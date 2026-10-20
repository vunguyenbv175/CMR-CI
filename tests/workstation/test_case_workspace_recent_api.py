"""Case-workspace recent-cases persistence & probe tests (CMR_CASE_WORKSPACE_RUNTIME_006).

Synthetic fixtures only (no PHI; SYNTH UIDs under ...99996 unique to this file).
Covers all acceptance criteria for slice 6:
- AC1: GET absent store returns 200 {entries: [], reason: "absent"}.
  Zero-length file reads as absent, not corrupt (explicit physical/codec distinction).
- AC2: Persist then relaunch: POST document then GET returns same entries with
  server-assigned openedSeq monotonic from durable high-water mark; client
  openedSeq is IGNORED.
- AC3: Stale snapshot & concurrent updates: mismatched base-digest gets 409
  recent.stale_snapshot + current document; retry with merge succeeds.
  Concurrent racing POSTs serialize without lost update or duplicate openedSeq.
- AC4: Codec-empty (OWNER DECISION A): single tampered persisted row (invalid UID,
  dest inside source, bad types, negative count, duplicate keys, >10 entries)
  degrades WHOLE document to empty with reason corrupt-rows.
- AC5: Failed writes preserve last-good: unwritable store keeps prior bytes + 400
  recent.store_unwritable. Unreadable store returns 400 recent.store_unreadable
  (never masquerades as absent).
- AC6: Tamper-row refusal parity on POST: blank/unsafe studyUid, namespace root,
  genuine-live on source/stage/derived-dest, dest inside source, symlinks all
  refused fail-closed with 400 before effects, store untouched.
- AC7: Authorization matrix: missing, spoofed, duplicate, or mismatched Host/Origin
  refused 403 {error: "forbidden"}.
- AC8: Staged identity & availability probe: ready (both present), source-missing
  (staged present -> reuse=true, reuse-first), staged-missing (source present ->
  reuse=false), unsafe-entry (dest inside source, genuine-live).
- AC9: Staged root census probe: GET /api/library/staged?dest_path= returns staged
  presence and series count.
- AC10: Source immutability: pre/post SHA-256 census on scan source files across
  all recent/probe/stage operations remains strictly identical.
- Review-round P1 fixes: durable high-water sidecar never regresses under a
  GET/POST interleave (deterministic race + fresh-process witness); sidecar
  symlink/bounded-read and temp O_EXCL refusals; /api/library/staged
  physical-path authority (frozen refusals, never suppressed 200); recent
  family pinned to loopback (listener + request authority); app.js
  first-open handoff wiring (queue-before-invoke, destination consumes).
"""

from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from services.workstation import recent as recent_mod
from services.workstation.recent import (
    RECENT_CASES_LIMIT,
    RECENT_CASES_SCHEMA_VERSION,
    RecentError,
    clear_pending_open,
    empty_document_digest,
    get_sidecar_seq_path,
    load_pending_open,
    load_recent_cases,
    persist_durable_high_water_seq,
    process_recent_post,
    read_durable_high_water_seq,
    save_pending_open,
)
from services.workstation.target_authority import GENUINE_LIVE_DIR, Refusal
from tests.synth.generate_dicom import write_synthetic_cine
from tests.workstation.support import (
    SeededStudy,
    json_body,
    json_value,
    make_seeded_study,
    request,
    running_server,
)

UID_BASE = "1.2.826.0.1.3680043.10.99996"
STUDY_A = f"{UID_BASE}.1"
STUDY_B = f"{UID_BASE}.2"
STUDY_C = f"{UID_BASE}.3"
SERIES_A = f"{STUDY_A}.1"
SERIES_B = f"{STUDY_B}.1"
SERIES_C = f"{STUDY_C}.1"


def _safe_request(
    server: Any,
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Request helper with retry on transient Windows loopback reset (WinError 10054).

    Mutating POSTs carry a per-logical-call ``open_id`` so a transport
    replay is deduplicated server-side instead of counted twice (review
    P2: real replay idempotence). Retries of the SAME logical call share
    the id; distinct logical calls must use distinct ids.
    """
    import time
    import uuid

    req_headers = {"Connection": "close"}
    if headers:
        req_headers.update(headers)
    open_id: str | None = None
    if method == "POST" and body:
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            # RecursionError: decoder RESOURCE failures are themselves valid
            # malformed bodies under test (review r4310462034); they simply
            # carry no open_id to sniff.
            decoded = None
        if isinstance(decoded, dict) and "open_id" not in decoded:
            open_id = "t" + uuid.uuid4().hex
            decoded = dict(decoded)
            decoded["open_id"] = open_id
            body = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            return request(
                server, base_url, path, method=method, body=body, headers=req_headers
            )
        except (ConnectionResetError, OSError) as exc:
            last_err = exc
            time.sleep(0.1 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError("request failed after retries")


def test_post_transport_retry_is_idempotent_by_direct_process_call(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """Real replay idempotence (review P2, contract merge_semantics).

    A duplicate delivery of the SAME logical open (same ``open_id``) is
    not a second successful open: the server returns the current document
    UNCHANGED — no ``openedSeq`` advance, no ``openCount`` movement, same
    digest. A DISTINCT logical open (different ``open_id``) still
    contributes exactly server-authoritative +1, whatever the client
    claims in ``openCount``.
    """
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)

    doc1 = process_recent_post(store, {"entry": entry, "open_id": "logical-open-1"})
    assert doc1["entries"][0]["openCount"] == 1
    first_seq = doc1["entries"][0]["openedSeq"]

    # Same logical open delivered twice (transport replay): the second
    # delivery changes NOTHING — not the counters, not the digest.
    replay = dict(entry)
    replay["openCount"] = 999
    doc2 = process_recent_post(
        store, {"entry": replay, "open_id": "logical-open-1"}
    )
    assert doc2["entries"][0]["openCount"] == 1
    assert doc2["entries"][0]["openedSeq"] == first_seq
    assert doc2["digest"] == doc1["digest"]

    # A distinct logical open still accumulates exactly prior+1, never the
    # client-supplied value.
    doc3 = process_recent_post(
        store, {"entry": replay, "open_id": "logical-open-2"}
    )
    assert doc3["entries"][0]["openCount"] == 2
    assert doc3["entries"][0]["openedSeq"] == first_seq + 1

    # Malformed open ids are refused, never trusted.
    with pytest.raises(RecentError):
        process_recent_post(store, {"entry": entry, "open_id": ""})
    with pytest.raises(RecentError):
        process_recent_post(store, {"entry": entry, "open_id": "../evil"})


def _census_tree(path: Path) -> dict[str, str]:
    """Digest every file in tree for immutability verification."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for root, _, files in os.walk(path):
        for f in sorted(files):
            full = Path(root) / f
            rel = str(full.relative_to(path)).replace("\\", "/")
            result[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
    return result


@pytest.fixture
def recent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Prepare isolated source, stage, and recent-store paths."""
    src = tmp_path / "scan_source"
    stage = tmp_path / "stage_base"
    store_file = tmp_path / "store" / "recent_cases.json"
    src.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CMR_RECENT_STORE_PATH", str(store_file))
    return src, stage, store_file


@pytest.fixture
def seeded_env(
    tmp_path: Path,
    recent_env: tuple[Path, Path, Path],
) -> tuple[SeededStudy, Path, Path, Path]:
    src, stage, store_file = recent_env
    seeded = make_seeded_study(tmp_path, series_uid=SERIES_A, study_uid=STUDY_A)
    return seeded, src, stage, store_file


def test_get_recent_absent_and_empty_file(seeded_env: tuple[SeededStudy, Path, Path, Path]) -> None:
    """AC1: GET /api/library/recent on absent or 0-byte file returns 200 reason=absent."""
    seeded, _, _, store_file = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        # 1. Store absent
        res = _safe_request(srv, base_url, "/api/library/recent")
        assert res.status == 200
        data = json_value(res)
        assert data["version"] == RECENT_CASES_SCHEMA_VERSION
        assert data["entries"] == []
        assert data["reason"] == "absent"
        assert data["digest"] == empty_document_digest()
        assert res.headers.get("Cache-Control") == "no-store"

        # 2. Store 0-byte file (explicit physical/codec distinction)
        store_file.parent.mkdir(parents=True, exist_ok=True)
        store_file.write_bytes(b"")
        res2 = _safe_request(srv, base_url, "/api/library/recent")
        assert res2.status == 200
        data2 = json_value(res2)
        assert data2["entries"] == []
        assert data2["reason"] == "absent"


def test_dangling_store_symlink_is_not_absent(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """A prohibited dangling store leaf must refuse, not masquerade as absent."""
    seeded, _src, _stage, store_file = seeded_env
    store_file.parent.mkdir(parents=True, exist_ok=True)
    missing_target = store_file.parent / "missing-target.json"
    try:
        store_file.symlink_to(missing_target)
    except OSError:
        pytest.skip("symlink creation unavailable without developer mode")
    assert store_file.is_symlink()
    assert not store_file.exists()
    with running_server(seeded.app) as (srv, base_url):
        response = _safe_request(srv, base_url, "/api/library/recent")
        assert response.status == 400
        assert json_value(response)["code"] == "recent.store_unreadable"
        assert store_file.is_symlink()
        assert not missing_target.exists()

        store_file.unlink()
        response = _safe_request(srv, base_url, "/api/library/recent")
        assert response.status == 200
        assert json_value(response)["reason"] == "absent"


def test_pending_post_symlinked_slot_returns_frozen_envelope(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """R8 Finding 1 HTTP leg: when the pending slot path contains a symlink,
    ``save_pending_open``'s entry safety check raises ``target_authority.Refusal``
    (reject_links does not raise RecentError). The pending POST must answer the
    frozen 400 envelope carrying the stable refusal code — never an
    internal-error response — and must not write any document through the
    hostile path."""
    seeded, src, stage, store_file = seeded_env
    store_file.parent.mkdir(parents=True, exist_ok=True)
    pending_path = recent_mod.get_pending_open_path(store_file)
    link_target = store_file.parent / "hostile-target"
    link_target.mkdir(parents=True, exist_ok=True)
    try:
        pending_path.symlink_to(link_target / "escaped.pending")
    except OSError:
        pytest.skip("symlink creation unavailable without developer mode")
    assert pending_path.is_symlink()
    entry = _fresh_entry(src, stage)
    with running_server(seeded.app) as (srv, base_url):
        res = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body({"id": "pend-link", "exp": 9999999999999, "entry": entry}),
        )
        assert res.status == 400, res
        payload = json_value(res)
        assert payload["error"] == "frozen_error", payload
        assert payload["code"] == "target_authority.symlink", payload
        # No bytes were ever written through the link:
        assert [p for p in link_target.rglob("*") if p.is_file()] == []
        assert pending_path.is_symlink()


def test_store_safety_precedes_absence_without_symlink_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-skipping ordering witness on hosts unable to create symlinks."""
    from services.workstation.target_authority import Refusal

    store = tmp_path / "missing.json"
    checked: list[Path] = []

    def refuse(path: Path) -> None:
        checked.append(path)
        raise Refusal("target_authority.symlink", "prohibited store leaf")

    monkeypatch.setattr(recent_mod, "check_store_path_safety", refuse)
    with pytest.raises(RecentError) as exc:
        load_recent_cases(store)
    assert exc.value.code == "recent.store_unreadable"
    assert checked == [store]


def test_main_document_leaf_swap_in_open_window_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main-document R3-2 analogue: a leaf swapped between validation and
    open must be refused. The handle still names the decoy it traversed to,
    so identity/location binding catches the object the decoy exposes."""
    _src, _stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": RECENT_CASES_SCHEMA_VERSION,
        "entries": [],
        "high_water_seq": 0,
    }
    store.write_text(json.dumps(doc), encoding="utf-8")

    real_open = recent_mod._open_main_document

    def swap_then_open(path: Path, flags: int) -> int:
        path.unlink()
        (path.parent / "decoy.json").write_text(
            '{"version": 1, "entries": "not-a-list"}', encoding="utf-8"
        )
        os.link(path.parent / "decoy.json", path)
        return real_open(path, flags)

    monkeypatch.setattr(recent_mod, "_open_main_document", swap_then_open)
    with pytest.raises(RecentError) as exc:
        load_recent_cases(store)
    # Either verdict is fail-closed: a same-location benign replacement
    # reads the new snapshot (recent.malformed on hostile content), while
    # a redirected ancestor still refuses target_authority.symlink.
    assert exc.value.code in {"target_authority.symlink", "recent.malformed"}
    assert store.read_text(encoding="utf-8").startswith('{"version": 1')


def test_main_document_parent_swap_in_open_window_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3 provenance-preserving ancestry: a parent swap (rename-aside +
    recreate) between validation and open is REFUSED with
    target_authority.symlink EVEN when the replacement bytes are
    schema-valid JSON. A swapped ancestry proves the approved directory
    was not continuously owned — no bytes from it are trusted, however
    well-formed. (Contrast the F5 interleave witness, where the parent is
    untouched and the new snapshot loads.)"""
    _src, _stage, store = recent_env
    parent = store.parent
    parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": RECENT_CASES_SCHEMA_VERSION,
        "entries": [],
        "high_water_seq": 0,
    }
    store.write_text(json.dumps(doc), encoding="utf-8")

    real_open = recent_mod._open_main_document
    moved = parent.with_name(parent.name + "-orig")

    def swap_parent_then_open(path: Path, flags: int) -> int:
        os.rename(parent, moved)
        parent.mkdir()
        (parent / store.name).write_text(
            json.dumps(
                {
                    "version": RECENT_CASES_SCHEMA_VERSION,
                    "entries": [],
                    "high_water_seq": 0,
                }
            ),
            encoding="utf-8",
        )
        return real_open(path, flags)

    monkeypatch.setattr(recent_mod, "_open_main_document", swap_parent_then_open)
    with pytest.raises(RecentError) as exc:
        load_recent_cases(store)
    assert exc.value.code == "target_authority.symlink"


def test_main_document_parent_swap_before_prestat_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12 P2 provenance-preserving ancestry: a parent swap executed BEFORE
    the main pre-open lstat (between parent_at_validation sampling and pre-stat)
    must be REFUSED with target_authority.symlink even when both pre-stat and
    opened handle agree on the replacement file and replacement bytes are
    schema-valid JSON."""
    _src, _stage, store = recent_env
    parent = store.parent
    parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": RECENT_CASES_SCHEMA_VERSION,
        "entries": [],
        "high_water_seq": 0,
    }
    store.write_text(json.dumps(doc), encoding="utf-8")

    real_read_hw = recent_mod.read_durable_high_water_seq
    moved = parent.with_name(parent.name + "-orig")

    def swap_parent_during_hw_read(store_path: Path) -> int:
        os.rename(parent, moved)
        parent.mkdir()
        (parent / store.name).write_text(
            json.dumps(
                {
                    "version": RECENT_CASES_SCHEMA_VERSION,
                    "entries": [
                        {
                            "studyUid": "1.2.826.0.1.3680043.10.99996.1",
                            "seriesUid": None,
                            "scanSourcePath": "/src",
                            "stageBaseDir": "/stage",
                            "destRoot": "/stage/1.2.826.0.1.3680043.10.99996.1",
                            "openedSeq": 1,
                            "openCount": 1,
                        }
                    ],
                    "high_water_seq": 1,
                }
            ),
            encoding="utf-8",
        )
        return real_read_hw(store_path)

    monkeypatch.setattr(recent_mod, "read_durable_high_water_seq", swap_parent_during_hw_read)
    try:
        with pytest.raises(RecentError) as exc:
            load_recent_cases(store)
        assert exc.value.code == "target_authority.symlink"
        assert "recent store parent changed between validation and open" in exc.value.message
    finally:
        if moved.exists():
            shutil.rmtree(parent, ignore_errors=True)
            os.rename(moved, parent)


def test_main_document_parent_absent_at_validation_then_created_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12 F2 matrix: if parent was absent at validation (parent_at_validation is None)
    but created before open, load_recent_cases must refuse fail-closed with
    target_authority.symlink rather than trusting unvalidated parent."""
    _src, _stage, store = recent_env
    parent = store.parent
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)

    real_read_hw = recent_mod.read_durable_high_water_seq

    def create_parent_during_hw(store_path: Path) -> int:
        parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [],
            "high_water_seq": 0,
        }
        store.write_text(json.dumps(doc), encoding="utf-8")
        return real_read_hw(store_path)

    monkeypatch.setattr(recent_mod, "read_durable_high_water_seq", create_parent_during_hw)
    try:
        with pytest.raises(RecentError) as exc:
            load_recent_cases(store)
        assert exc.value.code == "target_authority.symlink"
        assert "recent store parent changed between validation and open" in exc.value.message
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_main_document_growth_after_validation_is_never_ingested(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main-document growth analogue: bytes appended between size reasoning
    and the read never allocate more than cap + 1. The bounded handle read
    sees the excess and refuses before parsing anything."""
    _src, _stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"version": RECENT_CASES_SCHEMA_VERSION, "entries": []}),
        encoding="utf-8",
    )

    real_read = recent_mod._read_bounded_bytes

    def grow_then_read(fd: int, cap: int) -> bytes:
        store.write_bytes(b"x" * (recent_mod.MAX_RECENT_FILE_BYTES + 4096))
        return real_read(fd, cap)

    monkeypatch.setattr(recent_mod, "_read_bounded_bytes", grow_then_read)
    with pytest.raises(RecentError) as exc:
        load_recent_cases(store)
    assert exc.value.code == "recent.malformed"
    assert "exceeds bounded limit" in exc.value.message

def test_main_store_decoder_failures_are_frozen_refusals(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """PR3 review r4310080008 (fail-capable): under-byte-cap JSON that makes
    the DECODER itself fail before check_json_depth can run.

    ~6,000 nested arrays raise RecursionError and an integer beyond the
    interpreter digit limit raises plain ValueError; both are far below the
    byte cap, so neither the size guard nor the depth walk sees them first.
    Pre-fix they escaped to the generic HTTP boundary (500, no frozen code,
    no no-store); they must read as the frozen recent.malformed refusal with
    no-store and leave the prior store bytes exactly as planted."""
    seeded, _src, _stage, store_file = seeded_env
    store_file.parent.mkdir(parents=True, exist_ok=True)
    payloads = (
        (("[" * 6000 + "]" * 6000).encode("utf-8"), "nested-arrays"),
        (
            b'{"version": 1, "entries": [], "tamper": ' + b"9" * 4301 + b"}",
            "oversized-int",
        ),
    )
    with running_server(seeded.app) as (srv, base_url):
        for raw, label in payloads:
            assert len(raw) < recent_mod.MAX_RECENT_FILE_BYTES, (label, len(raw))
            store_file.write_bytes(raw)
            before = store_file.read_bytes()
            res = _safe_request(srv, base_url, "/api/library/recent")
            assert res.status == 400, (label, res.status, json_value(res))
            assert json_value(res).get("code") == "recent.malformed", (
                label,
                json_value(res),
            )
            assert res.headers.get("Cache-Control") == "no-store", (
                label,
                res.headers,
            )
            assert store_file.read_bytes() == before, (label, "prior bytes changed")


def test_recent_post_body_decoder_failures_are_frozen_route_refusals(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """Review 5567119593 P2 (comment r4310462034): decoder RESOURCE failures
    in an incoming recent POST body take the frozen route envelope.

    ``_read_json_body`` catches only ``UnicodeDecodeError`` /
    ``JSONDecodeError`` while the recent POST boundary translated only
    ``InvalidPayload``, so a body of ~6,000 nested arrays (12 KiB, far below
    the 4 MiB byte cap) raised ``RecursionError`` before ``check_json_depth``
    could run, and a JSON integer beyond the interpreter digit limit raised a
    plain ``ValueError``; both escaped the frozen envelope as generic 500 /
    ``invalid_request`` without ``no-store``. Actual malformed POST bytes on
    EITHER recent endpoint must answer the frozen ``400`` ``recent.malformed``
    with ``no-store``, leaving the main store document and the pending slot
    byte-identical (no side effect at all).
    """
    seeded, _src, _stage, store_file = seeded_env
    pending_file = recent_mod.get_pending_open_path(store_file)
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_bytes(b'{"version": 1, "entries": [], "high_water_seq": 0}')
    pending_file.write_bytes(b"planted-pending-bytes")
    payloads = (
        (("[" * 6000 + "]" * 6000).encode("utf-8"), "nested-arrays"),
        (b'{"entry": ' + b"9" * 4301 + b"}", "oversized-int"),
    )
    with running_server(seeded.app) as (srv, base_url):
        for path in ("/api/library/recent", "/api/library/recent/pending"):
            for raw, label in payloads:
                where = f"{path} {label}"
                assert len(raw) < 4 * 1024 * 1024, (where, len(raw))
                before_store = store_file.read_bytes()
                before_pending = pending_file.read_bytes()
                before_tree = _census_tree(store_file.parent)
                res = _safe_request(srv, base_url, path, method="POST", body=raw)
                assert res.status == 400, (where, res.status, json_value(res))
                body = json_value(res)
                assert body.get("code") == "recent.malformed", (where, body)
                assert res.headers.get("Cache-Control") == "no-store", (
                    where,
                    dict(res.headers),
                )
                assert store_file.read_bytes() == before_store, (
                    where,
                    "main store bytes changed",
                )
                assert pending_file.read_bytes() == before_pending, (
                    where,
                    "pending slot bytes changed",
                )
                assert _census_tree(store_file.parent) == before_tree, (
                    where,
                    "store directory gained or lost a file",
                )


def test_write_all_completes_short_writes_and_refuses_zero_progress(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 short-write (unit): a partial os.write is completed by looping;
    a writer reporting zero progress raises instead of committing a
    truncated buffer."""
    _, _, store = recent_env
    target = store.parent / "w.bin"
    target.parent.mkdir(parents=True, exist_ok=True)

    real_write = os.write
    calls = {"n": 0}

    def short_once(fd: int, data: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data[:1]))
        return real_write(fd, data)

    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        monkeypatch.setattr(os, "write", short_once)
        recent_mod._write_all(fd, b"0123456789")
    finally:
        os.close(fd)
    assert target.read_bytes() == b"0123456789"

    def zero_forever(fd: int, data: Any) -> int:
        return 0

    fd2 = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        monkeypatch.setattr(os, "write", zero_forever)
        with pytest.raises(OSError):
            recent_mod._write_all(fd2, b"0123456789")
    finally:
        os.close(fd2)


def test_short_write_never_replaces_last_good(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 short-write (integration): a persistently short writer fails the
    save with recent.store_unwritable; prior bytes are preserved and no
    truncated document is observable."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    doc1 = process_recent_post(store, {"entry": entry, "open_id": "w-short-1"})
    assert doc1["entries"][0]["openCount"] == 1
    good_bytes = store.read_bytes()

    def short_always(fd: int, data: Any) -> int:
        return 0

    monkeypatch.setattr(os, "write", short_always)
    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "w-short-2"})
    assert exc_info.value.code == "recent.store_unwritable"
    assert store.read_bytes() == good_bytes
    # Last-good still loads and serves the original open. (The global
    # short-writer is still patched here, so read the entries straight
    # from the preserved bytes instead of re-entering the sidecar path.)
    preserved = json.loads(good_bytes.decode("utf-8"))
    assert preserved["entries"][0]["openCount"] == 1


@pytest.mark.skipif(
    recent_mod._dir_fd_replace_supported(),
    reason=(
        "Windows/no-dir_fd fallback witness (WIN-WRITE-012 scope): the POSIX "
        "handle path is covered by the R7 F1 acquisition witnesses. On a "
        "dir_fd host the save leg verifies the parent identity at "
        "ACQUISITION, so this flip-seam shape no longer describes it."
    ),
)
def test_parent_swap_between_check_and_replace_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1 write-side ancestor swap: when the store parent identity changes
    between validation and replace, the write aborts fail-closed — no
    replace happens outside the approved directory and last-good is kept.

    Scope-note: this witness drives the MAIN-document save path. The POST
    first reloads the document (which persists the durable high-water
    sidecar through the same ``_atomic_replace`` helper), so the
    identity seam is save-leg-scoped: the flip shape pins truthfully on
    its FIRST save-path observation and reports the swap on every later
    one. The sidecar persist inside the POST's load is neutralized (it
    would otherwise consume the truthful call or hit the armed replace
    seam before the main save — review R2 on the F2 witness). The
    load itself still runs fully; only the derived sidecar write is
    skipped, which is invisible to the refusal under test.

    Platform scope (WIN-WRITE-012): the ``_storage_parent_identity`` /
    ``_replace_path`` seams consulted here exist ONLY on the non-dir_fd
    (Windows) fallback — on a POSIX host with dir_fd support those seams
    are dead code and this witness does not exercise the handle path.
    The POSIX handle path is covered instead by
    ``test_posix_handle_path_binds_replace_to_pinned_parent`` (fd
    binding) and ``test_posix_handle_path_refusal_on_identity_change``
    (detection); the suite is therefore Windows-scoped for refusal and
    construction-scoped for POSIX binding, as documented on
    ``_atomic_replace``."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "pswap-1"})
    good_bytes = store.read_bytes()

    def swapping_replace(src: Path, dst: Path) -> None:
        # Simulate the race deterministically: the parent was swapped for
        # a hostile directory AFTER validation, so the pathname replace
        # would land outside the approved dir — the helper must refuse
        # before calling through.
        raise AssertionError("replace must not run after a parent swap")

    identities = [(11, 22), (33, 44)]  # pin, then changed
    calls = {"n": 0}

    def flipping_identity(path: Path) -> tuple[int, int]:
        # Save-leg pin/then-changed flip: the FIRST THREE observations pin
        # the parent identity truthfully (temp-creation pre/post pair and
        # replace-side pre inside the save path), and every LATER observation
        # reports the swapped identity — so the temp-creation gate passes on the
        # still-valid parent while the replace-side re-verification
        # observes the swap and refuses. The witness arms this seam only
        # around the save leg (see scoped_post), so the load leg never
        # consumes the truthful calls.
        calls["n"] += 1
        return identities[0] if calls["n"] <= 3 else identities[1]

    real_post = recent_mod.process_recent_post
    real_persist = recent_mod.persist_durable_high_water_seq
    real_save = recent_mod.save_recent_cases_to_disk

    def scoped_post(payload: dict[str, Any]) -> dict[str, Any]:
        # Arm the save-leg witness seams, then run the real POST with the
        # sidecar persist neutralized: the sidecar persist inside the
        # POST's load would otherwise consume the single truthful
        # identity call (or hit the armed replace seam) before the main
        # save is reached (review R2 on the F2 witness). The POST's own
        # load still runs fully (validation, lstat, handle binding,
        # ancestry gate) — only the derived sidecar write is skipped,
        # which is invisible to the refusal under test.
        monkeypatch.setattr(recent_mod, "_replace_path", swapping_replace)
        monkeypatch.setattr(
            recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None
        )

        def scoped_save(*args: Any, **kwargs: Any) -> Any:
            monkeypatch.setattr(
                recent_mod, "_storage_parent_identity", flipping_identity
            )
            return real_save(*args, **kwargs)

        monkeypatch.setattr(recent_mod, "save_recent_cases_to_disk", scoped_save)
        try:
            return real_post(store, payload)
        finally:
            monkeypatch.setattr(
                recent_mod, "persist_durable_high_water_seq", real_persist
            )

    with pytest.raises(RecentError) as exc_info:
        scoped_post({"entry": entry, "open_id": "pswap-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert calls["n"] >= 2, calls
    assert store.read_bytes() == good_bytes


def test_write_parent_redirection_never_escapes_approved_dir(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 ancestor-redirection (REAL swap, no mocks): the approved parent
    is renamed aside and replaced by a hostile directory mid-write. Temp
    bytes must never escape into it: the write either lands in the pinned
    approved directory or aborts with last-good preserved, and the hostile
    directory holds no store/temp bytes afterwards."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "redir-1"})
    good_bytes = store.read_bytes()

    parent = store.parent
    moved = parent.with_name(parent.name + "-orig")
    hostile = parent.with_name(parent.name + "-hostile")
    hostile.mkdir(parents=True, exist_ok=True)
    real_atomic = recent_mod._atomic_replace

    def swap_then_replace(
        temp_path: Path, target: Path, *, action: str, **kwargs: Any
    ) -> None:
        # REAL ancestor swap between validation and use: the approved
        # parent moves aside; a hostile directory takes its pathname.
        # R7 F1: the save path now passes its ACQUISITION-verified parent
        # handle through (POSIX); the wrapper forwards it so the replace
        # stays bound to the approved inode — the swap then cannot
        # redirect the commit.
        os.rename(parent, moved)
        os.rename(hostile, parent)
        try:
            real_atomic(temp_path, target, action=action, **kwargs)
        finally:
            # Restore layout for post-assertions (never trust the swap).
            if parent.is_dir():
                try:
                    os.rename(parent, hostile)
                except OSError:
                    pass
            if moved.is_dir():
                try:
                    os.rename(moved, parent)
                except OSError:
                    pass

    monkeypatch.setattr(recent_mod, "_atomic_replace", swap_then_replace)
    try:
        process_recent_post(store, {"entry": entry, "open_id": "redir-2"})
        outcome = "committed"
    except RecentError as exc:
        outcome = exc.code

    # Either the write committed into the REAL approved directory (its
    # bytes are intact and parse as the new document) or it aborted
    # fail-closed (last-good preserved with a stable refusal code — the
    # parent-swap refusal, or the unwritable-store refusal when the swap
    # breaks a handle/path the write needs). What must NEVER happen is
    # temp bytes landing in the hostile directory.
    hostile_leftovers = [
        p for p in hostile.rglob("*") if p.is_file() and p.name != "seed.txt"
    ] if hostile.is_dir() else []
    assert hostile_leftovers == [], hostile_leftovers
    if outcome == "committed":
        doc = json.loads(store.read_bytes().decode("utf-8"))
        assert doc["entries"][0]["openCount"] == 2
    else:
        assert outcome in {"target_authority.symlink", "recent.store_unwritable"}
        assert store.read_bytes() == good_bytes


def test_posix_handle_path_binds_replace_to_pinned_parent(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIN-WRITE-012 POSIX leg: with the dir_fd branch forced (stubbed fds —
    Windows has no dir_fd, so the branch is unreachable natively), the
    replace MUST be issued through the pinned parent fd on BOTH sides
    (src_dir_fd/dst_dir_fd) with bare leaf names — no post-validation
    pathname re-traversal can escape the approved directory even if the
    parent pathname is swapped. Fully mocked by necessity; what it proves
    is the BINDING, which is the POSIX security property (fd-relative
    replace cannot name another directory)."""
    _src, _stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)
    temp_path = store.parent / ".tmp_probe_bind"
    temp_path.write_bytes(b"probe")

    calls: dict[str, Any] = {}

    def fake_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        calls["src"] = src
        calls["dst"] = dst
        calls["kwargs"] = kwargs

    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", lambda _p: 999)
    monkeypatch.setattr(recent_mod, "_storage_parent_identity_fd", lambda _fd: (7, 7))
    # Let the REAL _replace_in_parent run; capture the os.replace call it
    # issues (os.replace itself is stubbed: Windows os.replace accepts no
    # dir_fd kwargs, so this also proves the kwargs are really passed).
    monkeypatch.setattr(os, "replace", fake_replace)
    # The pathname fallback must never run on the handle path.
    monkeypatch.setattr(recent_mod, "_replace_path", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("pathname replace must not run on the handle path")
    ))
    recent_mod._atomic_replace(temp_path, store, action="replace")
    assert calls["kwargs"].get("src_dir_fd") == 999, calls
    assert calls["kwargs"].get("dst_dir_fd") == 999, calls
    assert calls["src"] == temp_path.name, calls  # bare leaf, no directory
    assert calls["dst"] == store.name, calls  # bare leaf, no directory
    assert os.sep not in str(calls["src"]) and "/" not in str(calls["src"])


def test_posix_handle_path_refusal_on_identity_change(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIN-WRITE-012 POSIX refusal leg: on the handle path, a parent-identity
    change observed through the pinned fd raises target_authority.symlink
    (detection defense-in-depth; the fd-relative replace itself cannot
    escape regardless). Same stubbed-fd necessity as the binding witness."""
    _src, _stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)
    temp_path = store.parent / ".tmp_probe_refuse"
    temp_path.write_bytes(b"probe")

    identities = [(7, 7), (8, 8)]
    calls = {"n": 0}

    def flipping_fd_identity(_fd: int) -> tuple[int, int]:
        calls["n"] += 1
        return identities[0] if calls["n"] == 1 else identities[1]

    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", lambda _p: 999)
    monkeypatch.setattr(recent_mod, "_storage_parent_identity_fd", flipping_fd_identity)
    monkeypatch.setattr(recent_mod, "_replace_in_parent", lambda *_a, **_k: None)
    with pytest.raises(RecentError) as exc_info:
        recent_mod._atomic_replace(temp_path, store, action="replace")
    assert exc_info.value.code == "target_authority.symlink"
    assert calls["n"] >= 2, calls


@pytest.mark.skipif(
    not recent_mod._dir_fd_replace_supported(),
    reason="POSIX dir_fd write path only (WIN-WRITE-012 boundary)",
)
def test_posix_main_parent_rename_replace_before_prereplace_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13 P2 write-side witness (real Linux): the approved parent PATHNAME is
    renamed aside and replaced by a NEW real directory containing a
    schema-valid store document AFTER acquisition binding and BEFORE the
    fd-relative replace. ``fstat(parent_fd)`` still names the detached
    original, so a handle-only re-check can never see the swap: the pre-replace
    pathname parent identity must be re-verified against the ACQUISITION-bound
    identity and refuse BEFORE the write — even though the replacement bytes
    are valid JSON. Proof properties: refusal raised (the request must NOT
    report success), the pinned original directory keeps its last-good bytes
    (the replace never ran — pre-fix it replaced them), no temp leak, and this
    process never writes the active pathname."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "rp-1"})
    good_bytes = store.read_bytes()

    parent = store.parent
    moved = parent.with_name(parent.name + "-orig")
    attacker_doc = json.dumps(
        {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [],
            "high_water_seq": 7,
        }
    ).encode("utf-8")

    real_atomic = recent_mod._atomic_replace
    swapped = {"done": False}

    def rename_replace_then_atomic(
        temp_path: Path, target: Path, *, action: str, **kwargs: Any
    ) -> None:
        # Real rename/replace of the parent pathname AFTER acquisition and
        # temp creation, BEFORE the fd-relative replace runs (first call only).
        if not swapped["done"]:
            swapped["done"] = True
            os.rename(parent, moved)
            parent.mkdir(parents=True, exist_ok=True)
            (parent / store.name).write_bytes(attacker_doc)
        return real_atomic(temp_path, target, action=action, **kwargs)

    monkeypatch.setattr(recent_mod, "_atomic_replace", rename_replace_then_atomic)
    try:
        with pytest.raises(RecentError) as exc_info:
            process_recent_post(store, {"entry": entry, "open_id": "rp-2"})
        assert exc_info.value.code == "target_authority.symlink"
        assert "parent changed between acquisition and replace" in exc_info.value.message
        # The pinned original directory still holds last-good bytes: the
        # fd-relative replace NEVER ran (pre-fix it replaced them in place).
        assert (moved / store.name).read_bytes() == good_bytes
        # No temp leak in the pinned directory (cleanup ran through the fd).
        assert list(moved.glob(".tmp_*")) == []
        # This process never wrote the active pathname (planted bytes intact).
        assert store.read_bytes() == attacker_doc
    finally:
        if moved.exists():
            shutil.rmtree(parent, ignore_errors=True)
            os.rename(moved, parent)


@pytest.mark.skipif(
    not recent_mod._dir_fd_replace_supported(),
    reason="POSIX dir_fd write path only (WIN-WRITE-012 boundary)",
)
def test_posix_sidecar_parent_rename_replace_before_prereplace_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13 P2 sidecar witness (real Linux): same rename/replace of the parent
    pathname before the fd-relative replace of the high-water sidecar. The
    sidecar writer must refuse against the ACQUISITION-bound pathname parent
    identity before writing, preserving the original sidecar bytes and never
    touching the active pathname — even though the planted replacement sidecar
    content is well-formed."""
    _src, _stage, store = recent_env
    parent = store.parent
    parent.mkdir(parents=True, exist_ok=True)
    seq_path = get_sidecar_seq_path(store)
    recent_mod.persist_durable_high_water_seq(store, 42)
    assert seq_path.read_text(encoding="utf-8") == "42"
    good_seq = seq_path.read_bytes()

    moved = parent.with_name(parent.name + "-orig")
    real_atomic = recent_mod._atomic_replace

    def rename_replace_then_atomic(
        temp_path: Path, target: Path, *, action: str, **kwargs: Any
    ) -> None:
        os.rename(parent, moved)
        parent.mkdir(parents=True, exist_ok=True)
        (parent / seq_path.name).write_text("99", encoding="utf-8")
        return real_atomic(temp_path, target, action=action, **kwargs)

    monkeypatch.setattr(recent_mod, "_atomic_replace", rename_replace_then_atomic)
    try:
        with pytest.raises(RecentError) as exc_info:
            recent_mod.persist_durable_high_water_seq(store, 55)
        assert exc_info.value.code == "target_authority.symlink"
        assert "parent changed between acquisition and replace" in exc_info.value.message
        # Original sidecar bytes preserved in the pinned detached directory:
        # pre-fix the fd-relative replace rewrote them to "55".
        assert (moved / seq_path.name).read_bytes() == good_seq
        assert list(moved.glob(".tmp_*")) == []
        # Active pathname never written by this process (planted bytes intact).
        assert (parent / seq_path.name).read_text(encoding="utf-8") == "99"
    finally:
        if moved.exists():
            shutil.rmtree(parent, ignore_errors=True)
            os.rename(moved, parent)


def test_sidecar_safety_refusal_is_not_self_healed(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1 sidecar writer: a path-SAFETY refusal on the sidecar (link leaf)
    is fail-closed — the write is skipped, never pushed through a hostile
    path. Content malformation still degrades to the 0 baseline."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "sc-1"})
    seq_path = get_sidecar_seq_path(store)
    assert seq_path.is_file()

    # Hostile content (non-digits) degrades to the 0 baseline, no refusal.
    seq_path.write_bytes(b"not-a-number")
    assert recent_mod._read_sidecar_seq(seq_path) == 0

    # A safety refusal must propagate out of persist (no silent overwrite).
    # Memory-mark hygiene: persist updates the in-process mark before the
    # sidecar write, so snapshot it to observe propagation cleanly.
    path_key = recent_mod._hw_path_key(store)
    saved_mark = recent_mod._HIGH_WATER_BY_PATH.get(path_key)

    def hostile_read(path: Path) -> int:
        raise RecentError("target_authority.symlink", "injected hostile sidecar")

    monkeypatch.setattr(recent_mod, "_read_sidecar_seq", hostile_read)
    with pytest.raises(RecentError) as exc_info:
        recent_mod.persist_durable_high_water_seq(store, 999)
    assert exc_info.value.code == "target_authority.symlink"
    if saved_mark is None:
        recent_mod._HIGH_WATER_BY_PATH.pop(path_key, None)
    else:
        recent_mod._HIGH_WATER_BY_PATH[path_key] = saved_mark


def _foreign_replace_store_document(
    store_path: Path,
    rows: list[dict[str, Any]],
    high_water_seq: int,
) -> None:
    """Atomically replace the store with a schema-valid document WITHOUT the
    in-process recent API — the FOREIGN (out-of-process) writer emulation.

    PR3 review F1 serializes the in-process GET snapshot against the in-process
    POST commit through the shared lock, so a replace can no longer land inside
    a load's validation window from this process. The only remaining shape for
    that window is a writer OUTSIDE the process (another process replacing the
    store file), which is exactly what this helper reproduces: same-dir temp +
    ``os.replace`` with no locked read-merge.
    """
    payload = {
        "version": RECENT_CASES_SCHEMA_VERSION,
        "entries": rows,
        "high_water_seq": high_water_seq,
        "seen_open_ids": [],
    }
    tmp = store_path.parent / f".foreign_{os.getpid()}_{time.monotonic_ns()}.json"
    tmp.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    os.replace(tmp, store_path)


def test_read_replace_interleave_returns_snapshot_not_refusal(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 GET/POST interleave (F1-serialized shape): a legitimate replace
    landing between a GET's validation and its open must NOT surface as 400
    target_authority.symlink. The GET reads the NEW snapshot from its
    already-bound handle (location-equal, regular file). Because F1 serializes
    the in-process POST against this load, the replace in that window is
    emulated by a FOREIGN writer (see the helper above); a REAL redirection
    (location mismatch) still refuses — covered by the existing
    leaf/parent-swap witnesses."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    doc1 = process_recent_post(store, {"entry": entry, "open_id": "il-1"})
    assert doc1["entries"][0]["openCount"] == 1

    real_open = recent_mod._open_main_document
    newer_row = dict(entry, openedSeq=2, openCount=2)
    calls = {"n": 0}

    def replace_then_open(path: Path, flags: int) -> int:
        if calls["n"] == 0:
            calls["n"] += 1
            # Foreign-writer replace INSIDE this load's validation window
            # (between the pre-open lstat and the open): the only shape F1's
            # serialization leaves reachable.
            _foreign_replace_store_document(store, [newer_row], 2)
        return real_open(path, flags)

    monkeypatch.setattr(recent_mod, "_open_main_document", replace_then_open)
    got = recent_mod.load_recent_cases(store)
    assert calls["n"] == 1, calls  # the replace really landed in the window
    # The bound handle reads the new snapshot; a symlink refusal is not.
    assert got["entries"][0]["openCount"] == 2, got
    assert got["reason"] == "none", got


def test_read_replace_interleave_proves_ancestry_untouched(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3 ancestry-positive companion: the interleave witness above loads
    the new snapshot ONLY because the parent identity is unchanged. Prove
    the gate is load-bearing — if the post-open parent observation reports
    a change for that same interleave, the load refuses instead.

    The replace in the validation window is the FOREIGN-writer emulation
    (F1 serializes the in-process POST out of it); the ancestry lie is armed
    when that replace lands, so the first armed observation is the load's own
    post-open ``parent_now``."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "ilb-1"})

    real_open = recent_mod._open_main_document
    real_identity = recent_mod._storage_parent_identity
    newer_row = dict(entry, openedSeq=2, openCount=2)
    calls = {"n": 0}

    def replace_then_open(path: Path, flags: int) -> int:
        if calls["n"] == 0:
            calls["n"] += 1
            _foreign_replace_store_document(store, [newer_row], 2)
            anchor_calls["armed"] = True
        return real_open(path, flags)

    # Lie about EVERY post-arm observation: the validation anchor is
    # sampled before the seam is armed (always truthful), and the outer
    # load's post-open `parent_now` is the first armed call — so the lie
    # must start at the first armed call, not the second.
    anchor_calls: dict[str, Any] = {"n": 0, "armed": False}

    def split_identity(path: Path) -> tuple[int, int]:
        ident = real_identity(path)
        if not anchor_calls["armed"]:
            return ident
        anchor_calls["n"] += 1
        dev, ino = ident
        return (dev, ino + 1)

    monkeypatch.setattr(recent_mod, "_open_main_document", replace_then_open)
    monkeypatch.setattr(recent_mod, "_storage_parent_identity", split_identity)
    with pytest.raises(RecentError) as exc:
        recent_mod.load_recent_cases(store)
    assert exc.value.code == "target_authority.symlink"
    # The armed gate observation fired (the lie was actually consulted).
    assert anchor_calls["n"] >= 1


def test_post_open_replacement_schedule_has_no_spurious_refusal(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3 post-open schedule: when the replacement lands AFTER the handle
    is already bound (open returns first, replace commits during the
    bounded read), the GET still serves a snapshot — no
    target_authority.symlink for a schedule the reader cannot control.
    The parent is untouched throughout, so ancestry stays proven.

    Windows sharing-note: the CRT read handle denies rename/replace of
    the store while it is open (ERROR_ACCESS_DENIED), so a same-thread
    inner POST can never commit under the open handle on Windows — and
    in production the two sides are serialized under _LOCK anyway (the
    inner POST blocks until the GET's handle is closed). The witness
    therefore defers the inner POST until AFTER the outer load returns
    (recorded, then executed outside the handle lifetime): the load
    serves the pre-replace snapshot, the deferred POST then commits the
    new snapshot, and a follow-up load observes it. No refusal at any
    step proves the post-open schedule is not a refusal schedule.
    """
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    doc1 = process_recent_post(store, {"entry": entry, "open_id": "po-1"})
    assert doc1["entries"][0]["openCount"] == 1

    real_read = recent_mod._read_bounded_bytes
    main_cap = recent_mod.MAX_RECENT_FILE_BYTES
    replaced = {"n": 0}
    deferred: list[Any] = []

    def replace_then_read(fd: int, cap: int) -> bytes:
        # The handle is already bound; a legitimate POST commits now,
        # during this GET's read. Fire ONLY on the main-document read
        # (cap == MAX_RECENT_FILE_BYTES): this wrapper is also reached
        # by the sidecar read, which runs with _HW_LOCK held — an inner
        # POST from the sidecar read would re-enter _HW_LOCK on this
        # same thread and self-deadlock.
        #
        # Deferred-commit discipline (see docstring): snapshot the bound
        # bytes through the real read and RECORD the inner POST for
        # after the outer load returns, when the handle is closed and the
        # replace lands on a closed destination. Serve the pre-replace
        # snapshot now: the reader cannot observe a replace that lands
        # after its handle is bound, and ancestry is proven (parent
        # untouched), so this is a snapshot, never a refusal.
        if cap == main_cap and replaced["n"] == 0:
            replaced["n"] += 1
            data = real_read(fd, cap)

            def _deferred_post() -> None:
                process_recent_post(store, {"entry": entry, "open_id": "po-2"})

            deferred.append(_deferred_post)
            return data
        return real_read(fd, cap)

    monkeypatch.setattr(recent_mod, "_read_bounded_bytes", replace_then_read)
    try:
        got = recent_mod.load_recent_cases(store)
    finally:
        monkeypatch.setattr(recent_mod, "_read_bounded_bytes", real_read)
    # The bound handle reads a consistent snapshot (old or new); the
    # post-open replace it cannot observe is not a refusal.
    assert got["entries"][0]["openCount"] in (1, 2)
    assert got["reason"] == "none"
    # Exactly one deferred writer was recorded; run it now (handle
    # closed) and prove the new snapshot lands and loads.
    assert len(deferred) == 1, deferred
    deferred[0]()
    got2 = recent_mod.load_recent_cases(store)
    assert got2["entries"][0]["openCount"] == 2
    assert got2["reason"] == "none"


def test_get_read_is_serialized_with_post_commit_under_real_overlap(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3 review F1 (real overlap, barrier-controlled): the GET snapshot read
    and the POST commit must share the ONE in-process lock.

    The reader is paused INSIDE its bound handle (barrier seam immediately
    after the real open) and released only once the writer's state is known:
    (a) the writer is blocked behind the reader's commit lock — the fixed
    shape, so the POST cannot replace the document under a live read handle;
    or (b) the writer reached the main-document replace while the reader's
    handle is still open — the pre-fix shape, where the reader's handle
    starves the POST (Windows: sharing violation -> recent.store_unwritable)
    or the reader's post-open lstat observes the replacement and refuses with
    the spurious target_authority.symlink (POSIX). No sleep decides the
    overlap: the reader's pause is a barrier and the writer's blocking is
    observed deterministically through the lock proxy."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "ov-seed"})

    reader_ident: dict[str, int] = {}
    writer_ident: dict[str, int] = {}
    reader_has_handle = threading.Event()
    release_reader = threading.Event()
    writer_attempted = threading.Event()
    writer_blocked = threading.Event()
    writer_at_replace = threading.Event()
    writer_done = threading.Event()
    overlap = {"during_reader_handle": False}
    reader_result: dict[str, Any] = {}
    writer_result: dict[str, Any] = {}

    real_lock = recent_mod._LOCK
    real_open = recent_mod._open_main_document
    real_replace = recent_mod._atomic_replace

    class _SignalLock:
        """Transparent proxy over the real lock (test seam): reports the
        writer's acquire attempt and, deterministically, whether it had to
        block behind an in-flight reader."""

        def __enter__(self) -> Any:
            if threading.get_ident() == writer_ident.get("id"):
                writer_attempted.set()
                if not real_lock.acquire(blocking=False):
                    writer_blocked.set()
                    real_lock.acquire()
            else:
                real_lock.acquire()
            return self

        def __exit__(self, *_exc: Any) -> bool:
            real_lock.release()
            return False

    def open_with_reader_barrier(path: Path, flags: int) -> int:
        fd = real_open(path, flags)
        if threading.get_ident() == reader_ident.get("id"):
            reader_has_handle.set()
            assert release_reader.wait(30), "reader barrier was never released"
        return fd

    def replace_tracker(*args: Any, **kwargs: Any) -> None:
        if (
            threading.get_ident() == writer_ident.get("id")
            and args[1] == store
            and reader_has_handle.is_set()
            and not release_reader.is_set()
        ):
            overlap["during_reader_handle"] = True
            writer_at_replace.set()
        return real_replace(*args, **kwargs)

    def reader() -> None:
        reader_ident["id"] = threading.get_ident()
        try:
            reader_result["doc"] = recent_mod.load_recent_cases(store)
        except BaseException as exc:  # noqa: BLE001 - the witness records any failure
            reader_result["error"] = exc

    def writer() -> None:
        writer_ident["id"] = threading.get_ident()
        try:
            writer_result["doc"] = process_recent_post(
                store, {"entry": entry, "open_id": "ov-next"}
            )
        except BaseException as exc:  # noqa: BLE001 - the witness records any failure
            writer_result["error"] = exc
        finally:
            writer_done.set()

    monkeypatch.setattr(recent_mod, "_LOCK", _SignalLock())
    monkeypatch.setattr(recent_mod, "_open_main_document", open_with_reader_barrier)
    monkeypatch.setattr(recent_mod, "_atomic_replace", replace_tracker)
    reader_thread = threading.Thread(target=reader, name="f1-reader")
    writer_thread = threading.Thread(target=writer, name="f1-writer")
    try:
        reader_thread.start()
        assert reader_has_handle.wait(30), "reader never bound its handle"
        writer_thread.start()
        assert writer_attempted.wait(30), "writer never attempted the commit lock"
        if not writer_blocked.wait(5):
            # Pre-fix shape: the writer is not serialized — let it reach the
            # replace (or fail) while the reader's handle stays open, so the
            # witness observes the real overlap instead of a scheduling guess.
            writer_at_replace.wait(5)
    finally:
        release_reader.set()
        reader_thread.join(30)
        writer_thread.join(30)

    assert writer_done.is_set(), "writer thread never completed"
    assert "error" not in writer_result, writer_result
    assert writer_result["doc"]["entries"][0]["openCount"] == 2, writer_result
    assert "error" not in reader_result, reader_result
    # Serialized: the reader entered its snapshot first, so it serves the
    # pre-commit snapshot; the commit lands after it releases the lock.
    assert reader_result["doc"]["entries"][0]["openCount"] == 1, reader_result
    assert overlap["during_reader_handle"] is False, (
        "the main document was replaced while the GET read handle was open"
    )
    final = recent_mod.load_recent_cases(store)
    assert final["entries"][0]["openCount"] == 2, final


def test_get_route_waits_for_in_flight_commit_lock(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """PR3 review F1 (API boundary): the GET route must go through the same
    serialized loader as the POST commit path. While a commit holds the shared
    lock, a concurrent GET must NOT answer (pre-fix code answers immediately
    from the unsynchronized loader); after release it serves 200."""
    seeded, src, stage, _store_file = seeded_env
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "entry": {
            "studyUid": STUDY_A,
            "seriesUid": None,
            "scanSourcePath": str(sdir),
            "stageBaseDir": str(stage),
            "destRoot": str(stage / STUDY_A),
            "openedSeq": 0,
            "openCount": 1,
        }
    }

    with running_server(seeded.app) as (srv, base_url):
        warm = _safe_request(srv, base_url, "/api/library/recent")
        assert warm.status == 200, json_value(warm)
        post = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload),
        )
        assert post.status == 200, json_value(post)

        recent_mod._LOCK.acquire()
        result: dict[str, Any] = {}

        def _get() -> None:
            try:
                res = _safe_request(srv, base_url, "/api/library/recent")
                result["status"] = res.status
                result["doc"] = json_value(res)
            except BaseException as exc:  # noqa: BLE001 - the witness records any failure
                result["error"] = exc

        get_thread = threading.Thread(target=_get, name="f1-api-get")
        try:
            get_thread.start()
            get_thread.join(1.0)
            assert get_thread.is_alive(), (
                "GET answered while a commit held the shared lock "
                f"(response: {result})"
            )
        finally:
            recent_mod._LOCK.release()
        get_thread.join(15)
        assert "error" not in result, result
        assert result["status"] == 200, result
        assert result["doc"]["entries"][0]["studyUid"] == STUDY_A, result


def test_staged_census_refusal_is_not_reported_as_missing(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """P2 census suppression: an EXISTING but malformed staged destination
    (unexpected layout entries) must surface the actionable
    target_authority.dest_layout refusal — never staged-missing with the
    false 'will re-stage on open' promise that a later open would refuse."""
    from services.workstation.recent import probe_recent_availability

    seeded, src, stage, _ = seeded_env
    _ = seeded
    dest = stage / STUDY_A
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "junk.txt").write_text("malformed staged layout", encoding="utf-8")

    with pytest.raises(Refusal) as exc_info:
        probe_recent_availability(str(src), str(stage), STUDY_A)
    assert exc_info.value.code == "target_authority.dest_layout"

    # Same over HTTP: 400 + code, not 200 staged-missing.
    with running_server(seeded.app) as (srv, base_url):
        params = {"source": str(src), "stage": str(stage), "study": STUDY_A}
        res = _safe_request(
            srv, base_url, f"/api/library/recent/probe?{urlencode(params)}"
        )
        assert res.status == 400
        assert json_value(res)["code"] == "target_authority.dest_layout"


def test_staged_census_enumeration_failure_is_not_reported_as_missing(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C4 (review F7) census-failure refusal: a nested enumeration OSError
    inside the staged census (e.g. a series directory that becomes
    unreadable mid-walk) must surface the actionable
    target_authority.dest_layout refusal — never staged-missing with the
    false 'will re-stage on open' promise. Pre-fix, the broad ``except
    Exception`` fallback in probe_recent_availability degraded any
    non-Refusal census failure to staged_exists=False."""
    from services.workstation.recent import probe_recent_availability

    seeded, src, stage, _ = seeded_env
    _ = seeded
    dest = stage / STUDY_A
    dicom = dest / "dicom" / SERIES_A
    dicom.mkdir(parents=True, exist_ok=True)
    (dicom / "frame0.dcm").write_bytes(b"0" * 64)

    real_iterdir = Path.iterdir

    def failing_iterdir(self: Path):  # type: ignore[no-untyped-def]
        if self == dicom:
            raise PermissionError("injected unreadable series directory")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    try:
        with pytest.raises(Refusal) as exc_info:
            probe_recent_availability(str(src), str(stage), STUDY_A)
    finally:
        monkeypatch.setattr(Path, "iterdir", real_iterdir)
    assert exc_info.value.code == "target_authority.dest_layout"

    # Same over HTTP: 400 + code, not 200 staged-missing.
    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    try:
        with running_server(seeded.app) as (srv, base_url):
            params = {"source": str(src), "stage": str(stage), "study": STUDY_A}
            res = _safe_request(
                srv, base_url, f"/api/library/recent/probe?{urlencode(params)}"
            )
            assert res.status == 400
            assert json_value(res)["code"] == "target_authority.dest_layout"
    finally:
        monkeypatch.setattr(Path, "iterdir", real_iterdir)


def test_duplicate_host_and_origin_headers_are_refused(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """G7 authorization matrix, duplicate-header leg: two Host headers (one
    valid) and two Origin headers (one valid) are each refused 403 —
    the single-value rule in authorize_host/authorize_mutation is not
    bypassable by header repetition."""
    import http.client
    import time

    def _raw_request(
        method: str,
        path: str,
        headers: list[tuple[str, str]],
        body: bytes | None = None,
    ) -> tuple[int, bytes]:
        """Single-attempt raw request with duplicate-header support.

        Retries the transient Windows loopback reset (WinError 10054) at
        THIS level: the duplicate-header assertions are refusal-shape
        assertions (403 + envelope), and a reset carries no refusal content
        — retrying the identical bytes is safe here (both legs are
        refusal-only: the GET reads nothing, and the POST is refused at
        auth BEFORE any state change, so a replay cannot double-apply).
        """
        last_err: Exception | None = None
        for attempt in range(5):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.putrequest(method, path)
                for name, value in headers:
                    conn.putheader(name, value)
                conn.putheader("Connection", "close")
                conn.endheaders(body)
                resp = conn.getresponse()
                return resp.status, resp.read()
            except (ConnectionResetError, OSError) as exc:
                last_err = exc
                time.sleep(0.08 * (attempt + 1))
            finally:
                conn.close()
        assert last_err is not None
        raise last_err

    seeded, _, _, _ = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _ = base_url
        port = int(srv.server_address[1])
        host = f"127.0.0.1:{port}"

        status, raw = _raw_request(
            "GET",
            "/api/library/recent",
            [("Host", host), ("Host", host)],  # duplicate, both valid
        )
        assert status == 403
        assert json.loads(raw.decode("utf-8"))["error"] == "forbidden"

        body = json_body({"entry": _fresh_entry(Path("."), Path("."))})
        status2, raw2 = _raw_request(
            "POST",
            "/api/library/recent",
            [
                ("Host", host),
                ("Origin", f"http://{host}"),
                ("Origin", f"http://{host}"),  # duplicate, both valid
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
            body,
        )
        assert status2 == 403
        assert json.loads(raw2.decode("utf-8"))["error"] == "forbidden"


def test_pending_producer_shape_without_counters_is_valid(
    tmp_path: Path,
) -> None:
    """C5 (review N1) producer/validator contract: the first-open producer
    (app.js setPendingOpen) queues the opened identity WITHOUT server
    counters — no openedSeq/openCount keys at all. The pending validator
    must ACCEPT that shape (defaulting 0 / 1) instead of rejecting it
    with missing-field malformed. A PRESENT counter must still be
    well-formed; the record path clamps both to server-authoritative
    values regardless."""
    from services.workstation.recent import save_pending_open, validate_pending_intent

    src = tmp_path / "src"
    stage = tmp_path / "stage"
    src.mkdir(parents=True)
    stage.mkdir(parents=True)
    store = tmp_path / "store" / "recent_cases.json"

    producer_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": str(src),
        "stageBaseDir": str(stage),
        "destRoot": "",
    }
    assert "openedSeq" not in producer_entry and "openCount" not in producer_entry
    intent = validate_pending_intent(
        {"id": "po-producer-1", "exp": 9999999999999, "entry": dict(producer_entry)}
    )
    assert intent["entry"]["openedSeq"] == 0
    assert intent["entry"]["openCount"] == 1
    saved = save_pending_open(
        store, {"id": "po-producer-1", "exp": 9999999999999, "entry": dict(producer_entry)}
    )
    assert saved["id"] == "po-producer-1"

    # Hostile counters through the pending path are still refused.
    for bad_counters in (
        {"openedSeq": -1, "openCount": 1},
        {"openedSeq": 0, "openCount": 0},
        {"openedSeq": True, "openCount": 1},
        {"openedSeq": 0, "openCount": "99"},
    ):
        bad = dict(producer_entry)
        bad.update(bad_counters)
        with pytest.raises(RecentError):
            validate_pending_intent(
                {"id": "po-bad", "exp": 9999999999999, "entry": bad}
            )


def test_pending_intent_lifecycle_round_trip(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """P1 pending slot (HTTP): queue → read → clear with id correlation;
    hostile/expired content never serves; the slot is single and replaced."""
    seeded, src, stage, store_file = seeded_env
    _ = store_file
    study_dir = src / "study_pend"
    write_synthetic_cine(study_dir, series_uid=SERIES_A, study_uid=STUDY_A)
    intent_entry = {
        "studyUid": STUDY_A,
        "seriesUid": SERIES_A,
        "scanSourcePath": str(study_dir),
        "stageBaseDir": str(stage),
        "destRoot": "",
        "openedSeq": 0,
        "openCount": 1,
    }
    with running_server(seeded.app) as (srv, base_url):
        # Empty slot reads absent.
        res0 = _safe_request(srv, base_url, "/api/library/recent/pending")
        assert res0.status == 200
        assert json_value(res0) == {"pending": False}

        # Queue the intent (server validates the entry fail-closed).
        res1 = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body(
                {"id": "pend-1", "exp": 9999999999999, "entry": intent_entry}
            ),
        )
        assert res1.status == 200
        assert json_value(res1) == {
            "pending": True,
            "id": "pend-1",
            "exp": 9999999999999,
        }

        # Read-back carries the full entry for the bound destination.
        res2 = _safe_request(srv, base_url, "/api/library/recent/pending")
        got = json_value(res2)
        assert got["pending"] is True
        assert got["id"] == "pend-1"
        assert got["entry"]["studyUid"] == STUDY_A

        # Wrong-id clear is a no-op (returns cleared False, slot kept).
        res3 = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body({"clear": True, "id": "pend-other"}),
        )
        assert json_value(res3) == {"pending": False, "cleared": False}
        assert json_value(
            _safe_request(srv, base_url, "/api/library/recent/pending")
        )["pending"] is True

        # Correlated clear drops the slot.
        res4 = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body({"clear": True, "id": "pend-1"}),
        )
        assert json_value(res4) == {"pending": False, "cleared": True}
        assert json_value(
            _safe_request(srv, base_url, "/api/library/recent/pending")
        ) == {"pending": False}

        # Hostile entry content is refused at queue time, never stored.
        res5 = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body(
                {
                    "id": "pend-hostile",
                    "exp": 9999999999999,
                    "entry": {"studyUid": "", "seriesUid": None},
                }
            ),
        )
        assert res5.status == 400
        assert json_value(
            _safe_request(srv, base_url, "/api/library/recent/pending")
        ) == {"pending": False}


def _pending_intent(entry: dict[str, Any], intent_id: str, exp: int) -> dict[str, Any]:
    """Queue-shaped pending intent (full server-side entry, producer hint id)."""
    return {"id": intent_id, "exp": exp, "entry": dict(entry)}


def test_pending_clear_serializes_with_queue_replace_under_real_overlap(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3 review #2 F2 (real overlap, barrier-controlled): the correlated
    clear's compare-and-unlink must be atomic against a queue replacement.

    The clear is paused INSIDE its comparison, below the pending-slot lock.
    A concurrent queue then installs B. Serialized shape: the queue waits, and
    B survives. Pre-fix shape: B installs into the comparison/unlink gap and
    the resumed clear deletes B while reporting success for A — the next
    successfully opened case then has no durable intent to consume. No sleep
    decides the outcome: the clear's pause is a barrier and the queue's
    blocking is bounded-wait observed."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    save_pending_open(store, _pending_intent(entry, "intent-a", 9999999999999))

    clear_paused = threading.Event()
    release_clear = threading.Event()
    queue_done = threading.Event()
    real_read = recent_mod._read_pending_bytes
    clear_result: dict[str, Any] = {}
    queue_result: dict[str, Any] = {}

    def paused_read(pending_path: Path) -> bytes:
        # Pause AFTER the comparison read returns A's bytes (the comparison is
        # done by then): pre-fix, B installs in the compare/unlink gap and the
        # resumed clear deletes it; fixed, B blocks on the pending lock.
        data = real_read(pending_path)
        if threading.current_thread().name == "pending-clear":
            clear_paused.set()
            assert release_clear.wait(30), "clear barrier was never released"
        return data

    def run_clear() -> None:
        try:
            clear_result["cleared"] = clear_pending_open(store, "intent-a")
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            clear_result["error"] = exc

    def run_queue_b() -> None:
        try:
            queue_result["saved"] = save_pending_open(
                store, _pending_intent(entry, "intent-b", 9999999999999)
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            queue_result["error"] = exc
        finally:
            queue_done.set()

    monkeypatch.setattr(recent_mod, "_read_pending_bytes", paused_read)
    clear_thread = threading.Thread(target=run_clear, name="pending-clear")
    queue_thread = threading.Thread(target=run_queue_b, name="pending-queue")
    try:
        clear_thread.start()
        assert clear_paused.wait(30), "clear never reached its comparison"
        queue_thread.start()
        # Pre-fix: the replacement is not serialized, so it completes inside
        # the paused comparison window. Fixed: it blocks on the pending lock
        # and this bounded wait expires before the clear is released.
        queue_done.wait(1.0)
    finally:
        release_clear.set()
        clear_thread.join(30)
        queue_thread.join(30)

    assert "error" not in clear_result, clear_result
    assert clear_result["cleared"] is True, clear_result
    assert "error" not in queue_result, queue_result
    assert queue_result["saved"]["id"] == "intent-b", queue_result
    survivor = load_pending_open(store)
    assert survivor is not None, "queue replacement B was deleted by the clear"
    assert survivor["id"] == "intent-b", survivor


def test_pending_expiry_cleanup_serializes_with_queue_replace(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3 review #2 F2: the load-time EXPIRY removal must not delete an intent
    queued after it. Barrier-controlled: the loader reads the expired intent and
    pauses; a queue installs a fresh B; the loader resumes and removes "the"
    slot. Serialized shape keeps B; the pre-fix shape deletes the fresh
    replacement, stranding the newest successfully opened case."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    save_pending_open(store, _pending_intent(entry, "intent-expired", 1))

    loader_paused = threading.Event()
    release_loader = threading.Event()
    queue_done = threading.Event()
    real_read = recent_mod._read_pending_bytes
    loader_result: dict[str, Any] = {}
    queue_result: dict[str, Any] = {}

    def paused_read(pending_path: Path) -> bytes:
        # Pause AFTER the loader read the EXPIRED intent's bytes, before its
        # parse/unlink: pre-fix the fresh replacement is deleted by the resumed
        # expiry cleanup; fixed the replacement blocks on the pending lock.
        data = real_read(pending_path)
        if threading.current_thread().name == "pending-loader":
            loader_paused.set()
            assert release_loader.wait(30), "loader barrier was never released"
        return data

    def run_loader() -> None:
        try:
            loader_result["intent"] = load_pending_open(store)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            loader_result["error"] = exc

    def run_queue_b() -> None:
        try:
            queue_result["saved"] = save_pending_open(
                store, _pending_intent(entry, "intent-fresh", 9999999999999)
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            queue_result["error"] = exc
        finally:
            queue_done.set()

    monkeypatch.setattr(recent_mod, "_read_pending_bytes", paused_read)
    loader_thread = threading.Thread(target=run_loader, name="pending-loader")
    queue_thread = threading.Thread(target=run_queue_b, name="pending-queue")
    try:
        loader_thread.start()
        assert loader_paused.wait(30), "loader never read the expired intent"
        queue_thread.start()
        queue_done.wait(1.0)
    finally:
        release_loader.set()
        loader_thread.join(30)
        queue_thread.join(30)

    assert "error" not in loader_result, loader_result
    assert loader_result["intent"] is None, loader_result  # expired -> dropped
    assert "error" not in queue_result, queue_result
    survivor = load_pending_open(store)
    assert survivor is not None, "expiry cleanup deleted the fresh replacement"
    assert survivor["id"] == "intent-fresh", survivor


def test_pending_wrong_id_and_correlated_clear_are_exact(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """F2 companion (sequential exactness): a wrong-id clear never drops a
    different intent, and the correlated clear drops exactly the intent it
    names."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    save_pending_open(store, _pending_intent(entry, "intent-exact", 9999999999999))
    assert clear_pending_open(store, "intent-other") is False
    kept = load_pending_open(store)
    assert kept is not None and kept["id"] == "intent-exact", kept
    assert clear_pending_open(store, "intent-exact") is True
    assert load_pending_open(store) is None


def test_pending_malformed_bytes_take_deliberate_cleanup_and_recover(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """PR3 review #4 F3 (live route): unusable pending-slot bytes (truncated
    JSON, invalid UTF-8) must reach the deliberate invalid-content cleanup —
    the slot is dropped under the pending lock — instead of escaping as a raw
    decode ValueError through the generic HTTP handler (which answers
    invalid_request without no-store and leaves the unusable slot on disk).

    Chosen frozen response (documented in the report): the same fail-closed
    shape the hostile-content branch already uses, because an unusable client
    hint is not a store failure — ``{pending: false}`` 200 with no-store. The
    id-correlated clear answers the same frozen shape, and the next queue
    recovers without manual file repair."""
    seeded, src, stage, store_file = seeded_env
    study_dir = src / "study_pending_bytes"
    write_synthetic_cine(study_dir, series_uid=SERIES_A, study_uid=STUDY_A)
    entry = _fresh_entry(study_dir, stage)
    pending_path = recent_mod.get_pending_open_path(store_file)

    def _plant(raw: bytes) -> None:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_bytes(raw)

    with running_server(seeded.app) as (srv, base_url):
        # Truncated JSON and invalid UTF-8 both read as an unusable slot:
        # frozen absent response, no-store, and nothing left on disk.
        for raw, label in (
            (b'{"id": "pend-truncated", "exp": 99', "truncated-json"),
            (b"\xff\xfe\x00\x01{\x00", "invalid-utf8"),
            # PR3 review r4310080008: decoder resource failure (RecursionError
            # at ~6,000 nested arrays) and a JSON integer beyond the
            # interpreter digit limit (plain ValueError) are under the byte
            # cap; both must take this same cleanup path.
            (("[" * 6000 + "]" * 6000).encode("utf-8"), "nested-json"),
            (
                b'{"id": "pend-digits", "exp": 99, "tamper": ' + b"9" * 4301 + b"}",
                "oversized-int",
            ),
        ):
            _plant(raw)
            res = _safe_request(srv, base_url, "/api/library/recent/pending")
            assert res.status == 200, (label, res.status, json_value(res))
            assert json_value(res) == {"pending": False}, (label, json_value(res))
            assert res.headers.get("Cache-Control") == "no-store", (
                label,
                res.headers,
            )
            assert not pending_path.exists(), (label, "unusable slot left on disk")

        # The id-correlated clear on malformed bytes answers the frozen
        # response too (never the generic handler) and still cleans the slot.
        _plant(b'{"broken":')
        clear = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body({"clear": True, "id": "pend-malformed"}),
        )
        assert clear.status == 200, (clear.status, json_value(clear))
        assert json_value(clear) == {"pending": False, "cleared": True}, json_value(
            clear
        )
        assert clear.headers.get("Cache-Control") == "no-store", clear.headers
        assert not pending_path.exists()

        # Recovery without manual file repair: the next queue installs a real
        # intent and it round-trips.
        queued = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body(_pending_intent(entry, "pend-recovered", 9999999999999)),
        )
        assert queued.status == 200, (queued.status, json_value(queued))
        assert json_value(queued)["pending"] is True, json_value(queued)
        back = _safe_request(srv, base_url, "/api/library/recent/pending")
        assert json_value(back)["pending"] is True, json_value(back)
        assert json_value(back)["id"] == "pend-recovered", json_value(back)


NONFINITE_EXP_LITERALS = (
    # (wire literal, label). The literal is spliced verbatim into the request
    # body; each one decodes to a value the pre-fix type check accepted:
    # 1e309 -> +inf (valid JSON number), -1e309 -> -inf, NaN/Infinity are the
    # decoder-accepted non-JSON constants, and the 400-digit integer is finite
    # but beyond float range (+- it must not raise OverflowError out of the
    # validator either).
    ("1e309", "json-number-overflow"),
    ("-1e309", "json-number-underflow"),
    ("NaN", "json-nan-constant"),
    ("Infinity", "json-infinity-constant"),
    ("9" * 400, "int-beyond-float-range"),
)


def _raw_pending_body(
    literal: str, entry: dict[str, Any], intent_id: str, *, include_open_id: bool = True
) -> bytes:
    """Pending intent body carrying ``exp`` as the VERBATIM literal.

    For POSTs, ``open_id`` is included so the retrying request helper sees it
    and sends these exact bytes instead of re-serializing the payload (which
    would normalize ``1e309`` to ``Infinity``); the pending route ignores
    unknown top-level fields. For an ON-DISK slot the flag must be False:
    ``validate_pending_intent`` rejects unknown fields, so a planted
    ``open_id`` would make the slot invalid for the wrong reason and the
    witness would be vacuous.
    """
    body = (
        '{"entry": '
        + json.dumps(entry, sort_keys=True, separators=(",", ":"))
        + ', "exp": '
        + literal
        + ', "id": '
        + json.dumps(intent_id)
    )
    if include_open_id:
        body += ', "open_id": ' + json.dumps("witness-" + literal[:12])
    return (body + "}").encode("utf-8")


@pytest.mark.parametrize("literal,label", NONFINITE_EXP_LITERALS)
def test_pending_post_nonfinite_expiry_refused_and_prior_slot_preserved(
    seeded_env: tuple[SeededStudy, Path, Path, Path], literal: str, label: str
) -> None:
    """PR3 review r4311421633 (live route, fail-capable): a pending POST whose
    ``exp`` is non-finite (or beyond float range) must be REFUSED before
    anything is written.

    Pre-fix: the type check accepted every literal, ``_json_bytes`` re-emitted
    ``Infinity``/``NaN`` (or a 400-digit integer a browser reads as Infinity)
    in a 200 body that ``response.json()`` cannot parse, and the stored intent
    never expired because ``now > exp`` is false for inf/NaN. Post-fix the
    write is refused with the frozen ``recent.malformed`` envelope plus
    ``no-store``, and a previously queued GOOD slot survives byte-for-byte.
    """
    seeded, src, stage, store_file = seeded_env
    study_dir = src / "study_pending_nonfinite_post"
    write_synthetic_cine(study_dir, series_uid=SERIES_A, study_uid=STUDY_A)
    entry = _fresh_entry(study_dir, stage)
    pending_path = recent_mod.get_pending_open_path(store_file)

    with running_server(seeded.app) as (srv, base_url):
        good = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body(_pending_intent(entry, "pend-good", 9999999999999)),
        )
        assert good.status == 200, (good.status, json_value(good))
        assert json_value(good)["pending"] is True, json_value(good)
        good_bytes = pending_path.read_bytes()

        hostile = _raw_pending_body(literal, entry, "pend-hostile-" + label)
        res = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=hostile,
        )
        assert res.status == 400, (label, res.status, json_value(res))
        assert json_value(res)["code"] == "recent.malformed", (label, json_value(res))
        assert res.headers.get("Cache-Control") == "no-store", (label, dict(res.headers))
        assert pending_path.read_bytes() == good_bytes, (
            label,
            "refused write changed the prior good slot",
        )

        # The preserved slot still serves a FROZEN, browser-parseable body.
        back = _safe_request(srv, base_url, "/api/library/recent/pending")
        assert back.status == 200, (label, back.status, json_value(back))
        assert json_value(back)["pending"] is True, (label, json_value(back))
        assert json_value(back)["id"] == "pend-good", (label, json_value(back))
        assert b"Infinity" not in back.body and b"NaN" not in back.body, (
            label,
            back.body,
        )


@pytest.mark.parametrize("literal,label", NONFINITE_EXP_LITERALS)
def test_pending_slot_nonfinite_expiry_takes_deliberate_cleanup(
    seeded_env: tuple[SeededStudy, Path, Path, Path], literal: str, label: str
) -> None:
    """PR3 review r4311421633 (live route, fail-capable): a PLANTED pending
    slot whose ``exp`` is non-finite (or beyond float range) is unusable
    content and must take the existing deliberate invalid-content cleanup.

    Pre-fix: GET served ``200 {"pending": true, "exp": Infinity|NaN}`` — a body
    no browser ``response.json()`` can parse — and the slot never expired, so
    the malformed transient slot stayed on disk and kept breaking confirmation.
    Post-fix: the frozen ``{pending: false}`` 200 with ``no-store``, the slot
    is dropped under the pending lock, and the next queue round-trips without
    manual file repair.
    """
    seeded, src, stage, store_file = seeded_env
    study_dir = src / "study_pending_nonfinite_slot"
    write_synthetic_cine(study_dir, series_uid=SERIES_A, study_uid=STUDY_A)
    entry = _fresh_entry(study_dir, stage)
    pending_path = recent_mod.get_pending_open_path(store_file)

    with running_server(seeded.app) as (srv, base_url):
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_bytes(
            _raw_pending_body(literal, entry, "pend-planted", include_open_id=False)
        )

        res = _safe_request(srv, base_url, "/api/library/recent/pending")
        assert res.status == 200, (label, res.status, json_value(res))
        assert json_value(res) == {"pending": False}, (label, json_value(res))
        assert res.headers.get("Cache-Control") == "no-store", (
            label,
            dict(res.headers),
        )
        assert b"Infinity" not in res.body and b"NaN" not in res.body, (
            label,
            res.body,
        )
        assert not pending_path.exists(), (label, "unusable slot left on disk")

        # Recovery without manual file repair.
        queued = _safe_request(
            srv,
            base_url,
            "/api/library/recent/pending",
            method="POST",
            body=json_body(_pending_intent(entry, "pend-recovered", 9999999999999)),
        )
        assert queued.status == 200, (label, queued.status, json_value(queued))
        assert json_value(queued)["pending"] is True, (label, json_value(queued))
        back = _safe_request(srv, base_url, "/api/library/recent/pending")
        assert json_value(back)["pending"] is True, (label, json_value(back))
        assert json_value(back)["id"] == "pend-recovered", (label, json_value(back))


def test_main_document_unavailable_handle_provenance_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuous guard: when handle provenance is unavailable, the main
    document is refused rather than read from an unbound handle. (The
    sidecar can degrade to its document field; these bytes have no such
    second source, so fail-closed is the only safe behaviour.)"""
    _src, _stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"version": RECENT_CASES_SCHEMA_VERSION, "entries": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(recent_mod, "_handle_location", lambda _fd: None)
    with pytest.raises(RecentError) as exc:
        load_recent_cases(store)
    assert exc.value.code == "recent.store_unreadable"


def test_oversized_display_field_is_refused_before_replace(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """A single display-only field that serializes past the load bound must
    be refused with an actionable code BEFORE any replacement — the prior
    bytes survive and the store stays usable (a later clear still works)."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    doc1 = process_recent_post(store, {"entry": entry})
    assert doc1["entries"][0]["openCount"] == 1
    last_good = store.read_bytes()

    oversized = _fresh_entry(src, stage)
    oversized["destRoot"] = "D" * (recent_mod.MAX_RECENT_FILE_BYTES + 1024)
    with pytest.raises(RecentError) as exc:
        process_recent_post(store, {"entry": oversized})
    assert exc.value.code == "recent.document_too_large"
    assert store.read_bytes() == last_good

    # The store is still usable after the refusal: an empty-list clear works.
    cleared = process_recent_post(store, {"entries": []})
    assert cleared["entries"] == []
    assert cleared["reason"] == "none"


def test_cumulative_document_growth_is_refused_before_replace(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """Cumulative growth past the load bound refuses the same way: every
    single row is valid, but the serialized WHOLE no longer fits, so the
    save is refused and the prior document is preserved byte-for-byte."""
    src, stage, store = recent_env
    store.parent.mkdir(parents=True, exist_ok=True)

    # Build cumulative rows whose SERVER-MERGED serialization (fresh
    # openedSeq/openCount, capped to RECENT_CASES_LIMIT rows) exceeds the
    # load bound: probe against the exact merge output shape, not the raw
    # client rows.
    rows: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    index = 0
    while True:
        row = _fresh_entry(src, stage)
        row["studyUid"] = f"{UID_BASE}.{5000 + index}"
        row["destRoot"] = f"display-padding-{index}-" + ("p" * 8000)
        rows.append(row)
        merged = [
            {
                "studyUid": r["studyUid"],
                "seriesUid": r["seriesUid"],
                "scanSourcePath": r["scanSourcePath"],
                "stageBaseDir": r["stageBaseDir"],
                "destRoot": r["destRoot"],
                "openedSeq": pos + 1,
                "openCount": 1,
            }
            for pos, r in enumerate(rows[: recent_mod.RECENT_CASES_LIMIT])
        ]
        probe = {"entries": merged, "high_water_seq": 0, "version": 1}
        index += 1
        if len(recent_mod._json_bytes(probe)) > recent_mod.MAX_RECENT_FILE_BYTES:
            break
        assert index < 100, "padding loop did not converge"
    assert len(merged) >= 2

    with pytest.raises(RecentError) as exc:
        process_recent_post(store, {"entries": rows})
    assert exc.value.code == "recent.document_too_large"
    assert not store.exists()

    small = process_recent_post(store, {"entry": _fresh_entry(src, stage)})
    assert len(small["entries"]) == 1


def test_persist_then_relaunch_monotonic_opened_seq(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC2: POST then GET persists entries with monotonic openedSeq from high-water mark."""
    seeded, src, stage, store_file = seeded_env
    study_a_dir = src / "study_a"
    write_synthetic_cine(study_a_dir, series_uid=SERIES_A, study_uid=STUDY_A)

    with running_server(seeded.app) as (srv, base_url):
        # Warmup loopback socket so Windows threaded server is actively listening
        _safe_request(srv, base_url, "/api/studies")

        entry_payload = {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": SERIES_A,
                "scanSourcePath": str(study_a_dir),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 9999,  # Client openedSeq must be IGNORED
                "openCount": 1,
            },
        }
        post_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(entry_payload),
        )
        assert post_res.status == 200
        post_data = json_value(post_res)
        assert len(post_data["entries"]) == 1
        # POST success body must expose the SAME document shape as GET
        assert post_data["discarded"] == 0
        entry_saved = post_data["entries"][0]
        assert entry_saved["studyUid"] == STUDY_A
        first_seq = entry_saved["openedSeq"]
        assert first_seq >= 1
        assert entry_saved["openCount"] == 1

        # GET confirms persistence
        get_res = _safe_request(srv, base_url, "/api/library/recent")
        assert get_res.status == 200
        get_data = json_value(get_res)
        assert len(get_data["entries"]) == 1
        assert get_data["entries"][0]["openedSeq"] == first_seq
        assert set(post_data) == set(get_data)

    # Simulate relaunch with a second server on the same store
    with running_server(seeded.app) as (srv2, base_url2):
        get_relaunch = _safe_request(srv2, base_url2, "/api/library/recent")
        assert get_relaunch.status == 200
        relaunch_seq = json_value(get_relaunch)["entries"][0]["openedSeq"]
        assert relaunch_seq == first_seq  # Preserved across relaunch

        # Second open accumulates openCount and advances openedSeq monotonically
        entry_payload2 = {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": SERIES_A,
                "scanSourcePath": str(study_a_dir),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            },
        }
        post_res2 = _safe_request(
            srv2,
            base_url2,
            "/api/library/recent",
            method="POST",
            body=json_body(entry_payload2),
        )
        assert post_res2.status == 200
        updated = json_value(post_res2)["entries"][0]
        assert updated["openedSeq"] > relaunch_seq  # Monotonic increment
        assert updated["openCount"] >= 2  # Accumulated openCount


def test_mru_ordering_and_cap_limit(seeded_env: tuple[SeededStudy, Path, Path, Path]) -> None:
    """AC2: MRU ordering (newest front) and bounded limit cap (max 10 entries)."""
    seeded, src, stage, _ = seeded_env

    with running_server(seeded.app) as (srv, base_url):
        for i in range(12):
            uid = f"{UID_BASE}.{100 + i}"
            sdir = src / f"study_{i}"
            sdir.mkdir(parents=True, exist_ok=True)
            payload = {
                "entry": {
                    "studyUid": uid,
                    "seriesUid": None,
                    "scanSourcePath": str(sdir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / uid),
                    "openedSeq": 0,
                    "openCount": 1,
                }
            }
            res = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body(payload),
            )
            assert res.status == 200

        get_res = _safe_request(srv, base_url, "/api/library/recent")
        data = json_value(get_res)
        entries = data["entries"]
        assert len(entries) == RECENT_CASES_LIMIT
        # Most recent (i=11) is first and sequences are strictly descending
        assert entries[0]["studyUid"] == f"{UID_BASE}.111"
        assert entries[0]["openedSeq"] >= 12
        for idx in range(len(entries) - 1):
            assert entries[idx]["openedSeq"] > entries[idx + 1]["openedSeq"]
        # Oldest kept is i=2
        assert entries[-1]["studyUid"] == f"{UID_BASE}.102"


def test_client_opencount_never_authority(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """Contract merge_semantics: client openCount NEVER authority (fixed +1 per open)."""
    seeded, src, stage, _ = seeded_env
    dir_a = src / "study_a"
    dir_b = src / "study_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    def _post(entry: dict[str, Any]) -> tuple[int, Any]:
        res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": entry}),
        )
        return res.status, json_value(res)

    with running_server(seeded.app) as (srv, base_url):
        # 1. First open of A with a legitimate counter
        status, doc = _post(
            {
                "studyUid": STUDY_A,
                "seriesUid": SERIES_A,
                "scanSourcePath": str(dir_a),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        )
        assert status == 200
        assert doc["entries"][0]["openCount"] == 1

        # 2. Inflation attempt: client sends openCount=999 for the SAME
        #    identity -> accumulates to exactly prior+1 (server +1), never 1000.
        status, doc = _post(
            {
                "studyUid": STUDY_A,
                "seriesUid": SERIES_A,
                "scanSourcePath": str(dir_a),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 999,
            }
        )
        assert status == 200
        assert doc["entries"][0]["openCount"] == 2  # exactly prior+1, NOT 1000

        # 3. New identity via multi-row import with inflated counters -> clamped to 1
        res3 = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(
                {
                    "entries": [
                        {
                            "studyUid": STUDY_A,
                            "seriesUid": SERIES_A,
                            "scanSourcePath": str(dir_a),
                            "stageBaseDir": str(stage),
                            "destRoot": str(stage / STUDY_A),
                            "openedSeq": 0,
                            "openCount": 500,
                        },
                        {
                            "studyUid": STUDY_B,
                            "seriesUid": SERIES_B,
                            "scanSourcePath": str(dir_b),
                            "stageBaseDir": str(stage),
                            "destRoot": str(stage / STUDY_B),
                            "openedSeq": 0,
                            "openCount": 500,
                        },
                    ]
                }
            ),
        )
        status = res3.status
        doc = json_value(res3)
        assert status == 200
        by_uid = {e["studyUid"]: e for e in doc["entries"]}
        assert by_uid[STUDY_A]["openCount"] == 3  # still exactly prior+1
        assert by_uid[STUDY_B]["openCount"] == 1  # new identity clamped to 1


def test_source_identity_key_is_os_aware_canonical_form() -> None:
    """PR3 review #4 F1 (unit): identity is the OS-aware canonical form of the
    validated source spelling — dot segments and redundant separators collapse,
    case folds only where the platform path identity is case-insensitive, and
    distinct paths stay distinct. Lexical only (no filesystem access), never
    consulted for path-safety decisions."""
    key = recent_mod.safe_canonical_path_key

    # Dot-segment / redundant-separator / separator-style equivalents are ONE
    # key (`/copy/library` and `/copy/./library` are the reviewer's example).
    assert key("/copy/library") == "/copy/library"
    assert key("/copy/./library") == "/copy/library"
    assert key("/copy//library/") == "/copy/library"
    assert key("/copy/other/../library") == "/copy/library"
    assert key("\\copy\\library") == "/copy/library"

    # Distinct paths stay distinct.
    assert key("/copy/library") != key("/copy/library2")
    assert key("/copy/library") != key("/copy/other")

    # Degenerate input never yields an empty key segment.
    assert key("") == "/"
    assert key("/") == "/"

    original = recent_mod._CASE_FOLD_PATH_IDENTITY
    try:
        # Windows semantics: case-insensitive identity, drive letter included.
        recent_mod._CASE_FOLD_PATH_IDENTITY = True
        assert key("/Copy/Library") == "/copy/library"
        assert key("C:\\Copy\\Library") == "c:/copy/library"
        # POSIX semantics: case-distinct paths stay distinct.
        recent_mod._CASE_FOLD_PATH_IDENTITY = False
        assert key("/Copy/Library") == "/Copy/Library"
        assert key("/Copy/Library") != key("/copy/library")
    finally:
        recent_mod._CASE_FOLD_PATH_IDENTITY = original

    # The live host folds case exactly when its filesystem identity does.
    assert recent_mod._CASE_FOLD_PATH_IDENTITY is (os.name == "nt")


def test_equivalent_source_spellings_merge_into_one_recent_identity(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """PR3 review #4 F1 (live route): opening the same physical source spelled
    with dot segments or redundant separators (and, on Windows, different case)
    accumulates openCount on the EXISTING row instead of inserting a duplicate
    row with openCount 1 that can evict a different case. The raw validated
    spelling remains the stored/displayed one, and the path-safety validation
    still runs on that original spelling (unchanged)."""
    seeded, src, stage, _ = seeded_env
    study_dir = src / "study_identity"
    write_synthetic_cine(study_dir, series_uid=SERIES_A, study_uid=STUDY_A)
    other_dir = src / "study_identity_other"
    write_synthetic_cine(other_dir, series_uid=SERIES_B, study_uid=STUDY_B)
    base = str(study_dir)
    slash_base = base.replace("\\", "/")

    def _entry(
        source: str, study_uid: str = STUDY_A, series_uid: str | None = SERIES_A
    ) -> dict[str, Any]:
        return {
            "studyUid": study_uid,
            "seriesUid": series_uid,
            "scanSourcePath": source,
            "stageBaseDir": str(stage),
            "destRoot": str(stage / study_uid),
            "openedSeq": 0,
            "openCount": 1,
        }

    def _identity_key(entry: dict[str, Any]) -> str:
        return recent_mod.recent_case_key(entry)

    def _row(doc: Any, entry: dict[str, Any]) -> dict[str, Any]:
        want = _identity_key(entry)
        rows = [e for e in doc["entries"] if recent_mod.recent_case_key(e) == want]
        assert len(rows) == 1, (entry["scanSourcePath"], doc)
        return rows[0]

    with running_server(seeded.app) as (srv, base_url):

        def _post(entry: dict[str, Any]) -> tuple[int, Any]:
            res = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body({"entry": entry}),
            )
            return res.status, json_value(res)

        status, doc = _post(_entry(base))
        assert status == 200, doc
        assert len(doc["entries"]) == 1, doc
        assert _row(doc, _entry(base))["openCount"] == 1, doc

        # A DIFFERENT source keeps its own identity: merging is not a collapse.
        other_entry = _entry(str(other_dir), STUDY_B, SERIES_B)
        status, doc = _post(other_entry)
        assert status == 200, doc
        assert len(doc["entries"]) == 2, doc
        assert _row(doc, _entry(base))["openCount"] == 1, doc
        assert _row(doc, other_entry)["openCount"] == 1, doc

        # Equivalent spellings of the SAME source accumulate on one row, and
        # the stored spelling stays the raw validated one (identity is separate
        # from display text).
        for expected_count, variant in enumerate(
            (base + "/.", slash_base + "//", slash_base + "/./"), start=2
        ):
            variant_entry = _entry(variant)
            status, doc = _post(variant_entry)
            assert status == 200, (variant, doc)
            assert len(doc["entries"]) == 2, (variant, doc)
            row = _row(doc, variant_entry)
            assert row["openCount"] == expected_count, (variant, doc)
            assert row["scanSourcePath"] == variant, (variant, doc)

        # Windows path identity is case-insensitive, so a case variant of the
        # SAME live directory is the same case; on POSIX it stays a distinct
        # identity (and must never accumulate onto the lowercase row).
        status, doc = _post(_entry(base.upper()))
        assert status == 200, doc
        if os.name == "nt":
            assert len(doc["entries"]) == 2, doc
            assert _row(doc, _entry(base))["openCount"] == 5, doc
        else:
            assert len(doc["entries"]) == 3, doc
            assert _row(doc, _entry(base))["openCount"] == 4, doc
            assert _row(doc, _entry(base.upper()))["openCount"] == 1, doc


def test_stale_snapshot_rejection_and_merge_retry(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC3: Stale base-digest gets 409 recent.stale_snapshot; retry with merge succeeds."""
    seeded, src, stage, _ = seeded_env
    dir_a = src / "study_a"
    dir_b = src / "study_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    with running_server(seeded.app) as (srv, base_url):
        # 1. Initial write
        p1 = {
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": SERIES_A,
                "scanSourcePath": str(dir_a),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        res1 = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(p1),
        )
        assert res1.status == 200
        init_digest = json_value(res1)["digest"]

        # 2. Advance store with study B
        p2 = {
            "entry": {
                "studyUid": STUDY_B,
                "seriesUid": SERIES_B,
                "scanSourcePath": str(dir_b),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_B),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        res2 = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(p2),
        )
        assert res2.status == 200
        new_digest = json_value(res2)["digest"]

        # 3. Attempt write using the OLD snapshot digest
        stale_write = {
            "entry": {
                "studyUid": STUDY_C,
                "seriesUid": SERIES_C,
                "scanSourcePath": str(dir_a),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_C),
                "openedSeq": 0,
                "openCount": 1,
            },
            "base_digest": init_digest,
        }
        stale_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(stale_write),
        )
        assert stale_res.status == 409
        err = json_value(stale_res)
        assert err["error"] == "frozen_error"
        assert err["code"] == "recent.stale_snapshot"
        assert "current" in err
        assert err["current"]["digest"] == new_digest

        # 4. Retry with current digest succeeds
        stale_write["base_digest"] = err["current"]["digest"]
        retry_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(stale_write),
        )
        assert retry_res.status in {200, 409}
        if retry_res.status == 200:
            assert json_value(retry_res)["entries"][0]["studyUid"] == STUDY_C
        else:
            # Under socket RST retry, the initial attempt may have already committed
            assert json_value(retry_res)["current"]["entries"][0]["studyUid"] == STUDY_C


def test_concurrent_racing_posts_serialize(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC3: Racing POSTs serialize cleanly under in-process lock without lost updates."""
    seeded, src, stage, _ = seeded_env

    with running_server(seeded.app) as (srv, base_url):
        def _post_one(index: int) -> int:
            uid = f"{UID_BASE}.200.{index}"
            sdir = src / f"race_{index}"
            sdir.mkdir(parents=True, exist_ok=True)
            payload = {
                "entry": {
                    "studyUid": uid,
                    "seriesUid": None,
                    "scanSourcePath": str(sdir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / uid),
                    "openedSeq": 0,
                    "openCount": 1,
                }
            }
            res = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body(payload),
            )
            return res.status

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_post_one, i) for i in range(5)]
            statuses = [f.result() for f in futures]

        assert all(s == 200 for s in statuses)
        get_res = _safe_request(srv, base_url, "/api/library/recent")
        entries = json_value(get_res)["entries"]
        assert len(entries) == 5
        seqs = [e["openedSeq"] for e in entries]
        # All assigned sequence numbers must be distinct
        assert len(set(seqs)) == 5


def test_codec_empty_on_corrupt_rows_owner_decision_a(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC4: Single tampered row degrades WHOLE document to empty list with reason corrupt-rows."""
    seeded, src, stage, store_file = seeded_env
    valid_dir = src / "study_valid"
    valid_dir.mkdir(parents=True, exist_ok=True)

    with running_server(seeded.app) as (srv, base_url):
        # 1. Tampered UID row: '../escape'
        corrupt_doc = {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [
                {
                    "studyUid": "../escape",
                    "seriesUid": None,
                    "scanSourcePath": str(valid_dir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / "escape"),
                    "openedSeq": 1,
                    "openCount": 1,
                },
                {
                    "studyUid": STUDY_A,
                    "seriesUid": None,
                    "scanSourcePath": str(valid_dir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / STUDY_A),
                    "openedSeq": 2,
                    "openCount": 1,
                },
            ],
        }
        store_file.parent.mkdir(parents=True, exist_ok=True)
        store_file.write_text(json.dumps(corrupt_doc), encoding="utf-8")

        res = _safe_request(srv, base_url, "/api/library/recent")
        assert res.status == 200
        data = json_value(res)
        assert data["entries"] == []
        assert data["discarded"] == 2
        assert data["reason"] == "corrupt-rows"

        # 2. Duplicate identity rows in store
        dup_doc = {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [
                {
                    "studyUid": STUDY_A,
                    "seriesUid": None,
                    "scanSourcePath": str(valid_dir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / STUDY_A),
                    "openedSeq": 1,
                    "openCount": 1,
                },
                {
                    "studyUid": STUDY_A,
                    "seriesUid": None,
                    "scanSourcePath": str(valid_dir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / STUDY_A),
                    "openedSeq": 2,
                    "openCount": 2,
                },
            ],
        }
        store_file.write_text(json.dumps(dup_doc), encoding="utf-8")
        res_dup = _safe_request(srv, base_url, "/api/library/recent")
        assert res_dup.status == 200
        assert json_value(res_dup)["reason"] == "corrupt-rows"
        assert json_value(res_dup)["entries"] == []


def test_failed_writes_preserve_last_good_and_unreadable_store(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC5: Store unwritable preserves last-good bytes; unreadable store returns 400."""
    seeded, src, stage, store_file = seeded_env
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)

    with running_server(seeded.app) as (srv, base_url):
        # 1. Establish valid store state
        p = {
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": str(sdir),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        ok_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(p),
        )
        assert ok_res.status == 200
        last_good_bytes = store_file.read_bytes()

        # 2. Mock replace failure to test unwritable store
        orig_replace = os.replace

        def _failing_replace(_src: Any, _dst: Any, **_kwargs: Any) -> None:
            # POSIX production replacement is dir_fd-relative and passes
            # src_dir_fd/dst_dir_fd; Windows uses the bare two-argument form.
            # The injector must reject BOTH shapes to test the same failure path.
            raise PermissionError("Access denied mock")

        monkeypatch.setattr(os, "replace", _failing_replace)
        p_fail = {
            "entry": {
                "studyUid": STUDY_B,
                "seriesUid": None,
                "scanSourcePath": str(sdir),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_B),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        fail_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(p_fail),
        )
        assert fail_res.status == 400
        fail_body = json_value(fail_res)
        assert fail_body["error"] == "frozen_error"
        assert fail_body["code"] == "recent.store_unwritable"

        # Last good bytes remain untouched
        assert store_file.read_bytes() == last_good_bytes

        # Restore replace without undoing environment variable
        monkeypatch.setattr(os, "replace", orig_replace)

        # 3. Store unreadable: break the store's link refusal (raised before
        #    any byte is read), proving failure precedes ingestion.
        def _refusing_safety(_path: Any) -> None:
            raise RecentError("recent.store_unreadable", "Unreadable file mock")

        monkeypatch.setattr(recent_mod, "check_store_path_safety", _refusing_safety)
        unread_res = _safe_request(srv, base_url, "/api/library/recent")
        assert unread_res.status == 400
        unread_body = json_value(unread_res)
        assert unread_body["error"] == "frozen_error"
        assert unread_body["code"] == "recent.store_unreadable"


def test_sidecar_failure_after_main_replace_reports_committed_result(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3 review F2 (post-commit sidecar failure, EVERY error class): the
    main-document replace is the explicit commit point. A durable high-water
    (sidecar) failure AFTER that replace must report the COMMITTED result —
    never 400 and never a "prior document preserved" claim while the new rows,
    counters, and replay id are already durable.

    Three injected classes cover the whole post-commit escape set: storage
    failure (RecentError.storage_failure), security refusal as RecentError
    (target_authority.symlink), and a raw Refusal (the widened defensive
    clause). The high-water mark survives in the main document's own
    ``high_water_seq`` field, is read back on restart with its counters, heals
    on the next successful write, and the committed delivery still replays
    idempotently. The injection targets ONLY the sidecar operation that runs
    after the main replacement (load-time sidecar writes inside the same
    request still delegate to production)."""
    seeded, src, stage, store_file = seeded_env
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)
    entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": str(sdir),
        "stageBaseDir": str(stage),
        "destRoot": str(stage / STUDY_A),
        "openedSeq": 0,
        "openCount": 1,
    }
    real_replace = recent_mod._atomic_replace
    real_persist = recent_mod.persist_durable_high_water_seq
    main_replaced = {"n": 0}
    injected: dict[str, Any] = {"factory": None}
    # Per-request arm: the injection must fire ONLY after THIS request's own
    # main-document replace (a global counter would also poison the next
    # request's load-time sidecar write).
    armed = {"value": False}

    def replace_tracker(*args: Any, **kwargs: Any) -> None:
        result = real_replace(*args, **kwargs)
        if args[1] == store_file:
            main_replaced["n"] += 1
            armed["value"] = True
        return result

    def failing_sidecar(store_path: Path, seq: int) -> None:
        # ONLY the sidecar write that runs after this request's main-document
        # replace fails; every earlier sidecar effect delegates to production.
        if armed["value"]:
            raise injected["factory"]()
        return real_persist(store_path, seq)

    injections: list[tuple[str, Any]] = [
        (
            "storage-failure",
            lambda: RecentError.storage_failure(
                "recent.store_unwritable",
                "sidecar replace",
                PermissionError("injected post-commit sidecar failure"),
            ),
        ),
        (
            "security-refusal-recent-error",
            lambda: RecentError(
                "target_authority.symlink",
                "sidecar leaf is a link; refusing fail-closed",
            ),
        ),
        (
            "security-refusal-raw-refusal",
            lambda: Refusal(
                "target_authority.symlink",
                "sidecar leaf is a link; refusing fail-closed",
            ),
        ),
    ]

    committed_digest = ""
    expected_seq = 1
    with running_server(seeded.app) as (srv, base_url):
        seed = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": entry}),
        )
        assert seed.status == 200, json_value(seed)

        monkeypatch.setattr(recent_mod, "_atomic_replace", replace_tracker)
        monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", failing_sidecar)
        for label, factory in injections:
            injected["factory"] = factory
            armed["value"] = False
            open_id = f"sc-{label}"
            expected_seq += 1
            post = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body({"entry": entry, "open_id": open_id}),
            )
            assert main_replaced["n"] == expected_seq - 1, (label, main_replaced)
            assert post.status == 200, (label, json_value(post))
            committed = json_value(post)
            # The COMMITTED document is reported: no frozen error envelope and
            # no "prior document preserved" claim on ANY injected class.
            assert "message" not in committed, (label, committed)
            assert "code" not in committed, (label, committed)
            assert committed["entries"][0]["openCount"] == expected_seq, (label, committed)
            assert committed["entries"][0]["openedSeq"] == expected_seq, (label, committed)
            # ... and the main document already carries the same state.
            raw_now = json.loads(store_file.read_text(encoding="utf-8"))
            assert raw_now["high_water_seq"] == expected_seq, (label, raw_now)
            assert raw_now["entries"][0]["openCount"] == expected_seq, (label, raw_now)
            assert open_id in raw_now.get("seen_open_ids", []), (label, raw_now)
            committed_digest = committed["digest"]

    # Restart with a stale/absent sidecar and no in-process memory: the durable
    # high-water mark AND the counters are read back from the MAIN document.
    monkeypatch.setattr(recent_mod, "_atomic_replace", real_replace)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", real_persist)
    sidecar = get_sidecar_seq_path(store_file)
    if sidecar.exists():
        sidecar.unlink()
    recent_mod._HIGH_WATER_BY_PATH.pop(recent_mod._hw_path_key(store_file), None)
    reloaded = load_recent_cases(store_file)
    assert reloaded["high_water_seq"] == expected_seq, reloaded
    assert reloaded["entries"][0]["openCount"] == expected_seq, reloaded
    # Healed on the subsequent write: the sidecar is re-persisted from the
    # main document's own high-water mark.
    assert sidecar.exists(), "sidecar was not re-persisted from the main document"
    assert sidecar.read_text(encoding="utf-8").strip() == str(expected_seq), sidecar
    assert read_durable_high_water_seq(store_file) == expected_seq

    # Idempotent replay of the last committed delivery: no counter movement and
    # the same digest — the refusal classes never corrupted the replay tail.
    last_open_id = f"sc-{injections[-1][0]}"
    replay = process_recent_post(store_file, {"entry": entry, "open_id": last_open_id})
    assert replay["high_water_seq"] == expected_seq, replay
    assert replay["entries"][0]["openCount"] == expected_seq, replay
    assert replay["digest"] == committed_digest, replay


def test_get_reads_committed_entries_when_only_sidecar_write_fails(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR3 review #2 F1: a readable, valid main document must stay readable when
    only the derived SIDECAR cannot be written (read-only directory, disk full).
    The committed entries are served with the high-water kept as the
    memory/main-document maximum; a genuinely unwritable MAIN store still
    refuses with last-good bytes preserved; sidecar path-SAFETY refusals are NOT
    weakened (they still fail the read closed)."""
    seeded, src, stage, store_file = seeded_env
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)
    entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": str(sdir),
        "stageBaseDir": str(stage),
        "destRoot": str(stage / STUDY_A),
        "openedSeq": 0,
        "openCount": 1,
    }
    real_persist = recent_mod.persist_durable_high_water_seq

    with running_server(seeded.app) as (srv, base_url):
        for _ in range(2):
            post = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body({"entry": entry}),
            )
            assert post.status == 200, json_value(post)
        baseline = _safe_request(srv, base_url, "/api/library/recent")
        assert baseline.status == 200, json_value(baseline)
        baseline_doc = json_value(baseline)
        assert baseline_doc["entries"][0]["openCount"] == 2, baseline_doc
        last_good = store_file.read_bytes()
        hw_key = recent_mod._hw_path_key(store_file)

        # ONLY sidecar writes fail (the main document stays readable/valid).
        def failing_sidecar(store_path: Path, seq: int) -> None:
            raise RecentError.storage_failure(
                "recent.store_unwritable",
                "sidecar replace",
                PermissionError("mock read-only sidecar directory"),
            )

        monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", failing_sidecar)
        got = _safe_request(srv, base_url, "/api/library/recent")
        assert got.status == 200, json_value(got)
        doc = json_value(got)
        assert doc["entries"][0]["openCount"] == 2, doc
        assert doc["digest"] == baseline_doc["digest"], doc
        assert doc["reason"] == "none", doc
        # The loaded maximum is kept in memory (the durable copy of the
        # high-water mark is the main document's own high_water_seq field).
        assert recent_mod._HIGH_WATER_BY_PATH.get(hw_key, 0) >= 2, (
            recent_mod._HIGH_WATER_BY_PATH
        )

        # A genuinely unwritable MAIN store still refuses, last-good preserved:
        # the read-path tolerance must not leak into the write path.
        monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", real_persist)
        real_replace = recent_mod._atomic_replace

        def failing_replace(*args: Any, **kwargs: Any) -> None:
            raise OSError(13, "Access denied (mock read-only main store)")

        monkeypatch.setattr(recent_mod, "_atomic_replace", failing_replace)
        refused = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": entry}),
        )
        assert refused.status == 400, json_value(refused)
        assert json_value(refused)["code"] == "recent.store_unwritable", json_value(refused)
        assert store_file.read_bytes() == last_good
        monkeypatch.setattr(recent_mod, "_atomic_replace", real_replace)

        # Path-SAFETY refusals are not weakened: a sidecar safety refusal still
        # fails the read closed (fail-closed is preserved for hostile paths).
        def refusing_sidecar(store_path: Path, seq: int) -> None:
            raise RecentError(
                "target_authority.symlink",
                "sidecar path refused; high-water mark kept in memory only",
            )

        monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", refusing_sidecar)
        safe = _safe_request(srv, base_url, "/api/library/recent")
        assert safe.status == 400, json_value(safe)
        assert json_value(safe)["code"] == "target_authority.symlink", json_value(safe)
        monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", real_persist)


def test_recent_failure_responses_never_disclose_store_paths(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentinel-path witness: read/stat/write failures whose OSError text
    embeds the server-owned store path must NOT leak that path into the
    frozen route response — only the stable code + fixed client wording."""
    seeded, src, stage, store_file = seeded_env
    sentinel = store_file.parent / "SENTINEL-DO-NOT-LEAK-9f3"
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)
    sentinel_store = sentinel / "nested" / "recent_cases.json"

    sentinel_store.parent.mkdir(parents=True, exist_ok=True)
    sentinel_store.write_text(
        '{"version": 1, "entries": [], "high_water_seq": 0}', encoding="utf-8"
    )
    with running_server(seeded.app) as (srv, base_url):
        real_lstat = Path.lstat

        def failing_stat(self: Any) -> Any:
            if self == store_file:
                raise OSError(13, "Permission denied", str(sentinel_store))
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", failing_stat)
        stat_res = _safe_request(srv, base_url, "/api/library/recent")
        assert stat_res.status == 400
        stat_body = json_value(stat_res)
        assert stat_body["code"] == "recent.store_unreadable"
        assert "SENTINEL-DO-NOT-LEAK-9f3" not in stat_body["message"]
        assert str(sentinel_store) not in stat_body["message"]
        monkeypatch.undo()

        def failing_read(_fd: int, _n: int) -> bytes:
            raise OSError(5, "I/O error", str(sentinel_store))

        monkeypatch.setattr(
            recent_mod,
            "_read_bounded_bytes",
            lambda _fd, _cap: failing_read(_fd, _cap),
        )
        monkeypatch.setenv("CMR_RECENT_STORE_PATH", str(sentinel_store))
        read_res = _safe_request(srv, base_url, "/api/library/recent")
        assert read_res.status == 400
        read_body = json_value(read_res)
        assert read_body["code"] == "recent.store_unreadable"
        assert "SENTINEL-DO-NOT-LEAK-9f3" not in read_body["message"]
        monkeypatch.undo()

        orig_replace = os.replace

        def failing_replace(_src: Any, _dst: Any, **_kwargs: Any) -> None:
            # Accept the POSIX dir_fd kwargs as well as the Windows two-arg form;
            # the intended failure is the injected OSError, not a TypeError from
            # the injector itself.
            raise OSError(13, "Access denied", str(sentinel_store))

        monkeypatch.setattr(os, "replace", failing_replace)
        write_res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
        )
        assert write_res.status == 400
        write_body = json_value(write_res)
        assert write_body["code"] == "recent.store_unwritable"
        assert "SENTINEL-DO-NOT-LEAK-9f3" not in write_body["message"]
        monkeypatch.setattr(os, "replace", orig_replace)

        # Sanity: the stable messages are actionable without the path.
        assert stat_body["message"]
        assert read_body["message"]
        assert write_body["message"]


def test_tamper_row_refusal_parity_on_post(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC6: All tamper rows refused 400 before effects, matching backend authority order."""
    seeded, src, stage, store_file = seeded_env
    sdir = src / "valid_src"
    sdir.mkdir(parents=True, exist_ok=True)

    with running_server(seeded.app) as (srv, base_url):
        # 1. Blank / unsafe studyUid
        for bad_uid in ("", "   ", ".", "..", "../escape", "bad/slash", "bad\\slash", "uid space"):
            payload = {
                "entry": {
                    "studyUid": bad_uid,
                    "seriesUid": None,
                    "scanSourcePath": str(sdir),
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / "x"),
                    "openedSeq": 0,
                    "openCount": 1,
                }
            }
            res = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body(payload),
            )
            assert res.status == 400
            assert json_value(res)["code"] in {"target_authority.uid_invalid", "recent.malformed"}

        # 2. Windows namespace prefix in scanSourcePath
        for ns in (r"\\?\C:\safe", r"\\.\C:\safe"):
            payload = {
                "entry": {
                    "studyUid": STUDY_A,
                    "seriesUid": None,
                    "scanSourcePath": ns,
                    "stageBaseDir": str(stage),
                    "destRoot": str(stage / STUDY_A),
                    "openedSeq": 0,
                    "openCount": 1,
                }
            }
            res = _safe_request(
                srv,
                base_url,
                "/api/library/recent",
                method="POST",
                body=json_body(payload),
            )
            assert res.status == 400
            assert json_value(res)["code"] == "target_authority.namespace_prefix"

        # 3. Genuine-live deny marker on source, stage, and derived dest
        payload_gl_src = {
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": GENUINE_LIVE_DIR,
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        res_gl = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload_gl_src),
        )
        assert res_gl.status == 400
        assert json_value(res_gl)["code"] == "target_authority.genuine_live"

        payload_gl_dest = {
            "entry": {
                "studyUid": "owner_test_session",
                "seriesUid": None,
                "scanSourcePath": str(sdir),
                "stageBaseDir": "/tmp/opencode",
                "destRoot": "/tmp/opencode/owner_test_session",
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        res_gl_dest = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload_gl_dest),
        )
        assert res_gl_dest.status == 400
        assert json_value(res_gl_dest)["code"] in {
            "target_authority.genuine_live",
            "target_authority.root_relative",
        }

        # 4. Dest inside scan source
        payload_inside = {
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": str(stage),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        res_inside = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload_inside),
        )
        assert res_inside.status == 400
        assert json_value(res_inside)["code"] == "library.dest_inside_source"

        # Verify zero side-effects: store file was never created
        assert not store_file.exists()


def test_authorization_matrix_recent_endpoints(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC7: Spoofed, duplicate, missing, or mismatched Host/Origin refused with existing 403."""
    seeded, src, stage, _ = seeded_env
    sdir = src / "study_a"
    sdir.mkdir(parents=True, exist_ok=True)

    with running_server(seeded.app) as (srv, base_url):
        port = int(srv.server_address[1])
        valid_host = f"127.0.0.1:{port}"
        payload = {
            "entry": {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": str(sdir),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_A),
                "openedSeq": 0,
                "openCount": 1,
            }
        }

        # 1. Missing Host header on GET
        res_no_host = _safe_request(srv, base_url, "/api/library/recent", headers={"Host": ""})
        assert res_no_host.status == 403
        assert json_value(res_no_host)["error"] == "forbidden"

        # 2. Spoofed Host header
        res_spoof_host = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            headers={"Host": "evil.com:8080"},
        )
        assert res_spoof_host.status == 403

        # 3. Missing Origin on POST
        res_no_orig = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload),
            headers={"Host": valid_host, "Origin": ""},
        )
        assert res_no_orig.status == 403

        # 4. Mismatched Origin on POST
        res_mismatch_orig = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload),
            headers={"Host": valid_host, "Origin": "http://evil.com"},
        )
        assert res_mismatch_orig.status == 403

        # 5. Non-http Origin (e.g. https or uppercase HTTP)
        res_non_http = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(payload),
            headers={"Host": valid_host, "Origin": f"HTTP://127.0.0.1:{port}"},
        )
        assert res_non_http.status == 403


def test_availability_and_staged_probes(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC8 & AC9: Availability probe covers reuse-first, source-missing, and staged."""
    seeded, src, stage, _ = seeded_env
    study_dir = src / "study_probe"
    dest_dir = stage / STUDY_B
    write_synthetic_cine(study_dir, series_uid=SERIES_B, study_uid=STUDY_B)

    with running_server(seeded.app) as (srv, base_url):
        # 1. Staged missing, source present -> staged-missing (reuse=False)
        params_staged_missing = {
            "source": str(study_dir),
            "stage": str(stage),
            "study": STUDY_B,
        }
        res1 = _safe_request(
            srv,
            base_url,
            f"/api/library/recent/probe?{urlencode(params_staged_missing)}",
        )
        assert res1.status == 200
        d1 = json_value(res1)
        assert d1["availability"] == "staged-missing"
        assert d1["reuse"] is False
        assert d1["source_exists"] is True
        assert d1["staged_exists"] is False

        # Check /api/library/staged before staging
        res_staged_pre = _safe_request(
            srv,
            base_url,
            f"/api/library/staged?dest_path={dest_dir}",
        )
        assert res_staged_pre.status == 200
        assert json_value(res_staged_pre)["staged"] is False

        # Stage the study via existing /api/library/stage
        stage_payload = {
            "source_path": str(study_dir),
            "study_uid": STUDY_B,
            "dest_path": str(dest_dir),
        }
        stage_res = _safe_request(
            srv,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body(stage_payload),
        )
        assert stage_res.status in {200, 400}
        if stage_res.status == 400:
            assert json_value(stage_res)["code"] == "library.dest_not_empty"

        # Check /api/library/staged after staging
        res_staged_post = _safe_request(
            srv,
            base_url,
            f"/api/library/staged?dest_path={dest_dir}",
        )
        assert res_staged_post.status == 200
        staged_info = json_value(res_staged_post)
        assert staged_info["staged"] is True
        assert SERIES_B in staged_info["series_uids"]

        # 2. Both present -> ready (reuse=True)
        res2 = _safe_request(
            srv,
            base_url,
            f"/api/library/recent/probe?{urlencode(params_staged_missing)}",
        )
        assert res2.status == 200
        d2 = json_value(res2)
        assert d2["availability"] == "ready"
        assert d2["reuse"] is True
        assert d2["staged_exists"] is True

        # 3. Source missing, staged present -> source-missing (REUSE-FIRST: reuse=True!)
        # Rename/move source folder
        moved_dir = src / "study_probe_moved"
        study_dir.rename(moved_dir)
        res3 = _safe_request(
            srv,
            base_url,
            f"/api/library/recent/probe?{urlencode(params_staged_missing)}",
        )
        assert res3.status == 200
        d3 = json_value(res3)
        assert d3["availability"] == "source-missing"
        assert d3["reuse"] is True  # Opens staged copy!
        assert d3["source_exists"] is False
        assert d3["staged_exists"] is True


def test_source_immutability_census(seeded_env: tuple[SeededStudy, Path, Path, Path]) -> None:
    """AC10: Pre/post file census and SHA-256 hashes of scan source remain untouched."""
    seeded, src, stage, _ = seeded_env
    source_study = src / "immutable_source"
    write_synthetic_cine(source_study, series_uid=SERIES_C, study_uid=STUDY_C)
    pre_census = _census_tree(source_study)
    assert len(pre_census) > 0

    with running_server(seeded.app) as (srv, base_url):
        # Perform probe, recent write, and stage
        probe_params = {
            "source": str(source_study),
            "stage": str(stage),
            "study": STUDY_C,
        }
        _safe_request(srv, base_url, f"/api/library/recent/probe?{urlencode(probe_params)}")

        recent_payload = {
            "entry": {
                "studyUid": STUDY_C,
                "seriesUid": SERIES_C,
                "scanSourcePath": str(source_study),
                "stageBaseDir": str(stage),
                "destRoot": str(stage / STUDY_C),
                "openedSeq": 0,
                "openCount": 1,
            }
        }
        _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body(recent_payload),
        )

        stage_payload = {
            "source_path": str(source_study),
            "study_uid": STUDY_C,
            "dest_path": str(stage / STUDY_C),
        }
        _safe_request(
            srv,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body(stage_payload),
        )

        # Perform probe again on staged target
        _safe_request(srv, base_url, f"/api/library/recent/probe?{urlencode(probe_params)}")

    post_census = _census_tree(source_study)
    assert pre_census == post_census, "Scan source files were modified!"


# ---------------------------------------------------------------------------
# Review-round P1 fixes
# ---------------------------------------------------------------------------


def _fresh_entry(src: Path, stage: Path, *, study_uid: str = STUDY_A) -> dict[str, Any]:
    """Entry shape exactly as recordRecentCase builds it (first open)."""
    return {
        "studyUid": study_uid,
        "seriesUid": None,
        "scanSourcePath": str(src),
        "stageBaseDir": str(stage),
        "destRoot": "",
        "openedSeq": 0,
        "openCount": 1,
    }


def test_get_post_interleave_never_regresses_high_water(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """P1-1 (F1-serialized shape): a GET paused inside its snapshot must never
    let a POST commit underneath it, the durable sidecar must never roll back,
    and a fresh process must never reuse a sequence.

    The pre-F1 shape paused the loader after it read sequence 1 and let a POST
    assign sequence 2 underneath it. PR3 review F1 serializes the loader's
    whole read — including its high-water persist — under the shared commit
    lock, so that POST cannot commit until the loader releases; this witness
    asserts exactly that (bounded), so the loader's observed sequence can never
    be stale relative to the commit that follows it. The fresh-process leg
    keeps the original no-reuse invariant."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)

    doc1 = process_recent_post(store, {"entry": entry})
    seq1 = doc1["entries"][0]["openedSeq"]
    assert seq1 >= 1

    entered = threading.Event()
    release = threading.Event()
    post_done = threading.Event()
    assert hasattr(recent_mod, "persist_durable_high_water_seq")
    real_persist = recent_mod.persist_durable_high_water_seq

    def paused_persist(path: Path, seq: int) -> None:
        if threading.current_thread().name == "recent-get":
            entered.set()
            assert release.wait(timeout=30), "release signal never arrived"
        real_persist(path, seq)

    get_results: list[Any] = []
    post_results: list[Any] = []

    def run_get() -> None:
        try:
            get_results.append(load_recent_cases(store))
        except BaseException as exc:  # noqa: BLE001 - surfaced in main thread
            get_results.append(exc)

    def run_post() -> None:
        try:
            post_results.append(process_recent_post(store, {"entry": entry}))
        except BaseException as exc:  # noqa: BLE001 - surfaced in main thread
            post_results.append(exc)
        finally:
            post_done.set()

    test_patch = pytest.MonkeyPatch()
    test_patch.setattr(recent_mod, "persist_durable_high_water_seq", paused_persist)
    try:
        getter = threading.Thread(target=run_get, name="recent-get", daemon=True)
        getter.start()
        assert entered.wait(timeout=30), "loader never reached the document read"

        # While the GET is paused (holding the commit lock), a POST attempted
        # now must NOT commit: pre-F1 code commits sequence 2 underneath it.
        poster = threading.Thread(target=run_post, name="recent-post", daemon=True)
        poster.start()
        assert not post_done.wait(timeout=1.0), (
            "POST committed while the GET snapshot still held the commit lock"
        )

        release.set()
        getter.join(timeout=30)
        poster.join(timeout=30)
        assert not getter.is_alive() and not poster.is_alive()
        assert len(get_results) == 1, get_results
        assert isinstance(get_results[0], dict), get_results[0]
        assert len(post_results) == 1, post_results
        assert isinstance(post_results[0], dict), post_results[0]
    finally:
        release.set()
        test_patch.undo()

    # The loader served its pre-commit snapshot; the POST then assigned the
    # next sequence — never the other way round.
    assert get_results[0]["entries"][0]["openedSeq"] == seq1, get_results[0]
    seq2 = post_results[0]["entries"][0]["openedSeq"]
    assert seq2 == seq1 + 1, post_results[0]

    # Serialized order: the durable sidecar holds the committed maximum.
    sidecar = get_sidecar_seq_path(store)
    assert sidecar.is_file()
    assert int(sidecar.read_text(encoding="utf-8")) >= seq2

    # Fresh-process witness: drop process-local state and zero the document
    # (supported absent state) — the next POST must EXCEED the sidecar value,
    # never reuse a sequence.
    recent_mod._HIGH_WATER_BY_PATH.pop(str(store.resolve()), None)
    store.write_bytes(b"")
    doc3 = process_recent_post(store, {"entry": entry})
    assert doc3["entries"][0]["openedSeq"] > seq2


def test_high_water_persist_writes_merged_maximum_not_stale_arg(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """P1-1 unit seam: persisting a stale caller argument must never lower
    the durable sidecar (writes max(memory, sidecar, argument))."""
    _, _, store = recent_env
    persist_durable_high_water_seq(store, 1)
    persist_durable_high_water_seq(store, 5)
    persist_durable_high_water_seq(store, 2)  # stale writer
    assert int(get_sidecar_seq_path(store).read_text(encoding="utf-8")) == 5
    assert read_durable_high_water_seq(store) == 5


def test_sidecar_bounded_read_never_ingests_oversized_content(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """P1-2: an oversized sidecar is bounded out — its content is never read
    into the high-water value (treated as unusable; the document's own
    high_water_seq still recovers the true mark on load)."""
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("9" * 100, encoding="utf-8")
    assert read_durable_high_water_seq(store) == 0

    # Non-regular sidecar refused fail-closed — a DIRECTORY needs no symlink
    # privilege, so this refusal witness always runs (unlike the link test).
    sidecar.unlink()
    sidecar.mkdir()
    with pytest.raises(RecentError) as exc:
        read_durable_high_water_seq(store)
    assert exc.value.code == "recent.store_unreadable"
    sidecar.rmdir()


def test_sidecar_symlink_refused_and_never_followed(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """P1-2: a symlinked sidecar is refused fail-closed with the LINK refusal
    code the production path actually raises (``target_authority.symlink`` —
    round-2 P2 corrected this expectation, which asserted
    ``recent.store_unreadable`` and would have failed on any symlink-capable
    runner); its target bytes are never read; persisting replaces the LINK
    entry itself (rename) without ever writing through it."""
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    victim = store.parent / "victim.txt"
    victim.write_text("12345", encoding="utf-8")
    try:
        sidecar.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable without developer mode")

    with pytest.raises(RecentError) as exc:
        read_durable_high_water_seq(store)
    assert exc.value.code == "target_authority.symlink"
    assert victim.read_text(encoding="utf-8") == "12345"

    # Hostile sidecar path: persist refuses (fail-closed, never written
    # through the link) instead of self-healing over it. The merged mark
    # survives in memory for this process (review P1 follow-up).
    with pytest.raises(RecentError) as exc2:
        persist_durable_high_water_seq(store, 7)
    assert exc2.value.code == "target_authority.symlink"
    assert sidecar.is_symlink()
    assert victim.read_text(encoding="utf-8") == "12345"


def test_sidecar_link_refusal_code_is_witnessed_without_privilege(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 P2: the link-refusal expectation must be a real witness on ANY
    runner, not one that disappears behind a symlink-privilege skip.

    Exercises the production refusal seam directly (``reject_links`` raising)
    so the security regression — a link being followed instead of refused —
    fails deterministically without needing filesystem privileges.
    """
    from services.workstation.target_authority import Refusal

    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    def refuse_links(_path: str, _field: str) -> None:
        raise Refusal(
            "target_authority.symlink",
            "recent_sidecar_path contains a symlink or junction component; refusing fail-closed",
        )

    monkeypatch.setattr(recent_mod, "reject_links", refuse_links)
    with pytest.raises(RecentError) as exc:
        read_durable_high_water_seq(store)
    assert exc.value.code == "target_authority.symlink"


def test_sidecar_identity_comparison_distinguishes_objects(tmp_path: Path) -> None:
    """R3-2 unit: identity (dev/ino), not size or mtime, decides whether two
    observations name the same filesystem object.

    Round-4 review called the original test tautological (it only compared a
    file with itself and two stats of the SAME file). This witness pins both
    directions: a DISTINCT object with identical content, size AND mtime
    compares False, while the same object under a DIFFERENT name (hard link)
    compares True — so the assertion actually discriminates on identity.
    """
    import os as os_mod

    a = os_mod.stat(__file__)
    b = os_mod.stat(__file__)
    assert recent_mod._same_file(a, b) is True
    assert recent_mod._same_file(a, a) is True

    # Distinct object, deliberately made metadata-identical to the target.
    twin = tmp_path / "twin.bin"
    twin.write_bytes(Path(__file__).read_bytes())
    os_mod.utime(twin, ns=(a.st_atime_ns, a.st_mtime_ns))
    c = os_mod.stat(twin)
    assert c.st_size == a.st_size
    assert c.st_mtime_ns == a.st_mtime_ns
    assert recent_mod._same_file(a, c) is False, (
        "a distinct object with identical size/mtime must NOT compare equal"
    )

    # Same object reached under another name: identity follows the object,
    # not the path spelling.
    alias = tmp_path / "alias.bin"
    os_mod.link(twin, alias)
    d = os_mod.stat(alias)
    assert recent_mod._same_file(c, d) is True


def test_sidecar_replaced_between_validation_and_open_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-2 (portable production-path witness): a swap installed inside the
    open window — after link refusal and the pre-open ``lstat``, before the
    handle opens — must be refused, because the handle then names an object
    different from the one that was validated. The pre-fix shape kept only a
    post-open comparison, which agrees with whatever the swap produced.

    Residual same-inode-reuse boundary (explicit, not claimed covered): if the
    original inode is freed and the filesystem reuses the same dev/ino for the
    replacement at the same path, identity-only ``_same_file`` cannot prove a
    substitution. On POSIX this witness therefore holds the original handle
    open and renames the original outside the validated path before creating
    the replacement, forcing a genuinely different identity. The same-inode
    reuse case remains a documented limitation and is not asserted as refused.
    """
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    real_open = recent_mod._open_sidecar
    held_fd: int | None = None
    moved: Path | None = None

    if os.name == "nt":
        # The existing Windows witness already observes a different file ID
        # after unlink/recreate; keep its native shape.
        def swap_then_open(path: Path, flags: int) -> int:
            path.unlink()
            path.write_text("12345", encoding="utf-8")
            return real_open(path, flags)
    else:
        # POSIX deterministic different-identity shape: retain the original
        # inode via an open handle AND move it outside the validated path so
        # the filesystem cannot reuse its inode for the replacement.
        held_fd = os.open(sidecar, os.O_RDONLY)
        moved = sidecar.with_name(sidecar.name + ".original")

        def swap_then_open(path: Path, flags: int) -> int:
            os.rename(path, moved)
            path.write_text("12345", encoding="utf-8")
            return real_open(path, flags)

    try:
        monkeypatch.setattr(recent_mod, "_open_sidecar", swap_then_open)
        with pytest.raises(RecentError) as exc:
            read_durable_high_water_seq(store)
        assert exc.value.code == "target_authority.symlink"
    finally:
        if held_fd is not None:
            os.close(held_fd)
        if moved is not None:
            moved.unlink(missing_ok=True)


def test_sidecar_parent_swap_in_open_window_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-2 (reviewer's ancestor-redirection repro, made portable): the PARENT
    directory is moved aside and a different directory holding a same-named
    sidecar with different content takes its place, inside the open window.
    The redirected handle resolves to another object than the validated one,
    so the read is refused rather than ingesting the decoy's 12345."""
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    parent = sidecar.parent
    parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    real_open = recent_mod._open_sidecar
    moved = parent.with_name(parent.name + "-orig")

    def swap_parent_then_open(path: Path, flags: int) -> int:
        os.rename(parent, moved)
        parent.mkdir()
        (parent / sidecar.name).write_text("12345", encoding="utf-8")
        return real_open(path, flags)

    monkeypatch.setattr(recent_mod, "_open_sidecar", swap_parent_then_open)
    with pytest.raises(RecentError) as exc:
        read_durable_high_water_seq(store)
    assert exc.value.code == "target_authority.symlink"


@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions are Windows-only")
def test_sidecar_junction_parent_swap_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-2 (exact attack shape from the round-3 review): the parent directory
    is replaced by a REAL NTFS junction pointing at a decoy directory whose
    same-named sidecar contains 12345, installed between link validation and
    open. ``mklink /J`` needs no privilege, so this witness runs on an ordinary
    Windows host (unlike the symlink test, which needs developer mode)."""
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    parent = sidecar.parent
    parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    real_open = recent_mod._open_sidecar
    moved = parent.with_name(parent.name + "-orig")
    decoy = parent.with_name(parent.name + "-decoy")

    def junction_then_open(path: Path, flags: int) -> int:
        os.rename(parent, moved)
        decoy.mkdir()
        (decoy / sidecar.name).write_text("12345", encoding="utf-8")
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(parent), str(decoy.resolve())],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mklink /J failed: {proc.stderr.strip()}")
        return real_open(path, flags)

    monkeypatch.setattr(recent_mod, "_open_sidecar", junction_then_open)
    try:
        with pytest.raises(RecentError) as exc:
            read_durable_high_water_seq(store)
        assert exc.value.code == "target_authority.symlink"
    finally:
        try:
            parent.rmdir()  # removes the junction itself, not its target
        except OSError:
            pass


def test_sidecar_pre_lstat_ancestor_redirect_window_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-2 round-4 (exact review repro): the redirect is installed INSIDE
    the first ``lstat`` — after ``reject_links`` already returned and BEFORE
    the pre-open identity is captured — then removed again before the handle
    is returned, with the SAME decoy object hard-linked at the validated
    name. Every path-based comparison then agrees (pre == opened == final
    post-hoc ``lstat``, no link left anywhere to find), so the decoy's 12345
    was ingested as trusted high-water state at the reviewed head. Only a
    location binding derived from the open handle itself — never another
    path re-walk, which merely moves the window — can see that the open
    actually traversed the decoy and refuse.

    ``mklink /J`` is Windows-only; POSIX witnesses the identical shape with
    an ordinary directory symlink (no privilege needed), so the regression is
    witnessed on both the Windows workstation and the Linux CI runner.
    """
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    parent = sidecar.parent
    parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    moved = parent.with_name(parent.name + "-orig")
    decoy = parent.with_name(parent.name + "-decoy")
    decoy.mkdir()
    decoy_sidecar = decoy / sidecar.name
    decoy_sidecar.write_text("12345", encoding="utf-8")

    real_lstat = Path.lstat
    real_open = recent_mod._open_sidecar
    lstat_calls = {"n": 0}
    observed_ino = {"v": 0}

    def install_redirect() -> None:
        os.rename(parent, moved)
        if os.name == "nt":
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(parent), str(decoy.resolve())],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"mklink /J failed: {proc.stderr.strip()}")
        else:
            parent.symlink_to(decoy, target_is_directory=True)

    def remove_redirect() -> None:
        """Remove ONLY the link itself; never touches a real directory."""
        if os.name == "nt":
            if os.path.isjunction(str(parent)):
                parent.rmdir()
        else:
            if parent.is_symlink():
                parent.unlink()

    def lstat_then_redirect(
        self: Path, *args: Any, **kwargs: Any
    ) -> os.stat_result:
        if self == sidecar:
            lstat_calls["n"] += 1
            if lstat_calls["n"] == 1:
                # reject_links has already returned; the pre-open observation
                # is the window under review — install the redirect NOW.
                install_redirect()
            st = real_lstat(self, *args, **kwargs)
            if lstat_calls["n"] == 1:
                observed_ino["v"] = st.st_ino
            return st
        return real_lstat(self, *args, **kwargs)

    def open_then_unredirect(path: Path, flags: int) -> int:
        fd = real_open(path, flags)
        try:
            # Remove the redirect, put a REAL parent back, and expose the
            # same decoy object at the validated name: after this point every
            # post-hoc path check is link-free and identity-consistent.
            remove_redirect()
            parent.mkdir()
            os.link(decoy_sidecar, sidecar)
            return fd
        except BaseException:
            os.close(fd)
            raise

    monkeypatch.setattr(Path, "lstat", lstat_then_redirect)
    monkeypatch.setattr(recent_mod, "_open_sidecar", open_then_unredirect)
    try:
        with pytest.raises(RecentError) as exc:
            read_durable_high_water_seq(store)
        assert exc.value.code == "target_authority.symlink"
        # The window really happened: the pre-open observation recorded the
        # DECOY object (not the original sidecar) — the fixture is not
        # accidentally testing an empty or already-swapped setup.
        assert lstat_calls["n"] >= 1, "the pre-open observation never ran"
        assert observed_ino["v"] == decoy_sidecar.lstat().st_ino
        # Post-hoc state is link-free and holds the decoy bytes at the
        # validated path: a path-only recheck would have accepted them.
        assert not os.path.islink(str(parent))
        if os.name == "nt":
            assert not os.path.isjunction(str(parent))
        assert sidecar.read_text(encoding="utf-8") == "12345"
    finally:
        remove_redirect()


def test_sidecar_post_open_link_revalidation_runs_and_refuses(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-2: the ancestors are re-validated AFTER the handle is anchored — a
    post-open ``lstat`` alone would follow a swapped-in junction and agree
    with the handle. This witness proves the second refusal is executed on
    the production path and that its refusal propagates as a typed error."""
    from services.workstation.target_authority import Refusal

    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    real_reject = recent_mod.reject_links
    calls = {"n": 0}

    def counting_reject(text: str, field: str) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise Refusal(
                "target_authority.symlink",
                "ancestor became a link during validation; refusing fail-closed",
            )
        return real_reject(text, field)

    monkeypatch.setattr(recent_mod, "reject_links", counting_reject)
    with pytest.raises(RecentError) as exc:
        read_durable_high_water_seq(store)
    assert exc.value.code == "target_authority.symlink"
    assert calls["n"] >= 2, "the post-open ancestor re-validation never ran"


def test_sidecar_read_oserror_falls_back_to_unusable(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-3: a read failure on the open handle keeps the documented
    unusable-sidecar fallback (0) instead of raising a bare OSError that the
    recent handlers do not catch and that would escape as a generic 500."""
    import errno as errno_mod

    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    def failing_read(_fd: int) -> bytes:
        raise OSError(errno_mod.EIO, "injected sidecar read failure")

    monkeypatch.setattr(recent_mod, "_read_sidecar_bytes", failing_read)
    assert read_durable_high_water_seq(store) == 0


def test_sidecar_fstat_oserror_falls_back_to_unusable(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-3: handle validation failures behave like read failures — unusable
    (0), never a raw OSError. The module's ``os`` reference is swapped for a
    delegating proxy so only this module's ``fstat`` is affected."""
    import errno as errno_mod

    class _FstatFailingOS:
        def __getattr__(self, name: str):
            return getattr(os, name)

        @staticmethod
        def fstat(_fd: int) -> os.stat_result:
            raise OSError(errno_mod.EIO, "injected fstat failure")

    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    monkeypatch.setattr(recent_mod, "os", _FstatFailingOS())
    assert read_durable_high_water_seq(store) == 0


def test_recent_get_envelope_survives_sidecar_read_failure(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-3 at the HTTP boundary: a sidecar read failure must not escape the
    frozen recent envelope as a generic 500 — the GET route still answers
    with its normal recent payload (unusable sidecar simply recovers 0).

    Round-4 review: the original witness seeded a store with NO sidecar, so
    the patched reader was never called and the assertion passed vacuously.
    This version creates a REAL regular sidecar and proves the injected
    reader was actually invoked while the route still returned its normal
    200 envelope."""
    import errno as errno_mod

    seeded, _src, _stage, store = seeded_env
    seq_path = get_sidecar_seq_path(store)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path.write_text("7", encoding="utf-8")

    calls: list[int] = []

    def failing_read(fd: int) -> bytes:
        calls.append(fd)
        raise OSError(errno_mod.EIO, "injected sidecar read failure")

    monkeypatch.setattr(recent_mod, "_read_sidecar_bytes", failing_read)
    with running_server(seeded.app) as (srv, base_url):
        res = _safe_request(srv, base_url, "/api/library/recent")
        assert res.status == 200
        body = json_value(res)
        assert "entries" in body
        assert body["version"] == 1
        assert body["reason"] == "absent"  # sidecar recovered 0, store absent
        assert calls, (
            "the injected sidecar reader was never invoked — "
            "this witness would pass vacuously"
        )


def test_sidecar_growth_after_validation_is_never_ingested(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-2 TOCTOU witness (round-2 P1): growing the sidecar between the
    ``fstat`` size check and the actual read must NOT be ingested.

    The read is bounded at ``_MAX_SIDECAR_BYTES + 1`` bytes from the already
    open handle, so a file that becomes 100 digits after validation yields an
    unusable result (0), never the 100-digit integer. With the pre-fix
    implementation (``lstat`` size check then unbounded ``read_text``) this
    ingests the whole value; the patched seam also fails-capable there.
    """
    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("1", encoding="utf-8")

    real_read = recent_mod._read_sidecar_bytes

    def grow_then_read(fd: int) -> bytes:
        # The file is still valid at fstat (1 byte); it grows only now.
        sidecar.write_text("9" * 100, encoding="utf-8")
        return real_read(fd)

    monkeypatch.setattr(recent_mod, "_read_sidecar_bytes", grow_then_read)
    assert read_durable_high_water_seq(store) == 0


def test_sidecar_byte_read_is_capped_one_past_the_bound(
    recent_env: tuple[Path, Path, Path],
) -> None:
    """P1-2: the cap is a property of the READ itself — from an already open
    handle it yields at most ``_MAX_SIDECAR_BYTES + 1`` bytes, so excess can
    be detected without ever ingesting it. Behavioural witness independent of
    the growth test's seam (the pre-fix shape has no bounded reader at all)."""
    import os as os_mod

    _, _, store = recent_env
    sidecar = get_sidecar_seq_path(store)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("9" * 100, encoding="utf-8")
    fd = os_mod.open(sidecar, os_mod.O_RDONLY)
    try:
        data = recent_mod._read_sidecar_bytes(fd)
    finally:
        os_mod.close(fd)
    assert len(data) <= recent_mod._MAX_SIDECAR_BYTES + 1
    assert len(data) < 100  # the 100-digit body is never returned in full


def test_sidecar_unicode_digit_get_falls_back_to_document_mark(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """PR3 review r4310770081 (fail-capable, GET leg): a sidecar whose UTF-8
    content is ``str.isdigit()``-true but not convertible (U+00B2 SUPERSCRIPT
    TWO) must take the DOCUMENTED unusable-sidecar fallback.

    The sidecar is read BEFORE the main document, so the ``ValueError`` from
    ``int(content)`` made a valid last-good store unavailable: GET answered the
    generic ``400 invalid_request`` without ``no-store`` instead of falling
    back to the document's own ``high_water_seq`` (the 2-byte content is far
    below the 32-byte sidecar cap, so no size guard sees it first). The store
    written through the live route must remain byte-identical and readable;
    the read path may leave the hostile sidecar alone or re-persist the
    recovered mark as ASCII decimal, never a partial value. Pre-fix: RED -
    ``400 {"code":"invalid_request"}`` with no ``no-store``.
    """
    seeded, src, stage, store_file = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")  # warm the listener
        warm = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
        )
        assert warm.status == 200, (warm.status, json_value(warm))
        good_store = store_file.read_bytes()

        seq_path = get_sidecar_seq_path(store_file)
        assert seq_path.exists()
        hostile = "\u00b2".encode("utf-8")  # '²': isdigit() is True, int() raises
        seq_path.write_bytes(hostile)
        assert hostile.decode("utf-8").isdigit()
        with pytest.raises(ValueError):
            int(hostile.decode("utf-8"))

        res = _safe_request(srv, base_url, "/api/library/recent")
        assert res.status == 200, (res.status, json_value(res))
        body = json_value(res)
        assert body["entries"], body
        assert body["reason"] not in {"absent", "corrupt-rows"}, body
        assert res.headers.get("Cache-Control") == "no-store", dict(res.headers)
        assert store_file.read_bytes() == good_store, "main store bytes changed"
        after = seq_path.read_bytes()
        if after != hostile:
            assert after.isascii() and after.strip().isdigit(), after


def test_sidecar_unicode_digit_post_still_commits(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """PR3 review r4310770081 (fail-capable, POST leg): the same hostile
    sidecar content must not block a valid commit.

    Pre-fix the POST path raised ``ValueError`` out of ``_read_sidecar_seq``
    and answered the generic ``400 invalid_request`` without ``no-store``, so a
    valid last-good store was unusable for writes. Post-fix the mark recovers
    from the document's own ``high_water_seq`` and the commit advances
    monotonically. Pre-fix: RED - ``400 {"code":"invalid_request"}``.
    """
    seeded, src, stage, store_file = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")
        warm = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
        )
        assert warm.status == 200, (warm.status, json_value(warm))
        prior_seq = json_value(warm)["entries"][0]["openedSeq"]

        seq_path = get_sidecar_seq_path(store_file)
        seq_path.write_bytes("\u00b2".encode("utf-8"))

        res = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
        )
        assert res.status == 200, (res.status, json_value(res))
        assert res.headers.get("Cache-Control") == "no-store", dict(res.headers)
        merged = json_value(res)["entries"][0]
        assert merged["studyUid"] == STUDY_A, merged
        assert merged["openedSeq"] > prior_seq, (merged, prior_seq)
        assert merged["openCount"] >= 2, merged


def test_temp_path_exclusive_refusal_preserves_last_good(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-2: a pre-planted file at the writer's temp path is refused via
    O_EXCL — the main store is neither truncated nor followed, last-good
    bytes survive, and the caller gets an actionable store_unwritable."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)

    doc1 = process_recent_post(store, {"entry": entry})
    assert doc1["entries"][0]["openedSeq"] >= 1
    last_good = store.read_bytes()

    # Make the temp name deterministic, then pre-plant it.
    fixed = type("Fixed", (), {"hex": "deadbeef" * 4})()
    monkeypatch.setattr(recent_mod, "uuid4", lambda: fixed)
    planted = store.parent / f".tmp_{store.name}_{os.getpid()}_{fixed.hex}"
    planted.write_bytes(b"PLANTED")

    with pytest.raises(RecentError) as exc:
        process_recent_post(store, {"entry": entry})
    assert exc.value.code == "recent.store_unwritable"
    assert store.read_bytes() == last_good


def test_staged_route_physical_path_authority(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """P1-3: /api/library/staged validates dest_path with the existing
    authority policy BEFORE any filesystem inspection. Unsafe paths get a
    stable frozen 400 refusal — never a suppressed 200 {staged: false}."""
    seeded, _, stage, _ = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")

        # 1. Windows namespace prefix refused.
        res_ns = _safe_request(
            srv, base_url, "/api/library/staged?dest_path=" + "\\\\?\\C:\\x"
        )
        assert res_ns.status == 400
        assert json_value(res_ns)["code"] == "target_authority.namespace_prefix"

        # 2. Frozen genuine-live marker refused on the RAW text.
        res_gl = _safe_request(
            srv, base_url, f"/api/library/staged?dest_path={GENUINE_LIVE_DIR}"
        )
        assert res_gl.status == 400
        assert json_value(res_gl)["code"] == "target_authority.genuine_live"

        # 3. Relative path refused (fully-anchored absolute required).
        res_rel = _safe_request(
            srv, base_url, "/api/library/staged?dest_path=relative/not_absolute"
        )
        assert res_rel.status == 400
        assert json_value(res_rel)["code"] == "target_authority.root_relative"

        # 4. Existing NON-staged directory: frozen layout refusal, not 200.
        plain = stage / "plain_dir"
        plain.mkdir()
        (plain / "stray.txt").write_text("x", encoding="utf-8")
        res_layout = _safe_request(
            srv, base_url, f"/api/library/staged?dest_path={plain}"
        )
        assert res_layout.status == 400
        assert json_value(res_layout)["code"] == "target_authority.dest_layout"

        # 5. Absent dest remains an honest staged:false answer (validated first).
        res_absent = _safe_request(
            srv, base_url, f"/api/library/staged?dest_path={stage / STUDY_C}"
        )
        assert res_absent.status == 200
        assert json_value(res_absent)["staged"] is False

        # 6. Ancestor symlink refused (skip where links need privileges).
        link = stage / "linkdir"
        try:
            link.symlink_to(stage, target_is_directory=True)
        except (OSError, NotImplementedError):
            link = None
        if link is not None:
            res_link = _safe_request(
                srv, base_url, f"/api/library/staged?dest_path={link / 'sub'}"
            )
            assert res_link.status == 400
            assert json_value(res_link)["code"] == "target_authority.symlink"


def test_bound_root_route_reports_server_bound_root(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """R7 F5 server leg: GET /api/library/bound-root reports the root the
    server was bootstrapped against (the genuinely bound live root for its
    page), behind the same authorize_host + loopback policy as the recent
    family, with no-store and no path leakage beyond the bound root itself."""
    seeded, _, _, _ = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")  # warmup
        res = _safe_request(srv, base_url, "/api/library/bound-root")
        assert res.status == 200
        assert json_value(res) == {"root": str(seeded.root)}
        assert res.headers.get("Cache-Control") == "no-store"

        # Spoofed Host is refused by the existing allow-list boundary.
        res_spoof = _safe_request(
            srv,
            base_url,
            "/api/library/bound-root",
            headers={"Host": "evil.example:9999"},
        )
        assert res_spoof.status == 403
        assert json_value(res_spoof)["error"] == "forbidden"

        # Wildcard listener: loopback request authority still refused.
        saved_host = seeded.app.host
        try:
            seeded.app.host = "0.0.0.0"
            res_wild = _safe_request(srv, base_url, "/api/library/bound-root")
            assert res_wild.status == 403
            assert "loopback" in json_value(res_wild)["message"]
        finally:
            seeded.app.host = saved_host

        # R12 P2 review witness: bootstrap spelling containing dot-segments (..)
        # must be reported as the physically resolved, validated bound root.
        saved_root = seeded.app.root
        try:
            existing_sibling = seeded.root.parent / "existing_sibling"
            existing_sibling.mkdir(parents=True, exist_ok=True)
            seeded.app.root = seeded.root.parent / "existing_sibling" / ".." / seeded.root.name
            res_dots = _safe_request(srv, base_url, "/api/library/bound-root")
            assert res_dots.status == 200
            assert json_value(res_dots) == {"root": str(seeded.root.resolve())}
        finally:
            seeded.app.root = saved_root

        # R12 P2 review witness: if app.root is a link/symlink, /api/library/bound-root
        # refuses fail-closed with target_authority.symlink.
        link_dir = seeded.root.parent / "bound_link"
        try:
            link_dir.symlink_to(seeded.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            link_dir = None
        if link_dir is not None:
            saved_root = seeded.app.root
            try:
                seeded.app.root = link_dir
                res_link = _safe_request(srv, base_url, "/api/library/bound-root")
                assert res_link.status == 400
                assert json_value(res_link)["code"] == "target_authority.symlink"
            finally:
                seeded.app.root = saved_root


def test_recent_routes_refuse_non_loopback_deployment(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """AC/auth acceptance: non-loopback deployment refuses recent routes —
    on BOTH seams (configured listener and authorized request authority),
    layered on top of the existing Host/Origin allow-list."""
    seeded, src, stage, _ = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        port = int(srv.server_address[1])
        nonloop = f"nonloopback.example:{port}"
        seeded.app.allowed_authorities = tuple(seeded.app.allowed_authorities) + (
            nonloop,
        )

        # 1. Allow-listed non-loopback GET authority refused.
        res_get = _safe_request(
            srv, base_url, "/api/library/recent", headers={"Host": nonloop}
        )
        assert res_get.status == 403
        assert json_value(res_get)["error"] == "forbidden"
        assert "loopback" in json_value(res_get)["message"]

        # 2. Probe + staged + bound-root sibling routes refused too.
        for path in (
            "/api/library/recent/probe",
            "/api/library/staged",
            "/api/library/bound-root",
        ):
            res = _safe_request(srv, base_url, path, headers={"Host": nonloop})
            assert res.status == 403, path
            assert json_value(res)["error"] == "forbidden"

        # 3. POST with matching non-loopback Origin refused (body drained).
        res_post = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
            headers={"Host": nonloop, "Origin": f"http://{nonloop}"},
        )
        assert res_post.status == 403
        assert json_value(res_post)["error"] == "forbidden"

        # 4. Listener seam: wildcard listener refuses even a loopback Host.
        saved_host = seeded.app.host
        try:
            seeded.app.host = "0.0.0.0"
            res_wild = _safe_request(srv, base_url, "/api/library/recent")
            assert res_wild.status == 403
            assert "loopback" in json_value(res_wild)["message"]
        finally:
            seeded.app.host = saved_host

        # 5. Normal loopback deployment unaffected.
        res_ok = _safe_request(srv, base_url, "/api/library/recent")
        assert res_ok.status == 200


def _response_complete(data: bytes) -> bool:
    """True once ``data`` holds a complete HTTP response (headers + body)."""
    head_end = data.find(b"\r\n\r\n")
    if head_end == -1:
        return False
    length = 0
    for line in data[:head_end].decode("latin-1").split("\r\n")[1:]:
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length":
            try:
                length = int(value.strip())
            except ValueError:
                length = 0
    return len(data) >= head_end + 4 + length


def _raw_exchange_once(
    port: int,
    payload: bytes,
    deadline_seconds: float,
) -> tuple[bytes, bool, float]:
    """One raw exchange; returns ``(response_bytes, ended, elapsed_seconds)``.

    The socket deadline bounds the whole exchange, so a server that waits for
    a body the client never sends can never hang this witness.
    """
    import socket

    started = time.monotonic()
    received = bytearray()
    ended = False
    with socket.create_connection(("127.0.0.1", port), timeout=deadline_seconds) as sock:
        sock.sendall(payload)
        deadline = started + deadline_seconds
        while True:
            budget = deadline - time.monotonic()
            if budget <= 0:
                break
            sock.settimeout(budget)
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                break
            except ConnectionResetError:
                ended = True
                break
            if not chunk:
                ended = True
                break
            received.extend(chunk)
    return bytes(received), ended, time.monotonic() - started


def _refusal_response(
    port: int,
    payload: bytes,
    *,
    attempts: int = 4,
    deadline_seconds: float = 5.0,
) -> tuple[bytes, bool, float]:
    """Prompt-refusal probe tolerant of the documented Windows loopback reset.

    Only a transport loss (reset/timeout) that carried no complete response is
    retried: both legs of this witness are refusal-only (the request is
    rejected before any state change), so replaying identical bytes cannot
    double-apply anything. The pre-fix defect -- a handler parked on the
    withheld body -- yields no response on ANY attempt, so the retry cannot
    mask it.
    """
    for attempt in range(attempts):
        try:
            response, ended, elapsed = _raw_exchange_once(
                port, payload, deadline_seconds
            )
        except OSError:
            response, ended, elapsed = b"", False, 0.0
        if _response_complete(response):
            return response, ended, elapsed
        time.sleep(0.05 * (attempt + 1))
    raise AssertionError(
        f"no complete refusal response within {attempts} attempts"
    )


def test_post_refusal_answers_before_body_and_closes_connection(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """Review P2 (discussion r4310080005): a rejected POST must never wait
    for the body it advertises.

    The previous helper read the advertised ``Content-Length`` bytes before
    answering, so a client could send rejected Host/Origin headers with a
    withheld body and keep the handler thread and its socket parked. Every
    rejected POST path -- the mutation allow-list (recent, stage, and action
    routes alike) and the recent non-loopback refusal -- must answer
    promptly, and the refused connection must close instead of waiting for
    bytes the client may never send.
    """
    seeded, _src, _stage, _store = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")  # warm the loopback listener
        port = int(srv.server_address[1])
        host = f"127.0.0.1:{port}"

        refused_requests = {
            "recent (rejected Host)": (
                f"POST /api/library/recent HTTP/1.1\r\nHost: spoof.example:9\r\n"
                f"Origin: http://{host}\r\nContent-Type: application/json\r\n"
                "Content-Length: 128\r\n\r\n"
            ),
            "stage (rejected Origin)": (
                f"POST /api/library/stage HTTP/1.1\r\nHost: {host}\r\n"
                "Origin: http://evil.example:9\r\nContent-Type: application/json\r\n"
                "Content-Length: 128\r\n\r\n"
            ),
            "action (rejected Host)": (
                f"POST /api/study/{SERIES_A}/action HTTP/1.1\r\nHost: spoof.example:9\r\n"
                f"Origin: http://{host}\r\nContent-Type: application/json\r\n"
                "Content-Length: 128\r\n\r\n"
            ),
        }
        for label, raw_request in refused_requests.items():
            raw, ended, elapsed = _refusal_response(
                port, raw_request.encode("ascii")
            )
            assert raw.startswith(b"HTTP/1.1 403 "), (label, raw[:200])
            assert b'"error":"forbidden"' in raw, label
            assert ended, f"{label}: refused connection stayed open"
            assert elapsed < 2.0, (label, elapsed)

        # Allow-listed non-loopback request authority on the recent route:
        # the loopback refusal must answer before the advertised body too.
        nonloop = f"nonloopback.example:{port}"
        saved_authorities = tuple(seeded.app.allowed_authorities)
        seeded.app.allowed_authorities = saved_authorities + (nonloop,)
        try:
            raw_request = (
                f"POST /api/library/recent HTTP/1.1\r\nHost: {nonloop}\r\n"
                f"Origin: http://{nonloop}\r\nContent-Type: application/json\r\n"
                "Content-Length: 128\r\n\r\n"
            )
            raw, ended, elapsed = _refusal_response(
                port, raw_request.encode("ascii")
            )
            assert raw.startswith(b"HTTP/1.1 403 "), raw[:200]
            assert b"loopback" in raw
            assert ended, "non-loopback refusal: connection stayed open"
            assert elapsed < 2.0, elapsed
        finally:
            seeded.app.allowed_authorities = saved_authorities


def _trickle_exchange_once(
    port: int,
    headers: bytes,
    *,
    byte_interval: float,
    deadline_seconds: float,
) -> tuple[bytes, bool, float]:
    """One raw exchange whose client TRICKLES the body it advertised.

    The header block advertises a large ``Content-Length`` the client never
    sends at once; afterwards the client sends one byte per ``byte_interval``
    while polling for the response. That is exactly the shape a buffered
    ``read(n)`` cannot bound: every underlying receive returns well inside
    the socket timeout, so only an absolute elapsed-time bound re-checked
    between reads can end the drain. The client's own deadline always bounds
    the exchange, so a server that never closes cannot hang this witness.
    """
    import socket

    started = time.monotonic()
    deadline = started + deadline_seconds
    received = bytearray()
    ended = False
    with socket.create_connection(("127.0.0.1", port), timeout=deadline_seconds) as sock:
        sock.sendall(headers)
        next_send = started
        while True:
            now = time.monotonic()
            budget = deadline - now
            if budget <= 0:
                break
            if now >= next_send:
                try:
                    sock.sendall(b"x")
                except OSError:
                    ended = True
                    break
                next_send = now + byte_interval
            sock.settimeout(min(0.05, budget))
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            except (ConnectionResetError, ConnectionAbortedError):
                ended = True
                break
            if not chunk:
                ended = True
                break
            received.extend(chunk)
    return bytes(received), ended, time.monotonic() - started


def test_trickling_refused_body_closes_within_total_budget(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """Review 5567119593 P2 (comment r4310462032): a rejected POST whose
    client advertises 65536 body bytes and then TRICKLES them one byte at a
    time must still see the refused connection close within the budget.

    ``self.rfile`` is a buffered reader, so a single ``read(65536)`` performs
    many underlying receives and the socket timeout bounds each receive, never
    the call: a client sending one byte per interval shorter than that timeout
    keeps the handler and its socket parked for hours, and the outer
    ``time.monotonic()`` deadline check never runs. The withheld-body witness
    does not cover this shape. Every read must be a bounded single receive
    (``read1``) with the socket timeout capped to the remaining budget, and
    the refused connection must close inside the total budget.
    """
    seeded, _src, _stage, _store = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")  # warm the loopback listener
        port = int(srv.server_address[1])
        host = f"127.0.0.1:{port}"
        advertised = "65536"
        legs: dict[str, bytes] = {
            "recent (rejected Host)": (
                f"POST /api/library/recent HTTP/1.1\r\nHost: spoof.example:9\r\n"
                f"Origin: http://{host}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {advertised}\r\n\r\n"
            ).encode("ascii"),
        }
        nonloop = f"nonloopback.example:{port}"
        saved_authorities = tuple(seeded.app.allowed_authorities)
        seeded.app.allowed_authorities = saved_authorities + (nonloop,)
        try:
            legs["recent (non-loopback authority)"] = (
                f"POST /api/library/recent HTTP/1.1\r\nHost: {nonloop}\r\n"
                f"Origin: http://{nonloop}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {advertised}\r\n\r\n"
            ).encode("ascii")
            for label, raw_request in legs.items():
                response, ended, elapsed = b"", False, 0.0
                for attempt in range(3):
                    response, ended, elapsed = _trickle_exchange_once(
                        port,
                        raw_request,
                        byte_interval=0.02,
                        deadline_seconds=2.0,
                    )
                    if _response_complete(response):
                        break
                    time.sleep(0.05 * (attempt + 1))
                assert _response_complete(response), (label, response[:200])
                assert response.startswith(b"HTTP/1.1 403 "), (label, response[:200])
                assert b'"error":"forbidden"' in response, label
                assert ended, f"{label}: trickling body kept the refused socket open"
                assert elapsed < 1.5, (label, elapsed)
        finally:
            seeded.app.allowed_authorities = saved_authorities


def test_first_open_destination_record_persists_across_relaunch(
    seeded_env: tuple[SeededStudy, Path, Path, Path]
) -> None:
    """P1-4 witness: the destination page's first-open payload (seriesUid
    null, destRoot empty exactly as recordRecentCase builds it) lands on a
    FRESH store, gets a server-assigned sequence, fills destRoot from the
    anchored stage+study derivation, and survives a relaunch."""
    seeded, src, stage, store_file = seeded_env
    study_a_dir = src / "study_a"
    write_synthetic_cine(study_a_dir, series_uid=SERIES_A, study_uid=STUDY_A)

    with running_server(seeded.app) as (srv, base_url):
        _safe_request(srv, base_url, "/api/studies")  # warmup
        fresh = _safe_request(srv, base_url, "/api/library/recent")
        assert json_value(fresh)["entries"] == []
        assert json_value(fresh)["reason"] == "absent"

        post = _safe_request(
            srv,
            base_url,
            "/api/library/recent",
            method="POST",
            body=json_body({"entry": _fresh_entry(src, stage)}),
        )
        assert post.status == 200
        saved = json_value(post)["entries"][0]
        assert saved["studyUid"] == STUDY_A
        assert saved["seriesUid"] is None
        assert saved["openedSeq"] == 1  # server-assigned (client sent 0)
        assert saved["openCount"] == 1
        # destRoot was sent empty: server fills the anchored derivation.
        assert saved["destRoot"] == str((stage / STUDY_A).resolve())

    with running_server(seeded.app) as (srv2, base_url2):
        relaunch = _safe_request(srv2, base_url2, "/api/library/recent")
        assert relaunch.status == 200
        kept = json_value(relaunch)["entries"][0]
        assert kept["openedSeq"] == saved["openedSeq"]
        assert kept["studyUid"] == STUDY_A
        assert store_file.is_file()


def test_app_js_first_open_handoff_wiring() -> None:
    """P1-4 wiring witness: app.js queues the pending open BEFORE the shell
    invoke (the outgoing page cannot POST — A is dead before the command
    returns), the bound destination evaluates it at startup, and every
    failure path (including the `#switch-failed` restored page) clears it.

    Round-2 P1: the wiring must also carry the four guards — init-time
    eligibility snapshot, producer-authority exclusion, bound-study
    verification, and id-correlated clear/consume (the executable
    ``test_handoff_*`` scenarios drive these for real)."""
    app_js_path = Path(__file__).parents[2] / "services" / "workstation" / "static" / "app.js"
    app_js = app_js_path.read_text(encoding="utf-8")

    # Helpers exist and use a same-host, port-agnostic channel with TTL.
    for helper in (
        "function setPendingOpen(entry)",
        "function clearPendingOpen(expectedId)",
        "function maybeConsumePendingOpen()",
        "function readPendingIntent()",
        "function discardIncomingIntent(expectedId)",
        "function beginHandoff()",
        "function endHandoff()",
    ):
        assert helper in app_js, helper
    assert 'PENDING_OPEN_COOKIE = "cmr_pending_open"' in app_js
    assert "SameSite=Lax" in app_js
    assert "max-age=120" in app_js

    # Round-3 R3-1: handoffs are single-flight and every continuation clears
    # only the id its own action created (never a page-global).
    assert "if (handoffOutstanding) return false;" in app_js
    assert app_js.count("if (!beginHandoff()) {") == 2  # switch + reopen
    assert app_js.count("const intentId = setPendingOpen(") == 2
    assert app_js.count("clearPendingOpen(intentId);") >= 6
    assert "producedIntentId" not in app_js

    # Round-2 P1 guards are present in production code.
    assert "incomingIntent = readPendingIntent();" in app_js  # snapshot at init
    assert "producer: window.location.host" in app_js        # producer stamp
    assert "live.producer === window.location.host" in app_js
    assert "pending.entry.studyUid" in app_js  # binding via server slot
    assert "if (consumedIntentId) return;" in app_js         # exactly once

    # Switch affordance: cookie, then ONE awaited queue POST, then invoke.
    switch_start = app_js.index('show("Switch requested:')
    switch_end = app_js.index("section.appendChild(button);", switch_start)
    switch_region = app_js[switch_start:switch_end]
    assert switch_region.index("setPendingOpen(") < switch_region.index("queuePendingOpen(intentId")
    assert switch_region.index("queuePendingOpen(intentId") < switch_region.index(
        'invoke("switch_study"'
    )

    # Reopen affordance: same single-awaited-queue-then-invoke order.
    reopen_start = app_js.index('show("Reopen requested for study:')
    reopen_region = app_js[reopen_start : reopen_start + 6000]
    assert reopen_region.index("setPendingOpen(") < reopen_region.index("queuePendingOpen(intentId")
    assert reopen_region.index("queuePendingOpen(intentId") < reopen_region.index(
        'invoke("switch_study"'
    )

    # QUEUE-011 single-fire: exactly one awaited queue call per leg (switch +
    # reopen) - setPendingOpen itself fires no POST, so each click queues
    # exactly once and the invoke is gated on its settlement. The correlated
    # staging renewal (PR3 review 5564276070 F2) is a separate same-id re-queue
    # inside each needs_stage chain, never a second click-time queue.
    assert switch_region.count("queuePendingOpen(intentId") == 1
    assert reopen_region.count("queuePendingOpen(intentId") == 1
    assert app_js.count("await queuePendingOpen(") == 3  # 2 click legs + renewal
    assert switch_region.count("renewPendingOpen(intentId, entryPayload)") == 1
    assert reopen_region.count("renewPendingOpen(intentId, entry)") == 1
    # Queue deadline preserved without AbortController (P2 review fix):
    # queuePendingOpen must race against timeout even when AbortController is absent.
    assert 'typeof AbortController !== "undefined" ? new AbortController() : null' in app_js
    assert "Promise.race([attempt, timeout])" in app_js

    # The outgoing page must NOT POST the record (A is dead post-invoke).
    assert "return recordRecentCase(entry)" not in app_js

    # The bound destination evaluates the handoff at startup and again once
    # the bound study is known, recording through the single sanctioned writer.
    startup = app_js.index("mountRecentAffordance();")
    consume_at = app_js.index("maybeConsumePendingOpen();", startup)
    assert consume_at > startup
    assert "confirmPendingOpen(live)" in app_js
    assert "recordRecentCase(pending.entry, hint.id)" in app_js

    # Every non-switched outcome and catch path clears the intent, correlated
    # to the id this page produced. The ONLY uncorrelated force-clear is the
    # restored page's `discardIncomingIntent()` (no id argument).
    assert app_js.count("clearPendingOpen(intentId);") >= 6
    assert app_js.count("discardIncomingIntent();") == 1
    assert app_js.count("discardIncomingIntent(live.id);") == 1

    # The restored-page navigation (`#switch-failed`) discards it too.
    restored = app_js.index('window.location.hash === "#switch-failed"')
    assert "discardIncomingIntent();" in app_js[restored : restored + 500]

    # Failed saves are surfaced AND propagated (PENDING-SUCCESS-015): the typed
    # rejection lets confirmPendingOpen observe the failure so a failed
    # persistence can never clear the pending slot as successful.
    assert "Recent case save failed" in app_js
    assert 'rejectErr.code = "record-rejected";' in app_js
    assert "throw rejectErr;" in app_js

    # SAMEROOT-014 shell grant (defensive): every `switched` outcome still
    # feeds a shell-provided dest_root into the live-root tracker when the
    # shell reports one (production `switched` results carry only
    # status/authority, so the server bound-root route below is the real
    # initialization path).
    assert (
        'if (typeof retryResult.dest_root === "string") noteLiveRoot(retryResult.dest_root);'
        in app_js
    )
    assert 'if (typeof outcome.dest_root === "string") noteLiveRoot(outcome.dest_root);' in app_js
    assert (
        'if (typeof finalResult.dest_root === "string") noteLiveRoot(finalResult.dest_root);'
        in app_js
    )

    # R7 F5: the live root is the SERVER-reported bound root, learned on
    # every initialization (independent of pending intents and of the recent
    # store); the pending entry no longer seeds it. Genuine granted-shell
    # dest_root handling stays defensive.
    assert '/api/library/bound-root' in app_js
    assert "function loadBoundRoot()" in app_js
    assert "noteLiveRoot(data.root);" in app_js
    assert "noteLiveRoot(canonicalDestRoot(pending.entry));" not in app_js
    # Bound-root initialization is memoized and STARTED on every page load,
    # and every live-root decision awaits it (delayed bound-root review fix).
    assert "function boundRootInitialized()" in app_js
    assert "loadBoundRoot().catch(function () {});" not in app_js
    assert "\n  boundRootInitialized();\n" in app_js
    assert "await boundRootInitialized();" in app_js
    # Supplementary review P2: complete fetch+body operation raced against
    # deadline, abort controller kept through body parsing, attempt isolation.
    assert "Promise.race([fetchAndParse, timeoutPromise])" in app_js
    assert "thisAttempt !== boundRootAttempt" in app_js

    # PR3 review #4 (app.js:1230): no client lexical destination derivation
    # remains. Pending confirmation compares SERVER-RESOLVED destinations (the
    # same probe Reopen uses) and still never consults the display-only
    # persisted destRoot (R7 F2: a contradictory destRoot must not suppress a
    # genuine switch).
    assert "canonicalDestRoot" not in app_js
    assert "noteLiveRoot(canonicalDestRoot(pending.entry));" not in app_js
    confirm_start = app_js.index("async function confirmPendingOpen(hint)")
    confirm_end = app_js.index("\n  function schedulePendingConfirmRetry(", confirm_start)
    confirm_body = app_js[confirm_start:confirm_end]
    assert "fetchResolvedDestRoot(pending.entry" in confirm_body
    assert "entry.destRoot" not in confirm_body
    # PR3 review #4 (app.js:1219): a different-UID consumer STANDS BY instead of
    # retiring the shared hint/slot (only absent/id-mismatched slots are refused).
    assert (
        'state.study_uid !== pending.entry.studyUid) return "standby"'
        in confirm_body
    )
    # Shared bounded resolve: pending confirmation and Reopen consume ONE
    # destination fetch (complete fetch+body race, attempt isolation) — the row
    # probe keeps no second copy of the request.
    assert "function fetchResolvedDestRoot(entry, isStale)" in app_js
    assert "return fetchResolvedDestRoot(entry, function () {" in app_js
    assert app_js.count('"/api/library/recent/probe?') == 1
    row_probe_start = app_js.index("function loadEntryProbe()")
    row_probe_end = app_js.index("\n      function ensureEntryProbe()", row_probe_start)
    row_probe_body = app_js[row_probe_start:row_probe_end]
    assert "fetchResolvedDestRoot(entry" in row_probe_body
    assert "fetch(" not in row_probe_body

    # PR3 review 5564032012 F1: a failed/timed-out bound-root lookup is NOT a
    # proven non-destination — the confirmation retries with a FRESH lookup
    # (failed memoization reset) instead of standing by silently.
    assert (
        'boundRootStatus === "timed_out" || boundRootStatus === "failed"'
        in confirm_body
    )
    assert "boundRootReady = null;" in confirm_body
    assert confirm_body.count('return "retry";') >= 2

    # PR3 review 5564032012 F2: ONE coherent bounded record acknowledgement
    # (POST headers+body, 409 body+retry, ack body), unknown-commit retry with
    # the SAME open id, and ancillary refresh/cleanup bounded so a KNOWN ack is
    # never blocked or downgraded.
    assert "function recordAckTimeoutMs()" in app_js
    assert "Promise.race([exchange, deadline])" in app_js
    assert 'unknownErr.code = "record-unknown";' in app_js
    assert "if (digest !== null) recentDigest = digest;" in app_js
    assert "refreshRecentListBounded();" in app_js
    assert "function clearPendingSlotBounded(id)" in app_js
    assert "await clearPendingSlotBounded(hint.id);" in app_js
    assert 'error.code === "record-unknown"' in confirm_body
    record_start = app_js.index("async function recordRecentCase(entry, openId)")
    record_end = app_js.index("\n  function refreshRecentListBounded()", record_start)
    record_body = app_js[record_start:record_end]
    # PR3 review 5564147464: definite rejections are handled FIRST (never
    # retried); every other failure is normalized to record-unknown, so an
    # immediate fetch rejection or an interrupted ack body is retried with the
    # same id instead of being reported as a lost save.
    rejected_at = record_body.index('err.code === "record-rejected"')
    unknown_at = record_body.index('unknownErr.code = "record-unknown";')
    assert rejected_at < unknown_at
    assert "timedOut ||" not in record_body
    # The exchange itself never mutates shared state or surfaces notices
    # (late-settlement isolation): only the race winner may publish a digest.
    exchange_start = record_body.index("const exchange = (async function () {")
    exchange_end = record_body.index("})();", exchange_start)
    exchange_body = record_body[exchange_start:exchange_end]
    assert "recentDigest" not in exchange_body
    assert "show(" not in exchange_body
    assert "await loadRecentCases();" not in record_body

    # R11 F2: Reopen awaits ensureEntryProbe() and uses the server-resolved
    # probedDestRoot as authoritative destination before evaluating liveRoot equality.
    assert "await ensureEntryProbe();" in app_js
    assert "function loadEntryProbe()" in app_js
    assert "const entryDest = canonicalRootString(probedDestRoot);" in app_js

    # R7 F4: the queue-failure notice is armed in BOTH legs before the
    # invoke, and consumed exactly once by the surviving destination page.
    assert app_js.count('armRecentNotice("queue-error");') == 4
    assert 'RECENT_NOTICE_COOKIE = "cmr_recent_notice"' in app_js
    assert "function consumeRecentNotice()" in app_js
    assert "consumeRecentNotice();" in app_js

# ---------------------------------------------------------------------------
# Round-2 P1: executable cookie-handoff lifecycle witnesses (real app.js)
# ---------------------------------------------------------------------------

_HANDOFF_HARNESS_JS = r"""
const fs = require('fs');
const vm = require('vm');

const SCENARIO = process.env.SCENARIO || '';
const HOST = process.env.HOST || '127.0.0.1:2222';
const PRODUCER = process.env.PRODUCER || '127.0.0.1:1111';
const TARGET_STUDY = process.env.TARGET_STUDY || '';
const BOUND_STUDY = process.env.BOUND_STUDY || '';
const INIT_COOKIE = process.env.INIT_COOKIE === '1';
const PAGE_HASH = process.env.PAGE_HASH || '';
const PENDING_READ_FAILS = Number(process.env.PENDING_READ_FAILS || 0);
const PENDING_READ_HANG = process.env.PENDING_READ_HANG === '1';
const PENDING_READ_HANG_BODY = process.env.PENDING_READ_HANG_BODY === '1';
const PENDING_TIMEOUT_MS = process.env.PENDING_TIMEOUT_MS || '';
const PENDING_MAX_ATTEMPTS = process.env.PENDING_MAX_ATTEMPTS || '';
const PENDING_RETRY_DELAY_MS = process.env.PENDING_RETRY_DELAY_MS || '';
const DELAY_RECENT_GET = process.env.DELAY_RECENT_GET === '1';
const HOLD_STARTUP_RECENT_BODY = process.env.HOLD_STARTUP_RECENT_BODY === '1';
// PR3 review r4311831210 witness knobs: a genuine DIFFERENT series to select
// mid-confirmation, and a hold on EVERY /state answer while the scenario has it
// armed, so the pending read can settle inside the transient `state === null`
// window a series change (chooseSeries) opens.
const EXTRA_SERIES_UID = process.env.EXTRA_SERIES_UID || '';
const HOLD_STATE_FOR_RACE = process.env.HOLD_STATE_FOR_RACE === '1';
const STALE_RECENT_ENTRIES_JSON = process.env.STALE_RECENT_ENTRIES_JSON || '';
const RECENT_ENTRIES_JSON = process.env.RECENT_ENTRIES_JSON || '';
const DELAY_QUEUE = process.env.DELAY_QUEUE === '1';
const HANG_QUEUE = process.env.HANG_QUEUE === '1';
const QUEUE_FAIL = process.env.QUEUE_FAIL === '1';
const RECORD_FAIL = process.env.RECORD_FAIL === '1';
const BOUND_ROOT = process.env.BOUND_ROOT || '';
const BOUND_ROOT_AFTER_RELOAD = process.env.BOUND_ROOT_AFTER_RELOAD || '';
const BOUND_STUDY_AFTER_RELOAD = process.env.BOUND_STUDY_AFTER_RELOAD || '';
const PENDING_STAGE_BASE = process.env.PENDING_STAGE_BASE || '/stage';
const PROBE_DEST_ROOT = process.env.PROBE_DEST_ROOT || '';
const DELAY_PROBE = process.env.DELAY_PROBE === '1';
const HANG_PROBE = process.env.HANG_PROBE === '1';
const PROBE_FAIL = process.env.PROBE_FAIL === '1';
let releaseProbe = null;
const probeGate = new Promise(function (resolve) { releaseProbe = resolve; });
let probeRequests = 0;
let probeAborted = false;
const DELAY_BOUND_ROOT = process.env.DELAY_BOUND_ROOT === '1';
const HANG_BOUND_ROOT = process.env.HANG_BOUND_ROOT === '1';
const HANG_BOUND_ROOT_BODY = process.env.HANG_BOUND_ROOT_BODY === '1';
const BOUND_ROOT_FAILS = Number(process.env.BOUND_ROOT_FAILS || 0);
const RECORD_ACK_TIMEOUT_MS = process.env.RECORD_ACK_TIMEOUT_MS || '';
const HANG_RECORD_HEADERS = process.env.HANG_RECORD_HEADERS === '1';
const HANG_RECORD_BODY = process.env.HANG_RECORD_BODY === '1';
const HANG_409_BODY = process.env.HANG_409_BODY === '1';
const RECORD_FETCH_REJECT = process.env.RECORD_FETCH_REJECT === '1';
const RECORD_BODY_REJECT = process.env.RECORD_BODY_REJECT === '1';
const RECORD_409_BODY_REJECT = process.env.RECORD_409_BODY_REJECT === '1';
const HANG_REFRESH_AFTER_RECORD = process.env.HANG_REFRESH_AFTER_RECORD === '1';
const HANG_CLEAR = process.env.HANG_CLEAR === '1';
const LATE_RECORD = process.env.LATE_RECORD === '1';
const INVOKE_REFUSED = process.env.INVOKE_REFUSED === '1';
const HANG_STAGE = process.env.HANG_STAGE === '1';
const STAGE_RENEW_FAILS = Number(process.env.STAGE_RENEW_FAILS || 0);
const STAGE_RENEW_HANG = process.env.STAGE_RENEW_HANG === '1';
const DELAY_STAGE = process.env.DELAY_STAGE === '1';
const DELAY_INITIAL_STATE = process.env.DELAY_INITIAL_STATE === '1';
// PR3 review r4310770084 knob: '' | 'headers' | 'body' — the FIRST recent GET
// after the scenario resets the counter stalls either before any response
// (headers) or after headers with a never-settling body.
const STALL_RECENT_GET = process.env.STALL_RECENT_GET || '';
const INIT_INTENT_EXP_MS = Number(process.env.INIT_INTENT_EXP_MS || 0);
// Controlled clock (PR3 review 5564276070 F2 witness): scenarios can advance
// page time past the original 120s intent deadline without waiting.
let clockOffset = 0;
const realDateNow = Date.now;
Date.now = function () { return realDateNow() + clockOffset; };
const cookieExpiries = {};
let cookieExpiryCount = 0;
let releaseInitialState = null;
let initialStateReleased = false;
const initialStateGate = new Promise(function (resolve) { releaseInitialState = resolve; });
function releaseInitialStateGate() {
  // Review 5566059819 (P2-2 witness): a delayed FIRST /state response can be
  // released after the claimed intent's expiry without waiting out the test.
  initialStateReleased = true;
  if (typeof releaseInitialState === 'function') releaseInitialState();
}
let boundRootFirstBodyResolve = null;
let boundRootBodyAborted = false;

const COOKIE_NAME = 'cmr_pending_open';
const NOTICE_COOKIE = 'cmr_recent_notice';

// ---- document.cookie shim (multi-cookie jar with max-age=0 deletion;
//      `jar` remains the pending-cookie view the existing witnesses assert
//      on, so all previous shape checks keep their meaning) ----
const cookies = {};            // name -> encoded value
let jar = null;                // pending-cookie view (legacy witnesses)
const jarShapes = [];
const noticeShapes = [];
function decodedShape(raw) {
  try {
    const decoded = JSON.parse(decodeURIComponent(raw));
    const keys = Object.keys(decoded).sort().join(',');
    const flat = JSON.stringify(decoded);
    const hasPaths = /scanSourcePath|stageBaseDir|destRoot|src\/|stage\/|:\/\//.test(flat);
    return {keys: keys, has_paths: hasPaths};
  } catch (_) {
    return {keys: null, has_paths: false};
  }
}
function syncJarView() {
  cookieExpiryCount += 1;
  for (const [name, expiry] of Object.entries(cookieExpiries)) {
    if (expiry !== null && Date.now() >= expiry) {
      delete cookies[name];
      delete cookieExpiries[name];
    }
  }
  jar = Object.prototype.hasOwnProperty.call(cookies, COOKIE_NAME) ? cookies[COOKIE_NAME] : null;
}
function noteJarShape() {
  try {
    if (jar !== null) jarShapes.push(decodedShape(jar));
  } catch (_) {}
}
function noteNoticeShape() {
  if (Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE)) {
    noticeShapes.push(decodedShape(cookies[NOTICE_COOKIE]));
  }
}
function writeCookieRaw(name, value, maxAgeSeconds) {
  if (value === null) {
    delete cookies[name];
    delete cookieExpiries[name];
  } else {
    cookies[name] = value;
    cookieExpiries[name] = typeof maxAgeSeconds === 'number'
      ? Date.now() + maxAgeSeconds * 1000
      : null;
  }
  if (name === COOKIE_NAME) {
    syncJarView();
    noteJarShape();
  }
  if (name === NOTICE_COOKIE) noteNoticeShape();
}
function writeIntent(producer, studyUid, intentId, expMs) {
  // Identity-only hint (confidentiality review P1): the cookie carries NO
  // entry and NO filesystem paths — only the opaque slot id, expiry, and
  // producer stamp. An explicit expMs lets a witness use a short-lived real
  // intent so claim-expiry cleanup is observable without waiting out 60s.
  const exp = Date.now() + (typeof expMs === 'number' && expMs > 0 ? expMs : 60000);
  writeCookieRaw(COOKIE_NAME, encodeURIComponent(JSON.stringify({
    id: intentId || 'poHarness1',
    exp: exp,
    producer: producer
  })), Math.max(1, Math.ceil((exp - Date.now()) / 1000)));
}
if (INIT_COOKIE) writeIntent(PRODUCER, TARGET_STUDY, undefined, INIT_INTENT_EXP_MS || undefined);

// ---- DOM shim (first-writer-wins id map: app.js reuses the id
//      `switch-study` for BOTH the UID field and the Switch button) ----
const allElements = [];
const elements = new Map();
function makeElement(tag, id, cls) {
  const listeners = new Map();
  const attrs = new Map();
  const element = {
    tagName: (tag || 'div').toUpperCase(),
    id: id || '',
    className: cls || '',
    value: id === 'fps' ? '10' : '',
    textContent: '',
    style: {},
    dataset: {},
    children: [],
    parentNode: null,
    setAttribute: function (k, v) { attrs.set(k, String(v)); },
    getAttribute: function (k) { return attrs.get(k) || null; },
    // PR3 review r4311831218 witness: the active marker must be RECONCILED, so
    // a row that stops being current must lose aria-current (not keep "true").
    removeAttribute: function (k) { attrs.delete(k); },
    hasAttribute: function (k) { return attrs.has(k); },
    classList: {
      toggle: function (c, val) {
        const parts = new Set((element.className || '').split(' ').filter(Boolean));
        if (val === undefined ? !parts.has(c) : Boolean(val)) parts.add(c);
        else parts.delete(c);
        element.className = Array.from(parts).join(' ');
      },
      add: function (c) { element.classList.toggle(c, true); },
      remove: function (c) { element.classList.toggle(c, false); },
      contains: function (c) { return (element.className || '').split(' ').includes(c); }
    },
    addEventListener: function (name, fn) { listeners.set(name, fn); },
    appendChild: function (child) {
      child.parentNode = element;
      element.children.push(child);
      if (child.id && !elements.has(child.id)) elements.set(child.id, child);
      return child;
    },
    replaceChildren: function (...kids) {
      element.children = [];
      for (const k of kids) element.appendChild(k);
    },
    insertBefore: function (child, ref) {
      child.parentNode = element;
      const idx = element.children.indexOf(ref);
      if (idx !== -1) element.children.splice(idx, 0, child);
      else element.children.push(child);
      if (child.id && !elements.has(child.id)) elements.set(child.id, child);
      return child;
    },
    removeChild: function (child) {
      const idx = element.children.indexOf(child);
      if (idx !== -1) element.children.splice(idx, 1);
      child.parentNode = null;
      return child;
    },
    getContext: function () {
      return {clearRect: function () {}, drawImage: function () {}, putImageData: function () {}};
    },
    click: function () {
      const fn = listeners.get('click');
      if (fn) fn({target: element});
    }
  };
  allElements.push(element);
  if (id && !elements.has(id)) elements.set(id, element);
  return element;
}
function el(id) {
  if (!elements.has(id)) makeElement('div', id);
  return elements.get(id);
}
function findById(node, id) {
  if (node.id === id) return node;
  for (const child of node.children || []) {
    const hit = findById(child, id);
    if (hit) return hit;
  }
  return null;
}
function findButton(id) {
  return allElements.find(function (e) { return e.id === id && e.tagName === 'BUTTON'; }) || null;
}

const viewport = makeElement('section', '', 'viewport');
const message = makeElement('p', 'message');
// R5-ratified: capture EVERY show() write (app.js sets textContent on this
// element). The message line is a shared channel — an unrelated async init
// failure can overwrite the final text AFTER the notice — so witnesses
// assert on the log (notice was shown), never only on the final text.
const messageLog = [];
(function () {
  let current = message.textContent;
  Object.defineProperty(message, 'textContent', {
    configurable: true,
    get: function () { return current; },
    set: function (v) { current = String(v); messageLog.push(String(v)); }
  });
}());
viewport.appendChild(message);
const controls = makeElement('aside', '', 'controls');

const documentShim = {
  body: makeElement('body'),
  getElementById: el,
  createElement: function (tag) { return makeElement(tag); },
  createTextNode: function (val) { return {textContent: val}; },
  querySelector: function (sel) {
    if (sel === 'section.viewport') return viewport;
    if (sel === 'aside.controls') return controls;
    return null;
  },
  addEventListener: function () {},
  get cookie() {
    syncJarView();
    return Object.keys(cookies).map(function (n) { return n + '=' + cookies[n]; }).join('; ');
  },
  set cookie(v) {
    const parts = String(v).split(';').map(function (s) { return s.trim(); });
    const nv = parts[0];
    const eq = nv.indexOf('=');
    if (eq <= 0) return;
    const name = nv.slice(0, eq);
    const val = nv.slice(eq + 1);
    let maxAge = null;
    for (const p of parts.slice(1)) {
      const match = /^max-age=(-?\d+)$/i.exec(p);
      if (match) maxAge = Number(match[1]);
    }
    const expired = maxAge !== null && maxAge <= 0;
    writeCookieRaw(name, (expired || val === '') ? null : val, maxAge);
  }
};

global.document = documentShim;

// ---- window (host + hash are load-bearing for the handoff guards) ----
// The host is mutable so a scenario can model the DESTINATION authority after a
// navigation (a real switch lands on a different loopback port).
let currentHost = HOST;
const windowShim = {
  document: documentShim,
  location: {hash: PAGE_HASH, pathname: '/', search: '', get host() { return currentHost; }},
  history: {replaceState: function () {}},
  CMR_KEYBOARD_SHORTCUTS: {},
  setInterval: function () {},
  setTimeout: setTimeout,
  createImageBitmap: async function () { return {width: 100, height: 100, close: function () {}}; },
  __CMR_BOUND_ROOT_TIMEOUT_MS: (
    process.env.BOUND_ROOT_TIMEOUT_MS ? Number(process.env.BOUND_ROOT_TIMEOUT_MS) : 2500
  ),
  // PR3 review r4310770076 knob: bounded wait for the independently
  // established bound study identity (the live backend's DICOM-derived state
  // projection) before a same-root Reopen no-op decision.
  __CMR_BOUND_IDENTITY_TIMEOUT_MS: (
    process.env.BOUND_IDENTITY_TIMEOUT_MS
      ? Number(process.env.BOUND_IDENTITY_TIMEOUT_MS)
      : 2500
  ),
  __CMR_QUEUE_TIMEOUT_MS: (
    process.env.QUEUE_TIMEOUT_MS ? Number(process.env.QUEUE_TIMEOUT_MS) : 5000
  ),
  __CMR_PROBE_TIMEOUT_MS: (
    process.env.PROBE_TIMEOUT_MS ? Number(process.env.PROBE_TIMEOUT_MS) : 2500
  ),
  // PR3 review F3 knobs: full fetch+body deadline, bounded attempt budget and
  // retry delay for the pending-open confirmation read.
  __CMR_PENDING_TIMEOUT_MS: (PENDING_TIMEOUT_MS ? Number(PENDING_TIMEOUT_MS) : 2500),
  __CMR_PENDING_MAX_ATTEMPTS: (PENDING_MAX_ATTEMPTS ? Number(PENDING_MAX_ATTEMPTS) : 3),
  __CMR_PENDING_RETRY_DELAY_MS: (PENDING_RETRY_DELAY_MS ? Number(PENDING_RETRY_DELAY_MS) : 400),
  // PR3 review 5564032012 F2 knobs: one coherent record-ack deadline and the
  // bounded ancillary cleanup/refresh budget.
  __CMR_RECORD_ACK_TIMEOUT_MS: (
    process.env.RECORD_ACK_TIMEOUT_MS ? Number(process.env.RECORD_ACK_TIMEOUT_MS) : 8000
  ),
  __CMR_PENDING_CLEANUP_TIMEOUT_MS: (
    process.env.PENDING_CLEANUP_TIMEOUT_MS ? Number(process.env.PENDING_CLEANUP_TIMEOUT_MS) : 1500
  )
};
global.window = windowShim;
if (process.env.NO_ABORT_CONTROLLER === '1') {
  delete global.AbortController;
  delete windowShim.AbortController;
}
global.URLSearchParams = URLSearchParams;
global.createImageBitmap = windowShim.createImageBitmap;

// ---- cine bundle (copied from the repository's switch-study harness) ----
const frameRow = {frame_index: 0, byte_offset: 0, byte_length: 4};
const manifestObj = {version: 1, media: 'application/png-frame-bundle', frames: [frameRow]};
const manifestBuf = Buffer.from(JSON.stringify(manifestObj), 'utf8');
const headerBuf = Buffer.alloc(8);
headerBuf.writeBigUInt64BE(BigInt(manifestBuf.length), 0);
const fullBundle = Buffer.concat([headerBuf, manifestBuf, Buffer.from([1, 2, 3, 4])]);

// ---- fetch mock: captures recent POSTs, optionally gates the startup GET ----
const posts = [];
let boundRootRequests = 0;
let releaseRecentGet = null;
const recentGetGate = new Promise(function (resolve) { releaseRecentGet = resolve; });
// Held startup recent GET body (PR3 review r4310080007 witness): the FIRST recent
// GET's response body stays pending until releaseStartupRecentBody(), so a
// witness can drive a save + newer refresh and only THEN let the stale startup
// body arrive. Subsequent GETs answer normally.
let releaseStartupRecentBody = null;
const startupRecentBodyGate = new Promise(function (resolve) {
  releaseStartupRecentBody = resolve;
});
let recentGetRequests = 0;
// Delayed bound-root (review fix witness): when DELAY_BOUND_ROOT is set, the
// /api/library/bound-root answer stays pending until releaseBoundRoot(), so a
// witness can click Reopen while currentLiveRoot() would still be null.
let releaseBoundRoot = null;
const boundRootGate = new Promise(function (resolve) { releaseBoundRoot = resolve; });

// PR3 review r4311831210 witness: while the scenario has the state hold armed,
// every /state answer stays pending until releaseHeldStateNow(), so the
// transient `state === null` window a series change opens (chooseSeries) stays
// open exactly as long as the witness needs. The interval heartbeat is stubbed
// in this harness, so no other state answer can close that window.
let stateHoldArmed = false;
let stateHolds = 0;
let releaseHeldState = null;
const stateHoldGate = new Promise(function (resolve) { releaseHeldState = resolve; });
function releaseHeldStateNow() {
  stateHoldArmed = false;
  if (typeof releaseHeldState === 'function') {
    const resolve = releaseHeldState;
    releaseHeldState = null;
    resolve();
  }
}

// Server-side pending slot mock (mirrors POST/GET/recent/pending + clear):
// the producer queues the FULL entry here; the cookie carries only the hint.
// The harness pre-seeds the slot for INIT_COOKIE scenarios (the producer page
// queued before THIS page loaded, then died with backend A).
// QUEUE-011: every queue POST body is captured in pendingPosts (the record
// POSTs in posts[] never include queue writes); DELAY_QUEUE holds non-clear
// queue POSTs unresolved until releaseQueue() so witnesses can prove the
// shell invoke is gated on queue persistence (killed-A race).
let pendingSlot = null;
// Server-reported bound root AND the state's study UID for THIS page: both
// start as the scenario values and are mutable so a destination reload can
// model the real destination (matching UID + root) inside one run.
let currentBoundRoot = BOUND_ROOT;
let currentBoundStudy = BOUND_STUDY;
const pendingPosts = [];
// PR3 review F3 witness state: read attempts, correlated clears, ordered
// event log (a clear must never precede its record), and the late-settlement
// releases for the hang/body-hang injections.
let pendingGets = 0;
let pendingSlotAborted = false;
let pendingFirstBodyResolve = null;
let pendingHangPayload = null;
let pendingBodyResolved = 0;
let releaseHeldPendingRead = null;
const pendingClears = [];
const handoffEvents = [];
// PR3 review 5564032012 F2 witness state: record attempts, held ack resolvers
// (late-settlement isolation) and the ancillary-refresh release.
let recordPosts = 0;
let heldRecordResolvers = [];
function releaseRecord() {
  const rs = heldRecordResolvers;
  heldRecordResolvers = [];
  rs.forEach(function (r) { r(); });
}
let heldStageResolvers = [];
function releaseStage() {
  const rs = heldStageResolvers;
  heldStageResolvers = [];
  rs.forEach(function (r) { r(); });
}
function releasePendingRead() {
  if (typeof releaseHeldPendingRead === 'function') {
    const resolve = releaseHeldPendingRead;
    releaseHeldPendingRead = null;
    resolve();
  }
}
function releasePendingReadBody() {
  if (typeof pendingFirstBodyResolve === 'function') {
    const resolve = pendingFirstBodyResolve;
    pendingFirstBodyResolve = null;
    pendingBodyResolved += 1;
    resolve(pendingHangPayload || {pending: false});
  }
}
let heldQueueResolvers = [];
function releaseQueue() {
  const rs = heldQueueResolvers;
  heldQueueResolvers = [];
  rs.forEach(function (r) { r(); });
}
function seedPendingSlot(producer, studyUid) {
  pendingSlot = {id: 'poHarness1', exp: Date.now() + 60000, entry: {
    studyUid: studyUid, seriesUid: null, scanSourcePath: '/src',
    stageBaseDir: PENDING_STAGE_BASE, destRoot: '', openedSeq: 0, openCount: 1
  }};
}
if (INIT_COOKIE) seedPendingSlot(PRODUCER, TARGET_STUDY);

global.fetch = async function (url, opts) {
  const method = (opts && opts.method) || 'GET';
  const u = String(url);
  if (u.indexOf('/api/library/bound-root') !== -1) {
    // F5: server-reported genuinely bound root (empty string when unset).
    boundRootRequests += 1;
    if (BOUND_ROOT_FAILS > 0 && boundRootRequests <= BOUND_ROOT_FAILS) {
      // PR3 review #5 (app.js:1316): transient lookup failure (non-OK). The
      // confirmation must treat this as unavailable, never as a different
      // workspace, and retry with a FRESH lookup.
      return {ok: false, status: 503, json: async function () {
        return {error: 'unavailable'};
      }};
    }
    if (HANG_BOUND_ROOT) {
      // Finding 2: never-settling response models a hung server or stalled body.
      return new Promise(function () {});
    }
    if (HANG_BOUND_ROOT_BODY) {
      // Supplementary review P2 witness: 200 headers arrive immediately, but
      // body stream never settles on request 1 until explicitly resolved.
      if (boundRootRequests === 1) {
        return {
          ok: true,
          status: 200,
          json: function () {
            return new Promise(function (resolve, reject) {
              boundRootFirstBodyResolve = resolve;
              if (opts && opts.signal) {
                opts.signal.addEventListener('abort', function () {
                  boundRootBodyAborted = true;
                  const abortErr = new Error('The operation was aborted');
                  abortErr.name = 'AbortError';
                  reject(abortErr);
                });
              }
            });
          }
        };
      }
      return {
        ok: true,
        status: 200,
        json: async function () {
          return {root: currentBoundRoot};
        }
      };
    }
    if (DELAY_BOUND_ROOT) await boundRootGate;
    return {ok: true, status: 200, json: async function () {
      return {root: currentBoundRoot};
    }};
  }
  if (u.indexOf('/api/library/recent/probe') !== -1) {
    // Authoritative probe answer; dest_root is the server's resolved
    // derivation for the entry.
    probeRequests += 1;
    if (HANG_PROBE) {
      return new Promise(function (_, reject) {
        if (opts && opts.signal) {
          opts.signal.addEventListener('abort', function () {
            probeAborted = true;
            const abortErr = new Error('The operation was aborted');
            abortErr.name = 'AbortError';
            reject(abortErr);
          });
        }
      });
    }
    if (PROBE_FAIL) {
      return {ok: false, status: 500, json: async function () {
        return {error: 'probe_failed'};
      }};
    }
    if (DELAY_PROBE) await probeGate;
    const urlObj = new URL('http://' + HOST + u);
    const qStage = urlObj.searchParams.get('stage') || '';
    const qStudy = urlObj.searchParams.get('study') || '';
    let defaultDestRoot = PROBE_DEST_ROOT;
    if (!defaultDestRoot && qStage && qStudy) {
      defaultDestRoot = qStage.replace(/\/+$/, '') + '/' + qStudy;
    }
    return {ok: true, status: 200, json: async function () {
      return {availability: 'ready', reuse: true, dest_root: defaultDestRoot};
    }};
  }
  if (u.indexOf('/api/library/recent/pending') !== -1) {
    if (method === 'POST') {
      const body = JSON.parse(opts.body);
      if (body && body.clear === true) {
        pendingClears.push(body.id || null);
        handoffEvents.push('pending-clear');
        if (HANG_CLEAR) {
          // PR3 review 5564032012 F2: cleanup never acknowledges — the KNOWN
          // commit must still complete inside its bounded cleanup budget.
          return new Promise(function () {});
        }
        if (!body.id || (pendingSlot && pendingSlot.id === body.id)) pendingSlot = null;
        return {ok: true, status: 200, json: async function () {
          return {pending: false};
        }};
      }
      pendingPosts.push(body);
      if (STAGE_RENEW_FAILS > 0 && pendingPosts.length > 1) {
        // Initial queue remains successful; the correlated staging renewal is
        // rejected with a definite non-OK so both needs_stage chains must arm
        // the path-free warning before their final invoke.
        return {ok: false, status: 503, json: async function () {
          return {error: 'recent.store_unwritable', message: 'rejected'};
        }};
      }
      if (STAGE_RENEW_HANG && pendingPosts.length > 1) {
        // Staging renewal queue times out while the active producer lease still
        // owns its id; timeout is best-effort but must arm the warning.
        return new Promise(function () {});
      }
      if (QUEUE_FAIL) {
        // R5-ratified queue-error leg: the queue write fails fast (500) —
        // the page must still invoke exactly once (unrecorded switch, never
        // broken, never fabricated) WITH the explicit "may not be saved"
        // notice. Pre-fix/pre-notice code invokes silently.
        return {ok: false, status: 500, json: async function () {
          return {error: 'recent.store_unwritable', message: 'rejected'};
        }};
      }
      if (DELAY_QUEUE || HANG_QUEUE) {
        // Held unresolved: the producer page waits for backend A to
        // acknowledge persistence. Released via releaseQueue() — EARLY by
        // the delayed_queue scenario (QUEUE-011 gating proof), LATE (after
        // the page's bounded queue timeout) by the queue_hang scenario
        // (R5 timeout + late-response-blocking proof). The held mock
        // deliberately ignores AbortSignal: the page's timeout race must
        // settle the await even when the fetch never settles nor aborts.
        return new Promise(function (resolve) {
          heldQueueResolvers.push(function () {
            pendingSlot = {id: body.id, exp: body.exp, entry: body.entry};
            resolve({ok: true, status: 200, json: async function () {
              return {pending: true, id: body.id, exp: body.exp};
            }});
          });
        });
      }
      pendingSlot = {id: body.id, exp: body.exp, entry: body.entry};
      return {ok: true, status: 200, json: async function () {
        return {pending: true, id: body.id, exp: body.exp};
      }};
    }
    pendingGets += 1;
    handoffEvents.push('pending-read');
    if (PENDING_READ_FAILS > 0 && pendingGets <= PENDING_READ_FAILS) {
      // F3: transient read failure (transport-class non-OK). The page must
      // retain the hint and retry the SAME idempotent open id.
      return {ok: false, status: 503, json: async function () {
        return {error: 'recent.store_unreadable', message: 'transient'};
      }};
    }
    if (PENDING_READ_HANG && pendingGets === 1) {
      // F3: the first read never settles until releasePendingRead() — a
      // concurrent second evaluation must not start another read.
      return new Promise(function (resolve, reject) {
        releaseHeldPendingRead = resolve;
        if (opts && opts.signal) {
          opts.signal.addEventListener('abort', function () {
            pendingSlotAborted = true;
            const abortErr = new Error('The operation was aborted');
            abortErr.name = 'AbortError';
            reject(abortErr);
          });
        }
      });
    }
    if (PENDING_READ_HANG_BODY && pendingGets === 1) {
      // F3: 200 headers immediately, body held until releasePendingReadBody().
      pendingHangPayload = pendingSlot
        ? {pending: true, id: pendingSlot.id, exp: pendingSlot.exp,
           entry: pendingSlot.entry}
        : {pending: false};
      return {ok: true, status: 200, json: function () {
        return new Promise(function (resolve, reject) {
          pendingFirstBodyResolve = resolve;
          if (opts && opts.signal) {
            opts.signal.addEventListener('abort', function () {
              pendingSlotAborted = true;
              const abortErr = new Error('The operation was aborted');
              abortErr.name = 'AbortError';
              reject(abortErr);
            });
          }
        });
      }};
    }
    if (!pendingSlot) {
      return {ok: true, status: 200, json: async function () {
        return {pending: false};
      }};
    }
    const slot = pendingSlot;
    return {ok: true, status: 200, json: async function () {
      return {pending: true, id: slot.id, exp: slot.exp, entry: slot.entry};
    }};
  }
  if (method === 'POST' && u.indexOf('/api/library/recent') !== -1) {
    posts.push(JSON.parse(opts.body));
    handoffEvents.push('record');
    recordPosts += 1;
    // PR3 review 5564032012 F2: acknowledgements that never settle (headers or
    // body), a 409 whose body never settles, and a held ack for the
    // late-settlement witness — each models a distinct unknown-commit window.
    if (HANG_RECORD_HEADERS && recordPosts === 1) {
      return new Promise(function () {});
    }
    if (HANG_RECORD_BODY && recordPosts === 1) {
      return {ok: true, status: 200, json: function () {
        return new Promise(function () {});
      }};
    }
    if (HANG_409_BODY && recordPosts === 1) {
      return {ok: false, status: 409, json: function () {
        return new Promise(function () {});
      }};
    }
    if (RECORD_FETCH_REJECT && recordPosts === 1) {
      // PR3 review 5564147464: immediate transport failure (connection reset)
      // with no acknowledgement at all — INDETERMINATE, never a rejection.
      return Promise.reject(new TypeError('fetch failed'));
    }
    if (RECORD_BODY_REJECT && recordPosts === 1) {
      // 200 headers, then a truncated/interrupted ack body AFTER a possible
      // commit — also INDETERMINATE.
      return {ok: true, status: 200, json: function () {
        return Promise.reject(new TypeError('terminated'));
      }};
    }
    if (RECORD_409_BODY_REJECT && recordPosts === 1) {
      // 409 headers, then a body failure before the stale-digest retry could be
      // issued — INDETERMINATE (nothing committed, nothing refused cleanly).
      return {ok: false, status: 409, json: function () {
        return Promise.reject(new TypeError('terminated'));
      }};
    }
    if (LATE_RECORD && recordPosts === 1) {
      return new Promise(function (resolve) {
        heldRecordResolvers.push(function () {
          resolve({ok: true, status: 200, json: async function () {
            return {digest: 'd-late', entries: []};
          }});
        });
      });
    }
    if (RECORD_FAIL) {
      // PENDING-SUCCESS-015: the record write fails (400, no 409 retry) —
      // the page must surface it and keep the pending slot, never clear it
      // as successful.
      return {ok: false, status: 400, json: async function () {
        return {error: 'recent.malformed', code: 'recent.malformed', message: 'rejected'};
      }};
    }
    return {ok: true, status: 200, json: async function () {
      return {digest: 'd-post', entries: []};
    }};
  }
  if (u.indexOf('/api/library/recent') !== -1) {
    if (HANG_REFRESH_AFTER_RECORD && recordPosts > 0) {
      // PR3 review 5564032012 F2: the post-save refresh hangs — the KNOWN ack
      // must still complete (bounded, detached), never become unknown.
      return new Promise(function () {});
    }
    if (STALL_RECENT_GET === 'headers' && recentGetRequests === 0) {
      // PR3 review r4310770084 witness: the startup recent GET never answers
      // at all (headers never arrive). The cross-navigation notice must still
      // be delivered, independently of this load.
      recentGetRequests += 1;
      return new Promise(function () {});
    }
    recentGetRequests += 1;
    if ((HOLD_STARTUP_RECENT_BODY || STALL_RECENT_GET === 'body') && recentGetRequests === 1) {
      // PR3 review r4310080007 witness: the startup GET answers with headers
      // immediately, but its (old-snapshot) body stays pending until the
      // witness has driven a save and a newer refresh.
      return {ok: true, status: 200, json: async function () {
        await startupRecentBodyGate;
        if (STALE_RECENT_ENTRIES_JSON) {
          return JSON.parse(STALE_RECENT_ENTRIES_JSON);
        }
        return {version: 1, entries: [], discarded: 0, reason: 'absent', digest: 'd-stale'};
      }};
    }
    if (DELAY_RECENT_GET) await recentGetGate;
    return {ok: true, status: 200, json: async function () {
      if (RECENT_ENTRIES_JSON) {
        return JSON.parse(RECENT_ENTRIES_JSON);
      }
      return {version: 1, entries: [], discarded: 0, reason: 'absent', digest: 'd0'};
    }};
  }
  if (u.indexOf('/api/library/stage') !== -1) {
    if (DELAY_STAGE || HANG_STAGE) {
      // Review 5564803264: model browser-equivalent time passing during stage;
      // the controlled-clock scenario explicitly advances 200s before release.
      return new Promise(function (resolve) {
        heldStageResolvers.push(function () {
          resolve({ok: true, status: 200, json: async function () { return {}; }});
        });
      });
    }
    return {ok: true, status: 200, json: async function () { return {}; }};
  }
  if (u.indexOf('/api/studies') !== -1) {
    return {ok: true, json: async function () {
      const studies = [{uid: 'S1', slices: 1, frames: 2}];
      // PR3 review r4311831210 witness: a genuine SECOND series the scenario can
      // select mid-confirmation (loadStudies still auto-selects the first).
      if (EXTRA_SERIES_UID) studies.push({uid: EXTRA_SERIES_UID, slices: 1, frames: 2});
      return studies;
    }};
  }
  if (u.indexOf('/state') !== -1) {
    if (stateHoldArmed) {
      // PR3 review r4311831210 witness: hold every state answer while armed.
      stateHolds += 1;
      return {ok: true, json: async function () {
        await stateHoldGate;
        return {series_uid: 'S1', study_uid: currentBoundStudy, slice: 0, frame: 0,
                fps: 10, playing: false};
      }};
    }
    if (DELAY_INITIAL_STATE && !initialStateReleased) {
      return {ok: true, json: async function () {
        await initialStateGate;
        return {series_uid: 'S1', study_uid: currentBoundStudy, slice: 0, frame: 0,
                fps: 10, playing: false};
      }};
    }
    return {ok: true, json: async function () {
      return {series_uid: 'S1', study_uid: currentBoundStudy, slice: 0, frame: 0,
              fps: 10, playing: false};
    }};
  }
  if (u.indexOf('/cine') !== -1) {
    return {ok: true, headers: {get: function () { return 'etag'; }},
            arrayBuffer: async function () { return fullBundle.buffer.slice(
              fullBundle.byteOffset, fullBundle.byteOffset + fullBundle.byteLength); }};
  }
  return {ok: false, status: 404};
};

// ---- desktop IPC mock (switch outcomes per scenario) ----
let invokeCalls = 0;
const invokeTimes = [];
// Review 5566059819: capture whether the path-free warning was already armed
// at EACH shell invoke, so a witness can prove warning-before-invoke.
const invokeNotices = [];
const pendingInvoke = [];
windowShim.__TAURI_INTERNALS__ = {
  invoke: function () {
    invokeCalls += 1;
    invokeTimes.push(Date.now());
    invokeNotices.push(Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE));
    if (INVOKE_REFUSED) {
      // PR3 review 5564276070 F1 witness: a handoff that reaches the shell is
      // refused (nothing is terminated), so only the durable intent survives.
      return Promise.resolve({status: 'refused', detail: 'mock refusal'});
    }
    if (SCENARIO === 'navigate_during_handoff') {
      // F4: the shell navigates the window BEFORE the invoke resolves, so
      // this page's context is disposed and its continuation never runs.
      // The never-settling promise models exactly that disposal.
      return new Promise(function () {});
    }
    if (SCENARIO === 'double_switch') {
      // Leave every call unresolved so two clicks can overlap in real time.
      return new Promise(function (resolve) { pendingInvoke.push(resolve); });
    }
    if (SCENARIO === 'staging_renew' || SCENARIO === 'staging_renew_warning' ||
        SCENARIO === 'reopen_staging_renew' || SCENARIO === 'reopen_staging_renew_warning') {
      // PR3 review 5564276070 F2 / review 5566059819 witness: the first invoke
      // asks for staging; after the (held) stage POST completes the retry
      // succeeds. The warning scenarios report producer-side instead of
      // navigating to a destination.
      if (invokeCalls === 1) {
        return Promise.resolve({status: 'needs_stage', dest_root: '/stage/' + TARGET_STUDY});
      }
      return Promise.resolve({status: 'switched', authority: HOST});
    }
    if (SCENARIO.startsWith('switch_')) {
      if (SCENARIO === 'switch_staged_refused' && invokeCalls === 1) {
        return Promise.resolve({status: 'needs_stage', dest_root: '/stage/' + TARGET_STUDY});
      }
      return Promise.resolve({status: 'refused', detail: 'mock refusal'});
    }
    return Promise.resolve({status: 'switched', authority: HOST});
  }
};

function loadApp() {
  const code = fs.readFileSync(process.env.APP_JS_PATH, 'utf8');
  vm.runInThisContext(code, {filename: 'app.js'});
  // C2 live-root seeding (harness only): replay the equivalent of a real
  // granted `switched` outcome through the page's own hook.
  if (process.env.LIVE_ROOT) {
    const hooks = global.window.__cmrTestHooks;
    if (!hooks || typeof hooks.noteLiveRoot !== 'function') {
      console.log(JSON.stringify({error: 'no live-root hook'})); process.exit(2);
    }
    hooks.noteLiveRoot(process.env.LIVE_ROOT);
  }
}

// SAMEROOT-014: count test-hook live-root assignments separately from the
// production path (which calls the closure noteLiveRoot directly, never via
// the hook object). A no-hook bootstrap witness must show hook_calls == 0
// while still learning the live root.
let hookCalls = 0;
(function () {
  const hooks = global.window.__cmrTestHooks;
  if (hooks && typeof hooks.noteLiveRoot === 'function' && !process.env.LIVE_ROOT) {
    const orig = hooks.noteLiveRoot;
    hooks.noteLiveRoot = function (r) { hookCalls += 1; return orig(r); };
  } else if (process.env.LIVE_ROOT) {
    hookCalls += 1;  // the seeded assignment above went through the hook
  }
}());

loadApp();

// PR3 review r4311831218 witness: the "current case" marker is reconciled state,
// so the harness must be able to observe it repeatedly (and observe a marker
// being REMOVED). Read the LIVE list children, not the append-only element log.
function recentRowElements() {
  const list = el('recent-list');
  return (list.children || []).filter(function (c) {
    return typeof c.className === 'string' && c.className.split(' ').includes('recent-item');
  });
}
function recentMarkerSnapshot() {
  return recentRowElements().map(function (row) {
    const info = (row.children || []).find(function (k) {
      return typeof k.className === 'string' && k.className.split(' ').includes('recent-info');
    });
    // PR3 review r4313043501: the marker must be VISIBLE, so the snapshot also
    // reads the sighted-user label (text + hidden state), not only aria/class.
    const label = (row.children || []).find(function (k) {
      return typeof k.className === 'string'
        && k.className.split(' ').includes('recent-current-label');
    });
    const text = info ? String(info.textContent) : '';
    return {
      study: text.split(' (opened:')[0] || null,
      text: text,
      aria_current: row.getAttribute('aria-current'),
      active_class: typeof row.className === 'string'
        && row.className.split(' ').includes('active'),
      current_label_text: label ? String(label.textContent) : null,
      current_label_hidden: label ? label.hidden === true : null
    };
  });
}

function report(extra) {
  let cookieKeys = null;
  let cookieHasPaths = false;
  try {
    if (jar !== null) {
      const decoded = JSON.parse(decodeURIComponent(jar));
      cookieKeys = Object.keys(decoded).sort();
      const flat = JSON.stringify(decoded);
      cookieHasPaths = /scanSourcePath|stageBaseDir|destRoot|src\/|stage\/|:\/\//.test(flat);
    }
  } catch (_) {}
  // cookie_jar_shapes: every distinct jar content observed during the run
  // (decoded key sets), so tests can prove no queued hint ever carried paths.
  const hooks = global.window && global.window.__cmrTestHooks;
  const liveRoot = (hooks && typeof hooks.currentLiveRoot === 'function')
    ? hooks.currentLiveRoot()
    : null;
  console.log(JSON.stringify(Object.assign({
    posts: posts.length,
    post_studies: posts.map(function (p) { return p.entry ? p.entry.studyUid : null; }),
    post_open_ids: posts.map(function (p) { return p.open_id || null; }),
    pending_posts: pendingPosts.length,
    pending_post_ids: pendingPosts.map(function (p) { return p.id || null; }),
    pending_slot_present: pendingSlot !== null,
    pending_gets: pendingGets,
    pending_clears: pendingClears.slice(),
    handoff_events: handoffEvents.slice(),
    pending_read_aborted: pendingSlotAborted,
    pending_body_resolved: pendingBodyResolved,
    hook_calls: hookCalls,
    cookie_present: jar !== null,
    cookie_keys: cookieKeys,
    cookie_has_paths: cookieHasPaths,
    cookie_jar_shapes: jarShapes.slice(),
    cookie_names: Object.keys(cookies).sort(),
    notice_cookie_present: Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE),
    notice_cookie_shapes: noticeShapes.slice(),
    bound_root_requests: boundRootRequests,
    invoke_calls: invokeCalls,
    invoke_times: invokeTimes.slice(),
    invoke_notices: invokeNotices.slice(),
    message_text: message.textContent,
    message_log: messageLog.slice(),
    live_root: liveRoot,
    body_aborted: boundRootBodyAborted,
    probe_requests: probeRequests,
    probe_aborted: probeAborted
  }, extra || {})));
}

if (SCENARIO === 'delayed_a_startup' || SCENARIO === 'cookie_after_init') {
  // The intent appears only AFTER this page finished initialising, then the
  // previously pending startup GET is released.
  writeIntent(PRODUCER, TARGET_STUDY);
  releaseRecentGet();
  setTimeout(function () { report({phase: 'after_release'}); }, 200);
} else if (SCENARIO === 'double_switch') {
  // Round-3 R3-1: click the real Switch button twice while the first IPC is
  // still outstanding, refuse it, then click again to prove the slot was
  // released rather than left wedged.
  // QUEUE-011: the shell invoke is chained on the awaited queue POST, so the
  // first observation happens after async work flushes (setTimeout macrotask
  // runs only once microtasks drain). The second click is still refused
  // synchronously — beginHandoff fails while the first handoff is claimed.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const btn = findButton('switch-study');
  if (!btn) { console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2); }
  btn.click();
  btn.click();
  setTimeout(function () {
  const invokeAfterTwoClicks = invokeCalls;
  const cookieAfterTwoClicks = jar !== null;
  const queueAfterTwoClicks = pendingPosts.length;
  setTimeout(function () {
    if (pendingInvoke[0]) pendingInvoke[0]({status: 'refused', detail: 'first refused'});
    setTimeout(function () {
      const cookieAfterFirstRefusal = jar !== null;
      btn.click();                       // must be allowed again now
      setTimeout(function () {
        if (pendingInvoke[1]) pendingInvoke[1]({status: 'refused', detail: 'second refused'});
        setTimeout(function () {
          report({
            phase: 'done',
            invoke_after_two_clicks: invokeAfterTwoClicks,
            cookie_after_two_clicks: cookieAfterTwoClicks,
            queue_after_two_clicks: queueAfterTwoClicks,
            cookie_after_first_refusal: cookieAfterFirstRefusal
          });
        }, 150);
      }, 80);
    }, 150);
  }, 80);
  }, 50);
} else if (SCENARIO.startsWith('switch_')) {
  // Drive the real Switch Study affordance end to end.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const button = findButton('switch-study');
  if (!button) { console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2); }
  button.click();
  setTimeout(function () { report({phase: 'after_click'}); }, 300);
} else if (SCENARIO === 'reload_after_consume') {
  // Consumed on the first load; load the page a second time (reload) and
  // prove the intent is not recorded again.
  setTimeout(function () {
    const first = posts.length;
    loadApp();
    setTimeout(function () { report({phase: 'after_reload', first_load_posts: first}); }, 250);
  }, 250);
} else if (SCENARIO === 'superseded_startup_recent') {
  // PR3 review r4310080007 witness: the startup recent GET announces old
  // snapshot headers but its body is HELD. loadState (via the auto-selected
  // series) confirms the pending open through the real flow, records it, and
  // the post-save refresh renders the committed case. Releasing the stale
  // startup body afterwards must neither replace the list nor reset the
  // digest to the stale value: the digest is observed through the
  // data-plane (the next record's base_digest), which works identically on
  // pre-fix and post-fix app.js.
  function recentRowCount() {
    // Count CURRENT list children (the global allElements log also retains
    // detached nodes, so it would not observe a replaceChildren() clobber).
    const list = el('recent-list');
    return (list.children || []).filter(function (c) {
      return typeof c.className === 'string'
        && c.className.split(' ').includes('recent-item');
    }).length;
  }
  function latestBaseDigest() {
    return posts.length ? posts[posts.length - 1].base_digest : null;
  }
  let waitedMs = 0;
  (function pollForCommittedCase() {
    if (posts.length >= 1 && recentRowCount() >= 1) {
      const rowsBefore = recentRowCount();
      const postsBefore = posts.length;
      const getsBefore = recentGetRequests;
      releaseStartupRecentBody();  // the OLD startup body arrives now
      setTimeout(function () {
        const rowsAfter = recentRowCount();
        // Data-plane probe: the next record must carry the CURRENT digest as
        // base_digest. Pre-fix the stale startup body has already reset it.
        window.CMR_WORKSTATION.recordRecentCase({
          studyUid: TARGET_STUDY, seriesUid: null, scanSourcePath: '/src',
          stageBaseDir: '/stage', destRoot: '/stage/' + TARGET_STUDY,
          openedSeq: 99, openCount: 1
        }, 'probe-digest').catch(function () {});
        setTimeout(function () {
          report({
            phase: 'after_stale_release',
            posts_before_release: postsBefore,
            recent_gets_before_release: getsBefore,
            rows_before_release: rowsBefore,
            rows_after_release: rowsAfter,
            digest_after_release: latestBaseDigest(),
            message_text: message.textContent
          });
        }, 300);
      }, 400);
      return;
    }
    waitedMs += 100;
    if (waitedMs > 6000) {
      report({
        phase: 'timeout_waiting_for_committed_case',
        posts: posts.length,
        rows: recentRowCount(),
        recent_gets: recentGetRequests
      });
      return;
    }
    setTimeout(pollForCommittedCase, 100);
  })();
} else if (SCENARIO === 'reopen_same_root') {
  // Same-root no-op (review P2, S10 `unchanged` leg): the recent list shows
  // the ACTIVE (already-open) case; clicking its Reopen must NOT handoff,
  // queue, or invoke — the shell would kill/respawn the live backend.
  // Seed the list with the bound study, then click its Reopen button.
  pendingSlot = {id: 'poUnused', exp: Date.now() + 60000, entry: {
    studyUid: BOUND_STUDY, seriesUid: null, scanSourcePath: '/src',
    stageBaseDir: '/stage', destRoot: '', openedSeq: 0, openCount: 1
  }};
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click();
    setTimeout(function () { report({phase: 'after_click', buttons: buttons.length}); }, 250);
  }, 400);
} else if (SCENARIO === 'reopen_identity_pending') {
  // PR3 review r4310770076 lifecycle witness: the FIRST /state answer — the
  // only INDEPENDENT bound-study identity source — is held by
  // DELAY_INITIAL_STATE. Clicking Reopen on the equal server-resolved root
  // while identity is UNKNOWN must not answer "already open" and must not
  // invoke/queue yet: the decision waits for the identity, then applies the
  // matching (anchored no-op) / mismatching (switch) rule.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click();   // clicked while the bound identity is still pending
    setTimeout(function () {
      const pendingWhileUnknown = {
        message: message.textContent,
        invoke_calls: invokeCalls,
        pending_posts: pendingPosts.length,
        cookie_present: jar !== null
      };
      releaseInitialStateGate();   // the delayed /state now resolves
      setTimeout(function () {
        report({
          phase: 'after_identity_release',
          pending_while_unknown: pendingWhileUnknown
        });
      }, 400);
    }, 150);
  }, 400);
} else if (SCENARIO === 'delayed_probe') {
  // R11 F2 witness: the entry probe is delayed while bound-root has already
  // answered. An early click with alternative path spelling (e.g. /stage/./)
  // must wait for the probe, rather than deciding from client derivation
  // and mistakenly invoking switch_study.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click();  // clicked while probe is pending
    setTimeout(function () {
      const invokeWhilePending = invokeCalls;
      const pendingWhilePending = pendingPosts.length;
      const cookieWhilePending = jar !== null;
      if (typeof releaseProbe === 'function') releaseProbe();
      setTimeout(function () {
        report({
          phase: 'after_probe_release',
          buttons: buttons.length,
          invoke_while_pending: invokeWhilePending,
          pending_while_pending: pendingWhilePending,
          cookie_while_pending: cookieWhilePending,
          invoke_calls: invokeCalls,
          pending_posts: pendingPosts.length,
          cookie_present: jar !== null,
          message_text: message.textContent
        });
      }, 250);
    }, 200);
  }, 300);
} else if (SCENARIO === 'hang_probe') {
  // R11 F2 witness: entry probe hangs past PROBE_TIMEOUT_MS.
  // Reopen must time out safely, exit visibly with an actionable retry notice,
  // produce 0 invokes, 0 pending writes, no cookie, and reset for retry.
  window.__CMR_PROBE_TIMEOUT_MS = 200;
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click();
    setTimeout(function () {
      const firstMessage = message.textContent;
      const firstInvokeCalls = invokeCalls;
      const firstPendingPosts = pendingPosts.length;
      const firstCookiePresent = jar !== null;

      // Actionable retry (click again)
      buttons[0].click();
      setTimeout(function () {
        report({
          phase: 'after_hang_probe',
          first_message: firstMessage,
          first_invoke_calls: firstInvokeCalls,
          first_pending_posts: firstPendingPosts,
          first_cookie_present: firstCookiePresent,
          second_invoke_calls: invokeCalls,
          second_pending_posts: pendingPosts.length,
          cookie_present: jar !== null,
          final_message: message.textContent,
          probe_aborted: probeAborted
        });
      }, 250);
    }, 250);
  }, 200);
} else if (SCENARIO === 'delayed_bound_root') {
  // Delayed bound-root (review fix witness): the recent list settles first
  // and the Reopen button is clickable while /api/library/bound-root is
  // STILL pending. The early click must not decide from a null live root:
  // no handoff, no queue write, no shell invoke while pending. Once the
  // bound-root answer releases, the SAME click's decision settles as the
  // anchored same-root no-op ("already open") instead of a kill/respawn.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click();  // early: bound-root initialization still pending
    setTimeout(function () {
      const invokeWhilePending = invokeCalls;
      const pendingWhilePending = pendingPosts.length;
      releaseBoundRoot();  // the bound-root answer arrives now
      setTimeout(function () {
        report({
          phase: 'after_release',
          buttons: buttons.length,
          invoke_while_pending: invokeWhilePending,
          pending_while_pending: pendingWhilePending
        });
      }, 250);
    }, 200);
  }, 400);
} else if (SCENARIO === 'hang_bound_root') {
  // Finding 2 witness: /api/library/bound-root never completes.
  // The Reopen decision must NOT hang indefinitely: it times out, surfaces
  // an actionable refusal, leaves the backend untouched, and resets for retry.
  window.__CMR_BOUND_ROOT_TIMEOUT_MS = 250;
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[0].click(); // First click: times out after 250ms
    setTimeout(function () {
      const firstMessage = message.textContent;
      const firstInvokeCalls = invokeCalls;
      const firstPendingPosts = pendingPosts.length;
      // Actionable retry: second click attempts again
      buttons[0].click();
      setTimeout(function () {
        report({
          phase: 'after_hang_timeout',
          first_message: firstMessage,
          first_invoke_calls: firstInvokeCalls,
          first_pending_posts: firstPendingPosts,
          second_invoke_calls: invokeCalls,
          second_pending_posts: pendingPosts.length,
          cookie_present: jar !== null,
          final_message: message.textContent
        });
      }, 350);
    }, 350);
  }, 400);
} else if (SCENARIO === 'hang_bound_root_body') {
  // Supplementary review P2 witness: /api/library/bound-root returns 200
  // headers immediately, but the body stream is held.
  // The Reopen decision must NOT hang indefinitely on await res.json():
  // 1) First attempt times out after BOUND_ROOT_TIMEOUT_MS, aborts the stream,
  //    exits visibly with actionable retry notice, and leaves backend untouched.
  // 2) Second attempt (actionable retry) successfully verifies active workspace.
  // 3) Late resolution of attempt 1's stalled body is isolated and CANNOT commit
  //    live root.
  window.__CMR_BOUND_ROOT_TIMEOUT_MS = 200;
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    // Click button during request 1's hanging body
    buttons[0].click();
    setTimeout(function () {
      const firstMessage = message.textContent;
      const firstInvokeCalls = invokeCalls;
      const firstPendingPosts = pendingPosts.length;
      const firstCookiePresent = jar !== null;
      const firstRequests = boundRootRequests;
      const wasAborted = boundRootBodyAborted;

      // Actionable retry: second click initiates request 2 which settles with BOUND_ROOT
      buttons[0].click();
      setTimeout(function () {
        const secondMessage = message.textContent;
        const secondInvokeCalls = invokeCalls;
        const secondPendingPosts = pendingPosts.length;
        const hooks = global.window && global.window.__cmrTestHooks;
        const rootAfterRetry = (hooks && typeof hooks.currentLiveRoot === 'function')
          ? hooks.currentLiveRoot()
          : null;

        // Late body completion on attempt 1: resolve the stalled first body
        // with a conflicting bogus root to prove isolation.
        if (typeof boundRootFirstBodyResolve === 'function') {
          try {
            boundRootFirstBodyResolve({root: '/late/bogus/root'});
          } catch (_) {}
        }

        setTimeout(function () {
          const finalLiveRoot = (hooks && typeof hooks.currentLiveRoot === 'function')
            ? hooks.currentLiveRoot()
            : null;
          report({
            phase: 'after_hang_body_and_retry',
            first_message: firstMessage,
            first_invoke_calls: firstInvokeCalls,
            first_pending_posts: firstPendingPosts,
            first_cookie_present: firstCookiePresent,
            first_requests: firstRequests,
            body_aborted: wasAborted,
            second_message: secondMessage,
            second_invoke_calls: secondInvokeCalls,
            second_pending_posts: secondPendingPosts,
            root_after_retry: rootAfterRetry,
            final_live_root: finalLiveRoot,
            late_body_committed: finalLiveRoot === '/late/bogus/root',
            cookie_present: jar !== null
          });
        }, 100);
      }, 150);
    }, 250);
  }, 50);
} else if (SCENARIO === 'delayed_queue') {
  // QUEUE-011 killed-A witness: the queue POST is held unresolved (backend A
  // not yet durable). The shell invoke must NOT fire while the queue is held;
  // it fires only after the queue is released as persisted.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const qbutton = findButton('switch-study');
  if (!qbutton) {
    console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2);
  }
  qbutton.click();
  setTimeout(function () {
    const invokeWhileQueueHeld = invokeCalls;
    const queueDispatched = pendingPosts.length;
    releaseQueue();  // backend A acknowledges persistence
    setTimeout(function () {
      report({
        phase: 'done',
        invoke_while_queue_held: invokeWhileQueueHeld,
        queue_dispatched: queueDispatched
      });
    }, 150);
  }, 100);
} else if (SCENARIO === 'queue_hang') {
  // QUEUE-HANG (R5): the queue POST is held; the page's bounded queue
  // timeout (~5s, or QUEUE_TIMEOUT_MS) must fire the shell invoke anyway
  // (no wedge) with the explicit "may not be saved" notice — and the LATE
  // ok:true released after the deadline must be ignored.
  const queueTimeoutMs = (typeof window !== 'undefined' && window.__CMR_QUEUE_TIMEOUT_MS) || 5000;
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const hbutton = findButton('switch-study');
  if (!hbutton) {
    console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2);
  }
  const clickAt = Date.now();
  hbutton.click();
  setTimeout(function () {
    releaseQueue();  // late ok:true — the page must already have timed out
    const releasedAt = Date.now();
    setTimeout(function () {
      report({
        phase: 'done',
        queue_dispatched: pendingPosts.length,
        invoke_after_hang: invokeCalls,
        click_at: clickAt,
        released_at: releasedAt,
        abort_controller_present: typeof AbortController !== 'undefined'
      });
    }, 200);
  }, queueTimeoutMs + 1000);
} else if (SCENARIO === 'queue_error') {
  // QUEUE-ERROR (R5-ratified): the queue POST fails fast (QUEUE_FAIL 500).
  // The switch must still invoke exactly once (unrecorded, never broken,
  // never fabricated) WITH the explicit "may not be saved" notice.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const ebutton = findButton('switch-study');
  if (!ebutton) {
    console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2);
  }
  ebutton.click();
  setTimeout(function () {
    report({
      phase: 'done',
      queue_dispatched: pendingPosts.length,
      invoke_after_error: invokeCalls
    });
  }, 800);
} else if (SCENARIO === 'navigate_during_handoff') {
  // F4: the queue POST fails; the page arms the cross-navigation notice and
  // invokes the shell; the shell navigates away, so the outgoing page's
  // continuation never runs (the invoke mock never settles). A fresh page
  // (B) must still show the "may not be saved" warning, consumed once from
  // the notice cookie — never from the disposed page's message log.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const nbutton = findButton('switch-study');
  if (!nbutton) {
    console.log(JSON.stringify({error: 'switch button not found'}));
    process.exit(2);
  }
  nbutton.click();
  setTimeout(function () {
    const queueDispatched = pendingPosts.length;
    const invokeBeforeNavigation = invokeCalls;
    const logBeforeReload = messageLog.length;
    const noticeBeforeReload = Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE);
    loadApp();  // B: fresh closure; A's continuation never ran
    setTimeout(function () {
      report({
        phase: 'after_navigation',
        queue_dispatched: queueDispatched,
        invoke_before_navigation: invokeBeforeNavigation,
        log_before_reload: logBeforeReload,
        notice_before_reload: noticeBeforeReload,
        notice_shown_after_reload: messageLog.slice(logBeforeReload).some(function (m) {
          return m.indexOf('may not be saved') !== -1;
        }),
        notice_cleared_after_reload: !Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE)
      });
    }, 300);
  }, 300);
} else if (SCENARIO === 'notice_with_stalled_recent_get') {
  // PR3 review r4310770084 witness: the cross-navigation queue-error warning
  // must be delivered INDEPENDENTLY of the recent-list load. The notice cookie
  // is planted exactly as the surviving destination of a failed queue write
  // would find it, then a fresh page is evaluated while the startup recent GET
  // stalls (headers never arrive, or the body never settles). Pre-fix:
  // consumeRecentNotice was chained on loadRecentCases, so the stalled GET
  // withheld the warning forever — and the eval-time snapshot had already
  // cleared the cookie, so a reload could not recover it either.
  writeCookieRaw(NOTICE_COOKIE, encodeURIComponent(JSON.stringify({
    v: 1, exp: Date.now() + 60000, code: 'queue-error'
  })), 60);
  recentGetRequests = 0;   // the fresh page's startup GET is the stalled one
  loadApp();
  setTimeout(function () {
    const shownFirstPage = messageLog.some(function (m) {
      return m.indexOf('may not be saved') !== -1;
    });
    const cookieAfterDelivery = Object.prototype.hasOwnProperty.call(cookies, NOTICE_COOKIE);
    // The warning must also survive on the durable banner surface: the list
    // load's synchronous banner reset runs BEFORE this delivery, so a
    // delivery that is merely chained before the load would be wiped here.
    const bannerVisible = el('recent-banner').style.display === 'block';
    const logBeforeReload = messageLog.length;
    loadApp();   // reload: a consumed notice must not be shown twice
    setTimeout(function () {
      report({
        phase: 'after_stalled_get_and_reload',
        notice_shown_first_page: shownFirstPage,
        notice_banner_visible: bannerVisible,
        notice_cookie_after_delivery: cookieAfterDelivery,
        notice_shown_after_reload: messageLog.slice(logBeforeReload).some(function (m) {
          return m.indexOf('may not be saved') !== -1;
        })
      });
    }, 300);
  }, 600);
} else if (SCENARIO === 'record_failure_keeps_slot') {
  // PENDING-SUCCESS-015: the seeded intent is consumed by the bound
  // destination but the record POST fails (RECORD_FAIL) — the pending slot
  // must be KEPT (never cleared as successful) and no second page may
  // observe a confirmed record.
  setTimeout(function () {
    report({phase: 'after_consume'});
  }, 400);
} else if (SCENARIO === 'pending_read_fail_then_retry') {
  // F3: the first pending read fails transiently (503). The page must retain
  // the hint, retry the SAME idempotent open id under its bounded budget, and
  // record exactly once; cookie and slot are cleared only AFTER the confirmed
  // record (handoff_events order).
  setTimeout(function () {
    report({phase: 'after_retry'});
  }, 600);
} else if (SCENARIO === 'pending_read_fail_reload') {
  // F3: the read fails while the bounded retry is still pending (long retry
  // delay). The retained hint must let a RELOAD confirm and record exactly
  // once — the durable pending slot is never stranded by the transient read
  // failure.
  setTimeout(function () {
    const beforeReload = {
      posts: posts.length,
      cookie_present: jar !== null,
      slot_present: pendingSlot !== null
    };
    loadApp();
    setTimeout(function () {
      report({phase: 'after_reload', before_reload: beforeReload});
    }, 400);
  }, 300);
} else if (SCENARIO === 'pending_read_fail_exhausted') {
  // F3: every read fails; the bounded retry budget exhausts on this page.
  // The page must surface an actionable failure, keep the hint + slot
  // (retryable by a later reload), never clear early and never fabricate.
  setTimeout(function () {
    report({phase: 'after_exhaustion'});
  }, 500);
} else if (SCENARIO === 'pending_read_hang_body') {
  // F3: the first read returns headers but the body never settles. The full
  // fetch+body deadline must fire and abort it; the retry must record exactly
  // once; the LATE body of the timed-out attempt must be isolated (no second
  // record, no spurious clear).
  setTimeout(function () {
    const readsAfterRetry = pendingGets;
    const abortedAtTimeout = pendingSlotAborted;
    const postsBeforeLateBody = posts.length;
    const clearsBeforeLateBody = pendingClears.length;
    releasePendingReadBody();       // late settlement of the timed-out attempt
    setTimeout(function () {
      report({
        phase: 'after_late_body',
        reads_after_retry: readsAfterRetry,
        aborted_at_timeout: abortedAtTimeout,
        posts_before_late_body: postsBeforeLateBody,
        clears_before_late_body: clearsBeforeLateBody
      });
    }, 300);
  }, 600);
} else if (SCENARIO === 'pending_read_retry_singleflight') {
  // F3: the first read fails; the retry is scheduled after the configured
  // delay. Repeated concurrent evaluations while the retry timer is pending
  // must NOT start extra reads (single-flight); the eventual retry records
  // exactly once.
  setTimeout(function () {
    const hooks = global.window.__cmrTestHooks;
    if (!hooks || typeof hooks.maybeConsumePendingOpen !== 'function') {
      console.log(JSON.stringify({error: 'no pending-open test hook'}));
      process.exit(2);
    }
    for (let i = 0; i < 5; i += 1) hooks.maybeConsumePendingOpen();
    setTimeout(function () {
      // Capture while the retry timer is still pending (delay 150ms): the
      // re-entrant evaluations must not have added a read.
      const readsWhileRetryPending = pendingGets;
      setTimeout(function () {
        report({
          phase: 'after_retry',
          reads_while_retry_pending: readsWhileRetryPending
        });
      }, 450);
    }, 20);
  }, 80);
} else if (SCENARIO === 'pending_series_change_race') {
  // PR3 review r4311831210 (app.js:1501) lifecycle witness: the pending-slot
  // read is held; a NORMAL series change then clears `state` (chooseSeries)
  // while its /state answers are held too; the read settles while the page
  // state is unknown. Null state is NOT proof of a different destination:
  // pre-fix this returned "standby", set pendingConfirmExhaustedId, and the
  // successful open stayed unrecorded with no notice. Post-fix the unknown
  // state RETRIES under the bounded budget (hint + durable slot retained) and
  // records exactly once when the identity lands again.
  setTimeout(function () {
    const seriesButton = allElements.find(function (e) {
      return e.tagName === 'BUTTON' && typeof e.textContent === 'string'
        && e.textContent.indexOf(EXTRA_SERIES_UID) === 0;
    });
    if (!seriesButton) {
      console.log(JSON.stringify({error: 'series button not found'}));
      process.exit(2);
    }
    stateHoldArmed = true;      // hold the /state answers (state stays null)
    seriesButton.click();       // chooseSeries(other): state = null
    setTimeout(function () {
      releasePendingRead();     // the held read settles while state === null
      setTimeout(function () {
        const readsAfterRelease = pendingGets;
        const postsAfterRelease = posts.length;
        const cookieAfterRelease = jar !== null;
        const slotAfterRelease = pendingSlot !== null;
        releaseHeldStateNow();  // the identity lands; the retry must record
        setTimeout(function () {
          report({
            phase: 'after_state_release',
            reads_after_release: readsAfterRelease,
            posts_after_release: postsAfterRelease,
            cookie_after_release: cookieAfterRelease,
            slot_after_release: slotAfterRelease,
            state_holds: stateHolds
          });
        }, 700);
      }, 250);
    }, 250);
  }, 500);
} else if (SCENARIO === 'marker_reconcile_delayed_state') {
  // PR3 review r4311831218 witness (a): the recent-list response renders BEFORE
  // the initial /state answer. Pre-fix the one-time UID-only marker snapshot ran
  // with state === null, so no row was ever marked and loadState never
  // reconciled the markers afterwards — the ordinary relaunch could keep an
  // unmarked current case indefinitely.
  setTimeout(function () {
    const before = recentMarkerSnapshot();
    releaseInitialStateGate();   // the delayed bound identity now arrives
    setTimeout(function () {
      const after = recentMarkerSnapshot();
      // PR3 review r4313043501 relocation leg: the bound live root moves to a
      // DIFFERENT case root, so the marker AND its visible label must be
      // removed (not merely left unset).
      let relocated = null;
      if (process.env.LIVE_ROOT_AFTER) {
        window.__cmrTestHooks.noteLiveRoot(process.env.LIVE_ROOT_AFTER);
        relocated = recentMarkerSnapshot();
      }
      report({
        phase: 'after_state',
        marker_before_state: before,
        marker_after_state: after,
        marker_after_relocation: relocated
      });
    }, 500);
  }, 350);
} else if (SCENARIO === 'marker_reconcile_same_uid_root') {
  // PR3 review r4311831218 witness (b): two rows carry the SAME study UID but
  // different server-resolved destinations. Only the row whose destination IS
  // the live bound root is the current case; pre-fix every same-UID row got
  // aria-current, contradicting the same-root rule the Reopen handler applies.
  // The recent GET is held until the state identity is known so the pre-fix
  // one-time snapshot is actually taken with `state` present (non-vacuous).
  setTimeout(function () {
    releaseRecentGet();
    setTimeout(function () {
      report({phase: 'after_render', marker: recentMarkerSnapshot()});
    }, 400);
  }, 300);
} else if (SCENARIO === 'root_match') {
  // PR3 review #3 positive control: a matching server-reported bound root is
  // the legitimate destination — record exactly once, clear the handoff.
  setTimeout(function () {
    report({phase: 'after_match'});
  }, 500);
} else if (SCENARIO === 'root_mismatch_then_destination') {
  // PR3 review #3 (root binding): a same-UID consumer whose SERVER-REPORTED
  // bound root differs from the pending entry's derived destination must
  // record NOTHING and must not consume the shared handoff. The real
  // destination then reloads with the matching root and records exactly once.
  setTimeout(function () {
    const wrongConsumerPosts = posts.length;
    const wrongConsumerSlot = pendingSlot !== null;
    const wrongConsumerCookie = jar !== null;
    const wrongConsumerClears = pendingClears.length;
    currentBoundRoot = BOUND_ROOT_AFTER_RELOAD;   // the real destination's root
    loadApp();                                    // the actual destination page
    setTimeout(function () {
      report({
        phase: 'after_destination',
        wrong_consumer_posts: wrongConsumerPosts,
        wrong_consumer_slot_present: wrongConsumerSlot,
        wrong_consumer_cookie_present: wrongConsumerCookie,
        wrong_consumer_clears: wrongConsumerClears
      });
    }, 400);
  }, 500);
} else if (SCENARIO === 'consume_then_reopen') {
  // SAMEROOT-014 no-hook bootstrap: the fresh destination page consumes the
  // seeded intent (bound study matches), learns its live root from the
  // server-validated entry — WITHOUT any test-hook seeding — then clicking
  // Reopen on the same entry must no-op without a shell invoke. The LAST
  // button is the live one: the consume record triggers a list re-render,
  // and the shim (unlike a real DOM) retains the orphaned pre-render
  // button element in its registry.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'})); process.exit(2);
    }
    buttons[buttons.length - 1].click();
    setTimeout(function () { report({phase: 'after_reopen', buttons: buttons.length}); }, 250);
  }, 400);
} else if (SCENARIO === 'different_uid_then_destination') {
  // PR3 review #4 (app.js:1219): a different-UID window that captured the
  // shared hint must STAND BY — record nothing, clear nothing, keep the hint
  // and the durable slot — so the real destination can still consume them.
  // The run then reloads as the real destination (matching UID + resolved
  // root) and proves it records exactly once and clears exactly once.
  setTimeout(function () {
    const wrongPosts = posts.length;
    const wrongClears = pendingClears.length;
    const wrongCookie = jar !== null;
    const wrongSlot = pendingSlot !== null;
    currentBoundStudy = BOUND_STUDY_AFTER_RELOAD || TARGET_STUDY;
    currentBoundRoot = BOUND_ROOT_AFTER_RELOAD;
    loadApp();                                    // the actual destination page
    setTimeout(function () {
      report({
        phase: 'after_destination',
        wrong_consumer_posts: wrongPosts,
        wrong_consumer_clears: wrongClears,
        wrong_consumer_cookie_present: wrongCookie,
        wrong_consumer_slot_present: wrongSlot
      });
    }, 500);
  }, 400);
} else if (SCENARIO === 'pending_destination_probe') {
  // PR3 review #4 (app.js:1230): the pending confirmation must AWAIT the
  // server-resolved destination before deciding. A delayed probe must produce
  // no record/clear while it is still pending; the released answer then
  // settles the decision (confirm when it matches the bound root, stand by
  // otherwise). Used by the delayed, failure and hang probe witnesses.
  setTimeout(function () {
    const postsWhileProbePending = posts.length;
    const clearsWhileProbePending = pendingClears.length;
    const cookieWhileProbePending = jar !== null;
    if (typeof releaseProbe === 'function') releaseProbe();
    setTimeout(function () {
      report({
        phase: 'after_probe_release',
        posts_while_probe_pending: postsWhileProbePending,
        clears_while_probe_pending: clearsWhileProbePending,
        cookie_while_probe_pending: cookieWhileProbePending
      });
    }, 500);
  }, 300);
} else if (SCENARIO === 'bound_root_pending_confirm') {
  // PR3 review 5564032012 F1 (app.js:1316): a failed/timed-out bound-root
  // lookup is NOT a proven non-destination. The confirmation retries under its
  // bounded budget with a FRESH lookup (failed memoization reset); a transient
  // failure therefore still records exactly once after recovery, while an
  // exhausted lookup ends on the actionable "may not be saved" notice with the
  // hint and durable slot retained.
  setTimeout(function () { report({phase: 'after_settle'}); }, 900);
} else if (SCENARIO === 'late_record_retry') {
  // PR3 review 5564032012 F2: the first record ack is held past the coherent
  // deadline (commit UNKNOWN -> retry with the SAME open id); the retry
  // commits exactly once. Releasing the timed-out ack LATE must add no second
  // record, no clear and no digest change.
  setTimeout(function () {
    const beforeLate = {
      posts: posts.length,
      clears: pendingClears.length,
      cookie_present: jar !== null
    };
    releaseRecord();          // late ack of the timed-out attempt
    setTimeout(function () {
      report({phase: 'after_late_ack', before_late: beforeLate});
    }, 300);
  }, 700);
} else if (SCENARIO === 'incoming_confirm_blocks_handoff') {
  // PR3 review 5564276070 F1: while B's incoming intent is being confirmed
  // (read held, or retry backoff), a Switch request for C must be refused by the
  // page's own single-flight guard — zero invokes, zero queue writes, the shared
  // hint (same id) and the durable slot untouched — so B can still record with
  // its ORIGINAL open id afterwards.
  setTimeout(function () {
    el('switch-source').value = '/src-c';
    el('switch-stage').value = '/stage-c';
    el('switch-study').value = '1.2.826.0.1.3680043.10.99996.3';
    const btn = findButton('switch-study');
    if (!btn) { console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2); }
    btn.click();
    setTimeout(function () {
      const invokeAfterC = invokeCalls;
      const queueAfterC = pendingPosts.length;
      const cookieAfterC = jar !== null;
      const cookieIdAfterC = jar ? JSON.parse(decodeURIComponent(jar)).id : null;
      const slotAfterC = pendingSlot !== null;
      releasePendingRead();          // the held read now settles (hang leg)
      setTimeout(function () {
        report({
          phase: 'after_release',
          invoke_after_c: invokeAfterC,
          queue_after_c: queueAfterC,
          cookie_after_c: cookieAfterC,
          cookie_id_after_c: cookieIdAfterC,
          slot_after_c: slotAfterC
        });
      }, 500);
    }, 200);
  }, 300);
} else if (SCENARIO === 'staging_renew') {
  // PR3 review 5564276070 F2: staging is held past the ORIGINAL 120s deadline
  // (controlled clock), then completes. The correlated intent must be renewed
  // (same id, fresh exp) before the final switch invoke, so the destination page
  // still confirms and records exactly once. With FOREIGN_INTENT the shared hint
  // is replaced by another window's newer id while staging runs: the renewal must
  // NOT overwrite it (and must not re-queue the page's own intent).
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage/./';       // raw dot-segment stage spelling
  el('switch-study').value = TARGET_STUDY;
  const renewBtn = findButton('switch-study');
  if (!renewBtn) {
    console.log(JSON.stringify({error: 'switch button not found'}));
    process.exit(2);
  }
  renewBtn.click();
  setTimeout(function () {
    const hintBefore = jar ? JSON.parse(decodeURIComponent(jar)) : null;
    clockOffset += 200000;                      // past the original TTL
    if (process.env.FOREIGN_INTENT === '1') {
      writeIntent('127.0.0.1:9999', TARGET_STUDY, 'poForeign');
    }
    releaseStage();                             // staging completes -> renewal -> invoke #2
    setTimeout(function () {
      const hintAfter = jar ? JSON.parse(decodeURIComponent(jar)) : null;
      const slotExp = pendingSlot ? pendingSlot.exp : null;
      const invokeAfter = invokeCalls;
      if (process.env.FOREIGN_INTENT === '1') {
        report({
          phase: 'after_foreign',
          hint_before: hintBefore,
          hint_after: hintAfter,
          slot_exp: slotExp,
          invoke_after: invokeAfter
        });
        return;
      }
      // Navigate: the destination page is B (matching study + resolved root).
      currentBoundStudy = TARGET_STUDY;
      currentBoundRoot = BOUND_ROOT_AFTER_RELOAD;
      currentHost = '127.0.0.1:3333';            // the bound destination authority
      loadApp();
      setTimeout(function () {
        report({
          phase: 'after_destination',
          hint_before: hintBefore,
          hint_after: hintAfter,
          slot_exp: slotExp,
          invoke_after: invokeAfter
        });
      }, 500);
    }, 400);
  }, 300);
} else if (SCENARIO === 'restored_hash_then_switch') {
  // Review 5566059819 P2-1 witness: the #switch-failed restored page
  // discarded its captured intent; any correlated incoming reservation must
  // be released too, so the restored page can immediately start a NEW handoff
  // (the failed switch recorded nothing).
  setTimeout(function () {
    el('switch-source').value = '/src';
    el('switch-stage').value = '/stage';
    el('switch-study').value = TARGET_STUDY;
    const btn = findButton('switch-study');
    if (!btn) { console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2); }
    btn.click();
    setTimeout(function () { report({phase: 'after_new_handoff'}); }, 300);
  }, 300);
} else if (SCENARIO === 'delayed_initial_expiry_then_switch') {
  // Review 5566059819 P2-2 witness: the reservation is armed at CLAIM time,
  // so a delayed first /state (confirmation cannot start) plus the intent's
  // own expiry must release the page's handoff slot; a LATER handoff is then
  // accepted and the delayed state release never records the expired intent.
  setTimeout(function () {
    const hooks = global.window && global.window.__cmrTestHooks;
    const claimHeld = (hooks && typeof hooks.incomingReservationHeld === 'function')
      ? hooks.incomingReservationHeld() : null;
    el('switch-source').value = '/src';
    el('switch-stage').value = '/stage';
    el('switch-study').value = TARGET_STUDY;
    const btn = findButton('switch-study');
    if (btn) btn.click();
    setTimeout(function () {
      const invokeAfterClick = invokeCalls;
      const queueAfterClick = pendingPosts.length;
      releaseInitialStateGate();   // the delayed /state now resolves
      setTimeout(function () {
        report({
          phase: 'after_state_release',
          claim_held_after_expiry: claimHeld,
          invoke_after_click: invokeAfterClick,
          queue_after_click: queueAfterClick
        });
      }, 400);
    }, 300);
  }, 700);
} else if (SCENARIO === 'replaced_intent_cleanup_then_switch') {
  // Review 5566059819 P2-2 (correlated cleanup): while the claim is held the
  // shared hint is replaced by ANOTHER window's newer intent. Claim-expiry
  // cleanup may release only THIS page's reservation — it must never clear or
  // release the replacement — and the page must remain able to hand off.
  setTimeout(function () { writeIntent('127.0.0.1:9999', TARGET_STUDY, 'poForeign2'); }, 150);
  setTimeout(function () {
    const cookieIdAfterExpiry = jar ? JSON.parse(decodeURIComponent(jar)).id : null;
    const hooks = global.window && global.window.__cmrTestHooks;
    const claimHeld = (hooks && typeof hooks.incomingReservationHeld === 'function')
      ? hooks.incomingReservationHeld() : null;
    el('switch-source').value = '/src';
    el('switch-stage').value = '/stage';
    el('switch-study').value = TARGET_STUDY;
    const btn = findButton('switch-study');
    if (btn) btn.click();
    setTimeout(function () {
      report({
        phase: 'after_new_handoff',
        cookie_id_after_expiry: cookieIdAfterExpiry,
        claim_held_after_expiry: claimHeld,
        invoke_after_click: invokeCalls
      });
    }, 300);
  }, 700);
} else if (SCENARIO === 'staging_renew_warning') {
  // Review 5566059819 witness (Switch leg): renewal non-OK/timeout must arm
  // the path-free warning BEFORE the final invoke; report producer-side.
  el('switch-source').value = '/src';
  el('switch-stage').value = '/stage';
  el('switch-study').value = TARGET_STUDY;
  const btn = findButton('switch-study');
  if (!btn) { console.log(JSON.stringify({error: 'switch button not found'})); process.exit(2); }
  btn.click();
  setTimeout(function () {
    const hintBefore = jar ? JSON.parse(decodeURIComponent(jar)) : null;
    clockOffset += 200000;                      // past the original TTL
    releaseStage();
    setTimeout(function () {
      report({
        phase: 'renew_warning',
        hint_before: hintBefore,
        hint_after: jar ? JSON.parse(decodeURIComponent(jar)) : null,
        slot_exp: pendingSlot ? pendingSlot.exp : null
      });
    }, 900);
  }, 300);
} else if (SCENARIO === 'reopen_staging_renew') {
  // Review 5566059819 witness (Reopen leg, real positive Max-Age expiry):
  // staging is held past the ORIGINAL deadline; the Reopen chain must renew
  // its correlated intent through the in-memory lease and the bound
  // destination must still record exactly once with the original open id.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'}));
      process.exit(2);
    }
    buttons[0].click();
    setTimeout(function () {
      const hintBefore = jar ? JSON.parse(decodeURIComponent(jar)) : null;
      clockOffset += 200000;                    // past the original TTL
      releaseStage();
      setTimeout(function () {
        const hintAfter = jar ? JSON.parse(decodeURIComponent(jar)) : null;
        const slotExp = pendingSlot ? pendingSlot.exp : null;
        const invokeAfter = invokeCalls;
        currentBoundStudy = TARGET_STUDY;
        currentBoundRoot = BOUND_ROOT_AFTER_RELOAD;
        currentHost = '127.0.0.1:3333';
        loadApp();
        setTimeout(function () {
          report({
            phase: 'after_destination',
            hint_before: hintBefore,
            hint_after: hintAfter,
            slot_exp: slotExp,
            invoke_after: invokeAfter
          });
        }, 500);
      }, 600);
    }, 300);
  }, 400);
} else if (SCENARIO === 'reopen_staging_renew_warning') {
  // Review 5566059819 witness (Reopen leg): renewal non-OK/timeout must arm
  // the path-free warning BEFORE the final reopen invoke; producer-side report.
  setTimeout(function () {
    const buttons = allElements.filter(function (e) {
      return e.className === 'recent-reopen-button';
    });
    if (!buttons.length) {
      console.log(JSON.stringify({error: 'no reopen buttons'}));
      process.exit(2);
    }
    buttons[0].click();
    setTimeout(function () {
      const hintBefore = jar ? JSON.parse(decodeURIComponent(jar)) : null;
      clockOffset += 200000;                    // past the original TTL
      releaseStage();
      setTimeout(function () {
        report({
          phase: 'renew_warning',
          hint_before: hintBefore,
          hint_after: jar ? JSON.parse(decodeURIComponent(jar)) : null,
          slot_exp: pendingSlot ? pendingSlot.exp : null
        });
      }, 900);
    }, 300);
  }, 400);
} else if (SCENARIO === 'stale_hint_after_clock') {
  // PR3 review 5564276070 F2 (abandoned-intent bound preserved): with the clock
  // advanced past the hint's own exp and NO renewal, the destination page must
  // not consume anything (the expired hint is never even read, the slot is never
  // confirmed), so abandoned handoffs still expire.
  setTimeout(function () {
    const getsBefore = pendingGets;
    const postsBefore = posts.length;
    clockOffset += 200000;
    currentBoundStudy = TARGET_STUDY;      // the real destination page
    currentBoundRoot = BOUND_ROOT_AFTER_RELOAD;
    loadApp();
    setTimeout(function () {
      report({
        phase: 'after_stale_reload',
        gets_before_reload: getsBefore,
        posts_before_reload: postsBefore
      });
    }, 500);
  }, 300);
} else {
  setTimeout(function () { report({phase: 'settled'}); }, 300);
}
"""


def _run_handoff_scenario(
    scenario: str,
    *,
    host: str = "127.0.0.1:2222",
    producer: str = "127.0.0.1:1111",
    target_study: str = STUDY_B,
    bound_study: str | None = None,
    init_cookie: bool = True,
    page_hash: str = "",
    delay_recent_get: bool = False,
    hold_startup_recent_body: bool = False,
    stale_recent_entries: dict[str, Any] | None = None,
    recent_entries: dict[str, Any] | None = None,
    live_root: str | None = None,
    live_root_after: str | None = None,
    delay_queue: bool = False,
    record_fail: bool = False,
    hang_queue: bool = False,
    queue_fail: bool = False,
    bound_root: str | None = None,
    bound_root_after_reload: str | None = None,
    bound_study_after_reload: str | None = None,
    pending_stage_base: str | None = None,
    bound_root_fails: int = 0,
    record_ack_timeout_ms: int | None = None,
    pending_cleanup_timeout_ms: int | None = None,
    hang_record_headers: bool = False,
    hang_record_body: bool = False,
    hang_409_body: bool = False,
    record_fetch_reject: bool = False,
    record_body_reject: bool = False,
    record_409_body_reject: bool = False,
    hang_refresh_after_record: bool = False,
    hang_clear: bool = False,
    late_record: bool = False,
    invoke_refused: bool = False,
    delay_stage: bool = False,
    stage_renew_fails: int = 0,
    stage_renew_hang: bool = False,
    delay_initial_state: bool = False,
    stall_recent_get: str | None = None,
    init_intent_exp_ms: int | None = None,
    foreign_intent: bool = False,
    probe_dest_root: str | None = None,
    delay_bound_root: bool = False,
    hang_bound_root: bool = False,
    hang_bound_root_body: bool = False,
    bound_root_timeout_ms: int | None = None,
    bound_identity_timeout_ms: int | None = None,
    no_abort_controller: bool = False,
    queue_timeout_ms: int | None = None,
    delay_probe: bool = False,
    hang_probe: bool = False,
    probe_fail: bool = False,
    probe_timeout_ms: int | None = None,
    pending_read_fails: int = 0,
    pending_read_hang: bool = False,
    pending_read_hang_body: bool = False,
    pending_timeout_ms: int | None = None,
    pending_max_attempts: int | None = None,
    pending_retry_delay_ms: int | None = None,
    extra_series_uid: str | None = None,
    hold_state_for_race: bool = False,
) -> dict[str, Any]:
    """Drive the real static/app.js in Node and return its JSON report.

    Uses the repository's own vm + DOM-shim harness pattern (the one the
    switch-study tests use), extended with a document.cookie jar and a
    host-bearing window.location so the handoff guards are exercisable.
    """
    app_js = Path(__file__).parents[2] / "services" / "workstation" / "static" / "app.js"
    assert app_js.is_file()
    import json as _json

    proc = subprocess.run(
        ["node", "-"],
        input=_HANDOFF_HARNESS_JS,
        env={
            **os.environ,
            "APP_JS_PATH": str(app_js),
            "SCENARIO": scenario,
            "HOST": host,
            "PRODUCER": producer,
            "TARGET_STUDY": target_study,
            "BOUND_STUDY": bound_study if bound_study is not None else target_study,
            "INIT_COOKIE": "1" if init_cookie else "0",
            "PAGE_HASH": page_hash,
            "DELAY_RECENT_GET": "1" if delay_recent_get else "0",
            "HOLD_STARTUP_RECENT_BODY": "1" if hold_startup_recent_body else "0",
            "STALE_RECENT_ENTRIES_JSON": (
                _json.dumps(stale_recent_entries) if stale_recent_entries else ""
            ),
            "RECENT_ENTRIES_JSON": _json.dumps(recent_entries) if recent_entries else "",
            "LIVE_ROOT": live_root or "",
            "LIVE_ROOT_AFTER": live_root_after or "",
            "DELAY_QUEUE": "1" if delay_queue else "0",
            "RECORD_FAIL": "1" if record_fail else "",
            "HANG_QUEUE": "1" if hang_queue else "0",
            "QUEUE_FAIL": "1" if queue_fail else "0",
            "BOUND_ROOT": bound_root or "",
            "BOUND_ROOT_AFTER_RELOAD": bound_root_after_reload or "",
            "BOUND_STUDY_AFTER_RELOAD": bound_study_after_reload or "",
            "PENDING_STAGE_BASE": pending_stage_base or "",
            "BOUND_ROOT_FAILS": str(bound_root_fails) if bound_root_fails else "",
            "RECORD_ACK_TIMEOUT_MS": (
                str(record_ack_timeout_ms) if record_ack_timeout_ms is not None else ""
            ),
            "PENDING_CLEANUP_TIMEOUT_MS": (
                str(pending_cleanup_timeout_ms)
                if pending_cleanup_timeout_ms is not None
                else ""
            ),
            "HANG_RECORD_HEADERS": "1" if hang_record_headers else "0",
            "HANG_RECORD_BODY": "1" if hang_record_body else "0",
            "HANG_409_BODY": "1" if hang_409_body else "0",
            "RECORD_FETCH_REJECT": "1" if record_fetch_reject else "0",
            "RECORD_BODY_REJECT": "1" if record_body_reject else "0",
            "RECORD_409_BODY_REJECT": "1" if record_409_body_reject else "0",
            "HANG_REFRESH_AFTER_RECORD": "1" if hang_refresh_after_record else "0",
            "HANG_CLEAR": "1" if hang_clear else "0",
            "LATE_RECORD": "1" if late_record else "0",
            "INVOKE_REFUSED": "1" if invoke_refused else "0",
            "DELAY_STAGE": "1" if delay_stage else "0",
            "STAGE_RENEW_FAILS": str(stage_renew_fails) if stage_renew_fails else "",
            "STAGE_RENEW_HANG": "1" if stage_renew_hang else "0",
            "DELAY_INITIAL_STATE": "1" if delay_initial_state else "0",
            "STALL_RECENT_GET": stall_recent_get or "",
            "INIT_INTENT_EXP_MS": (
                str(init_intent_exp_ms) if init_intent_exp_ms is not None else ""
            ),
            "FOREIGN_INTENT": "1" if foreign_intent else "0",
            "PROBE_DEST_ROOT": probe_dest_root or "",
            "DELAY_BOUND_ROOT": "1" if delay_bound_root else "0",
            "HANG_BOUND_ROOT": "1" if hang_bound_root else "0",
            "HANG_BOUND_ROOT_BODY": "1" if hang_bound_root_body else "0",
            "BOUND_ROOT_TIMEOUT_MS": (
                str(bound_root_timeout_ms) if bound_root_timeout_ms is not None else ""
            ),
            "BOUND_IDENTITY_TIMEOUT_MS": (
                str(bound_identity_timeout_ms)
                if bound_identity_timeout_ms is not None
                else ""
            ),
            "NO_ABORT_CONTROLLER": "1" if no_abort_controller else "0",
            "QUEUE_TIMEOUT_MS": (
                str(queue_timeout_ms) if queue_timeout_ms is not None else ""
            ),
            "DELAY_PROBE": "1" if delay_probe else "0",
            "HANG_PROBE": "1" if hang_probe else "0",
            "PROBE_FAIL": "1" if probe_fail else "0",
            "PROBE_TIMEOUT_MS": (
                str(probe_timeout_ms) if probe_timeout_ms is not None else ""
            ),
            "PENDING_READ_FAILS": str(pending_read_fails) if pending_read_fails else "",
            "PENDING_READ_HANG": "1" if pending_read_hang else "0",
            "PENDING_READ_HANG_BODY": "1" if pending_read_hang_body else "0",
            "PENDING_TIMEOUT_MS": (
                str(pending_timeout_ms) if pending_timeout_ms is not None else ""
            ),
            "PENDING_MAX_ATTEMPTS": (
                str(pending_max_attempts) if pending_max_attempts is not None else ""
            ),
            "PENDING_RETRY_DELAY_MS": (
                str(pending_retry_delay_ms) if pending_retry_delay_ms is not None else ""
            ),
            # PR3 review r4311831210 witness knobs: a genuine second series to
            # select while the confirmation read is in flight, and the hold on
            # every /state answer that keeps the transient null window open.
            "EXTRA_SERIES_UID": extra_series_uid or "",
            "HOLD_STATE_FOR_RACE": "1" if hold_state_for_race else "0",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"handoff harness failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    assert lines, f"harness produced no output; stderr:\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1])


def test_handoff_delayed_startup_never_records_unswitched_target() -> None:
    """Round-2 P1 (VERIFIED_BY_EXECUTION repro): page A's startup recent GET is
    still pending when a switch to B is requested, so the intent cookie appears
    AFTER page A initialised. When A's pending GET settles, A must NOT record
    B's identity — A is not the bound destination and its switch has not
    succeeded. The pre-fix consumer read whichever cookie existed at that
    later moment and recorded it."""
    res = _run_handoff_scenario(
        "delayed_a_startup",
        host="127.0.0.1:1111",          # page A
        producer="127.0.0.1:1111",      # queued by A itself
        target_study=STUDY_B,
        bound_study=STUDY_A,            # A is still bound to A
        init_cookie=False,              # nothing eligible at A's page init
        delay_recent_get=True,
    )
    assert res["posts"] == 0, res
    assert res["post_studies"] == [], res


def test_handoff_ignores_intent_that_arrives_after_page_init() -> None:
    """Snapshot guard in isolation: an intent whose producer is a DIFFERENT
    authority and whose target matches this page's bound study still must not
    be consumed when it only appeared after page initialisation — eligibility
    is captured once, synchronously, at page init. The pre-fix consumer
    re-read the cookie later and would record it."""
    res = _run_handoff_scenario(
        "cookie_after_init",
        host="127.0.0.1:3333",          # this page
        producer="127.0.0.1:1111",      # produced elsewhere
        target_study=STUDY_B,
        bound_study=STUDY_B,            # binding WOULD match
        init_cookie=False,
        delay_recent_get=True,
    )
    assert res["posts"] == 0, res


def test_handoff_records_exactly_once_for_bound_destination() -> None:
    """Happy path: the destination page (different authority, bound to the
    target study, intent present at its init) records exactly once — and a
    subsequent reload of that page does not record again."""
    res = _run_handoff_scenario(
        "reload_after_consume",
        host="127.0.0.1:2222",          # page B
        producer="127.0.0.1:1111",      # queued by A
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        # PR3 review #3: the destination must match the SERVER-REPORTED bound
        # root too, not only the UID; the entry's derived destination is
        # /stage/<studyUid>.
        bound_root="/stage/" + STUDY_B,
    )
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["cookie_present"] is False, res
    assert res["first_load_posts"] == 1, res

def test_recent_list_ignores_superseded_startup_response() -> None:
    """PR3 review r4310080007 (fail-capable): a delayed startup recent GET
    released AFTER a successful save and its newer refresh must not publish.

    The startup GET's body is held while loadState confirms the pending open
    through the real flow, records it, and the bounded post-save refresh
    renders the committed case (digest 'd-list'). Releasing the old startup
    body then must neither replace the rendered list nor reset the digest to
    the stale value; pre-fix it emptied the list and regressed the digest, so
    the committed case looked unsaved until another refresh/reload."""
    committed_entry = {
        "studyUid": STUDY_B,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "/stage/" + STUDY_B,
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "superseded_startup_recent",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        bound_root="/stage/" + STUDY_B,
        init_cookie=True,
        recent_entries=_single_entry_document(committed_entry),
        hold_startup_recent_body=True,
        stale_recent_entries={
            "version": 1,
            "entries": [],
            "discarded": 0,
            "reason": "absent",
            "digest": "d-stale",
        },
    )
    assert res["phase"] == "after_stale_release", res
    assert res["posts_before_release"] == 1, res
    assert res["recent_gets_before_release"] >= 2, res
    assert res["rows_before_release"] == 1, res
    # Both symptoms are asserted together so the RED run records the list
    # clobber AND the digest regression in one failure: pre-fix the stale
    # release empties the rendered list and the next record carries the stale
    # base_digest 'd-stale' instead of the committed document digest.
    assert {
        "rows_after_release": res["rows_after_release"],
        "digest_after_release": res["digest_after_release"],
    } == {"rows_after_release": 1, "digest_after_release": "d-list"}, res


def test_handoff_different_uid_consumer_stands_by_and_preserves_handoff() -> None:
    """PR3 review #4 (app.js:1219): a consumer whose study UID differs from the
    intent's target is NOT the destination, but it must STAND BY rather than
    retire the shared handoff. The hint cookie is shared across loopback ports
    and the pending slot uses the same per-user store, so clearing either here
    strands the real destination (it finds no hint at its own load and the
    durable slot is never consumed, with no warning). Pre-fix this returned
    `refused`, which deleted the shared cookie."""
    res = _run_handoff_scenario(
        "dest_mismatch",
        host="127.0.0.1:4444",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,            # bound to A, intent targets B
        bound_root="/stage/" + STUDY_A,
        init_cookie=True,
    )
    assert res["posts"] == 0, res
    assert res["post_studies"] == [], res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    # Standby is decided from the UID mismatch alone: this page never even asks
    # the server to resolve a destination for an entry that is not its own.
    assert res["probe_requests"] == 0, res


def test_handoff_different_uid_consumer_then_real_destination_records_once() -> None:
    """PR3 review #4 (two-consumer lifecycle): an unrelated C window (different
    UID, other authority) evaluates the shared hint FIRST — it records and
    clears nothing, leaving the hint + durable slot intact — and the real B
    destination (matching UID and server-resolved root) then records exactly
    once and clears exactly once."""
    res = _run_handoff_scenario(
        "different_uid_then_destination",
        host="127.0.0.1:4444",               # window C (wrong study)
        producer="127.0.0.1:1111",           # queued by A
        target_study=STUDY_B,
        bound_study=STUDY_A,                 # C is bound to a DIFFERENT study
        bound_root="/stage/" + STUDY_A,
        bound_study_after_reload=STUDY_B,    # B: the real destination
        bound_root_after_reload="/stage/" + STUDY_B,
        init_cookie=True,
    )
    assert res["wrong_consumer_posts"] == 0, res
    assert res["wrong_consumer_clears"] == 0, res
    assert res["wrong_consumer_cookie_present"] is True, res
    assert res["wrong_consumer_slot_present"] is True, res
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res


@pytest.mark.parametrize("raw_stage", ["/stage/./", "/stage/other/../"])
def test_handoff_raw_dot_segment_stage_confirms_via_server_resolved_destination(
    raw_stage: str,
) -> None:
    """PR3 review #4 (app.js:1230): `save_pending_open` preserves the RAW
    validated stage spelling while the bound root is physically resolved. A
    client lexical derivation compares `/stage/./<uid>` with `/stage/<uid>`,
    stands by forever, and silently leaves the successful open unrecorded (on
    every reload too). The destination must be the SERVER-RESOLVED answer (the
    same probe Reopen uses) — never the display-only destRoot."""
    res = _run_handoff_scenario(
        "root_match",
        host="127.0.0.1:2222",           # the destination page
        producer="127.0.0.1:1111",       # queued by A
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_stage_base=raw_stage,              # raw dot-segment spelling
        probe_dest_root="/stage/" + STUDY_B,       # server-resolved answer
        bound_root="/stage/" + STUDY_B,            # matching bound root
    )
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res
    assert res["probe_requests"] >= 1, res


def test_handoff_raw_dot_segment_stage_with_different_resolved_root_stands_by() -> None:
    """PR3 review #4 negative control: server resolution must not weaken the
    different-root refusal. The same raw dot-segment spelling whose SERVER
    answer names a different destination leaves the handoff intact (zero
    records, zero clears) for the real destination."""
    res = _run_handoff_scenario(
        "root_match",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_stage_base="/stage/./",
        probe_dest_root="/stage/OTHER-ROOT",   # server says: a different root
        bound_root="/stage/" + STUDY_B,
    )
    assert res["posts"] == 0, res
    assert res["post_studies"] == [], res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert res["probe_requests"] >= 1, res


def _assert_probe_failure_retains_handoff(res: dict[str, Any]) -> None:
    assert res["posts"] == 0, res
    assert res["post_studies"] == [], res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_pending_destination_probe_failure_is_bounded_and_keeps_handoff() -> None:
    """PR3 review #4: a failing server resolve NEVER decides. The bounded retry
    exhausts into the actionable "may not be saved" notice with the hint and
    the durable slot retained (a later reload retries) — never a silent
    stand-by, never a fabricated record."""
    res = _run_handoff_scenario(
        "root_match",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        probe_fail=True,
        pending_max_attempts=1,
    )
    _assert_probe_failure_retains_handoff(res)
    assert res["probe_requests"] >= 1, res


def test_handoff_pending_destination_probe_hang_times_out_and_keeps_handoff() -> None:
    """PR3 review #4: the bounded whole-fetch+body deadline applies to the
    pending destination resolve too. A hung probe times out, aborts, and lands
    on the same actionable retained-handoff failure instead of wedging the page
    or confirming without a resolved destination."""
    res = _run_handoff_scenario(
        "root_match",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        hang_probe=True,
        probe_timeout_ms=200,
        pending_max_attempts=1,
    )
    _assert_probe_failure_retains_handoff(res)
    assert res["probe_aborted"] is True, res


def test_handoff_pending_destination_probe_delayed_confirms_only_after_resolution() -> None:
    """PR3 review #4 (awaited server resolve): while the destination probe is
    still pending the page records nothing and clears nothing; the matching
    answer then records exactly once and clears exactly once."""
    res = _run_handoff_scenario(
        "pending_destination_probe",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        delay_probe=True,
    )
    assert res["posts_while_probe_pending"] == 0, res
    assert res["clears_while_probe_pending"] == 0, res
    assert res["cookie_while_probe_pending"] is True, res
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res


def test_handoff_transient_bound_root_failure_recovers_and_records_once() -> None:
    """PR3 review 5564032012 F1 (app.js:1316): a transient non-OK bound-root
    lookup is NOT proof of a different workspace. The confirmation retries inside
    its bounded budget with a FRESH lookup (the failed memoized initialization is
    reset), so the destination still records exactly once after recovery.
    Pre-fix the failed lookup stood by silently: the page stopped evaluating and
    the successful open was never recorded."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",           # the destination page
        producer="127.0.0.1:1111",       # queued by A
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        bound_root_fails=1,              # first lookup: transient 503
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res
    # The retry re-asked the server (the failed memoization was reset).
    assert res["bound_root_requests"] >= 2, res


def test_handoff_pending_destination_bound_root_http_failure_exhausts_with_visible_retry() -> None:
    """PR3 review 5564032012 F1: an unmovable bound-root HTTP failure must not
    masquerade as a different workspace. The bounded budget exhausts on the
    actionable "may not be saved" notice with the hint and durable slot retained
    for a later reload; nothing is recorded or cleared."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        bound_root_fails=99,             # the lookup never succeeds
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 0, res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert any("may not be saved" in line for line in res["message_log"]), res
    assert res["bound_root_requests"] >= 1, res


def test_handoff_pending_destination_bound_root_body_timeout_exhausts_with_visible_retry() -> None:
    """PR3 review 5564032012 F1: a bound-root request whose BODY never completes
    times out inside the bounded initialization, and the confirmation lands on
    the same actionable retained-handoff failure instead of a silent standby."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        hang_bound_root_body=True,
        bound_root_timeout_ms=200,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 0, res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert any("may not be saved" in line for line in res["message_log"]), res
    assert res["body_aborted"] is True, res


def _assert_unknown_commit_retains_handoff(res: dict[str, Any]) -> None:
    # Exactly one attempt was made and no second record/clear happened; the
    # commit state is UNKNOWN, so the handoff is retained with a visible notice.
    assert res["posts"] == 1, res
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_record_ack_headers_hang_is_unknown_and_retains_handoff() -> None:
    """PR3 review 5564032012 F2: a record ack that never delivers headers is an
    UNKNOWN commit, not a rejection. The page keeps the hint + durable slot,
    stops with the actionable retry notice, and never claims success."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        hang_record_headers=True,
        record_ack_timeout_ms=200,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    _assert_unknown_commit_retains_handoff(res)


def test_handoff_record_ack_body_hang_is_unknown_and_retains_handoff() -> None:
    """PR3 review 5564032012 F2: 200 headers arrive but the ack BODY never
    settles — the coherent deadline spans body consumption, so the commit stays
    UNKNOWN and the handoff is retained with the actionable notice."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        hang_record_body=True,
        record_ack_timeout_ms=200,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    _assert_unknown_commit_retains_handoff(res)


def test_handoff_record_409_body_hang_is_unknown_and_retains_handoff() -> None:
    """PR3 review 5564032012 F2: a 409 whose body never settles leaves the
    stale-digest retry incomplete — still an UNKNOWN commit (the first POST was
    rejected, the retry never issued), with hint + slot retained and a visible
    retry notice."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        hang_409_body=True,
        record_ack_timeout_ms=200,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    _assert_unknown_commit_retains_handoff(res)


def test_handoff_record_ack_late_retry_commits_once_and_late_ack_is_isolated() -> None:
    """PR3 review 5564032012 F2: an UNKNOWN first ack retries with the SAME
    idempotent open id and commits exactly once; releasing the timed-out ack
    LATE adds no second record, no clear, and cannot republish state."""
    res = _run_handoff_scenario(
        "late_record_retry",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        late_record=True,
        record_ack_timeout_ms=200,
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    # One attempt + one idempotent retry, the SAME open id, one correlated clear.
    assert res["posts"] == 2, res
    assert res["post_open_ids"] == ["poHarness1", "poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert not any("may not be saved" in line for line in res["message_log"]), res
    # Late settlement of the timed-out ack is isolated.
    assert res["posts"] == res["before_late"]["posts"], res
    assert res["pending_clears"] == ["poHarness1"], res


def test_handoff_known_ack_survives_hung_refresh_and_cleanup() -> None:
    """PR3 review 5564032012 F2: once the ack is KNOWN, ancillary work can never
    block or downgrade the confirmation. A hung post-save refresh and a hung
    pending-slot cleanup still produce exactly one record, one attempted clear,
    the released hint, and no unknown-commit notice or retry."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        hang_refresh_after_record=True,
        hang_clear=True,
        pending_cleanup_timeout_ms=150,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert not any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_record_immediate_transport_failure_retries_same_id_and_recovers() -> None:
    """PR3 review 5564147464 (app.js:1945): an IMMEDIATE fetch rejection
    (connection reset) is an indeterminate outcome, never a rejection. It must be
    normalized to record-unknown and retried with the SAME idempotent open id,
    recording exactly once when the service recovers."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_fetch_reject=True,
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 2, res
    assert res["post_open_ids"] == ["poHarness1", "poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res
    assert not any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_record_immediate_transport_failure_exhaustion_retains_handoff() -> None:
    """PR3 review 5564147464: when the transport never recovers, the bounded
    budget exhausts and the hint + durable slot are retained with the actionable
    reload/retry notice — never a silent failure and never a fabricated record."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_fetch_reject=True,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    _assert_unknown_commit_retains_handoff(res)


def test_handoff_record_ack_body_failure_after_commit_retries_same_id() -> None:
    """PR3 review 5564147464: the acknowledgement BODY can fail AFTER the server
    committed. That is indeterminate, not a rejection — the retry carries the
    SAME open id, so the committed write is deduplicated and the case is
    recorded exactly once with one correlated clear."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_body_reject=True,
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 2, res
    assert res["post_open_ids"] == ["poHarness1", "poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res
    assert not any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_record_ack_body_failure_after_commit_exhaustion_retains_handoff() -> None:
    """PR3 review 5564147464: an unrelenting acknowledgement-body failure keeps
    the hint + durable slot with the reload notice, so a possible commit is never
    reported as success nor silently dropped."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_body_reject=True,
        pending_max_attempts=1,
        pending_retry_delay_ms=50,
    )
    _assert_unknown_commit_retains_handoff(res)


def test_handoff_record_409_body_failure_retries_same_id_and_recovers() -> None:
    """PR3 review 5564147464: a 409 whose BODY fails leaves the stale-digest
    retry unissued — indeterminate (nothing committed, nothing refused cleanly)
    — so the bounded retry re-sends with the SAME open id and records once."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_409_body_reject=True,
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 2, res
    assert res["post_open_ids"] == ["poHarness1", "poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert not any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_record_definite_rejection_is_not_retried() -> None:
    """PR3 review 5564147464 (preserved contract): a DEFINITE non-OK response is
    record-rejected — surfaced, never retried (no budget is spent), and the
    handoff is retained rather than cleared as successful."""
    res = _run_handoff_scenario(
        "bound_root_pending_confirm",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        probe_dest_root="/stage/" + STUDY_B,
        bound_root="/stage/" + STUDY_B,
        record_fail=True,
        pending_max_attempts=3,
        pending_retry_delay_ms=50,
    )
    assert res["posts"] == 1, res          # exactly one attempt: no retry of a refusal
    assert res["pending_clears"] == [], res
    assert res["cookie_present"] is True, res
    assert res["pending_slot_present"] is True, res
    assert any("Recent case save failed: HTTP 400" in line for line in res["message_log"]), res
    assert not any("may not be saved" in line for line in res["message_log"]), res


def test_handoff_incoming_confirmation_reserves_intent_against_new_handoff() -> None:
    """PR3 review 5564276070 F1: while B's incoming intent is confirming (its
    pending-slot read is still in flight), a Switch request for C must be refused
    by the page's own single-flight guard — C never reaches the shell, never
    queues, and the shared hint (same original id) plus the only durable slot are
    untouched — so B still records exactly once with its ORIGINAL open id."""
    res = _run_handoff_scenario(
        "incoming_confirm_blocks_handoff",
        host="127.0.0.1:2222",           # B: the destination page
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        pending_read_hang=True,
        invoke_refused=True,
        pending_max_attempts=3,
    )
    assert res["invoke_after_c"] == 0, res
    assert res["queue_after_c"] == 0, res
    assert res["cookie_id_after_c"] == "poHarness1", res
    assert res["slot_after_c"] is True, res
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res


def test_handoff_incoming_confirmation_reserves_intent_across_retry_backoff() -> None:
    """PR3 review 5564276070 F1 (retry backoff leg): the reservation is held
    while the confirmation is between bounded retries too. A delayed/failed B
    confirmation, a refused C request, and then B's retry still records exactly
    once with the original id — C is never recorded."""
    res = _run_handoff_scenario(
        "incoming_confirm_blocks_handoff",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        pending_read_fails=1,            # first read fails -> bounded retry
        pending_retry_delay_ms=600,      # C is requested during the backoff
        invoke_refused=True,
        pending_max_attempts=3,
    )
    assert res["invoke_after_c"] == 0, res
    assert res["queue_after_c"] == 0, res
    assert res["cookie_id_after_c"] == "poHarness1", res
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res


def test_pending_confirmation_retries_through_a_series_change() -> None:
    """PR3 review r4311831210 (fail-capable lifecycle witness): a NORMAL series
    change must not retire a successful pending confirmation.

    The pending-slot read is held; another series is then selected
    (`chooseSeries` clears `state`, and every /state answer is held as well), so
    the read settles while the page state is unknown. A null state is NOT proof
    of a different destination. Pre-fix this returned "standby" ->
    `pendingConfirmExhaustedId`, so no retry ever ran and the successful open
    stayed unrecorded with no failure notice. Post-fix the unknown state retries
    under the bounded budget with the correlated hint and the durable slot
    retained, then records EXACTLY once with the original open id.
    """
    res = _run_handoff_scenario(
        "pending_series_change_race",
        host="127.0.0.1:2222",           # B: the destination page
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        pending_read_hang=True,          # the read is held across the change
        extra_series_uid="S2",           # a genuine different series
        hold_state_for_race=True,        # /state answers held: state stays null
        pending_retry_delay_ms=150,      # several bounded retries inside the window
        pending_max_attempts=6,
    )
    # The race actually happened: state answers were held, the read settled
    # FIRST (nothing recorded yet), and the correlated hint + durable slot were
    # RETAINED across the unknown window instead of being retired.
    assert res["state_holds"] >= 1, res
    assert res["reads_after_release"] >= 1, res
    assert res["posts_after_release"] == 0, res
    assert res["cookie_after_release"] is True, res
    assert res["slot_after_release"] is True, res
    # Retried through the unknown window and recorded exactly once.
    assert res["pending_gets"] >= 2, res
    assert res["posts"] == 1, res
    assert res["post_studies"] == [STUDY_B], res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res


def test_switch_needs_stage_renewal_survives_original_ttl() -> None:
    """PR3 review 5564276070 F2: staging has no two-minute bound. With the clock
    advanced past the ORIGINAL 120s deadline while the stage POST is in flight,
    the page must renew the CORRELATED intent (same id, fresh exp) before the
    final switch invoke so the destination still confirms — exactly one persisted
    open, with the original open id."""
    res = _run_handoff_scenario(
        "staging_renew",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,                        # the switching page is A
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        bound_root_after_reload="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        delay_stage=True,                           # staging held past the TTL
        pending_max_attempts=3,
    )
    assert res["invoke_after"] == 2, res            # needs_stage, then switched
    hint_before = res["hint_before"]
    hint_after = res["hint_after"]
    assert hint_after is not None, res
    assert hint_after["id"] == hint_before["id"], res        # same open id
    assert hint_after["exp"] > hint_before["exp"], res       # renewed deadline
    assert res["slot_exp"] == hint_after["exp"], res         # slot renewed too
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == [hint_before["id"]], res
    assert res["pending_clears"] == [hint_before["id"]], res
    assert res["cookie_present"] is False, res


def test_switch_staging_renewal_never_overwrites_newer_intent() -> None:
    """PR3 review 5564276070 F2 (guard): if another window's newer handoff
    replaces the shared hint while staging runs, the renewal must NOT overwrite
    it and must not re-queue this page's intent — while the switch itself still
    proceeds."""
    res = _run_handoff_scenario(
        "staging_renew",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        delay_stage=True,
        foreign_intent=True,
    )
    assert res["invoke_after"] == 2, res
    assert res["hint_after"]["id"] == "poForeign", res
    assert res["pending_posts"] == 1, res            # only the click-time queue


def _reopen_staging_entries() -> dict[str, Any]:
    """Recent-list payload for the Reopen-leg staging witnesses below."""
    return {
        "version": 1,
        "entries": [
            {
                "studyUid": STUDY_B,
                "seriesUid": None,
                "scanSourcePath": "/src",
                "stageBaseDir": "/stage",
                "destRoot": "/stage/" + STUDY_B,
                "openedSeq": 1,
                "openCount": 1,
            }
        ],
        "discarded": 0,
        "reason": "none",
        "digest": "d-list",
    }


def test_switch_staging_renewal_non_ok_arms_warning_before_final_invoke() -> None:
    """Review 5566059819 (Switch leg): a DEFINITE non-OK renewal must arm the
    path-free warning BEFORE the final invoke — never a silent unrecorded
    switch. The renewal attempt itself must have been issued (same id)."""
    res = _run_handoff_scenario(
        "staging_renew_warning",
        host="127.0.0.1:2224",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        delay_stage=True,
        stage_renew_fails=1,
        pending_max_attempts=3,
    )
    assert res["invoke_calls"] == 2, res
    assert res["invoke_notices"][0] is False, res   # nothing armed before #1
    assert res["invoke_notices"][1] is True, res    # warning armed before #2
    assert res["notice_cookie_present"] is True, res
    assert res["pending_posts"] == 2, res           # click queue + renewal attempt
    assert res["hint_after"]["id"] == res["hint_before"]["id"], res
    assert res["hint_after"]["exp"] > res["hint_before"]["exp"], res


def test_switch_staging_renewal_timeout_arms_warning_before_final_invoke() -> None:
    """Review 5566059819 (Switch leg): a HUNG renewal that times out inside the
    bounded queue budget must arm the same path-free warning before the final
    invoke (degraded unrecorded switch, never broken, never silent)."""
    res = _run_handoff_scenario(
        "staging_renew_warning",
        host="127.0.0.1:2226",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        delay_stage=True,
        stage_renew_hang=True,
        queue_timeout_ms=300,
        pending_max_attempts=3,
    )
    assert res["invoke_calls"] == 2, res
    assert res["invoke_notices"][1] is True, res    # warning armed before #2
    assert res["notice_cookie_present"] is True, res
    assert res["pending_posts"] == 2, res           # click queue + hung renewal


def test_reopen_staging_renewal_survives_original_ttl_and_records_once() -> None:
    """Review 5566059819 (Reopen leg, real positive Max-Age expiry): staging is
    held past the ORIGINAL deadline; the browser-equivalent cookie is
    physically expired by the controlled clock. The Reopen chain must renew
    the correlated intent through its in-memory lease (same id, fresh exp) and
    the bound destination must record exactly once with the original id."""
    res = _run_handoff_scenario(
        "reopen_staging_renew",
        host="127.0.0.1:2225",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        bound_root_after_reload="/stage/" + STUDY_B,
        probe_dest_root="/stage/" + STUDY_B,
        recent_entries=_reopen_staging_entries(),
        delay_stage=True,
    )
    assert res["invoke_after"] == 2, res
    hint_before = res["hint_before"]
    hint_after = res["hint_after"]
    assert hint_before is not None, res
    assert hint_after is not None, res
    assert hint_after["id"] == hint_before["id"], res        # same open id
    assert hint_after["exp"] > hint_before["exp"], res       # renewed deadline
    assert res["slot_exp"] == hint_after["exp"], res         # slot renewed too
    assert res["posts"] == 1, res                            # destination records once
    assert res["post_open_ids"] == [hint_before["id"]], res
    assert res["pending_clears"] == [hint_before["id"]], res
    assert res["cookie_present"] is False, res


def test_reopen_staging_renewal_non_ok_arms_warning_before_final_invoke() -> None:
    """Review 5566059819 (Reopen leg): a definite non-OK renewal must arm the
    path-free warning BEFORE the final reopen invoke."""
    res = _run_handoff_scenario(
        "reopen_staging_renew_warning",
        host="127.0.0.1:2227",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_B,
        recent_entries=_reopen_staging_entries(),
        delay_stage=True,
        stage_renew_fails=1,
    )
    assert res["invoke_calls"] == 2, res
    assert res["invoke_notices"][0] is False, res   # nothing armed before #1
    assert res["invoke_notices"][1] is True, res    # warning armed before #2
    assert res["notice_cookie_present"] is True, res
    assert res["pending_posts"] == 2, res           # click queue + renewal attempt
    assert res["hint_after"]["id"] == res["hint_before"]["id"], res


def test_reopen_staging_renewal_timeout_arms_warning_before_final_invoke() -> None:
    """Review 5566059819 (Reopen leg): a HUNG renewal that times out inside the
    bounded queue budget must arm the same path-free warning before the final
    reopen invoke (best-effort reopen, never silent)."""
    res = _run_handoff_scenario(
        "reopen_staging_renew_warning",
        host="127.0.0.1:2228",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_B,
        recent_entries=_reopen_staging_entries(),
        delay_stage=True,
        stage_renew_hang=True,
        queue_timeout_ms=300,
    )
    assert res["invoke_calls"] == 2, res
    assert res["invoke_notices"][1] is True, res    # warning armed before #2
    assert res["notice_cookie_present"] is True, res
    assert res["pending_posts"] == 2, res           # click queue + hung renewal


def test_handoff_expired_hint_after_clock_is_not_consumed() -> None:
    """PR3 review 5564276070 F2 (abandoned-intent bound preserved): with the
    clock past the hint's own expiry and no renewal, the destination consumes
    nothing — the expired hint is never even read and the slot is never
    confirmed, so abandoned handoffs still expire."""
    res = _run_handoff_scenario(
        "stale_hint_after_clock",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,                          # the first page is A
        init_cookie=True,
        bound_root="/stage/" + STUDY_A,
        bound_root_after_reload="/stage/" + STUDY_B,  # the destination is B
        probe_dest_root="/stage/" + STUDY_B,
    )
    assert res["posts_before_reload"] == 0, res
    assert res["posts"] == 0, res
    assert res["pending_gets"] == res["gets_before_reload"], res
    assert res["pending_clears"] == [], res


def test_handoff_init_claim_expiry_releases_slot_for_later_handoff() -> None:
    """Review 5566059819 P2-2: the reservation armed at init must clean up by
    the intent's own expiry even while the first /state is still delayed
    (confirmation never starts). The later handoff is accepted and releasing
    the delayed state records nothing for the expired intent."""
    res = _run_handoff_scenario(
        "delayed_initial_expiry_then_switch",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        delay_initial_state=True,
        init_intent_exp_ms=300,
    )
    assert res["claim_held_after_expiry"] is False, res       # claim released
    assert res["invoke_after_click"] == 1, res                # later handoff accepted
    assert res["queue_after_click"] == 1, res                 # ... queued exactly once
    assert res["posts"] == 0, res                             # no expired-intent record
    assert res["post_open_ids"] == [], res


def test_handoff_init_claim_expiry_keeps_replaced_intent_untouched() -> None:
    """Review 5566059819 P2-2 (correlation): while the claim is held the shared
    hint is replaced by another window's newer intent. Expiry cleanup may
    release only THIS page's reservation — the replacement hint and slot are
    never cleared — and the page can still start its own handoff."""
    res = _run_handoff_scenario(
        "replaced_intent_cleanup_then_switch",
        host="127.0.0.1:2223",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        delay_initial_state=True,
        init_intent_exp_ms=300,
    )
    assert res["claim_held_after_expiry"] is False, res         # correlated release
    assert res["cookie_id_after_expiry"] == "poForeign2", res  # replacement kept
    assert res["pending_clears"] == [], res                    # slot never cleared
    assert res["invoke_after_click"] == 1, res                 # later handoff accepted
    assert res["posts"] == 0, res


def test_handoff_producer_or_its_reload_never_consumes_own_intent() -> None:
    """Producer exclusion: a page whose authority matches the intent's producer
    stamp — the producing page itself, or a reload of it during staging — must
    not consume its own intent (it is not the destination)."""
    res = _run_handoff_scenario(
        "producer_reload",
        host="127.0.0.1:1111",          # same authority as the producer
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=True,
    )
    assert res["posts"] == 0, res
    assert res["cookie_present"] is True, res  # not ours to drop


def test_handoff_restored_page_discards_intent_before_consumption() -> None:
    """Refused/restored outcome: the shell's `#switch-failed` navigation must
    discard the queued intent synchronously, before any startup consumption,
    so a failed switch can never fabricate a successful-open record."""
    res = _run_handoff_scenario(
        "restored_hash",
        host="127.0.0.1:5555",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,            # binding would otherwise match
        init_cookie=True,
        page_hash="#switch-failed",
    )
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_handoff_restored_page_releases_reservation_for_new_handoff() -> None:
    """Review 5566059819 P2-1: the #switch-failed restored page discards its
    captured intent; the correlated incoming reservation must be released with
    it. The failed switch recorded nothing, so the restored page must be able
    to start a NEW handoff instead of staying single-flighted until a reload."""
    res = _run_handoff_scenario(
        "restored_hash_then_switch",
        host="127.0.0.1:5556",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        page_hash="#switch-failed",
    )
    assert res["posts"] == 0, res                 # failed switch: nothing recorded
    assert res["invoke_calls"] == 1, res          # new handoff accepted
    assert res["pending_posts"] == 1, res         # ... and queued exactly once
    assert res["cookie_present"] is True, res     # new intent survives for A->B


def test_handoff_refused_switch_clears_queued_intent() -> None:
    """Refused outcome through the REAL Switch Study affordance: the intent is
    queued before the IPC invoke, and a non-switched result clears it (correlated
    by id), leaving nothing for any later page to consume."""
    res = _run_handoff_scenario(
        "switch_refused",
        host="127.0.0.1:6666",
        producer="127.0.0.1:6666",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
    )
    assert res["invoke_calls"] == 1, res
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_handoff_staging_then_refusal_clears_queued_intent() -> None:
    """Staging outcome: needs_stage -> stage on the current backend -> re-invoke
    -> non-switched result. The queued intent must be cleared exactly as in the
    direct-refusal path (nothing is left behind for a later page)."""
    res = _run_handoff_scenario(
        "switch_staged_refused",
        host="127.0.0.1:7777",
        producer="127.0.0.1:7777",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
    )
    assert res["invoke_calls"] == 2, res
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_handoff_second_switch_is_blocked_while_one_is_outstanding() -> None:
    """Round-3 R3-1 (executable): Switch and Reopen share ONE cookie slot, so
    a second click while the first IPC is outstanding must be refused BEFORE
    it writes a cookie or invokes the shell — otherwise the first action's
    refusal deletes the newer intent and a later legitimate switch reaches its
    destination with no first-open record (the reviewer's two-click probe).

    Serialization makes the overlapping-id case unreachable by construction;
    the per-action `intentId` capture asserted in the wiring test is the
    defence-in-depth behind it. The third click proves the slot is released
    rather than left wedged."""
    res = _run_handoff_scenario(
        "double_switch",
        host="127.0.0.1:8888",
        producer="127.0.0.1:8888",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
    )
    # Second click never reached the shell, never queued, and never wrote
    # a cookie... (QUEUE-011: the refused click returns before setPendingOpen,
    # so exactly ONE queue POST exists after two clicks.)
    assert res["invoke_after_two_clicks"] == 1, res
    assert res["queue_after_two_clicks"] == 1, res  # single queue, no double-fire
    assert res["cookie_after_two_clicks"] is True, res  # only the first intent
    # ...the first refusal cleared only that intent...
    assert res["cookie_after_first_refusal"] is False, res
    # ...and the slot was released: a later switch invokes again (with its
    # own single queue POST), and no record was ever fabricated by this
    # unswitched page.
    assert res["invoke_calls"] == 2, res
    assert res["pending_posts"] == 2, res  # first click + third click, never the refused one
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_handoff_hint_cookie_carries_no_filesystem_paths() -> None:
    """Confidentiality P1 (executable): the cookie queued by REAL Switch
    Study clicks is an opaque hint only — at no point does the jar hold an
    entry or filesystem paths. The declined-then-requeued flow (double
    switch: first click queues, refusal clears, third click re-queues, all
    refused) exercises queue + clear + re-queue with the jar observed
    path-free throughout; the full entry travels server-side instead."""
    res = _run_handoff_scenario(
        "double_switch",
        host="127.0.0.1:9991",
        producer="127.0.0.1:9991",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
    )
    assert res["invoke_after_two_clicks"] == 1, res
    assert res["queue_after_two_clicks"] == 1, res
    assert res["cookie_after_two_clicks"] is True, res
    assert res["cookie_after_first_refusal"] is False, res
    assert res["invoke_calls"] == 2, res
    assert res["pending_posts"] == 2, res
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res
    assert res["cookie_has_paths"] is False, res
    # Every queued hint observed during the run was the path-free shape.
    assert len(res["cookie_jar_shapes"]) >= 1, res
    for shape in res["cookie_jar_shapes"]:
        assert shape["keys"] == "exp,id,producer", shape
        assert shape["has_paths"] is False, shape


def test_queue_post_is_awaited_before_shell_invoke() -> None:
    """QUEUE-011 killed-A witness (executable): with the queue POST held
    unresolved (backend A not yet durable), the shell invoke must NOT fire;
    it fires only after the queue is released as persisted. Pre-fix code
    invoked without awaiting (invoke_while_queue_held == 1)."""
    res = _run_handoff_scenario(
        "delayed_queue",
        host="127.0.0.1:8889",
        producer="127.0.0.1:8889",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        delay_queue=True,
    )
    assert res["queue_dispatched"] == 1, res  # exactly one queue POST, dispatched
    assert res["invoke_while_queue_held"] == 0, res  # ... and no invoke until it settles
    assert res["invoke_calls"] == 1, res  # released queue -> single invoke
    assert res["pending_posts"] == 1, res  # still exactly one queue POST total


def test_queue_hang_degrades_to_unrecorded_switch() -> None:
    """QUEUE-HANG (R5) witness (executable): the queue POST is held past the
    page's bounded timeout (backend A killed mid-flight). All five owner
    conditions are witnessed: (1) the await settles bounded (~5s, single
    invoke predates the 6s late release — no wedge); (2) the switch still
    invokes (never broken); (3) exactly one queue POST, zero records
    (unrecorded, never fabricated); (4) the explicit "may not be saved"
    notice is user-visible; (5) the late ok:true is blocked (no second
    invoke, no record). Pre-fix code awaited with no timeout: its invoke
    lands only after the late release, with no notice."""
    res = _run_handoff_scenario(
        "queue_hang",
        host="127.0.0.1:8890",
        producer="127.0.0.1:8890",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        hang_queue=True,
    )
    assert res["queue_dispatched"] == 1, res  # exactly one queue POST dispatched
    assert res["pending_posts"] == 1, res  # ... no retry storm
    assert res["invoke_after_hang"] == 1, res  # the switch still invoked (never broken)
    assert res["invoke_calls"] == 1, res
    assert len(res["invoke_times"]) == 1, res  # ... exactly once (late ok:true blocked)
    assert res["invoke_times"][0] < res["released_at"], res  # timeout fired BEFORE the late release
    assert res["invoke_times"][0] - res["click_at"] >= 4000, res  # awaited the bounded timeout
    assert res["posts"] == 0, res  # unrecorded, never fabricated
    assert any("may not be saved" in m for m in res["message_log"]), res  # notice shown


def test_queue_hang_without_abort_controller_degrades_to_unrecorded_switch() -> None:
    """P2 regression witness (fail-capable): when AbortController is unavailable
    and the queue POST hangs, the non-abort timeout race must still settle
    within QUEUE_TIMEOUT_MS and proceed with the switch in degraded mode.

    Pre-fix code in the no-AbortController branch awaited fetch() directly
    with no timeout or Promise.race, leaving handoffOutstanding true forever
    and wedging every subsequent action without invoking the shell."""
    res = _run_handoff_scenario(
        "queue_hang",
        host="127.0.0.1:8892",
        producer="127.0.0.1:8892",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        hang_queue=True,
        no_abort_controller=True,
        queue_timeout_ms=300,
    )
    assert res["abort_controller_present"] is False, res
    assert res["queue_dispatched"] == 1, res  # exactly one queue POST dispatched
    assert res["pending_posts"] == 1, res  # ... no retry storm
    assert res["invoke_after_hang"] == 1, res  # switch still invoked (never broken)
    assert res["invoke_calls"] == 1, res
    assert len(res["invoke_times"]) == 1, res  # ... exactly once (late ok:true blocked)
    assert res["invoke_times"][0] < res["released_at"], res  # timeout fired BEFORE late release
    assert res["invoke_times"][0] - res["click_at"] >= 250, res  # awaited bounded timeout
    assert res["posts"] == 0, res  # unrecorded, never fabricated
    assert any("may not be saved" in m for m in res["message_log"]), res  # notice shown


def test_queue_error_shows_may_not_be_saved_notice() -> None:
    """QUEUE-ERROR (R5-ratified) witness (executable): the queue POST fails
    fast (500). The switch must still invoke exactly once (unrecorded,
    never broken, never fabricated) WITH the explicit "may not be saved"
    notice. Pre-notice code invokes silently (no notice)."""
    res = _run_handoff_scenario(
        "queue_error",
        host="127.0.0.1:8891",
        producer="127.0.0.1:8891",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        queue_fail=True,
    )
    assert res["queue_dispatched"] == 1, res  # exactly one queue POST dispatched
    assert res["pending_posts"] == 1, res  # ... no retry storm
    assert res["invoke_after_error"] == 1, res  # the switch still invoked (never broken)
    assert res["invoke_calls"] == 1, res
    assert res["posts"] == 0, res  # unrecorded, never fabricated
    assert any("may not be saved" in m for m in res["message_log"]), res  # ratified notice shown


def test_record_failure_keeps_pending_slot() -> None:
    """PENDING-SUCCESS-015 witness (executable): the bound destination
    consumes the seeded intent but the record POST fails — the pending slot
    must be KEPT (never cleared as successful) and exactly one record
    attempt is made (400, no 409 retry). Pre-fix code cleared the slot
    unconditionally after the failed record."""
    res = _run_handoff_scenario(
        "record_failure_keeps_slot",
        host="127.0.0.1:2223",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,  # binding matches, so consume runs
        init_cookie=True,
        record_fail=True,
        bound_root="/stage/" + STUDY_B,  # root binding matches too (review #3)
    )
    assert res["posts"] == 1, res  # single failed record attempt, no retry loop
    assert res["pending_slot_present"] is True, res  # slot KEPT, not cleared


def test_handoff_pending_read_failure_retries_same_id_and_records_once() -> None:
    """F3 (PR3 review): one transient pending-route read failure must NOT
    consume the only handoff hint. The page retains the hint, retries the SAME
    idempotent open id, records exactly once, and clears the cookie/slot only
    AFTER the confirmed record (clear strictly after record in the event log).
    Pre-fix code dropped the hint before the read and silent-returned false:
    zero retries, zero records, a permanently omitted open."""
    res = _run_handoff_scenario(
        "pending_read_fail_then_retry",
        host="127.0.0.1:2224",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_read_fails=1,
        pending_retry_delay_ms=30,
        bound_root="/stage/" + STUDY_B,  # destination root (review #3)
    )
    assert res["pending_gets"] >= 2, res  # the retained hint was retried
    assert res["posts"] == 1, res  # exactly one record
    assert res["post_open_ids"] == ["poHarness1"], res  # same idempotent open id
    assert res["pending_clears"] == ["poHarness1"], res  # one correlated clear
    events = res["handoff_events"]
    assert events.count("record") == 1, res
    assert events.index("pending-clear") > events.index("record"), events
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res


def test_handoff_pending_read_failure_then_reload_records_once() -> None:
    """F3 (PR3 review): with the per-page retry still pending, a RELOAD must
    find the retained hint and record exactly once — the durable pending slot
    is never stranded by a transient read failure. Pre-fix code cleared the
    cookie before the read, so the reload had no hint and the successfully
    switched case stayed permanently omitted."""
    res = _run_handoff_scenario(
        "pending_read_fail_reload",
        host="127.0.0.1:2225",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_read_fails=1,
        pending_retry_delay_ms=100000,  # page 1 cannot retry inside the window
        bound_root="/stage/" + STUDY_B,
    )
    before = res["before_reload"]
    assert before["posts"] == 0, res
    assert before["cookie_present"] is True, res  # hint RETAINED, not consumed
    assert before["slot_present"] is True, res
    assert res["posts"] == 1, res  # the reload recorded exactly once
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_gets"] == 2, res  # page-1 read + page-2 read only
    assert res["cookie_present"] is False, res  # cleared only after confirmation


def test_handoff_pending_read_exhaustion_is_actionable_and_keeps_hint() -> None:
    """F3 (PR3 review): bounded retry — when every attempt fails the page must
    stop (no retry storm), surface an actionable failure, and KEEP the hint +
    slot so a later reload can still confirm. Never a silent no-op, never a
    fabricated record, never a premature clear."""
    res = _run_handoff_scenario(
        "pending_read_fail_exhausted",
        host="127.0.0.1:2226",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_read_fails=99,
        pending_max_attempts=2,
        pending_retry_delay_ms=20,
        bound_root="/stage/" + STUDY_B,
    )
    assert res["posts"] == 0, res  # never fabricated
    assert res["pending_gets"] == 2, res  # bounded: exactly the budget
    assert res["pending_clears"] == [], res  # no premature clear
    assert res["cookie_present"] is True, res  # retained for a later reload
    assert res["pending_slot_present"] is True, res
    assert any("may not be saved" in m for m in res["message_log"]), res


def test_handoff_pending_read_body_deadline_and_late_isolation() -> None:
    """F3 (PR3 review): the pending read deadline must cover the BODY, not just
    the headers. A stalled body is aborted at the deadline, the hint retained,
    the retry recorded exactly once — and the late body of the timed-out
    attempt is isolated: no second record, no spurious clear."""
    res = _run_handoff_scenario(
        "pending_read_hang_body",
        host="127.0.0.1:2227",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_read_hang_body=True,
        pending_timeout_ms=200,
        pending_retry_delay_ms=30,
        bound_root="/stage/" + STUDY_B,
    )
    assert res["aborted_at_timeout"] is True, res  # deadline aborted the body
    assert res["reads_after_retry"] == 2, res  # retried after the deadline
    assert res["posts_before_late_body"] == 1, res  # exactly one record
    assert res["pending_body_resolved"] == 1, res  # the late body really settled
    assert res["posts"] == 1, res  # ... and could not record again
    assert res["pending_clears"] == ["poHarness1"], res
    events = res["handoff_events"]
    assert events.count("record") == 1, res
    assert events.index("pending-clear") > events.index("record"), events


def test_handoff_pending_retry_is_single_flight_under_concurrent_reentry() -> None:
    """F3 (PR3 review): concurrent evaluations while the bounded retry is
    pending (startup/heartbeat re-entry) must not start extra reads — the retry
    owns the attempt — and the eventual retry records exactly once with the
    same idempotent open id. Pre-fix code never retried at all (zero records)."""
    res = _run_handoff_scenario(
        "pending_read_retry_singleflight",
        host="127.0.0.1:2228",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        pending_read_fails=1,
        pending_retry_delay_ms=150,
        bound_root="/stage/" + STUDY_B,
    )
    assert res["reads_while_retry_pending"] == 1, res  # no re-entrant read
    assert res["pending_gets"] == 2, res  # exactly one retry
    assert res["posts"] == 1, res  # exactly one record
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res


def test_handoff_same_uid_other_root_consumer_records_nothing_and_keeps_handoff() -> None:
    """PR3 review #3 (root binding): a consumer bound to a DIFFERENT workspace
    root but the SAME study UID is not the destination. It must record nothing
    and must not consume the shared hint cookie / durable pending slot — the
    real destination (matching server-reported bound root) must still record
    exactly once afterwards. Pre-fix the UID check alone let the wrong window
    record the queued target, clear the shared slot/hint, and strand the real
    handoff."""
    res = _run_handoff_scenario(
        "root_mismatch_then_destination",
        host="127.0.0.1:3333",           # the OTHER window
        producer="127.0.0.1:1111",       # queued by A
        target_study=STUDY_B,
        bound_study=STUDY_B,             # same UID: the old check passed here
        init_cookie=True,
        bound_root="/stage/OTHER-ROOT",  # different root, same UID
        bound_root_after_reload="/stage/" + STUDY_B,  # the real destination
    )
    # Wrong consumer: zero records, zero clears, handoff fully intact.
    assert res["wrong_consumer_posts"] == 0, res
    assert res["wrong_consumer_clears"] == 0, res
    assert res["wrong_consumer_slot_present"] is True, res
    assert res["wrong_consumer_cookie_present"] is True, res
    # The real destination (matching bound root) then records exactly once.
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res


def test_handoff_matching_bound_root_consumer_records_once_positive_control() -> None:
    """PR3 review #3 positive control: a destination whose SERVER-REPORTED bound
    root equals the pending entry's independently derived destination (stage
    base + UID, never the display-only destRoot) records exactly once and clears
    the handoff — the root binding must not block the legitimate destination."""
    res = _run_handoff_scenario(
        "root_match",
        host="127.0.0.1:2222",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_B,
        init_cookie=True,
        bound_root="/stage/" + STUDY_B,   # matching root
    )
    assert res["posts"] == 1, res
    assert res["post_open_ids"] == ["poHarness1"], res
    assert res["pending_clears"] == ["poHarness1"], res
    assert res["cookie_present"] is False, res
    assert res["pending_slot_present"] is False, res


def test_fresh_destination_bootstraps_live_root_without_hook() -> None:
    """SAMEROOT-014 residual-leg witness (executable, no test hook): a fresh
    destination page consumes the seeded intent (bound study matches) and
    learns its live root from the SERVER-reported bound root (R7 F5) — then
    clicking Reopen on the same entry no-ops without a shell invoke.
    hook_calls == 0 proves no test-hook seeding was involved; the bound-root
    request proves the genuine initialization path was used."""
    same_entry = {
        "version": 1,
        "entries": [
            {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": "/src",
                "stageBaseDir": "/stage",
                "destRoot": "",  # producer shape: derived as /stage/<study>
                "openedSeq": 1,
                "openCount": 1,
            }
        ],
        "discarded": 0,
        "reason": "none",
        "digest": "d-list",
    }
    res = _run_handoff_scenario(
        "consume_then_reopen",
        host="127.0.0.1:9994",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,  # page is bound to the listed case
        init_cookie=True,  # seeded intent present at page init
        recent_entries=same_entry,
        bound_root="/stage/" + STUDY_A,  # server-reported genuinely bound root
        # NO live_root: the hook must never fire on this path.
    )
    assert res["buttons"] >= 1, res  # live button + shim-retained pre-render orphan
    assert res["posts"] == 1, res  # the consume record happened (bootstrap had a source)
    assert res["hook_calls"] == 0, res  # ... with zero test-hook involvement
    assert res["bound_root_requests"] >= 1, res  # root learned from the server
    assert res["invoke_calls"] == 0, res  # ... and the reopen no-ops without invoke
    assert res["cookie_present"] is False, res


def test_pending_hint_cookie_shape_has_no_paths() -> None:
    """Confidentiality P1 (static): the producer writes exactly the hint
    keys {id, exp, producer} — any entry/paths key in setPendingOpen is a
    regression back to the JS-readable leak. QUEUE-011: setPendingOpen fires
    no POST itself (function body contains no queue call) — each click
    queues exactly once via the caller's single awaited queuePendingOpen."""
    app_js_path = (
        Path(__file__).parents[2] / "services" / "workstation" / "static" / "app.js"
    )
    app_js = app_js_path.read_text(encoding="utf-8")
    start = app_js.index("function setPendingOpen(entry)")
    end = app_js.index("\n  }\n", start)
    body = app_js[start:end]
    assert "entry: entry" not in body
    assert "scanSourcePath" not in body
    assert "stageBaseDir" not in body
    assert "destRoot" not in body
    assert "queuePendingOpen(" not in body
    assert "return id;" in body


def test_reopen_same_root_is_noop_without_invoke() -> None:
    """Same-root no-op C2 (executable, S10 `unchanged` leg): clicking Reopen
    on the genuinely live root must NOT handoff, queue, or invoke — the
    shell would kill/respawn the live backend. Identity is the RESOLVED
    destination versus the live root, never a bare UID match."""
    live_root = "/stage/" + STUDY_A
    active_entry = {
        "version": 1,
        "entries": [
            {
                "studyUid": STUDY_A,
                "seriesUid": None,
                "scanSourcePath": "/src",
                "stageBaseDir": "/stage",
                "destRoot": live_root,
                "openedSeq": 1,
                "openCount": 1,
            }
        ],
        "discarded": 0,
        "reason": "none",
        "digest": "d-list",
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9992",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,  # page is bound to the listed case
        init_cookie=False,
        recent_entries=active_entry,
        live_root=live_root,  # this root is genuinely live
    )
    assert res["buttons"] == 1, res
    assert res["invoke_calls"] == 0, res
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_reopen_same_uid_different_root_still_invokes() -> None:
    """C2 control: the SAME study UID restaged under a DIFFERENT stage base
    is a different root — the no-op must NOT fire, and the shell must be
    invoked exactly once. UID equality alone never proves same-root."""
    live_root = "/stage/" + STUDY_A
    other_entry = {
        "version": 1,
        "entries": [
            {
                "studyUid": STUDY_A,  # same UID ...
                "seriesUid": None,
                "scanSourcePath": "/src",
                "stageBaseDir": "/stageB",  # ... different stage base
                "destRoot": "/stageB/" + STUDY_A,
                "openedSeq": 1,
                "openCount": 1,
            }
        ],
        "discarded": 0,
        "reason": "none",
        "digest": "d-list",
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9993",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=other_entry,
        live_root=live_root,  # /stage/A live; entry resolves to /stageB/A
    )
    assert res["buttons"] == 1, res
    assert res["invoke_calls"] == 1, res
    assert res["cookie_present"] is True, res


# ---------------------------------------------------------------------------
# R7 review (native Review-Hard at 31377c53) — finding witnesses
# ---------------------------------------------------------------------------

def _single_entry_document(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "entries": [entry],
        "discarded": 0,
        "reason": "none",
        "digest": "d-list",
    }


def _bound_entry(study_uid: str, *, stage_base: str = "/stage") -> dict[str, Any]:
    """Synthetic recent row for ``study_uid`` staged under ``stage_base``."""
    return {
        "studyUid": study_uid,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": stage_base,
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }


def test_recent_active_marker_reconciled_when_state_arrives_after_render() -> None:
    """PR3 review r4311831218 (fail-capable): the "current case" marker must be
    reconciled, not snapshotted once during the recent-list render.

    The recent-list response renders BEFORE the initial /state answer (the
    ordinary relaunch ordering when the store is fast and the study load is
    slow). Pre-fix the marker decision ran with ``state === null``, so no row
    was marked and ``loadState()`` never updated the markers afterwards — the
    current case stayed unmarked indefinitely even though the bound identity
    was known moments later. Post-fix the marker is applied when the identity
    becomes available, and an unknown identity never claims to be current.
    """
    res = _run_handoff_scenario(
        "marker_reconcile_delayed_state",
        host="127.0.0.1:9993",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(_bound_entry(STUDY_A)),
        bound_root="/stage/" + STUDY_A,   # genuine bound-root route, no test hook
        delay_initial_state=True,         # the identity arrives AFTER the render
        live_root_after="/stageB/" + STUDY_A,  # relocation leg: live root moves away
    )
    before = res["marker_before_state"]
    after = res["marker_after_state"]
    assert len(before) == 1, res
    # Unknown identity is never reported as the current case.
    assert before[0]["aria_current"] is None, res
    assert before[0]["active_class"] is False, res
    # ...and no visible current-case claim is shown either (r4313043501).
    assert before[0]["current_label_text"] == "Current case", res
    assert before[0]["current_label_hidden"] is True, res
    # ... and the marker appears once the bound identity is established.
    assert len(after) == 1, res
    assert after[0]["study"] == STUDY_A, res
    assert after[0]["aria_current"] == "true", res
    assert after[0]["active_class"] is True, res
    # The sighted operator can SEE the current case, not only query its class.
    assert after[0]["current_label_text"] == "Current case", res
    assert after[0]["current_label_hidden"] is False, res
    # Relocation: the live root moves to a different case root, so the marker
    # (class, aria-current AND the visible label) must be withdrawn.
    relocated = res["marker_after_relocation"]
    assert relocated is not None, res
    assert len(relocated) == 1, res
    assert relocated[0]["aria_current"] is None, res
    assert relocated[0]["active_class"] is False, res
    assert relocated[0]["current_label_hidden"] is True, res


def test_recent_active_marker_requires_matching_resolved_destination() -> None:
    """PR3 review r4311831218 (fail-capable): a same-UID row is only the current
    case when its SERVER-RESOLVED destination is the live bound root.

    Two recent rows carry the same study UID but different stage bases, so the
    server resolves them to different workspace roots (``/stage/<A>`` and
    ``/stageB/<A>``) while the bound root is ``/stage/<A>``. The Reopen handler
    treats that distinction as requiring a switch, so the marker must not claim
    both rows are open. Pre-fix every same-UID row got ``aria-current``; the
    row whose destination is a different root was marked as current.
    """
    res = _run_handoff_scenario(
        "marker_reconcile_same_uid_root",
        host="127.0.0.1:9993",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries={
            "version": 1,
            "entries": [
                _bound_entry(STUDY_A, stage_base="/stage"),    # -> /stage/<A>  (live)
                _bound_entry(STUDY_A, stage_base="/stageB"),   # -> /stageB/<A> (switch)
            ],
            "discarded": 0,
            "reason": "none",
            "digest": "d-list",
        },
        bound_root="/stage/" + STUDY_A,
        delay_recent_get=True,   # the state identity is known BEFORE the render
    )
    marker = res["marker"]
    assert len(marker) == 2, res
    assert res["probe_requests"] >= 2, res   # destinations were actually resolved
    assert marker[0]["aria_current"] == "true", res
    assert marker[0]["active_class"] is True, res
    assert marker[0]["current_label_text"] == "Current case", res
    assert marker[0]["current_label_hidden"] is False, res
    assert marker[1]["aria_current"] is None, res
    assert marker[1]["active_class"] is False, res
    assert marker[1]["current_label_text"] == "Current case", res
    assert marker[1]["current_label_hidden"] is True, res


def test_served_stylesheet_renders_the_current_case_marker_visibly(
    seeded_env: tuple[SeededStudy, Path, Path, Path],
) -> None:
    """PR3 review r4313043501 (fail-capable): the reconciled marker must change
    VISUAL output on the served profile, not only DOM state.

    The reconciler set ``active`` and ``aria-current``, but ``style.css`` had no
    active/current selector and the row text carried only the UID and open
    count, so a sighted operator could not tell which case was open and S7's
    ACTIVE observation was impossible. This asserts the SERVED stylesheet
    carries the visible treatment the row label relies on, so the visual
    distinction cannot silently disappear again.
    """
    seeded, _src, _stage, _store = seeded_env
    with running_server(seeded.app) as (srv, base_url):
        res = _safe_request(srv, base_url, "/static/style.css")
    assert res.status == 200, res
    css = res.body.decode("utf-8")
    assert ".recent-item.active" in css, css
    assert "border-left" in css, css
    assert ".recent-current-label" in css, css
    assert ".recent-current-label[hidden]" in css, css


def test_reopen_equal_root_different_bound_study_never_reports_already_open() -> None:
    """PR3 review r4310770076 (fail-capable): path equality alone must never
    answer "already open".

    A stale recent row for study B can resolve to a root that is genuinely the
    live bound root while that root's DICOM now serves study A: the recent
    probe only checks layout and echoes the requested study_uid, so it cannot
    prove DICOM study identity. The shell prove step exists to reject exactly
    that mismatch on a real switch; the page must not short-circuit it. With
    equal server-resolved roots but a DIFFERENT independently established
    bound study (the live backend's own DICOM-derived state projection),
    Reopen must invoke the shell once and queue its intent instead of
    reporting the case already open. Pre-fix (`app.js` 3f495265): RED - the
    page reported "already open" with 0 invokes and no queued intent, masking
    the mismatch.
    """
    live_root = "/stage/" + STUDY_B  # equals the entry's server-resolved dest
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9988",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,  # the live backend actually serves A
        init_cookie=False,
        recent_entries=_single_entry_document(_bound_entry(STUDY_B)),
        bound_root=live_root,
    )
    assert res["buttons"] == 1, res
    assert "already open" not in res["message_text"], res
    assert res["invoke_calls"] == 1, res
    assert res["pending_posts"] == 1, res
    assert res["cookie_present"] is True, res


def test_reopen_waits_for_bound_study_identity_before_same_root_noop() -> None:
    """PR3 review r4310770076 lifecycle witness (fail-capable): the only
    INDEPENDENT bound-study identity source (the live backend's DICOM-derived
    state projection) has not loaded yet when Reopen is clicked.

    Unknown identity must never be reported as "already open", and the click
    must not invoke or queue while it is pending: the decision waits (bounded)
    for the identity. Once the delayed state arrives with the MATCHING study,
    the anchored same-root no-op applies. Pre-fix: RED - "already open" was
    already shown while identity was unknown (a decision made from unknown
    identity).
    """
    live_root = "/stage/" + STUDY_A
    res = _run_handoff_scenario(
        "reopen_identity_pending",
        host="127.0.0.1:9987",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(_bound_entry(STUDY_A)),
        bound_root=live_root,
        delay_initial_state=True,
        bound_identity_timeout_ms=2500,
    )
    pending = res["pending_while_unknown"]
    assert "already open" not in pending["message"], res
    assert pending["invoke_calls"] == 0, res
    assert pending["pending_posts"] == 0, res
    assert pending["cookie_present"] is False, res
    # Identity released as the MATCHING study -> anchored no-op, still nothing
    # invoked or queued. The no-op is asserted on the message LOG because the
    # page's own "Series loaded" status can legitimately overwrite the current
    # message immediately after the delayed state lands.
    assert res["invoke_calls"] == 0, res
    assert res["pending_posts"] == 0, res
    assert res["cookie_present"] is False, res
    assert any("already open" in line for line in res["message_log"]), res


def test_reopen_identity_released_as_different_study_switches() -> None:
    """PR3 review r4310770076 lifecycle control (fail-capable): unknown
    identity must not be treated as a match EITHER once the wait ends.

    The delayed state arrives with a DIFFERENT bound study while the server-
    resolved root equals the entry's destination, so the equal root is NOT the
    open case: the page must invoke the shell exactly once (its prove step
    rejects the mismatch) instead of silently reporting the case open.
    Pre-fix: RED - "already open" at +150 ms and 0 invokes.
    """
    live_root = "/stage/" + STUDY_B
    res = _run_handoff_scenario(
        "reopen_identity_pending",
        host="127.0.0.1:9986",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(_bound_entry(STUDY_B)),
        bound_root=live_root,
        delay_initial_state=True,
        bound_identity_timeout_ms=2500,
    )
    pending = res["pending_while_unknown"]
    assert "already open" not in pending["message"], res
    assert pending["invoke_calls"] == 0, res
    assert pending["pending_posts"] == 0, res
    assert res["invoke_calls"] == 1, res
    assert res["pending_posts"] == 1, res
    assert res["cookie_present"] is True, res


def test_reopen_no_cookie_no_hook_reload_is_noop() -> None:
    """R7 F5 witness (fail-capable): after a plain reload with NO pending
    cookie and NO test hook, the same-root Reopen is still an anchored no-op
    — the live root is the SERVER-reported bound root learned on every
    initialization. Pre-fix code reset the live root on every load and
    learned it only from a pending intent/hook, so this reload invoked the
    shell and killed/respawned the bound backend (invoke_calls == 1)."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "",  # display-only; derivation is authoritative
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9995",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,  # genuinely bound, server-reported
    )
    assert res["bound_root_requests"] >= 1, res
    assert res["hook_calls"] == 0, res
    assert res["invoke_calls"] == 0, res
    assert res["posts"] == 0, res
    assert res["cookie_present"] is False, res


def test_reopen_waits_for_delayed_bound_root_before_deciding() -> None:
    """Review follow-up (app.js Reopen decision): the recent list settles
    independently of ``/api/library/bound-root``, so the button can be
    clicked while the live root is still unknown. The decision must WAIT for
    that initialization to settle instead of deciding from a null root.
    Pre-fix: the early click saw null, skipped the same-root no-op, wrote the
    handoff cookie, queued, and invoked the shell — killing/respawning the
    bound backend — and never re-evaluated after the answer arrived.
    Post-fix: zero queue/invoke/cookie while pending; after release the SAME
    click settles as the anchored same-root no-op message."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "",  # display-only; derivation is authoritative
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "delayed_bound_root",
        host="127.0.0.1:9997",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,  # genuinely bound, server-reported
        delay_bound_root=True,           # ... but answered LATE here
    )
    assert res["bound_root_requests"] >= 1, res
    assert res["invoke_while_pending"] == 0, res
    assert res["pending_while_pending"] == 0, res
    assert res["cookie_present"] is False, res
    assert res["invoke_calls"] == 0, res
    assert res["posts"] == 0, res
    # The decision ran only after the bound-root answer settled, and chose
    # the anchored same-root no-op rather than a shell invoke.
    assert "already open" in res["message_text"], res


def test_reopen_never_settling_bound_root_exits_visibly_without_invoke() -> None:
    """R8 Finding 2 witness: when ``/api/library/bound-root`` never completes
    (stalled connection / never-settling body), the Reopen decision must NOT
    hang indefinitely. It bounds the initialization wait, leaves the current
    backend untouched (0 shell invokes, 0 pending queue writes, 0 cookies),
    exits visibly with an actionable retry message, and resets so subsequent
    clicks can retry. Pre-fix code unconditionally awaited the memoized
    request with no timeout, so every click hung forever without feedback."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "hang_bound_root",
        host="127.0.0.1:9998",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        hang_bound_root=True,  # server never completes the request
        bound_root_timeout_ms=200,
    )
    # Backend was NEVER invoked across either click:
    assert res["first_invoke_calls"] == 0, res
    assert res["first_pending_posts"] == 0, res
    assert res["second_invoke_calls"] == 0, res
    assert res["second_pending_posts"] == 0, res
    assert res["cookie_present"] is False, res
    # Exited visibly with actionable notice:
    assert "timed out" in res["first_message"], res
    assert "Cannot verify active workspace" in res["first_message"], res
    # Actionable retry: second click also produced the visible notice:
    assert "Cannot verify active workspace" in res["final_message"], res


def test_reopen_immediate_headers_never_settling_body_times_out_and_retries_safely() -> None:
    """Supplementary review P2 witness (fail-capable): /api/library/bound-root
    answers 200 headers immediately, but the body stream never settles.

    1. Pre-fix code cleared the timeout after headers before awaiting res.json(),
       hanging boundRootReady and every Reopen indefinitely.
    2. Post-fix code races the COMPLETE fetch + body operation against the
       deadline, aborts the stream on timeout, exits visibly with actionable
       retry notice, and touches nothing (0 invokes, 0 pending posts, no cookie).
    3. Actionable retry (second click) successfully verifies the active workspace
       and completes the Reopen action (same-root no-op when destination matches).
    4. Late completion of attempt 1's stalled body is isolated and CANNOT commit
       or overwrite the live root.
    """
    live_root = "/stage/" + STUDY_A
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "hang_bound_root_body",
        host="127.0.0.1:9998",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        hang_bound_root_body=True,
        bound_root=live_root,
        bound_root_timeout_ms=200,
    )
    # Phase 1: timed out on hung body with zero shell/queue effects
    assert res["first_invoke_calls"] == 0, res
    assert res["first_pending_posts"] == 0, res
    assert res["first_cookie_present"] is False, res
    assert "timed out" in res["first_message"], res
    assert "Cannot verify active workspace" in res["first_message"], res
    assert res["body_aborted"] is True, res

    # Phase 2: actionable retry succeeded and recognized active workspace
    assert res["second_invoke_calls"] == 0, res
    assert res["second_pending_posts"] == 0, res
    assert "already open" in res["second_message"], res
    assert res["root_after_retry"] == live_root, res

    # Phase 3: late body completion on attempt 1 was isolated and did not corrupt live root
    assert res["late_body_committed"] is False, res
    assert res["final_live_root"] == live_root, res


def test_reopen_immediate_headers_hung_body_retry_switches_different_case() -> None:
    """Supplementary review P2 witness (switch leg): when reopening a DIFFERENT
    study while the bound root body hangs, the first click times out safely, and
    the retry click verifies the bound root and executes the switch invoke."""
    live_root = "/stage/" + STUDY_A
    different_entry = {
        "studyUid": STUDY_B,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "hang_bound_root_body",
        host="127.0.0.1:9998",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(different_entry),
        hang_bound_root_body=True,
        bound_root=live_root,
        bound_root_timeout_ms=200,
    )
    # Phase 1: timed out on hung body
    assert res["first_invoke_calls"] == 0, res
    assert res["first_pending_posts"] == 0, res
    assert "timed out" in res["first_message"], res
    assert res["body_aborted"] is True, res

    # Phase 2: actionable retry recognized different study and invoked switch
    assert res["second_invoke_calls"] == 1, res
    assert res["cookie_present"] is True, res

    # Phase 3: late body completion on attempt 1 is isolated
    assert res["late_body_committed"] is False, res


def test_reopen_contradictory_display_destroot_still_switches() -> None:
    """R7 F2 witness (fail-capable): the persisted destRoot is DISPLAY-ONLY.
    An entry whose real stage base points to /stageB while its destRoot field
    claims the live /stage/<A> root must still switch. The live root is seeded
    through the page's own hook so this witness isolates the destRoot-vs-
    derivation decision: pre-fix code compared the display field and no-op'd,
    leaving the wrong root displayed; post-fix only the authoritative
    derivation is consulted."""
    live_root = "/stage/" + STUDY_A
    contradictory = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stageB",     # authoritative derivation: /stageB/<A>
        "destRoot": live_root,         # display-only, contradicts the above
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9996",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(contradictory),
        live_root=live_root,           # genuinely live /stage/<A>, hook-seeded
        bound_root=live_root,          # ... and server-reported
    )
    assert res["bound_root_requests"] >= 1, res
    assert res["invoke_calls"] == 1, res      # B still switches (never suppressed)
    assert res["cookie_present"] is True, res  # handoff queued for the switch


def test_reopen_authoritative_probe_dest_beats_stale_derivation() -> None:
    """R7 F2 corroboration (executable): when the server probe has answered,
    its RESOLVED destination is the no-op authority — a lexical derivation
    mismatch that resolves to the same bound root still no-ops."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage/./",  # lexical derivation: /stage/./<A>
        "destRoot": "",                # display-only
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9997",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_A,  # server-resolved authority
    )
    assert res["invoke_calls"] == 0, res


def test_reopen_waits_for_delayed_entry_probe_before_deciding_same_root() -> None:
    """R11 F2 witness (fail-capable): when bound-root is known but the entry's
    destination probe is still pending, an early click on an entry whose
    stageBaseDir has alternative anchored spelling (/stage/./) must WAIT for the
    authoritative server probe rather than deciding from client lexical
    derivation.

    Pre-fix code: fell back immediately to canonicalDestRoot(entry), which derived
    /stage/./<A> != /stage/<A>, triggering an unwanted switch_study invoke that
    respawned the backend for an already-open case.

    Post-fix code: awaits ensureEntryProbe(); while pending, zero shell invokes,
    zero queue writes, and no cookie. Once probe resolves with /stage/<A>, the
    same click settles as the anchored same-root no-op without respawning.
    """
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage/./",  # equivalent anchored spelling
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "delayed_probe",
        host="127.0.0.1:9995",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_A,  # resolves identically to bound root
        delay_probe=True,
    )
    # While probe was pending: strictly 0 cookies, 0 queue writes, 0 invokes
    assert res["invoke_while_pending"] == 0, res
    assert res["pending_while_pending"] == 0, res
    assert res["cookie_while_pending"] is False, res
    # After probe released: same-root no-op, strictly 0 cookies, 0 queue writes, 0 invokes
    assert res["invoke_calls"] == 0, res
    assert res["pending_posts"] == 0, res
    assert res["cookie_present"] is False, res
    assert "already open" in res["message_text"], res


def test_reopen_resolved_bound_root_with_dot_segments_is_same_root() -> None:
    """R12 P2 frontend witness: when server reports the physically resolved
    bound root for a startup path containing '..', Reopen's destination probe
    matches the resolved root, resulting in the anchored same-root no-op
    without invoking switch_study or respawning."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage",
        "destRoot": "/stage/" + STUDY_A,
        "openedSeq": 1,
        "openCount": 1,
    }
    # Resolved root matches probe destination: 0 invokes, 0 writes, 0 cookies
    res = _run_handoff_scenario(
        "delayed_probe",
        host="127.0.0.1:9995",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_A,
        delay_probe=False,
    )
    assert res["invoke_calls"] == 0, res
    assert res["pending_posts"] == 0, res
    assert res["cookie_present"] is False, res
    assert "already open" in res["message_text"], res

    # Control: unnormalized bootstrap root containing '..' fails lexical
    # match against resolved probe destination and would invoke switch_study.
    res_unnormalized = _run_handoff_scenario(
        "delayed_probe",
        host="127.0.0.1:9995",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/existing/../" + STUDY_A,
        probe_dest_root="/stage/" + STUDY_A,
        delay_probe=False,
    )
    assert res_unnormalized["invoke_calls"] == 1, res_unnormalized
    assert res_unnormalized["pending_posts"] == 1, res_unnormalized
    assert res_unnormalized["cookie_present"] is True, res_unnormalized


def test_reopen_hung_probe_exits_visibly_without_invoke_and_resets() -> None:
    """R11 F2 witness (hung probe timeout & retry): when the entry probe hangs,
    the Reopen click must NOT hang forever and must NOT invoke with unverified
    destination. It bounds the probe wait, exits visibly with an actionable retry
    notice, produces zero side effects (0 invokes, 0 pending writes, no cookie),
    and resets so a subsequent click can retry."""
    active_entry = {
        "studyUid": STUDY_A,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stage/./",
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "hang_probe",
        host="127.0.0.1:9994",
        producer="127.0.0.1:1111",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,
        hang_probe=True,
        probe_timeout_ms=200,
    )
    # First click: timed out safely without effects
    assert res["first_invoke_calls"] == 0, res
    assert res["first_pending_posts"] == 0, res
    assert res["first_cookie_present"] is False, res
    assert "timed out" in res["first_message"], res
    assert "Cannot verify case destination" in res["first_message"], res

    # Actionable retry: second click also exits visibly without side effects
    assert res["second_invoke_calls"] == 0, res
    assert res["second_pending_posts"] == 0, res
    assert res["cookie_present"] is False, res
    assert "Cannot verify case destination" in res["final_message"], res
    assert res["probe_aborted"] is True, res


def test_reopen_probe_different_case_still_switches() -> None:
    """R11 F2 positive control: when the authoritative probe resolves to a
    DIFFERENT destination than the live root, Reopen awaits the probe and
    proceeds with the switch invoke."""
    active_entry = {
        "studyUid": STUDY_B,
        "seriesUid": None,
        "scanSourcePath": "/src",
        "stageBaseDir": "/stageB",
        "destRoot": "",
        "openedSeq": 1,
        "openCount": 1,
    }
    res = _run_handoff_scenario(
        "reopen_same_root",
        host="127.0.0.1:9992",
        producer="127.0.0.1:1111",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        recent_entries=_single_entry_document(active_entry),
        bound_root="/stage/" + STUDY_A,
        probe_dest_root="/stageB/" + STUDY_B,
    )
    assert res["invoke_calls"] == 1, res
    assert res["cookie_present"] is True, res


def test_queue_error_warning_survives_shell_navigation() -> None:
    """R7 F4 witness (fail-capable): the queue POST fails, then the shell
    navigates and THIS page's continuation never runs (the invoke promise
    never settles — context disposed). The warning must survive in the
    surviving destination page, carried by the opaque notice cookie and
    consumed once at startup. Pre-fix code only wrote the warning into the
    outgoing page's message log, so after navigation nobody saw it."""
    res = _run_handoff_scenario(
        "navigate_during_handoff",
        host="127.0.0.1:8892",
        producer="127.0.0.1:8892",
        target_study=STUDY_B,
        bound_study=STUDY_A,
        init_cookie=False,
        queue_fail=True,
    )
    assert res["queue_dispatched"] == 1, res
    assert res["invoke_before_navigation"] == 1, res   # the switch still proceeded
    assert res["notice_before_reload"] is True, res    # armed BEFORE the invoke
    assert res["notice_shown_after_reload"] is True, res
    assert res["notice_cleared_after_reload"] is True, res
    # The notice payload is opaque: version/expiry/fixed code, no entry, no
    # filesystem path, no free text.
    assert res["notice_cookie_shapes"], res
    for shape in res["notice_cookie_shapes"]:
        assert shape["keys"] == "code,exp,v", shape
        assert shape["has_paths"] is False, shape


@pytest.mark.parametrize("stall", ["headers", "body"])
def test_queue_error_notice_delivered_despite_stalled_recent_get(stall: str) -> None:
    """PR3 review r4310770084 (fail-capable): the cross-navigation queue-error
    warning must be delivered INDEPENDENTLY of the recent-list load.

    `loadRecentCases` has no deadline around its fetch or its body consumption,
    so a destination page with an incoming notice and a stalled recent GET
    (headers never arrive, or headers arrive and the body never settles) never
    reached the chained `consumeRecentNotice` — and because the eval-time
    snapshot had ALREADY cleared the cookie, a reload lost the warning
    permanently. Pre-fix: RED - "may not be saved" never appeared on either
    stalled leg. Post-fix the notice is delivered at init (decoupled), it
    survives on the durable banner surface even though the load's synchronous
    banner reset has already run, and the cookie is retired only AFTER delivery,
    so the reload shows nothing twice.
    """
    res = _run_handoff_scenario(
        "notice_with_stalled_recent_get",
        host="127.0.0.1:8884",
        producer="127.0.0.1:8884",
        target_study=STUDY_A,
        bound_study=STUDY_A,
        init_cookie=False,
        stall_recent_get=stall,
    )
    assert res["notice_shown_first_page"] is True, res
    # Durable surface: the warning is on the banner too, so a later unrelated
    # message cannot retire it from view.
    assert res["notice_banner_visible"] is True, res
    # Retired only after the warning was displayed (never before delivery).
    assert res["notice_cookie_after_delivery"] is False, res
    # Delivered exactly once: the reload does not repeat it.
    assert res["notice_shown_after_reload"] is False, res
    # The notice payload stays opaque: version/expiry/fixed code, no paths.
    assert res["notice_cookie_shapes"], res
    for shape in res["notice_cookie_shapes"]:
        assert shape["keys"] == "code,exp,v", shape
        assert shape["has_paths"] is False, shape


@pytest.mark.parametrize("missing_key", ["openedSeq", "openCount"])
def test_persisted_row_missing_counter_degrades_whole_document(
    recent_env: tuple[Path, Path, Path],
    missing_key: str,
) -> None:
    """R7 F3 witness: a persisted row missing openedSeq/openCount is DAMAGED,
    not repairable. The whole document degrades to empty with reason
    corrupt-rows (OWNER DECISION A) instead of being silently repaired with
    defaulted counters (which left it selectable with reason none and could
    reset counts). Counterless INCOMING/pending rows stay accepted (see
    test_pending_producer_shape_without_counters_is_valid)."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": f"strict-{missing_key}"})
    persisted = json.loads(store.read_bytes().decode("utf-8"))
    assert len(persisted["entries"]) == 1
    del persisted["entries"][0][missing_key]
    store.write_bytes(
        json.dumps(persisted, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    loaded = load_recent_cases(store)
    assert loaded["reason"] == "corrupt-rows", loaded
    assert loaded["entries"] == []
    assert loaded["discarded"] == 1


def test_dir_fd_save_path_anchors_parent_identity_before_write(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7 F1 portable leg (stubbed dir_fd; runs on any host): the write path
    samples the parent identity BEFORE opening and passes it as the
    acquisition anchor; an acquisition refusal aborts fail-closed with no
    temp bytes and last-good preserved."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "acq-port-1"})
    good_bytes = store.read_bytes()

    anchors: list[tuple[Any, Any]] = []

    def refusing_open(parent_dir: Path, *, expect_identity: Any = None) -> int:
        anchors.append((parent_dir, expect_identity))
        raise RecentError("target_authority.symlink", "acquisition refusal witness")

    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", refusing_open)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)
    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "acq-port-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert anchors, "acquisition seam was never called"
    _parent_dir, expect_identity = anchors[-1]
    assert expect_identity is not None, "identity must be sampled BEFORE the open"
    assert expect_identity == recent_mod._storage_parent_identity(store.parent)
    assert store.read_bytes() == good_bytes
    assert list(store.parent.glob(".tmp_*")) == []


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX dir_fd handle path (real os.open of a directory)",
)
def test_posix_parent_symlink_swap_at_acquisition_is_refused(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7 F1 POSIX real-swap witness (acquisition window): the approved
    parent is renamed aside and its pathname replaced by a SYMLINK to a
    hostile directory before the acquisition open. O_NOFOLLOW must refuse
    (target_authority.symlink): no bytes may land through the redirect and
    last-good survives. Pre-fix code followed the link and wrote the
    complete payload into the hostile directory."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "acq-pos-1"})
    good_bytes = store.read_bytes()

    parent = store.parent
    moved = parent.with_name(parent.name + "-orig")
    hostile = parent.with_name(parent.name + "-hostile")
    hostile.mkdir(parents=True, exist_ok=True)

    real_open = recent_mod._open_parent_dir

    def swap_then_open(parent_dir: Path, *, expect_identity: Any = None) -> int:
        os.rename(parent, moved)
        os.symlink(hostile, parent, target_is_directory=True)
        try:
            return real_open(parent_dir, expect_identity=expect_identity)
        finally:
            try:
                os.unlink(parent)
            except OSError:
                pass
            try:
                os.rename(moved, parent)
            except OSError:
                pass

    monkeypatch.setattr(recent_mod, "_open_parent_dir", swap_then_open)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)
    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "acq-pos-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert store.read_bytes() == good_bytes
    assert [p for p in hostile.rglob("*") if p.is_file()] == []


def test_parent_open_refusal_translates_enotdir_only_for_a_real_leaf_link(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Review follow-up on F1 (recent.py:156): Linux reports ENOTDIR — not
    ELOOP — for ``os.open(..., O_DIRECTORY | O_NOFOLLOW)`` when the final
    parent component is a symlink. ENOTDIR must normalize to the stable
    ``target_authority.symlink`` refusal only when the final component
    really is a link; an ordinary non-directory, an unobservable
    component, or an unrelated errno must keep the original OSError so
    downstream mapping stays ``recent.store_unwritable``/``store_unreadable``
    and never a link refusal. Portable classification witness (the POSIX
    native-errno leg is covered by the test below)."""
    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"x")
    enotdir = OSError(errno.ENOTDIR, "Not a directory")

    # Native Linux shape: ENOTDIR + final component observed as a link.
    monkeypatch.setattr(recent_mod, "_leaf_link_at", lambda p: True)
    refusal = recent_mod._parent_open_refusal(enotdir, ordinary)
    assert refusal is not None
    assert refusal.code == "target_authority.symlink"

    # Ordinary non-directory: ENOTDIR but no link at the final component.
    monkeypatch.setattr(recent_mod, "_leaf_link_at", lambda p: False)
    assert recent_mod._parent_open_refusal(enotdir, ordinary) is None

    # Observation failure (component vanished): no classification at all.
    assert recent_mod._parent_open_refusal(enotdir, tmp_path / "gone") is None

    # ELOOP: definitively the no-follow leaf-link failure; always refuse.
    loop_refusal = recent_mod._parent_open_refusal(
        OSError(errno.ELOOP, "Too many levels of symbolic links"), ordinary
    )
    assert loop_refusal is not None
    assert loop_refusal.code == "target_authority.symlink"

    # Unrelated errnos are never link refusals.
    for code in (errno.EACCES, errno.ENOENT, errno.EIO, errno.EISDIR):
        assert recent_mod._parent_open_refusal(OSError(code, "x"), ordinary) is None


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX directory open (O_DIRECTORY | O_NOFOLLOW native errno)",
)
def test_posix_open_parent_dir_native_leaf_link_errno_is_refused(
    tmp_path: Path,
) -> None:
    """Native-errno coverage for the review follow-up: through the REAL
    ``os.open`` path (no stubs), a final component that is a symlink to a
    directory must fail the no-follow directory open with this kernel's
    link errno — ENOTDIR on Linux, ELOOP on macOS/BSD — and
    ``_open_parent_dir`` must normalize either one to
    ``target_authority.symlink``. An ordinary regular file at the same
    path raises the plain ENOTDIR OSError and is never misclassified."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform lacks O_DIRECTORY | O_NOFOLLOW")
    target = tmp_path / "realdir"
    target.mkdir()
    link = tmp_path / "linkdir"
    os.symlink(target, link, target_is_directory=True)

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(link, flags)
    except OSError as raw:
        assert raw.errno in (errno.ELOOP, errno.ENOTDIR), raw.errno
    else:
        os.close(fd)
        pytest.fail("no-follow directory open followed the leaf link")

    with pytest.raises(RecentError) as exc_info:
        recent_mod._open_parent_dir(link)
    assert exc_info.value.code == "target_authority.symlink"

    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"x")
    with pytest.raises(OSError) as plain:
        recent_mod._open_parent_dir(ordinary)
    assert not isinstance(plain.value, RecentError)
    assert plain.value.errno == errno.ENOTDIR


def test_acquisition_closes_handle_and_translates_refusal_envelope(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review follow-up on F1 (post-bound re-check): ``check_store_path_safety``
    delegates to ``reject_links``, which raises ``target_authority.Refusal`` —
    NOT ``RecentError``. The acquisition must CLOSE the acquired descriptor on
    that path and translate the refusal WITHOUT losing its stable code/message
    (the frozen error envelope stays RecentError-shaped). Pre-fix code caught
    only ``RecentError``: the refusal escaped and the handle leaked.
    Portable witness (the POSIX real-ancestor-link leg is
    ``test_posix_ancestor_link_before_acquisition_is_refused_by_bound_recheck``,
    which this Windows run skips)."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "acq-env-1"})
    good_bytes = store.read_bytes()

    # Acquisition seam: hand out a fake descriptor and remember that the
    # handle exists, so the refusal check below fires ONLY on the post-bind
    # call (the pre-acquisition call at writer entry must still pass).
    sentinel_fd = 424242
    state = {"opened": False}

    def fake_open(parent_dir: Path, *, expect_identity: Any = None) -> int:
        state["opened"] = True
        return sentinel_fd

    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", fake_open)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)

    real_close = os.close
    closed: list[int] = []

    def recording_close(fd: int) -> None:
        if fd == sentinel_fd:
            closed.append(fd)  # never hand the fake fd to the real close
            return
        real_close(fd)

    monkeypatch.setattr(recent_mod.os, "close", recording_close)

    refusal_message = (
        "recent_store_path contains a symlink or junction component; "
        "refusing fail-closed"
    )

    def refusing_check(path: Path) -> None:
        if state["opened"]:  # the post-bind re-check inside acquisition
            raise Refusal("target_authority.symlink", refusal_message)

    monkeypatch.setattr(recent_mod, "check_store_path_safety", refusing_check)

    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "acq-env-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert exc_info.value.message == refusal_message  # stable code/message intact
    assert closed == [sentinel_fd], "the acquired handle was not closed"
    assert store.read_bytes() == good_bytes
    assert list(store.parent.glob(".tmp_*")) == []


def test_prereplace_refusal_cleans_resources_and_translates_envelope(
    recent_env: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8 Finding 1 portable witness: after acquisition and temp creation, the
    real pre-replace ``check_store_path_safety()`` in ``_atomic_replace`` can
    still raise ``target_authority.Refusal``. All three pinned writers (main
    store, pending slot, and sidecar) must:
    1. Translate the refusal to ``RecentError`` without losing stable code/msg;
    2. Close the acquired parent descriptor;
    3. Unlink the temp file relative to that handle.
    Pre-fix code caught only RecentError/OSError in the caller write loops, so
    the Refusal escaped (breaking HTTP envelope on pending) while leaking the
    descriptor and leaving temp documents behind."""
    src, stage, store = recent_env
    entry = _fresh_entry(src, stage)

    sentinel_fd = 555555
    closed: list[int] = []
    unlinked: list[str] = []

    real_close = os.close
    real_unlink = os.unlink
    real_persist = recent_mod.persist_durable_high_water_seq

    def recording_close(fd: int) -> None:
        if fd == sentinel_fd:
            closed.append(fd)
            return
        real_close(fd)

    def recording_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        name = str(getattr(path, "name", path))
        unlinked.append(name)
        if kwargs.get("dir_fd") == sentinel_fd:
            return  # do not call real unlink with fake fd
        try:
            real_unlink(path, *args, **kwargs)
        except OSError:
            pass

    monkeypatch.setattr(recent_mod.os, "close", recording_close)
    monkeypatch.setattr(recent_mod.os, "unlink", recording_unlink)
    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", lambda *a, **k: sentinel_fd)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)

    refusal_msg = "recent_store_path contains a symlink or junction component; refusing fail-closed"

    # --- 1. Main store writer (save_recent_cases / process_recent_post) ---
    temp_created = {"name": None}

    def intercept_temp(parent_fd: int, temp_name: str, flags: int, mode: int) -> int:
        temp_created["name"] = temp_name
        real_tmp = store.parent / temp_name
        return os.open(real_tmp, flags, mode)

    monkeypatch.setattr(recent_mod, "_create_temp_in_parent", intercept_temp)

    # Pre-replace check raises Refusal AFTER temp creation:
    def prereplace_refusal(path: Path) -> None:
        if temp_created["name"] is not None:
            raise Refusal("target_authority.symlink", refusal_msg)

    monkeypatch.setattr(recent_mod, "check_store_path_safety", prereplace_refusal)

    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "post-acq-1"})
    assert exc_info.value.code == "target_authority.symlink"
    assert exc_info.value.message == refusal_msg
    assert sentinel_fd in closed, "main writer did not close parent handle"
    assert temp_created["name"] in unlinked, "main writer did not unlink temp file"

    # --- 2. Pending slot writer (save_pending_open) ---
    closed.clear()
    unlinked.clear()
    temp_created["name"] = None
    with pytest.raises(RecentError) as exc_info2:
        recent_mod.save_pending_open(
            store,
            {"id": "poTest1", "exp": int(time.time() * 1000) + 60000, "entry": entry},
        )
    assert exc_info2.value.code == "target_authority.symlink"
    assert exc_info2.value.message == refusal_msg
    assert sentinel_fd in closed, "pending writer did not close parent handle"
    assert temp_created["name"] in unlinked, "pending writer did not unlink temp file"

    # --- 3. High-water sidecar writer (persist_durable_high_water_seq) ---
    closed.clear()
    unlinked.clear()
    temp_created["name"] = None
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", real_persist)
    monkeypatch.setattr(recent_mod.os, "close", recording_close)
    monkeypatch.setattr(recent_mod.os, "unlink", recording_unlink)
    monkeypatch.setattr(recent_mod, "_dir_fd_replace_supported", lambda: True)
    monkeypatch.setattr(recent_mod, "_open_parent_dir", lambda *a, **k: sentinel_fd)
    monkeypatch.setattr(recent_mod, "_create_temp_in_parent", intercept_temp)
    monkeypatch.setattr(recent_mod, "check_store_path_safety", prereplace_refusal)

    with pytest.raises(RecentError) as exc_info3:
        recent_mod.persist_durable_high_water_seq(store, 42)
    assert exc_info3.value.code == "target_authority.symlink"
    assert exc_info3.value.message == refusal_msg
    assert sentinel_fd in closed, "sidecar writer did not close parent handle"
    assert temp_created["name"] in unlinked, "sidecar writer did not unlink temp file"


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX dir_fd handle path (real os.open of a directory)",
)
def test_posix_parent_ancestry_redirect_at_acquisition_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7 F1 POSIX real-swap witness (identity anchor): the ANCESTOR of the
    approved parent is renamed aside and a fresh REAL directory takes its
    pathname — no link anywhere, so link checks cannot help — strictly after
    the validation-time identity sample and before the open. The opened
    directory is a different object, so the identity anchor must refuse
    (target_authority.symlink); no bytes may land in the redirected tree."""
    src = tmp_path / "scan"
    stage = tmp_path / "stage"
    src.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "deep" / "anc" / "store"
    parent.mkdir(parents=True, exist_ok=True)
    store = parent / "recent_cases.json"
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "acq-anc-1"})
    good_bytes = store.read_bytes()

    anc = parent.parent
    moved = anc.with_name(anc.name + "-orig")
    real_open = recent_mod._open_parent_dir

    def redirect_then_open(parent_dir: Path, *, expect_identity: Any = None) -> int:
        os.rename(anc, moved)
        os.makedirs(parent_dir, exist_ok=True)
        try:
            return real_open(parent_dir, expect_identity=expect_identity)
        finally:
            shutil.rmtree(anc, ignore_errors=True)
            os.rename(moved, anc)

    monkeypatch.setattr(recent_mod, "_open_parent_dir", redirect_then_open)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)
    pre_attempt_files = sorted(p.name for p in parent.glob("*"))
    assert store.name in pre_attempt_files, "store file must exist before second attempt"
    assert (store.name + ".seq") in pre_attempt_files, "legitimate .seq sidecar must exist"
    assert pre_attempt_files == [store.name, store.name + ".seq"], (
        "pre-attempt parent must contain only store and sidecar"
    )

    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "acq-anc-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert store.read_bytes() == good_bytes
    assert list(parent.glob(".tmp_*")) == []
    # R11 review fix: compare against pre-attempt directory snapshot (which
    # legitimately contains recent_cases.json and recent_cases.json.seq created
    # during initial setup before persist_durable_high_water_seq was patched).
    # Verifies both legitimate files are preserved and no unexpected/temp files leak.
    assert sorted(p.name for p in parent.glob("*")) == pre_attempt_files


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX dir_fd handle path (real os.open of a directory)",
)
def test_posix_ancestor_link_before_acquisition_is_refused_by_bound_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7 F1 POSIX real-swap witness (post-bind re-check leg): an ANCESTOR is
    replaced by a symlink to a twin tree BEFORE the identity sample, so the
    sample and the opened handle AGREE on the redirected directory (an
    identity anchor alone cannot see it). The post-bind path-safety re-check
    must still refuse the link (target_authority.symlink), and no bytes may
    land in the twin tree."""
    src = tmp_path / "scan"
    stage = tmp_path / "stage"
    src.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "deep2"
    anc = base / "anc"
    parent = anc / "store"
    parent.mkdir(parents=True, exist_ok=True)
    twin_parent = base / "twin" / "store"
    twin_parent.mkdir(parents=True, exist_ok=True)
    store = parent / "recent_cases.json"
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "acq-link-1"})
    good_bytes = store.read_bytes()

    real_acquire = recent_mod._acquire_verified_parent

    def link_then_acquire(parent_dir: Path, store_path: Path) -> int:
        moved = base / "anc-orig"
        os.rename(anc, moved)
        os.symlink(base / "twin", anc, target_is_directory=True)
        try:
            return real_acquire(parent_dir, store_path)
        finally:
            try:
                os.unlink(anc)
            except OSError:
                pass
            os.rename(moved, anc)

    monkeypatch.setattr(recent_mod, "_acquire_verified_parent", link_then_acquire)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)
    with pytest.raises(RecentError) as exc_info:
        process_recent_post(store, {"entry": entry, "open_id": "acq-link-2"})
    assert exc_info.value.code == "target_authority.symlink"
    assert store.read_bytes() == good_bytes
    assert [p for p in twin_parent.rglob("*") if p.is_file()] == []


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX dir_fd handle path (real os.open of a directory)",
)
def test_posix_ancestor_link_after_acquisition_is_refused_by_prereplace_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8 Finding 1 POSIX real-swap witness: after parent acquisition, the
    ancestor directory is renamed aside and replaced by a symlink during temp
    creation (strictly POST-acquisition). The real pre-replace
    ``check_store_path_safety()`` in ``_atomic_replace`` encounters the real
    link and must refuse (target_authority.symlink). Resource cleanup must
    unlink the temp document from the pinned parent and close the descriptor.
    No bytes may land in the twin tree."""
    src = tmp_path / "scan"
    stage = tmp_path / "stage"
    src.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "deep3"
    anc = base / "anc"
    parent = anc / "store"
    parent.mkdir(parents=True, exist_ok=True)
    twin_parent = base / "twin" / "store"
    twin_parent.mkdir(parents=True, exist_ok=True)
    store = parent / "recent_cases.json"
    entry = _fresh_entry(src, stage)
    process_recent_post(store, {"entry": entry, "open_id": "post-link-1"})
    good_bytes = store.read_bytes()

    real_create_temp = recent_mod._create_temp_in_parent

    def link_during_temp(parent_fd: int, temp_name: str, flags: int, mode: int) -> int:
        fd = real_create_temp(parent_fd, temp_name, flags, mode)
        # Strictly POST-acquisition: swap the ancestor for a symlink to twin
        moved = base / "anc-orig"
        os.rename(anc, moved)
        os.symlink(base / "twin", anc, target_is_directory=True)
        return fd

    monkeypatch.setattr(recent_mod, "_create_temp_in_parent", link_during_temp)
    monkeypatch.setattr(recent_mod, "persist_durable_high_water_seq", lambda *a, **k: None)

    try:
        with pytest.raises(RecentError) as exc_info:
            process_recent_post(store, {"entry": entry, "open_id": "post-link-2"})
        assert exc_info.value.code == "target_authority.symlink"
        # The link is STILL in place, so `store` resolves through it; the
        # untouched original bytes live in the renamed-aside directory.
        original_store = base / "anc-orig" / "store" / "recent_cases.json"
        assert original_store.read_bytes() == good_bytes
        assert store.is_symlink() or anc.is_symlink()
        # No temp document leaked in the original (pinned) directory...
        assert [p for p in (base / "anc-orig" / "store").glob(".tmp_*")] == []
        # ... and none in the twin tree either:
        assert [p for p in twin_parent.rglob("*") if p.is_file()] == []
    finally:
        try:
            os.unlink(anc)
        except OSError:
            pass
        moved = base / "anc-orig"
        if moved.exists():
            os.rename(moved, anc)
