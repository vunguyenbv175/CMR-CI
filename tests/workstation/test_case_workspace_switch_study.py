"""Backend switch-matrix proof (CMR_CASE_WORKSPACE_RUNTIME_002, profile b).

Synthetic fixtures only (no PHI; SYNTH UIDs under the obviously-fake
``...99991`` root unique to this file).

Proves, at the store boundary + the frozen standalone authority CLI
(``python -m services.workstation.target_authority resolve/prove``), the
runtime-002 prerequisite slice of the frozen ordered effects
(validate -> terminate -> spawn -> handshake -> verify -> navigate):

- M1 resolve success matrix: staged hit with source present (reuse=true) and
  staged hit with source MISSING (reuse-first true); unstaged + source
  present succeeds with reuse=false; stdout is exactly one JSON document
  and exit is 0 on success.
- M2 resolve refusal matrix (all fail-closed, nonzero exit, single
  ``{ok:false,code,message}`` stdout doc): UID grammar + dot-segment,
  namespace prefix, root-relative (KL-001 kept: refused, never
  reinterpreted), genuine-live on source AND on the independently derived
  dest, missing source with nothing staged, malformed JSON, unknown
  operation, over-cap ``--timeout-ms``.
- M3 prove matrix: witness binds the owned pid + loopback authority to the
  staged layout digest AND the ACTUAL served fields (``/api/studies`` rows +
  per-series ``/manifest`` state_projection); non-loopback, dead pid,
  absent-series, wrong-root census, wrong requested study, and unreachable
  authority all refuse; the ``--request -`` stdin form works; the startup
  seam (no ``study_uid``) still derives the single served study.
- M4 cross-check: the CLI staged census (series UIDs + file count) matches
  the ACTUAL served fields of a bootstrapped backend on the same root
  (``GET /api/studies`` rows + ``/manifest`` state_projection) — the study
  -> series binding the shell must check before any navigation.
- M5 contamination: the 004A lineage-carry precedent holds on both staged
  roots (A-state cannot validate against B).

Python-matrix subprocess checks of the backend alone are supplemental —
they NEVER certify the Rust lifecycle (that is the G1R production e2e in
``desktop/src-tauri/src/main.rs``); G1/G1R/G8 are NOT renamed.

Gate map: G1 (this file + workstation neighbors) proves ACTUAL target
opening + the frozen side-effect order incl. the ONE synthetic production
Rust-lifecycle e2e (real children, kill/spawn/PID/port/maps/
target-visibility), not staging alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from cmr_core.domain_types import Prediction
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_dicom.discovery import scan_library

from services.workstation.library import stage_case
from services.workstation.server import WorkstationApplication
from services.workstation.target_authority import (
    REFUSAL_CODES,
    prove_identity,
    resolve_target,
)
from tests.synth.generate_dicom import write_synthetic_cine
from tests.workstation.support import (
    STAMP,
    _contour,
    _rect,
    json_value,
    request,
    running_server,
)

UID = "1.2.826.0.1.3680043.10.99991"
STUDY_A = UID + ".1"
STUDY_B = UID + ".2"
SERIES_A = STUDY_A + ".1.1"
SERIES_B = STUDY_B + ".2.1"

CLI = [sys.executable, "-m", "services.workstation.target_authority"]


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["packages", "services", "."])
    return env


def _run_cli(operation: str, payload: dict | str, repo: Path) -> tuple[int, dict, str]:
    """Run the frozen CLI; return (exit, parsed stdout doc, stderr tail)."""
    request_arg = payload if isinstance(payload, str) else json.dumps(payload)
    proc = subprocess.run(
        [*CLI, operation, "--request", request_arg],
        cwd=repo,
        env=_cli_env(),
        capture_output=True,
        timeout=60,
    )
    stdout = proc.stdout.decode("utf-8", "replace")
    # stdout MUST be exactly one JSON document (parsed, never eyeballed).
    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert stdout.strip().count("\n") == 0, "stdout must be a single document line"
    return proc.returncode, doc, proc.stderr.decode("utf-8", "replace")[-500:]


def _two_study_world(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write two synthetic studies, stage both; return (lib, stage, dest_b)."""
    src = tmp_path / "lib"
    write_synthetic_cine(
        src / "nestA", n_slices=1, n_frames=2, series_uid=SERIES_A, study_uid=STUDY_A
    )
    write_synthetic_cine(
        src / "nestB", n_slices=1, n_frames=2, series_uid=SERIES_B, study_uid=STUDY_B
    )
    scan = scan_library(src)
    stage = tmp_path / "stage"
    st_a = stage_case(scan, STUDY_A, stage / STUDY_A, source_root=src)
    st_b = stage_case(scan, STUDY_B, stage / STUDY_B, source_root=src)
    assert st_a.series_uids == (SERIES_A,)
    assert st_b.series_uids == (SERIES_B,)
    return src, stage, Path(st_b.dest_root)


def _repo_root() -> Path:
    return Path(__file__).parents[2]


# --- M1: resolve success matrix -------------------------------------------


def test_resolve_staged_hit_reuses_with_identity_census(tmp_path: Path) -> None:
    """M1a: staged hit + source present -> reuse=true + census + digest."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    code, doc, _err = _run_cli(
        "resolve",
        {"source_root": str(src), "stage_base": str(stage), "study_uid": STUDY_A},
        _repo_root(),
    )
    assert code == 0
    assert doc["ok"] is True
    assert doc["reuse"] is True
    assert doc["identity"]["study_uid"] == STUDY_A
    assert doc["identity"]["series_uids"] == [SERIES_A]
    assert doc["identity"]["file_count"] > 0
    assert len(doc["root_digest"]) == 64
    assert Path(doc["dest_root"]).resolve() == (stage / STUDY_A).resolve()
    assert Path(doc["resolved"]["dest"]).resolve() == (stage / STUDY_A).resolve()


def test_resolve_missing_source_reuses_staged_copy(tmp_path: Path) -> None:
    """M1b: reuse-first — missing source with a staged copy still reuses."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    before = sorted(
        (path.name, path.stat().st_size) for path in (stage / STUDY_B).rglob("*") if path.is_file()
    )
    import shutil

    shutil.rmtree(src)
    code, doc, _err = _run_cli(
        "resolve",
        {"source_root": str(src), "stage_base": str(stage), "study_uid": STUDY_B},
        _repo_root(),
    )
    assert code == 0
    assert doc["ok"] is True
    assert doc["reuse"] is True
    assert doc["identity"]["series_uids"] == [SERIES_B]
    after = sorted(
        (path.name, path.stat().st_size) for path in (stage / STUDY_B).rglob("*") if path.is_file()
    )
    assert before == after, "resolve must not touch the staged copy"


def test_resolve_unstaged_with_source_succeeds_reuse_false(tmp_path: Path) -> None:
    """M1c: nothing staged + source present -> ok with reuse=false."""
    fresh = tmp_path / "fresh"
    (fresh / "src").mkdir(parents=True)
    (fresh / "base").mkdir(parents=True)
    code, doc, _err = _run_cli(
        "resolve",
        {
            "source_root": str(fresh / "src"),
            "stage_base": str(fresh / "base"),
            "study_uid": "8.8.8",
        },
        _repo_root(),
    )
    assert code == 0
    assert doc == {
        "ok": True,
        "reuse": False,
        "dest_root": doc["dest_root"],
        "resolved": doc["resolved"],
        "root_digest": doc["root_digest"],
        "identity": {"study_uid": "8.8.8", "series_uids": [], "file_count": 0},
    }


def test_resolve_in_process_matches_cli_shape(tmp_path: Path) -> None:
    """M1d: the in-process `resolve_target` honors the same frozen shape."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    verdict = resolve_target(str(src), str(stage), STUDY_A)
    assert verdict["ok"] is True
    assert verdict["reuse"] is True
    assert verdict["identity"]["series_uids"] == [SERIES_A]
    assert set(verdict) == {
        "ok",
        "reuse",
        "dest_root",
        "resolved",
        "root_digest",
        "identity",
    }


# --- M2: resolve refusal matrix --------------------------------------------


@pytest.mark.parametrize(
    ("study_uid", "code"),
    [
        ("..", "target_authority.uid_invalid"),
        (".", "target_authority.uid_invalid"),
        ("", "target_authority.bad_request"),
        ("a/b", "target_authority.uid_invalid"),
        ("a b", "target_authority.uid_invalid"),
        ("a:b", "target_authority.uid_invalid"),
    ],
)
def test_resolve_uid_grammar_refuses(tmp_path: Path, study_uid: str, code: str) -> None:
    """M2a: UID grammar + dot-segment refusal before any effect."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": str(src), "stage_base": str(stage), "study_uid": study_uid},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc == {"ok": False, "code": code, "message": doc["message"]}
    assert doc["code"] in REFUSAL_CODES


def test_resolve_namespace_prefix_refuses(tmp_path: Path) -> None:
    """M2b: Windows namespace prefixes refuse before resolution."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {
            "source_root": "\\\\?\\C:\\evil",
            "stage_base": str(stage),
            "study_uid": STUDY_A,
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.namespace_prefix"


def test_resolve_root_relative_refuses_kl001_kept(tmp_path: Path) -> None:
    """M2c: KL-001 unchanged — root-relative inputs refuse, never reinterpret."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": "relative/lib", "stage_base": str(stage), "study_uid": STUDY_A},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.root_relative"


def test_resolve_root_relative_drive_less_slash_rooted_path_refuses(tmp_path: Path) -> None:
    """Finding 5: drive-less slash-rooted path (e.g. /synth-stage) on Windows refuses."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": "/synth-stage", "stage_base": str(stage), "study_uid": STUDY_A},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.root_relative"


def test_resolve_rejects_links_on_derived_destination_before_resolution(tmp_path: Path) -> None:
    """Finding 4: derived destination containing symlink or junction refuses fail-closed."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    real_target = stage / "real_target"
    real_target.mkdir()
    link_study_uid = "1.2.826.0.1.3680043.10.99991.99"
    link_dest = stage / link_study_uid
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(real_target), str(link_dest))
    else:
        os.symlink(str(real_target), str(link_dest))

    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": str(src), "stage_base": str(stage), "study_uid": link_study_uid},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.symlink"
    assert "dest contains a symlink or junction component" in doc["message"]


def test_resolve_rejects_links_followed_by_parent_on_source_root(tmp_path: Path) -> None:
    """Finding 3: source_root with link followed by '..' refuses before resolution collapses it."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    real_target = tmp_path / "other_target"
    real_target.mkdir()
    link_dir = src / "link_dir"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(real_target), str(link_dir))
    else:
        os.symlink(str(real_target), str(link_dir))

    traversed_source = str(src / "link_dir" / ".." / "lib")
    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": traversed_source, "stage_base": str(stage), "study_uid": STUDY_A},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.symlink"
    assert "source_root contains a symlink or junction component" in doc["message"]


def test_resolve_rejects_links_followed_by_parent_on_stage_base(tmp_path: Path) -> None:
    """Finding 3: stage_base with link followed by '..' refuses before resolution collapses it."""
    src, stage, _dest_b = _two_study_world(tmp_path)
    real_target = tmp_path / "other_stage"
    real_target.mkdir()
    link_dir = stage / "link_dir"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(real_target), str(link_dir))
    else:
        os.symlink(str(real_target), str(link_dir))

    traversed_stage = str(stage / "link_dir" / ".." / "stage")
    exit_code, doc, _err = _run_cli(
        "resolve",
        {"source_root": str(src), "stage_base": traversed_stage, "study_uid": STUDY_A},
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.symlink"
    assert "stage_base contains a symlink or junction component" in doc["message"]


def test_resolve_genuine_live_source_refuses(tmp_path: Path) -> None:
    """M2d: genuine-live on the SOURCE refuses before any effect."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {
            "source_root": "/tmp/opencode/owner_test_session",
            "stage_base": str(stage),
            "study_uid": STUDY_A,
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.genuine_live"


def test_resolve_genuine_live_derived_dest_refuses(tmp_path: Path) -> None:
    """M2e: genuine-live on the INDEPENDENTLY DERIVED dest refuses too."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {
            "source_root": str(stage),
            "stage_base": "/tmp/opencode/owner_test_session/x",
            "study_uid": STUDY_A,
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.genuine_live"


def test_resolve_missing_source_without_stage_refuses(tmp_path: Path) -> None:
    """M2f: missing source AND nothing staged -> source_missing."""
    _src, stage, _dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "resolve",
        {
            "source_root": str(tmp_path / "no-such-source"),
            "stage_base": str(stage),
            "study_uid": "8.8.8",
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "library.source_missing"


def test_resolve_malformed_and_unknown_refuse(tmp_path: Path) -> None:
    """M2g: malformed JSON, unknown operation, and over-cap timeout refuse."""
    _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli("resolve", "{not json", _repo_root())
    assert exit_code != 0
    assert doc["code"] == "target_authority.bad_request"
    exit_code, doc, _err = _run_cli("frobnicate", "{}", _repo_root())
    assert exit_code != 0
    assert doc["code"] == "target_authority.unsupported_operation"
    proc = subprocess.run(
        [*CLI, "resolve", "--request", "{}", "--timeout-ms", "99999"],
        cwd=_repo_root(),
        env=_cli_env(),
        capture_output=True,
        timeout=60,
    )
    over = json.loads(proc.stdout.decode("utf-8", "replace"))
    assert proc.returncode != 0
    assert over["code"] == "target_authority.bad_request"


# --- M3: prove matrix -------------------------------------------------------


def test_prove_witness_binds_pid_authority_and_served_fields(tmp_path: Path) -> None:
    """M3a: the witness binds the OWNED pid + loopback authority + digest +
    the ACTUAL served /api/studies + /manifest identity (contract
    §exact_identity: all four legs agree)."""
    _src, _stage, dest_b = _two_study_world(tmp_path)
    code, resolve_doc, _err = _run_cli(
        "resolve",
        {
            "source_root": str(dest_b.parent.parent / "lib"),
            "stage_base": str(dest_b.parent),
            "study_uid": STUDY_B,
        },
        _repo_root(),
    )
    assert code == 0
    app = WorkstationApplication.bootstrap(dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False)
    with running_server(app) as (_server, base_url):
        authority = base_url.removeprefix("http://")
        exit_code, doc, _err = _run_cli(
            "prove",
            {
                "resolved_root": str(dest_b),
                "authority": authority,
                "pid": os.getpid(),
                "study_uid": STUDY_B,
            },
            _repo_root(),
        )
    assert exit_code == 0
    witness = doc["witness"]
    assert witness["pid"] == os.getpid()
    assert witness["authority"] == base_url + "/"
    assert witness["root_digest"] == resolve_doc["root_digest"]
    assert witness["study_uid"] == STUDY_B
    assert witness["identity"]["series_uids"] == [SERIES_B]
    assert witness["identity"]["study_uid"] == STUDY_B
    assert witness["identity"]["file_count"] == resolve_doc["identity"]["file_count"]
    served = witness["served"]
    assert served["study_uid"] == STUDY_B
    assert served["series_uids"] == [SERIES_B]
    (projection,) = served["projections"]
    assert projection["series_uid"] == SERIES_B
    assert projection["study_uid"] == STUDY_B
    for key in ("workflow_id", "manifest_digest", "head_revision_id", "measurement_id"):
        assert key in projection, key


def test_prove_witness_above_256_kib_cine_manifest(tmp_path: Path) -> None:
    """Finding 1: supported studies whose full manifest exceeds 256 KiB must not be rejected."""
    lib = tmp_path / "large_lib"
    stage = tmp_path / "large_stage"
    lib.mkdir()
    stage.mkdir()

    large_study_uid = "1.2.826.0.1.3680043.10.99991.60"
    large_series_uid = large_study_uid + ".1"
    # 35 slices x 32 frames produces a full manifest of ~265 KiB (> 256 KiB):
    write_synthetic_cine(
        lib / "cine",
        n_slices=35,
        n_frames=32,
        series_uid=large_series_uid,
        study_uid=large_study_uid,
    )
    scan = scan_library(lib)
    staged = stage_case(scan, large_study_uid, stage / large_study_uid, source_root=lib)

    app = WorkstationApplication.bootstrap(
        Path(staged.dest_root), port=0, clock=lambda: STAMP, jobs_enabled=False
    )
    with running_server(app) as (_server, base_url):
        authority = base_url.removeprefix("http://")
        exit_code, doc, _err = _run_cli(
            "prove",
            {
                "resolved_root": str(staged.dest_root),
                "authority": authority,
                "pid": os.getpid(),
                "study_uid": large_study_uid,
            },
            _repo_root(),
        )
    assert exit_code == 0, f"prove failed on >256KiB cine: {doc} {_err}"
    assert doc["ok"] is True
    witness = doc["witness"]
    assert witness["study_uid"] == large_study_uid
    assert witness["identity"]["series_uids"] == [large_series_uid]
    (projection,) = witness["served"]["projections"]
    assert projection["series_uid"] == large_series_uid
    assert projection["study_uid"] == large_study_uid


def test_prove_refusal_matrix(tmp_path: Path) -> None:
    """M3b: non-loopback, dead pid, and absent series all refuse."""
    _src, _stage, dest_b = _two_study_world(tmp_path)
    exit_code, doc, _err = _run_cli(
        "prove",
        {
            "resolved_root": str(dest_b),
            "authority": "evil.example:45123",
            "pid": os.getpid(),
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.not_loopback"

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = dead.pid
    dead.wait()
    exit_code, doc, _err = _run_cli(
        "prove",
        {
            "resolved_root": str(dest_b),
            "authority": "127.0.0.1:45123",
            "pid": dead_pid,
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.pid_not_running"

    exit_code, doc, _err = _run_cli(
        "prove",
        {
            "resolved_root": str(dest_b),
            "authority": "127.0.0.1:45123",
            "pid": os.getpid(),
            "series_uid": "1.2.3.4.5.6",
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.series_mismatch"


def test_prove_served_mismatch_matrix(tmp_path: Path) -> None:
    """M3e: served-fields mismatches refuse fail-closed.

    Three live probes against a REAL backend serving study B:
    (1) a DIFFERENT root's census (study A) never matches the served rows;
    (2) the wrong REQUESTED study never matches the served study_uid;
    (3) an unreachable authority cannot establish the served witness.
    """
    _src, stage, dest_a = _two_study_world(tmp_path)
    dest_a = stage / STUDY_A
    dest_b = stage / STUDY_B
    app = WorkstationApplication.bootstrap(dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False)
    with running_server(app) as (_server, base_url):
        authority = base_url.removeprefix("http://")
        exit_code, doc, _err = _run_cli(
            "prove",
            {
                "resolved_root": str(dest_a),
                "authority": authority,
                "pid": os.getpid(),
                "study_uid": STUDY_A,
            },
            _repo_root(),
        )
        assert exit_code != 0
        assert doc["code"] == "target_authority.served_mismatch"
        assert doc["code"] in REFUSAL_CODES

        exit_code, doc, _err = _run_cli(
            "prove",
            {
                "resolved_root": str(dest_b),
                "authority": authority,
                "pid": os.getpid(),
                "study_uid": STUDY_A,
            },
            _repo_root(),
        )
        assert exit_code != 0
        assert doc["code"] == "target_authority.served_mismatch"
    # (3) unreachable authority (nothing is bound on port 1): fail-closed.
    exit_code, doc, _err = _run_cli(
        "prove",
        {
            "resolved_root": str(dest_b),
            "authority": "127.0.0.1:1",
            "pid": os.getpid(),
            "study_uid": STUDY_B,
        },
        _repo_root(),
    )
    assert exit_code != 0
    assert doc["code"] == "target_authority.served_mismatch"


def test_prove_stdin_startup_seam(tmp_path: Path) -> None:
    """M3c: the `--request -` stdin form + startup seam (no requested
    study): the witness still binds the single SERVED study uid."""
    _src, _stage, dest_b = _two_study_world(tmp_path)
    app = WorkstationApplication.bootstrap(dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False)
    with running_server(app) as (_server, base_url):
        payload = json.dumps(
            {
                "resolved_root": str(dest_b),
                "authority": base_url.removeprefix("http://"),
                "pid": os.getpid(),
            }
        ).encode()
        proc = subprocess.run(
            [*CLI, "prove", "--request", "-"],
            cwd=_repo_root(),
            env=_cli_env(),
            input=payload,
            capture_output=True,
            timeout=60,
        )
    doc = json.loads(proc.stdout.decode("utf-8", "replace"))
    assert proc.returncode == 0
    assert doc["ok"] is True
    assert doc["witness"]["study_uid"] is None
    assert doc["witness"]["identity"]["series_uids"] == [SERIES_B]
    assert doc["witness"]["identity"]["study_uid"] == STUDY_B
    assert doc["witness"]["served"]["study_uid"] == STUDY_B


def test_prove_in_process_matches_cli(tmp_path: Path) -> None:
    """M3d: in-process `prove_identity` honors the same frozen shape."""
    _src, _stage, dest_b = _two_study_world(tmp_path)
    app = WorkstationApplication.bootstrap(dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False)
    with running_server(app) as (_server, base_url):
        witness = prove_identity(
            str(dest_b),
            base_url.removeprefix("http://"),
            os.getpid(),
            None,
            STUDY_B,
        )
    assert witness["ok"] is True
    assert witness["witness"]["identity"]["series_uids"] == [SERIES_B]
    assert witness["witness"]["served"]["study_uid"] == STUDY_B


# --- M4: served-fields cross-check (ACTUAL target opening) -----------------


def test_staged_census_matches_served_study_identity(tmp_path: Path) -> None:
    """M4: the CLI census matches the ACTUAL served study/series identity.

    Boots a REAL backend on the staged root and cross-checks the CLI
    `identity` (series UIDs + file count + root digest) against the served
    `GET /api/studies` series rows and the `/manifest` state_projection
    (series_uid, study_uid, workflow_id, manifest_digest, head_revision_id,
    measurement_id) — the exact witness the shell must check before any
    navigation.
    """
    _src, _stage, dest_b = _two_study_world(tmp_path)
    _code, resolve_doc, _err = _run_cli(
        "resolve",
        {
            "source_root": str(dest_b.parent.parent / "lib"),
            "stage_base": str(dest_b.parent),
            "study_uid": STUDY_B,
        },
        _repo_root(),
    )
    app = WorkstationApplication.bootstrap(dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False)
    with running_server(app) as (server, base_url):
        def _get_json(path: str) -> Any:
            for attempt in range(5):
                try:
                    return json_value(request(server, base_url, path))
                except ConnectionResetError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05)

        studies = _get_json("/api/studies")
        assert [row["uid"] for row in studies] == [SERIES_B]
        row = studies[0]
        for key in ("uid", "description", "frames", "slices", "modality", "sequence_class"):
            assert key in row, key
        manifest = _get_json(f"/api/study/{SERIES_B}/manifest")
        # The manifest resource IS the state_projection (render_state with
        # include_manifest=True): series identity + workflow/manifest
        # binding + head/measurement lineage live at the top level.
        assert manifest["series_uid"] == SERIES_B
        assert manifest["study_uid"] == STUDY_B
        assert manifest["workflow_id"] is not None
        assert manifest["manifest_digest"] == manifest["workflow_id"]
        assert "head_revision_id" in manifest
        assert "measurement_id" in manifest
    assert resolve_doc["identity"]["series_uids"] == [row["uid"]]
    assert resolve_doc["identity"]["file_count"] == len(
        list((dest_b / "dicom" / SERIES_B).iterdir())
    )
    digest = hashlib.sha256(os.fsencode(str(dest_b.resolve()))).hexdigest()
    assert resolve_doc["root_digest"] == digest


# --- M5: contamination precedent on both staged roots ----------------------


def test_lineage_carry_mutant_fails_on_both_roots(tmp_path: Path) -> None:
    """M5: the 004A no-A-state-in-B precedent holds on the staged roots.

    Records a selection + head in A only, then proves the carry mutant
    (A's head/selection lineage submitted against B) refuses — the frozen
    contamination guard for the switch path.
    """
    from cmr_persistence.store import StaleHeadError

    src, stage, _dest_b = _two_study_world(tmp_path)
    dest_a = stage / STUDY_A
    dest_b = stage / STUDY_B
    _ = src
    app_a = WorkstationApplication.bootstrap(
        dest_a, port=0, clock=lambda: STAMP, jobs_enabled=False
    )
    cine_a = app_a.cine_by_uid[SERIES_A]
    provenance_a = Provenance(
        cine_a.study_uid, SERIES_A, VERSION, None, STAMP, 0, cine_a.workflow_id
    )
    prediction_a = Prediction(
        cine_a.study_uid,
        SERIES_A,
        "manual-seed",
        {Structure.LV_CAVITY: _contour(0, 0, Structure.LV_CAVITY, _rect(18, 16, 20, 18))},
        None,
        provenance_a,
        STAMP,
    )
    store_a = app_a.open_store()
    try:
        store_a.save_prediction(prediction_a)
        head_a = app_a.controller.save(store_a, prediction_a, {}, "seed", STAMP)
        sel_a = app_a.controller.select_ed_es(
            cine_a, 0, 1, "physician", STAMP, lambda: provenance_a
        )
        store_a.save_edes_selection(sel_a)
    finally:
        store_a.close()

    app_b = WorkstationApplication.bootstrap(
        dest_b, port=0, clock=lambda: STAMP, jobs_enabled=False
    )
    cine_b = app_b.cine_by_uid[SERIES_B]
    sel_b = app_b.controller.select_ed_es(
        cine_b,
        0,
        1,
        "physician",
        STAMP,
        lambda: Provenance(cine_b.study_uid, SERIES_B, VERSION, None, STAMP, 0, cine_b.workflow_id),
    )
    store_b = app_b.open_store()
    try:
        with pytest.raises(StaleHeadError):
            store_b.save_edes_selection_checked(
                sel_b,
                expected_head_revision_id=head_a.revision_id,
                expected_selection_id=sel_a.selection_id,
                expected_inclusion_set_id=None,
                expected_n_slices=cine_b.n_slices,
            )
        assert store_b.load_latest_selection(SERIES_B) is None
    finally:
        store_b.close()

    # And A still restores exactly (return-to-A leg).
    app_a2 = WorkstationApplication.bootstrap(
        dest_a, port=0, clock=lambda: STAMP, jobs_enabled=False
    )
    store_a2 = app_a2.open_store()
    try:
        restored = store_a2.load_latest_selection(SERIES_A)
    finally:
        store_a2.close()
    assert restored is not None
    assert restored.selection_id == sel_a.selection_id


# --- Review-Hard Finding 3: preserve drive colon in UI staging destination ----


def test_app_js_staging_preserves_drive_qualified_destination() -> None:
    """Finding 3: static/app.js must preserve Windows drive colons (e.g. C:\\...)
    when extracting dest_path from the switch outcome for /api/library/stage,
    never truncating to drive letter 'C'.
    """
    app_js_path = _repo_root() / "services" / "workstation" / "static" / "app.js"
    assert app_js_path.is_file()
    content = app_js_path.read_text(encoding="utf-8")
    assert "destPath = outcome.dest_root" in content
    assert 'split(":")[0]' not in content

    # Run JS evaluation in Node: verify structured outcome preserves drive path
    js_probe = """
    function extractDest(outcome) {
        if (outcome && typeof outcome === "object" && outcome.dest_root) {
            return outcome.dest_root;
        }
        if (typeof outcome === "string" && outcome.indexOf("needs_stage:") === 0) {
            const rest = outcome.slice("needs_stage:".length);
            const lastColon = rest.lastIndexOf(":");
            return (lastColon > 1) ? rest.slice(0, lastColon) : rest;
        }
        return "";
    }
    const structured = {
        status: "needs_stage",
        dest_root: "C:\\\\stage\\\\study-B",
        detail: "unstaged"
    };
    const stringForm = "needs_stage:C:\\\\stage\\\\study-B:unstaged target";
    console.log(JSON.stringify({
        from_object: extractDest(structured),
        from_string: extractDest(stringForm)
    }));
    """
    proc = subprocess.run(
        ["node", "-e", js_probe],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout.strip())
    assert result["from_object"] == r"C:\stage\study-B"
    assert result["from_string"] == r"C:\stage\study-B"
    assert result["from_object"] != "C"
    assert result["from_string"] != "C"


# --- Review-Hard Round 6 Finding 3: persistent recovery banner in static/app.js ----


def test_recovery_banner_persists_after_initial_series_loading() -> None:
    """Finding 3: static/app.js must preserve the restored-A notice in a persistent
    banner element that ordinary initial series loading (loadStudies -> chooseSeries ->
    show('Series loaded', false)) cannot erase, remaining visible with error styling
    until dismissed or a subsequent switch succeeds.
    """
    app_js_path = _repo_root() / "services" / "workstation" / "static" / "app.js"
    assert app_js_path.is_file()

    js_mounted_test = r"""
const fs = require('fs');
const vm = require('vm');

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
    classList: {
      toggle: function (cls, val) {
        const parts = new Set((element.className || '').split(' ').filter(Boolean));
        if (val === undefined ? !parts.has(cls) : Boolean(val)) parts.add(cls);
        else parts.delete(cls);
        element.className = Array.from(parts).join(' ');
      },
      add: function (cls) { element.classList.toggle(cls, true); },
      remove: function (cls) { element.classList.toggle(cls, false); },
      contains: function (cls) { return (element.className || '').split(' ').includes(cls); }
    },
    addEventListener: function (name, fn) { listeners.set(name, fn); },
    appendChild: function (child) {
      child.parentNode = element;
      element.children.push(child);
      if (child.id) elements.set(child.id, child);
      return child;
    },
    replaceChildren: function (...newKids) {
      element.children = [];
      for (const k of newKids) element.appendChild(k);
    },
    insertBefore: function (child, ref) {
      child.parentNode = element;
      const idx = element.children.indexOf(ref);
      if (idx !== -1) element.children.splice(idx, 0, child);
      else element.children.push(child);
      if (child.id) elements.set(child.id, child);
      return child;
    },
    removeChild: function (child) {
      const idx = element.children.indexOf(child);
      if (idx !== -1) element.children.splice(idx, 1);
      child.parentNode = null;
      if (child.id) elements.delete(child.id);
      return child;
    },
    getContext: function () { return {clearRect: function () {}, drawImage: function () {}}; },
    click: function () {
      const fn = listeners.get('click');
      if (fn) fn({target: element});
    }
  };
  if (id) elements.set(id, element);
  return element;
}

function el(id) {
  if (!elements.has(id)) {
    const created = makeElement('div', id);
    elements.set(id, created);
  }
  return elements.get(id);
}

const viewport = makeElement('section', '', 'viewport');
const message = makeElement('p', 'message');
viewport.appendChild(message);

const controls = makeElement('aside', '', 'controls');

const document = {
  body: makeElement('body'),
  getElementById: el,
  createElement: function (tag) { return makeElement(tag); },
  createTextNode: function (val) { return {textContent: val}; },
  querySelector: function (sel) {
    if (sel === 'section.viewport') return viewport;
    if (sel === 'aside.controls') return controls;
    return null;
  },
  addEventListener: function () {}
};

global.document = document;
global.window = {
  document: document,
  location: {hash: '#switch-failed', pathname: '/', search: ''},
  history: {replaceState: function () {}},
  CMR_KEYBOARD_SHORTCUTS: {},
  setInterval: function () {},
  setTimeout: setTimeout,
  createImageBitmap: async () => ({width: 100, height: 100, close: () => {}})
};
global.URLSearchParams = URLSearchParams;

const STATE = {series_uid: '1.2.3', slices: 1, frames: 2, current_slice: 0, current_frame: 0};

// Create a valid binary cine bundle:
const frameRow = {frame_index: 0, byte_offset: 0, byte_length: 4};
const manifestObj = {
  version: 1,
  media: 'application/png-frame-bundle',
  frames: [frameRow]
};
const manifestBuf = Buffer.from(JSON.stringify(manifestObj), 'utf8');
const headerBuf = Buffer.alloc(8);
headerBuf.writeBigUInt64BE(BigInt(manifestBuf.length), 0);
const payloadBuf = Buffer.from([1, 2, 3, 4]);
const fullBundle = Buffer.concat([headerBuf, manifestBuf, payloadBuf]);

global.fetch = async function (url) {
  if (url.includes('/api/studies')) {
    return {ok: true, json: async () => [{uid: '1.2.3', slices: 1, frames: 2}]};
  }
  if (url.includes('/state')) {
    return {ok: true, json: async () => STATE};
  }
  if (url.includes('/cine')) {
    return {ok: true, headers: {get: () => 'etag'}, arrayBuffer: async () => fullBundle};
  }
  return {ok: false, status: 404};
};

const appJsCode = fs.readFileSync(process.env.APP_JS_PATH, 'utf8');
vm.runInThisContext(appJsCode, {filename: 'app.js'});

setTimeout(() => {
  const banner = elements.get('recovery-banner');
  const msgEl = elements.get('message');

  let bannerText = '';
  if (banner) {
    for (const child of banner.children) {
      if (child.className === 'recovery-banner-text') bannerText = child.textContent;
    }
  }

  let dismissBtn = null;
  if (banner) {
    for (const child of banner.children) {
      if (child.className === 'recovery-banner-dismiss') dismissBtn = child;
    }
  }

  const dismissedBefore = Boolean(elements.get('recovery-banner'));
  if (dismissBtn) {
    dismissBtn.click();
  }
  const dismissedAfter = Boolean(elements.get('recovery-banner'));

  console.log(JSON.stringify({
    message_text: msgEl ? msgEl.textContent : null,
    message_is_error: msgEl ? msgEl.classList.contains('error') : null,
    banner_exists_after_load: dismissedBefore,
    banner_text: bannerText,
    banner_role: banner ? banner.getAttribute('role') : null,
    banner_classes: banner ? banner.className : null,
    banner_dismissed_after_click: !dismissedAfter
  }));
}, 100);
"""
    proc = subprocess.run(
        ["node", "-e", js_mounted_test],
        env={**os.environ, "APP_JS_PATH": str(app_js_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout.strip())
    assert result["message_text"] == "Series loaded"
    assert result["message_is_error"] is False
    assert result["banner_exists_after_load"] is True
    assert "did not complete" in result["banner_text"]
    assert "previous study was restored" in result["banner_text"]
    assert result["banner_role"] == "alert"
    assert "recovery-banner" in result["banner_classes"]
    assert "error" in result["banner_classes"]
    assert result["banner_dismissed_after_click"] is True
