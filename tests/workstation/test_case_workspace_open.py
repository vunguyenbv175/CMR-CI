"""Case-workspace open tests (CMR_CASE_WORKSPACE_OPEN_002, slice 2).

Synthetic fixtures only (no PHI; SYNTH UIDs under an obviously-fake root).
Covers all 7 contract acceptance criteria:

- AC1: scan endpoint on a 3-study nested fixture returns all 3 studies with
  the correct series split (by UID, not folder).
- AC2: staging one study yields a root loadable via the EXISTING
  ``load_cine_volume`` + ``controller.select_series`` path (no new loader).
- AC3: unknown study_uid refuses fail-closed, no destination created.
- AC4: non-empty destination refuses, source untouched.
- AC5: destination inside source refuses (plus destination == source).
- AC6: bad scan path refuses 400, no raise through HTTP.
- AC7: existing Host/Origin policy unchanged on the new routes (plus the
  existing auth/neighbor suites stay green — run as gates, not here).
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from urllib.parse import urlencode

from cmr_core.errors import CMRError
from cmr_dicom import scan_library
from cmr_dicom.pipeline import load_cine_volume
from cmr_persistence.store import SqliteCmrStore
from cmr_workflow.controller import WorkflowController

from services.workstation.library import (
    DEST_EQUALS_SOURCE,
    DEST_INSIDE_SOURCE,
    DEST_NOT_EMPTY,
    UNKNOWN_STUDY,
    stage_case,
)
from tests.synth.generate_dicom import make_instance, write_synthetic_cine
from tests.workstation.support import (
    json_body,
    json_value,
    make_seeded_study,
    request,
    running_server,
)

UID_ROOT = "1.2.826.0.1.3680043.10.99998"

_sop_counter = itertools.count(1)


def _write(
    root: Path,
    rel: str,
    *,
    study: str,
    series: str,
    **kwargs,
) -> Path:
    """Write one synthetic DICOM instance at ``root/rel`` (parents made)."""
    n = next(_sop_counter)
    ds = make_instance(
        study_uid=study,
        series_uid=series,
        sop_uid=f"{UID_ROOT}.70.{n}",
        position_mm=(0.0, 0.0, 0.0),
        radius_px=10.0,
        trigger_time_ms=0.0,
        temporal_position=1,
        instance_number=1,
        **kwargs,
    )
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return path


def _study(i: int) -> str:
    return f"{UID_ROOT}.{i}"


def _series(i: int, j: int) -> str:
    return f"{UID_ROOT}.{i}.{j}"


def _three_study_tree(root: Path) -> dict[str, list[str]]:
    """Build the AC1/AC3 fixture: 3 nested studies + interleaved junk.

    Returns the expected {study_uid: [series_uids]} mapping (series sorted).
    """
    # Identical folder layouts carrying different UIDs (UID, not folder).
    _write(root, "siteA/series/img01.dcm", study=_study(1), series=_series(1, 1))
    _write(root, "siteA/series/img02.dcm", study=_study(1), series=_series(1, 1))
    _write(root, "siteB/series/img01.dcm", study=_study(2), series=_series(2, 1))
    _write(root, "siteB/series/img02.dcm", study=_study(2), series=_series(2, 1))
    # Deeply nested third study with two series.
    _write(root, "deep/n/study3a.dcm", study=_study(3), series=_series(3, 1))
    _write(root, "deep/n/o/p/study3b.dcm", study=_study(3), series=_series(3, 2))
    # Interleaved junk: silent non-DICOM + one DICOM-shaped corrupt file.
    (root / "notes.txt").write_text("not dicom", encoding="utf-8")
    (root / "corrupt.dcm").write_bytes(b"\x00" * 128 + b"DICM" + b"\xff" * 64)
    return {
        _study(1): [_series(1, 1)],
        _study(2): [_series(2, 1)],
        _study(3): [_series(3, 1), _series(3, 2)],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_url(base: str, route: str, params: dict[str, str]) -> str:
    return f"{route}?{urlencode(params)}"


def test_scan_endpoint_returns_three_studies_split_by_uid(tmp_path: Path) -> None:
    """AC1: GET /api/library/scan lists all 3 nested studies, UID-split."""
    src = tmp_path / "library"
    expected = _three_study_tree(src)
    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)

    with running_server(host.app) as (server, base_url):
        response = request(
            server, base_url, _scan_url(base_url, "/api/library/scan", {"path": str(src)})
        )
    assert response.status == 200
    body = json_value(response)

    assert [s["study_uid"] for s in body["studies"]] == sorted(expected)
    by_study = {s["study_uid"]: s for s in body["studies"]}
    for study_uid, series_uids in expected.items():
        got = by_study[study_uid]
        assert [s["series_uid"] for s in got["series"]] == sorted(series_uids)
        for row in got["series"]:
            assert row["count"] >= 1
            assert row["modality"] == "MR"
    # No series mixing across studies: UID sets are disjoint.
    seen: set[str] = set()
    for study in body["studies"]:
        for row in study["series"]:
            assert row["series_uid"] not in seen
            seen.add(row["series_uid"])
    # Identical basenames stayed with their own study (folder is not identity).
    s1_files = {row["series_uid"]: row["count"] for row in by_study[_study(1)]["series"]}
    s2_files = {row["series_uid"]: row["count"] for row in by_study[_study(2)]["series"]}
    assert s1_files == {_series(1, 1): 2}
    assert s2_files == {_series(2, 1): 2}
    # The corrupt file surfaces as a per-file error, never a raise.
    reasons = {err["path"]: err["reason"] for err in body["errors"]}
    assert reasons.get("corrupt.dcm") == "unreadable"
    assert "notes.txt" not in reasons


def test_stage_then_load_and_select_series(tmp_path: Path) -> None:
    """AC2: staged study loads via load_cine_volume + select_series."""
    src = tmp_path / "library"
    series_a1 = f"{UID_ROOT}.11.1"
    series_a2 = f"{UID_ROOT}.11.2"
    study_a = f"{UID_ROOT}.11"
    study_b = f"{UID_ROOT}.12"
    write_synthetic_cine(
        src / "cineA1", n_slices=2, n_frames=3,
        series_uid=series_a1, study_uid=study_a,
    )
    write_synthetic_cine(
        src / "cineA2", n_slices=1, n_frames=2,
        series_uid=series_a2, study_uid=study_a,
    )
    _write(src, "decoy/img.dcm", study=study_b, series=f"{UID_ROOT}.12.1")
    (src / "notes.txt").write_text("not dicom", encoding="utf-8")

    scan = scan_library(src)
    dest = tmp_path / "staged"
    staged = stage_case(scan, study_a, dest, source_root=src)

    assert staged.study_uid == study_a
    assert staged.series_uids == (series_a1, series_a2)
    assert staged.file_count == 2 * 3 + 1 * 2
    assert Path(staged.dest_root) == dest.resolve()

    # Staged layout: dest/dicom/<SeriesUID>/ flat files, names preserved.
    dicom_root = dest / "dicom"
    assert sorted(p.name for p in dicom_root.iterdir()) == [series_a1, series_a2]
    for series_uid, count in ((series_a1, 6), (series_a2, 2)):
        series_dir = dicom_root / series_uid
        files = sorted(p for p in series_dir.iterdir() if p.is_file())
        assert len(files) == count
        assert not [p for p in series_dir.iterdir() if p.is_dir()]
        for staged_file in files:
            origin = next(
                (src / rel for rel in _relatives(scan, study_a, series_uid)
                 if Path(rel).name == staged_file.name),
                None,
            )
            assert origin is not None
            assert _sha256(staged_file) == _sha256(origin)

    # The staged root reuses the EXISTING load path — no new loader.
    store = SqliteCmrStore(tmp_path / "verify.sqlite3")
    try:
        controller = WorkflowController()
        for series_uid, (n_slices, n_frames) in (
            (series_a1, (2, 3)),
            (series_a2, (1, 2)),
        ):
            series_dir = dicom_root / series_uid
            volume, series = load_cine_volume(series_dir, series_uid=series_uid)
            assert volume is not None and series is not None
            assert (volume.n_slices, volume.n_frames) == (n_slices, n_frames)
            selected_volume, selected_series = controller.select_series(
                store, series_dir, series_uid
            )
            assert selected_volume is not None and selected_series is not None
            manifest = selected_volume.source_manifest
            assert manifest is not None
            assert manifest.series_uid == series_uid
            assert manifest.study_uid == study_a
            assert store.load_source_manifest(manifest.workflow_id) == manifest
    finally:
        store.close()


def _relatives(scan, study_uid: str, series_uid: str) -> list[str]:
    for study in scan.studies:
        if study.study_uid != study_uid:
            continue
        for series in study.series:
            if series.series_uid == series_uid:
                return list(series.files)
    raise AssertionError("fixture series missing from scan")


def test_stage_unknown_study_refuses_without_creating_dest(tmp_path: Path) -> None:
    """AC3: unknown study_uid refuses fail-closed via helper and HTTP."""
    src = tmp_path / "library"
    _three_study_tree(src)
    scan = scan_library(src)

    dest = tmp_path / "never-created"
    try:
        stage_case(scan, f"{UID_ROOT}.absent", dest, source_root=src)
    except CMRError as exc:
        assert exc.code == UNKNOWN_STUDY
    else:
        raise AssertionError("unknown study must refuse")
    assert not dest.exists()

    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)
    with running_server(host.app) as (server, base_url):
        response = request(
            server,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body({
                "source_path": str(src),
                "study_uid": f"{UID_ROOT}.absent",
                "dest_path": str(tmp_path / "never-created-http"),
            }),
        )
    assert response.status == 404
    assert json_value(response) == {
        "code": UNKNOWN_STUDY,
        "error": "frozen_error",
        "message": json_value(response)["message"],
    }
    assert not (tmp_path / "never-created-http").exists()


def test_stage_nonempty_dest_refused_source_untouched(tmp_path: Path) -> None:
    """AC4: occupied destination refuses; source and dest keep their bytes."""
    src = tmp_path / "library"
    expected = _three_study_tree(src)
    study = next(iter(expected))
    scan = scan_library(src)
    before = {p: _sha256(p) for p in sorted(src.rglob("*")) if p.is_file()}

    dest = tmp_path / "occupied"
    dest.mkdir()
    sentinel = dest / "sentinel.txt"
    sentinel.write_text("physician data", encoding="utf-8")

    try:
        stage_case(scan, study, dest, source_root=src)
    except CMRError as exc:
        assert exc.code == DEST_NOT_EMPTY
    else:
        raise AssertionError("non-empty dest must refuse")
    assert sentinel.read_text(encoding="utf-8") == "physician data"
    assert {p: _sha256(p) for p in sorted(src.rglob("*")) if p.is_file()} == before

    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)
    with running_server(host.app) as (server, base_url):
        response = request(
            server,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body({
                "source_path": str(src),
                "study_uid": study,
                "dest_path": str(dest),
            }),
        )
    assert response.status == 400
    assert json_value(response)["code"] == DEST_NOT_EMPTY
    assert sentinel.read_text(encoding="utf-8") == "physician data"


def test_stage_dest_inside_source_or_equal_refused(tmp_path: Path) -> None:
    """AC5 (+ task dest==source): overlapping destinations refuse cleanly."""
    src = tmp_path / "library"
    expected = _three_study_tree(src)
    study = next(iter(expected))
    scan = scan_library(src)

    for dest, code in (
        (src / "inner", DEST_INSIDE_SOURCE),
        (src / "inner" / "deeper", DEST_INSIDE_SOURCE),
        (src, DEST_EQUALS_SOURCE),
    ):
        try:
            stage_case(scan, study, dest, source_root=src)
        except CMRError as exc:
            assert exc.code == code
        else:
            raise AssertionError(f"{dest} must refuse")
    assert not (src / "inner").exists()

    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)
    with running_server(host.app) as (server, base_url):
        response = request(
            server,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body({
                "source_path": str(src),
                "study_uid": study,
                "dest_path": str(src / "inner-http"),
            }),
        )
    assert response.status == 400
    assert json_value(response)["code"] == DEST_INSIDE_SOURCE
    assert not (src / "inner-http").exists()


def test_stage_without_source_root_refuses(tmp_path: Path) -> None:
    """Guard: source_root omission refuses (scan holds relative paths only)."""
    src = tmp_path / "library"
    expected = _three_study_tree(src)
    scan = scan_library(src)
    dest = tmp_path / "staged"
    try:
        stage_case(scan, next(iter(expected)), dest)
    except CMRError as exc:
        assert exc.code == "library.bad_path"
    else:
        raise AssertionError("missing source_root must refuse")
    assert not dest.exists()


def test_scan_bad_path_is_400_without_raise(tmp_path: Path) -> None:
    """AC6: missing/non-dir/relative scan paths refuse 400 via HTTP."""
    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)
    missing = tmp_path / "absent"
    probe_file = tmp_path / "file.txt"
    probe_file.write_text("synthetic", encoding="utf-8")
    with running_server(host.app) as (server, base_url):
        cases = [
            "/api/library/scan",
            "/api/library/scan?path=",
            _scan_url(base_url, "/api/library/scan", {"path": str(missing)}),
            _scan_url(base_url, "/api/library/scan", {"path": str(probe_file)}),
            "/api/library/scan?path=relative/dir",
            "/api/library/scan?path=/a&path=/b",
        ]
        for path in cases:
            response = request(server, base_url, path)
            assert response.status == 400, path
            assert json_value(response)["code"] == "library.bad_path", path


def test_library_routes_keep_host_origin_policy(tmp_path: Path) -> None:
    """AC7: new routes sit behind the unchanged Host/Origin policy."""
    src = tmp_path / "library"
    _three_study_tree(src)
    host = make_seeded_study(tmp_path / "host", jobs_enabled=False)
    with running_server(host.app) as (server, base_url):
        port = int(server.server_address[1])
        spoofed_host = {"Host": "spoof.example:8787"}
        forbidden_scan = request(
            server,
            base_url,
            _scan_url(base_url, "/api/library/scan", {"path": str(src)}),
            headers=spoofed_host,
        )
        assert forbidden_scan.status == 403
        assert json_value(forbidden_scan)["error"] == "forbidden"

        forbidden_stage = request(
            server,
            base_url,
            "/api/library/stage",
            method="POST",
            body=json_body({
                "source_path": str(src),
                "study_uid": "anything",
                "dest_path": str(tmp_path / "x"),
            }),
            headers={"Origin": "http://spoof.example:8787"},
        )
        assert forbidden_stage.status == 403
        assert json_value(forbidden_stage)["error"] == "forbidden"
        assert port != 0  # listener is the real bound server, not a stub
