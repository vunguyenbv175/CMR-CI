"""HTTP API, bootstrap, and authority contract tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from services.workstation.frames import decode_cine_bundle
from services.workstation.server import (
    WorkstationApplication,
    WorkstationConfigurationError,
    authorize_host,
    authorize_mutation,
    normalize_authority,
    normalize_origin,
)
from tests.synth.generate_dicom import write_synthetic_cine
from tests.workstation.support import (
    SeededStudy,
    action_payload,
    all_included,
    decode_png_rgb,
    json_body,
    json_value,
    make_seeded_study,
    request,
    running_server,
    view_query,
)


def test_disabled_jobs_omit_projection_and_do_not_enqueue(tmp_path: Path, monkeypatch) -> None:
    study = make_seeded_study(tmp_path, jobs_enabled=False)
    assert study.app.job_runner is None
    store = study.app.open_store()
    try:
        original = store.load_analysis_jobs

        def fail_if_read(series_uid: str):
            raise AssertionError("disabled application must not read job projection")

        monkeypatch.setattr(type(store), "load_analysis_jobs", fail_if_read)
        state = study.controller.reopen(store, study.series_uid, cine=study.cine)
        assert state["series_uid"] == study.series_uid
        monkeypatch.setattr(type(store), "load_analysis_jobs", original)
    finally:
        store.close()


def test_studies_state_frame_and_cine_endpoints(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    with running_server(study.app) as (server, base_url):
        studies = request(server, base_url, "/api/studies")
        assert studies.status == 200
        assert json_value(studies) == [
            {
                "description": "cine SAX",
                "frames": 2,
                "modality": "MR",
                "slices": 1,
                "uid": study.series_uid,
                # Additive series-kind signal (R1 viewer series selection).
                "sequence_class": "cine_sax",
            }
        ]

        state_response = request(server, base_url, f"/api/study/{study.series_uid}/state")
        assert state_response.status == 200
        state = json_value(state_response)
        assert state["series_uid"] == study.series_uid
        assert state["study_uid"] == study.cine.study_uid
        assert state["workflow_id"] == study.cine.workflow_id
        assert state["manifest_digest"] == study.cine.workflow_id
        assert state["head_revision_id"] is not None
        assert state["measurement_id"] is None
        assert len(state["inclusion_draft"]) == 5

        frame = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/frame?{view_query()}",
        )
        assert frame.status == 200
        assert frame.body.startswith(b"\x89PNG\r\n\x1a\n")
        assert frame.headers["Cache-Control"] == "no-cache"

        cine = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/cine?{view_query(include_frame=False)}",
        )
        assert cine.status == 200
        assert cine.headers["Content-Type"] == "application/vnd.cmr.cine-bundle"
        assert cine.headers["Cache-Control"] == "no-cache"
        assert int(cine.headers["Content-Length"]) == len(cine.body)


def test_cine_conditional_304_and_forbidden_frame_query(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, seed_history=False)
    with running_server(study.app) as (server, base_url):
        path = f"/api/study/{study.series_uid}/cine?{view_query(include_frame=False)}"
        first = request(server, base_url, path)
        second = request(
            server,
            base_url,
            path,
            headers={"If-None-Match": first.headers["ETag"]},
        )
        assert first.status == 200
        assert second.status == 304
        assert second.body == b""
        assert second.headers["Content-Length"] == "0"
        assert second.headers["ETag"] == first.headers["ETag"]
        assert second.headers["Cache-Control"] == "no-cache"

        forbidden = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/cine?{view_query(include_frame=True)}",
        )
        assert forbidden.status == 400
        assert json_value(forbidden)["error"] == "frozen_error"


def test_cine_manifest_rows_match_equivalent_frame_endpoints(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=3, seed_history=False)
    with running_server(study.app) as (server, base_url):
        cine_path = f"/api/study/{study.series_uid}/cine?{view_query(include_frame=False)}"
        bundle_response = request(server, base_url, cine_path)
        assert bundle_response.status == 200
        body = bundle_response.body
        assert bundle_response.headers["ETag"] == (
            f'"cine-{hashlib.sha256(body).hexdigest()}"'
        )
        manifest_length = int.from_bytes(body[:8], byteorder="big", signed=False)
        manifest_bytes = body[8 : 8 + manifest_length]
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        assert manifest_bytes == json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        _decoded_manifest, pngs = decode_cine_bundle(body)
        payload = body[8 + manifest_length :]
        assert len(pngs) == study.cine.n_frames
        for row, bundle_png in zip(manifest["frames"], pngs):
            frame_path = (
                f"/api/study/{study.series_uid}/frame?"
                f"{view_query(frame_index=row['frame_index'])}"
            )
            frame_response = request(server, base_url, frame_path)
            assert frame_response.status == 200
            assert frame_response.body == bundle_png
            assert (
                payload[row["byte_offset"] : row["byte_offset"] + row["byte_length"]]
                == bundle_png
            )
            assert row["digest"] == frame_response.headers["X-Frame-Digest"]
            assert frame_response.headers["ETag"] == f'"frame-{row["digest"]}"'
            assert decode_png_rgb(frame_response.body).shape == (500, 800, 3)


def test_cine_context_changes_revalidate_and_same_body_returns_304(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=2, n_frames=2, seed_history=False)
    with running_server(study.app) as (server, base_url):
        initial_path = (
            f"/api/study/{study.series_uid}/cine?"
            f"{view_query(slice_index=0, fps=10, playing=False, include_frame=False)}"
        )
        initial = request(server, base_url, initial_path)
        assert initial.status == 200
        draft = all_included(2)
        draft[4]["included"] = False
        changed_queries = (
            view_query(slice_index=0, fps=11, playing=False, include_frame=False),
            view_query(slice_index=0, fps=10, playing=True, include_frame=False),
            view_query(
                slice_index=0,
                fps=10,
                playing=False,
                reference_sex="men",
                include_frame=False,
            ),
            view_query(
                slice_index=0,
                fps=10,
                playing=False,
                inclusion_draft=draft,
                include_frame=False,
            ),
        )
        for query in changed_queries:
            current = request(
                server,
                base_url,
                f"/api/study/{study.series_uid}/cine?{query}",
            )
            response = request(
                server,
                base_url,
                f"/api/study/{study.series_uid}/cine?{query}",
                headers={"If-None-Match": initial.headers["ETag"]},
            )
            if current.body == initial.body:
                assert response.status == 304
                assert response.body == b""
            else:
                assert response.status == 200
                assert response.body == current.body
                assert response.headers["ETag"] != initial.headers["ETag"]

        equivalent = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/cine?"
            "playing=false&inclusion_draft=null&fps=10&slice=0",
            headers={"If-None-Match": initial.headers["ETag"]},
        )
        assert equivalent.status == 304
        assert equivalent.body == b""


def test_unknown_and_out_of_range_routes_fail_closed(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, seed_history=False)
    with running_server(study.app) as (server, base_url):
        unknown = request(server, base_url, "/api/study/not-a-series/state")
        assert unknown.status == 404
        assert json_value(unknown)["error"] == "not_found"

        out_of_range = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/frame?{view_query(slice_index=1)}",
        )
        assert out_of_range.status == 400
        assert json_value(out_of_range)["error"] == "frozen_error"

        traversal = request(server, base_url, "/static/../RUNBOOK.md")
        assert traversal.status in {404, 400}


def test_checked_action_round_trip_and_stale_response_reset(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path)
    with running_server(study.app) as (server, base_url):
        initial_response = request(server, base_url, f"/api/study/{study.series_uid}/state")
        initial = json_value(initial_response)
        select = action_payload(
            initial,
            action="select_ed_es",
            params={"ed_frame": 0, "es_frame": 1},
            nonce="select-1",
        )
        selected_response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body(select),
            headers={"Content-Type": "application/json"},
        )
        assert selected_response.status == 200
        selected = json_value(selected_response)
        assert selected["action"] == "select_ed_es"
        assert selected["state"]["selection"]["ed_frame"] == 0
        assert selected["state"]["selection"]["es_frame"] == 1
        assert selected["inclusion_draft"] == all_included(1)

        # The frozen validator rejects an all-false group when N=1.  The
        # multi-slice nontrivial draft is covered by the end-to-end contract
        # test; this round-trip uses the valid default grid.
        draft = all_included(1)
        apply = action_payload(
            selected["state"],
            action="apply_inclusion_and_refresh",
            params={},
            nonce="apply-1",
            draft=draft,
        )
        applied_response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body(apply),
            headers={"Content-Type": "application/json"},
        )
        assert applied_response.status == 200
        applied = json_value(applied_response)
        assert applied["state"]["measurement_id"] is not None
        assert applied["state"]["inclusion_set_id"] is not None
        assert applied["inclusion_draft"] == draft

        stale = action_payload(
            selected["state"],
            action="apply_inclusion_and_refresh",
            params={},
            nonce="stale-1",
            draft=draft,
        )
        stale_response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body(stale),
            headers={"Content-Type": "application/json"},
        )
        assert stale_response.status == 409
        stale_value = json_value(stale_response)
        assert stale_value["error"] == "stale_view"
        assert stale_value["client_nonce"] == "stale-1"
        assert stale_value["inclusion_draft_reset"] is True
        assert stale_value["inclusion_draft"] == all_included(1)
        assert stale_value["state"]["measurement_id"] == applied["state"]["measurement_id"]


def test_malformed_and_unknown_actions_are_400(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, seed_history=False)
    with running_server(study.app) as (server, base_url):
        unknown = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body({"action": "nope"}),
            headers={"Content-Type": "application/json"},
        )
        assert unknown.status == 400
        assert json_value(unknown)["error"] == "invalid_payload"

        malformed = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body({}),
            headers={"Content-Type": "application/json"},
        )
        assert malformed.status == 400
        assert json_value(malformed)["error"] == "invalid_payload"


def test_authority_normalization_and_mutation_matrix() -> None:
    assert normalize_authority("LOCALHOST:8787") == "localhost:8787"
    assert normalize_authority("127.0.0.1:8787") == "127.0.0.1:8787"
    assert normalize_authority("[2001:0DB8::1]:8787") == "[2001:db8::1]:8787"
    assert normalize_authority("127.0.0.1", require_port=False) == "127.0.0.1:80"
    assert normalize_origin("http://LOCALHOST:8787") == "localhost:8787"
    assert normalize_origin("http://[2001:0DB8::1]:8787") == "[2001:db8::1]:8787"

    allowed = ("localhost:8787", "192.168.10.20:8787", "[2001:db8::1]:8787")
    assert authorize_host({"Host": "LOCALHOST:8787"}, allowed) == "localhost:8787"
    assert authorize_mutation(
        {"Host": "192.168.10.20:8787", "Origin": "http://192.168.10.20:8787"},
        allowed,
    ) == "192.168.10.20:8787"
    assert authorize_mutation(
        {"Host": "[2001:DB8::1]:8787", "Origin": "http://[2001:db8::1]:8787"},
        allowed,
    ) == "[2001:db8::1]:8787"

    negative_headers: tuple[dict[str, object], ...] = (
        {},
        {"Host": ["localhost:8787", "localhost:8787"]},
        {"Host": "spoof.example:8787"},
        {"Host": "localhost:8787", "Origin": "null"},
        {"Host": "localhost:8787"},
        {"Host": "localhost:8787", "Origin": "https://localhost:8787"},
        {"Host": "localhost:8787", "Origin": "http://192.168.10.20:8787"},
        {"Host": "localhost:8787", "Origin": "http://localhost:8787/path"},
        {"Host": "localhost:8787", "Origin": "http://user@localhost:8787"},
    )
    for headers in negative_headers:
        with pytest.raises(Exception):
            authorize_mutation(headers, allowed)

    for malformed in (
        "0.0.0.0:8787",
        "[::]:8787",
        "2001:db8::1:8787",
        "[127.0.0.1]:8787",
        "localhost:0",
        "localhost:65536",
    ):
        with pytest.raises(ValueError):
            normalize_authority(malformed)


def test_wildcard_bootstrap_requires_explicit_non_wildcard_authority(tmp_path: Path) -> None:
    root = tmp_path / "study"
    uid = "1.2.826.0.1.3680043.10.999.910"
    write_synthetic_cine(
        root / "dicom" / uid,
        n_slices=1,
        n_frames=1,
        series_uid=uid,
        study_uid="1.2.826.0.1.3680043.10.999.911",
    )

    with pytest.raises(WorkstationConfigurationError, match="wildcard"):
        WorkstationApplication.bootstrap(root, host="0.0.0.0", port=8787)
    with pytest.raises(WorkstationConfigurationError, match="wildcard"):
        WorkstationApplication.bootstrap(
            root,
            host="0.0.0.0",
            port=8787,
            allow_authorities=["0.0.0.0:8787"],
        )

    app = WorkstationApplication.bootstrap(
        root,
        host="0.0.0.0",
        port=8787,
        allow_authorities=["192.168.10.20:8787"],
    )
    assert app.host == "0.0.0.0"
    assert app.allowed_authorities == ("192.168.10.20:8787",)


def test_bootstrap_enumerates_explicit_uids_lexically_and_persists_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study"
    uid_z = "1.2.826.0.1.3680043.10.999.920"
    uid_a = "1.2.826.0.1.3680043.10.999.919"
    write_synthetic_cine(
        root / "dicom" / uid_z,
        n_slices=1,
        n_frames=1,
        series_uid=uid_z,
        study_uid="1.2.826.0.1.3680043.10.999.921",
    )
    write_synthetic_cine(
        root / "dicom" / uid_a,
        n_slices=1,
        n_frames=1,
        series_uid=uid_a,
        study_uid="1.2.826.0.1.3680043.10.999.922",
    )

    app = WorkstationApplication.bootstrap(root, port=0)
    assert list(app.cine_by_uid) == [uid_a, uid_z]
    assert list(app.series_by_uid) == [uid_a, uid_z]
    store = app.open_store()
    try:
        for uid in (uid_a, uid_z):
            cine = app.cine_by_uid[uid]
            manifest = cine.source_manifest
            assert manifest is not None
            assert manifest.series_uid == uid
            assert store.load_source_manifest(manifest.workflow_id) == manifest
            selected_cine, selected_series = app.controller.select_series(
                store, root / "dicom" / uid, uid
            )
            assert selected_cine is not None
            assert selected_cine.workflow_id == cine.workflow_id
            assert selected_cine.cine_volume_digest == cine.cine_volume_digest
            assert selected_series is not None and selected_series.series_uid == uid
    finally:
        store.close()


@pytest.mark.parametrize(
    "invalid_layout",
    (
        "root-file",
        "nested",
        "empty-series",
        "unusable",
        "mismatch",
        "duplicate-sop",
    ),
)
def test_bootstrap_rejects_invalid_layout_before_creating_store(
    tmp_path: Path, invalid_layout: str
) -> None:
    root = tmp_path / "study"
    uid = "1.2.826.0.1.3680043.10.999.930"
    series_dir = root / "dicom" / uid
    if invalid_layout == "root-file":
        write_synthetic_cine(
            series_dir,
            n_slices=1,
            n_frames=1,
            series_uid=uid,
            study_uid="1.2.826.0.1.3680043.10.999.931",
        )
        (root / "unexpected.txt").write_text("synthetic", encoding="utf-8")
    elif invalid_layout == "nested":
        write_synthetic_cine(
            series_dir,
            n_slices=1,
            n_frames=1,
            series_uid=uid,
            study_uid="1.2.826.0.1.3680043.10.999.931",
        )
        (series_dir / "nested").mkdir()
    elif invalid_layout == "empty-series":
        series_dir.mkdir(parents=True)
    elif invalid_layout == "unusable":
        series_dir.mkdir(parents=True)
        (series_dir / "notes.txt").write_text("not DICOM", encoding="utf-8")
    elif invalid_layout == "mismatch":
        write_synthetic_cine(
            series_dir,
            n_slices=1,
            n_frames=1,
            series_uid="1.2.826.0.1.3680043.10.999.932",
            study_uid="1.2.826.0.1.3680043.10.999.931",
        )
    else:
        paths = write_synthetic_cine(
            series_dir,
            n_slices=1,
            n_frames=2,
            series_uid=uid,
            study_uid="1.2.826.0.1.3680043.10.999.931",
        )
        shutil.copyfile(paths[0], series_dir / "duplicate.dcm")

    with pytest.raises(WorkstationConfigurationError):
        WorkstationApplication.bootstrap(root, port=0)
    assert not (root / "cmr.sqlite3").exists()


def test_malformed_display_params_are_400_and_valid_ones_are_echoed(tmp_path: Path) -> None:
    """S3-008B §3.1/§3.2: 400 on malformed display params, 200 + echo on valid.

    The negatives are the frozen §3.2 list; the positives prove the two new
    flat /state keys are appended LAST and carry the canonical values.
    """
    study = make_seeded_study(tmp_path)
    with running_server(study.app) as (server, base_url):
        for raw in (
            "%7B%22center%22%3A1e2%2C%22width%22%3A400%7D",          # 1e2
            "%7B%22center%22%3A-0.0%2C%22width%22%3A400%7D",         # -0.0
            "%7B%22center%22%3A%2240%22%2C%22width%22%3A400%7D",     # string
            "%7B%22center%22%3Anull%2C%22width%22%3A400%7D",         # inner null
            "%7B%22center%22%3A40%2C%22width%22%3A1%7D",             # width <= 1
            "%5B%22unknown_structure%22%5D",                         # unknown
            "%5B%22rv_cavity%22%2C%22lv_cavity%22%5D",               # out-of-order
            "%5B%22lv_cavity%22%2C%22lv_cavity%22%5D",               # duplicate
        ):
            for name in ("window_level", "visible_structures"):
                response = request(
                    server,
                    base_url,
                    f"/api/study/{study.series_uid}/state?{view_query()}"
                    f"&{name}={raw}",
                )
                assert response.status == 400, (name, raw, response.status)

        # A literal JSON null is ACCEPTED and normalized to the default.
        response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/state?{view_query()}"
            "&window_level=null&visible_structures=null",
        )
        assert response.status == 200
        assert json_value(response)["window_level"] is None
        assert json_value(response)["visible_structures"] is None

        # A valid explicit display context is echoed with canonical values.
        response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/state?{view_query()}"
            "&window_level=%7B%22center%22%3A40%2C%22width%22%3A400%7D"
            "&visible_structures=%5B%22lv_cavity%22%5D",
        )
        assert response.status == 200
        body = json_value(response)
        assert body["window_level"] == {"center": 40.0, "width": 400.0}
        assert body["visible_structures"] == ["lv_cavity"]

        # A display param is NOT an action field.
        payload = action_payload(
            json_value(
                request(server, base_url, f"/api/study/{study.series_uid}/state")
            ),
            action="select_ed_es",
            params={"ed_frame": 0, "es_frame": 1},
            nonce="server-api-display-in-action",
        )
        payload["view"]["window_level"] = {"center": 40, "width": 400}
        response = request(
            server,
            base_url,
            f"/api/study/{study.series_uid}/action",
            method="POST",
            body=json_body(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400


def _analysis_job_fixture(tmp_path: Path, monkeypatch) -> SeededStudy:
    """Seeded study + one enqueued/claimed job for cancellation route tests."""

    study = make_seeded_study(tmp_path, jobs_enabled=True, n_slices=1, n_frames=2)
    store = study.app.open_store()
    try:
        from services.workstation.jobs import request_identity

        identity = request_identity(
            series_uid=study.series_uid,
            head_revision_id=study.controller.reopen(
                store, study.series_uid, cine=study.cine
            )["chain"][-1].revision_id,
            selection_id=None,
            inclusion_set_id=None,
            coverage_identity={"ed_frame": 0, "es_frame": 1},
            software_version="sw",
            algorithm_version="av",
            range_table_version="rt",
        )
        store.enqueue_analysis_job(
            series_uid=study.series_uid,
            kind="selected_phase_recompute",
            request_payload=identity,
            dedup_identity=identity,
            created_at="2026-09-11T00:00:00+00:00",
        )
    finally:
        store.close()
    return study


def test_preapplied_cancel_never_reaches_inclusion_mutation_no_draft(
    tmp_path: Path, monkeypatch
) -> None:
    """R2 review finding 1 (probe): a cancellation request routed through the
    runner (cancel_already_applied=True) with NO inclusion draft must not
    fall through into the inclusion action (no AssertionError, no unrelated
    clinical mutation)."""
    from cmr_workflow.controller import WorkflowController

    study = _analysis_job_fixture(tmp_path, monkeypatch)
    real_cancel = WorkflowController.apply_inclusion_and_refresh_analysis_checked

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("cancellation request reached inclusion mutation")

    monkeypatch.setattr(
        WorkflowController, "apply_inclusion_and_refresh_analysis_checked", fail_if_called
    )
    try:
        with running_server(study.app) as (server, base_url):
            state = json_value(
                request(server, base_url, f"/api/study/{study.series_uid}/state")
            )
            jobs = study.app.open_store().load_analysis_jobs(study.series_uid)
            assert jobs, "fixture must enqueue one job"
            payload = action_payload(
                state,
                action="cancel_analysis_job",
                params={"job_id": jobs[0].job_id},
                nonce="n-cancel-1",
                draft=None,
            )
            # No draft: the wire payload carries null inclusion_draft.
            payload["view"]["inclusion_draft"] = None
            response = request(
                server,
                base_url,
                f"/api/study/{study.series_uid}/action",
                method="POST",
                body=json_body(payload),
            )
            assert response.status in (200, 409), response.status
    finally:
        monkeypatch.setattr(
            WorkflowController,
            "apply_inclusion_and_refresh_analysis_checked",
            real_cancel,
        )


def test_preapplied_cancel_never_reaches_inclusion_mutation_with_draft(
    tmp_path: Path, monkeypatch
) -> None:
    """R2 review finding 1 (probe, draft case): with a NON-NULL inclusion
    draft the cancellation request must still not invoke the inclusion
    mutation; the response is a plain cancellation/state result."""
    from cmr_workflow.controller import WorkflowController

    study = _analysis_job_fixture(tmp_path, monkeypatch)

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("cancellation request reached inclusion mutation")

    real = WorkflowController.apply_inclusion_and_refresh_analysis_checked
    monkeypatch.setattr(
        WorkflowController, "apply_inclusion_and_refresh_analysis_checked", fail_if_called
    )
    try:
        with running_server(study.app) as (server, base_url):
            store = study.app.open_store()
            state = json_value(
                request(server, base_url, f"/api/study/{study.series_uid}/state")
            )
            jobs = store.load_analysis_jobs(study.series_uid)
            assert jobs, "fixture must enqueue one job"
            # A COMPLETE 5xN inclusion grid (from the current state wire) so
            # payload validation accepts the draft and the request reaches
            # the action path — the reviewer's draft case.
            payload = action_payload(
                state,
                action="cancel_analysis_job",
                params={"job_id": jobs[0].job_id},
                nonce="n-cancel-2",
                draft=state["inclusion_draft"],
            )
            response = request(
                server,
                base_url,
                f"/api/study/{study.series_uid}/action",
                method="POST",
                body=json_body(payload),
            )
            assert response.status in (200, 409), response.status
            # Exactly one job row, terminal or unchanged — never a second
            # cancellation path or an inclusion mutation.
            store2 = study.app.open_store()
            try:
                rows = store2.load_analysis_jobs(study.series_uid)
                assert len(rows) == 1
                assert rows[0].state in {"queued", "running", "cancelled"}
                inclusion_rows = store2._require_connection().execute(
                    "SELECT COUNT(*) FROM slice_inclusion_sets"
                ).fetchone()[0]
                assert inclusion_rows == 0, "cancellation must not write inclusion sets"
            finally:
                store2.close()
    finally:
        monkeypatch.setattr(
            WorkflowController,
            "apply_inclusion_and_refresh_analysis_checked",
            real,
        )
