"""Backend cross-case isolation proof (CMR_CASE_WORKSPACE_SWITCH_004A).

Synthetic fixtures only (no PHI; SYNTH UIDs under the obviously-fake
``...99990`` root unique to this file).

Proves, at the store boundary, that switching the case workspace from case
A to case B cannot leak A's physician state into B, and that returning to
A restores exactly what was recorded:

- T1 fresh-B-none-of-A: a freshly bootstrapped app on staged case B shows
  no selection, no measurement, and an empty revision chain for B's
  series — none of A's ED/ES/inclusion/measurement ids.
- T2 ED/ES-carry mutant: submitting A's REAL persisted selection/head
  lineage against B's checked save refuses with :class:`StaleHeadError`
  and writes zero rows to B.
- T3 inclusion-carry mutant: submitting A's selection id with a null head
  (the no-history state B actually has) also refuses, leaving B untouched.
- T4 B->A restore: after visiting B, a second bootstrap on A's staged
  root reloads exactly the recorded selection AND revision head.

Guard evidence cited here (all VERIFIED_FROM_SOURCE by the manager probe):
``packages/cmr_persistence/store.py:2692`` ``save_edes_selection_checked``
(CAS before idempotency lookup); ``store.py:2701-2707`` docstring;
``services/workstation/actions.py:1912-1921`` (head/selection mismatch
``StaleHeadError``); ``actions.py:2642-2656`` (approve-path
selection/inclusion mismatch).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cmr_core.domain_types import Prediction
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_dicom.discovery import scan_library
from cmr_persistence.store import StaleHeadError

from services.workstation.library import stage_case
from services.workstation.server import WorkstationApplication
from tests.synth.generate_dicom import write_synthetic_cine
from tests.workstation.support import STAMP, _contour, _rect

UID = "1.2.826.0.1.3680043.10.99990"


def _two_case_world(tmp_path: Path):
    """Stage two synthetic cases and record selection + head in case A only.

    Returns ``(serA, serB, selA, headA, stA_dest, stB_dest)``. The caller owns
    the stores it opens; bootstrapped apps hold no open connections themselves.
    """
    src = tmp_path / "lib"
    write_synthetic_cine(
        src / "nestA",
        n_slices=2,
        n_frames=4,
        series_uid=UID + ".1.1",
        study_uid=UID + ".1",
    )
    write_synthetic_cine(
        src / "nestB",
        n_slices=2,
        n_frames=4,
        series_uid=UID + ".2.1",
        study_uid=UID + ".2",
    )
    scan = scan_library(src)
    stA = stage_case(scan, UID + ".1", tmp_path / "stA", source_root=src)
    stB = stage_case(scan, UID + ".2", tmp_path / "stB", source_root=src)
    serA, serB = stA.series_uids[0], stB.series_uids[0]

    appA = WorkstationApplication.bootstrap(
        Path(stA.dest_root), port=0, clock=lambda: STAMP, jobs_enabled=False
    )
    cineA = appA.cine_by_uid[serA]
    provenanceA = Provenance(cineA.study_uid, serA, VERSION, None, STAMP, 0, cineA.workflow_id)
    predictionA = Prediction(
        cineA.study_uid,
        serA,
        "manual-seed",
        {
            Structure.LV_CAVITY: _contour(0, 0, Structure.LV_CAVITY, _rect(18, 16, 20, 18)),
        },
        None,
        provenanceA,
        STAMP,
    )
    storeA = appA.open_store()
    try:
        storeA.save_prediction(predictionA)
        headA = appA.controller.save(storeA, predictionA, {}, "seed", STAMP)
        selA = appA.controller.select_ed_es(cineA, 0, 3, "physician", STAMP, lambda: provenanceA)
        storeA.save_edes_selection(selA)
    finally:
        storeA.close()
    return serA, serB, selA, headA.revision_id, Path(stA.dest_root), Path(stB.dest_root)


def test_fresh_case_b_shows_none_of_case_a(tmp_path: Path) -> None:
    """T1: fresh B shows no selection, no measurement, empty chain."""
    serA, serB, selA, _headA, _destA, destB = _two_case_world(tmp_path)
    assert serA != serB

    appB = WorkstationApplication.bootstrap(destB, port=0, clock=lambda: STAMP, jobs_enabled=False)
    storeB = appB.open_store()
    try:
        assert storeB.load_latest_selection(serB) is None
        assert storeB.load_latest_measurement(serB) is None
        assert storeB.load_chain(serB) == []
    finally:
        storeB.close()


def test_edes_carry_mutant_refuses_and_writes_nothing(tmp_path: Path) -> None:
    """T2: A's REAL head+selection lineage against B refuses; B stays empty."""
    _serA, serB, selA, headA, _destA, destB = _two_case_world(tmp_path)

    appB = WorkstationApplication.bootstrap(destB, port=0, clock=lambda: STAMP, jobs_enabled=False)
    cineB = appB.cine_by_uid[serB]
    selB = appB.controller.select_ed_es(
        cineB,
        0,
        3,
        "physician",
        STAMP,
        lambda: Provenance(cineB.study_uid, serB, VERSION, None, STAMP, 0, cineB.workflow_id),
    )
    storeB = appB.open_store()
    try:
        with pytest.raises(StaleHeadError):
            storeB.save_edes_selection_checked(
                selB,
                expected_head_revision_id=headA,
                expected_selection_id=selA.selection_id,
                expected_inclusion_set_id=None,
                expected_n_slices=cineB.n_slices,
            )
        assert storeB.load_latest_selection(serB) is None
    finally:
        storeB.close()


def test_inclusion_carry_mutant_refuses_and_leaves_b_untouched(
    tmp_path: Path,
) -> None:
    """T3: A's selection id with null head refuses; B chain/mid empty."""
    _serA, serB, selA, _headA, _destA, destB = _two_case_world(tmp_path)

    appB = WorkstationApplication.bootstrap(destB, port=0, clock=lambda: STAMP, jobs_enabled=False)
    cineB = appB.cine_by_uid[serB]
    selB = appB.controller.select_ed_es(
        cineB,
        0,
        3,
        "physician",
        STAMP,
        lambda: Provenance(cineB.study_uid, serB, VERSION, None, STAMP, 0, cineB.workflow_id),
    )
    storeB = appB.open_store()
    try:
        with pytest.raises(StaleHeadError):
            storeB.save_edes_selection_checked(
                selB,
                expected_head_revision_id=None,
                expected_selection_id=selA.selection_id,
                expected_inclusion_set_id=None,
                expected_n_slices=cineB.n_slices,
            )
        assert storeB.load_latest_selection(serB) is None
        assert storeB.load_latest_measurement(serB) is None
        assert storeB.load_chain(serB) == []
    finally:
        storeB.close()


def test_return_to_case_a_restores_recorded_selection(tmp_path: Path) -> None:
    """T4: visit B, then return to A — A reloads exactly the recorded state."""
    serA, serB, selA, headA, destA, destB = _two_case_world(tmp_path)

    # Visit case B (switch away): B is active, A untouched.
    appB = WorkstationApplication.bootstrap(destB, port=0, clock=lambda: STAMP, jobs_enabled=False)
    storeB = appB.open_store()
    try:
        assert storeB.load_latest_selection(serB) is None
    finally:
        storeB.close()

    # Return to case A: A's recorded selection AND head restore exactly.
    appA2 = WorkstationApplication.bootstrap(destA, port=0, clock=lambda: STAMP, jobs_enabled=False)
    storeA2 = appA2.open_store()
    try:
        restored = storeA2.load_latest_selection(serA)
        chainA2 = storeA2.load_chain(serA)
    finally:
        storeA2.close()
    assert restored is not None
    assert restored.selection_id == selA.selection_id
    assert (
        (restored.ed_frame, restored.es_frame)
        == (
            selA.ed_frame,
            selA.es_frame,
        )
        == (0, 3)
    )
    assert [r.revision_id for r in chainA2] == [headA]
