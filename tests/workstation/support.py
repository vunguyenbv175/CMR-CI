"""Shared synthetic fixtures and stdlib HTTP helpers for S3-008 tests."""

from __future__ import annotations

import json
import struct
import threading
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from cmr_contours.model import Contour
from cmr_core.domain_types import Prediction
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_dicom.pipeline import load_cine_volume
from cmr_qc.snapshot import run_revision_qc
from cmr_quantification.measurements import (
    build_cine_contour_index,
    compute_cavity_volume_curve_ml,
)
from cmr_workflow.controller import WorkflowController

from services.workstation.actions import all_included_wire
from services.workstation.server import (
    WorkstationApplication,
    create_http_server,
)
from tests.synth.generate_dicom import write_synthetic_cine

SERIES = "1.2.826.0.1.3680043.10.999.900"
STUDY = "1.2.826.0.1.3680043.10.999.901"
STAMP = "2026-08-30T00:00:00+00:00"
LATER = "2026-08-30T00:00:01+00:00"


def _rect(x0: float, y0: float, width: float, height: float) -> list[tuple[float, float]]:
    return [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]


def _contour(
    slice_index: int,
    frame_index: int,
    structure: Structure,
    points: list[tuple[float, float]],
) -> Contour:
    return Contour(slice_index, frame_index, structure, points, source="manual")


@dataclass(frozen=True)
class SeededStudy:
    root: Path
    series_uid: str
    app: WorkstationApplication
    cine: Any
    controller: WorkflowController


def make_seeded_study(
    parent: Path,
    *,
    n_slices: int = 1,
    n_frames: int = 2,
    series_uid: str = SERIES,
    study_uid: str = STUDY,
    seed_history: bool = True,
    jobs_enabled: bool = True,
) -> SeededStudy:
    """Create a bound DICOM study and, optionally, a renderable workflow head."""
    root = parent / "study"
    dicom_dir = root / "dicom" / series_uid
    write_synthetic_cine(
        dicom_dir,
        n_slices=n_slices,
        n_frames=n_frames,
        series_uid=series_uid,
        study_uid=study_uid,
    )
    app = WorkstationApplication.bootstrap(
        root, port=0, clock=lambda: STAMP, jobs_enabled=jobs_enabled
    )
    cine = app.cine_by_uid[series_uid]
    controller = app.controller
    if seed_history:
        store = app.open_store()
        try:
            workflow_id = cine.workflow_id
            if workflow_id is None or cine.study_uid is None:
                raise AssertionError("bootstrap did not bind a source workflow")
            provenance = Provenance(
                cine.study_uid,
                series_uid,
                VERSION,
                None,
                STAMP,
                0,
                workflow_id,
            )
            prediction = Prediction(
                cine.study_uid,
                series_uid,
                "manual-seed",
                {
                    Structure.LV_CAVITY: _contour(
                        0, 0, Structure.LV_CAVITY, _rect(18, 16, 20, 18)
                    ),
                    Structure.RV_CAVITY: _contour(
                        0, 0, Structure.RV_CAVITY, _rect(43, 38, 9, 9)
                    ),
                    Structure.LV_MYOCARDIUM: _contour(
                        0, 0, Structure.LV_MYOCARDIUM, _rect(15, 13, 26, 24)
                    ),
                },
                None,
                provenance,
                STAMP,
            )
            store.save_prediction(prediction)
            selection = controller.select_ed_es(
                cine,
                0,
                n_frames - 1,
                "seed",
                STAMP,
                lambda: provenance,
            )
            revisions = [controller.save(store, prediction, {}, "seed", STAMP)]
            current_revision = revisions[0]
            for slice_index in range(n_slices):
                for frame_index in range(n_frames):
                    if slice_index == 0 and frame_index == 0:
                        continue
                    edits = {
                        Structure.LV_CAVITY: _contour(
                            slice_index,
                            frame_index,
                            Structure.LV_CAVITY,
                            _rect(
                                18 + slice_index,
                                16 + frame_index,
                                20 if frame_index == 0 else 16,
                                18 if frame_index == 0 else 14,
                            ),
                        ),
                        Structure.RV_CAVITY: _contour(
                            slice_index,
                            frame_index,
                            Structure.RV_CAVITY,
                            _rect(
                                43 + slice_index,
                                38 + frame_index,
                                9 if frame_index == 0 else 7,
                                9 if frame_index == 0 else 7,
                            ),
                        ),
                        Structure.LV_MYOCARDIUM: _contour(
                            slice_index,
                            frame_index,
                            Structure.LV_MYOCARDIUM,
                            _rect(15 + slice_index, 13 + frame_index, 26, 24),
                        ),
                    }
                    current_revision = controller.save(
                        store, current_revision, edits, "seed", LATER
                    )
                    revisions.append(current_revision)
            index = build_cine_contour_index(revisions, series_uid)
            curve = compute_cavity_volume_curve_ml(index, cine)
            report = run_revision_qc(index, cine, selection, volumes_by_frame=curve)
            controller.save(
                store,
                current_revision,
                {},
                "seed",
                LATER,
                qc_report=report,
            )
        finally:
            store.close()
    return SeededStudy(root, series_uid, app, cine, controller)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: HTTPMessage
    body: bytes


@contextmanager
def running_server(app: WorkstationApplication) -> Iterator[tuple[Any, str]]:
    server = create_http_server(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request(
    server: Any,
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    port = int(server.server_address[1])
    request_headers = {"Host": f"127.0.0.1:{port}"}
    if method == "POST":
        request_headers["Origin"] = f"http://127.0.0.1:{port}"
    request_headers.update(headers or {})
    req = Request(base_url + path, data=body, method=method, headers=request_headers)
    try:
        with urlopen(req, timeout=10) as response:
            return HttpResponse(
                response.status,
                cast(HTTPMessage, response.headers),
                response.read(),
            )
    except HTTPError as error:
        return HttpResponse(error.code, cast(HTTPMessage, error.headers), error.read())


def json_body(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_value(response: HttpResponse) -> Any:
    return json.loads(response.body.decode("utf-8"))


def view_query(
    *,
    slice_index: int = 0,
    frame_index: int = 0,
    fps: int = 10,
    playing: bool = False,
    reference_sex: str | None = None,
    inclusion_draft: list[dict[str, Any]] | None = None,
    include_frame: bool = True,
) -> str:
    values: dict[str, str] = {
        "slice": str(slice_index),
        "fps": str(fps),
        "playing": "true" if playing else "false",
    }
    if include_frame:
        values["frame"] = str(frame_index)
    if reference_sex is not None:
        values["reference_sex"] = reference_sex
    if inclusion_draft is not None:
        values["inclusion_draft"] = json.dumps(
            inclusion_draft,
            sort_keys=True,
            separators=(",", ":"),
        )
    return urlencode(values)


def action_payload(
    state: dict[str, Any],
    *,
    action: str,
    params: dict[str, Any],
    nonce: str,
    draft: list[dict[str, Any]] | None = None,
    frame: int | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "params": params,
        "client_nonce": nonce,
        "view": {
            "slice": state["slice"],
            "frame": state["frame"] if frame is None else frame,
            "fps": state["fps"],
            "playing": state["playing"],
            "reference_sex": state["reference_sex"],
            "inclusion_draft": state["inclusion_draft"] if draft is None else draft,
        },
        "lineage": {
            "head_revision_id": state["head_revision_id"],
            **(
                {"whole_cine_head_id": state["whole_cine_head_id"]}
                if "whole_cine_head_id" in state
                else {}
            ),
            "selection_id": state["selection_id"],
            "inclusion_set_id": state["inclusion_set_id"],
        },
    }


def decode_png_rgb(data: bytes) -> np.ndarray:
    """Decode the first-party filter-0 RGB PNG without Pillow."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("PNG signature is missing")
    position = 8
    width = height = None
    compressed = bytearray()
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += length + 12
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (depth, color_type, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
                raise AssertionError("PNG does not use the frozen RGB24 envelope")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError("PNG IHDR is missing")
    raw = zlib.decompress(bytes(compressed))
    row_length = width * 3
    expected = height * (row_length + 1)
    if len(raw) != expected:
        raise AssertionError("PNG scanline payload length is wrong")
    rows = []
    for row in range(height):
        start = row * (row_length + 1)
        if raw[start] != 0:
            raise AssertionError("PNG row filter is not zero")
        rows.append(np.frombuffer(raw[start + 1 : start + 1 + row_length], dtype=np.uint8))
    return np.stack(rows).reshape((height, width, 3)).copy()


def isolated_tempdir() -> TemporaryDirectory[str]:
    """Keep wheel/build probes outside the repository."""
    return TemporaryDirectory(prefix="s3-008-", dir="/dev/shm")


def all_included(n_slices: int) -> list[dict[str, Any]]:
    return all_included_wire(n_slices)


def source_reload(root: Path, series_uid: str) -> tuple[Any, Any]:
    return load_cine_volume(root / "dicom" / series_uid, series_uid=series_uid)


__all__ = [
    "LATER",
    "SERIES",
    "STAMP",
    "STUDY",
    "SeededStudy",
    "all_included",
    "action_payload",
    "decode_png_rgb",
    "isolated_tempdir",
    "json_body",
    "json_value",
    "make_seeded_study",
    "request",
    "running_server",
    "source_reload",
    "view_query",
]
