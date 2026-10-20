"""R1 viewer series-selection API: /api/studies must expose sequence_class.

Owner bug class: the viewer auto-selected the FIRST series in the list (a
3-slice localizer) because /api/studies carried uid/description/frames/slices/
modality only — no series-kind signal. The canonical cine SAX is identified
today only by shape heuristics on the client.

This module pins the additive contract:

1. Every /api/studies row carries a ``sequence_class`` key whose value is the
   value ALREADY stored on the domain series (cmr_dicom.pipeline /
   cmr_dicom.study call classify_series and store it).
2. A series with no stored ``sequence_class`` (legacy / unclassified) yields
   the canonical fallback ``"unknown"`` — never guessed, never omitted.
3. The pre-existing rows keys/values (uid, description, frames, slices,
   modality) are UNCHANGED (additive only, no behavior change).
4. The value flows from the real classifier: a synthetic "cine SAX" study
   classifies as ``"cine_sax"`` end-to-end through the HTTP handler.

Synthetic UIDs only (no PHI). Display/selection only — no contour geometry,
persistence, or inference behavior is exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cmr_dicom.classify import classify_series
from services.workstation.api import make_handler
from services.workstation.server import WorkstationApplication
from tests.workstation.support import make_seeded_study, running_server, request

# Pre-existing /api/studies row keys (pinned before R1) — additive changes only.
LEGACY_ROW_KEYS = {"uid", "description", "frames", "slices", "modality"}


class _FakeCine:
    """Minimal cine stand-in: only what _studies reads (n_frames/n_slices)."""

    def __init__(self, n_slices: int, n_frames: int) -> None:
        self.n_slices = n_slices
        self.n_frames = n_frames


class _FakeSeries:
    """Minimal series stand-in; sequence_class absent by default (legacy path)."""

    def __init__(
        self,
        *,
        series_description: str = "",
        modality: str = "MR",
        sequence_class: str | None = None,
    ) -> None:
        self.series_description = series_description
        self.modality = modality
        if sequence_class is not None:
            # Attribute present only when set (mimics a dataclass default
            # that some legacy constructors never assign).
            self.sequence_class = sequence_class


def _fake_app(series_by_uid: dict[str, Any]) -> WorkstationApplication:
    """Build a WorkstationApplication-shaped app with fake series + cines."""
    app = WorkstationApplication.__new__(WorkstationApplication)
    cine_by_uid = {
        uid: _FakeCine(n_slices=10 if "sax" in uid else 3, n_frames=25 if "sax" in uid else 1)
        for uid in series_by_uid
    }
    object.__setattr__(app, "cine_by_uid", cine_by_uid)
    object.__setattr__(app, "series_by_uid", series_by_uid)
    return app


def _studies_rows(app: WorkstationApplication) -> list[dict[str, Any]]:
    """Drive the real handler's _studies() and decode the JSON body."""
    from http.server import BaseHTTPRequestHandler

    captured: dict[str, bytes] = {}

    class _Capture:
        def send_response(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise AssertionError("_studies must use _send_json, not send_response")

    handler_cls = make_handler(app)

    class _Probe(handler_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:  # bypass socket plumbing
            self.wfile = _Capture()
            self.rfile = None  # type: ignore[assignment]
            self.client_address = ("127.0.0.1", 0)
            self.requestline = "GET /api/studies HTTP/1.1"
            self.request_version = "HTTP/1.1"
            self.command = "GET"
            self.path = "/api/studies"
            self.headers: Any = {}

        def send_response_only(self, *args: Any, **kwargs: Any) -> None:
            return None

        def send_header(self, *args: Any, **kwargs: Any) -> None:
            return None

        def end_headers(self) -> None:
            return None

        def _send_json(self, status: Any, payload: Any, headers: Any = None) -> None:
            captured["body"] = json.dumps(payload).encode("utf-8")
            captured["status"] = str(status).encode("utf-8")

    assert issubclass(handler_cls, BaseHTTPRequestHandler)
    probe = _Probe()  # type: ignore[call-arg]
    probe._studies()
    body = captured.get("body")
    assert body is not None, "_studies produced no JSON body"
    return json.loads(body.decode("utf-8"))


def test_studies_row_carries_sequence_class_from_stored_series() -> None:
    """sequence_class present on the series -> echoed verbatim in the row."""
    sax_uid = "1.2.826.0.1.3680043.10.999.1001.sax"
    loc_uid = "1.2.826.0.1.3680043.10.999.1002.localizer"
    app = _fake_app(
        {
            sax_uid: _FakeSeries(series_description="cine SAX", sequence_class="cine_sax"),
            loc_uid: _FakeSeries(series_description="localizer", sequence_class="unknown"),
        }
    )
    rows = _studies_rows(app)
    by_uid = {row["uid"]: row for row in rows}
    assert set(by_uid) == {sax_uid, loc_uid}
    assert by_uid[sax_uid]["sequence_class"] == "cine_sax"
    assert by_uid[loc_uid]["sequence_class"] == "unknown"


def test_studies_row_sequence_class_defaults_to_unknown_when_absent() -> None:
    """No stored sequence_class (legacy series) -> canonical 'unknown' fallback."""
    uid = "1.2.826.0.1.3680043.10.999.1100.legacy"
    app = _fake_app({uid: _FakeSeries(series_description="legacy series")})
    assert not hasattr(app.series_by_uid[uid], "sequence_class")
    rows = _studies_rows(app)
    assert len(rows) == 1
    assert rows[0]["sequence_class"] == "unknown"


def test_studies_row_is_additive_legacy_fields_unchanged() -> None:
    """Existing keys keep their exact values; sequence_class is the only new key."""
    uid = "1.2.826.0.1.3680043.10.999.1200.additive"
    app = _fake_app(
        {
            uid: _FakeSeries(
                series_description="cine short axis",
                modality="MR",
                sequence_class="cine_sax",
            )
        }
    )
    row = _studies_rows(app)[0]
    assert LEGACY_ROW_KEYS <= set(row)
    assert row["uid"] == uid
    assert row["description"] == "cine short axis"
    assert row["modality"] == "MR"
    # uid contains no "sax" -> the fake cine is the 3x1 (non-cine) shape.
    assert row["frames"] == 1
    assert row["slices"] == 3
    assert set(row) - LEGACY_ROW_KEYS == {"sequence_class"}


def test_classifier_value_flows_from_stored_series_end_to_end(tmp_path: Path) -> None:
    """Real synthetic cine-SAX study -> classifier says cine_sax over HTTP."""
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    with running_server(study.app) as (server, base_url):
        response = request(server, base_url, "/api/studies")
        assert response.status == 200
        rows = json.loads(response.body.decode("utf-8"))
        assert len(rows) == 1
        row = rows[0]
        assert row["uid"] == study.series_uid
        # The stored series must agree with the classifier for this description.
        stored = study.app.series_by_uid[study.series_uid]
        assert row["sequence_class"] == getattr(stored, "sequence_class", "unknown")
        assert row["sequence_class"] == classify_series(
            series_description=stored.series_description,
            protocol_name=getattr(stored, "protocol_name", ""),
            modality=getattr(stored, "modality", "MR"),
        )


def test_sequence_class_is_one_of_canonical_classes() -> None:
    """Never invent class names: the wire value is a canonical class string."""
    from cmr_dicom.classify import classify_series as classify

    canonical = {"cine_sax", "unknown"}
    for description in ("cine SAX", "localizer", "cine 2ch long axis", ""):
        assert classify(series_description=description, modality="MR") in canonical
    uid = "1.2.826.0.1.3680043.10.999.1300.canonical"
    app = _fake_app({uid: _FakeSeries(series_description="cine SAX", sequence_class="cine_sax")})
    assert _studies_rows(app)[0]["sequence_class"] in canonical


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
