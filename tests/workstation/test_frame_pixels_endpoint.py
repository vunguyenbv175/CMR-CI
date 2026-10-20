"""Tests for GET /api/study/<uid>/frame-pixels and series manifest endpoint."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from tests.workstation.support import (
    json_value,
    make_seeded_study,
    request,
    running_server,
)

FROZEN_FRAME_PIXELS_KEYS = frozenset({
    "rows",
    "cols",
    "pixel_spacing",
    "orientation",
    "image_position_patient",
    "rescale_slope",
    "rescale_intercept",
    "bits_allocated",
    "pixel_representation",
    "photometric_interpretation",
    "window_center",
    "window_width",
    "pixel_padding_value",
    "voi_lut",
    "sop_instance_uid",
    "slice_index",
    "frame_index",
    "trigger_time_ms",
})

FROZEN_VOI_LUT_KEYS = frozenset({
    "voi_lut_type",
    "voi_lut_data",
    "voi_lut_explanation",
    "is_signed",
})

FORBIDDEN_PHI_TERMS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "InstitutionName",
)


def test_frame_pixels_allowlist_exact_and_no_store(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    with running_server(study.app) as (server, base_url):
        # 0. Test studies, state, and manifest endpoints for Cache-Control: no-store
        studies_res = request(server, base_url, "/api/studies")
        assert studies_res.status == 200
        assert studies_res.headers.get("Cache-Control") == "no-store"

        state_res = request(server, base_url, f"/api/study/{study.series_uid}/state")
        assert state_res.status == 200
        assert state_res.headers.get("Cache-Control") == "no-store"

        manifest_res = request(server, base_url, f"/api/study/{study.series_uid}/manifest")
        assert manifest_res.status == 200
        assert manifest_res.headers.get("Cache-Control") == "no-store"

        # 1. Test binary response
        url = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=0"
        res = request(server, base_url, url)
        assert res.status == 200
        assert res.headers.get("Cache-Control") == "no-store"
        assert res.headers.get("Content-Type") == "application/octet-stream"

        body = res.body
        assert len(body) > 4
        header_len = struct.unpack(">I", body[:4])[0]
        header_json = body[4 : 4 + header_len].decode("utf-8")
        header = json.loads(header_json)
        pixel_bytes = body[4 + header_len :]

        # Check exact allowlist
        assert set(header.keys()) == FROZEN_FRAME_PIXELS_KEYS

        # Check VOI LUT keys if present
        if header.get("voi_lut") is not None:
            assert set(header["voi_lut"].keys()) == FROZEN_VOI_LUT_KEYS

        # Check expected pixel size
        expected_bytes = header["rows"] * header["cols"] * (header["bits_allocated"] // 8)
        assert len(pixel_bytes) == expected_bytes

        # Verify no PHI
        for term in FORBIDDEN_PHI_TERMS:
            assert term not in header
            assert term.lower() not in header

        # 2. Test format=json response
        url_json = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=0&format=json"
        res_json = request(server, base_url, url_json)
        assert res_json.status == 200
        assert res_json.headers.get("Cache-Control") == "no-store"
        assert res_json.headers.get("Content-Type") == "application/json"
        json_header = json_value(res_json)
        assert set(json_header.keys()) == FROZEN_FRAME_PIXELS_KEYS
        assert json_header == header


def test_frame_pixels_by_sop_uid(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    sop_frame0 = study.cine.frames[0][0].sop_instance_uid
    assert sop_frame0 is not None

    with running_server(study.app) as (server, base_url):
        url = f"/api/study/{study.series_uid}/frame-pixels?sop_uid={sop_frame0}&format=json"
        res = request(server, base_url, url)
        assert res.status == 200
        header = json_value(res)
        assert header["sop_instance_uid"] == sop_frame0
        assert header["frame_index"] == 0

        # Unknown SOP UID -> 404
        bad_url = f"/api/study/{study.series_uid}/frame-pixels?sop_uid=9.9.9.unknown"
        bad_res = request(server, base_url, bad_url)
        assert bad_res.status == 404

        # Out of bounds slice / frame -> 404 or 400
        bad_idx_url = f"/api/study/{study.series_uid}/frame-pixels?slice=99&frame=99"
        bad_idx = request(server, base_url, bad_idx_url)
        assert bad_idx.status in {400, 404}


def test_frame_pixels_voi_lut_and_defaults(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    sop0 = study.cine.frames[0][0].sop_instance_uid
    dicom_files = sorted((tmp_path / "study" / "dicom" / study.series_uid).iterdir())
    target_file = None
    for f in dicom_files:
        d = pydicom.dcmread(str(f))
        if d.SOPInstanceUID == sop0:
            target_file = f
            ds = d
            break
    assert target_file is not None
    item = Dataset()
    item.LUTDescriptor = [4, 10, 16]
    item.LUTData = struct.pack("<4H", 100, 200, 300, 400)
    item.LUTExplanation = "Test LUT"
    ds.VOILUTSequence = Sequence([item])
    ds.save_as(str(target_file))

    with running_server(study.app) as (server, base_url):
        # Frame 0: first frame serves TABLE LUT data
        url0 = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=0&format=json"
        res0 = request(server, base_url, url0)
        assert res0.status == 200
        assert res0.headers.get("Cache-Control") == "no-store"
        header0 = json_value(res0)
        assert header0["voi_lut"] is not None
        assert header0["voi_lut"]["voi_lut_type"] == "TABLE"
        assert header0["voi_lut"]["voi_lut_data"] == [[10, 100], [11, 200], [12, 300], [13, 400]]
        assert header0["voi_lut"]["voi_lut_explanation"] == "Test LUT"

        # Frame 1: subsequent frame omits TABLE LUT data (served once per series)
        url1 = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=1&format=json"
        res1 = request(server, base_url, url1)
        assert res1.status == 200
        assert res1.headers.get("Cache-Control") == "no-store"
        header1 = json_value(res1)
        assert header1["voi_lut"] is not None
        assert header1["voi_lut"]["voi_lut_type"] == "TABLE"
        assert header1["voi_lut"]["voi_lut_data"] is None

        # Frame 1 with explicit include_lut_data=true returns LUT data
        url1_explicit = (
            f"/api/study/{study.series_uid}/frame-pixels"
            "?slice=0&frame=1&format=json&include_lut_data=true"
        )
        res1_explicit = request(server, base_url, url1_explicit)
        assert res1_explicit.status == 200
        header1_explicit = json_value(res1_explicit)
        expected_lut = [[10, 100], [11, 200], [12, 300], [13, 400]]
        assert header1_explicit["voi_lut"]["voi_lut_data"] == expected_lut

        # Binary format for frame 0 and frame 1
        bin_url0 = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=0"
        bin_res0 = request(server, base_url, bin_url0)
        assert bin_res0.status == 200
        assert bin_res0.headers.get("Cache-Control") == "no-store"
        h0_len = struct.unpack(">I", bin_res0.body[:4])[0]
        bin_h0 = json.loads(bin_res0.body[4 : 4 + h0_len].decode("utf-8"))
        assert bin_h0["voi_lut"]["voi_lut_data"] == expected_lut

        bin_url1 = f"/api/study/{study.series_uid}/frame-pixels?slice=0&frame=1"
        bin_res1 = request(server, base_url, bin_url1)
        assert bin_res1.status == 200
        assert bin_res1.headers.get("Cache-Control") == "no-store"
        h1_len = struct.unpack(">I", bin_res1.body[:4])[0]
        bin_h1 = json.loads(bin_res1.body[4 : 4 + h1_len].decode("utf-8"))
        assert bin_h1["voi_lut"]["voi_lut_data"] is None


def test_manifest_endpoint_and_state_manifest(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=1, n_frames=2)
    with running_server(study.app) as (server, base_url):
        # 1. GET /api/study/<uid>/manifest
        man_res = request(server, base_url, f"/api/study/{study.series_uid}/manifest")
        assert man_res.status == 200
        manifest = json_value(man_res)
        assert "geometry" in manifest
        assert "canonical_phase_vector" in manifest
        assert "frame_time_vector" in manifest
        assert "frames_manifest" in manifest
        assert "default_window_level" in manifest

        # Check geometry fields
        geom = manifest["geometry"]
        assert "rows" in geom
        assert "cols" in geom
        assert "pixel_spacing" in geom
        assert "row_cosines" in geom
        assert "column_cosines" in geom
        assert "image_position_patient" in geom
        assert "frame_of_reference_uid" in geom

        # Check frame manifest
        assert len(manifest["frames_manifest"]) == 2
        assert manifest["frames_manifest"][0]["image_id"].startswith("cmr-study://")

        # 2. GET /api/study/<uid>/state?manifest=true
        url_st_man = f"/api/study/{study.series_uid}/state?manifest=true"
        st_man_res = request(server, base_url, url_st_man)
        assert st_man_res.status == 200
        st_manifest = json_value(st_man_res)
        assert "geometry" in st_manifest
        assert "canonical_phase_vector" in st_manifest

        # 3. GET /api/study/<uid>/state without manifest retains legacy keys
        st_res = request(server, base_url, f"/api/study/{study.series_uid}/state")
        assert st_res.status == 200
        st_legacy = json_value(st_res)
        assert "geometry" not in st_legacy
        assert "canonical_phase_vector" not in st_legacy


def test_manifest_unsorted_trigger_times_preserves_canonical_binding(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, n_slices=2, n_frames=3)
    slice0 = study.cine.frames[0]
    slice1 = study.cine.frames[1]
    s0_f0, s0_f1, s0_f2 = slice0[0], slice0[1], slice0[2]
    s1_f0, s1_f1, s1_f2 = slice1[0], slice1[1], slice1[2]

    # Deliberately assign interleaved, unsorted trigger times across slices
    s0_f0.trigger_time_ms = 200.0
    s0_f1.trigger_time_ms = 50.0
    s0_f2.trigger_time_ms = 350.0

    s1_f0.trigger_time_ms = 300.0
    s1_f1.trigger_time_ms = 100.0
    s1_f2.trigger_time_ms = 500.0

    with running_server(study.app) as (server, base_url):
        url = f"/api/study/{study.series_uid}/manifest"
        res = request(server, base_url, url)
        assert res.status == 200
        manifest = json_value(res)

        # Vector is canonical sorted order across all frames of the multi-slice series
        expected_phases = [50.0, 100.0, 200.0, 300.0, 350.0, 500.0]
        assert manifest["canonical_phase_vector"] == expected_phases

        frames = manifest["frames_manifest"]
        assert len(frames) == 6

        expected_binding = [
            (s0_f1, 0, 50.0),
            (s1_f1, 1, 100.0),
            (s0_f0, 0, 200.0),
            (s1_f0, 1, 300.0),
            (s0_f2, 0, 350.0),
            (s1_f2, 1, 500.0),
        ]

        for seq_idx, (frame_obj, expected_slice, expected_tt) in enumerate(expected_binding):
            assert frames[seq_idx]["sop_instance_uid"] == frame_obj.sop_instance_uid
            assert frames[seq_idx]["slice_index"] == expected_slice
            assert frames[seq_idx]["frame_index"] == seq_idx
            assert frames[seq_idx]["trigger_time_ms"] == expected_tt
            assert (
                frames[seq_idx]["image_id"]
                == f"cmr-study://{study.series_uid}/{frame_obj.sop_instance_uid}"
            )

        # Timing intervals match deltas between consecutive canonical phases:
        # [100-50, 200-100, 300-200, 350-300, 500-350, mean(50, 100, 100, 50, 150)]
        expected_dts = [50.0, 100.0, 100.0, 50.0, 150.0, 90.0]
        assert manifest["frame_time_vector"] == expected_dts

