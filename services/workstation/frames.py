"""Frozen-renderer frame and cine-bundle transport helpers."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from cmr_viewer.renderer import RenderedFrame, ViewerRenderer, frame_digest
from cmr_viewer.state import ViewerState

__all__ = [
    "CineBundle",
    "build_cine_bundle",
    "canonical_manifest_bytes",
    "decode_cine_bundle",
    "encode_png",
    "frame_etag",
    "cine_etag",
]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def encode_png(canvas: np.ndarray) -> bytes:
    """Encode a contiguous RGB24 canvas with the frozen PNG parameters."""
    array = np.asarray(canvas)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("PNG input must be an HxWx3 uint8 RGB array")
    height, width, _channels = array.shape
    if width < 1 or height < 1:
        raise ValueError("PNG input must have positive width and height")
    rgb = np.ascontiguousarray(array)
    scanlines = b"".join(b"\x00" + rgb[row].tobytes(order="C") for row in range(height))
    compressed = zlib.compress(scanlines, level=6)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        )
    )


def frame_etag(digest: str) -> str:
    """Return the quoted entity tag for one frozen frame digest."""
    return f'"frame-{digest}"'


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize a cine manifest using the exact frozen JSON settings."""
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def cine_etag(body: bytes) -> str:
    """Return the only permitted cine ETag: the exact-body SHA-256."""
    return f'"cine-{hashlib.sha256(body).hexdigest()}"'


@dataclass(frozen=True)
class CineBundle:
    """Exact bundle bytes plus its decoded manifest and rendered frames."""

    body: bytes
    manifest: dict[str, Any]
    rendered_frames: tuple[RenderedFrame, ...]
    png_frames: tuple[bytes, ...]

    @property
    def etag(self) -> str:
        return cine_etag(self.body)


def build_cine_bundle(
    renderer: ViewerRenderer,
    state: ViewerState,
    *,
    slice_index: int | None = None,
) -> CineBundle:
    """Render every frame of one slice using a context without ``frame``.

    The supplied state contributes all view/HUD context.  The only field
    changed for each row is ``current_frame_index`` set to that row's index;
    current playhead position is therefore never a bundle-cache input.
    """
    selected_slice = state.current_slice_index if slice_index is None else slice_index
    if not 0 <= selected_slice < state.cine.n_slices:
        raise ValueError("slice_index is outside the cine stack")
    rendered: list[RenderedFrame] = []
    pngs: list[bytes] = []
    rows: list[dict[str, Any]] = []
    offset = 0
    for frame_index in range(state.cine.n_frames):
        frame = renderer.render(
            replace(
                state,
                current_slice_index=selected_slice,
                current_frame_index=frame_index,
            )
        )
        png = encode_png(frame.canvas)
        rendered.append(frame)
        pngs.append(png)
        rows.append(
            {
                "frame_index": frame_index,
                "digest": frame_digest(frame),
                "byte_offset": offset,
                "byte_length": len(png),
            }
        )
        offset += len(png)
    manifest: dict[str, Any] = {
        "version": 1,
        "media": "application/png-frame-bundle",
        "frames": rows,
    }
    manifest_bytes = canonical_manifest_bytes(manifest)
    body = struct.pack(">Q", len(manifest_bytes)) + manifest_bytes + b"".join(pngs)
    return CineBundle(
        body=body,
        manifest=manifest,
        rendered_frames=tuple(rendered),
        png_frames=tuple(pngs),
    )


def decode_cine_bundle(body: bytes) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """Decode and validate the transport envelope for contract tests/tools."""
    if len(body) < 8:
        raise ValueError("cine bundle is missing its 8-byte manifest length")
    manifest_length = struct.unpack(">Q", body[:8])[0]
    manifest_end = 8 + manifest_length
    if manifest_end > len(body):
        raise ValueError("cine bundle manifest extends past the body")
    manifest_bytes = body[8:manifest_end]
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cine bundle manifest is not UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("cine bundle manifest must be a JSON object")
    if canonical_manifest_bytes(manifest) != manifest_bytes:
        raise ValueError("cine bundle manifest is not canonical JSON")
    if set(manifest) != {"version", "media", "frames"}:
        raise ValueError("cine bundle manifest schema is invalid")
    if manifest["version"] != 1 or manifest["media"] != "application/png-frame-bundle":
        raise ValueError("cine bundle manifest schema is invalid")
    rows = manifest["frames"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("cine bundle manifest frames must be a non-empty array")
    payload = body[manifest_end:]
    pngs: list[bytes] = []
    frame_indexes: set[int] = set()
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "frame_index",
            "digest",
            "byte_offset",
            "byte_length",
        }:
            raise ValueError("cine bundle frame row is invalid")
        try:
            frame_index = row["frame_index"]
            offset = row["byte_offset"]
            length = row["byte_length"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cine bundle frame row is invalid") from exc
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or frame_index in frame_indexes
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
        ):
            raise ValueError("cine bundle frame row is invalid")
        digest = row["digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("cine bundle frame digest is invalid")
        if offset < 0 or length < 1 or offset + length > len(payload):
            raise ValueError("cine bundle frame offset/length is outside payload")
        png = payload[offset : offset + length]
        _validate_png_bytes(png)
        frame_indexes.add(frame_index)
        ranges.append((offset, offset + length))
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("cine bundle frame is not PNG")
        pngs.append(png)
    if frame_indexes != set(range(len(rows))):
        raise ValueError("cine bundle frame indexes are not contiguous")
    covered = sorted(ranges)
    if covered[0][0] != 0 or any(a != b for (_a, b), (a, _b) in zip(covered, covered[1:])):
        raise ValueError("cine bundle frame offsets are not contiguous")
    if covered[-1][1] != len(payload):
        raise ValueError("cine bundle has bytes outside declared PNG frames")
    return manifest, tuple(pngs)


def _validate_png_bytes(png: bytes) -> None:
    """Validate the fixed PNG chunk envelope without a third-party decoder."""
    signature = b"\x89PNG\r\n\x1a\n"
    if not png.startswith(signature):
        raise ValueError("cine bundle frame is not PNG")
    position = len(signature)
    seen_ihdr = False
    seen_iend = False
    while position < len(png):
        if position + 12 > len(png):
            raise ValueError("cine bundle PNG chunk is truncated")
        length = struct.unpack(">I", png[position : position + 4])[0]
        chunk_end = position + 12 + length
        if chunk_end > len(png):
            raise ValueError("cine bundle PNG chunk extends past frame")
        kind = png[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        data = png[data_start:data_end]
        expected_crc = struct.unpack(">I", png[data_end : data_end + 4])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("cine bundle PNG CRC is invalid")
        if kind == b"IHDR":
            if seen_ihdr or length != 13:
                raise ValueError("cine bundle PNG IHDR is invalid")
            seen_ihdr = True
        elif kind == b"IEND":
            if not seen_ihdr or length != 0 or chunk_end != len(png):
                raise ValueError("cine bundle PNG IEND is invalid")
            seen_iend = True
            break
        position = chunk_end
    if not seen_ihdr or not seen_iend:
        raise ValueError("cine bundle PNG is missing IHDR/IEND")
