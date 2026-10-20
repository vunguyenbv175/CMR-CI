"""HTTP routing for the stdlib workstation server."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import sqlite3
import struct
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import pydicom
from cmr_core.errors import CMRError
from cmr_dicom.discovery import scan_library
from cmr_persistence.store import StaleHeadError
from cmr_quantification.reference_ranges import ReferenceSex
from cmr_viewer.renderer import frame_digest

from .actions import (
    ActionConflict,
    InvalidPayload,
    ViewIntent,
    _best_effort_analysis_jobs,
    all_included_wire,
    apply_checked_action,
    build_evidence_projection,
    build_viewer_state,
    parse_action_intent,
    resolve_study_contours,
    state_projection,
)
from .frames import build_cine_bundle, encode_png
from .library import UNKNOWN_STUDY, scan_to_payload, stage_case
from .recent import (
    RecentError,
    clear_pending_open,
    get_recent_store_path,
    load_pending_open,
    load_recent_cases,
    probe_recent_availability,
    process_recent_post,
    save_pending_open,
)
from .server import (
    AuthorizationError,
    WorkstationApplication,
    authorize_host,
    authorize_mutation,
)
from .target_authority import (
    Refusal,
    census_staged_root,
    check_raw_path_text,
    is_genuine_denied,
    reject_links,
    resolve_path,
)

ALLOWLISTED_FRAME_PIXELS_KEYS: frozenset[str] = frozenset({
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

ALLOWLISTED_VOI_LUT_KEYS: frozenset[str] = frozenset({
    "voi_lut_type",
    "voi_lut_data",
    "voi_lut_explanation",
    "is_signed",
})

LOGGER = logging.getLogger(__name__)

# A refused request is answered before any body byte is trusted. The
# lingering close discards only the bytes already in flight so the socket
# close cannot race the refusal response; both bounds are local to this
# module and independent of the client's advertised Content-Length.
REFUSAL_DRAIN_SECONDS = 0.5
REFUSAL_DRAIN_MAX_BYTES = 4 * 1024 * 1024
REFUSAL_DRAIN_CHUNK_BYTES = 64 * 1024

def _sparse_contours_diagnostic(exc: BaseException) -> str:
    """Path-safe diagnostic: exception type + stable code, never raw text.

    Store read failures can spell absolute paths (database file, weights
    sentinels); the log must carry only the type name and this code.
    """
    return f"{type(exc).__name__}:sparse_contours_unavailable"


def _evidence_diagnostic(exc: BaseException) -> str:
    """Path-safe diagnostic: exception type + stable code, never raw text."""
    return f"{type(exc).__name__}:evidence_unavailable"

__all__ = [
    "ALLOWLISTED_FRAME_PIXELS_KEYS",
    "ALLOWLISTED_VOI_LUT_KEYS",
    "make_handler",
]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _asset(name: str) -> bytes:
    if name not in {"index.html", "app.js", "style.css", "kbmap.js"}:
        raise FileNotFoundError(name)
    return (
        resources.files("services.workstation")
        .joinpath("static")
        .joinpath(name)
        .read_bytes()
    )


FRONTEND_DIST_MIME: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".map": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def _single_query(query: str, name: str) -> str | None:
    from urllib.parse import parse_qs

    values = parse_qs(query, keep_blank_values=True).get(name)
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError(f"query parameter {name!r} must occur exactly once")
    return values[0]


def _series_uid_from_segments(segments: list[str]) -> str:
    if len(segments) < 3 or segments[0] != "api" or segments[1] != "study":
        raise KeyError("route")
    uid = segments[2]
    if not uid or uid in {".", ".."} or "/" in uid or "\\" in uid:
        raise KeyError("series")
    return uid


RECENT_FAMILY_PATHS = frozenset({
    "/api/library/recent",
    "/api/library/recent/probe",
    "/api/library/probe",
    "/api/library/staged",
    "/api/library/recent/pending",
    "/api/library/bound-root",
})


def _authority_host_is_loopback(authority: str) -> bool:
    """True only when the authority's host is provably loopback.

    Understands the normalized ``host:port`` and ``[v6]:port`` forms that
    ``normalize_authority`` produces. DNS names other than localhost and any
    wildcard/non-IP host are NOT loopback (fail-closed for the recent store).
    """
    host = authority
    if host.startswith("["):
        closing = host.find("]")
        host = host[1:closing] if closing != -1 else host
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    host = host.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _recent_loopback_refusal(app: WorkstationApplication, authority: str) -> str | None:
    """Contract acceptance: non-loopback deployment refuses recent routes.

    The recent store exposes remembered source/stage paths, so runtime-006
    pins these routes to loopback on BOTH seams — the configured listener
    and the authorized request authority — on top of (never replacing) the
    existing Host/Origin allow-list check. Returns a stable refusal message
    when the route must not run, None when it may.
    """
    if not _authority_host_is_loopback(str(getattr(app, "host", "") or "")):
        return "recent routes require a loopback listener"
    if not _authority_host_is_loopback(authority):
        return "recent routes require a loopback request authority"
    return None


def _error_payload(error: str, message: str | None = None) -> bytes:
    value: dict[str, Any] = {"error": error}
    if message is not None:
        value["message"] = message
    return _json_bytes(value)


def make_handler(app: WorkstationApplication) -> type[Any]:
    """Return a handler class bound to one bootstrapped application."""

    class WorkstationHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "CMRStudioWorkstation/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            # HTTP request logs may contain route UIDs; the v1 workstation is
            # intentionally quiet and keeps provenance in the application
            # state rather than writing an additional journal.
            return

        def _send(
            self,
            status: int,
            body: bytes = b"",
            *,
            content_type: str = "application/octet-stream",
            headers: Mapping[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _send_json(
            self,
            status: int,
            value: Any,
            headers: dict[str, str] | None = None,
        ) -> None:
            self._send(
                status,
                _json_bytes(value),
                content_type="application/json",
                headers=headers,
            )

        def _forbidden(self, message: str) -> None:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden", "message": message})

        def _refuse_and_close(self, message: str) -> None:
            """Answer a rejected request first, then close the connection.

            The refusal must never wait on client-controlled bytes: a client
            can advertise ``Content-Length`` and withhold the body while
            holding the connection open, so draining before answering parks
            this handler thread (and its socket) indefinitely. The 403 is
            written to the wire first, the connection is marked
            non-reusable, and only then does a strictly time-bounded
            lingering close discard whatever body bytes already arrived, so
            the close cannot reset the socket before the client reads the
            refusal. The drain can delay the close, never the answer.
            """
            self.close_connection = True
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "forbidden", "message": message},
                headers={"Connection": "close"},
            )
            # Flush before the drain so the refusal cannot sit in a writer
            # buffer while the lingering close waits on client bytes.
            self.wfile.flush()
            self._lingering_close()

        def _lingering_close(self) -> None:
            """Best-effort bounded discard of an advertised request body.

            Every read is ONE bounded receive (``read1``) whose socket timeout
            is the remaining budget, and the absolute deadline is re-derived
            from a single start before each read, so neither a withheld body
            nor a byte-TRICKLING body can extend the drain past
            ``REFUSAL_DRAIN_SECONDS``. A buffered ``read(n)`` cannot honour
            that bound: it loops over underlying receives internally, and the
            socket timeout bounds each receive rather than the call, so a
            client trickling one byte per interval shorter than the timeout
            keeps this handler (and its socket) parked past the advertised
            close, with the outer deadline check never reached.
            """
            try:
                raw_length = self.headers.get("Content-Length")
                if not raw_length:
                    return
                length = int(raw_length)
                if length <= 0 or length > REFUSAL_DRAIN_MAX_BYTES:
                    return
                # Fallback: an unbuffered reader has no ``read1``; a raw
                # ``read`` is a single underlying receive there too.
                read_once = getattr(self.rfile, "read1", None) or self.rfile.read
                original_timeout = self.connection.gettimeout()
                try:
                    deadline = time.monotonic() + REFUSAL_DRAIN_SECONDS
                    remaining = length
                    while remaining > 0:
                        budget = deadline - time.monotonic()
                        if budget <= 0:
                            break
                        self.connection.settimeout(budget)
                        chunk = read_once(min(remaining, REFUSAL_DRAIN_CHUNK_BYTES))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                finally:
                    self.connection.settimeout(original_timeout)
            except Exception:
                pass

        def _path(self) -> tuple[str, str]:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if "\x00" in path or "\\" in path:
                raise KeyError("path")
            return path, parsed.query

        def do_GET(self) -> None:  # noqa: N802
            try:
                authority = authorize_host(self.headers, app.allowed_authorities)
            except AuthorizationError as exc:
                self._forbidden(str(exc))
                return
            try:
                path, query = self._path()
                dist_root = getattr(app, "frontend_dist", None)
                if dist_root is not None:
                    if path == "/":
                        self._serve_dist_file(Path(dist_root), "index.html", csp=True)
                        return
                    if path.startswith("/assets/"):
                        # Preserve the assets/ prefix: a vite dist lays out
                        # index.html + assets/<hash>.js, so /assets/<p> maps
                        # to <dist>/assets/<p>.
                        self._serve_dist_file(
                            Path(dist_root), path[1:], csp=False
                        )
                        return
                if path == "/":
                    self._serve_asset("index.html", csp=True)
                    return
                if path.startswith("/static/"):
                    name = path[len("/static/") :]
                    if "/" in name or not name:
                        raise KeyError("static")
                    self._serve_asset(name, csp=False)
                    return
                if path == "/api/studies":
                    self._studies()
                    return
                if path == "/api/library/scan":
                    self._library_scan(query)
                    return
                if path in RECENT_FAMILY_PATHS:
                    refusal = _recent_loopback_refusal(app, authority)
                    if refusal is not None:
                        self._forbidden(refusal)
                        return
                if path == "/api/library/recent":
                    self._library_recent()
                    return
                if path == "/api/library/bound-root":
                    self._library_bound_root()
                    return
                if path == "/api/library/recent/pending":
                    self._library_recent_pending()
                    return
                if path in {"/api/library/recent/probe", "/api/library/probe"}:
                    self._library_recent_probe(query)
                    return
                if path == "/api/library/staged":
                    self._library_staged(query)
                    return
                segments = path.strip("/").split("/")
                if len(segments) == 4 and segments[0:2] == ["api", "study"]:
                    uid = _series_uid_from_segments(segments)
                    resource = segments[3]
                    if resource == "state":
                        self._state(uid, query)
                        return
                    if resource == "jobs":
                        self._jobs(uid)
                        return
                    if resource == "manifest":
                        self._manifest(uid, query)
                        return
                    if resource == "frame-pixels":
                        self._frame_pixels(uid, query)
                        return
                    if resource == "frame":
                        self._frame(uid, query)
                        return
                    if resource == "cine":
                        self._cine(uid, query)
                        return
                    if resource == "contours":
                        self._contours(uid, query)
                        return
                    if resource == "evidence":
                        self._evidence(uid)
                        return
                    if resource in {"report.json", "report.csv", "report.pdf"}:
                        self._report(uid, resource, query)
                        return
                raise KeyError("route")
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": str(exc)})
            except CMRError as exc:
                if getattr(exc, "code", None) == "report.unavailable":
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "report_unavailable"})
                    return
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": getattr(exc, "code", "invalid_request"),
                        "message": str(exc),
                    },
                )
            except (CMRError, ValueError, sqlite3.Error) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": getattr(exc, "code", "invalid_request"),
                        "message": str(exc),
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "message": str(exc)},
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                authority = authorize_mutation(self.headers, app.allowed_authorities)
            except AuthorizationError as exc:
                self._refuse_and_close(str(exc))
                return
            try:
                path, _query = self._path()
                if path == "/api/library/stage":
                    payload = self._read_json_body()
                    self._library_stage(payload)
                    return
                if path in {"/api/library/recent", "/api/library/recent/pending"}:
                    refusal = _recent_loopback_refusal(app, authority)
                    if refusal is not None:
                        self._refuse_and_close(refusal)
                        return
                    try:
                        payload = self._read_json_body()
                    except InvalidPayload as exc:
                        self._recent_body_refusal(str(exc))
                        return
                    except (RecursionError, ValueError):
                        # PR3 review r4310462034: the decoder ITSELF can fail
                        # below the byte cap -- ~6,000 nested arrays raise
                        # RecursionError before check_json_depth can run, and
                        # a JSON integer beyond the interpreter digit limit
                        # raises a plain ValueError. Both are client body
                        # failures, so they take the same frozen
                        # recent.malformed envelope (400 + no-store) as every
                        # other refused recent POST body. Scoped to this
                        # recent boundary on purpose: the pre-existing
                        # metric/action routes keep their frozen envelopes
                        # byte-identical.
                        self._recent_body_refusal(
                            "request body is not decodable JSON"
                        )
                        return
                    if path == "/api/library/recent/pending":
                        self._library_recent_pending_post(payload)
                    else:
                        self._library_recent_post(payload)
                    return
                segments = path.strip("/").split("/")
                if (
                    len(segments) != 4
                    or segments[0:2] != ["api", "study"]
                    or segments[3] != "action"
                ):
                    raise KeyError("route")
                uid = _series_uid_from_segments(segments)
                payload = self._read_json_body()
                self._action(uid, payload)
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": str(exc)})
            except ActionConflict as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": exc.error, "message": str(exc)},
                )
            except InvalidPayload as exc:
                if str(exc).startswith("unknown action:"):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown_action"})
                else:
                    # Path-safe (F5): InvalidPayload texts quote only
                    # client-supplied spellings (grammar diagnostics), never
                    # host paths — the message is echoed so existing
                    # grammar-diagnostic consumers keep working; absolute
                    # paths never appear in these texts (sentinel-tested).
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "invalid_payload", "message": str(exc)},
                    )
            except CMRError as exc:
                if getattr(exc, "code", None) == "report.unavailable":
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "report_unavailable"})
                    return
                # Path-safe (F5): typed code + stable detail, never raw
                # host text. CMRError messages are curated frozen strings
                # (grammar/topology diagnostics); checkpoint spellings never
                # appear in them. The INFERENCE surface (enqueue, execution,
                # terminal-write) uses type-only codes and is sentinel-tested
                # separately — this shared boundary keeps its detail shape
                # for the pre-existing topology/grammar consumers.
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": getattr(exc, "code", "invalid_request"),
                        "message": str(exc),
                    },
                )
            except (ValueError, sqlite3.Error) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": getattr(exc, "code", "invalid_request"),
                    },
                )
            except Exception:  # pragma: no cover - defensive HTTP boundary
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error"},
                )

        def _serve_asset(self, name: str, *, csp: bool) -> None:
            try:
                body = _asset(name)
            except (FileNotFoundError, ModuleNotFoundError):
                raise KeyError("static asset")
            digest = hashlib.sha256(body).hexdigest()[:12]
            headers = {
                "ETag": f'"{digest}"',
                "Cache-Control": "public, max-age=31536000, immutable",
            }
            if csp:
                # Allow data: images (incl. inline-SVG favicon); scripts/styles/objects stay 'self'.
                headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:"
            mime = {
                "index.html": "text/html; charset=utf-8",
                "app.js": "text/javascript; charset=utf-8",
                "style.css": "text/css; charset=utf-8",
                "kbmap.js": "text/javascript; charset=utf-8",
            }[name]
            self._send(HTTPStatus.OK, body, content_type=mime, headers=headers)

        def _serve_dist_file(self, dist_root: Path, rel: str, *, csp: bool) -> None:
            if not rel or rel.startswith("/") or "\\" in rel:
                raise KeyError("dist asset")
            parts = rel.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise KeyError("dist asset")
            suffix = Path(parts[-1]).suffix.lower()
            mime = FRONTEND_DIST_MIME.get(suffix)
            if mime is None:
                raise KeyError("dist asset")
            full = dist_root.joinpath(*parts)
            try:
                resolved = full.resolve()
                root_resolved = dist_root.resolve()
            except OSError:
                raise KeyError("dist asset")
            if resolved != root_resolved and root_resolved not in resolved.parents:
                raise KeyError("dist asset")
            if not resolved.is_file() or resolved.is_symlink():
                raise KeyError("dist asset")
            body = resolved.read_bytes()
            digest = hashlib.sha256(body).hexdigest()[:12]
            headers = {
                "ETag": f'"{digest}"',
                "Cache-Control": "public, max-age=31536000, immutable",
            }
            if csp:
                # Allow data: images (incl. inline-SVG favicon); scripts/styles/objects stay 'self'.
                headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:"
            self._send(HTTPStatus.OK, body, content_type=mime, headers=headers)

        def _studies(self) -> None:
            rows = [
                {
                    "uid": uid,
                    "description": str(getattr(series, "series_description", "")),
                    "frames": cine.n_frames,
                    "slices": cine.n_slices,
                    "modality": str(getattr(series, "modality", "MR")),
                    # Additive series-kind signal (R1 viewer series selection):
                    # the classifier value already stored on the domain series by
                    # cmr_dicom.pipeline (classify_series). One of the canonical
                    # SEQUENCE_CLASSES strings ("cine_sax", "unknown"). Absent on
                    # legacy/mock series -> "unknown" (fail-visible, never guessed).
                    "sequence_class": str(
                        getattr(series, "sequence_class", None) or "unknown"
                    ),
                }
                for uid, cine in sorted(app.cine_by_uid.items())
                for series in [app.series_by_uid[uid]]
            ]
            self._send_json(HTTPStatus.OK, rows, headers={"Cache-Control": "no-store"})

        def _library_scan(self, query: str) -> None:
            """Read-only library scan (slice 2): folder -> Study/Series list.

            Authorization is the unchanged GET policy: ``authorize_host``
            already ran at the top of ``do_GET``. Fail-closed 400 with the
            frozen ``frozen_error``/code shape on any bad path; nothing here
            raises a new envelope through the generic boundary.
            """
            try:
                raw = _single_query(query, "path")
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": str(exc),
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            if raw is None or not raw.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": "query parameter 'path' is required",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            candidate = Path(raw)
            if not candidate.is_absolute() or not candidate.is_dir():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": "scan path must be an existing absolute directory",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            try:
                scan = scan_library(candidate)
            except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": f"scan path is unreadable: {type(exc).__name__}",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                scan_to_payload(scan),
                headers={"Cache-Control": "no-store"},
            )

        def _library_stage(self, payload: Any) -> None:
            """Filesystem-only staging (slice 2): copy one study, no switch.

            Authorization is the unchanged POST policy: ``authorize_mutation``
            already ran at the top of ``do_POST`` (Host + Origin equality —
            strictly stronger than ``authorize_host``, so no weakening).
            This performs filesystem staging ONLY: it never touches
            ``app.cine_by_uid`` / ``app.root``, so the active study cannot
            change here (switch stays the existing respawn path). Unknown
            studies refuse 404; every other staging refusal is 400, all in
            the frozen ``frozen_error``/code shape.
            """
            if not isinstance(payload, dict):
                raise InvalidPayload("stage body must be a JSON object")
            source_path = payload.get("source_path")
            study_uid = payload.get("study_uid")
            dest_path = payload.get("dest_path")
            for name, value in (
                ("source_path", source_path),
                ("study_uid", study_uid),
                ("dest_path", dest_path),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise InvalidPayload(
                        f"stage field {name!r} must be a non-empty string"
                    )
            assert isinstance(source_path, str)
            assert isinstance(study_uid, str)
            assert isinstance(dest_path, str)
            if not Path(source_path).is_absolute() or not Path(
                source_path
            ).is_dir():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": "source path must be an existing absolute directory",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            try:
                scan = scan_library(Path(source_path))
            except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": f"source path is unreadable: {type(exc).__name__}",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            try:
                staged = stage_case(
                    scan, study_uid, dest_path, source_root=source_path
                )
            except CMRError as exc:
                status = (
                    HTTPStatus.NOT_FOUND
                    if exc.code == UNKNOWN_STUDY
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(
                    status,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": str(exc),
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                staged.to_dict(),
                headers={"Cache-Control": "no-store"},
            )

        def _library_bound_root(self) -> None:
            """Read-only bound-root identity for the pinned profile (R7 F5).

            The root THIS backend was spawned against is the genuinely bound
            live root for the page it serves: every initialization learns it
            here, independent of the recent store's health and of any pending
            first-open intent, so a same-root Reopen after a plain reload is
            an anchored no-op instead of an unnecessary kill/respawn.
            Loopback-gated with the recent family (it names a local
            filesystem path); no-store.
            """
            try:
                reject_links(str(app.root), "boundRoot")
                resolved = resolve_path(str(app.root), "boundRoot")
            except Refusal as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "message": str(exc)},
                    headers={"Cache-Control": "no-store"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {"root": str(resolved)},
                headers={"Cache-Control": "no-store"},
            )

        def _library_recent(self) -> None:
            """Read-only recent cases list (slice 6 / runtime-006).

            PR3 review F1: ``load_recent_cases`` acquires the shared
            in-process commit lock, so this snapshot can never overlap a POST
            main-document replace — the read handle cannot starve a valid
            commit (Windows) and no replace can land mid-validation (POSIX).
            """
            store_path = get_recent_store_path(app)
            try:
                doc = load_recent_cases(store_path)
            except RecentError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            headers = {
                "Cache-Control": "no-store",
                "ETag": f'"{doc["digest"]}"',
            }
            body = {
                "version": doc["version"],
                "entries": doc["entries"],
                "reason": doc["reason"],
                "discarded": doc["discarded"],
                "digest": doc["digest"],
            }
            self._send_json(HTTPStatus.OK, body, headers=headers)

        def _library_recent_post(self, payload: Any) -> None:
            """Bounded, tamper-evident recent study persistence (runtime-006)."""
            store_path = get_recent_store_path(app)
            if_match = self.headers.get("If-Match")
            if if_match:
                if_match = if_match.strip('"')
            try:
                doc = process_recent_post(store_path, payload, base_digest_header=if_match)
            except RecentError as exc:
                status = (
                    HTTPStatus.CONFLICT
                    if exc.code == "recent.stale_snapshot"
                    else HTTPStatus.BAD_REQUEST
                )
                err_body: dict[str, Any] = {
                    "error": "frozen_error",
                    "code": exc.code,
                    "message": exc.message,
                }
                if exc.current is not None:
                    err_body["current"] = exc.current
                self._send_json(status, err_body, headers={"Cache-Control": "no-store"})
                return
            except Refusal as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            headers = {
                "Cache-Control": "no-store",
                "ETag": f'"{doc["digest"]}"',
            }
            body = {
                "version": doc["version"],
                "entries": doc["entries"],
                "reason": doc["reason"],
                "discarded": doc.get("discarded", 0),
                "digest": doc["digest"],
            }
            self._send_json(HTTPStatus.OK, body, headers=headers)

        def _library_recent_pending(self) -> None:
            """Read-only pending first-open intent slot (runtime-006).

            Returns the live intent or ``{pending: false}`` when the slot
            is empty/expired. Authorization is the unchanged GET policy.
            """
            store_path = get_recent_store_path(app)
            try:
                intent = load_pending_open(store_path)
            except RecentError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            if intent is None:
                self._send_json(
                    HTTPStatus.OK,
                    {"pending": False},
                    headers={"Cache-Control": "no-store"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "pending": True,
                    "id": intent["id"],
                    "exp": intent["exp"],
                    "entry": intent["entry"],
                },
                headers={"Cache-Control": "no-store"},
            )

        def _recent_body_refusal(self, message: str) -> None:
            """Frozen ``recent.malformed`` envelope for a refused POST body.

            Shared by every recent POST body refusal (invalid payload shapes
            and decoder RESOURCE failures alike) so the route answers one
            frozen 400 with ``no-store`` regardless of which decoder failure
            class the client body triggered.
            """
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "frozen_error",
                    "code": "recent.malformed",
                    "message": message,
                },
                headers={"Cache-Control": "no-store"},
            )

        def _pending_refusal(self, exc: RecentError) -> None:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "frozen_error",
                    "code": exc.code,
                    "message": exc.message,
                },
                headers={"Cache-Control": "no-store"},
            )

        def _library_recent_pending_post(self, payload: Any) -> None:
            """Queue or clear the pending first-open intent slot.

            ``{"entry": {...}}`` (plus ``id``/``exp``) queues the single
            slot (replace); ``{"clear": true}`` (plus optional ``id``)
            drops it with id correlation. Authorization is the unchanged
            POST policy (Host + Origin equality). Shapes are frozen
            ``frozen_error``/code; ``no-store`` throughout.
            """
            store_path = get_recent_store_path(app)
            try:
                if not isinstance(payload, dict):
                    raise RecentError(
                        "recent.malformed", "pending payload must be a JSON object"
                    )
                if payload.get("clear") is True:
                    intent_id = payload.get("id")
                    if intent_id is not None and (
                        not isinstance(intent_id, str) or not intent_id
                    ):
                        raise RecentError(
                            "recent.malformed", "pending clear id is invalid"
                        )
                    cleared = clear_pending_open(store_path, intent_id)
                    self._send_json(
                        HTTPStatus.OK,
                        {"pending": False, "cleared": cleared},
                        headers={"Cache-Control": "no-store"},
                    )
                    return
                saved = save_pending_open(
                    store_path,
                    {
                        "id": payload.get("id"),
                        "exp": payload.get("exp"),
                        "entry": payload.get("entry"),
                    },
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"pending": True, "id": saved["id"], "exp": saved["exp"]},
                    headers={"Cache-Control": "no-store"},
                )
            except (RecentError, Refusal) as exc:
                if isinstance(exc, Refusal):
                    exc = RecentError(exc.code, exc.message)
                self._pending_refusal(exc)

        def _library_recent_probe(self, query: str) -> None:
            """Read-only availability & reuse probe for a case (runtime-006)."""
            from urllib.parse import parse_qs

            params = parse_qs(query, keep_blank_values=True)
            source = (params.get("source") or params.get("source_path") or [None])[0]
            stage = (
                params.get("stage")
                or params.get("stage_base")
                or params.get("stage_base_dir")
                or [None]
            )[0]
            study = (params.get("study") or params.get("study_uid") or [None])[0]

            for name, val in (("source", source), ("stage", stage), ("study", study)):
                if val is None or not val.strip():
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "error": "frozen_error",
                            "code": "library.bad_path",
                            "message": f"query parameter {name!r} is required",
                        },
                        headers={"Cache-Control": "no-store"},
                    )
                    return
            try:
                res = probe_recent_availability(source, stage, study)
            except Refusal as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": f"probe failed: {type(exc).__name__}",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            self._send_json(HTTPStatus.OK, res, headers={"Cache-Control": "no-store"})

        def _library_staged(self, query: str) -> None:
            """Read-only check for staged root census (runtime-006)."""
            try:
                dest_path = _single_query(query, "dest_path")
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": str(exc),
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            if dest_path is None or not dest_path.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": "query parameter 'dest_path' is required",
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            # Physical-path authority BEFORE any filesystem inspection
            # (review P1-3): raw namespace/root checks, the frozen
            # genuine-live denial, and ancestor link refusal — the same
            # policy probe_recent_availability applies to its stage paths.
            # Unsafe paths get a STABLE frozen refusal (400 + code), never
            # a suppressed 200 {staged: false}.
            try:
                check_raw_path_text(dest_path, "dest_path")
                if is_genuine_denied(dest_path):
                    raise Refusal(
                        "target_authority.genuine_live",
                        "dest_path matches the genuine-live marker",
                    )
                resolved = resolve_path(dest_path, "dest_path")
                reject_links(dest_path, "dest_path")
            except Refusal as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                return
            if not resolved.is_dir():
                self._send_json(
                    HTTPStatus.OK,
                    {"staged": False},
                    headers={"Cache-Control": "no-store"},
                )
                return
            try:
                census = census_staged_root(resolved)
                staged = bool(census["series_uids"])
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "staged": staged,
                        "series_uids": census["series_uids"],
                        "file_count": census["file_count"],
                    },
                    headers={"Cache-Control": "no-store"},
                )
            except Refusal as exc:
                # Existing but unsafe/non-staged layout: a stable frozen
                # refusal, never a suppressed 200 staged:false (review P1-3).
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": exc.code,
                        "message": exc.message,
                    },
                    headers={"Cache-Control": "no-store"},
                )
            except Exception:
                # Unexpected census failure: frozen refusal with a type-only
                # message (no raw exception text, no path leak).
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "frozen_error",
                        "code": "library.bad_path",
                        "message": "staged census failed",
                    },
                    headers={"Cache-Control": "no-store"},
                )

        def _jobs(self, uid: str) -> None:
            """Merged series-scoped job feed over BOTH tables (A6).

            Rows from ``analysis_jobs`` (kind ``selected_phase_recompute``)
            and ``inference_jobs`` (kind ``selected_phase_inference``) are
            merged, sorted by ``(created_at, job_id)``, each carrying a
            STABLE ``kind`` field. Every pre-existing row key and the
            response shape are additive (inference rows expose the same
            keys; inference-only diagnostics stay server-side typed codes).
            """
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            store = app.open_store()
            try:
                analysis = store.load_analysis_jobs(uid)
                try:
                    inference = store.load_inference_jobs(uid)
                except Exception:
                    inference = []
                rows: list[dict[str, Any]] = [
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "state": job.state,
                        "error_code": job.error_code,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at,
                        "timings": json.loads(job.timings),
                    }
                    for job in analysis
                ]
                rows.extend(
                    {
                        "job_id": job.job_id,
                        "kind": "selected_phase_inference",
                        "state": job.state,
                        "error_code": job.error_code,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at,
                        "timings": json.loads(job.timings),
                    }
                    for job in inference
                )
                rows.sort(key=lambda row: (row["created_at"], row["job_id"]))
            finally:
                store.close()
            self._send_json(
                HTTPStatus.OK,
                rows,
                headers={"Cache-Control": "no-store"},
            )

        def _state(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            from urllib.parse import parse_qs

            q = parse_qs(query, keep_blank_values=True)
            include_manifest = (
                q.get("manifest") == ["true"]
                or q.get("include_manifest") == ["true"]
            )
            view = app.parse_query_view(uid, query)
            state, _reopened, projection = app.render_state(
                uid, view, include_manifest=include_manifest
            )
            del state
            self._send_json(HTTPStatus.OK, projection, headers={"Cache-Control": "no-store"})

        def _manifest(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            view = app.parse_query_view(uid, query)
            state, _reopened, projection = app.render_state(
                uid, view, include_manifest=True
            )
            del state
            self._send_json(HTTPStatus.OK, projection, headers={"Cache-Control": "no-store"})

        def _frame_pixels(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            from urllib.parse import parse_qs

            q = parse_qs(query, keep_blank_values=True)
            sop_uid = q.get("sop_uid", [None])[0]
            raw_slice = q.get("slice", [None])[0]
            raw_frame = q.get("frame", [None])[0]
            raw_format = q.get("format", [None])[0]

            cine = app.cine_by_uid[uid]
            target_frame = None
            target_slice_idx = -1
            target_frame_idx = -1

            if sop_uid:
                for s_idx, slice_frames in enumerate(cine.frames):
                    for f_idx, frame in enumerate(slice_frames):
                        if frame.sop_instance_uid == sop_uid:
                            target_frame = frame
                            target_slice_idx = s_idx
                            target_frame_idx = f_idx
                            break
                    if target_frame is not None:
                        break
                if target_frame is None:
                    raise KeyError(f"unknown sop_uid: {sop_uid}")
            else:
                try:
                    target_slice_idx = int(raw_slice or 0)
                    target_frame_idx = int(raw_frame or 0)
                except ValueError as exc:
                    raise ValueError("slice and frame must be integers") from exc
                if not (0 <= target_slice_idx < cine.n_slices):
                    raise KeyError(
                        f"slice index {target_slice_idx} out of range (0..{cine.n_slices - 1})"
                    )
                if not (0 <= target_frame_idx < cine.n_frames):
                    raise KeyError(
                        f"frame index {target_frame_idx} out of range (0..{cine.n_frames - 1})"
                    )
                target_frame = cine.frames[target_slice_idx][target_frame_idx]

            geometry = cine.geometries[target_slice_idx]
            resolved_sop = str(
                target_frame.sop_instance_uid
                or f"{uid}.{target_slice_idx}.{target_frame_idx}"
            )
            trigger_time_ms = (
                float(target_frame.trigger_time_ms)
                if target_frame.trigger_time_ms is not None
                else None
            )

            ds = None
            series_dir = app.root / "dicom" / uid
            if series_dir.is_dir():
                dcm_file = None
                if cine.source_manifest is not None:
                    for mf in cine.source_manifest.files:
                        if mf.sop_instance_uid == resolved_sop:
                            cand = app.root / mf.relative_path
                            if cand.is_file():
                                dcm_file = cand
                                break
                            cand = series_dir / Path(mf.relative_path).name
                            if cand.is_file():
                                dcm_file = cand
                                break
                if dcm_file is None:
                    for entry in series_dir.iterdir():
                        if entry.is_file() and resolved_sop in entry.name:
                            dcm_file = entry
                            break
                if dcm_file is None:
                    files = sorted(entry for entry in series_dir.iterdir() if entry.is_file())
                    if files:
                        idx = target_slice_idx * cine.n_frames + target_frame_idx
                        if 0 <= idx < len(files):
                            dcm_file = files[idx]
                        else:
                            dcm_file = files[0]
                if dcm_file is not None:
                    try:
                        ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                    except Exception:
                        ds = None

            rescale_slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0) if ds else 1.0
            rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0) if ds else 0.0
            bits_allocated = int(getattr(ds, "BitsAllocated", 16) or 16) if ds else 16
            pixel_representation = int(getattr(ds, "PixelRepresentation", 0) or 0) if ds else 0
            photometric_interpretation = (
                str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "MONOCHROME2")
                if ds
                else "MONOCHROME2"
            )
            pixel_padding_value = None
            if ds and hasattr(ds, "PixelPaddingValue") and ds.PixelPaddingValue is not None:
                try:
                    pixel_padding_value = int(ds.PixelPaddingValue)
                except (ValueError, TypeError):
                    pixel_padding_value = None

            window_center = None
            window_width = None
            if ds and hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                raw_wc = getattr(ds, "WindowCenter", None)
                raw_ww = getattr(ds, "WindowWidth", None)
                if raw_wc is not None and raw_ww is not None:
                    try:
                        wc_val = float(raw_wc[0] if isinstance(raw_wc, (list, tuple)) else raw_wc)
                        ww_val = float(raw_ww[0] if isinstance(raw_ww, (list, tuple)) else raw_ww)
                        window_center = wc_val
                        window_width = ww_val
                    except (ValueError, TypeError, IndexError):
                        pass

            if window_center is None or window_width is None:
                raw_frame_data = np.asarray(target_frame.pixel_data, dtype=np.float64)
                val_min = float(raw_frame_data.min())
                val_max = float(raw_frame_data.max())
                if val_max <= val_min:
                    val_max = val_min + 1.0
                window_center = float((val_min + val_max + 1.0) / 2.0)
                window_width = float((val_max - val_min) + 1.0)

            table_cache = getattr(app, "_series_table_lut_cache", None)
            if table_cache is None:
                table_cache = {}
                setattr(app, "_series_table_lut_cache", table_cache)

            is_first_frame = (target_slice_idx == 0 and target_frame_idx == 0)
            include_lut_data = (
                is_first_frame
                or q.get("include_lut_data") == ["true"]
                or q.get("include_lut") == ["true"]
            )

            voi_lut = None
            voi_lut_seq = getattr(ds, "VOILUTSequence", None) if ds else None
            voi_lut_func = getattr(ds, "VOILUTFunction", None) if ds else None
            if voi_lut_seq is not None and len(voi_lut_seq) > 0:
                item = voi_lut_seq[0]
                desc = getattr(item, "LUTDescriptor", [0, 0, 16])
                first_val = int(desc[1]) if len(desc) > 1 else 0
                lut_data_raw = getattr(item, "LUTData", [])
                if isinstance(lut_data_raw, (bytes, bytearray)):
                    num_vals = len(lut_data_raw) // 2
                    lut_vals = list(struct.unpack(f"<{num_vals}H", lut_data_raw))
                else:
                    lut_vals = [int(v) for v in lut_data_raw]
                table_data = [[first_val + i, val] for i, val in enumerate(lut_vals)]
                expl = str(getattr(item, "LUTExplanation", "")) or None
                table_cache[uid] = {
                    "table_data": table_data,
                    "explanation": expl,
                }
                voi_lut = {
                    "voi_lut_type": "TABLE",
                    "voi_lut_data": table_data if include_lut_data else None,
                    "voi_lut_explanation": expl,
                    "is_signed": pixel_representation == 1,
                }
            elif uid in table_cache:
                cached = table_cache[uid]
                voi_lut = {
                    "voi_lut_type": "TABLE",
                    "voi_lut_data": cached["table_data"] if include_lut_data else None,
                    "voi_lut_explanation": cached["explanation"],
                    "is_signed": pixel_representation == 1,
                }
            elif voi_lut_func is not None:
                voi_lut = {
                    "voi_lut_type": str(voi_lut_func).strip().upper(),
                    "voi_lut_data": None,
                    "voi_lut_explanation": None,
                    "is_signed": pixel_representation == 1,
                }
            elif ds and hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                voi_lut = {
                    "voi_lut_type": "LINEAR",
                    "voi_lut_data": None,
                    "voi_lut_explanation": None,
                    "is_signed": pixel_representation == 1,
                }
            else:
                voi_lut = {
                    "voi_lut_type": "LINEAR",
                    "voi_lut_data": None,
                    "voi_lut_explanation": None,
                    "is_signed": pixel_representation == 1,
                }

            header: dict[str, Any] = {
                "rows": int(geometry.rows),
                "cols": int(geometry.cols),
                "pixel_spacing": [float(geometry.row_spacing_mm), float(geometry.col_spacing_mm)],
                "orientation": [float(v) for v in geometry.image_orientation_patient],
                "image_position_patient": [float(v) for v in geometry.image_position_patient],
                "rescale_slope": rescale_slope,
                "rescale_intercept": rescale_intercept,
                "bits_allocated": bits_allocated,
                "pixel_representation": pixel_representation,
                "photometric_interpretation": photometric_interpretation,
                "window_center": window_center,
                "window_width": window_width,
                "pixel_padding_value": pixel_padding_value,
                "voi_lut": voi_lut,
                "sop_instance_uid": resolved_sop,
                "slice_index": target_slice_idx,
                "frame_index": target_frame_idx,
                "trigger_time_ms": trigger_time_ms,
            }

            extra_keys = set(header) - ALLOWLISTED_FRAME_PIXELS_KEYS
            if extra_keys:
                raise ValueError(f"unallowlisted keys in frame-pixels header: {extra_keys}")
            if voi_lut is not None:
                extra_voi = set(voi_lut) - ALLOWLISTED_VOI_LUT_KEYS
                if extra_voi:
                    raise ValueError(f"unallowlisted keys in voi_lut: {extra_voi}")

            header_bytes = _json_bytes(header)
            if raw_format == "json" or self.headers.get("Accept") == "application/json":
                self._send(
                    HTTPStatus.OK,
                    header_bytes,
                    content_type="application/json",
                    headers={"Cache-Control": "no-store"},
                )
                return

            if bits_allocated == 16:
                dtype = "<i2" if pixel_representation == 1 else "<u2"
            elif bits_allocated == 8:
                dtype = "u1"
            elif bits_allocated == 32:
                dtype = "<f4"
            else:
                dtype = "<u2"

            pixel_bytes = np.ascontiguousarray(target_frame.pixel_data, dtype=dtype).tobytes()
            body = struct.pack(">I", len(header_bytes)) + header_bytes + pixel_bytes
            self._send(
                HTTPStatus.OK,
                body,
                content_type="application/octet-stream",
                headers={"Cache-Control": "no-store"},
            )

        def _frame(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            from urllib.parse import parse_qs

            query_values = parse_qs(query, keep_blank_values=True)
            if "w" in query_values or "h" in query_values:
                raise ValueError("server-side resize parameters w/h are not supported")
            view = app.parse_query_view(uid, query)
            state, _reopened, _projection = app.render_state(uid, view)
            rendered = app.renderer.render(state)
            digest = frame_digest(rendered)
            self._send(
                HTTPStatus.OK,
                encode_png(rendered.canvas),
                content_type="image/png",
                headers={
                    "ETag": f'"frame-{digest}"',
                    "X-Frame-Digest": digest,
                    "Cache-Control": "no-cache",
                },
            )

        def _cine(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            view = app.parse_query_view(uid, query, cine_context=True)
            state, _reopened, _projection = app.render_state(uid, view)
            bundle = build_cine_bundle(app.renderer, state, slice_index=view.slice_index)
            if self.headers.get("If-None-Match") == bundle.etag:
                self._send(
                    HTTPStatus.NOT_MODIFIED,
                    b"",
                    content_type="application/vnd.cmr.cine-bundle",
                    headers={"ETag": bundle.etag, "Cache-Control": "no-cache"},
                )
                return
            self._send(
                HTTPStatus.OK,
                bundle.body,
                content_type="application/vnd.cmr.cine-bundle",
                headers={"ETag": bundle.etag, "Cache-Control": "no-cache"},
            )

        def _contours(self, uid: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            cine = app.cine_by_uid[uid]
            raw_slice = _single_query(query, "slice")
            raw_frame = _single_query(query, "frame")
            if raw_slice is None or raw_frame is None:
                raise ValueError("slice and frame query parameters are required")
            try:
                slice_index = int(raw_slice)
                frame_index = int(raw_frame)
            except ValueError as exc:
                raise ValueError("slice and frame must be integers") from exc
            if not (0 <= slice_index < cine.n_slices):
                raise ValueError(
                    f"slice index {slice_index} out of cine grid bounds (0..{cine.n_slices - 1})"
                )
            if not (0 <= frame_index < cine.n_frames):
                raise ValueError(
                    f"frame index {frame_index} out of cine grid bounds (0..{cine.n_frames - 1})"
                )
            store = app.open_store()
            try:
                try:
                    result = resolve_study_contours(
                        app.controller,
                        store,
                        uid,
                        cine,
                        slice_index=slice_index,
                        frame_index=frame_index,
                    )
                except CMRError:
                    # Curated typed refusals keep their code and fixed text.
                    raise
                except Exception as exc:
                    # Path-safe (C3/F5): the current-eligible sparse read is
                    # a store read whose failures can spell host paths
                    # (db file, weights sentinels). Map EVERY other failure
                    # to a typed refusal with a fixed message — never raw
                    # exception text.
                    LOGGER.warning(
                        "contours read failed: %s",
                        _sparse_contours_diagnostic(exc),
                    )
                    raise CMRError(
                        "sparse_contours_unavailable",
                        "contour overlay is temporarily unavailable",
                    ) from exc
                self._send_json(HTTPStatus.OK, result, headers={"Cache-Control": "no-store"})
            finally:
                store.close()

        def _evidence(self, uid: str) -> None:
            """Read-only FNC-G1 evidence projection (one new response shape).

            Opens the store, composes existing records, closes it. No writes,
            no job enqueue, no recompute trigger, no geometry bytes. A series
            with no history returns empty sections, never an error; an unknown
            series is a 404 like ``_jobs``.
            """
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            cine = app.cine_by_uid[uid]
            # Complete store-lifecycle sanitization (R1 P1/P2): open, read
            # AND close all sit inside the protected block — an open_store
            # or close failure carries the same fixed typed refusal as a
            # read failure (absolute store paths must never reach the
            # generic GET boundary's str(exc) mapping).
            try:
                store = app.open_evidence_store()
            except Exception as exc:
                LOGGER.warning(
                    "evidence open failed: %s", _evidence_diagnostic(exc)
                )
                raise CMRError(
                    "evidence_unavailable",
                    "evidence projection is temporarily unavailable",
                ) from exc
            try:
                try:
                    projection = build_evidence_projection(
                        app.controller, store, uid, cine
                    )
                except CMRError as exc:
                    # R2 P1: NO typed error passes through with its own
                    # text — a store failure can spell host paths inside ANY
                    # exception, typed or not. Only the curated
                    # report.unavailable refusal keeps its shape; everything
                    # else becomes the fixed evidence_unavailable refusal.
                    if getattr(exc, "code", None) == "report.unavailable":
                        raise
                    LOGGER.warning(
                        "evidence refused: %s", _evidence_diagnostic(exc)
                    )
                    raise CMRError(
                        "evidence_unavailable",
                        "evidence projection is temporarily unavailable",
                    ) from exc
                except Exception as exc:
                    # Path-safe (F5/C3): store read failures can spell host
                    # paths. Map every other failure to a typed refusal with
                    # a fixed message — never raw exception text.
                    LOGGER.warning("evidence read failed: %s", _evidence_diagnostic(exc))
                    raise CMRError(
                        "evidence_unavailable",
                        "evidence projection is temporarily unavailable",
                    ) from exc
                self._send_json(
                    HTTPStatus.OK, projection, headers={"Cache-Control": "no-store"}
                )
            finally:
                try:
                    store.close()
                except Exception as exc:
                    LOGGER.warning(
                        "evidence close failed: %s", _evidence_diagnostic(exc)
                    )

        def _report(self, uid: str, resource: str, query: str) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            raw_sex = _single_query(query, "reference_sex")
            if raw_sex is None:
                sex = None
            elif raw_sex == ReferenceSex.MEN.value:
                sex = ReferenceSex.MEN
            elif raw_sex == ReferenceSex.WOMEN.value:
                sex = ReferenceSex.WOMEN
            else:
                raise ValueError("reference_sex must be men or women")
            fmt, content_type, extension = (
                ("json", "application/json", "json")
                if resource.endswith(".json")
                else ("csv", "text/csv", "csv")
                if resource.endswith(".csv")
                else ("pdf", "application/pdf", "pdf")
            )
            body = app.report_bytes(uid, sex, fmt)
            self._send(
                HTTPStatus.OK,
                body,
                content_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="report.{extension}"'},
            )

        def _read_json_body(self) -> Any:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise InvalidPayload("Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise InvalidPayload("Content-Length is invalid") from exc
            if length < 0 or length > 4 * 1024 * 1024:
                raise InvalidPayload("request body is outside the permitted size")
            try:
                raw = self.rfile.read(length)
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InvalidPayload("request body must be UTF-8 JSON") from exc
            return value

        def _action(self, uid: str, payload: Any) -> None:
            if uid not in app.cine_by_uid:
                raise KeyError("unknown series")
            intent = parse_action_intent(
                payload, app.cine_by_uid[uid], legacy_clinical_path=True
            )
            cancel_already_applied = False
            if (
                intent.action == "cancel_analysis_job"
                and app.job_runner is not None
                and app.jobs_enabled
            ):
                # A6: kind-routed cancel through the existing route.
                # Inference ids cancel via the runner/store inference path
                # (runner falls back across both tables); analysis ids are
                # unchanged. Wrong-series ids change neither row nor token:
                # the runner's scoped store mutation raises KeyError before
                # any token request (request_cancel_token no-ops on unknown
                # ids), and the pre-existing do_POST boundary maps KeyError
                # to 404 without touching state. No new error shape here.
                app.job_runner.cancel(str(intent.params["job_id"]), series_uid=uid)
                cancel_already_applied = True
            store = app.open_store()
            try:
                try:
                    result = apply_checked_action(
                        app.controller,
                        store,
                        uid,
                        app.cine_by_uid[uid],
                        intent,
                        user=app.user,
                        timestamp=app.clock(),
                        legacy_clinical_path=True,
                        jobs_enabled=app.jobs_enabled,
                        cancel_already_applied=cancel_already_applied,
                        x1_weights_root=app.x1_weights_root,
                    )
                except StaleHeadError:
                    # The checked writer has rolled back.  Reopen from the
                    # same request connection after rollback and reset the
                    # client draft to the canonical all-included grid.
                    reset_view = ViewIntent(
                        intent.view.slice_index,
                        intent.view.frame_index,
                        intent.view.fps,
                        intent.view.playing,
                        intent.view.reference_sex,
                        tuple(all_included_wire(app.cine_by_uid[uid].n_slices)),
                    )
                    fresh_state, fresh_reopened = build_viewer_state(
                        app.controller,
                        store,
                        uid,
                        app.cine_by_uid[uid],
                        reset_view,
                    )
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {
                            "error": "stale_view",
                            "client_nonce": intent.client_nonce,
                            "state": state_projection(
                                fresh_state,
                                analysis=fresh_reopened.get("analysis"),
                                analysis_stale_reason=fresh_reopened.get(
                                    "analysis_stale_reason"
                                ),
                            ),
                            "inclusion_draft_reset": True,
                            "inclusion_draft": all_included_wire(
                                app.cine_by_uid[uid].n_slices
                            ),
                        },
                    )
                    return
                except ActionConflict as exc:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": exc.error, "message": str(exc)},
                    )
                    return
                final_rows = tuple(result["inclusion_draft"])
                final_view = ViewIntent(
                    intent.view.slice_index,
                    intent.view.frame_index,
                    intent.view.fps,
                    intent.view.playing,
                    intent.view.reference_sex,
                    final_rows,
                )
                final_state, final_reopened = build_viewer_state(
                    app.controller,
                    store,
                    uid,
                    app.cine_by_uid[uid],
                    final_view,
                )
                result["state"] = state_projection(
                    final_state,
                    analysis_jobs=(
                        _best_effort_analysis_jobs(store, uid)
                        if app.jobs_enabled
                        else None
                    ),
                    analysis=final_reopened.get("analysis"),
                    analysis_stale_reason=final_reopened.get("analysis_stale_reason"),
                )
                rendered = app.renderer.render(final_state)
                result["frame_digest"] = frame_digest(rendered)
                self._send_json(HTTPStatus.OK, result)
            finally:
                store.close()

    return WorkstationHandler
