"""Frozen standalone target-authority CLI (CMR_CASE_WORKSPACE_RUNTIME_002 F1).

Exactly TWO frozen operations, consumed by the shell-owned profile-(b)
switch path BEFORE any effect (validate -> terminate -> spawn -> handshake
-> verify -> navigate), and consumed later by runtime-006 by re-invoking
this SAME module with different requests (never a forked copy):

``python -m services.workstation.target_authority resolve --request <json>``
    Ordered physical validation + staged-study reuse verdict.
``python -m services.workstation.target_authority prove --request <json>``
    Identity-witness binding to the owned pid + loopback authority, plus the
    ACTUAL served fields (``/api/studies`` rows + ``/manifest``
    state_projection) whenever the request carries ``study_uid``.

Contract seams (VERIFIED_FROM_SOURCE this cycle):

- The genuine-live marker is EXACTLY ``/tmp/opencode/owner_test_session``
  (the single ``GENUINE_LIVE_DIR`` marker in ``desktop/src-tauri/src/main.rs``).
  No new marker, hostname, or sentinel is invented; the lexical
  equality-or-prefix comparison mirrors ``canonical_study_root`` +
  ``is_genuine_denied`` and runs over RESOLVED physical paths.
- KL-001 single-leading-separator semantics are UNCHANGED: a root-relative
  (drive-less) input is refused, never reinterpreted as absolute.
- Refusal codes reuse the frozen ``library.*`` vocabulary where the
  semantics are identical and add the ``target_authority.*`` family (the
  exact frozen list lives in :data:`REFUSAL_CODES` and the implementation
  report).

Fail-closed contract: any malformed request, unknown operation, missing or
mistyped field, timeout, or internal error produces a single bounded
``{ok:false,...}`` JSON document on stdout (stderr = diagnostics only,
never parsed) and a NONZERO exit. Success prints exactly one
``{ok:true,...}`` JSON document on stdout and exits 0. This module NEVER
spawns, kills, stages, navigates, or writes any file other than its single
stdout document. Its ONLY network access is ``prove``'s bounded loopback
GETs (``/api/studies`` + ``/manifest`` on the OWNED authority the caller
already proved loopback + pid-alive) that produce the contract's
``§exact_identity`` served-fields witness — an unauthenticated GET on the
EXISTING frozen routes under the EXISTING ``authorize_host`` policy (Host
header carried verbatim; no auth weakening, no new route, no mutation).
Pure stdlib (``argparse`` is deliberately NOT used so stdout stays a single
JSON document for every invocation); ``__main__.py`` and ``api.py`` remain
untouched.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

# The ONE genuine-live deny marker, mirrored from main.rs (never extended).
GENUINE_LIVE_DIR = "/tmp/opencode/owner_test_session"
# UID grammar: [A-Za-z0-9._-] with explicit dot-segment refusal for . and ..
UID_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")
# Frozen process bound: mirrors main.rs HANDSHAKE_TIMEOUT (hard cap 15s).
DEFAULT_TIMEOUT_MS = 15000
MAX_TIMEOUT_MS = 15000
# Bounded request payload (argv JSON and the `--request -` stdin form).
MAX_REQUEST_BYTES = 64 * 1024
# Bounded served response (16 MiB bound supports full manifests with frame vectors).
MAX_SERVED_RESPONSE_BYTES = 16 * 1024 * 1024
# Windows namespace prefixes refused before any resolution (main.rs parity).
_NAMESPACE_PREFIXES = ("\\\\?\\", "\\\\.\\")
# Loopback host set, mirrored from main.rs parse_handshake.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# Staged-root layout entries accepted by WorkstationApplication._validate_root_layout.
_STAGED_ROOT_ENTRIES = frozenset({"dicom", "cmr.sqlite3"})

# The FROZEN refusal vocabulary for this contract: `library.*` reuse +
# additive `target_authority.*`. Runtime-006 re-uses this exact list; 002
# never invents codes outside it.
REFUSAL_CODES = (
    "library.bad_path",
    "library.dest_equals_source",
    "library.dest_inside_source",
    "library.source_missing",
    "target_authority.authority_invalid",
    "target_authority.bad_request",
    "target_authority.dest_layout",
    "target_authority.dest_not_staged",
    "target_authority.dest_outside_stage_base",
    "target_authority.genuine_live",
    "target_authority.internal_error",
    "target_authority.namespace_prefix",
    "target_authority.not_loopback",
    "target_authority.pid_not_running",
    "target_authority.root_missing",
    "target_authority.root_relative",
    "target_authority.series_mismatch",
    "target_authority.served_mismatch",
    "target_authority.symlink",
    "target_authority.timeout",
    "target_authority.uid_invalid",
    "target_authority.unsupported_operation",
)


class Refusal(Exception):
    """A frozen fail-closed refusal: ``code`` must be in :data:`REFUSAL_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _ServedMismatch(Exception):
    """Internal: a served-field cross-check failed (mapped to one code)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _Deadline:
    """Best-effort in-process bound (the hard bound is the caller's wait)."""

    def __init__(self, timeout_ms: int) -> None:
        self._at = time.monotonic() + timeout_ms / 1000.0

    def check(self) -> None:
        if time.monotonic() >= self._at:
            raise Refusal(
                "target_authority.timeout",
                "operation exceeded its bounded timeout; refusing fail-closed",
            )


def canonical_study_root(text: str) -> str:
    """Lexical forward-slash canonicalization mirroring main.rs exactly.

    Duplicate slashes collapse, a trailing slash is dropped (except for the
    root), empty and ``.`` segments vanish, and ``..`` pops (a no-op on an
    empty stack, same as ``Vec::pop`` in Rust). Backslashes are NOT
    rewritten: the Rust mirror keeps Windows separators untouched, and this
    module must reach the SAME genuine-live verdict for the SAME text.
    """
    collapsed: list[str] = []
    prev_slash = False
    for ch in text:
        if ch == "/":
            if not prev_slash:
                collapsed.append(ch)
            prev_slash = True
        else:
            collapsed.append(ch)
            prev_slash = False
    out = "".join(collapsed)
    if len(out) > 1 and out.endswith("/"):
        out = out[:-1]
    stack: list[str] = []
    for part in out.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
            continue
        stack.append(part)
    return "/" + "/".join(stack)


def is_genuine_denied(text: str) -> bool:
    """True for the single frozen marker (equality or prefix, never widened)."""
    canon = canonical_study_root(text)
    return canon == GENUINE_LIVE_DIR or canon.startswith(GENUINE_LIVE_DIR + "/")


def _require_str(request: dict[str, Any], key: str) -> str:
    if key not in request:
        raise Refusal("target_authority.bad_request", f"missing required field: {key}")
    value = request[key]
    if not isinstance(value, str) or not value.strip():
        raise Refusal("target_authority.bad_request", f"field {key} must be a non-blank string")
    return value


def _require_pid(request: dict[str, Any], key: str) -> int:
    if key not in request:
        raise Refusal("target_authority.bad_request", f"missing required field: {key}")
    value = request[key]
    # bool is an int subclass in Python: refuse it explicitly (fail-closed).
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Refusal("target_authority.bad_request", f"field {key} must be a positive integer")
    return value


def _require_exact_keys(request: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    missing = sorted(required - set(request))
    if missing:
        raise Refusal(
            "target_authority.bad_request",
            "missing required field: " + ", ".join(missing),
        )
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise Refusal(
            "target_authority.bad_request",
            "unexpected field: " + ", ".join(unknown),
        )


def validate_uid(value: str, field: str) -> str:
    """Frozen UID grammar: [A-Za-z0-9._-]; dot-segments and separators refused."""
    if not UID_RE.match(value) or value in {".", ".."}:
        raise Refusal(
            "target_authority.uid_invalid",
            f"{field} must match [A-Za-z0-9._-] with no dot-segment",
        )
    return value


def check_raw_path_text(text: str, field: str) -> None:
    """Anchored + namespace + root-relative + above-root refusal (KL-001 kept).

    These checks run on the RAW input, before resolution: ``Path.resolve``
    would erase the very evidence they refuse (a namespace prefix collapses,
    a drive-less ``/x`` becomes absolute on POSIX), so a post-resolve check
    would never fire and the boundary would fail open.
    """
    if text.startswith(_NAMESPACE_PREFIXES):
        raise Refusal(
            "target_authority.namespace_prefix",
            f"{field} uses a Windows namespace prefix; refusing fail-closed",
        )
    path = Path(text)
    if not path.is_absolute():
        # KL-001 unchanged: a drive-less rooted input (``/x``) is
        # root-relative, NOT re-interpreted as absolute. A RELATIVE input
        # (``relative/lib``) is refused outright. A drive-less ROOTED input
        # flows through to fully-anchored resolution + genuine-live denial so
        # the single frozen marker (``/tmp/...``) stays enforced for that
        # textual form; any such path that resolves OUTSIDE the marker is
        # still refused root-relative AFTER resolution (fail-closed).
        if not text.startswith("/"):
            raise Refusal(
                "target_authority.root_relative",
                f"{field} must be a fully-anchored absolute path",
            )
    # Above-root refusal: after normalization the input must not climb above
    # its own root (e.g. more leading ".." components than the path has).
    depth = 0
    for part in text.replace("\\", "/").split("/"):
        if part in ("", ".", ":"):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                raise Refusal(
                    "target_authority.root_relative",
                    f"{field} climbs above its root; refusing fail-closed",
                )
            continue
        depth += 1


def _is_junction(path: str) -> bool:
    """Windows junction/reparse detection (os.path.isjunction, stdlib 3.12+)."""
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def reject_links(text: str, field: str) -> None:
    """Symlink/junction fail-closed on EVERY component of the raw input.

    Walks raw components in order WITHOUT pre-normalizing '..' via abspath,
    so an input like '/safe/link/../target' where 'link' is a symlink or junction
    is inspected and refused before parent traversal collapses it.
    """
    drive, rest = os.path.splitdrive(text)
    if drive:
        anchor = drive + os.sep
        raw_parts = [p for p in rest.replace("/", os.sep).split(os.sep) if p]
    elif text.startswith(("/", "\\")):
        cwd_drive = os.path.splitdrive(os.getcwd())[0]
        anchor = (cwd_drive + os.sep) if cwd_drive else "/"
        raw_parts = [p for p in text.replace("/", os.sep).split(os.sep) if p]
    else:
        cwd = os.getcwd()
        cwd_drive, cwd_rest = os.path.splitdrive(cwd)
        anchor = (cwd_drive + os.sep) if cwd_drive else "/"
        raw_parts = [p for p in cwd_rest.replace("/", os.sep).split(os.sep) if p] + [
            p for p in text.replace("/", os.sep).split(os.sep) if p
        ]

    current = anchor
    for part in raw_parts:
        if part == ".":
            continue
        if part == "..":
            parent = os.path.dirname(current.rstrip(os.sep))
            current = parent if parent else anchor
            continue
        current = os.path.join(current, part)
        if os.path.islink(current) or _is_junction(current):
            raise Refusal(
                "target_authority.symlink",
                f"{field} contains a symlink or junction component; refusing fail-closed",
            )


def resolve_path(text: str, field: str) -> Path:
    """Anchored resolution + the single frozen genuine-live marker.

    Order (fail-closed): raw anchored/namespace/above-root refusal ->
    genuine-live on the RAW input text (this is EXACTLY how ``main.rs``
    ``resolve_study_root`` applies ``is_genuine_denied`` — on the candidate
    text, before any filesystem access; on Windows the POSIX spelling
    ``/tmp/...`` is rewritten by resolution, so the raw-text form is the
    one that fires) -> fully-anchored ``Path.resolve`` -> genuine-live on
    the RESOLVED physical path (the contract's resolved-path seam) ->
    post-resolved absolute refusal (KL-001 kept for drive-less inputs that
    resolved outside the marker).
    """
    check_raw_path_text(text, field)
    if is_genuine_denied(text):
        raise Refusal(
            "target_authority.genuine_live",
            f"{field} matches the genuine-live marker; refusing fail-closed",
        )
    if os.name == "nt" and not Path(text).is_absolute():
        # Drive-less rooted input on Windows (e.g. '/synth-lib') begins with '/'
        # and has no drive letter. Path.resolve() would supply the current drive,
        # converting it to an absolute path and bypassing post-resolve checks.
        # Refused here AFTER genuine-live denial on raw text (fail-closed ordering).
        raise Refusal(
            "target_authority.root_relative",
            f"{field} must be a fully-anchored absolute path",
        )
    try:
        resolved = Path(text).resolve()
    except (OSError, RuntimeError) as exc:
        raise Refusal(
            "target_authority.bad_request",
            f"{field} could not be resolved: {type(exc).__name__}",
        ) from exc
    if is_genuine_denied(str(resolved)):
        raise Refusal(
            "target_authority.genuine_live",
            f"{field} resolves into the genuine-live marker; refusing fail-closed",
        )
    if not resolved.is_absolute():
        # A drive-less rooted input (``/x``) that resolved OUTSIDE the
        # frozen marker can never anchor on Windows: KL-001 unchanged,
        # refused here, AFTER genuine-live denial (fail-closed ordering).
        raise Refusal(
            "target_authority.root_relative",
            f"{field} must be a fully-anchored absolute path",
        )
    return resolved


def _sha256_text(text: str) -> str:
    return hashlib.sha256(os.fsencode(text)).hexdigest()


class _TransportReset(Exception):
    """A loopback connection RESET (the documented Windows RST flake)."""


def _served_json(host: str, port: str, path: str, *, deadline: _Deadline) -> Any:
    """GET one JSON document from the OWNED loopback authority (stdlib only).

    Retried ONLY on a connection reset (this host's documented transient
    ``WinError 10054`` loopback flake — a read-only idempotent GET); semantic
    failures (non-200, bad JSON, oversize, shape) NEVER retry and fail closed
    on the first observation.
    """
    last_reset: str | None = None
    for attempt in range(5):
        try:
            return _served_json_once(host, port, path, deadline=deadline)
        except _TransportReset as exc:
            last_reset = str(exc)
            if attempt < 4:
                time.sleep(0.1)
    raise _ServedMismatch(
        f"served check reset repeatedly on the loopback connection ({last_reset})"
    )


def _served_json_once(host: str, port: str, path: str, *, deadline: _Deadline) -> Any:
    """One bounded loopback GET + JSON parse (``_served_json`` retry unit).

    The socket is opened ONLY to the loopback host + bound port the caller
    already proved (``_parse_authority`` + pid liveness run BEFORE this
    function is called), with ``Host: <host>:<port>`` so the frozen server
    policy (``authorize_host``: exactly one valid Host in the allow-list)
    applies unchanged — no auth weakening, no new surface. Bounded read
    (16 MiB) under the caller's deadline; any transport/shape failure maps
    to ``target_authority.served_mismatch`` (fail-closed before navigation).
    """
    import socket

    deadline.check()
    target = (host, int(port))
    try:
        sock = socket.create_connection(target, timeout=5)
    except ConnectionResetError as exc:
        raise _TransportReset(f"connect reset ({type(exc).__name__})") from exc
    except (OSError, ValueError) as exc:
        raise _ServedMismatch(
            f"served check could not reach the owned authority ({type(exc).__name__})"
        ) from exc
    body = b""
    content_length: int | None = None
    try:
        request_line = f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode("ascii")
        try:
            sock.sendall(request_line)
        except ConnectionResetError as exc:
            raise _TransportReset(f"send reset ({type(exc).__name__})") from exc
        except (OSError, ValueError) as exc:
            raise _ServedMismatch(f"served check send failed ({type(exc).__name__})") from exc
        sock.settimeout(5)
        chunks: list[bytes] = []
        total = 0
        body_len = 0
        while True:
            deadline.check()
            try:
                chunk = sock.recv(65536)
            except ConnectionResetError as exc:
                if content_length is not None and body_len >= content_length:
                    break
                raise _TransportReset(f"read reset ({type(exc).__name__})") from exc
            except (OSError, ValueError) as exc:
                if content_length is not None and body_len >= content_length:
                    break
                raise _ServedMismatch(f"served check read failed ({type(exc).__name__})") from exc
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SERVED_RESPONSE_BYTES:
                raise _ServedMismatch("served response exceeds the bounded size")
            chunks.append(chunk)

            if content_length is None:
                joined = b"".join(chunks)
                if b"\r\n\r\n" in joined:
                    h, b = joined.split(b"\r\n\r\n", 1)
                    body_len = len(b)
                    for line in h.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            try:
                                content_length = int(line.split(b":", 1)[1].strip())
                            except ValueError:
                                pass
                            break
            else:
                body_len += len(chunk)

            if content_length is not None and body_len >= content_length:
                break
        body = b"".join(chunks)
    finally:
        try:
            sock.close()
        except OSError:
            pass
    try:
        head, payload = body.split(b"\r\n\r\n", 1)
    except ValueError as exc:
        raise _ServedMismatch("served response has no header/body split") from exc
    if content_length is not None:
        payload = payload[:content_length]
    try:
        status = head.split(b"\r\n", 1)[0].decode("latin-1")
    except (ValueError, UnicodeDecodeError) as exc:
        raise _ServedMismatch("served response has no status line") from exc
    parts = status.split(" ")
    status_code = parts[1] if len(parts) > 1 else ""
    if not status.startswith("HTTP/1.") or status_code != "200":
        raise _ServedMismatch(f"served check got non-200 status ({status_code or '?'})")
    try:
        return json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise _ServedMismatch(f"served body is not a JSON document ({type(exc).__name__})") from exc


def _check_served_witness(
    host: str,
    port: str,
    requested_study: str | None,
    census: dict[str, Any],
    *,
    deadline: _Deadline,
) -> dict[str, Any]:
    """Cross-check the staged census against the ACTUAL served study identity.

    Fetches ``GET /api/studies`` series rows + the per-series
    ``/manifest`` state_projection from the OWNED authority and binds the
    frozen manifest fields (series_uid, study_uid, workflow_id,
    manifest_digest, head_revision_id, measurement_id — values may be null,
    exact types beyond series/study UID are not asserted) to the requested
    study. With ``requested_study`` the served study must equal it; without
    it (startup seam) the served series must still agree with the staged
    census and resolve to exactly ONE study uid. Any mismatch or any
    transport failure maps to ``target_authority.served_mismatch``: the
    filesystem census alone cannot read the study->series binding, so only
    the served fields prove it, BEFORE any navigation.
    """
    series_uids = list(census["series_uids"])
    if not series_uids:
        raise Refusal(
            "target_authority.dest_not_staged",
            "resolved_root has no staged series directories",
        )
    try:
        rows = _served_json(host, port, "/api/studies", deadline=deadline)
        if not isinstance(rows, list):
            raise _ServedMismatch("/api/studies did not return a row list")
        served_uids: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("uid"), str):
                raise _ServedMismatch("/api/studies row is missing a string uid")
            served_uids.append(row["uid"])
        served_uids.sort()
        if served_uids != sorted(series_uids):
            raise _ServedMismatch(
                f"served series rows {served_uids} disagree with the staged census"
            )
        projections: list[dict[str, Any]] = []
        served_studies: set[str] = set()
        for series_uid in sorted(series_uids):
            projection = _served_json(
                host, port, f"/api/study/{series_uid}/manifest", deadline=deadline
            )
            if not isinstance(projection, dict):
                raise _ServedMismatch(f"manifest for {series_uid} is not an object")
            for key in (
                "series_uid",
                "study_uid",
                "workflow_id",
                "manifest_digest",
                "head_revision_id",
                "measurement_id",
            ):
                if key not in projection:
                    raise _ServedMismatch(f"manifest for {series_uid} is missing field {key}")
            if projection["series_uid"] != series_uid:
                raise _ServedMismatch(f"manifest series_uid disagrees for {series_uid}")
            study_value = projection["study_uid"]
            if not isinstance(study_value, str) or not study_value.strip():
                raise _ServedMismatch(f"manifest study_uid is blank for {series_uid}")
            served_studies.add(study_value)
            projections.append(
                {
                    key: projection[key]
                    for key in (
                        "series_uid",
                        "study_uid",
                        "workflow_id",
                        "manifest_digest",
                        "head_revision_id",
                        "measurement_id",
                    )
                }
            )
        if len(served_studies) != 1:
            raise _ServedMismatch("served series do not resolve to exactly one study uid")
        served_study = next(iter(served_studies))
        if requested_study is not None and served_study != requested_study:
            raise _ServedMismatch(f"served study_uid {served_study!r} is not the requested study")
    except _ServedMismatch as exc:
        raise Refusal("target_authority.served_mismatch", exc.message) from exc
    return {"study_uid": served_study, "series_uids": served_uids, "projections": projections}


def census_staged_root(root: Path) -> dict[str, Any]:
    """Staged ``dicom/<SeriesUID>/`` layout census (UIDs + counts only).

    Mirrors ``WorkstationApplication._validate_root_layout`` +
    ``_validate_series_layout``: only ``dicom/`` (and an optional regular
    ``cmr.sqlite3``) at the root, one level of real non-symlink series
    directories named by valid UIDs, files only. Runs AFTER resolution and
    genuine-live denial, BEFORE any reuse verdict.
    """
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except (OSError, RuntimeError) as exc:
        raise Refusal(
            "target_authority.dest_layout",
            f"staged root is unreadable ({type(exc).__name__})",
        ) from exc
    unexpected = sorted(set(entries) - _STAGED_ROOT_ENTRIES)
    if unexpected:
        raise Refusal(
            "target_authority.dest_layout",
            "staged root contains unexpected entries: " + ", ".join(unexpected[:3]),
        )
    dicom = entries.get("dicom")
    if dicom is None or not dicom.is_dir() or dicom.is_symlink() or _is_junction(str(dicom)):
        raise Refusal(
            "target_authority.dest_layout",
            "staged root requires a real dicom directory",
        )
    database = entries.get("cmr.sqlite3")
    if database is not None and (not database.is_file() or database.is_symlink()):
        raise Refusal(
            "target_authority.dest_layout",
            "cmr.sqlite3 must be a regular file",
        )
    series_uids: list[str] = []
    file_count = 0
    try:
        series_dirs = sorted(dicom.iterdir(), key=lambda item: item.name)
    except (OSError, RuntimeError) as exc:
        raise Refusal(
            "target_authority.dest_layout",
            f"staged dicom directory is unreadable ({type(exc).__name__})",
        ) from exc
    for series_dir in series_dirs:
        name = series_dir.name
        if (
            not series_dir.is_dir()
            or series_dir.is_symlink()
            or _is_junction(str(series_dir))
            or not UID_RE.match(name)
            or name in {".", ".."}
        ):
            raise Refusal(
                "target_authority.dest_layout",
                "dicom children must be real series directories named by UID",
            )
        series_uids.append(name)
        try:
            children = list(series_dir.iterdir())
        except (OSError, RuntimeError) as exc:
            raise Refusal(
                "target_authority.dest_layout",
                f"staged series directory is unreadable ({type(exc).__name__})",
            ) from exc
        for child in children:
            if child.is_dir() or child.is_symlink() or _is_junction(str(child)):
                raise Refusal(
                    "target_authority.dest_layout",
                    "series directories must contain files only, without nesting",
                )
            file_count += 1
    return {
        "study_uid": None,
        "series_uids": series_uids,
        "file_count": file_count,
    }


def _process_alive(pid: int) -> bool:
    """Portable liveness probe (NEVER os.kill on Windows: it would KILL)."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if handle:
            code = wintypes.DWORD(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            kernel32.CloseHandle(handle)
            if not ok:
                return False
            return code.value == 259  # STILL_ACTIVE
        error = ctypes.get_last_error()
        # ERROR_ACCESS_DENIED (5): the process exists but is not queryable.
        return error == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parse_authority(authority: str) -> tuple[str, str]:
    """Parse + loopback-validate ``host:port`` / ``[v6]:port`` (main.rs parity)."""
    if authority.startswith("["):
        rest = authority[1:]
        end = rest.find("]")
        if end == -1:
            raise Refusal(
                "target_authority.authority_invalid",
                "authority is not a valid host:port",
            )
        host = rest[:end]
        port = rest[end + 1 :]
        if not port.startswith(":"):
            raise Refusal(
                "target_authority.authority_invalid",
                "authority is not a valid host:port",
            )
        port = port[1:]
    else:
        if authority.count(":") != 1:
            raise Refusal(
                "target_authority.authority_invalid",
                "authority is not a valid host:port",
            )
        host, port = authority.rsplit(":", 1)
    if not host or not port:
        raise Refusal(
            "target_authority.authority_invalid",
            "authority is not a valid host:port",
        )
    if not port.isdigit():
        raise Refusal(
            "target_authority.authority_invalid",
            "authority port is not a number",
        )
    if int(port) == 0 or int(port) > 65535:
        raise Refusal(
            "target_authority.authority_invalid",
            "authority port is out of range",
        )
    normalized = host.lower()
    if normalized not in _LOOPBACK_HOSTS:
        raise Refusal(
            "target_authority.not_loopback",
            "authority host is not loopback; refusing fail-closed",
        )
    return normalized, port


def resolve_target(
    source_root: str,
    stage_base: str,
    study_uid: str,
    *,
    deadline: _Deadline | None = None,
) -> dict[str, Any]:
    """Ordered physical validation + staged-study reuse verdict (frozen order).

    1. UID validation (``[A-Za-z0-9._-]``, dot-segment refusal).
    2. RAW anchored/namespace/root-relative/above-root refusal (KL-001 kept:
       root-relative is refused, never reinterpreted).
    3. Fully-anchored ``Path.resolve`` on every input (source, stage-base).
    4. Genuine-live denial on source, stage-base AND the independently
       derived dest (``stage_base/study_uid``), over RESOLVED paths.
    5. Symlink/junction fail-closed on every raw component.
    6. Resolved equality/containment (dest vs source vs stage-base).
    7. Staged-layout identity check (``dest/dicom/<SeriesUID>/`` census +
       staged-presence verdict) -> reuse verdict BEFORE any source touch.
       Pre-spawn the CLI proves the staged LAYOUT and counts series; the
       study->series binding is proven later by the shell against the
       ACTUAL served fields (``/api/studies`` + ``/manifest``) before any
       navigation, because a filesystem census alone cannot read it.

    Success returns the frozen ``{ok, reuse, dest_root, resolved,
    root_digest, identity}`` shape. Refusals raise :class:`Refusal`.
    """
    if deadline is None:
        deadline = _Deadline(MAX_TIMEOUT_MS)
    validate_uid(study_uid, "study_uid")
    source_text = source_root
    stage_text = stage_base
    check_raw_path_text(source_text, "source_root")
    check_raw_path_text(stage_text, "stage_base")

    source = resolve_path(source_text, "source_root")
    stage = resolve_path(stage_text, "stage_base")
    raw_dest = stage / study_uid

    # Validate raw derived destination against links BEFORE canonicalizing it (Finding 4):
    reject_links(source_text, "source_root")
    reject_links(stage_text, "stage_base")
    reject_links(str(raw_dest), "dest")

    dest = raw_dest.resolve()
    if is_genuine_denied(str(dest)):
        raise Refusal(
            "target_authority.genuine_live",
            "dest resolves into the genuine-live marker; refusing fail-closed",
        )

    # Resolved equality/containment before any reuse/stage/open decision.
    if dest == source:
        raise Refusal(
            "library.dest_equals_source",
            "dest must not equal the source root",
        )
    if source in dest.parents:
        raise Refusal(
            "library.dest_inside_source",
            "dest must not lie inside the source root",
        )
    if stage not in dest.parents:
        raise Refusal(
            "target_authority.dest_outside_stage_base",
            "dest must lie strictly inside the stage base",
        )

    dest_exists = dest.exists() or dest.is_symlink()
    if dest_exists:
        if not dest.is_dir():
            raise Refusal(
                "target_authority.dest_layout",
                "dest exists but is not a real directory",
            )
        deadline.check()
        identity = census_staged_root(dest)
        if not identity["series_uids"]:
            raise Refusal(
                "target_authority.dest_not_staged",
                "dest has no staged series directories",
            )
        identity["study_uid"] = study_uid
        reuse = True
    else:
        # Nothing staged: the source must exist or there is nothing to open.
        if not source.is_dir():
            raise Refusal(
                "library.source_missing",
                "source root is missing and no staged copy exists",
            )
        identity = {"study_uid": study_uid, "series_uids": [], "file_count": 0}
        reuse = False

    return {
        "ok": True,
        "reuse": reuse,
        "dest_root": str(dest),
        "resolved": {
            "source": str(source),
            "stage_base": str(stage),
            "dest": str(dest),
        },
        "root_digest": _sha256_text(str(dest)),
        "identity": identity,
    }


def prove_identity(
    resolved_root: str,
    authority: str,
    pid: int,
    series_uid: str | None = None,
    study_uid: str | None = None,
    *,
    deadline: _Deadline | None = None,
) -> dict[str, Any]:
    """Bind the identity witness to the OWNED pid + loopback authority.

    Frozen order: raw anchored/namespace refusal -> fully-anchored
    resolution + genuine-live denial -> symlink fail-closed -> existence ->
    staged layout census -> authority parse + loopback -> pid liveness ->
    optional named-series presence in the staged layout -> ACTUAL served
    fields cross-check (``/api/studies`` rows + per-series ``/manifest``
    state_projection vs the staged census). With ``study_uid`` the served
    study must equal it; without it (startup seam, no ``resolve`` verdict
    yet) the served series must still match the census and resolve to
    exactly ONE study uid.

    The witness carries the canonical physical-root digest, the staged
    census, and the served projections (UIDs + counts + ids only: no
    credentials, no PHI). The READY line's authority/port/pid remains the
    CHANNEL binding; this witness is the IDENTITY the shell must cross-check
    against the pre-effect ``resolve`` verdict before any navigation.
    """
    if deadline is None:
        deadline = _Deadline(MAX_TIMEOUT_MS)
    root = resolve_path(resolved_root, "resolved_root")
    reject_links(resolved_root, "resolved_root")
    if not root.is_dir():
        raise Refusal(
            "target_authority.root_missing",
            "resolved_root is not a real directory",
        )
    deadline.check()
    identity = census_staged_root(root)
    host, port = _parse_authority(authority)
    if not _process_alive(pid):
        raise Refusal(
            "target_authority.pid_not_running",
            "owned pid is not running; refusing fail-closed",
        )
    if series_uid is not None:
        validate_uid(series_uid, "series_uid")
        if series_uid not in identity["series_uids"]:
            raise Refusal(
                "target_authority.series_mismatch",
                "requested series is not present in the staged layout",
            )
    study_text: str | None = None
    if study_uid is not None:
        study_text = validate_uid(study_uid, "study_uid")
    # The served-fields witness ALWAYS runs (startup too): census agreement +
    # a single served study uid are the identity the shell navigates to.
    served = _check_served_witness(host, port, study_text, identity, deadline=deadline)
    identity["study_uid"] = served["study_uid"]
    return {
        "ok": True,
        "witness": {
            "root_digest": _sha256_text(str(root)),
            "authority": f"http://{host}:{port}/",
            "pid": pid,
            "series_uid": series_uid,
            "study_uid": study_text,
            "identity": identity,
            "served": served,
        },
    }


def _load_request(raw: str | None) -> dict[str, Any]:
    """Exactly one bounded JSON document; anything else is bad_request."""
    if raw is None:
        raise Refusal(
            "target_authority.bad_request",
            "missing required field: --request",
        )
    if raw == "-":
        data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise Refusal(
                "target_authority.bad_request",
                "request exceeds the bounded size",
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Refusal(
                "target_authority.bad_request",
                "request is not valid UTF-8",
            ) from exc
    else:
        encoded = raw.encode("utf-8", errors="strict")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise Refusal(
                "target_authority.bad_request",
                "request exceeds the bounded size",
            )
        text = raw
    try:
        request = json.loads(text)
    except ValueError as exc:
        raise Refusal(
            "target_authority.bad_request",
            "request is not exactly one JSON document",
        ) from exc
    if not isinstance(request, dict):
        raise Refusal(
            "target_authority.bad_request",
            "request must be a JSON object",
        )
    return request


def _parse_timeout_ms(raw: str | None) -> int:
    if raw is None:
        return DEFAULT_TIMEOUT_MS
    if not raw.isdigit() or int(raw) == 0 or int(raw) > MAX_TIMEOUT_MS:
        raise Refusal(
            "target_authority.bad_request",
            f"--timeout-ms must be 1..{MAX_TIMEOUT_MS}",
        )
    return int(raw)


def _dispatch(operation: str, argv: list[str]) -> dict[str, Any]:
    if operation not in {"resolve", "prove"}:
        raise Refusal(
            "target_authority.unsupported_operation",
            "operation must be `resolve` or `prove`",
        )
    raw_request: str | None = None
    raw_timeout: str | None = None
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag not in {"--request", "--timeout-ms"}:
            raise Refusal(
                "target_authority.bad_request",
                f"unexpected argument: {flag}",
            )
        if index + 1 >= len(argv):
            raise Refusal(
                "target_authority.bad_request",
                f"missing value for {flag}",
            )
        value = argv[index + 1]
        if flag == "--request":
            if raw_request is not None:
                raise Refusal(
                    "target_authority.bad_request",
                    "--request given more than once",
                )
            raw_request = value
        else:
            if raw_timeout is not None:
                raise Refusal(
                    "target_authority.bad_request",
                    "--timeout-ms given more than once",
                )
            raw_timeout = value
        index += 2
    timeout_ms = _parse_timeout_ms(raw_timeout)
    deadline = _Deadline(timeout_ms)
    request = _load_request(raw_request)
    if operation == "resolve":
        _require_exact_keys(
            request,
            allowed={"source_root", "stage_base", "study_uid"},
            required={"source_root", "stage_base", "study_uid"},
        )
        deadline.check()
        return resolve_target(
            _require_str(request, "source_root"),
            _require_str(request, "stage_base"),
            _require_str(request, "study_uid"),
            deadline=deadline,
        )
    _require_exact_keys(
        request,
        allowed={"resolved_root", "authority", "pid", "series_uid", "study_uid"},
        required={"resolved_root", "authority", "pid"},
    )
    series_uid = request.get("series_uid")
    if series_uid is not None and (not isinstance(series_uid, str) or not series_uid.strip()):
        raise Refusal(
            "target_authority.bad_request",
            "field series_uid must be a non-blank string when present",
        )
    study_uid = request.get("study_uid")
    if study_uid is not None and (not isinstance(study_uid, str) or not study_uid.strip()):
        raise Refusal(
            "target_authority.bad_request",
            "field study_uid must be a non-blank string when present",
        )
    deadline.check()
    return prove_identity(
        _require_str(request, "resolved_root"),
        _require_str(request, "authority"),
        _require_pid(request, "pid"),
        series_uid,
        study_uid,
        deadline=deadline,
    )


def _emit(document: dict[str, Any]) -> None:
    """The ONLY stdout write: exactly one bounded JSON document + newline."""
    sys.stdout.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """CLI entry: bounded single-JSON-doc stdout; nonzero exit on refusal."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _emit(
            {
                "ok": False,
                "code": "target_authority.unsupported_operation",
                "message": "operation must be `resolve` or `prove`",
            }
        )
        return 2
    operation, rest = args[0], args[1:]
    try:
        document = _dispatch(operation, rest)
    except Refusal as refusal:
        _emit({"ok": False, "code": refusal.code, "message": refusal.message})
        return 2
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        # stderr = diagnostics (never parsed); stdout stays one JSON document.
        import traceback

        traceback.print_exc(file=sys.stderr)
        _emit(
            {
                "ok": False,
                "code": "target_authority.internal_error",
                "message": f"internal error ({type(exc).__name__})",
            }
        )
        return 3
    _emit(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
