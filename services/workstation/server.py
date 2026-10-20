"""Bootstrap, authority policy and HTTP-server construction for the workstation."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cmr_core.errors import CMRError
from cmr_dicom.pipeline import load_cine_volume
from cmr_persistence.store import SCHEMA_USER_VERSION, SqliteCmrStore
from cmr_reporting.model import ReportInput
from cmr_reporting.reproduce import reproduce_report
from cmr_viewer.renderer import ViewerRenderer
from cmr_workflow.controller import WorkflowController

from .actions import (
    ViewIntent,
    build_viewer_state,
    parse_view_intent,
    reject_duplicate_json_keys,
    reject_non_canonical_encoding,
    reject_non_grammar_number_text,
    state_projection,
)
from .jobs import JobRunner, _typed_diagnostic

__all__ = [
    "AuthorizationError",
    "WorkstationApplication",
    "WorkstationConfigurationError",
    "authorize_host",
    "authorize_mutation",
    "bootstrap_application",
    "bound_handshake",
    "create_http_server",
    "normalize_authority",
    "normalize_origin",
    "run",
]


class WorkstationConfigurationError(ValueError):
    """A study root or listener configuration cannot be served safely."""


class AuthorizationError(PermissionError):
    """A request does not satisfy the frozen Host/Origin authority policy."""


LOGGER = logging.getLogger(__name__)

_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _authority_host(host: str) -> tuple[str, bool]:
    if not isinstance(host, str) or not host or any(char.isspace() for char in host):
        raise ValueError("authority host is empty or contains whitespace")
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if any(not label or not _DNS_LABEL.fullmatch(label) for label in labels):
            raise ValueError(f"invalid DNS authority host {host!r}")
        return host.lower(), False
    return parsed.compressed.lower(), parsed.version == 6


def _port(value: str, *, default: int | None, required: bool) -> int:
    if value == "":
        if required or default is None:
            raise ValueError("authority port is required")
        return default
    if not value.isdigit():
        raise ValueError(f"invalid authority port {value!r}")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"authority port {port} is outside 1..65535")
    return port


def normalize_authority(
    authority: str,
    *,
    require_port: bool = True,
    allow_wildcard: bool = False,
) -> str:
    """Normalize one HTTP authority to lowercase DNS/canonical IP form.

    CLI allow-list authorities require an explicit port by default.  Incoming
    Host/Origin values call this function with ``require_port=False`` so an
    omitted port is normalized to HTTP's port 80, exactly as the frozen
    security matrix requires.
    """
    if not isinstance(authority, str) or not authority or any(
        char.isspace() for char in authority
    ):
        raise ValueError("authority must be a non-empty host[:port] string")
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            raise ValueError("bracketed IPv6 authority is missing closing bracket")
        raw_host = authority[1:closing]
        suffix = authority[closing + 1 :]
        if suffix and not suffix.startswith(":"):
            raise ValueError("authority has data after bracketed IPv6 host")
        host, is_ipv6 = _authority_host(raw_host)
        if not is_ipv6:
            raise ValueError("brackets are permitted only around IPv6 authorities")
        if not allow_wildcard and ipaddress.ip_address(raw_host).is_unspecified:
            raise ValueError("wildcard authority is not an allowed browser authority")
        port = _port(suffix[1:] if suffix else "", default=80, required=require_port)
        return f"[{host}]:{port}"
    if ":" in authority:
        if authority.count(":") != 1:
            raise ValueError("IPv6 authorities must use brackets")
        raw_host, raw_port = authority.rsplit(":", 1)
    else:
        raw_host, raw_port = authority, ""
    host, is_ipv6 = _authority_host(raw_host)
    if is_ipv6:
        raise ValueError("IPv6 authorities must use brackets")
    port = _port(raw_port, default=80, required=require_port)
    if not allow_wildcard:
        try:
            if ipaddress.ip_address(raw_host).is_unspecified:
                raise ValueError("wildcard authority is not an allowed browser authority")
        except ValueError as exc:
            if "wildcard authority" in str(exc):
                raise
    return f"{host}:{port}"


def _listen_host(host: str) -> str:
    if not isinstance(host, str) or not host:
        raise WorkstationConfigurationError("listen host must be non-empty")
    candidate = host
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if any(char.isspace() for char in candidate) or "/" in candidate:
        raise WorkstationConfigurationError("listen host is malformed")
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        normalized, is_ipv6 = _authority_host(candidate)
        if is_ipv6:
            return normalized
        return normalized


def _is_wildcard(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_unspecified
    except ValueError:
        return False


def _host_port_authority(host: str, port: int) -> str:
    """Format a normalized listen host as an HTTP authority."""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _headers_values(headers: Any, name: str) -> list[str]:
    if hasattr(headers, "get_all"):
        values = headers.get_all(name)
        if values is not None:
            return [str(value) for value in values]
    if isinstance(headers, Mapping):
        collected: list[str] = []
        for key, value in headers.items():
            if str(key).lower() != name.lower():
                continue
            if isinstance(value, (list, tuple)):
                collected.extend(str(item) for item in value)
            else:
                collected.append(str(value))
        return collected
    try:
        value = headers.get(name)
    except AttributeError:
        return []
    return [] if value is None else [str(value)]


def normalize_origin(origin: str) -> str:
    """Normalize a strict lowercase ``http://authority`` Origin."""
    from urllib.parse import urlsplit

    if not isinstance(origin, str) or origin == "null" or not origin.startswith("http://"):
        raise ValueError("Origin must be a non-null lowercase http origin")
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Origin must have no userinfo, path, query or fragment")
    return normalize_authority(parsed.netloc, require_port=False)


def authorize_host(headers: Any, allowed_authorities: Iterable[str]) -> str:
    """Require exactly one valid Host in the configured authority allow-list."""
    host_values = _headers_values(headers, "Host")
    if len(host_values) != 1:
        raise AuthorizationError("exactly one Host header is required")
    try:
        normalized = normalize_authority(host_values[0], require_port=False)
    except ValueError as exc:
        raise AuthorizationError("Host authority is malformed") from exc
    allowed = set(allowed_authorities)
    if normalized not in allowed:
        raise AuthorizationError("Host authority is not allowed")
    return normalized


def authorize_mutation(headers: Any, allowed_authorities: Iterable[str]) -> str:
    """Authorize a mutation using the exact Host/Origin equality rule."""
    host = authorize_host(headers, allowed_authorities)
    origins = _headers_values(headers, "Origin")
    if len(origins) != 1:
        raise AuthorizationError("exactly one non-null Origin header is required")
    try:
        origin = normalize_origin(origins[0])
    except ValueError as exc:
        raise AuthorizationError("Origin is malformed or not lowercase http") from exc
    if origin != host:
        raise AuthorizationError("Origin authority does not match Host")
    return host


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class WorkstationApplication:
    """Immutable study/cache configuration used by each request handler."""

    def __init__(
        self,
        *,
        root: Path,
        cine_by_uid: Mapping[str, Any],
        series_by_uid: Mapping[str, Any],
        controller: WorkflowController,
        host: str,
        port: int,
        allowed_authorities: Iterable[str],
        user: str,
        clock: Callable[[], str] = _utc_now,
        jobs_enabled: bool = False,
        job_runner: JobRunner | None = None,
        x1_weights_root: Path | None = None,
        frontend_dist: Path | str | None = None,
    ) -> None:
        self.root = root
        self.db_path = root / "cmr.sqlite3"
        self.cine_by_uid = MappingProxyType(dict(cine_by_uid))
        self.series_by_uid = MappingProxyType(dict(series_by_uid))
        self.controller = controller
        self.host = host
        self.port = port
        self.allowed_authorities = tuple(allowed_authorities)
        self.user = user
        self.clock = clock
        self.renderer = ViewerRenderer()
        self._series_table_lut_cache: dict[str, Any] = {}
        self.jobs_enabled = bool(jobs_enabled)
        self.x1_weights_root = None if x1_weights_root is None else Path(x1_weights_root)
        if frontend_dist is None:
            self.frontend_dist: Path | None = None
        else:
            candidate = Path(frontend_dist)
            if not candidate.is_dir() or candidate.is_symlink():
                raise WorkstationConfigurationError(
                    "frontend dist must be a real directory"
                )
            self.frontend_dist = candidate
        self.job_runner = job_runner if self.jobs_enabled else None
        if self.jobs_enabled and self.job_runner is None:
            self.job_runner = JobRunner(
                self.db_path,
                controller=controller,
                cine_by_uid=self.cine_by_uid,
                clock=clock,
                x1_weights_root=self.x1_weights_root,
            )
            self.job_runner.start()

    @classmethod
    def bootstrap(
        cls,
        study_root: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        allow_authorities: Sequence[str] | None = None,
        user: str = "workstation",
        controller: WorkflowController | None = None,
        clock: Callable[[], str] = _utc_now,
        jobs_enabled: bool = True,
        job_runner: JobRunner | None = None,
        x1_weights_root: Path | None = None,
        frontend_dist: Path | str | None = None,
    ) -> WorkstationApplication:
        root = Path(study_root)
        listen = _listen_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise WorkstationConfigurationError("port must be an integer in 0..65535")
        if not isinstance(user, str) or not user.strip():
            raise WorkstationConfigurationError("user must be a non-empty string")
        if _is_wildcard(listen) and not allow_authorities:
            raise WorkstationConfigurationError(
                "wildcard listen requires at least one explicit non-wildcard --allow-authority"
            )
        if allow_authorities is None:
            if port == 0:
                authorities: tuple[str, ...] = ()
            else:
                authorities = (normalize_authority(_host_port_authority(listen, port)),)
        else:
            normalized: list[str] = []
            for authority in allow_authorities:
                try:
                    # Parse wildcard authorities here so the configuration
                    # layer can reject them with its explicit policy error;
                    # incoming browser authorities still use the default
                    # non-wildcard parser.
                    canonical = normalize_authority(
                        authority, require_port=True, allow_wildcard=True
                    )
                except ValueError as exc:
                    raise WorkstationConfigurationError(
                        f"--allow-authority is malformed: {authority!r}"
                    ) from exc
                authority_port = int(canonical.rsplit(":", 1)[1])
                if port == 0 or authority_port != port:
                    raise WorkstationConfigurationError(
                        "each --allow-authority port must equal the configured listen port"
                    )
                host_part = canonical.rsplit(":", 1)[0]
                if host_part.startswith("["):
                    wildcard = host_part == "[::]"
                else:
                    try:
                        wildcard = ipaddress.ip_address(host_part).is_unspecified
                    except ValueError:
                        wildcard = False
                if wildcard:
                    raise WorkstationConfigurationError(
                        "wildcard listen addresses cannot be browser authorities"
                    )
                if canonical not in normalized:
                    normalized.append(canonical)
            if not normalized:
                raise WorkstationConfigurationError(
                    "--allow-authority must contain at least one authority"
                )
            authorities = tuple(normalized)
        if not root.is_dir() or root.is_symlink():
            raise WorkstationConfigurationError("study root must be a real directory")
        cls._validate_root_layout(root)
        dicom_root = root / "dicom"
        series_dirs = sorted(dicom_root.iterdir(), key=lambda item: item.name)
        if not series_dirs:
            raise WorkstationConfigurationError("study root contains no DICOM series directories")
        for entry in series_dirs:
            cls._validate_series_layout(entry)

        # Full source validation happens before SqliteCmrStore is opened and
        # therefore before create_http_server can bind a socket.
        validated: list[tuple[str, Any, Any, Path]] = []
        seen_uids: set[str] = set()
        for directory in series_dirs:
            explicit_uid = directory.name
            try:
                cine, series = load_cine_volume(directory, series_uid=explicit_uid)
            except Exception as exc:
                raise WorkstationConfigurationError(
                    f"DICOM validation failed for {explicit_uid!r}: {exc}"
                ) from exc
            if cine is None or series is None:
                raise WorkstationConfigurationError(
                    f"DICOM series {explicit_uid!r} is unusable"
                )
            if (
                cine.series_uid != explicit_uid
                or series.series_uid != explicit_uid
                or cine.source_manifest is None
                or series.workflow_id != cine.workflow_id
            ):
                raise WorkstationConfigurationError(
                    f"DICOM series {explicit_uid!r} failed explicit identity/manifest validation"
                )
            if explicit_uid in seen_uids:
                raise WorkstationConfigurationError(f"duplicate series UID {explicit_uid!r}")
            seen_uids.add(explicit_uid)
            validated.append((explicit_uid, cine, series, directory))

        selected: dict[str, tuple[Any, Any]] = {}
        store = SqliteCmrStore(root / "cmr.sqlite3")
        try:
            active_controller = controller if controller is not None else WorkflowController()
            for uid, _cine, _series, directory in validated:
                cine, series = active_controller.select_series(store, directory, uid)
                if cine is None or series is None:
                    raise WorkstationConfigurationError(
                        f"controller could not select validated series {uid!r}"
                    )
                selected[uid] = (cine, series)
        except Exception:
            store.close()
            raise
        store.close()
        return cls(
            root=root,
            cine_by_uid={uid: pair[0] for uid, pair in selected.items()},
            series_by_uid={uid: pair[1] for uid, pair in selected.items()},
            controller=active_controller,
            host=listen,
            port=port,
            allowed_authorities=authorities,
            user=user,
            clock=clock,
            jobs_enabled=jobs_enabled,
            job_runner=job_runner,
            x1_weights_root=x1_weights_root,
            frontend_dist=frontend_dist,
        )

    @staticmethod
    def _validate_root_layout(root: Path) -> None:
        entries = {entry.name: entry for entry in root.iterdir()}
        unexpected = sorted(set(entries) - {"dicom", "cmr.sqlite3"})
        if unexpected:
            raise WorkstationConfigurationError(
                f"study root contains unexpected entries: {unexpected}"
            )
        dicom = entries.get("dicom")
        if dicom is None or not dicom.is_dir() or dicom.is_symlink():
            raise WorkstationConfigurationError("study root requires a real dicom directory")
        database = entries.get("cmr.sqlite3")
        if database is not None and (not database.is_file() or database.is_symlink()):
            raise WorkstationConfigurationError("cmr.sqlite3 must be a regular file")

    @staticmethod
    def _validate_series_layout(directory: Path) -> None:
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or not directory.name.strip()
            or directory.name in {".", ".."}
        ):
            raise WorkstationConfigurationError(
                f"DICOM child {directory.name!r} must be a non-empty real directory"
            )
        for entry in directory.iterdir():
            if entry.is_dir() or entry.is_symlink() or not entry.is_file():
                raise WorkstationConfigurationError(
                    f"DICOM series {directory.name!r} must contain files only, without nesting"
                )

    def set_bound_port(self, port: int) -> None:
        """Complete an ephemeral listener's default authority after binding."""
        if self.port == 0 and not self.allowed_authorities:
            self.port = port
            self.allowed_authorities = (normalize_authority(_host_port_authority(self.host, port)),)

    def cine_for(self, series_uid: str) -> Any:
        try:
            return self.cine_by_uid[series_uid]
        except KeyError as exc:
            raise KeyError(series_uid) from exc

    def open_store(self) -> SqliteCmrStore:
        """Open the per-request connection; callers must close it in finally."""
        return SqliteCmrStore(self.db_path)

    def open_evidence_store(self) -> SqliteCmrStore:
        """Open a NON-CREATING, NON-MIGRATING read connection for evidence GETs.

        R2 P2: the evidence endpoint must not create a missing store file or
        migrate an old one — reads refuse safely instead:
        - missing file → ``FileNotFoundError`` (mapped to the typed
          ``evidence_unavailable`` refusal by the handler);
        - stamped older than ``SCHEMA_USER_VERSION`` → ``RuntimeError``
          (same mapping; the endpoint never migrates — migration stays the
          exclusive job of the read-write open path);
        - newer-than-supported → the store's own fail-closed ``RuntimeError``
          (same mapping).
        The connection itself is a plain ``SqliteCmrStore`` on an existing
        current-schema file (no write transaction is opened by the endpoint;
        the projection performs reads only). Callers must close in finally.
        """
        if not self.db_path.is_file():
            raise FileNotFoundError(f"no store file at {self.db_path.name}")
        probe = sqlite3.connect(str(self.db_path))
        try:
            row = probe.execute("PRAGMA user_version").fetchone()
            stamped = int(row[0]) if row is not None else 0
        finally:
            probe.close()
        if stamped != SCHEMA_USER_VERSION:
            raise RuntimeError(
                f"evidence reads require store schema {SCHEMA_USER_VERSION}, "
                f"found {stamped}; refusing without migration"
            )
        return SqliteCmrStore(self.db_path)

    def parse_query_view(
        self, series_uid: str, query: str, *, cine_context: bool = False
    ) -> ViewIntent:
        from json import loads
        from urllib.parse import parse_qs

        cine = self.cine_for(series_uid)
        # S3-008B §3.2 / Terra R1 F3: validate the RAW percent-encoding
        # BEFORE parse_qs is allowed to decode. parse_qs accepts an unencoded
        # or partially-encoded value and decodes it to the SAME Python string
        # as the canonical spelling, so the encoding is unobservable after
        # decoding — exactly like the number-lex and duplicate-key spellings.
        reject_non_canonical_encoding(query, "query string")
        values = parse_qs(query, keep_blank_values=True, strict_parsing=False)
        if cine_context and "frame" in values:
            raise ValueError("cine context must not contain a frame query parameter")

        def one(name: str, default: str | None = None) -> str | None:
            selected = values.get(name)
            if selected is None:
                return default
            if len(selected) != 1:
                raise ValueError(f"query parameter {name!r} must occur exactly once")
            return selected[0]

        raw_slice = one("slice", "0")
        raw_frame = one("frame", "0")
        raw_fps = one("fps", str(10))
        raw_playing = one("playing", "false")
        raw_reference = one("reference_sex", None)
        raw_draft = one("inclusion_draft", None)
        # --- S3-008B §3.1: the two display-only view params ---------------
        # Carried exactly the way inclusion_draft already is: one URL-encoded
        # canonical compact JSON value per non-scalar param; OMISSION is the
        # canonical default (literal JSON null is accepted and normalized to
        # the omission default). They populate the two ViewIntent display
        # fields and are never persisted.
        raw_window = one("window_level", None)
        raw_visible = one("visible_structures", None)
        assert raw_slice is not None
        assert raw_frame is not None
        assert raw_fps is not None
        assert raw_playing is not None

        def parse_number_lex(raw: str, name: str) -> Any:
            """Load one display param, rejecting malformed percent/UTF-8 (§3.2).

            The RAW text is grammar-checked BEFORE ``json.loads``: a number's
            value cannot reveal its spelling (``1e2`` and ``100`` parse
            identically), so exponent/sign spellings are rejectable only here.
            """
            try:
                reject_non_grammar_number_text(raw, f"query parameter {name!r}")
                # Duplicate keys are erased by json.loads, so they must be
                # rejected on the RAW text (§3.2 negative).
                reject_duplicate_json_keys(raw, f"query parameter {name!r}")
                return loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"query parameter {name!r} is not valid JSON") from exc

        try:
            view_value: dict[str, Any] = {
                "slice": int(raw_slice),
                "frame": int(raw_frame),
                "fps": int(raw_fps),
                "playing": (
                    raw_playing == "true"
                    if raw_playing in ("true", "false")
                    else raw_playing
                ),
                "reference_sex": raw_reference,
                "inclusion_draft": (
                    None if raw_draft is None else loads(raw_draft)
                ),
            }
            if raw_window is not None:
                view_value["window_level"] = parse_number_lex(
                    raw_window, "window_level"
                )
            if raw_visible is not None:
                view_value["visible_structures"] = parse_number_lex(
                    raw_visible, "visible_structures"
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("view query contains malformed values") from exc
        return parse_view_intent(view_value, cine, require_all=True)

    def render_state(
        self,
        series_uid: str,
        view: ViewIntent,
        *,
        include_manifest: bool = False,
    ) -> tuple[Any, Any, dict[str, Any]]:
        store = self.open_store()
        try:
            state, reopened = build_viewer_state(
                self.controller,
                store,
                series_uid,
                self.cine_for(series_uid),
                view,
            )
            analysis_jobs = None
            if self.jobs_enabled:
                try:
                    # A6: merged feed over BOTH tables; counts/state are
                    # server-owned over both kinds (webapp arithmetic-free).
                    analysis_jobs = tuple(store.load_analysis_jobs(series_uid)) + tuple(
                        store.load_inference_jobs(series_uid)
                    )
                    analysis_jobs = tuple(
                        sorted(analysis_jobs, key=lambda job: (job.created_at, job.job_id))
                    )
                except Exception as exc:
                    # Job metadata is optional; never turn a clinical state read
                    # into an error after the authoritative state is available.
                    # Path-safe (F5): type + stable code, never raw text.
                    LOGGER.warning(
                        "analysis-job projection read failed: %s",
                        _typed_diagnostic(exc, "analysis_job_projection_failed"),
                    )
                    analysis_jobs = ()
            return state, reopened, state_projection(
                state,
                include_manifest=include_manifest,
                analysis_jobs=analysis_jobs,
                analysis=reopened.get("analysis"),
                analysis_stale_reason=reopened.get("analysis_stale_reason"),
            )
        finally:
            store.close()

    def close(self) -> None:
        if self.job_runner is not None:
            self.job_runner.stop()

    def report_bytes(self, series_uid: str, reference_sex: Any, fmt: str) -> bytes:
        store = self.open_store()
        try:
            data = self.controller.reopen(store, series_uid, cine=self.cine_for(series_uid))
        finally:
            store.close()
        chain = tuple(data["chain"])
        measurement = data.get("latest_measurement")
        workflow_id = data.get("workflow_id")
        manifest = data.get("source_manifest")
        if (
            not chain
            or measurement is None
            or data.get("has_stale_analysis")
            or workflow_id is None
            or manifest is None
            or chain[-1].qc_report is None
            or manifest.series_uid != series_uid
        ):
            raise CMRError("report.unavailable", "report_unavailable")
        report_input = ReportInput(
            measurement=measurement,
            qc_summary=chain[-1].qc_report,
            series_uid=series_uid,
            study_uid=manifest.study_uid,
            workflow_id=workflow_id,
            patient_sex=reference_sex,
            report_created_at=self.clock(),
        )
        return reproduce_report(report_input, fmt)


def bootstrap_application(
    study_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    allow_authorities: Sequence[str] | None = None,
    user: str = "workstation",
    controller: WorkflowController | None = None,
    clock: Callable[[], str] = _utc_now,
    jobs_enabled: bool = True,
    job_runner: JobRunner | None = None,
    x1_weights_root: Path | None = None,
    frontend_dist: Path | str | None = None,
) -> WorkstationApplication:
    return WorkstationApplication.bootstrap(
        study_root,
        host=host,
        port=port,
        allow_authorities=allow_authorities,
        user=user,
        controller=controller,
        clock=clock,
        jobs_enabled=jobs_enabled,
        job_runner=job_runner,
        x1_weights_root=x1_weights_root,
        frontend_dist=frontend_dist,
    )


def create_http_server(app: WorkstationApplication) -> Any:
    """Construct (and bind) the threaded stdlib HTTP server."""
    from http.server import ThreadingHTTPServer

    from .api import make_handler

    address: Any
    if ":" in app.host:
        address = (app.host, app.port, 0, 0)
    else:
        address = (app.host, app.port)
    server = ThreadingHTTPServer(address, make_handler(app))
    app.set_bound_port(int(server.server_address[1]))
    if not app.allowed_authorities:
        server.server_close()
        raise WorkstationConfigurationError(
            "wildcard listener cannot bind without an explicit browser authority"
        )
    setattr(server, "app", app)
    return server


def bound_handshake(app: WorkstationApplication) -> str:
    """Return the single machine-readable READY line for a bound application."""
    authority = app.allowed_authorities[0]
    return f"CMR_WORKSTATION_READY authority={authority} port={app.port} pid={os.getpid()}"


def run(
    study_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    allow_authorities: Sequence[str] | None = None,
    user: str = "workstation",
    x1_weights_root: Path | None = None,
    frontend_dist: Path | str | None = None,
) -> None:
    app = bootstrap_application(
        study_root,
        host=host,
        port=port,
        allow_authorities=allow_authorities,
        user=user,
        x1_weights_root=x1_weights_root,
        frontend_dist=frontend_dist,
    )
    httpd = create_http_server(app)
    authority = app.allowed_authorities[0]
    print(f"CMR Studio workstation listening at http://{authority}", flush=True)
    print(
        "LAN binds are UNAUTHENTICATED and UNENCRYPTED — anyone on the network "
        "can view/control the session; use only synthetic fixtures on a trusted "
        "network; PHI is PROHIBITED",
        flush=True,
    )
    print(bound_handshake(app), flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        app.close()
