"""Backend-owned recent-cases persistence and availability probe (CMR_CASE_WORKSPACE_RUNTIME_006).

Implements bounded, tamper-evident recent study persistence at a server-owned
stable private path, independent of client-supplied stage paths and changing
study roots.

Contract seams:
- Versioned allow-listed document schema (version 1, max 10 entries).
- Exact keys per entry: studyUid, seriesUid|null, scanSourcePath, stageBaseDir,
  destRoot (display-only), openedSeq (server-assigned), openCount (>= 1).
- Server-assigned openedSeq monotonic from durable high-water mark; client
  openedSeq is IGNORED and never trusted as authority.
- Serialized read-validate-merge-assign-replace under an in-process lock.
- Stale-write rejection via client base-digest (409 recent.stale_snapshot + current
  document for retry-with-merge).
- OWNER DECISION A: single tampered row degrades the whole document to an empty
  list with reason 'corrupt-rows'.
- Explicit physical/codec distinction: 0-byte file reads as absent, not corrupt.
- Last-good preservation: atomic same-dir temp file + os.replace + fsync; failed
  replace keeps prior bytes and returns actionable recent.store_unwritable.
- Symlink/reparse refusal on every path component; the sidecar high-water read
  additionally binds the open HANDLE to the validated location (handle-derived
  proof, never a path re-walk) so an ancestor redirected inside the
  validation-to-observation window is refused.
- Genuine-live denial on raw text and resolved physical paths for scanSourcePath,
  stageBaseDir, and derived dest (stageBaseDir / studyUid).
- Bounded reads (64 KiB file cap + JSON depth cap).
- Read-only availability / staged probe route helper.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import posixpath
import stat
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from .target_authority import (
    Refusal,
    census_staged_root,
    check_raw_path_text,
    is_genuine_denied,
    reject_links,
    resolve_path,
    validate_uid,
)

RECENT_CASES_SCHEMA_VERSION = 1
RECENT_CASES_LIMIT = 10
ENTRY_KEYS = (
    "studyUid",
    "seriesUid",
    "scanSourcePath",
    "stageBaseDir",
    "destRoot",
    "openedSeq",
    "openCount",
)

# Replay-idempotence horizon: how many consumed open ids the document
# remembers. A duplicate delivery carrying a remembered id returns the
# current document unchanged (no counter movement); ids evicted from the
# bounded tail may double-count on a very late replay (documented residual;
# the window covers every realistic transport-retry horizon).
MAX_SEEN_OPEN_IDS = 100
OPEN_ID_PATTERN = (
    "open ids are client-generated opaque tokens (1..128 chars of "
    "A-Za-z0-9._-); malformed ids are refused, never trusted"
)

MAX_RECENT_FILE_BYTES = 64 * 1024
MAX_JSON_DEPTH = 5
KEY_SEP = "\x1f"
DEFAULT_RECENT_DIR_NAME = ".cmr-recent"
DEFAULT_RECENT_FILE_NAME = "recent_cases.json"

_LOCK = threading.Lock()
_HIGH_WATER_BY_PATH: dict[str, int] = {}


def _write_all(fd: int, data: bytes) -> None:
    """Write the complete buffer or raise; a short write is never silent.

    ``os.write`` may write fewer bytes than requested (finding: the old
    code ignored the count, then fsynced and replaced, committing
    truncated JSON while reporting success). Loop until every byte lands;
    a writer that reports zero progress raises ``OSError`` (EIO) instead
    of spinning forever. Callers translate that into a stable
    ``recent.store_unwritable`` refusal with last-good bytes preserved.
    """
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except OSError:
            raise
        if written <= 0:
            raise OSError(errno.EIO, "short write with no progress")
        view = view[written:]


def _dir_fd_replace_supported() -> bool:
    """True only when the platform offers directory-handle writes.

    Windows has neither O_DIRECTORY directory opens nor dir_fd replace,
    so the handle path is never taken there; POSIX hosts without
    O_DIRECTORY/dir_fd support keep the re-verified pathname discipline.
    """
    return (
        os.name != "nt"
        and hasattr(os, "O_DIRECTORY")
        and "dir_fd" in getattr(os.replace, "__text_signature__", "dir_fd")
    )


def _leaf_link_at(path: Path) -> bool:
    """True only when the path's FINAL component is itself a symlink.

    Observed with ``lstat`` (no follow), so the classification below can
    never be redirected through the link's target; any observation
    failure (component gone, intermediate non-directory) answers False
    and leaves the caller with the original error.
    """
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except OSError:
        return False


def _parent_open_refusal(exc: OSError, parent_dir: Path) -> RecentError | None:
    """Classify a failed parent-directory open as a link refusal, or None.

    The no-follow directory open reports the leaf link differently by
    kernel: ELOOP (macOS/BSD, and the textbook O_NOFOLLOW result) versus
    ENOTDIR on Linux, where ``O_DIRECTORY`` is applied to a final
    component that is a symlink rather than a directory. Only a link can
    produce the ENOTDIR case here, but ENOTDIR is also the errno for an
    ORDINARY non-directory component, which must keep its original
    classification (``recent.store_unwritable``/``store_unreadable``
    downstream, never ``target_authority.symlink``). The distinction is
    made by observing the final component itself without following it.
    """
    err = getattr(exc, "errno", None)
    if err == errno.ELOOP:
        # O_NOFOLLOW on the final component: definitively a leaf link.
        return RecentError(
            "target_authority.symlink",
            "recent store parent is a link; refusing fail-closed",
        )
    if err == errno.ENOTDIR and _leaf_link_at(parent_dir):
        # Linux native errno for O_DIRECTORY|O_NOFOLLOW over a leaf link.
        return RecentError(
            "target_authority.symlink",
            "recent store parent is a link; refusing fail-closed",
        )
    return None


def _open_parent_dir(
    parent_dir: Path, *, expect_identity: tuple[int, int] | None = None
) -> int:
    """Open a validated parent directory and return its fd.

    One-line seam so write-side redirection witnesses can interpose a
    real parent swap between validation and use without touching
    production code.

    Acquisition-time binding (R7 review finding F1): the previous shape
    opened the parent and pinned the identity of whatever it had ALREADY
    opened, so a redirect that landed between ``check_store_path_safety``
    and this open produced a handle to the redirected directory and the
    complete payload was written through it. The handle is now verified
    BEFORE it is trusted:
    * ``O_NOFOLLOW`` refuses a parent that became a link; the native
      link-failure errno (ELOOP on macOS/BSD, ENOTDIR on Linux for
      ``O_DIRECTORY | O_NOFOLLOW``) is translated to the stable
      ``target_authority.symlink`` refusal — see ``_parent_open_refusal``,
      which classifies ENOTDIR by observing the final component without
      following it, so an ordinary non-directory is never reported as a
      link refusal;
    * ``expect_identity`` — sampled with ``lstat`` at validation time —
      refuses a parent whose opened directory is a different object
      (ancestor redirection resolving elsewhere).
    The verified handle is then carried through temp creation and
    replacement, so no later pathname re-walk can escape the approved
    directory.

    POSIX-only helper: Windows cannot os.open a directory at all
    (ERROR_ACCESS_DENIED), so all callers gate handle use on
    ``os.name != "nt"`` and keep the re-verified pathname discipline on
    Windows (WIN-WRITE-012 boundary unchanged).
    """
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(parent_dir, flags)
    except OSError as exc:
        refusal = _parent_open_refusal(exc, parent_dir)
        if refusal is not None:
            raise refusal from exc
        raise
    if expect_identity is not None:
        try:
            actual = _storage_parent_identity_fd(fd)
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        if actual != expect_identity:
            try:
                os.close(fd)
            except OSError:
                pass
            raise RecentError(
                "target_authority.symlink",
                "recent store parent changed between validation and open; "
                "refusing fail-closed",
            )
    return fd


def _acquire_verified_parent(
    parent_dir: Path, store_path: Path
) -> tuple[int, tuple[int, int] | None]:
    """Acquire the write-side parent handle with acquisition-time binding.

    POSIX-only (callers gate on ``_dir_fd_replace_supported()``). Samples
    the validated parent identity BEFORE opening (``lstat``, no-follow on
    the final component), opens through ``_open_parent_dir`` with that
    anchor, and re-runs the full path-safety check once the handle is
    bound: a link that landed before the sample would make the sample
    agree with a redirected ancestry, but it is still visible to the
    path check, while the bound handle keeps every write on the sampled
    inode. EVERY unsuccessful validation closes the handle before
    propagating, and a ``target_authority.Refusal`` from the post-bind
    link check (``reject_links``, which does not raise ``RecentError``)
    is translated WITHOUT losing its stable code/message, so the frozen
    error envelope stays RecentError-shaped and no descriptor leaks.

    Returns ``(parent_fd, bound_pathname_identity)``: the ACQUISITION-bound
    PATHNAME parent identity travels with the handle so the pre-replace
    gate can prove the pathname still names the bound directory — a
    ``fstat(parent_fd)`` observation alone can never see a rename/replace
    of the pathname that leaves the descriptor untouched (review R13 P2).
    ``None`` means the parent did not exist when sampled (originally
    absent parent): writers refuse fail-closed at replace time.

    Residual boundary (same class as WIN-WRITE-012): a rename-style swap
    landing before the identity sample AND leaving no link in the path is
    indistinguishable from the approved directory by pathname — this
    requires local same-user code execution in the same instant, outside
    the loopback threat model. Every wider window is bound or refused.
    """
    try:
        parent_identity: tuple[int, int] | None = _storage_parent_identity(parent_dir)
    except FileNotFoundError:
        parent_identity = None
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", "directory observation", exc
        ) from exc
    try:
        parent_fd = _open_parent_dir(parent_dir, expect_identity=parent_identity)
    except RecentError:
        raise
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", "directory observation", exc
        ) from exc
    try:
        check_store_path_safety(store_path)
    except Refusal as exc:
        # reject_links raises target_authority.Refusal, NOT RecentError:
        # without this arm the acquired descriptor leaks and the refusal
        # escapes the frozen RecentError envelope. Close first, translate
        # second, keeping the stable code/message intact.
        try:
            os.close(parent_fd)
        except OSError:
            pass
        raise RecentError(exc.code, exc.message) from exc
    except Exception:
        # Close on EVERY unsuccessful validation (RecentError from the
        # genuine-live gate here, or anything else the check raises).
        try:
            os.close(parent_fd)
        except OSError:
            pass
        raise
    return parent_fd, parent_identity


def _storage_parent_identity_fd(parent_fd: int) -> tuple[int, int]:
    """Identity (st_dev, st_ino) of an already-open parent directory."""
    st = os.fstat(parent_fd)
    return (st.st_dev, st.st_ino)


def _storage_parent_identity(dir_path: Path) -> tuple[int, int]:
    """Identity (st_dev, st_ino) of a directory, observed WITHOUT following
    a leaf link (lstat on the parent itself)."""
    st = dir_path.lstat()
    return (st.st_dev, st.st_ino)


def _create_temp_in_parent(
    parent_fd: int, temp_name: str, flags: int, mode: int
) -> int:
    """Create the temp file inside an already-open parent directory.

    One-line seam (dir_fd interposition point for write-side witnesses).
    """
    return os.open(temp_name, flags, mode, dir_fd=parent_fd)


def _replace_path(src: Path, dst: Path) -> None:
    """One-line replacement seam (test interposition point)."""
    os.replace(src, dst)


def _replace_in_parent(
    parent_fd: int, temp_name: str, store_path: Path
) -> None:
    """Replace the store with the temp through the pinned parent handle.

    One-line seam: POSIX binds both sides to the open directory fd so no
    post-validation pathname re-traversal can escape the approved
    directory; Windows lacks dir_fd replace support and falls back to the
    pathname replace only after the parent re-verification below.
    """
    os.replace(
        temp_name,
        os.fspath(store_path.name),
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
    )


def _atomic_replace(
    temp_path: Path,
    store_path: Path,
    *,
    action: str,
    parent_fd: int | None = None,
    parent_bound_identity: tuple[int, int] | None = None,
) -> None:
    """Replace the store with a fully-written temp through the PINNED parent.

    Required fix shape (write-side ancestor redirection): the validated
    parent directory is opened ONCE (directory handle, verified against
    the validation-time identity by ``_acquire_verified_parent`` when
    callers supply it), and the final replace is issued through that
    handle — no post-validation pathname re-traversal can escape the
    approved directory. ``parent_fd`` carries the already-verified handle
    from the write leg so creation and replacement share ONE verified
    binding (R7 review finding F1); it is never closed here because the
    caller owns it. The parent identity observed through the handle is
    re-verified immediately before the replace; a mismatch aborts with a
    stable refusal, the temp is unlinked, and last-good bytes are
    untouched.

    Review R13 P2 (pathname re-check): the fd observation alone CANNOT
    change while the descriptor is open, so a rename/replace of the parent
    PATHNAME after acquisition is invisible to it — the fd-relative replace
    would then commit into the detached original while the active
    ``store_path`` stayed unchanged and the request reported success. The
    ACQUISITION-bound pathname identity (``parent_bound_identity``, carried
    from ``_acquire_verified_parent``) is therefore re-checked against the
    CURRENT pathname parent immediately before the fd-relative replace; a
    mismatch — or a bound identity of ``None`` (originally absent parent) —
    refuses fail-closed with ``target_authority.symlink`` BEFORE any write.
    On the own-fd path (no caller-supplied handle) the bound identity is
    sampled from the pathname immediately before opening it.

    Platform note: POSIX binds both replace sides to the open directory
    fd (``src_dir_fd``/``dst_dir_fd``). Windows has no dir_fd replace, so
    the fallback re-validates the parent identity by pathname, re-runs the
    full path-safety check, and only then replaces by pathname — the
    residual window is one syscall, fail-closed on any observed change.

    Windows residual-window threat boundary (WIN-WRITE-012): a parent swap
    landing strictly INSIDE that final one-syscall window (after the last
    identity observation, before the replace completes) is not refused on
    Windows — no pathname API can close it, and a deterministic witness
    would be flaky by construction (it must win a syscall race), so none
    is attempted. The boundary is: exploiting the window requires local
    code execution as the same user in the same instant (rename the store
    parent between two adjacent syscalls) — outside the loopback network
    threat model, which treats same-user local writers as trusted for
    availability. Every wider window IS closed: pre-check swaps are refused
    (``test_parent_swap_between_check_and_replace_is_refused``), pre-replace
    swaps never escape (``test_write_parent_redirection_never_escapes_...``),
    and any hostile bytes that ever become readable fail closed at JSON /
    codec validation on the next load. The POSIX handle path provably
    cannot escape by construction (fd-relative replace — witnessed by
    ``test_posix_handle_path_binds_replace_to_pinned_parent``).
    """
    try:
        check_store_path_safety(store_path)
    except Refusal as exc:
        raise RecentError(exc.code, exc.message) from exc
    parent_dir = store_path.parent
    use_dir_fd = _dir_fd_replace_supported()
    own_fd = False
    if use_dir_fd and parent_fd is None:
        # Own-fd path: bind the ACQUISITION identity from the pathname
        # immediately before opening it, so the pre-replace pathname gate
        # below has a bound identity to compare against (R13 P2).
        try:
            parent_bound_identity = _storage_parent_identity(parent_dir)
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unwritable", "directory observation", exc
            ) from exc
        try:
            parent_fd = _open_parent_dir(parent_dir)
        except RecentError:
            raise
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unwritable", "directory observation", exc
            ) from exc
        own_fd = True
    try:
        if use_dir_fd:
            assert parent_fd is not None
            # R13 P2: prove the PATHNAME still names the acquisition-bound
            # directory before committing through the handle. The fd
            # observation below cannot change while the descriptor is open,
            # so it can never see a rename/replace of the pathname; this
            # pathname check can. An originally absent parent (bound
            # identity None) is refused fail-closed as well.
            if parent_bound_identity is None:
                raise RecentError(
                    "target_authority.symlink",
                    "recent store parent has no acquisition-bound identity; "
                    "refusing fail-closed",
                )
            try:
                current_parent = _storage_parent_identity(parent_dir)
            except OSError as exc:
                raise RecentError.storage_failure(
                    "recent.store_unwritable", "directory observation", exc
                ) from exc
            if current_parent != parent_bound_identity:
                raise RecentError(
                    "target_authority.symlink",
                    "recent store parent changed between acquisition and "
                    "replace; refusing fail-closed",
                )
            try:
                pinned = _storage_parent_identity_fd(parent_fd)
            except OSError as exc:
                raise RecentError.storage_failure(
                    "recent.store_unwritable", "directory observation", exc
                ) from exc
            _replace_in_parent(parent_fd, temp_path.name, store_path)
            try:
                if _storage_parent_identity_fd(parent_fd) != pinned:
                    raise RecentError(
                        "target_authority.symlink",
                        "recent store parent changed during validation; "
                        "refusing fail-closed",
                    )
            except OSError as exc:
                raise RecentError.storage_failure(
                    "recent.store_unwritable", "directory observation", exc
                ) from exc
        else:
            # Windows / no-dir_fd fallback: re-verify identity + full path
            # safety by pathname immediately before the replace, then
            # replace. The residual window is one syscall, fail-closed on
            # any observed change.
            try:
                check_store_path_safety(store_path)
            except Refusal as exc:
                raise RecentError(exc.code, exc.message) from exc
            try:
                before = _storage_parent_identity(parent_dir)
                try:
                    check_store_path_safety(store_path)
                except Refusal as exc:
                    raise RecentError(exc.code, exc.message) from exc
                if _storage_parent_identity(parent_dir) != before:
                    raise RecentError(
                        "target_authority.symlink",
                        "recent store parent changed during validation; "
                        "refusing fail-closed",
                    )
            except RecentError:
                raise
            except OSError as exc:
                raise RecentError.storage_failure(
                    "recent.store_unwritable", "directory observation", exc
                ) from exc
            _replace_path(temp_path, store_path)
    except (RecentError, Refusal) as exc:
        if isinstance(exc, Refusal):
            raise RecentError(exc.code, exc.message) from exc
        raise
    except (PermissionError, OSError) as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", action, exc
        ) from exc
    finally:
        if own_fd and parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


class RecentError(Exception):
    """Refusal or failure in the recent-cases subsystem.

    Contract data-minimization: ``message`` is the STABLE, actionable
    client text (typed code + fixed wording, never a raw host path).
    The originating exception's text lives ONLY in ``detail`` for
    internal diagnostics — the HTTP boundary must never forward it.
    """

    def __init__(
        self,
        code: str,
        message: str,
        current: dict[str, Any] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.current = current
        self.detail = detail

    @staticmethod
    def storage_failure(code: str, action: str, exc: BaseException) -> RecentError:
        """Stable client text + internal detail for a filesystem failure.

        ``action`` names the operation ("read", "stat", "write", "replace",
        "validate store path"); ``exc`` is kept out of the client message
        because ``str(OSError)`` normally embeds the full server path.
        """
        detail = f"{type(exc).__name__}: {getattr(exc, 'errno', '?')}"
        return RecentError(
            code,
            f"recent store {action} failed; the prior document is preserved",
            detail=detail,
        )


def check_json_depth(obj: Any, current_depth: int = 1, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Refuse JSON documents exceeding bounded depth."""
    if current_depth > max_depth:
        raise RecentError(
            "recent.malformed",
            f"JSON depth exceeds maximum permitted limit ({max_depth})",
        )
    if isinstance(obj, dict):
        for v in obj.values():
            check_json_depth(v, current_depth + 1, max_depth)
    elif isinstance(obj, list):
        for item in obj:
            check_json_depth(item, current_depth + 1, max_depth)


# Path identity is case-insensitive on Windows and case-sensitive on POSIX.
# Module-level so the identity policy is one decision and both semantics stay
# verifiable from either host (the POSIX leg can be exercised on Windows).
_CASE_FOLD_PATH_IDENTITY = os.name == "nt"


def safe_canonical_path_key(path_str: str) -> str:
    """Safe OS-aware key segment for identity mapping (never used for security decisions).

    PR3 review #4 F2: equivalent anchored spellings of the SAME source must
    merge into one recent-case identity. The canonical form collapses redundant
    separators and dot segments lexically and, on Windows (a case-insensitive
    filesystem), folds case; case-distinct paths stay distinct on POSIX. It is
    derived from the ALREADY-VALIDATED raw spelling: the safety validation
    (raw-text checks, genuine-live denial, link refusal) still runs on the
    original spelling before identity is ever computed, this helper performs no
    filesystem access, and the stored/displayed spelling is untouched.
    """
    if not isinstance(path_str, str) or not path_str:
        return "/"
    norm = path_str.replace("\\", "/")
    # Lexical canonicalization only (never a filesystem walk): collapse
    # redundant separators and dot segments exactly like the frozen frontend
    # identity policy, so `/copy/library` and `/copy/./library` are one key.
    collapsed = posixpath.normpath(norm)
    if _CASE_FOLD_PATH_IDENTITY:
        # Windows path identity is case-insensitive; POSIX keeps case-distinct
        # paths distinct.
        collapsed = collapsed.lower()
    return collapsed or "/"


def recent_case_key(entry: dict[str, Any]) -> str:
    """Stable identity for the same case:
    (studyUid, seriesUid, safeCanonicalForm(scanSourcePath)).
    """
    study = str(entry.get("studyUid", ""))
    series = str(entry.get("seriesUid", "") or "")
    source = safe_canonical_path_key(str(entry.get("scanSourcePath", "")))
    return KEY_SEP.join([study, series, source])


def get_recent_store_path(app: Any = None) -> Path:
    """Resolve the server-owned private stable recent-cases store path."""
    if app is not None:
        custom = getattr(app, "recent_store_path", None)
        if custom is not None:
            return Path(custom)
    env_path = os.environ.get("CMR_RECENT_STORE_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / DEFAULT_RECENT_DIR_NAME / DEFAULT_RECENT_FILE_NAME


def get_sidecar_seq_path(store_path: Path) -> Path:
    """Sidecar high-water mark path beside the store file."""
    return store_path.parent / f"{store_path.name}.seq"


def check_store_path_safety(path: Path) -> None:
    """Symlink and genuine-live refusal on every component of the store path."""
    path_str = str(path)
    reject_links(path_str, "recent_store_path")
    if is_genuine_denied(path_str):
        raise RecentError(
            "target_authority.genuine_live",
            "recent store path matches the genuine-live marker",
        )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def compute_document_digest(entries: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 digest over the versioned document."""
    payload = {
        "entries": entries,
        "version": RECENT_CASES_SCHEMA_VERSION,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def empty_document_digest() -> str:
    return compute_document_digest([])


def validate_open_id(value: Any) -> str:
    """Validate a client-supplied replay-idempotence token.

    Opaque, bounded, charset-restricted: it is NEVER authority for content
    (only for duplicate-delivery suppression) and never persisted outside
    the bounded ``seen_open_ids`` tail.
    """
    if not isinstance(value, str) or not value:
        raise RecentError("recent.malformed", "open_id must be a non-empty string")
    if len(value) > 128:
        raise RecentError("recent.malformed", "open_id exceeds the bounded length")
    if any(not (ch.isalnum() or ch in "._-") for ch in value):
        raise RecentError("recent.malformed", "open_id uses unexpected characters")
    return value


def validate_entry_row(row: Any, *, for_persistence: bool = False) -> dict[str, Any]:
    """Validate one recent case entry against strict schema and physical boundaries.

    Fails closed on any invalid, missing, extra, or unsafe field.

    N1 producer contract (incoming/pending rows only): the first-open
    producer (app.js setPendingOpen) queues the opened identity WITHOUT
    server counters (no openedSeq / openCount keys), so a MISSING counter
    is defaulted (0 / 1) on that path; a PRESENT counter must be
    well-formed — a hostile slot must not smuggle counter authority
    through the pending path, and the record path clamps both to
    server-authoritative values regardless.

    Persisted rows (``for_persistence=True``, R7 review finding F3): the
    counters are MANDATORY. A persisted row without them is damaged, not
    repairable: defaulting silently would leave the document selectable
    with reason "none" and could reset counts, instead of the required
    whole-document ``corrupt-rows`` degradation. ``destRoot`` remains
    display-only on both paths.
    """
    if not isinstance(row, dict):
        raise Refusal("recent.malformed", "entry must be a JSON object")

    missing = [
        k
        for k in ENTRY_KEYS
        if k not in row
        and (k != "destRoot" or for_persistence)
        and (k not in ("openedSeq", "openCount") or for_persistence)
    ]
    if missing:
        raise Refusal("recent.malformed", f"missing required entry fields: {', '.join(missing)}")
    if "destRoot" not in row:
        row = dict(row)
        row["destRoot"] = ""
    if "openedSeq" not in row:
        row = dict(row)
        row["openedSeq"] = 0
    if "openCount" not in row:
        row = dict(row)
        row["openCount"] = 1

    # No unknown keys permitted
    unknown = set(row) - set(ENTRY_KEYS)
    if unknown:
        raise Refusal("recent.malformed", f"unexpected entry fields: {', '.join(sorted(unknown))}")

    study_uid = row["studyUid"]
    if not isinstance(study_uid, str) or not study_uid.strip():
        raise Refusal("target_authority.uid_invalid", "studyUid must be a non-blank string")
    validate_uid(study_uid, "studyUid")

    series_uid = row["seriesUid"]
    if series_uid is not None:
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise Refusal(
                "target_authority.uid_invalid",
                "seriesUid must be null or non-blank string",
            )
        validate_uid(series_uid, "seriesUid")

    scan_source = row["scanSourcePath"]
    if not isinstance(scan_source, str) or not scan_source.strip():
        raise Refusal("library.bad_path", "scanSourcePath must be a non-blank string")
    check_raw_path_text(scan_source, "scanSourcePath")

    stage_base = row["stageBaseDir"]
    if not isinstance(stage_base, str) or not stage_base.strip():
        raise Refusal("library.bad_path", "stageBaseDir must be a non-blank string")
    check_raw_path_text(stage_base, "stageBaseDir")

    dest_root = row["destRoot"]
    if not isinstance(dest_root, str):
        raise Refusal("recent.malformed", "destRoot must be a string")

    opened_seq = row["openedSeq"]
    # bool is an int subclass in Python
    if isinstance(opened_seq, bool) or not isinstance(opened_seq, int) or opened_seq < 0:
        raise Refusal("recent.malformed", "openedSeq must be a non-negative integer")

    open_count = row["openCount"]
    if isinstance(open_count, bool) or not isinstance(open_count, int) or open_count < 1:
        raise Refusal("recent.malformed", "openCount must be an integer >= 1")

    # Physical / security boundary checks (genuine-live, symlink, containment)
    if is_genuine_denied(scan_source):
        raise Refusal(
            "target_authority.genuine_live",
            "scanSourcePath matches genuine-live marker",
        )
    if is_genuine_denied(stage_base):
        raise Refusal(
            "target_authority.genuine_live",
            "stageBaseDir matches genuine-live marker",
        )

    resolved_source = resolve_path(scan_source, "scanSourcePath")
    resolved_stage = resolve_path(stage_base, "stageBaseDir")
    raw_dest = resolved_stage / study_uid

    reject_links(scan_source, "scanSourcePath")
    reject_links(stage_base, "stageBaseDir")
    reject_links(str(raw_dest), "dest")

    resolved_dest = raw_dest.resolve()
    if is_genuine_denied(str(resolved_dest)):
        raise Refusal(
            "target_authority.genuine_live",
            "derived destination resolves into genuine-live marker",
        )

    if resolved_dest == resolved_source:
        raise Refusal(
            "library.dest_equals_source",
            "dest must not equal the scan source root",
        )
    if resolved_source in resolved_dest.parents:
        raise Refusal(
            "library.dest_inside_source",
            "dest must not lie inside the scan source root",
        )
    if resolved_stage not in resolved_dest.parents:
        raise Refusal(
            "target_authority.dest_outside_stage_base",
            "dest must lie strictly inside stage base",
        )

    return {
        "studyUid": study_uid,
        "seriesUid": series_uid,
        "scanSourcePath": scan_source,
        "stageBaseDir": stage_base,
        "destRoot": dest_root or str(resolved_dest),
        "openedSeq": opened_seq,
        "openCount": open_count,
    }


def _hw_path_key(store_path: Path) -> str:
    try:
        return str(store_path.resolve())
    except OSError:
        return str(store_path)


# Dedicated lock serializing ALL durable high-water operations (dict + sidecar
# file): without it an interleaved path could rewrite a STALE sequence over
# the sidecar a concurrent POST just wrote (review P1-1). It is never held
# while acquiring _LOCK; the only nested order in production is
# _LOCK -> _HW_LOCK (both load and commit hold _LOCK first).
_HW_LOCK = threading.Lock()

# Dedicated lock serializing ALL pending-slot operations: queue replacement
# (save), correlated clear (comparison + unlink), and load-time expiry /
# invalid-content removal. Without it the clear's comparison and its unlink are
# separate unlocked steps — a replacement installed between them is deleted
# while the clear reports success for the older intent — and a loader's expiry
# cleanup can delete a freshly queued intent (PR3 review #2 F2). Lock order:
# this lock is never held while acquiring _LOCK or _HW_LOCK, and those are never
# held while acquiring it (the pending slot is an independent file namespace
# with its own lock).
_PENDING_LOCK = threading.Lock()

# Sidecar content bound: a decimal sequence only (bytes cap for the read).
_MAX_SIDECAR_BYTES = 32


def _read_bounded_bytes(fd: int, cap: int) -> bytes:
    """Read at most ``cap + 1`` bytes from an already open handle.

    The bound is applied to the READ itself, never to an earlier ``stat``:
    a size check followed by a separate read is TOCTOU-able by a local
    filesystem actor that grows or replaces the file in between. Reading
    one byte past the cap lets the caller reject excess content without
    ever ingesting it.
    """
    chunks: list[bytes] = []
    remaining = cap + 1
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_sidecar_bytes(fd: int) -> bytes:
    """Read at most ``_MAX_SIDECAR_BYTES + 1`` bytes from an open handle.

    The bound is applied to the READ itself, not to an earlier ``lstat``:
    a size check followed by a separate ``read_text`` is TOCTOU-able by a
    local filesystem actor that grows or replaces the file in between
    (review P1-2). Reading one byte past the cap lets the caller reject
    excess content without ever ingesting it.
    """
    return _read_bounded_bytes(fd, _MAX_SIDECAR_BYTES)


def _same_file(a: os.stat_result, b: os.stat_result) -> bool:
    """True when two observations are the same object at the same content version.

    ``(st_dev, st_ino)`` is the object identity; ``st_size`` is added on all
    platforms. On POSIX the observations must also agree on ``st_ctime_ns``:
    ctime is the kernel-maintained metadata-change witness and catches the
    observed same-inode unlink/recreate even when the replacement has the same
    byte length. On Windows ``st_ctime`` is not a stable lstat/fstat
    observation (it can drift between the two calls) and is therefore not
    compared; Windows keeps dev/inode/size plus its existing file-ID
    unlink/recreate witness.

    Exact guarantee, not a proof of unforgeable identity: POSIX both
    observations must agree on dev, inode, size and ctime. A same-inode
    replacement that reproduces all four, a privileged actor able to backdate
    ctime, and a mutation after the final observation remain outside this
    guarantee. This is deliberately not an mtime check: mtime is
    user-settable. The remaining TOCTOU is reported, never waived.
    """
    if (
        a.st_dev,
        a.st_ino,
        a.st_size,
    ) != (
        b.st_dev,
        b.st_ino,
        b.st_size,
    ):
        return False
    if os.name == "nt":
        return True
    return a.st_ctime_ns == b.st_ctime_ns


def _open_sidecar(seq_path: Path, flags: int) -> int:
    """Open the sidecar for reading.

    Kept as a one-line seam so the swap regression witnesses can interpose
    exactly the moment between link validation and open (round-3 R3-2),
    without any repository-only hook in the production path.
    """
    return os.open(seq_path, flags)


def _open_main_document(store_path: Path, flags: int) -> int:
    """Open the main recent document for reading.

    Same one-line interposition seam as the sidecar: witnesses swap or grow
    the store between validation and open without touching production code.
    """
    return os.open(store_path, flags)


def _handle_location(fd: int) -> str | None:
    """Location of an open handle, derived FROM the handle itself.

    This never re-walks the path: a path re-check re-opens the
    validation-to-observation window it is meant to close (round-4 R3-2).

    Windows: ``GetFinalPathNameByHandleW`` with ``FILE_NAME_NORMALIZED |
    VOLUME_NAME_DOS``. The reported path is the link-free location the
    kernel actually traversed to produce the handle, so a parent that was a
    junction at open time reports the TARGET's location — and it keeps
    reporting that after the junction is removed and the same object is
    hard-linked back at the validated name (verified by execution on a real
    NTFS junction; the handle cannot be re-pointed after the fact).

    POSIX (Linux): ``/proc/self/fd/<fd>`` reports the resolved dentry path
    used at open — the same binding semantics.

    Returns ``None`` when the platform cannot derive it; the caller then
    keeps the documented unusable-sidecar fallback (0) instead of trusting a
    handle whose provenance cannot be proven (the document's own
    ``high_water_seq`` still recovers the durable mark).

    Resolution caveat (rename-visibility): the kernel reports the CURRENT
    name of the open file description. A swap that renames the validated
    parent aside and recreates the SAME pathname (same-named hostile file)
    is location-invisible: the handle reports the validated pathname even
    though the object differs — that case is caught by the same-object
    (dev/ino) proof, not by the location proof. Location binding catches
    redirection to a DIFFERENT pathname (junction/symlink targets,
    mount-table swaps); object identity catches same-pathname swaps.
    """
    if os.name == "nt":
        import ctypes
        import msvcrt

        try:
            handle = msvcrt.get_osfhandle(fd)
            buffer = ctypes.create_unicode_buffer(64 * 1024)
            length = ctypes.windll.kernel32.GetFinalPathNameByHandleW(
                ctypes.c_void_p(handle),
                buffer,
                len(buffer),
                0,
            )
        except (OSError, ValueError, TypeError):
            return None
        if length <= 0 or length >= len(buffer):
            return None
        return buffer.value
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        # Non-Linux POSIX has no procfs; degrade to the unusable-sidecar
        # fallback rather than ingest bytes from an unbound handle.
        return None


def _normalized_location(path: str) -> str:
    """Canonical comparison form for a validated or handle-derived location.

    Strips the Win32 extended-length prefix and applies ``normpath`` +
    ``normcase`` so ``\\\\?\\C:\\X`` and ``c:/x`` compare equal. Deliberately
    does NOT resolve links (no ``realpath``): the expected side is the path
    ``reject_links`` just walked, and re-resolving it would re-open the
    check/use window this comparison exists to close.
    """
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[len("\\\\?\\UNC\\"):]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return os.path.normcase(os.path.normpath(path))


def _read_sidecar_seq(seq_path: Path) -> int:
    """Bounded, symlink-refused read of the sidecar high-water file.

    Returns 0 when absent or when the content is unusable (the document's own
    high_water_seq field still recovers the true value on load). Content is
    ASCII decimal only: non-decimal Unicode digits (``str.isdigit()`` true,
    ``int()`` raising) are unusable content, never a route failure (review
    r4310770081). Refuses
    (RecentError) on a NON-REGULAR sidecar, any link/reparse component in its
    path, or a handle that does not sit at the validated location — the ways
    a local actor could redirect these bytes (round-3/round-4 R3-2).
    ``load_recent_cases`` reads the sidecar BEFORE the main store's own
    ``check_store_path_safety``, so the sidecar path carries its own refusal.

    Provenance model (round-4): the first link refusal runs BEFORE the
    pre-open ``lstat`` and the open, so a path-only identity chain has an
    uncloseable validation-to-observation window — an ancestor swapped for a
    junction inside that window makes ``pre`` record the DECOY, and after the
    junction is removed (with the same decoy object hard-linked at the
    validated name) every subsequent path-based comparison agrees. The bytes
    are therefore bound by the handle's OWN location (``_handle_location``),
    fixed at open time by the traversal the kernel performed: a redirect at
    open shows the target's location and can never be re-pointed afterwards.
    The pre/post identity comparisons remain as defence-in-depth for swaps
    of the file object itself. Re-walking the path instead would only move
    the window.
    The identity witness also compares size (and ``st_ctime_ns`` on POSIX) and
    the already-open handle is re-stat'ed after the bounded read. That refuses
    the observed same-inode unlink/recreate case (which changes size and, on
    POSIX, ctime) and rejects a content mutation during the read instead of
    ingesting a mixed version. It is a narrowing witness, not an unforgeable
    identity proof: same-inode replacements that reproduce size and ctime,
    privileged ctime manipulation, and mutation after the final observation
    remain outside the guarantee.
    Observation failures (``OSError`` from ``fstat``/``lstat``/read, or an
    underivable handle location) keep the documented unusable-sidecar
    fallback of 0 and never escape the frozen recent envelope (round-3
    R3-3); deliberate link/type/location refusals still raise.
    """
    # 0. Canonical comparison form of the VALIDATED location: pure string
    #    normalization only (abspath/normpath/normcase — never realpath,
    #    which would re-walk and re-open the window the handle binding
    #    below exists to close).
    validated_location = _normalized_location(os.path.abspath(str(seq_path)))

    # 1. Component link refusal on the path exactly as validated.
    try:
        reject_links(str(seq_path), "recent_sidecar_path")
    except Refusal as exc:
        raise RecentError(
            "target_authority.symlink",
            f"high-water sidecar path failed link refusal: {exc.message}",
        ) from exc

    # 2. Capture the pre-open identity: type and (dev, ino) of the object the
    #    refusal above just approved. A directory can fail to open with a bare
    #    OSError on some platforms, so the type check must precede the open.
    try:
        pre = seq_path.lstat()
    except FileNotFoundError:
        return 0
    except OSError:
        return 0
    if not stat.S_ISREG(pre.st_mode):
        raise RecentError(
            "recent.store_unreadable",
            "high-water sidecar is not a regular file; refusing fail-closed",
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        # POSIX-only leaf-link refusal at open: without O_NOFOLLOW a leaf
        # swapped for a symlink between the lstat gate above and this open
        # would be followed. Windows has no O_NOFOLLOW (absent from os), so
        # the leaf is covered there by the lstat gate + location + identity
        # binding below; POSIX keeps the open-time refusal as well.
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = _open_sidecar(seq_path, flags)
    except FileNotFoundError:
        return 0
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ELOOP:
            # A link appeared after reject_links; refuse rather than follow.
            raise RecentError(
                "target_authority.symlink",
                "high-water sidecar resolved to a link; refusing fail-closed",
            ) from exc
        return 0
    try:
        # 3. The handle itself: regular (3a), located exactly where it was
        #    validated (3b), and the very object step 2 observed (3c).
        try:
            opened = os.fstat(fd)
        except OSError:
            return 0
        if not stat.S_ISREG(opened.st_mode):
            raise RecentError(
                "recent.store_unreadable",
                "high-water sidecar is not a regular file; refusing fail-closed",
            )
        # 3b. LOCATION BINDING — the proof that these bytes come from the
        #     validated, link-free path (round-4 R3-2). Step 1's refusal runs
        #     BEFORE step 2's observation, so an ancestor could be swapped for
        #     a junction inside that validation-to-observation window: ``pre``
        #     then records the DECOY object, the junction is removed again
        #     before the handle returns, and the same decoy object is
        #     hard-linked at the validated name — pre, handle, and post-hoc
        #     ``lstat`` all agree and every PATH-based check passes. The
        #     handle's location is fixed at open time by the traversal the
        #     kernel actually performed and cannot be re-pointed afterwards,
        #     so a mismatch proves a redirected ancestor and refuses. Another
        #     path-only recheck here would merely move the window.
        actual_location = _handle_location(fd)
        if actual_location is None:
            # Location cannot be derived from the handle on this platform:
            # never trust an unproven handle — unusable-sidecar fallback (0).
            return 0
        if _normalized_location(actual_location) != validated_location:
            raise RecentError(
                "target_authority.symlink",
                "high-water sidecar handle was opened through a redirected "
                "ancestor; refusing fail-closed",
            )
        # 3c. The handle must be the very object step 2 observed. A swap
        #     inside the open window changes the identity.
        if not _same_file(opened, pre):
            raise RecentError(
                "target_authority.symlink",
                "high-water sidecar changed between validation and open; "
                "refusing fail-closed",
            )
        # 4. Re-validate the ancestors NOW that the handle is anchored: a
        #    parent replaced by a junction after step 1 is refused here even
        #    though the handle still agrees with whatever it opened.
        try:
            reject_links(str(seq_path), "recent_sidecar_path")
        except Refusal as exc:
            raise RecentError(
                "target_authority.symlink",
                f"high-water sidecar path changed during validation: {exc.message}",
            ) from exc
        # 5. The path must still name the pre-validated object: a redirected
        #    ancestor resolves lstat to a different file than step 2 saw.
        try:
            after = seq_path.lstat()
        except OSError:
            return 0
        if not _same_file(after, pre):
            raise RecentError(
                "target_authority.symlink",
                "high-water sidecar was replaced during validation; "
                "refusing fail-closed",
            )
        if opened.st_size > _MAX_SIDECAR_BYTES:
            # Fast path: already too large, never read it.
            return 0
        try:
            data = _read_sidecar_bytes(fd)
        except OSError:
            # Unusable sidecar, not a route failure: fall back to 0 so the
            # recent handlers keep their frozen envelope (round-3 R3-3).
            return 0
        # Post-read content-version witness: the bytes just read must belong
        # to the same observation (dev/ino/size/ctime) that passed the path
        # and identity gates. A write that lands during the read is treated
        # as an unusable sidecar (0), never ingested as a mixed version.
        try:
            final = os.fstat(fd)
        except OSError:
            return 0
        if not _same_file(final, opened):
            return 0
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if len(data) > _MAX_SIDECAR_BYTES:
        # The read itself exceeded the cap (grown after validation): reject.
        return 0
    try:
        content = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return 0
    # ASCII decimal ONLY (PR3 review r4310770081): ``str.isdigit()`` is true for
    # non-decimal Unicode digit characters such as SUPERSCRIPT TWO (U+00B2), and
    # ``int()`` then raises ValueError. This read runs BEFORE the main document,
    # so that ValueError made a valid last-good store unavailable on BOTH routes
    # (generic ``invalid_request`` without ``no-store``, with the raw conversion
    # text) instead of taking the documented unusable-sidecar fallback. The
    # writer only ever emits ASCII decimal, so any other content is unusable by
    # definition.
    if not content.isascii() or not content.isdigit():
        return 0
    try:
        return int(content)
    except ValueError:
        # Defence in depth: any residual conversion failure is unusable sidecar
        # content, never a route failure.
        return 0


def read_durable_high_water_seq(store_path: Path) -> int:
    """Read durable sequence high-water mark independent of surviving rows."""
    with _HW_LOCK:
        path_key = _hw_path_key(store_path)
        val = _HIGH_WATER_BY_PATH.get(path_key, 0)
        seq_path = get_sidecar_seq_path(store_path)
        val = max(val, _read_sidecar_seq(seq_path))
        _HIGH_WATER_BY_PATH[path_key] = val
        return val


def persist_durable_high_water_seq(store_path: Path, seq: int) -> None:
    """Atomically raise the durable high-water mark to max(current, seq).

    Always persists the MERGED maximum (memory, sidecar, argument) — never a
    stale caller argument — under _HW_LOCK, with an unpredictable exclusive
    temp file and fsync (review P1-1/P1-2).
    """
    with _HW_LOCK:
        path_key = _hw_path_key(store_path)
        merged = max(_HIGH_WATER_BY_PATH.get(path_key, 0), seq)
        seq_path = get_sidecar_seq_path(store_path)
        try:
            merged = max(merged, _read_sidecar_seq(seq_path))
        except RecentError as exc:
            # SAFETY refusals (symlink/genuine-live on the sidecar path)
            # are fail-closed: the sidecar must never be written through a
            # hostile path, so the write is skipped and the merged maximum
            # survives in memory for this process. CONTENT malformation
            # cannot reach here — _read_sidecar_seq returns 0 for hostile
            # content and raises only for path-safety refusals. The
            # refusal propagates (it is not self-healed) so a hostile
            # sidecar can never be silently overwritten.
            raise RecentError(
                exc.code,
                "sidecar path refused; high-water mark kept in memory only",
            ) from exc
        _HIGH_WATER_BY_PATH[path_key] = merged
        tmp = seq_path.parent / f".tmp_{seq_path.name}_{os.getpid()}_{uuid4().hex}"
        use_seq_dir_fd = _dir_fd_replace_supported()
        seq_parent_fd: int | None = None
        seq_parent_bound_identity: tuple[int, int] | None = None
        try:
            seq_path.parent.mkdir(parents=True, exist_ok=True)
            if use_seq_dir_fd:
                # F1: acquisition-time binding — the handle is verified
                # against the validation-time parent identity before any
                # byte is written through it (and the pathname is
                # re-checked after binding). The ACQUISITION-bound pathname
                # identity travels with the handle for the R13 P2
                # pre-replace pathname gate.
                seq_parent_fd, seq_parent_bound_identity = _acquire_verified_parent(
                    seq_path.parent, seq_path
                )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            if use_seq_dir_fd:
                assert seq_parent_fd is not None
                fd = _create_temp_in_parent(
                    seq_parent_fd, tmp.name, flags, 0o600
                )
            else:
                fd = os.open(tmp, flags, 0o600)
            try:
                _write_all(fd, str(merged).encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            if hasattr(os, "chmod") and os.name != "nt":
                try:
                    os.chmod(tmp, 0o600)
                except OSError:
                    pass
            _atomic_replace(
                tmp,
                seq_path,
                action="replace",
                parent_fd=seq_parent_fd,
                parent_bound_identity=seq_parent_bound_identity,
            )
        except (RecentError, Refusal) as exc:
            try:
                if use_seq_dir_fd and seq_parent_fd is not None:
                    os.unlink(tmp.name, dir_fd=seq_parent_fd)
                elif tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if isinstance(exc, Refusal):
                raise RecentError(exc.code, exc.message) from exc
            raise
        except OSError as exc:
            try:
                if use_seq_dir_fd and seq_parent_fd is not None:
                    os.unlink(tmp.name, dir_fd=seq_parent_fd)
                elif tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            # Sidecar durability failure must not mask a successful main-store
            # write; memory still holds the merged maximum for this process.
            raise RecentError.storage_failure(
                "recent.store_unwritable", "sidecar replace", exc
            ) from exc
        finally:
            if seq_parent_fd is not None:
                try:
                    os.close(seq_parent_fd)
                except OSError:
                    pass


def _persist_high_water_best_effort(store_path: Path, seq: int) -> None:
    """Re-persist the derived high-water sidecar from a READ path.

    PR3 review #2 F1: a readable, valid main document must not become
    unavailable because the derived sidecar cannot be WRITTEN (read-only
    directory, disk full). ``persist_durable_high_water_seq`` already keeps the
    merged maximum in process memory before it writes, and the main document's
    own ``high_water_seq`` field is the durable source on restart — so only the
    STORAGE-failure code is swallowed here. Every path-SAFETY refusal
    (``target_authority.symlink`` / genuine-live / any other refusal code) still
    propagates fail-closed, exactly as before.
    """
    try:
        persist_durable_high_water_seq(store_path, seq)
    except RecentError as exc:
        if exc.code != "recent.store_unwritable":
            raise


def load_recent_cases(store_path: Path) -> dict[str, Any]:
    """Load recent cases document with explicit absent/codec/corrupt distinction.

    PR3 review F1 (serialized snapshots): the whole read runs under the SAME
    in-process ``_LOCK`` that every POST commit holds across its
    read-merge-assign-replace. A GET snapshot therefore can never overlap a
    main-document replacement: no Windows CRT sharing violation can fail an
    otherwise valid concurrent POST with ``recent.store_unwritable``, and no
    replacement can land between the post-open checks and the final lstat to
    produce a spurious ``target_authority.symlink`` on POSIX. The lock is
    NON-reentrant: internal callers that already hold it must call
    ``_load_recent_cases_unlocked``.

    Sidecar effect under the lock: the durable high-water mark is re-persisted
    from the parsed document on this path exactly as before; ``_LOCK`` is
    always acquired before ``_HW_LOCK`` (never the reverse), so no lock-order
    inversion is introduced.

    NEVER overwrites or alters the store on load. Degrades whole document to
    empty list per OWNER DECISION A on corrupt rows.
    """
    with _LOCK:
        return _load_recent_cases_unlocked(store_path)


def _load_recent_cases_unlocked(store_path: Path) -> dict[str, Any]:
    """Unlocked load body (see ``load_recent_cases``).

    The caller MUST already hold the non-reentrant ``_LOCK`` — notably
    ``process_recent_post``, which holds it across load + merge + replace so
    the merge sees a snapshot no concurrent reader/commit can move.
    """
    # Path.exists() follows symlinks: a dangling store link looks absent.
    # Validate the leaf and its ancestors before interpreting absence.
    # NOTE: check_store_path_safety must run with the REAL reject_links —
    # a witness that patches reject_links (e.g. for inner concurrent
    # writers) MUST restore it before the outer load, or absence and
    # refusal semantics change underfoot. The interleave witnesses below
    # restore ALL production seams for their inner POSTs for exactly
    # this reason.
    try:
        check_store_path_safety(store_path)
    except RecentError:
        raise
    except Exception as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "path validation", exc
        ) from exc

    # Provenance anchor for the C3 ancestry check below: the parent
    # identity as observed at validation time. Compared again after the
    # handle is bound; a change proves the ancestry was swapped.
    # Missing parents are NOT a refusal here: the ordinary absent-store
    # case (and a first write into a not-yet-created directory) has no
    # parent to sample yet.
    try:
        parent_at_validation = _storage_parent_identity(store_path.parent)
    except FileNotFoundError:
        parent_at_validation = None
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "stat", exc
        ) from exc

    hw_seq = read_durable_high_water_seq(store_path)

    # Existence is decided WITHOUT following links: lstat a dangling store
    # leaf reports the link itself, while exists() follows it and would hide
    # evidence of a prohibited path. An absent leaf (FileNotFoundError from
    # lstat) is the ordinary missing-store case; any OTHER lstat failure is
    # an actionable unreadable-store refusal.
    #
    # Windows sharing-note (review C3): the CRT read handle below denies
    # rename/replace of the store while it is open (ERROR_ACCESS_DENIED
    # when a concurrent POST replaces during this read). The handle is
    # therefore snapshot-copied and closed BEFORE parsing: no step after
    # the snapshot holds the store open, so a concurrent POST always lands
    # on a closed destination. The bytes below are the bound snapshot;
    # the ancestry gate above already proved them.

    # Existence is decided WITHOUT following links: lstat a dangling store
    # leaf reports the link itself, while exists() follows it and would hide
    # evidence of a prohibited path. An absent leaf (FileNotFoundError from
    # lstat) is the ordinary missing-store case; any OTHER lstat failure is
    # an actionable unreadable-store refusal.
    try:
        pre = store_path.lstat()
    except FileNotFoundError:
        return {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [],
            "discarded": 0,
            "reason": "absent",
            "digest": empty_document_digest(),
            "high_water_seq": hw_seq,
        }
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "stat", exc
        ) from exc
    if stat.S_ISLNK(pre.st_mode):
        raise RecentError(
            "target_authority.symlink",
            "recent store leaf is a link; refusing fail-closed",
        )
    if not stat.S_ISREG(pre.st_mode):
        raise RecentError(
            "recent.store_unreadable",
            "recent store is not a regular file; refusing fail-closed",
        )

    # Bind the document read to the handle produced by the just-approved
    # traversal. A separate Path.read_bytes would re-walk the path and
    # re-open the check/use window: the leaf could be replaced, or an
    # ancestor redirected, between validation and that later open.
    validated_location = _normalized_location(os.path.abspath(str(store_path)))

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        # POSIX-only leaf-link refusal at open (see sidecar open above):
        # Windows covers the leaf via lstat gate + location + identity.
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = _open_main_document(store_path, flags)
    except FileNotFoundError as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "open", exc
        ) from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ELOOP:
            raise RecentError(
                "target_authority.symlink",
                "recent store resolved to a link; refusing fail-closed",
            ) from exc
        raise RecentError.storage_failure(
            "recent.store_unreadable", "open", exc
        ) from exc
    try:
        try:
            opened = os.fstat(fd)
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "stat", exc
            ) from exc
        if stat.S_ISLNK(opened.st_mode):
            raise RecentError(
                "target_authority.symlink",
                "recent store leaf is a link; refusing fail-closed",
            )
        if not stat.S_ISREG(opened.st_mode):
            raise RecentError(
                "recent.store_unreadable",
                "recent store is not a regular file; refusing fail-closed",
            )
        actual_location = _handle_location(fd)
        if actual_location is None:
            raise RecentError(
                "recent.store_unreadable",
                "recent store handle provenance is unavailable; refusing fail-closed",
            )
        if _normalized_location(actual_location) != validated_location:
            raise RecentError(
                "target_authority.symlink",
                "recent store handle was opened through a redirected "
                "ancestor; refusing fail-closed",
            )
        # The handle is now BOUND: regular file at the validated location.
        #
        # Provenance anchor (review R12 P2): the parent identity sampled at
        # validation time above must be checked INDEPENDENTLY of leaf replacement
        # classification. If the parent is renamed aside and replaced by a new
        # real directory before the pre-open lstat, both pre and opened agree on
        # the replacement file, but the parent ancestry is swapped. The parent
        # identity must continuously agree with validation time; otherwise refuse
        # with target_authority.symlink even when replacement bytes are schema-valid.
        try:
            parent_now = _storage_parent_identity(store_path.parent)
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "stat", exc
            ) from exc
        if parent_at_validation is None or parent_now != parent_at_validation:
            raise RecentError(
                "target_authority.symlink",
                "recent store parent changed between validation and "
                "open; refusing fail-closed",
            )

        # Benign concurrent replacement (POST replace between lstat and open):
        # parent identity was verified unchanged above; an identity difference
        # between opened handle and pre-open lstat indicates a legitimate POST replace
        # within the continuously-owned parent.
        benign_replaced = not _same_file(opened, pre)
        try:
            reject_links(str(store_path), "recent_store_path")
        except Refusal as exc:
            raise RecentError(
                "target_authority.symlink",
                f"recent store path changed during validation: {exc.message}",
            ) from exc
        try:
            after = store_path.lstat()
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "stat", exc
            ) from exc
        if not benign_replaced and not _same_file(after, pre):
            raise RecentError(
                "target_authority.symlink",
                "recent store was replaced during validation; "
                "refusing fail-closed",
            )

        # The bound itself is enforced by the READ from this handle: growth
        # or replacement after the check can never allocate more than
        # MAX_RECENT_FILE_BYTES + 1, and one byte past the cap is enough to
        # reject excess content without ingesting it.
        #
        # Snapshot-copy + close BEFORE parse (see the Windows sharing-note
        # at the top of this function): the raw bytes are copied out of
        # the handle here and the handle is closed in the finally below,
        # so the parse/validate/merge steps that follow never hold the
        # store open against a concurrent POST replace.
        try:
            raw_bytes = bytes(_read_bounded_bytes(fd, MAX_RECENT_FILE_BYTES))
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "read", exc
            ) from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if len(raw_bytes) > MAX_RECENT_FILE_BYTES:
        raise RecentError(
            "recent.malformed",
            f"recent store file exceeds bounded limit ({len(raw_bytes)} > {MAX_RECENT_FILE_BYTES})",
        )

    # Zero-length file = absent, NOT corrupt (explicit physical/codec distinction)
    if len(raw_bytes) == 0:
        return {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [],
            "discarded": 0,
            "reason": "absent",
            "digest": empty_document_digest(),
            "high_water_seq": hw_seq,
        }

    try:
        text = raw_bytes.decode("utf-8")
        parsed = json.loads(text)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # PR3 review r4310080008: the DECODER itself can fail before
        # check_json_depth runs, well under the byte cap: ~6,000 nested
        # arrays raise RecursionError, and a JSON integer beyond the
        # interpreter digit limit raises plain ValueError (JSONDecodeError is
        # a ValueError subclass). Both are malformed content and must read as
        # the frozen recent.malformed refusal, never escape to the generic
        # HTTP boundary.
        raise RecentError(
            "recent.malformed",
            "recent store contains malformed JSON",
        ) from exc

    try:
        check_json_depth(parsed)
    except RecentError:
        raise
    except Exception as exc:
        raise RecentError(
            "recent.malformed",
            "recent store JSON depth validation failed",
        ) from exc

    if not isinstance(parsed, dict) or isinstance(parsed, list):
        raise RecentError("recent.malformed", "recent store root must be a JSON object")

    version = parsed.get("version")
    if version != RECENT_CASES_SCHEMA_VERSION:
        raise RecentError(
            "recent.unsupported_version",
            f"recent store version {version!r} is unsupported "
            f"(expected {RECENT_CASES_SCHEMA_VERSION})",
        )

    raw_entries = parsed.get("entries")
    if not isinstance(raw_entries, list):
        raise RecentError("recent.malformed", "recent store entries must be an array")

    persisted_hw = parsed.get("high_water_seq")
    if isinstance(persisted_hw, int) and not isinstance(persisted_hw, bool):
        hw_seq = max(hw_seq, persisted_hw)
        # Read-path sidecar persistence is best-effort (PR3 review #2 F1): a
        # readable document must not fail because the derived sidecar cannot
        # be written. Safety refusals still propagate.
        _persist_high_water_best_effort(store_path, hw_seq)

    # Durable replay-idempotence tail. Document-level, digest-excluded, and
    # forward/backward tolerant: older readers ignore the unknown key, and
    # this reader ignores malformed content (a tampered tail degrades to
    # no-dedup, never to a refusal or a fabricated open).
    seen_open_ids: list[str] = []
    raw_seen = parsed.get("seen_open_ids")
    if isinstance(raw_seen, list):
        for item in raw_seen:
            if (
                isinstance(item, str)
                and item
                and len(item) <= 128
                and len(seen_open_ids) < MAX_SEEN_OPEN_IDS
            ):
                if item not in seen_open_ids:
                    seen_open_ids.append(item)

    # OWNER DECISION A: ANY unusable row degrades the WHOLE document to empty
    entries: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for row in raw_entries:
        try:
            entry = validate_entry_row(row, for_persistence=True)
        except Exception:
            # Corrupt row: whole document degrades to empty with reason corrupt-rows
            return {
                "version": RECENT_CASES_SCHEMA_VERSION,
                "entries": [],
                "discarded": len(raw_entries),
                "reason": "corrupt-rows",
                "digest": hashlib.sha256(raw_bytes).hexdigest(),
                "high_water_seq": hw_seq,
            }
        key = recent_case_key(entry)
        if key in seen_keys:
            # Duplicate identity key is evidence of corruption/tampering
            return {
                "version": RECENT_CASES_SCHEMA_VERSION,
                "entries": [],
                "discarded": len(raw_entries),
                "reason": "corrupt-rows",
                "digest": hashlib.sha256(raw_bytes).hexdigest(),
                "high_water_seq": hw_seq,
            }
        seen_keys.add(key)
        hw_seq = max(hw_seq, entry["openedSeq"])
        entries.append(entry)

    if len(entries) > RECENT_CASES_LIMIT:
        # Over-cap entries is evidence of tampering
        return {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": [],
            "discarded": len(raw_entries),
            "reason": "corrupt-rows",
            "digest": hashlib.sha256(raw_bytes).hexdigest(),
            "high_water_seq": hw_seq,
        }

    # Final read-path sidecar persistence: best-effort storage (PR3 review #2
    # F1) — a readable, valid document stays readable when only the sidecar
    # cannot be written. Safety refusals still propagate fail-closed.
    _persist_high_water_best_effort(store_path, hw_seq)

    return {
        "version": RECENT_CASES_SCHEMA_VERSION,
        "entries": entries,
        "discarded": 0,
        "reason": "none",
        "digest": compute_document_digest(entries),
        "high_water_seq": hw_seq,
        "seen_open_ids": seen_open_ids,
    }


def save_recent_cases_to_disk(
    store_path: Path,
    entries: list[dict[str, Any]],
    high_water_seq: int,
    seen_open_ids: list[str] | None = None,
) -> str:
    """Atomic write via same-dir temp + fsync + os.replace.

    Preserves last-good bytes on any failure. Restrictive file (0o600) and
    dir (0o700) permissions.

    The COMPLETE serialized payload is checked against the read-side byte
    cap BEFORE any temp file is created: a display field that passes
    per-field string validation can still serialize past the 64 KiB load
    bound, and accepting that write would brick the store — every later
    load, including an empty-list clear, would fail first. Oversize is an
    actionable frozen refusal, never a last-good replacement.

    ``seen_open_ids`` (durable replay-idempotence tail) is persisted in
    the same document but OUTSIDE the digest: dedup bookkeeping must never
    invalidate client base-digests or masquerade as content.
    """
    seen_tail = list(seen_open_ids or [])[:MAX_SEEN_OPEN_IDS]
    payload = {
        "entries": entries,
        "high_water_seq": high_water_seq,
        "seen_open_ids": seen_tail,
        "version": RECENT_CASES_SCHEMA_VERSION,
    }
    raw_bytes = _json_bytes(payload)
    if len(raw_bytes) > MAX_RECENT_FILE_BYTES:
        raise RecentError(
            "recent.document_too_large",
            f"serialized recent document exceeds bounded limit "
            f"({len(raw_bytes)} > {MAX_RECENT_FILE_BYTES}); "
            "prior store bytes preserved",
        )

    check_store_path_safety(store_path)
    parent_dir = store_path.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "chmod") and os.name != "nt":
            try:
                os.chmod(parent_dir, 0o700)
            except OSError:
                pass
    except (PermissionError, OSError) as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", "directory creation", exc
        ) from exc

    # Temp creation THROUGH the pinned parent handle: the directory is
    # opened once after validation and the temp is created relative to
    # that handle, so a post-validation ancestor swap cannot redirect the
    # bytes outside the approved directory. POSIX dir_fd is used ONLY when
    # the platform actually offers it (Linux O_DIRECTORY + dir_fd replace
    # support); otherwise — including ALL Windows hosts — the pathname
    # discipline below applies with same-expression re-verification
    # immediately before use.
    use_dir_fd = _dir_fd_replace_supported()
    parent_fd: int | None = None
    parent_bound_identity: tuple[int, int] | None = None
    if use_dir_fd:
        # F1: sample the validated parent identity BEFORE opening, verify
        # the opened handle against that anchor (O_NOFOLLOW), re-check the
        # pathname once bound, and carry the SAME verified handle through
        # temp creation and replacement — no post-validation path walk can
        # redirect the payload outside the approved directory. The
        # ACQUISITION-bound pathname identity travels with the handle for
        # the R13 P2 pre-replace pathname gate.
        parent_fd, parent_bound_identity = _acquire_verified_parent(parent_dir, store_path)
    temp_name = f".tmp_{store_path.name}_{os.getpid()}_{uuid4().hex}"
    temp_path = parent_dir / temp_name
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if use_dir_fd:
            assert parent_fd is not None
            fd = _create_temp_in_parent(parent_fd, temp_name, flags, 0o600)
        else:
            # Windows fallback: no directory handles and no dir_fd — narrow
            # the window with a same-expression identity re-verification
            # immediately before creation (fail-closed on any observed
            # change), mirroring the replace-side fallback.
            try:
                before = _storage_parent_identity(parent_dir)
                check_store_path_safety(store_path)
                if _storage_parent_identity(parent_dir) != before:
                    raise RecentError(
                        "target_authority.symlink",
                        "recent store parent changed during validation; "
                        "refusing fail-closed",
                    )
            except RecentError:
                raise
            except OSError as exc:
                raise RecentError.storage_failure(
                    "recent.store_unwritable", "directory observation", exc
                ) from exc
            fd = os.open(temp_path, flags, 0o600)
        try:
            _write_all(fd, raw_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)

        if hasattr(os, "chmod") and os.name != "nt":
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass

        _atomic_replace(
            temp_path,
            store_path,
            action="replace",
            parent_fd=parent_fd,
            parent_bound_identity=parent_bound_identity,
        )
    except (RecentError, Refusal) as exc:
        try:
            if use_dir_fd:
                assert parent_fd is not None
                os.unlink(temp_name, dir_fd=parent_fd)
            elif temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if isinstance(exc, Refusal):
            raise RecentError(exc.code, exc.message) from exc
        raise
    except (PermissionError, OSError) as exc:
        try:
            if use_dir_fd:
                assert parent_fd is not None
                os.unlink(temp_name, dir_fd=parent_fd)
            elif temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        raise RecentError.storage_failure(
            "recent.store_unwritable", "replace", exc
        ) from exc
    if parent_fd is not None:
        try:
            os.close(parent_fd)
        except OSError:
            pass

    try:
        persist_durable_high_water_seq(store_path, high_water_seq)
    except (RecentError, Refusal):
        # PR3 review F2 (explicit commit point): the main-document replace
        # above IS the commit point. EVERY error class the optional sidecar
        # can raise after that point reports the COMMITTED result and never
        # claims the prior document was preserved:
        #   * RecentError storage failures (temp create/write/fsync/replace);
        #   * RecentError SECURITY refusals (target_authority.symlink /
        #     genuine-live on the sidecar path) — the hostile path is never
        #     written through and never self-healed;
        #   * a raw Refusal, caught defensively: the sidecar helper wraps
        #     refusals into RecentError, so this clause is belt-and-braces
        #     against a future unwrapped path (a raw Refusal escaping here
        #     would otherwise surface as a 400 after a committed write).
        # The high-water mark survives in this document's own
        # ``high_water_seq`` field (read back by the next load) and heals on
        # the next successful write. Every PRE-commit failure above still
        # refuses with last-good bytes preserved. A non-structured exception
        # (a programming error) is deliberately NOT swallowed: it surfaces as
        # an internal error, never as a false "prior preserved" claim.
        pass
    return compute_document_digest(entries)


def process_recent_post(
    store_path: Path,
    payload: Any,
    base_digest_header: str | None = None,
) -> dict[str, Any]:
    """Serialized read-validate-merge-assign-replace under in-process lock.

    Rejects stale writes with 409 recent.stale_snapshot. Assigns monotonic
    openedSeq from durable high-water mark; client openedSeq is ignored.

    Replay idempotence: a POST carrying a previously-consumed ``open_id``
    returns the current document UNCHANGED (no counter movement, same
    digest) instead of recording a second open — a duplicate delivery is
    not a second successful open (contract merge_semantics: only the
    successfully-opened identity accumulates, once per open).
    """
    if not isinstance(payload, dict):
        raise RecentError("recent.malformed", "POST payload must be a JSON object")

    check_json_depth(payload)

    version = payload.get("version")
    if version is not None and version != RECENT_CASES_SCHEMA_VERSION:
        raise RecentError(
            "recent.unsupported_version",
            f"unsupported recent cases version: {version!r}",
        )

    open_id: str | None = None
    if payload.get("open_id") is not None:
        open_id = validate_open_id(payload.get("open_id"))

    client_digest = (
        payload.get("base_digest")
        or payload.get("baseDigest")
        or base_digest_header
    )

    with _LOCK:
        # The unlocked helper: this function already holds the non-reentrant
        # commit lock, so the merge reads the same frozen snapshot the commit
        # replaces, and no concurrent GET can hold the store open across it.
        current = _load_recent_cases_unlocked(store_path)
        current_digest = current["digest"]
        current_entries = list(current["entries"])
        hw_seq = current["high_water_seq"]
        seen_open_ids = list(current.get("seen_open_ids", []))

        if open_id is not None and open_id in seen_open_ids:
            # Duplicate delivery of an already-consumed open: return the
            # current document unchanged — no openedSeq advance, no
            # openCount movement, same digest (so the client's base-digest
            # stays valid and no spurious 409 follows).
            return {
                "version": RECENT_CASES_SCHEMA_VERSION,
                "entries": current_entries,
                "discarded": current["discarded"],
                "digest": current_digest,
                "reason": current["reason"],
                "high_water_seq": hw_seq,
            }

        if client_digest is not None and client_digest != current_digest:
            raise RecentError(
                "recent.stale_snapshot",
                "stale snapshot digest; client must retry with merge",
                current={
                    "version": RECENT_CASES_SCHEMA_VERSION,
                    "entries": current_entries,
                    "digest": current_digest,
                    "reason": current["reason"],
                    "high_water_seq": hw_seq,
                },
            )

        # Determine incoming entries to process
        raw_incoming: list[Any] = []
        if "entry" in payload:
            raw_incoming = [payload["entry"]]
        elif "entries" in payload:
            if not isinstance(payload["entries"], list):
                raise RecentError("recent.malformed", "payload entries must be an array")
            raw_incoming = payload["entries"]
        elif "studyUid" in payload:
            raw_incoming = [payload]
        else:
            raise RecentError("recent.malformed", "payload must carry 'entry' or 'entries'")

        if not raw_incoming:
            # Empty list write / clear
            if open_id is not None:
                seen_open_ids = [open_id, *seen_open_ids][:MAX_SEEN_OPEN_IDS]
            new_digest = save_recent_cases_to_disk(
                store_path, [], hw_seq, seen_open_ids
            )
            return {
                "version": RECENT_CASES_SCHEMA_VERSION,
                "entries": [],
                "discarded": 0,
                "digest": new_digest,
                "reason": "none",
                "high_water_seq": hw_seq,
            }

        # Validate incoming rows fail-closed BEFORE any state change
        validated_incoming: list[dict[str, Any]] = []
        for raw in raw_incoming:
            try:
                val = validate_entry_row(raw, for_persistence=False)
            except Refusal as exc:
                raise RecentError(exc.code, exc.message) from exc
            validated_incoming.append(val)

        # Merge semantics: successfully-opened identity moves to front
        # and accumulates openCount (workspaceRecent.ts:270 policy).
        # Client openedSeq is IGNORED, server assigns monotonic openedSeq;
        # client openCount is likewise NEVER authority — each successful open
        # contributes exactly +1 (contract merge_semantics).
        target = validated_incoming[0]
        target_key = recent_case_key(target)

        prior_match: dict[str, Any] | None = None
        for candidate in current_entries:
            if recent_case_key(candidate) == target_key:
                prior_match = candidate
                break

        hw_seq += 1
        prior_count = prior_match["openCount"] if prior_match else 0
        inc_count = 1  # server-authoritative; client openCount is never authority
        merged_front: dict[str, Any] = {
            "studyUid": target["studyUid"],
            "seriesUid": target["seriesUid"],
            "scanSourcePath": target["scanSourcePath"],
            "stageBaseDir": target["stageBaseDir"],
            "destRoot": target["destRoot"],
            "openedSeq": hw_seq,
            "openCount": prior_count + inc_count if prior_match else inc_count,
        }

        survivors = [e for e in current_entries if recent_case_key(e) != target_key]

        # If incoming contained multiple rows (initial population / import)
        for additional in validated_incoming[1:]:
            add_key = recent_case_key(additional)
            if add_key == target_key or any(recent_case_key(s) == add_key for s in survivors):
                continue
            hw_seq += 1
            additional_entry = {
                "studyUid": additional["studyUid"],
                "seriesUid": additional["seriesUid"],
                "scanSourcePath": additional["scanSourcePath"],
                "stageBaseDir": additional["stageBaseDir"],
                "destRoot": additional["destRoot"],
                "openedSeq": hw_seq,
                "openCount": 1,  # server-authoritative; client openCount clamped
            }
            survivors.append(additional_entry)

        new_entries = [merged_front, *survivors][:RECENT_CASES_LIMIT]
        if open_id is not None:
            seen_open_ids = [open_id, *seen_open_ids][:MAX_SEEN_OPEN_IDS]
        new_digest = save_recent_cases_to_disk(
            store_path, new_entries, hw_seq, seen_open_ids
        )

        return {
            "version": RECENT_CASES_SCHEMA_VERSION,
            "entries": new_entries,
            "discarded": 0,
            "digest": new_digest,
            "reason": "none",
            "high_water_seq": hw_seq,
        }


def get_pending_open_path(store_path: Path) -> Path:
    """Server-owned pending first-open intent slot beside the store file."""
    return store_path.parent / f"{store_path.name}.pending"


def validate_pending_intent(value: Any) -> dict[str, Any]:
    """Validate a pending first-open intent (server-side, never JS-readable
    cross-origin beyond the same-origin pending route)."""
    if not isinstance(value, dict):
        raise RecentError("recent.malformed", "pending intent must be a JSON object")
    unknown = set(value) - {"id", "exp", "entry"}
    if unknown:
        raise RecentError(
            "recent.malformed",
            f"unexpected pending intent fields: {', '.join(sorted(unknown))}",
        )
    intent_id = value.get("id")
    if not isinstance(intent_id, str) or not intent_id or len(intent_id) > 128:
        raise RecentError("recent.malformed", "pending intent id is invalid")
    exp = value.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        raise RecentError("recent.malformed", "pending intent expiry is invalid")
    # PR3 review r4311421633: the type check above accepts every non-finite
    # value the decoder can produce — the VALID JSON number `1e309` decodes to
    # +inf, and the decoder also accepts the non-JSON `NaN`/`Infinity`
    # constants. Such an expiry is re-emitted by ``_json_bytes`` as
    # ``Infinity``/``NaN`` in a 200 body no browser ``response.json()`` can
    # parse, and ``now > exp`` is false for inf and NaN, so the stored intent
    # would never expire: a malformed transient slot could stay on disk and
    # keep breaking confirmation. An integer beyond float range is equally
    # unusable (and ``math.isfinite`` raises OverflowError for it). Refuse all
    # of them HERE, before any write; a planted slot with such content takes
    # the deliberate invalid-content cleanup below.
    try:
        exp_usable = math.isfinite(exp)
    except OverflowError:
        exp_usable = False
    if not exp_usable:
        raise RecentError("recent.malformed", "pending intent expiry is invalid")
    # N1 producer/validator contract: the first-open producer (app.js
    # setPendingOpen) queues the opened identity WITHOUT server counters
    # (no openedSeq/openCount keys at all); validate_entry_row defaults
    # them (0 / 1) and counters are server-assigned at record time. A
    # PRESENT counter must still be well-formed here, or a hostile slot
    # could smuggle counter authority through the pending path.
    raw_entry = value.get("entry")
    if isinstance(raw_entry, dict):
        for counter_key in ("openedSeq", "openCount"):
            if counter_key in raw_entry:
                counter_val = raw_entry[counter_key]
                if (
                    isinstance(counter_val, bool)
                    or not isinstance(counter_val, int)
                    or counter_val < (1 if counter_key == "openCount" else 0)
                ):
                    raise RecentError(
                        "recent.malformed",
                        f"pending intent entry {counter_key} is malformed",
                    )
    try:
        entry = validate_entry_row(raw_entry, for_persistence=False)
    except Refusal as exc:
        raise RecentError(exc.code, exc.message) from exc
    return {"id": intent_id, "exp": exp, "entry": entry}


def _read_pending_bytes(pending_path: Path) -> bytes:
    """Bounded handle-bound read of the pending slot (same discipline as
    the main document: lstat type gate, O_NOFOLLOW open, fstat regular
    check, location binding, capped read)."""
    try:
        pre = pending_path.lstat()
    except FileNotFoundError:
        raise RecentError("recent.absent", "no pending first-open intent")
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "stat", exc
        ) from exc
    if stat.S_ISLNK(pre.st_mode):
        raise RecentError(
            "target_authority.symlink",
            "pending intent leaf is a link; refusing fail-closed",
        )
    if not stat.S_ISREG(pre.st_mode):
        raise RecentError(
            "recent.store_unreadable",
            "pending intent is not a regular file; refusing fail-closed",
        )
    validated_location = _normalized_location(os.path.abspath(str(pending_path)))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        # POSIX-only leaf-link refusal at open (see sidecar open above):
        # Windows covers the leaf via lstat gate + location + identity.
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(pending_path, flags)
    except FileNotFoundError as exc:
        raise RecentError("recent.absent", "no pending first-open intent") from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ELOOP:
            raise RecentError(
                "target_authority.symlink",
                "pending intent resolved to a link; refusing fail-closed",
            ) from exc
        raise RecentError.storage_failure(
            "recent.store_unreadable", "open", exc
        ) from exc
    try:
        try:
            opened = os.fstat(fd)
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "stat", exc
            ) from exc
        if not stat.S_ISREG(opened.st_mode):
            raise RecentError(
                "recent.store_unreadable",
                "pending intent is not a regular file; refusing fail-closed",
            )
        actual_location = _handle_location(fd)
        if actual_location is None or _normalized_location(
            actual_location
        ) != validated_location:
            raise RecentError(
                "target_authority.symlink",
                "pending intent handle was opened through a redirected "
                "ancestor; refusing fail-closed",
            )
        if not _same_file(opened, pre):
            raise RecentError(
                "target_authority.symlink",
                "pending intent changed between validation and open; "
                "refusing fail-closed",
            )
        try:
            raw_bytes = _read_bounded_bytes(fd, MAX_RECENT_FILE_BYTES)
        except OSError as exc:
            raise RecentError.storage_failure(
                "recent.store_unreadable", "read", exc
            ) from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if len(raw_bytes) > MAX_RECENT_FILE_BYTES:
        raise RecentError(
            "recent.malformed",
            "pending intent exceeds the bounded limit; refusing fail-closed",
        )
    return raw_bytes


def load_pending_open(store_path: Path, *, now_ms: int | None = None) -> dict[str, Any] | None:
    """Return the live pending intent, or None when absent/expired.

    Expired intents are removed; the slot holds at most one intent. The whole
    read PLUS the expiry / invalid-content removal runs under
    ``_PENDING_LOCK`` so it cannot race a queue replacement (PR3 review #2
    F2): internal callers that already hold the lock must use
    ``_load_pending_open_unlocked``.
    """
    with _PENDING_LOCK:
        return _load_pending_open_unlocked(store_path, now_ms=now_ms)


def _unlink_pending_slot(pending_path: Path) -> None:
    """Best-effort removal of an unusable pending slot.

    Callers hold ``_PENDING_LOCK`` (the public pending entry points), so the
    removal can never race a queue replacement or a correlated clear.
    """
    try:
        pending_path.unlink()
    except OSError:
        pass


def _load_pending_open_unlocked(
    store_path: Path, *, now_ms: int | None = None
) -> dict[str, Any] | None:
    """Unlocked body of ``load_pending_open`` (caller holds ``_PENDING_LOCK``)."""
    import time as _time

    pending_path = get_pending_open_path(store_path)
    try:
        check_store_path_safety(pending_path)
    except RecentError:
        raise
    except Exception as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "path validation", exc
        ) from exc
    try:
        raw_bytes = _read_pending_bytes(pending_path)
    except RecentError as exc:
        if exc.code == "recent.absent":
            return None
        raise
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
        intent = validate_pending_intent(parsed)
    except RecentError:
        # Hostile content in a transient slot: drop it fail-closed rather
        # than serving it; the producer re-queues on its next switch.
        _unlink_pending_slot(pending_path)
        return None
    except (UnicodeDecodeError, ValueError, RecursionError):
        # PR3 review #4 F3 + r4310080008: malformed BYTES (truncated JSON,
        # invalid UTF-8, an integer beyond the interpreter digit limit) AND
        # decoder resource failures (deeply nested content, RecursionError)
        # take the SAME deliberate invalid-content path. They must never
        # escape as a raw decode error through the generic HTTP handler
        # (which would answer invalid_request, without no-store, and leave
        # the unusable slot on disk): the slot is dropped under
        # _PENDING_LOCK, the frozen pending response is `{pending: false}`
        # with no-store, and the next queue recovers without manual file
        # repair.
        _unlink_pending_slot(pending_path)
        return None
    now = now_ms if now_ms is not None else int(_time.time() * 1000)
    if now > intent["exp"]:
        _unlink_pending_slot(pending_path)
        return None
    return intent


def save_pending_open(store_path: Path, intent: dict[str, Any]) -> dict[str, Any]:
    """Queue (replace) the single pending first-open intent slot.

    Serialized with the correlated clear and the load-time expiry/invalid
    removal through ``_PENDING_LOCK`` (PR3 review #2 F2).
    """
    with _PENDING_LOCK:
        return _save_pending_open_unlocked(store_path, intent)


def _save_pending_open_unlocked(store_path: Path, intent: dict[str, Any]) -> dict[str, Any]:
    """Unlocked body of ``save_pending_open`` (caller holds ``_PENDING_LOCK``)."""
    validated = validate_pending_intent(intent)
    pending_path = get_pending_open_path(store_path)
    check_store_path_safety(pending_path)
    parent_dir = pending_path.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "chmod") and os.name != "nt":
            try:
                os.chmod(parent_dir, 0o700)
            except OSError:
                pass
    except (PermissionError, OSError) as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", "directory creation", exc
        ) from exc
    raw_bytes = _json_bytes(validated)
    if len(raw_bytes) > MAX_RECENT_FILE_BYTES:
        raise RecentError(
            "recent.document_too_large",
            "pending intent exceeds the bounded limit; prior state preserved",
        )
    temp_path = parent_dir / f".tmp_{pending_path.name}_{os.getpid()}_{uuid4().hex}"
    use_pending_dir_fd = _dir_fd_replace_supported()
    pending_parent_fd: int | None = None
    pending_parent_bound_identity: tuple[int, int] | None = None
    if use_pending_dir_fd:
        # F1: same acquisition-time binding as the main store and sidecar;
        # the ACQUISITION-bound pathname identity travels with the handle
        # for the R13 P2 pre-replace pathname gate.
        pending_parent_fd, pending_parent_bound_identity = _acquire_verified_parent(
            parent_dir, pending_path
        )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if use_pending_dir_fd:
            assert pending_parent_fd is not None
            fd = _create_temp_in_parent(
                pending_parent_fd, temp_path.name, flags, 0o600
            )
        else:
            fd = os.open(temp_path, flags, 0o600)
        try:
            _write_all(fd, raw_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)
        if hasattr(os, "chmod") and os.name != "nt":
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
        _atomic_replace(
            temp_path,
            pending_path,
            action="replace",
            parent_fd=pending_parent_fd,
            parent_bound_identity=pending_parent_bound_identity,
        )
    except (RecentError, Refusal) as exc:
        try:
            if use_pending_dir_fd and pending_parent_fd is not None:
                os.unlink(temp_path.name, dir_fd=pending_parent_fd)
            elif temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        if pending_parent_fd is not None:
            try:
                os.close(pending_parent_fd)
            except OSError:
                pass
        if isinstance(exc, Refusal):
            raise RecentError(exc.code, exc.message) from exc
        raise
    except (PermissionError, OSError) as exc:
        try:
            if use_pending_dir_fd and pending_parent_fd is not None:
                os.unlink(temp_path.name, dir_fd=pending_parent_fd)
            elif temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        if pending_parent_fd is not None:
            try:
                os.close(pending_parent_fd)
            except OSError:
                pass
        raise RecentError.storage_failure(
            "recent.store_unwritable", "replace", exc
        ) from exc
    if pending_parent_fd is not None:
        try:
            os.close(pending_parent_fd)
        except OSError:
            pass
    return {"id": validated["id"], "exp": validated["exp"]}


def clear_pending_open(store_path: Path, intent_id: str | None = None) -> bool:
    """Drop the pending intent (correlated by id when given).

    The id comparison and the unlink are ONE critical section under
    ``_PENDING_LOCK``, so a queue replacement installed concurrently is never
    deleted by a clear that matched the older intent (PR3 review #2 F2).
    """
    with _PENDING_LOCK:
        return _clear_pending_open_unlocked(store_path, intent_id)


def _clear_pending_open_unlocked(store_path: Path, intent_id: str | None = None) -> bool:
    """Unlocked body of ``clear_pending_open`` (caller holds ``_PENDING_LOCK``)."""
    pending_path = get_pending_open_path(store_path)
    try:
        check_store_path_safety(pending_path)
    except RecentError:
        raise
    except Exception as exc:
        raise RecentError.storage_failure(
            "recent.store_unreadable", "path validation", exc
        ) from exc
    if intent_id is not None:
        # Unlocked helper: this function already holds _PENDING_LOCK.
        current = _load_pending_open_unlocked(store_path)
        if current is not None and current["id"] != intent_id:
            return False
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RecentError.storage_failure(
            "recent.store_unwritable", "clear", exc
        ) from exc
    return True


def probe_recent_availability(
    source_path: str,
    stage_base: str,
    study_uid: str,
) -> dict[str, Any]:
    """Read-only staged-identity and availability probe for a case.

    Implements reuse-first policy (ready, source-missing, staged-missing, unsafe-entry)
    with genuine-live and containment checks before any inspection.
    """
    validate_uid(study_uid, "studyUid")
    check_raw_path_text(source_path, "scanSourcePath")
    check_raw_path_text(stage_base, "stageBaseDir")

    if is_genuine_denied(source_path):
        raise Refusal("target_authority.genuine_live", "scanSourcePath matches genuine-live marker")
    if is_genuine_denied(stage_base):
        raise Refusal("target_authority.genuine_live", "stageBaseDir matches genuine-live marker")

    resolved_source = resolve_path(source_path, "scanSourcePath")
    resolved_stage = resolve_path(stage_base, "stageBaseDir")
    raw_dest = resolved_stage / study_uid

    reject_links(source_path, "scanSourcePath")
    reject_links(stage_base, "stageBaseDir")
    reject_links(str(raw_dest), "dest")

    resolved_dest = raw_dest.resolve()
    if is_genuine_denied(str(resolved_dest)):
        raise Refusal("target_authority.genuine_live", "dest resolves into genuine-live marker")

    if resolved_dest == resolved_source:
        raise Refusal("library.dest_equals_source", "dest must not equal scan source")
    if resolved_source in resolved_dest.parents:
        raise Refusal("library.dest_inside_source", "dest must not lie inside scan source")
    if resolved_stage not in resolved_dest.parents:
        raise Refusal("target_authority.dest_outside_stage_base", "dest must lie inside stage base")

    # Check filesystem staged layout vs source existence
    staged_exists = False
    file_count = 0
    series_uids: list[str] = []
    if resolved_dest.is_dir():
        try:
            census = census_staged_root(resolved_dest)
            series_uids = census["series_uids"]
            file_count = census["file_count"]
            staged_exists = bool(series_uids)
        except Refusal:
            # Safety/layout refusals (malformed staged destination) are NOT
            # genuine missing staging: re-raise so the probe answers with
            # the actionable 400 target_authority.dest_layout refusal
            # instead of the staged-missing "will re-stage on open" promise
            # that a later open would itself refuse (finding: the old
            # blanket except reported malformed destinations as
            # staged-missing). Only genuinely unexpected errors degrade to
            # missing (transient teardown races).
            raise
        except Exception:
            staged_exists = False

    source_exists = resolved_source.is_dir()

    if staged_exists:
        # Reuse-first: staged copy hit can open directly
        availability = "ready" if source_exists else "source-missing"
        reuse = True
    elif source_exists:
        # Not staged yet, but source exists: will stage on open
        availability = "staged-missing"
        reuse = False
    else:
        # Neither exists
        availability = "unsafe-entry"
        reuse = False

    return {
        "availability": availability,
        "reuse": reuse,
        "study_uid": study_uid,
        "dest_root": str(resolved_dest),
        "source_exists": source_exists,
        "staged_exists": staged_exists,
        "series_uids": series_uids,
        "file_count": file_count,
    }
