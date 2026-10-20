"""DICOM source manifest: content-addressed raw-source workflow identity.

S2C-001 cluster A. A :class:`DicomSourceManifest` binds ONE persisted
workflow to its exact contributing raw DICOM bytes:

```text
raw file bytes -> file_sha256 -> DicomSourceManifest payload
              + decoded CineVolume digest -> workflow_id
```

``workflow_id`` IS the manifest content digest
(``sha256(canonical_json(payload).encode("utf-8"))``); there is deliberately
no second redundant manifest id. The identity changes whenever any
contributing raw byte, UID, relative path, decoded geometry, ordered frame or
pixel value changes, because the cine volume digest participates in the
payload.

Privacy rule: only NORMALIZED POSIX RELATIVE paths participate in identity —
never an absolute import-directory path. Relative filenames may still contain
identifying text; storing them is a documented production-readiness (privacy)
gate outside this milestone.

Validation is fail-closed at construction: absolute paths, ``..`` traversal,
blank UIDs, blank digests, duplicate relative paths and duplicate SOP UIDs are
rejected before any identity exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cmr_core.domain_types import canonical_digest

__all__ = [
    "DICOM_SOURCE_MANIFEST_DIGEST_VERSION",
    "DicomSourceFile",
    "DicomSourceManifest",
    "normalize_relative_path",
]

#: Version tag inside the canonical manifest payload (identity-relevant).
DICOM_SOURCE_MANIFEST_DIGEST_VERSION = "dicom-source-manifest-v2"

_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_hex64(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(
            f"{name} must be 64-character lowercase hex, got {value!r}"
        )
    return value


def _require_non_blank(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def normalize_relative_path(raw_path: str) -> str:
    """Normalize one contributor path to a safe POSIX relative path.

    Backslashes become forward slashes; absolute paths (leading ``/``, drive
    prefixes such as ``C:``), any ``..`` traversal segment and any ``.``
    no-op segment are REJECTED — identity must never leak an import-directory
    path, escape the source folder, or collapse distinct spellings onto one
    entry. The result is returned otherwise unchanged.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"relative_path must be a non-empty string, got {raw_path!r}")
    candidate = raw_path.replace("\\", "/").strip()
    if candidate.startswith("/"):
        raise ValueError(
            f"absolute path {raw_path!r} is forbidden in source identity"
        )
    first = candidate.split("/", 1)[0]
    if len(first) >= 2 and first[1] == ":":
        raise ValueError(
            f"drive-qualified path {raw_path!r} is forbidden in source identity"
        )
    segments = [segment for segment in candidate.split("/") if segment != ""]
    if not segments:
        raise ValueError(f"relative_path {raw_path!r} normalizes to nothing")
    if any(segment == ".." for segment in segments):
        raise ValueError(
            f"path traversal {raw_path!r} is forbidden in source identity"
        )
    # Correction P2: "." segments are rejected (not collapsed) so two
    # different path spellings can never silently map onto one identity row.
    if any(segment == "." for segment in segments):
        raise ValueError(
            f"no-op segment {raw_path!r} is forbidden in source identity"
        )
    return "/".join(segments)


@dataclass(frozen=True)
class DicomSourceFile:
    """One contributing DICOM file inside a :class:`DicomSourceManifest`.

    ``ordinal`` is NOT part of construction: ordinals are contiguous positions
    derived deterministically from the manifest's canonical file ordering
    (normalized POSIX ``relative_path`` then SOP UID), so they can never drift
    from the sorted identity payload.
    """

    relative_path: str
    sop_instance_uid: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_path", normalize_relative_path(self.relative_path)
        )
        _require_non_blank("sop_instance_uid", self.sop_instance_uid)
        _require_hex64("file_sha256", self.file_sha256)


@dataclass(frozen=True)
class DicomSourceManifest:
    """Content-addressed identity of one raw-DICOM source workflow.

    The ``files`` tuple is re-sorted by ``(relative_path, sop_instance_uid)``
    at construction and duplicate keys fail closed, so two manifests built
    from the same contributing set always derive one identical
    :attr:`workflow_id`.
    """

    study_uid: str
    series_uid: str
    cine_volume_digest: str
    files: tuple[DicomSourceFile, ...]
    temporal_axis_provenance: Literal["measured", "derived"] = "measured"
    workflow_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_blank("study_uid", self.study_uid)
        _require_non_blank("series_uid", self.series_uid)
        _require_hex64("cine_volume_digest", self.cine_volume_digest)
        if self.temporal_axis_provenance not in ("measured", "derived"):
            raise ValueError(
                f"temporal_axis_provenance must be 'measured' or 'derived', "
                f"got {self.temporal_axis_provenance!r}"
            )
        # Correction P2: a manifest with NO contributing file has no source
        # identity to speak of — an empty contributor set fails closed here
        # instead of deriving a degenerate workflow id.
        if not self.files:
            raise ValueError(
                "a DicomSourceManifest requires at least one contributing file"
            )
        if isinstance(self.files, list):
            object.__setattr__(self, "files", tuple(self.files))
        for source_file in self.files:
            if not isinstance(source_file, DicomSourceFile):
                raise ValueError(
                    "manifest files must be DicomSourceFile instances, got "
                    f"{type(source_file).__name__}"
                )
        ordered = tuple(
            sorted(self.files, key=lambda item: (item.relative_path, item.sop_instance_uid))
        )
        seen_paths: set[str] = set()
        seen_sops: set[str] = set()
        for source_file in ordered:
            if source_file.relative_path in seen_paths:
                raise ValueError(
                    f"duplicate contributing relative_path "
                    f"{source_file.relative_path!r}"
                )
            if source_file.sop_instance_uid in seen_sops:
                raise ValueError(
                    f"duplicate contributing SOPInstanceUID "
                    f"{source_file.sop_instance_uid!r}"
                )
            seen_paths.add(source_file.relative_path)
            seen_sops.add(source_file.sop_instance_uid)
        object.__setattr__(self, "files", ordered)
        object.__setattr__(self, "workflow_id", canonical_digest(self.to_canonical()))

    @property
    def is_temporal_axis_derived(self) -> bool:
        return self.temporal_axis_provenance == "derived"

    @property
    def temporal_axis_derived(self) -> bool:
        return self.temporal_axis_provenance == "derived"

    @property
    def is_derived_phase_axis(self) -> bool:
        return self.temporal_axis_provenance == "derived"

    def to_canonical(self) -> dict[str, Any]:
        """Canonical identity payload (design spec §2.1, exact field set)."""
        return {
            "domain": "DicomSourceManifest",
            "digest_version": DICOM_SOURCE_MANIFEST_DIGEST_VERSION,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "cine_volume_digest": self.cine_volume_digest,
            "temporal_axis_provenance": self.temporal_axis_provenance,
            "files": [
                {
                    "ordinal": ordinal,
                    "relative_path": source_file.relative_path,
                    "sop_instance_uid": source_file.sop_instance_uid,
                    "file_sha256": source_file.file_sha256,
                }
                for ordinal, source_file in enumerate(self.files)
            ],
        }
