"""Contour/ContourSet domain model (Task F).

A :class:`Contour` is a closed polygon labelled with structure, frame and
slice indices, its origin (ai/manual) and provenance. AI predictions and
physician corrections are never mixed: an edited contour keeps its original
``source`` and gains ``edited=True`` plus a bumped revision; the unedited
prediction is preserved elsewhere (data flywheel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cmr_core.errors import CONTOUR_INVALID, CMRError
from cmr_core.provenance import ModelMetadata, Provenance
from cmr_core.segmentation import Structure


@dataclass
class Contour:
    """One closed contour polygon.

    ``points`` are (x, y) pairs in **pixel coordinates** of the slice image;
    physical (mm) coordinates are derived via ImageGeometry. Keeping pixel
    coordinates as the canonical frame avoids rounding during edits.
    """

    slice_index: int
    frame_index: int
    structure: Structure
    points: list[tuple[float, float]]
    source: str = "ai"  # "ai" | "manual"
    edited: bool = False
    approved: bool = False
    revision: int = 0
    model: ModelMetadata | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.slice_index < 0:
            raise CMRError(CONTOUR_INVALID, f"slice_index must be >= 0, got {self.slice_index}")
        if self.frame_index < 0:
            raise CMRError(CONTOUR_INVALID, f"frame_index must be >= 0, got {self.frame_index}")
        if self.source not in ("ai", "manual"):
            raise CMRError(CONTOUR_INVALID, f"source must be 'ai' or 'manual', got {self.source!r}")
        if len(self.points) < 3:
            raise CMRError(CONTOUR_INVALID, f"contour needs >= 3 points, got {len(self.points)}")

    @property
    def is_closed(self) -> bool:
        if not self.points:
            return False
        return self.points[0] == self.points[-1]

    def mark_edited(self, model: ModelMetadata | None = None) -> None:
        """Physician touched this contour: edit flag + revision bump.

        Review e5614a6 P1-2 (FORK_UNAPPROVED_DERIVED_REVISION_V1): this legacy
        IN-PLACE API fails closed BEFORE any mutation when ``approved`` is True
        — an approved revision is immutable upstream and edits must fork an
        unapproved derived contour instead. When called without a replacement
        model, the current model identity is PRESERVED (``model=None`` here
        means "no replacement", never "erase provenance").
        """
        if self.approved:
            raise CMRError(
                CONTOUR_INVALID,
                "approved contour cannot be edited in place; changed content "
                "must fork an unapproved derived revision "
                "(FORK_UNAPPROVED_DERIVED_REVISION_V1)",
            )
        self.edited = True
        self.revision += 1
        if self.provenance is not None:
            self.provenance = self.provenance.bumped()
        if model is not None:
            self.model = model

    def approve(self) -> None:
        self.approved = True

    def as_mm(
        self, pixel_spacing_mm: tuple[float, float] | None = None
    ) -> list[tuple[float, float]]:
        """Contour points converted to mm (row_spacing, col_spacing) or
        raw points when spacing is unknown (explicit — no silent units)."""
        if pixel_spacing_mm is None:
            return list(self.points)
        rs, cs = pixel_spacing_mm
        return [(x * cs, y * rs) for x, y in self.points]


@dataclass
class ContourSet:
    """All contours of one (slice, frame) pair, keyed by structure."""

    slice_index: int
    frame_index: int
    contours: dict[Structure, Contour] = field(default_factory=dict)

    def add(self, contour: Contour) -> None:
        if (contour.slice_index, contour.frame_index) != (self.slice_index, self.frame_index):
            raise CMRError(
                CONTOUR_INVALID,
                f"contour ({contour.slice_index},{contour.frame_index}) does not belong to set "
                f"({self.slice_index},{self.frame_index})",
            )
        self.contours[contour.structure] = contour

    def get(self, structure: Structure) -> Contour | None:
        return self.contours.get(structure)

    @property
    def structures(self) -> set[Structure]:
        return set(self.contours)

    def approve_all(self) -> None:
        for c in self.contours.values():
            c.approve()
