"""cmr_qc — deterministic QC rules producing QCWarning flags."""

from .report import QCReport
from .rules import (
    check_containment,
    check_missing_structures,
    check_self_intersection,
    check_volume_curve,
    run_qc_on_contour_set,
)

__all__ = [
    "QCReport",
    "check_containment",
    "check_missing_structures",
    "check_self_intersection",
    "check_volume_curve",
    "run_qc_on_contour_set",
]
