"""CPU-only, stdlib HTTP workstation for synthetic CMR cine review.

The package is deliberately a thin transport/composition boundary.  Pixels
and clinical values are produced by the frozen ``cmr_dicom``, ``cmr_viewer``,
``cmr_workflow`` and ``cmr_reporting`` surfaces; this package does not
reimplement those algorithms.
"""

from __future__ import annotations

from .server import (
    WorkstationApplication,
    WorkstationConfigurationError,
    bootstrap_application,
    create_http_server,
    normalize_authority,
    run,
)

__all__ = [
    "WorkstationApplication",
    "WorkstationConfigurationError",
    "bootstrap_application",
    "create_http_server",
    "normalize_authority",
    "run",
]
