"""cmr_viewer — Node 8 physician-review viewer over accepted primitives.

Headless-first composition layer (AMENDMENT-001): deterministic Agg
rendering + pure playback + testable application core are importable
everywhere; the interactive Tk shell is resolved LAZILY (PEP 562) so
importing :mod:`cmr_viewer` never requires ``tkinter`` or a display.

The package pulls in NO inference stack, no network client and no GPU/cloud
dependency — enforced by the negative import-boundary gate in
``tests/ui/test_viewer_smoke.py``.
"""

from __future__ import annotations

from typing import Any

from cmr_viewer.app import CmrStoreLike, ViewerApp, WorkflowControllerLike, reopen_viewer_state
from cmr_viewer.playback import PlayheadController
from cmr_viewer.renderer import RenderedFrame, ViewerRenderer, frame_digest
from cmr_viewer.state import (
    APPROVED_LABEL,
    DECISION_RECORDED_LABEL,
    DEFAULT_FPS,
    MAX_FPS,
    MIN_FPS,
    UNAPPROVED_LABEL,
    ViewerState,
)

_SHELL_NAMES = ("ViewerShell", "main", "VIEWER_WINDOW_TITLE", "BUTTON_LABELS")


def __getattr__(name: str) -> Any:
    if name in _SHELL_NAMES:
        from cmr_viewer import shell

        return getattr(shell, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SHELL_NAMES))


__all__ = [
    "APPROVED_LABEL",
    "BUTTON_LABELS",
    "CmrStoreLike",
    "DECISION_RECORDED_LABEL",
    "DEFAULT_FPS",
    "MAX_FPS",
    "MIN_FPS",
    "RenderedFrame",
    "UNAPPROVED_LABEL",
    "ViewerApp",
    "ViewerRenderer",
    "ViewerShell",
    "ViewerState",
    "VIEWER_WINDOW_TITLE",
    "PlayheadController",
    "WorkflowControllerLike",
    "frame_digest",
    "reopen_viewer_state",
]
