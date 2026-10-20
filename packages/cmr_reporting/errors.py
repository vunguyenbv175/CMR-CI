"""Errors raised by the structured reporting boundary."""

from __future__ import annotations


class ReportError(ValueError):
    """A fail-closed reporting error.

    The reporting API uses one exception type so callers cannot accidentally
    treat a malformed or unverifiable report as a usable artifact.  Messages
    begin with a stable ``REPORT_*:`` category prefix; the remainder is
    intentionally descriptive and is not a second error taxonomy.
    """


__all__ = ["ReportError"]
