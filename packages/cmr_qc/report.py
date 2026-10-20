"""Aggregate QC outcome for a study/series (PASS or REVIEW REQUIRED)."""

from __future__ import annotations

from dataclasses import dataclass, field

from cmr_core.errors import CMRWarning


@dataclass
class QCReport:
    """Summary of all QC flags collected for a result."""

    flags: list[CMRWarning] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flags

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "REVIEW REQUIRED"

    def add(self, flag: CMRWarning) -> None:
        self.flags.append(flag)

    def extend(self, flags: list[CMRWarning]) -> None:
        self.flags.extend(flags)

    def __str__(self) -> str:  # pragma: no cover - trivial
        lines = [f"QC: {self.verdict} ({len(self.flags)} flag(s))"]
        lines.extend(f"  - {f}" for f in self.flags)
        return "\n".join(lines)
