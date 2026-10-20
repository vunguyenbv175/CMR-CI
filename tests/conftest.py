"""Test path setup + shared deterministic test support.

- Makes packages/services importable without installation.
- Provides FakeProcess, the shared scripted subprocess stand-in used by the
  Orchestrator state-machine tests (quota pause, routing integration). It
  starts "not done" (poll() -> None) and advances only when the test
  explicitly calls mark_done()/mark_failed(), so every tick outcome is
  deterministic.

Import in test modules as: from tests.conftest import FakeProcess
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _dir in (ROOT / "packages", ROOT / "services", ROOT / "tests"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))


class FakeProcess:
    """Scripted subprocess stand-in for orchestrator state-machine tests."""

    def __init__(self):
        self._code = None

    def mark_done(self):
        self._code = 0

    def mark_failed(self, code=1):
        self._code = code

    def poll(self):
        return self._code
