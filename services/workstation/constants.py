"""Small frozen constants owned by the workstation web client."""

from __future__ import annotations

from typing import Final

# This is the single keyboard mapping source.  ``static/kbmap.js`` embeds
# the exact JSON object so the no-build browser client cannot drift from the
# server-reviewed shortcut vocabulary.
KEYBOARD_SHORTCUTS: Final[dict[str, str]] = {
    " ": "toggle_playback",
    "ArrowLeft": "step_backward",
    "ArrowRight": "step_forward",
    "ArrowUp": "previous_slice",
    "ArrowDown": "next_slice",
    "+": "increase_speed",
    "-": "decrease_speed",
    "e": "mark_ed",
    "s": "mark_es",
    "?": "toggle_shortcuts",
}

__all__ = ["KEYBOARD_SHORTCUTS"]
