"""Node 8 — pure playhead: deterministic frame advancement, zero threads.

The playhead is a PURE modulo stepper (AMENDMENT-001 E):

- :meth:`PlayheadController.advance` is stateless:
  ``(frame_index + 1) % n_frames`` — one tick is exactly one modulo step;
- :meth:`PlayheadController.advance_state` reads the frozen
  :class:`~cmr_viewer.state.ViewerState` and returns the next frame index;
  it never mutates state (the caller composes the new immutable state);
- a paused or unbound :meth:`~PlayheadController.tick` returns the current
  index UNCHANGED; ticks only step while an explicit session is playing;
- :meth:`~PlayheadController.set_fps` CLAMPS to the inclusive range
  :data:`~cmr_viewer.state.MIN_FPS`..:data:`~cmr_viewer.state.MAX_FPS`
  (the interactive surface must stay usable, so clamping — not failing —
  is the specified behavior here; the frozen state itself validates);
- there is NO ``threading.Thread`` anywhere in this milestone. The
  interactive shell schedules ticks through Tk's cooperative ``after()``
  timer at ``interval_ms`` milliseconds; headless tests drive ticks by hand.
"""

from __future__ import annotations

from cmr_core.errors import INVALID_ARGUMENT, CMRError

from cmr_viewer.state import DEFAULT_FPS, MAX_FPS, MIN_FPS, ViewerState

__all__ = ["PlayheadController"]


def _clamp_fps(fps: int) -> int:
    """Clamp any integer fps into the supported inclusive range."""
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise CMRError(
            INVALID_ARGUMENT,
            f"fps must be an integer number of frames per second, got {fps!r}",
        )
    return max(MIN_FPS, min(MAX_FPS, fps))


class PlayheadController:
    """Deterministic playback stepper over one bound cine frame count.

    The instance carries only presentation-session bookkeeping (fps, bound
    frame count, session index, playing flag) — never clinical data. All
    clinical state stays in the frozen :class:`~cmr_viewer.state.ViewerState`.
    """

    def __init__(self, fps: int = DEFAULT_FPS) -> None:
        self._fps = _clamp_fps(fps)
        self._n_frames: int | None = None
        self._current_frame_index = 0
        self.playing = False

    # --- configuration -------------------------------------------------------

    @property
    def fps(self) -> int:
        """Effective frames-per-second (always within 1..60)."""
        return self._fps

    @property
    def interval_ms(self) -> int:
        """Timer interval in whole milliseconds used by the shell scheduler."""
        return round(1000 / self._fps)

    def set_fps(self, fps: int) -> int:
        """Clamp ``fps`` into 1..60, store it, return the effective value."""
        self._fps = _clamp_fps(fps)
        return self._fps

    # --- pure advancement ------------------------------------------------------

    @staticmethod
    def advance(frame_index: int, n_frames: int) -> int:
        """Stateless modulo step: ``(frame_index + 1) % n_frames``.

        Fails closed on non-integer inputs, empty cine grids and out-of-range
        indices instead of wrapping garbage into the display.
        """
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise CMRError(
                INVALID_ARGUMENT,
                f"frame_index must be an integer, got {frame_index!r}",
            )
        if isinstance(n_frames, bool) or not isinstance(n_frames, int):
            raise CMRError(
                INVALID_ARGUMENT,
                f"n_frames must be an integer, got {n_frames!r}",
            )
        if n_frames <= 0:
            raise CMRError(
                INVALID_ARGUMENT,
                f"n_frames must be positive, got {n_frames}",
            )
        if not 0 <= frame_index < n_frames:
            raise CMRError(
                INVALID_ARGUMENT,
                f"frame_index {frame_index} is outside the frame range "
                f"0..{n_frames - 1}",
            )
        return (frame_index + 1) % n_frames

    def advance_state(self, state: ViewerState) -> int:
        """Return the next frame index for ``state`` without mutating it."""
        return self.advance(state.current_frame_index, state.cine.n_frames)

    # --- interactive session (shell-only bookkeeping) --------------------------

    def bind(self, n_frames: int, *, frame_index: int = 0) -> None:
        """Bind the session stepper to a cine frame count."""
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
            raise CMRError(
                INVALID_ARGUMENT,
                f"n_frames must be a positive integer, got {n_frames!r}",
            )
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise CMRError(
                INVALID_ARGUMENT,
                f"frame_index must be an integer, got {frame_index!r}",
            )
        if not 0 <= frame_index < n_frames:
            raise CMRError(
                INVALID_ARGUMENT,
                f"frame_index {frame_index} is outside the frame range "
                f"0..{n_frames - 1}",
            )
        self._n_frames = n_frames
        self._current_frame_index = frame_index

    def sync(self, frame_index: int) -> None:
        """Re-sync the session stepper after external navigation."""
        if self._n_frames is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "playhead session is not bound; call bind() first",
            )
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < self._n_frames
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                f"frame_index {frame_index!r} is outside the bound frame range",
            )
        self._current_frame_index = frame_index

    @property
    def current_frame_index(self) -> int:
        """Session frame index last produced (or seeded) by this controller."""
        return self._current_frame_index

    def start(self) -> None:
        """Resume ticking (requires a bound session)."""
        if self._n_frames is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "cannot start an unbound playhead session; call bind() first",
            )
        self.playing = True

    def stop(self) -> None:
        """Pause ticking; subsequent ticks return the unchanged index."""
        self.playing = False

    def tick(self) -> int:
        """Advance one modulo step while playing; otherwise stay put.

        One tick == one modulo step of the BOUND frame count. A paused or
        unbound tick is a no-op returning the unchanged index.
        """
        if not self.playing or self._n_frames is None:
            return self._current_frame_index
        self._current_frame_index = self.advance(
            self._current_frame_index, self._n_frames
        )
        return self._current_frame_index
