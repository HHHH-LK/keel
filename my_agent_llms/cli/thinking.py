"""Scroll-safe 'thinking…' spinner.

Why not `console.status()` / `rich.live.Live`?
    A Live region repaints on a timer by emitting *cursor-up N → clear → redraw*
    sequences. The instant the user scrolls the terminal, the viewport/cursor
    relationship breaks: the refresh thread keeps firing those vertical-cursor
    moves relative to where it *thinks* the bottom is, and ends up redrawing over
    the wrong region — the screen "reloads" and the streamed output is clobbered.

This spinner only ever touches the CURRENT line via '\\r' (carriage return) +
'erase-line'. No vertical cursor movement at all → scrolling can't desync it.
New output simply snaps the terminal back to the bottom and rewrites that one
line. Matches chat_view.py's "无 Live" design intent.

Contract: start() must be called with the cursor on a fresh (empty) line; stop()
leaves the cursor at column 0 of a cleared line (no trailing newline), exactly
like `rich.console.status` did, so the renderer's first print flows from there.
"""
from __future__ import annotations

import io
import threading
from typing import Optional

from rich.console import Console
from rich.text import Text

from . import theme

# Same glyphs as Rich's "dots" spinner.
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ERASE_LINE = "\r\x1b[2K"   # carriage return + clear entire line


def _render_ansi(text: Text, console: Console) -> str:
    """Render a Rich Text to a one-line ANSI string (no newline, no wrap)."""
    buf = io.StringIO()
    tmp = Console(
        file=buf,
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=200,
        highlight=False,
    )
    tmp.print(text, end="")
    return buf.getvalue()


class ThinkingSpinner:
    """A single-line, scroll-safe animated status indicator.

    Reusable: stop() then start() again restarts the animation. start() is a
    no-op on non-terminals (e.g. piped output), so logs stay clean.
    """

    def __init__(self, console: Console,
                 label: str = "thinking…",
                 role: str = "伙伴",
                 interval: float = 0.1):
        self.console = console
        self.interval = interval
        self._role = role
        self._active = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Pre-render every frame to ANSI once (on the main thread) so the
        # animation loop only does cheap raw writes and never touches Rich.
        self._frames = [
            _render_ansi(self._frame_text(ch, role, label), console)
            for ch in _FRAMES
        ]

    def set_label(self, label: str) -> None:
        """换状态文案(如'校验中…')。帧数不变 → 运行中的动画线程会无缝切到新文案。"""
        self._frames = [
            _render_ansi(self._frame_text(ch, self._role, label), self.console)
            for ch in _FRAMES
        ]

    @staticmethod
    def _frame_text(spin_char: str, role: str, label: str) -> Text:
        t = Text()
        t.append(spin_char, style=theme.AGENT)
        t.append(" ")
        t.append(role, style=theme.AGENT)
        t.append("  ·  ", style=theme.DIM)
        t.append(label, style=theme.DIM)
        return t

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._active or not self.console.is_terminal:
            return
        self._active = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Wipe the spinner, leave cursor at col 0 of a clean line.
        self.console.file.write(_ERASE_LINE)
        self.console.file.flush()

    def _run(self) -> None:
        f = self.console.file
        i = 0
        n = len(self._frames)
        while not self._stop.is_set():
            f.write(_ERASE_LINE + self._frames[i % n])
            f.flush()
            i += 1
            self._stop.wait(self.interval)
