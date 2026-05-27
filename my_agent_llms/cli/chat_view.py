"""Role-strip rendering for the main chat area.

Every message is rendered as:
  header line — role label + meta (DIM)
  body lines  — each prefixed with ┃ in the role color

Markdown stays highlighted: we render Markdown to ANSI via a temp Console,
split by line, and rebuild via Text.from_ansi which preserves styles.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from . import theme


def _now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _header(console: Console, role: str, role_color: str,
            meta: Optional[str] = None) -> None:
    """Print the message header line: ' role · 20:34 · meta'."""
    t = Text()
    t.append(" ")
    t.append(role, style=role_color)
    t.append("  ·  ", style=theme.DIM)
    t.append(_now_hhmm(), style=theme.DIM)
    if meta:
        t.append("  ·  ", style=theme.DIM)
        t.append(meta, style=theme.DIM)
    console.print(t)


def _bar_lines(text_or_ansi: str, bar_color: str, *,
               from_ansi: bool) -> Text:
    """Wrap each line with '┃ ' prefix in bar_color, preserving styles."""
    out = Text()
    for i, line in enumerate(text_or_ansi.split("\n")):
        if i:
            out.append("\n")
        out.append("┃ ", style=bar_color)
        if from_ansi:
            out.append_text(Text.from_ansi(line))
        else:
            out.append(line)
    return out


def render_user(console: Console, text: str) -> None:
    """Echo the user's input with cyan bar + role label."""
    console.print()
    _header(console, "you", theme.YOU)
    console.print(_bar_lines(text, theme.YOU, from_ansi=False))


def render_agent(console: Console, reply: str, *,
                 tools_used: int = 0, elapsed_seconds: float = 0.0) -> None:
    """Render an AI reply: header + markdown body with magenta bar."""
    parts: list[str] = []
    if tools_used > 0:
        parts.append(f"{tools_used} tools")
    parts.append(f"{elapsed_seconds:.1f}s")
    meta = "  ·  ".join(parts)

    console.print()
    _header(console, "伙伴", theme.AGENT, meta=meta)

    # Render markdown to ANSI in a sub-console, then prefix each line.
    buf = io.StringIO()
    sub = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=max(40, console.width - 4),
    )
    sub.print(Markdown(reply))
    ansi = buf.getvalue().rstrip("\n")
    console.print(_bar_lines(ansi, theme.AGENT, from_ansi=True))


def render_agent_error(console: Console, message: str) -> None:
    """Agent failed mid-call — red bar, red header marker."""
    console.print()
    t = Text()
    t.append(" ")
    t.append("伙伴", style=theme.AGENT)
    t.append("  ·  ", style=theme.DIM)
    t.append("● error", style=theme.ERR)
    console.print(t)
    console.print(_bar_lines(message, theme.ERR, from_ansi=False))


def print_not_ready_hint(console: Console) -> None:
    """Shown when user tries to chat but agent is None (no API key)."""
    console.print()
    console.print(
        f"[{theme.ERR}]●[/] [bold]Agent not ready[/] "
        f"[{theme.DIM}]— missing API key[/]"
    )
    console.print()
    console.print(f"  [{theme.DIM}]Run these to set up:[/]")
    console.print(f"    [{theme.ACCENT}]/config[/]                "
                  f"[{theme.DIM}](interactive wizard)[/]")
    console.print(f"    [{theme.ACCENT}]/config key[/]            "
                  f"[{theme.DIM}](just the api key)[/]")
