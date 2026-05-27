"""Per-turn status line printed before the ❯ prompt.

Layout:
    ─────────────────────────────────────────────
    minimax / Text-01  ·  turn 7  ·  L1 1234/4.0k tokens

Current L1 token count stays exact (so the user sees it grow precisely);
the max is abbreviated because it's a constant ceiling.
"""
from __future__ import annotations

from rich.console import Console

from . import theme


def _fmt_tokens(n: int) -> str:
    """Format token count: show exact for <1000, abbreviated for >=1000."""
    if n < 1000:
        return str(n)
    return f"{n/1000:.1f}k"


def render(
    console: Console,
    *,
    ready: bool,
    provider_key: str,
    model: str,
    turn: int,
    l1_tokens: int,
    l1_max_tokens: int,
    multiline: bool,
) -> None:
    rule = "─" * max(20, console.width)
    console.print(f"[{theme.RULE}]{rule}[/]")

    if not ready:
        msg = (f"[bold {theme.ERR}]not ready[/]  "
               f"[{theme.DIM}]·  run [/][{theme.ACCENT}]/config key[/]")
        if multiline:
            msg += f"[{theme.DIM}]  ·  multiline[/]"
        console.print(msg)
        return

    parts = [
        f"[{theme.YOU}]{provider_key}[/] [{theme.DIM}]/[/] "
        f"[{theme.YOU}]{model}[/]",
        f"[{theme.DIM}]turn {turn}[/]",
        f"[{theme.DIM}]L1 {l1_tokens}/{_fmt_tokens(l1_max_tokens)} tokens[/]",
    ]
    if multiline:
        parts.append(f"[{theme.DIM}]multiline[/]")
    console.print(f"[{theme.DIM}]  ·  [/]".join(parts))
