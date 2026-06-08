"""Inline scrolling banner — no outer frame.

Logo:   █▒  keel        (LOGO_L + LOGO_R + TITLE)
Tag:    A long-term AI partner with memory
Bullets: status / model / workspace
Footer:  /help · / for menu
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from . import theme


def _tilde(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def render(
    console: Console,
    *,
    ready: bool,
    provider_key: str,
    model: str,
    backend_label: str,
    tool_count: int,
    workspace: Optional[Path],
) -> None:
    console.print()
    # Logo + title
    console.print(
        f"  [{theme.LOGO_L}]█[/][{theme.LOGO_R}]▒[/]  "
        f"[{theme.TITLE}]keel[/]"
    )
    console.print()
    console.print(f"  [{theme.DIM}]A long-term AI partner with memory[/]")
    console.print()

    # Bullets
    dot_ok  = f"[{theme.OK}]●[/]"
    dot_err = f"[{theme.ERR}]●[/]"
    dot_dim = f"[{theme.DIM}]●[/]"

    if ready:
        # Bullet 1: provider / model
        console.print(
            f"  {dot_dim}  [{theme.YOU}]{provider_key}[/]  "
            f"[{theme.DIM}]/[/]  [{theme.YOU}]{model}[/]"
        )
        # Bullet 2: ● ready  ·  L4 cold: sqlite  ·  N tools loaded
        extras = []
        if backend_label:
            extras.append(backend_label)
        if tool_count > 0:
            extras.append(f"{tool_count} tools loaded")
        suffix = ""
        if extras:
            joined = "  ·  ".join(extras)
            suffix = f"  [{theme.DIM}]·  {joined}[/]"
        console.print(f"  {dot_ok}  [{theme.OK}]ready[/]{suffix}")
        # Bullet 3: workspace path
        if workspace is not None:
            console.print(
                f"  {dot_dim}  [{theme.DIM}]workspace: {_tilde(workspace)}[/]"
            )
    else:
        console.print(
            f"  {dot_err}  [bold {theme.ERR}]not ready[/]  "
            f"[{theme.DIM}]—  run [/][{theme.ACCENT}]/config key[/]"
        )

    console.print()
    console.print(
        f"  [{theme.DIM}]Type [/][{theme.ACCENT}]/help[/]"
        f"[{theme.DIM}] for commands  ·  [/][{theme.ACCENT}]/[/]"
        f"[{theme.DIM}] for menu[/]"
    )
    console.print()
