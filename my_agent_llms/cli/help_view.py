"""Rich-Table renderers for /help, /config show, /memory + unified print_* helpers.

`render_*` functions take a `Console` argument so they can be tested via
`console.capture()`. The default app `console` from `cli.console` is passed
in by chat.py at call sites.
"""
from __future__ import annotations

from typing import Dict

from rich.console import Console
from rich.table import Table

from . import theme
from .completer import SLASH_COMMANDS


# ──────────────────────────────────────────────────────────
# Unified status prints
# ──────────────────────────────────────────────────────────

def print_error(console: Console, msg: str) -> None:
    console.print(
        f"[{theme.ERR}]●[/] [bold {theme.ERR}]error[/] "
        f"[{theme.ERR}]{msg}[/]"
    )


def print_warn(console: Console, msg: str) -> None:
    console.print(f"[{theme.WARN}]⚠[/] [{theme.WARN}]{msg}[/]")


def print_ok(console: Console, msg: str) -> None:
    console.print(f"[{theme.OK}]✓[/] {msg}")


# ──────────────────────────────────────────────────────────
# Section helpers
# ──────────────────────────────────────────────────────────

def _section_header(console: Console, label: str) -> None:
    rule = "─" * max(20, console.width - 4)
    console.print()
    console.print(f"  [bold {theme.ACCENT}]{label}[/]")
    console.print(f"  [{theme.RULE}]{rule}[/]")


def _section_footer(console: Console) -> None:
    rule = "─" * max(20, console.width - 4)
    console.print(f"  [{theme.RULE}]{rule}[/]")
    console.print()


# ──────────────────────────────────────────────────────────
# /help
# ──────────────────────────────────────────────────────────

def render_help(console: Console) -> None:
    """Two-column command table grouped by 'group' field of SLASH_COMMANDS."""
    _section_header(console, "COMMANDS")

    groups: Dict[str, list[tuple[str, str]]] = {}
    for name, desc, group in SLASH_COMMANDS:
        groups.setdefault(group, []).append((name, desc))

    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(width=18)
    table.add_column()

    first = True
    for group_label, rows in groups.items():
        if not first:
            table.add_row("", "")
        first = False
        table.add_row(f"[bold {theme.YOU}]{group_label}[/]", "")
        for name, desc in rows:
            table.add_row(f"  [{theme.ACCENT}]{name}[/]",
                          f"[{theme.DIM}]{desc}[/]")

    console.print(table)
    _section_footer(console)


# ──────────────────────────────────────────────────────────
# /config show
# ──────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    if not key:
        return f"[{theme.DIM}](not set)[/]"
    if len(key) <= 10:
        return key[:2] + "…"
    return key[:6] + "…" + key[-4:]


def render_config_show(
    console: Console,
    cfg: dict,
    agent_ready: bool,
    config_path: str,
) -> None:
    _section_header(console, "CONFIG")

    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(width=20)
    table.add_column()

    def row(label: str, value) -> None:
        table.add_row(f"  [{theme.DIM}]{label}[/]", str(value))

    table.add_row(f"[bold {theme.YOU}]LLM[/]", "")
    row("provider_key", f"[{theme.YOU}]{cfg.get('provider_key', '?')}[/]")
    row("provider",     cfg.get("provider", ""))
    row("model",        cfg.get("model") or f"[{theme.DIM}](not set)[/]")
    row("base_url",     cfg.get("base_url") or f"[{theme.DIM}]—[/]")
    row("api_key",      _mask_key(cfg.get("api_key", "")))

    table.add_row("", "")
    table.add_row(f"[bold {theme.YOU}]Memory[/]", "")
    mem = cfg.get("memory", {})
    row("cold_backend",      mem.get("cold_backend", ""))
    row("vector_backend",    mem.get("vector_backend", ""))
    row("conflict_strength", mem.get("conflict_strength", ""))
    row("tick_mode",         mem.get("tick_mode", ""))
    row("use_embedding",     "on" if mem.get("use_embedding") else f"[{theme.DIM}]off[/]")

    table.add_row("", "")
    table.add_row(f"[bold {theme.YOU}]Meta[/]", "")
    row("config path", f"[{theme.DIM}]{config_path}[/]")
    row("agent",
        f"[{theme.OK}]ready[/]" if agent_ready else f"[{theme.ERR}]not ready[/]")

    console.print(table)
    _section_footer(console)


# ──────────────────────────────────────────────────────────
# /memory
# ──────────────────────────────────────────────────────────

_MEM_LABELS = {
    "l1_items":  "L1 items",
    "l1_tokens": "L1 tokens",
    "l2_tokens": "L2 summary tokens",
    "l4_items":  "L4 cold items",
    "l5_items":  "L5 vector items",
}


def render_memory_stats(console: Console, stats: dict) -> None:
    _section_header(console, "MEMORY STATS")

    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(width=24)
    table.add_column()

    for key, value in stats.items():
        label = _MEM_LABELS.get(key, key)
        table.add_row(f"  [{theme.DIM}]{label}[/]",
                      f"[bright_white]{value}[/]")

    console.print(table)
    _section_footer(console)
