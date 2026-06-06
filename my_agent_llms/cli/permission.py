"""工具审批 UI —— 同步弹一个内联 Panel + 单键 y/n。

被 chat.py 包成 callback 传给 agent.run(on_permission_request=...);
agent 主循环在 execute_tool 前调它,阻塞等用户决定。
"""
from __future__ import annotations

import enum
import sys
from typing import Any, Dict

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.panel import Panel
from rich.syntax import Syntax

from my_agent_llms.cli import theme
from my_agent_llms.cli.console import console


class PermissionDecision(enum.Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"


class TerminalNotInteractiveError(RuntimeError):
    """stdin 不是 TTY 时抛出。chat.py 应捕获并降级为拒绝。"""


def _is_tty() -> bool:
    """单独抽出来方便测试 monkeypatch。"""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _looks_like_diff(text: str) -> bool:
    """unified diff 总有 --- / +++ / @@ 中至少一个;只识别这些头标记,
    避免把 '-v' / '+1 item' 之类的普通文本误当成 diff 高亮。"""
    stripped = text.lstrip()
    return stripped.startswith(("---", "+++", "@@"))


def _read_decision_key() -> "PermissionDecision":
    """阻塞读一个键。y/Enter=允许一次, a=本会话总是允许, n/Esc/Ctrl-C=拒绝。"""
    decision = {"v": PermissionDecision.DENY}
    kb = KeyBindings()

    @kb.add("y")
    @kb.add("Y")
    @kb.add("enter")
    def _yes(event):
        decision["v"] = PermissionDecision.ALLOW_ONCE
        event.app.exit()

    @kb.add("a")
    @kb.add("A")
    def _always(event):
        decision["v"] = PermissionDecision.ALLOW_ALWAYS
        event.app.exit()

    @kb.add("n")
    @kb.add("N")
    @kb.add("escape")
    @kb.add("c-c")
    def _no(event):
        decision["v"] = PermissionDecision.DENY
        event.app.exit()

    @kb.add("1")
    def _opt1(event):
        decision["v"] = PermissionDecision.ALLOW_ONCE
        event.app.exit()

    @kb.add("2")
    def _opt2(event):
        decision["v"] = PermissionDecision.ALLOW_ALWAYS
        event.app.exit()

    @kb.add("3")
    def _opt3(event):
        decision["v"] = PermissionDecision.DENY
        event.app.exit()

    app = Application(
        layout=Layout(Window(FormattedTextControl(""), height=0)),
        key_bindings=kb,
        full_screen=False,
        erase_when_done=True,
    )
    app.run()
    return decision["v"]


def prompt_permission(name: str, args: Dict[str, Any], preview: str) -> "PermissionDecision":
    """阻塞地弹一个审批框,返回 PermissionDecision(ALLOW_ONCE/ALLOW_ALWAYS/DENY)。

    调用方必须保证调用前已把任何 rich Live 区域 close,
    以免 Panel 跟 Live 区域重叠。

    Raises:
        TerminalNotInteractiveError: 当 stdin/stdout 不是 TTY 时。
            调用方应捕获并按安全默认处理(降级为 DENY = 拒绝)。
    """
    if not _is_tty():
        raise TerminalNotInteractiveError("stdin/stdout 不是交互终端")

    body = (
        Syntax(preview, "diff", theme="ansi_dark", background_color="default")
        if _looks_like_diff(preview)
        else preview
    )
    console.print(
        Panel(body, title=f"  {name}", title_align="left", border_style=theme.AGENT)
    )
    console.print("  Do you want to proceed?")
    console.print(
        f"  [{theme.OK}]1.[/] Yes   [{theme.OK}]2.[/] Yes, 本会话总是允许   "
        f"[{theme.ERR}]3.[/] No   [{theme.DIM}](1/Enter=Yes · 3/Esc=No)[/]"
    )
    decision = _read_decision_key()
    if decision is PermissionDecision.DENY:
        console.print(f"  [{theme.ERR}]✗ 已拒绝[/]")
    elif decision is PermissionDecision.ALLOW_ALWAYS:
        console.print(f"  [{theme.OK}]✓ 已同意(本会话总是允许)[/]")
    else:
        console.print(f"  [{theme.OK}]✓ 已同意[/]")
    return decision
