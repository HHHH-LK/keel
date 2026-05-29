"""工具审批 UI —— 同步弹一个内联 Panel + 单键 y/n。

被 chat.py 包成 callback 传给 agent.run(on_permission_request=...);
agent 主循环在 execute_tool 前调它,阻塞等用户决定。
"""
from __future__ import annotations

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


class TerminalNotInteractiveError(RuntimeError):
    """stdin 不是 TTY 时抛出。chat.py 应捕获并降级为拒绝。"""


def _is_tty() -> bool:
    """单独抽出来方便测试 monkeypatch。"""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _looks_like_diff(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("---", "+++", "@@", "+", "-"))


def _read_decision_key() -> bool:
    """阻塞读一个键,返回 True=允许 / False=拒绝。抽离方便测试 monkeypatch。"""
    decision: Dict[str, Any] = {"v": None}
    kb = KeyBindings()

    @kb.add("y")
    @kb.add("Y")
    @kb.add("enter")
    def _yes(event):
        decision["v"] = True
        event.app.exit()

    @kb.add("n")
    @kb.add("N")
    @kb.add("escape")
    @kb.add("c-c")
    def _no(event):
        decision["v"] = False
        event.app.exit()

    app = Application(
        layout=Layout(Window(FormattedTextControl(""), height=0)),
        key_bindings=kb,
        full_screen=False,
        erase_when_done=True,
    )
    app.run()
    return bool(decision["v"]) if decision["v"] is not None else False


def prompt_permission(name: str, args: Dict[str, Any], preview: str) -> bool:
    """阻塞地弹一个审批框,返回 True=允许 / False=拒绝。

    调用方必须保证调用前已把任何 rich Live 区域 close,
    以免 Panel 跟 Live 区域重叠。

    Raises:
        TerminalNotInteractiveError: 当 stdin/stdout 不是 TTY 时。
            调用方应捕获并按安全默认处理(返回 False = 拒绝)。
    """
    if not _is_tty():
        raise TerminalNotInteractiveError("stdin/stdout 不是交互终端")

    body = (
        Syntax(preview, "diff", theme="ansi_dark", background_color="default")
        if _looks_like_diff(preview)
        else preview
    )
    console.print(
        Panel(body, title=f"审批工具调用  {name}", border_style=theme.AGENT)
    )
    console.print(
        f"  [{theme.DIM}][y] 允许   [n] 拒绝   [Enter=y · Esc=n][/]"
    )
    return _read_decision_key()
