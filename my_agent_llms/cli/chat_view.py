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

from rich.console import Console, Group
from rich.live import Live
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


def _step_lines(text_or_ansi: str, dot_color: str, *,
                from_ansi: bool) -> Text:
    """Claude Code 风格:第一行带彩色 ⏺,续行缩进 2 格。一个 ⏺ = 一个 step。"""
    out = Text()
    for i, line in enumerate(text_or_ansi.split("\n")):
        if i == 0:
            out.append("⏺ ", style=dot_color)
        else:
            out.append("\n  ")
        if from_ansi:
            out.append_text(Text.from_ansi(line))
        else:
            out.append(line)
    return out


def _result_dot_color(text: str) -> str:
    """工具结果首字符决定 ⏺ 颜色:✅=绿,❌/拒绝=红,其它=DIM。"""
    if not text:
        return theme.DIM
    stripped = text.lstrip()
    if stripped.startswith("✅"):
        return theme.OK
    if stripped.startswith("❌") or "拒绝" in stripped:
        return theme.ERR
    return theme.DIM


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


class StreamingAgentRenderer:
    """流式 agent 输出渲染器,支持"流式期纯文本 → close 时 swap 成 markdown"。

    实现原理:
    用 `rich.live.Live` 把整个 body 区域托管,流式期间每个 chunk 触发一次 update,
    渲染的是"┃ bar + 原始文本"。close 时把 body 渲染换成"┃ bar + Markdown→ANSI",
    然后 stop Live —— `transient=False` 让最后一次 update 留在 scrollback,
    于是用户最终看到的是带格式的 markdown,但流式期间也实时看到了 plain text。

    内部用 segments 序列保留时序:
      ("text", str)              累积纯文本
      ("tool", name, preview)    工具调用提示

    一段连续的 on_text_chunk 调用合并进同一 ("text", ...) segment。
    tool_notice 插入会"切段",方便和文本交错时按时序渲染。
    """

    def __init__(self, console: Console,
                 role: str = "伙伴",
                 role_color: str = theme.AGENT):
        self.console = console
        self.role = role
        self.role_color = role_color
        self._opened = False
        self._segments: list[tuple] = []
        self._live: Optional[Live] = None

    # ── 内部:渲染当前 segments ────────────────────────────
    def _render_body(self, *, markdown: bool):
        """把 segments 渲染成 Group。markdown=True 时文本走 Markdown→ANSI 通道。"""
        if not self._segments:
            return Text("")

        renderables = []
        for seg in self._segments:
            kind = seg[0]
            if kind == "text":
                content = seg[1]
                if not content:
                    continue
                # 普通回答 = 白色 ⏺(theme.DEFAULT 走终端默认前景色)
                if markdown:
                    buf = io.StringIO()
                    sub = Console(
                        file=buf,
                        force_terminal=True,
                        color_system="truecolor",
                        width=max(40, self.console.width - 4),
                    )
                    sub.print(Markdown(content))
                    ansi = buf.getvalue().rstrip("\n")
                    renderables.append(_step_lines(ansi, theme.DEFAULT, from_ansi=True))
                else:
                    renderables.append(_step_lines(content, theme.DEFAULT, from_ansi=False))
            elif kind == "tool":
                # tool_notice (即将调用) = DIM ⏺,只是 "我要做 X" 的预告
                name = seg[1]
                preview = seg[2] if len(seg) > 2 else ""
                body = name if not preview else f"{name}({preview})"
                renderables.append(_step_lines(body, theme.DIM, from_ansi=False))
            elif kind == "tool_result":
                # tool_result = ⏺ 颜色看结果首字符:✅=绿 / ❌/拒绝=红 / 其它=DIM
                text_line = seg[1]
                color = _result_dot_color(text_line)
                renderables.append(_step_lines(text_line, color, from_ansi=False))

        if not renderables:
            return Text("")
        if len(renderables) == 1:
            return renderables[0]
        return Group(*renderables)

    def _ensure_started(self) -> None:
        if self._opened:
            return
        self._opened = True
        # header 落进 scrollback,Live 只接管它下面的 body 区域
        self.console.print()
        _header(self.console, self.role, self.role_color)
        self._live = Live(
            self._render_body(markdown=False),
            console=self.console,
            refresh_per_second=12,
            transient=False,  # 保留最后一次 update (close 时的 markdown 版本)
        )
        self._live.start()

    # ── 公共回调 ──────────────────────────────────────────
    def text_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_started()
        # 跟上一段连续的 text 合并,避免 segments 膨胀
        if self._segments and self._segments[-1][0] == "text":
            old = self._segments[-1][1]
            self._segments[-1] = ("text", old + chunk)
        else:
            self._segments.append(("text", chunk))
        if self._live is not None:
            self._live.update(self._render_body(markdown=False))

    def tool_notice(self, name: str, args_preview: str = "") -> None:
        self._ensure_started()
        self._segments.append(("tool", name, args_preview))
        if self._live is not None:
            self._live.update(self._render_body(markdown=False))

    def tool_result(self, text: str) -> None:
        """工具刚跑完时立刻把结果落到屏上,不用等模型再 invoke 一次。"""
        if not text:
            return
        self._ensure_started()
        # 只取首行,长结果(比如 ReadFile 的大段内容)截断
        first = text.splitlines()[0] if text else ""
        if len(first) > 200:
            first = first[:200] + " …"
        self._segments.append(("tool_result", first))
        if self._live is not None:
            self._live.update(self._render_body(markdown=False))

    def close(self, tools_used: int = 0, elapsed_seconds: float = 0.0) -> None:
        """收尾:把流式纯文本 swap 成 markdown 渲染版本,然后停 Live。"""
        if not self._opened:
            return
        if self._live is not None:
            # 关键一步:最后一次 update 用 markdown 版本,stop 后这版本留在屏上
            self._live.update(self._render_body(markdown=True))
            self._live.stop()
            self._live = None

        parts: list[str] = []
        if tools_used > 0:
            parts.append(f"{tools_used} tools")
        if elapsed_seconds > 0:
            parts.append(f"{elapsed_seconds:.1f}s")
        if parts:
            meta = "  ·  ".join(parts)
            t = Text("  ")
            t.append(meta, style=theme.DIM)
            self.console.print(t)

    @property
    def has_output(self) -> bool:
        return self._opened


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
