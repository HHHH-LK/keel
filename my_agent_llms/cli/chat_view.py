"""Role-strip rendering for the main chat area.

Every message is rendered as:
  header line — role label + meta (DIM)
  body lines  — each prefixed with ┃ in the role color

Markdown stays highlighted: we render Markdown to ANSI via a temp Console,
split by line, and rebuild via Text.from_ansi which preserves styles.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.markup import escape as _rich_escape
from rich.text import Text

from . import theme
from .markdown_render import render_markdown, render_inline


def _render_inline_markdown(text: str) -> Text:
    """text → Rich Text,只渲 inline markdown。委托 markdown_render.render_inline。"""
    return render_inline(text)


def _step_lines_from_text(text_obj: Text, dot_color: str) -> Text:
    """跟 _step_lines 一样,但接受 Rich Text 而非 str —— 用于已带 inline style 的内容。"""
    out = Text()
    lines = text_obj.split("\n", include_separator=False)
    for i, line in enumerate(lines):
        if i == 0:
            out.append("⏺ ", style=dot_color)
        else:
            out.append("\n  ")
        out.append_text(line)
    return out


def _tail_cap(text_obj: Text, height: int, reserve: int = 6) -> Text:
    """把 Text 截到尾部 max(3, height-reserve) 行 —— 给底部 live 尾区用,
    保证活跃段渲染高度不超终端(避免 Live overflow 花屏)。
    reserve 预留给 spinner / 边距。短于上限则原样返回。"""
    cap = max(3, height - reserve)
    lines = text_obj.split("\n", include_separator=False)
    if len(lines) <= cap:
        return text_obj
    out = Text()
    for i, line in enumerate(lines[-cap:]):
        if i:
            out.append("\n")
        out.append_text(line)
    return out


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


def _continuation_lines(text: str, color: str, *, more: int = 0) -> Text:
    """上一步 ⏺ 的续行 —— 用 '  ⎿  ' 连接符,跟 Claude Code 一致。
    一次工具调用 = 一个 ⏺ tool_notice + 缩进的 ⎿ tool_result。
    more>0 时在末尾补一行 DIM '… +N lines',表示结果被折叠。"""
    out = Text()
    for i, line in enumerate(text.split("\n")):
        if i == 0:
            out.append("  ⎿  ", style=color)
        else:
            out.append("\n     ")
        out.append(line, style=color if i == 0 else "")
    if more:
        out.append("\n     ")
        out.append(f"… +{more} lines", style=theme.DIM)
    return out


def _tool_notice_lines(name: str, preview: str, dot_color: str) -> Text:
    """⏺ name(preview) —— preview 多行时续行对齐到 '(' 之后,跟 Claude Code 一致。

    对齐列宽 = len('⏺ ') + len(name) + len('(') = 2 + len(name) + 1。
    (与 _step_lines 一样假设 ⏺ 占 1 列 + 1 空格。)
    """
    out = Text()
    out.append("⏺ ", style=dot_color)
    if not preview:
        out.append(name)
        return out
    indent = " " * (2 + len(name) + 1)
    body = f"{name}({preview})"
    for i, line in enumerate(body.split("\n")):
        if i:
            out.append("\n" + indent)
        out.append(line)
    return out


# unified diff hunk 头:@@ -OLD_START[,OLD_COUNT] +NEW_START[,NEW_COUNT] @@
_DIFF_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _parse_diff_for_display(diff_text: str) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    """把 unified diff 解析成 (added, removed, lines)。

    lines: 每个元素 (line_num, kind, content)
      kind: '+' 新增 / '-' 删除 / ' ' 上下文
      line_num: 新文件里的行号(- 行用旧文件行号);空字符串 = 无行号
    """
    added = 0
    removed = 0
    out: List[Tuple[str, str, str]] = []
    new_line = 0
    old_line = 0
    for raw in diff_text.split("\n"):
        if not raw:
            continue
        if raw.startswith("--- ") or raw.startswith("+++ "):
            continue  # 跳过 file headers
        m = _DIFF_HUNK_RE.match(raw)
        if m:
            old_line = int(m.group(1))
            new_line = int(m.group(2))
            continue
        prefix = raw[0]
        content = raw[1:]
        if prefix == "+":
            added += 1
            out.append((str(new_line), "+", content))
            new_line += 1
        elif prefix == "-":
            removed += 1
            out.append((str(old_line), "-", content))
            old_line += 1
        elif prefix == " ":
            out.append((str(new_line), " ", content))
            new_line += 1
            old_line += 1
        # 其它(\, no newline at end of file 等)忽略
    return added, removed, out


def _diff_summary(added: int, removed: int) -> str:
    """生成 "Added N lines" / "Removed N lines" / "Added N, removed M lines"。"""
    if added and removed:
        return f"Added {added} lines, removed {removed} lines"
    if added:
        return f"Added {added} lines"
    if removed:
        return f"Removed {removed} lines"
    return "No changes"


def _fmt_elapsed(seconds: float) -> str:
    """美化耗时:<1s → 'XXXms';<60s → 'X.Xs';>=60s → 'XmYs'。"""
    if seconds < 1.0:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m{s:.0f}s"


def _render_tool_diff(summary: str, lines: List[Tuple[str, str, str]],
                      truncated: int = 0) -> Text:
    """渲染 '⎿  summary' + 缩进的行号化 diff,跟 Claude Code Update 一致。

    格式:
      ⎿  ✅ 已修改 foo.py  ·  Added 13 lines
          82      return theme.DIM
          83
          85 +def _continuation_lines(text, color):
          ...
    """
    out = Text()
    # 第一行:⎿  summary (绿色,因为这是个"成功完成"信号)
    out.append("  ⎿  ", style=theme.OK)
    out.append(summary, style=theme.OK)
    # 行号宽度对齐
    width = max((len(ln) for ln, _, _ in lines), default=4)
    for line_num, kind, content in lines:
        out.append("\n     ")
        # 行号 (右对齐, DIM)
        out.append(f"{line_num:>{width}} ", style=theme.DIM)
        # diff 前缀 + 内容,颜色看 kind
        if kind == "+":
            out.append("+", style=theme.OK)
            out.append(content, style=theme.OK)
        elif kind == "-":
            out.append("-", style=theme.ERR)
            out.append(content, style=theme.ERR)
        else:
            out.append(" ")
            out.append(content, style=theme.DIM)
    if truncated:
        out.append("\n     ")
        out.append(f"… ({truncated} more lines)", style=theme.DIM)
    return out


def render_user(console: Console, text: str) -> None:
    """Echo the user's input with cyan bar + role label."""
    console.print()
    _header(console, "you", theme.YOU)
    console.print(_bar_lines(text, theme.YOU, from_ansi=False))


def render_agent(console: Console, reply: str, *,
                 tools_used: int = 0, elapsed_seconds: float = 0.0) -> None:
    """Render an AI reply: ⏺ + 全量 markdown body (Claude Code 风格)。
    用于非流式/流式无 chunk 兜底,与流式主路径(_framed_render)一致走 render_markdown。"""
    console.print()
    body = render_markdown(reply, max(20, console.width - 2))
    console.print(_step_lines_from_text(body, theme.DEFAULT))


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
    """无 Live 真·流式渲染器(Claude Code 风格)。

    实现要点:
    - text_chunk 直接 console.file.write(chunk) + flush,字符级推送到 stdout。
      第一个 chunk 之前打 '⏺ ' 起头,换行后下一行非 '\\n' 字符前补 '  ' 缩进,
      保证一段连续 text = 一个 ⏺ step,跟 Claude Code 一致。
    - tool_notice **延迟** 到 tool_result / tool_diff_result 来时一起打。
      好处:⏺ 的颜色一开始就是终态(绿/红/DIM),不再需要 ANSI 回去重涂,
      也省掉了 Live + segments + _recolor_last_tool 一整套机器。
      代价:工具执行期间 ⏺ 不可见(用户依赖外层 spinner 的转动作为反馈)。
    - close() 只补 meta 行(elapsed/token/tools),不再有 swap、不再 stop Live。
    - 流式期间放弃 inline markdown(**bold** / *italic* / `code`)的渲染:
      跨 chunk 状态机过于脆弱,Claude Code 自己流式也是源码原样上屏。

    pop_last_tool_notice 保留是为了兼容 chat.py 的 permission 流程 ——
    审批弹起之前,把还没打印的 pending notice 弹出,审批完后由调用方在新
    renderer 上重新登记,等结果回来再统一染色。
    """

    def __init__(self, console: Console,
                 role: str = "伙伴",
                 role_color: str = theme.AGENT):
        self.console = console
        self.role = role
        self.role_color = role_color
        self._opened = False
        self._text_open = False          # 当前是否在一段未关闭的 text segment 里
        self._pending_indent = False     # 刚写完 \n,下一个非 \n 字符前要补 '  '
        # 等 result 来时一起打的 tool_notice **队列** (FIFO): [(name, preview, color_hint), ...]
        # 为什么要队列:执行器 Phase A 把同一轮所有 on_tool_call 先触发完(全部入队),
        # Phase C 才按原顺序逐个 on_tool_result —— 单个 slot 会被后来的 notice 覆盖,
        # 导致前面的调用丢失。队列让每个结果按 FIFO 配对到自己的 ⏺。
        # color_hint=None → 让 result 推断 (✅绿/❌红/其它DIM);
        # 显式传入 (e.g. theme.ERR for 拒绝) → 强制用这个色
        self._pending: List[tuple[str, str, Optional[str]]] = []
        self._text_buf = ""
        self._live: Optional[Live] = None
        self._use_live = bool(getattr(self.console, "is_terminal", False))

    # ── 内部 helpers ─────────────────────────────────────────
    def _ensure_started(self) -> None:
        """首次输出前打一个空行,跟前面 turn 隔开。"""
        if self._opened:
            return
        self._opened = True
        self.console.print()

    def _framed_render(self, text: str) -> Text:
        """累积文本渲成 markdown,再包 ⏺ + 续行缩进的 strip 框。"""
        width = max(20, self.console.width - 2)
        body = render_markdown(text, width)
        return _step_lines_from_text(body, theme.DEFAULT)

    def _close_text(self) -> None:
        """text segment 切段:确保当前一行已结束,后续非 text 输出从行首开始。"""
        if not self._text_open:
            return
        if self._live is not None:
            # 兜底:update/refresh 万一抛(rich 内部),也必须 stop,否则后台刷新线程
            # 泄漏 → 终端花屏。finally 保证 stop + 清状态。
            try:
                self._live.update(self._framed_render(self._text_buf))
                self._live.refresh()
            except Exception:
                pass
            finally:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None
                self._text_buf = ""
        elif not self._pending_indent:
            self.console.file.write("\n")
            self.console.file.flush()
        self._text_open = False
        self._pending_indent = False

    def _flush_tool_notice(self, result_color: Optional[str] = None) -> None:
        """把**队首** pending notice 打到屏上(FIFO,跟 Phase C 结果上报顺序一致)。

        颜色优先级:tool_notice(color=...) 显式给的 hint > result 推断色 > DEFAULT。
        队列空时 no-op,允许 tool_result 在无 tool_notice 的情况下调用。
        """
        if not self._pending:
            return
        name, preview, hint_color = self._pending.pop(0)
        color = hint_color or result_color or theme.DEFAULT
        self.console.print(_tool_notice_lines(name, preview, color))

    # ── 公共回调 ──────────────────────────────────────────
    def text_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_started()
        if not self._use_live:
            # 非 tty:保持原 raw 行为(⏺ 起头 + 换行补 2 空格)
            if not self._text_open:
                self.console.print("⏺ ", style=theme.DEFAULT, end="")
                self._text_open = True
                self._pending_indent = False
            buf: list[str] = []
            for ch in chunk:
                if ch == "\n":
                    buf.append("\n"); self._pending_indent = True
                else:
                    if self._pending_indent:
                        buf.append("  "); self._pending_indent = False
                    buf.append(ch)
            self.console.file.write("".join(buf)); self.console.file.flush()
            return
        # tty:累积 + Live 重绘 markdown
        if not self._text_open:
            self._text_open = True
            self._text_buf = ""
            self._live = Live(console=self.console, refresh_per_second=12,
                              transient=False, auto_refresh=True)
            self._live.start()
        self._text_buf += chunk
        self._live.update(self._framed_render(self._text_buf))

    def tool_notice(self, name: str, args_preview: str = "",
                    color: Optional[str] = None) -> None:
        """登记一个即将调用的工具,延迟到 tool_result 来时一起打。

        color 可选:None = 让结果推断;显式给 (e.g. theme.ERR for 拒绝) = 强制。
        """
        self._ensure_started()
        self._close_text()
        self._pending.append((name, args_preview, color))

    def pop_last_tool_notice(self) -> Optional[tuple[str, str]]:
        """弹出 pending 的 tool_notice,返回 (name, preview)。

        用于 permission gate:审批前把还没打印的 pending notice 弹出,审批后
        由调用方在新 renderer 上重新 tool_notice() 登记,等结果回来再统一染色。
        弹**队尾**(刚由 _on_tool 登记的当前审批工具)。
        """
        if not self._pending:
            return None
        name, preview, _color = self._pending.pop()
        return name, preview

    def tool_diff_result(self, summary_status: str, diff_text: str, *,
                         elapsed_sec: Optional[float] = None,
                         max_lines: int = 30) -> None:
        """工具跑完且手上有 unified diff —— 跟 Claude Code 的 Update 一样。"""
        added, removed, lines = _parse_diff_for_display(diff_text)
        if not lines:
            self.tool_result(summary_status, elapsed_sec=elapsed_sec)
            return
        truncated = 0
        if len(lines) > max_lines:
            truncated = len(lines) - max_lines
            lines = lines[:max_lines]
        summary = f"{summary_status}  ·  {_diff_summary(added, removed)}"
        if elapsed_sec is not None:
            summary = f"{summary}  ·  {_fmt_elapsed(elapsed_sec)}"
        self._ensure_started()
        self._close_text()
        # diff 路径意味着写入成功 —— ⏺ 染绿,然后接 '⎿ summary + 行号化 diff'
        self._flush_tool_notice(result_color=theme.OK)
        self.console.print(_render_tool_diff(summary, lines, truncated))

    def tool_result(self, text: str, *,
                    elapsed_sec: Optional[float] = None,
                    max_lines: int = 4, max_line_chars: int = 300) -> None:
        """工具跑完(普通文本结果):打 '⏺ tool(args)\\n  ⎿ result'。

        默认折叠到前 4 行,超出补 '… +N lines'(Claude Code 风格)。模型始终
        拿到完整结果,所以"细节"没丢,只是 UI 不堆。
        """
        if not text:
            return
        self._ensure_started()
        self._close_text()
        # 1. 处理结果文本:每行截宽 + 折叠行数 + elapsed 拼到首行末
        lines = text.rstrip("\n").splitlines() or [""]
        clipped = [
            (ln if len(ln) <= max_line_chars else ln[:max_line_chars - 1] + "…")
            for ln in lines
        ]
        hidden = 0
        if len(clipped) > max_lines:
            hidden = len(clipped) - max_lines
            clipped = clipped[:max_lines]
        body = "\n".join(clipped)
        if elapsed_sec is not None:
            body_lines = body.split("\n")
            body_lines[0] = f"{body_lines[0]}  ·  {_fmt_elapsed(elapsed_sec)}"
            body = "\n".join(body_lines)
        # 2. 用结果推断的颜色打 pending tool_notice,再接 ⎿ result(+折叠标记)
        color = _result_dot_color(body)
        self._flush_tool_notice(result_color=color)
        self.console.print(_continuation_lines(body, color, more=hidden))

    def close(self, tools_used: int = 0, elapsed_seconds: float = 0.0,
              tokens_in: int = 0, tokens_out: int = 0) -> None:
        """收尾:关掉未结束的 text 段、flush 残留 pending notice、补 meta 行。"""
        if not self._opened:
            return
        self._close_text()
        # 兜底:notice 入队后没等到对应 result 就 close —— 把残留的全部打成头(默认色),
        # 不吞掉,免得调用被静默丢失(如某些工具 report=False 不上报结果)。
        while self._pending:
            self._flush_tool_notice()

        parts: list[str] = []
        if tools_used > 0:
            parts.append(f"{tools_used} tools")
        if elapsed_seconds > 0:
            parts.append(_fmt_elapsed(elapsed_seconds))
        if tokens_in > 0 or tokens_out > 0:
            parts.append(f"{tokens_in}↑ {tokens_out}↓")
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
