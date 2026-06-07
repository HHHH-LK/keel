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


def _indent_only(text_obj: Text) -> Text:
    """每行补 2 空格续行缩进(无 ⏺ 头)—— 用于段内非首块的 progressive commit。"""
    out = Text()
    for i, line in enumerate(text_obj.split("\n", include_separator=False)):
        if i:
            out.append("\n")
        out.append("  ")
        out.append_text(line)
    return out


def _split_committable(buf: str) -> Tuple[str, str]:
    """把累积 buf 切成 (可提交的完整块, 进行中的残块)。

    提交点 = fence-depth 0 处的空行(markdown 块边界)。未闭合代码围栏整体留在
    残块,fence 内的空行不当提交点 —— 避免逐行 commit 破坏跨行 markdown
    (代码块/列表/表格)。无块边界 → 全部留残块(整块仍在 live 里刷)。"""
    lines = buf.split("\n")
    fence = False
    last = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if s == "" and not fence:
            last = i
    if last < 0:
        return "", buf
    committable = "\n".join(lines[:last])
    remainder = "\n".join(lines[last + 1:])
    return committable, remainder


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


def _render_thinking(text: str, fold: int = 3) -> Text:
    """思考块:✻ 起头,暗色;超过 fold 行折叠为 '首 fold 行 + … +N 行(思考)'。"""
    lines = text.rstrip("\n").split("\n")
    hidden = max(0, len(lines) - fold)
    shown = lines[:fold]
    out = Text()
    out.append("✻ ", style=theme.DIM)
    for i, ln in enumerate(shown):
        if i:
            out.append("\n  ")
        out.append(ln, style=theme.DIM)
    if hidden:
        out.append("\n  ")
        out.append(f"… +{hidden} 行(思考)", style=theme.DIM)
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
    """工具结果 ⏺ 颜色(Claude Code 风:完成即绿,出错才红)。

    出错信号:agent 把工具异常包成 '❌ 工具 … 执行异常'、拒绝包成含'拒绝'的串
    (见 function_call_agent),故 ❌/拒绝 → 红;其余(含普通成功输出,不要求 ✅
    前缀)→ 绿。空结果也按完成算绿。
    """
    if not text:
        return theme.OK
    stripped = text.lstrip()
    if stripped.startswith("❌") or "拒绝" in stripped:
        return theme.ERR
    return theme.OK


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


_TODO_MARK = {"[ ]": "☐", "[~]": "◐", "[x]": "☑"}


def _render_update_todos(result_text: str) -> Text:
    """把 TodoStore.render() 文本渲成 '  ⎿  ☐/◐/☑ 清单'(去掉 heading 行)。"""
    out = Text()
    out.append("  ⎿  ", style=theme.DIM)
    first = True
    for line in result_text.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("## "):
            continue
        mark = _TODO_MARK.get(line[:3], "☐")
        content = line[3:].strip()
        style = "grey50 strike" if mark == "☑" else ("default" if mark == "◐" else "grey50")
        if not first:
            out.append("\n     ")
        out.append(f"{mark} {content}", style=style)
        first = False
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


def _summarize_read(text: str) -> Optional[str]:
    n = len(text.rstrip("\n").split("\n")) if text.strip() else 0
    return f"Read {n} line{'s' if n != 1 else ''}" if n else None


def _summarize_grep(text: str) -> Optional[str]:
    n = len([l for l in text.split("\n") if l.strip()])
    return f"{n} matches" if n else None


# 工具名 → 首行摘要函数;返回 None 则回退泛型 body
_TOOL_RESULT_SUMMARY = {
    "ReadFile": _summarize_read,
    "GrepTool": _summarize_grep,
}


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


_NOTICE_PREFIX = {"info": "·", "warn": "⚠", "error": "⊘"}


def render_system_notice(console: Console, kind: str, text: str) -> None:
    """框架级系统提示(压缩/中断/context 告警):独立暗色样式,区别于 ⏺/┃。"""
    t = Text()
    t.append(f"  {_NOTICE_PREFIX.get(kind, '·')} ", style=theme.DIM)
    t.append(text, style=theme.DIM)
    console.print(t)


def render_user(console: Console, text: str) -> None:
    """Echo the user's input with cyan bar + role label."""
    console.print()
    _header(console, "you", theme.YOU)
    console.print(_bar_lines(text, theme.YOU, from_ansi=False))


def render_user_input(console: Console, text: str) -> None:
    """Claude Code 风:提交后把用户输入塌成一行 '❯ text' 留进 scrollback。
    配合 prompt.py 的 erase_when_done=True(圆角框提交即擦)——框消失,这行留底。
    多行输入续行缩进 2 格对齐。空输入不回显。"""
    if not text.strip():
        return
    out = Text()
    for i, line in enumerate(text.split("\n")):
        if i == 0:
            out.append("❯ ", style=theme.YOU)
        else:
            out.append("\n  ")
        out.append(line)
    console.print(out)


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
    """流式渲染器(Claude Code 风格)。

    实现要点:
    - tty 路径:text_chunk 把文本累积进 _text_buf,并用一个 transient=True /
      auto_refresh=False 的 Live 渲染活跃段的尾部(_active_frame 截断防超高)。
      段落结束时(_close_text)stop 掉 Live(尾区帧消失),再把全文经由
      console.print(_framed_render) 落进 scrollback(progressive commit),
      保证早期行不因 Live 截断而丢失。
    - 非 tty 路径:退回 raw 行为——text_chunk 直接 console.file.write(chunk) +
      flush,字符级推送;第一个 chunk 之前打 '⏺ ' 起头,换行后补 '  ' 缩进。
    - tool_notice **延迟** 到 tool_result / tool_diff_result 来时一起打。
      好处:⏺ 的颜色一开始就是终态(绿/红/DIM),不再需要 ANSI 回去重涂。
      代价:工具执行期间 ⏺ 不可见(用户依赖外层 spinner 作为反馈)。
    - close() 只补 meta 行(elapsed/token/tools)。
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
        self._dot_emitted = False         # 当前 text 段是否已发出 ⏺ 头(progressive commit 用)
        self._reason_open = False
        self._reason_buf = ""
        self._live: Optional[Live] = None
        self._use_live = bool(getattr(self.console, "is_terminal", False))

    # ── 内部 helpers ─────────────────────────────────────────
    def _ensure_started(self) -> None:
        """首次输出前打一个空行,跟前面 turn 隔开。"""
        if self._opened:
            return
        self._opened = True
        self.console.print()

    def _render_body(self, text: str, *, with_dot: bool) -> Text:
        """累积文本渲成 markdown,再包前缀:首块带 ⏺,段内后续块只缩进续行。"""
        width = max(20, self.console.width - 2)
        body = render_markdown(text, width)
        if with_dot:
            return _step_lines_from_text(body, theme.DEFAULT)
        return _indent_only(body)

    def _framed_render(self, text: str) -> Text:
        """累积文本渲成 markdown,再包 ⏺ + 续行缩进的 strip 框。"""
        return self._render_body(text, with_dot=True)

    def _active_height(self) -> int:
        return getattr(getattr(self.console, "size", None), "height", 24) or 24

    def _active_frame(self, text: str) -> Text:
        """活跃段的 live 帧:整段渲染后只取尾部若干行,防 Live 超终端高度。
        已完成块由 text_chunk 逐块落 scrollback,这里只渲进行中的残块。
        段内已发过 ⏺ 时残块用缩进续行(不再补第二个 ⏺)。"""
        return _tail_cap(self._render_body(text, with_dot=not self._dot_emitted),
                         self._active_height())

    def _close_text(self) -> None:
        """text/reasoning segment 切段(progressive commit)。
        tty:stop 掉 transient Live(清尾区帧)→ 把全文 print 进 scrollback;
        非 tty:确保当前一行已结束,后续非 text 输出从行首开始。
        reasoning 段:折叠暗色块 commit;text 段:_framed_render commit。"""
        if not self._text_open and not self._reason_open:
            return
        if self._live is not None:
            # transient Live:stop 即清掉尾区帧;再 print 全文进 scrollback(progressive commit)。
            try:
                self._live.stop()
            except Exception:
                pass
            finally:
                self._live = None
            if self._reason_open:
                buf, self._reason_buf = self._reason_buf, ""
                self._reason_open = False
                if buf.strip():
                    self.console.print(_render_thinking(buf))    # 折叠暗色块
            else:
                buf, self._text_buf = self._text_buf, ""
                if buf.strip():     # 提交进行中残块(段内首块带 ⏺,否则缩进续行)
                    self.console.print(
                        self._render_body(buf, with_dot=not self._dot_emitted))
                    self._dot_emitted = True
        elif not self._pending_indent:
            self.console.file.write("\n")
            self.console.file.flush()
        self._text_open = False
        self._reason_open = False
        self._pending_indent = False

    def reasoning_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_started()
        if self._text_open:              # 只在"从正文切到思考"时收尾正文段(对称于 text_chunk)
            self._close_text()
        if not self._use_live:
            self.console.print(chunk, style=theme.DIM, end="")
            self.console.file.flush()
            self._reason_open = True
            return
        if not self._reason_open:
            self._reason_open = True
            self._reason_buf = ""
            self._live = Live(console=self.console, transient=True, auto_refresh=False)
            self._live.start()
        self._reason_buf += chunk
        # 流式期间尾区渲染:仅尾部防超高;fold 与 commit 一致防止 StringIO 测试泄漏
        self._live.update(_tail_cap(_render_thinking(self._reason_buf),
                                    self._active_height()), refresh=True)

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
        # 与 reasoning 段互斥:从思考切到正文前先收尾思考段(commit ✻ 块 + stop 旧 Live)。
        # 否则旧 reasoning Live 不被停 → 二次 Live 冲突 + 正文 buf 在 _close_text 被跳过丢失。
        if self._reason_open:
            self._close_text()
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
        # tty:累积 + Live 重绘(只显尾部)。auto_refresh=False:不开后台线程,
        # 每 chunk 手动 refresh —— 与 thinking.py 一致,杜绝后台重绘 desync。
        if not self._text_open:
            self._text_open = True
            self._text_buf = ""
            self._dot_emitted = False
            self._live = Live(console=self.console, transient=True,
                              auto_refresh=False)
            self._live.start()
        self._text_buf += chunk
        # progressive commit:写完的块(fence 外空行为界)经 console.print 落进
        # scrollback(Rich Live 会把它打在 live 区上方,永久),只把进行中的残块
        # 留在 live 帧里。这样完成内容向上滚动定格,不再"原地覆盖"或"末尾一次性出现"。
        committable, remainder = _split_committable(self._text_buf)
        if committable.strip():
            self.console.print(
                self._render_body(committable, with_dot=not self._dot_emitted))
            self._dot_emitted = True
            self._text_buf = remainder
        self._live.update(self._active_frame(self._text_buf), refresh=True)

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
        # write_todo → Claude Code 'Update Todos' 内联清单(而非泛型 ⎿ 文本)
        if self._pending and self._pending[0][0] == "write_todo":
            self._pending.pop(0)
            self._ensure_started()
            self._close_text()
            self.console.print(_tool_notice_lines("Update Todos", "", theme.DEFAULT))
            self.console.print(_render_update_todos(text))
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
        # per-type 首行摘要(命中才用;否则泛型 body 不变)
        head_name = self._pending[0][0] if self._pending else ""
        summarizer = _TOOL_RESULT_SUMMARY.get(head_name)
        if summarizer is not None:
            summary = summarizer(text)
            if summary:
                if elapsed_sec is not None:
                    summary = f"{summary}  ·  {_fmt_elapsed(elapsed_sec)}"
                scolor = _result_dot_color(text)
                self._flush_tool_notice(result_color=scolor)
                self.console.print(_continuation_lines(summary, scolor))
                return
        # 2. 用结果推断的颜色打 pending tool_notice,再接 ⎿ result(+折叠标记)
        color = _result_dot_color(body)
        self._flush_tool_notice(result_color=color)
        self.console.print(_continuation_lines(body, color, more=hidden))

    def system_notice(self, kind: str, text: str) -> None:
        """系统提示:先收掉当前 text 段,再打独立暗色提示行。"""
        if not text:
            return
        self._ensure_started()
        self._close_text()
        render_system_notice(self.console, kind, text)

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
