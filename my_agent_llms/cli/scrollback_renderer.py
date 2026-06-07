"""把 agent 流式回调映射成 Claude Code 风渲染块(prompt_toolkit 无关)。

两个注入 sink:
  commit(text_obj: rich.Text)            —— 把一个【完成块】交出去(由调用方 print 进 scrollback)
  set_active(src: str, mode: str, dot)   —— 把【正在生成的残块】源文交出去(由调用方渲到活跃区)

复用 cli/chat_view.py 的纯渲染函数 + markdown_render.render_markdown,不重写,也不碰 Rich Live。
"""
from __future__ import annotations

from typing import Callable

from rich.text import Text

from . import chat_view, theme
from .markdown_render import render_markdown


class ScrollbackRenderer:
    def __init__(self,
                 commit: Callable[[Text], None],
                 set_active: Callable[[str, str, bool], None],
                 width: Callable[[], int]):
        self._commit = commit
        self._set_active = set_active
        self._width = width
        self._text_buf = ""
        self._reason_buf = ""
        self._mode = "text"           # text | reasoning
        self._dot = False             # 本 text 段是否已发出 ⏺ 头
        self._opened = False
        # 待结果配对的工具 notice FIFO 队列(name, preview, read_only)。
        # 必须是队列:agent Phase A 先把同轮所有 tool_call 入队,Phase C 再按序
        # tool_result —— 单槽会被后来的覆盖,导致前面的名字/只读标志丢失。
        self._pending: list = []

    # ── 渲染 ──
    def _render_md(self, src: str, *, with_dot: bool) -> Text:
        body = render_markdown(src, self._width())
        return (chat_view._step_lines_from_text(body, theme.DEFAULT) if with_dot
                else chat_view._indent_only(body))

    # ── 回调 ──
    def text_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._opened = True
        if self._mode == "reasoning":
            self._close_reasoning()
        self._mode = "text"
        self._text_buf += chunk
        committable, remainder = chat_view._split_committable(self._text_buf)
        if committable.strip():
            self._commit(self._render_md(committable, with_dot=not self._dot))
            self._dot = True
            self._text_buf = remainder
        self._set_active(self._text_buf, "text", self._dot)

    def reasoning_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._opened = True
        self._mode = "reasoning"
        self._reason_buf += chunk
        self._set_active(self._reason_buf, "reasoning", False)

    def _flush_text(self) -> None:
        """工具/收尾前:把当前 text 残块 commit 掉,让工具块从干净行开始。"""
        if self._mode == "reasoning":
            self._close_reasoning()
        buf, self._text_buf = self._text_buf, ""
        if buf.strip():
            self._commit(self._render_md(buf, with_dot=not self._dot))
            self._dot = True
        self._set_active("", "text", self._dot)

    def tool_call(self, name: str, args_preview: str = "",
                  read_only: bool = False) -> None:
        self._opened = True
        self._flush_text()
        self._pending.append((name, args_preview, read_only))

    def tool_result(self, text: str, *, elapsed_sec=None,
                    max_lines: int = 4, max_line_chars: int = 300) -> None:
        """工具结果(Claude Code 风编排,不全量倒出):
        - write_todo → 'Update Todos' 内联清单
        - 有 per-type 摘要(Read/Grep…)→ 一行摘要(如 'Read 42 lines')
        - 否则 → 折叠到前 max_lines 行 + '… +N lines',每行截宽
        """
        if not text:
            return
        if self._pending:
            name, preview, read_only = self._pending.pop(0)   # FIFO 配对
        else:
            name, preview, read_only = "", "", False
        # ⏺ 颜色:出错→红;只读→中性;改动类成功→绿。
        stripped = text.lstrip()
        if stripped.startswith("❌") or "拒绝" in stripped:
            color = theme.ERR
        elif read_only:
            color = theme.DEFAULT
        else:
            color = theme.OK

        # write_todo → 内联 Update Todos 清单
        if name == "write_todo":
            self._commit(chat_view._tool_notice_lines("Update Todos", "", theme.DEFAULT))
            self._commit(chat_view._render_update_todos(text))
            return

        self._commit(chat_view._tool_notice_lines(name, preview, color))

        # per-type 一行摘要(命中才用)
        summarizer = chat_view._TOOL_RESULT_SUMMARY.get(name)
        if summarizer is not None:
            summary = summarizer(text)
            if summary:
                if elapsed_sec is not None:
                    summary = f"{summary}  ·  {chat_view._fmt_elapsed(elapsed_sec)}"
                self._commit(chat_view._continuation_lines(summary, color))
                return

        # 泛型:截宽 + 折叠行数 + elapsed 拼首行
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
            bl = body.split("\n")
            bl[0] = f"{bl[0]}  ·  {chat_view._fmt_elapsed(elapsed_sec)}"
            body = "\n".join(bl)
        self._commit(chat_view._continuation_lines(body, color, more=hidden))

    def close(self, *, tools_used: int = 0, elapsed_seconds: float = 0.0,
              tokens_in: int = 0, tokens_out: int = 0) -> None:
        if not self._opened:
            return
        self._flush_text()
        parts: list[str] = []
        if tools_used > 0:
            parts.append(f"{tools_used} tools")
        if elapsed_seconds > 0:
            parts.append(chat_view._fmt_elapsed(elapsed_seconds))
        if tokens_in > 0 or tokens_out > 0:
            parts.append(f"{tokens_in}↑ {tokens_out}↓")
        if parts:
            meta = Text("  ")
            meta.append("  ·  ".join(parts), style=theme.DIM)
            self._commit(meta)

    def _close_reasoning(self) -> None:
        # 收尾思考段总是切回 text 模式 —— 在此自洽复位,避免被 close()/tool_call 等
        # 独立调用后 _mode 残留 "reasoning"(否则下个 text_chunk 会误触二次 close)。
        buf, self._reason_buf = self._reason_buf, ""
        self._mode = "text"
        if buf.strip():
            self._commit(chat_view._render_thinking(buf))
        self._set_active("", "text", self._dot)
