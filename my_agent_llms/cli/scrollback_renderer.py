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

    def close(self, *, tools_used: int = 0, elapsed_seconds: float = 0.0,
              tokens_in: int = 0, tokens_out: int = 0) -> None:
        if not self._opened:
            return
        if self._mode == "reasoning":
            self._close_reasoning()
        buf, self._text_buf = self._text_buf, ""
        if buf.strip():
            self._commit(self._render_md(buf, with_dot=not self._dot))
            self._dot = True
        self._set_active("", "text", self._dot)
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
