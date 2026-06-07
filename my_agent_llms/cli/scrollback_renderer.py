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
        # 连续同类只读工具的合并缓冲(Phase C 同名结果攒成一组 → 'Read N files')。
        # items: [(preview, result, elapsed)];仅当后面还有同名排队时才攒,否则即时落地。
        self._group: list = []
        self._group_name = None
        self._group_ro = False
        # 块间留白:每个新"步"(⏺/✻ 开头的块)落地前补一行空行,除了本轮首块 ——
        # 跟 Claude Code 一致,工具/思考/正文各块之间不挤在一起。
        self._committed_any = False

    # ── 渲染 ──
    def _commit_step(self, text_obj: Text) -> None:
        """落地一个新步块:非首块时先补一行空行做分隔。"""
        if self._committed_any:
            self._commit(Text(""))
        self._commit(text_obj)
        self._committed_any = True

    def _render_md(self, src: str, *, with_dot: bool) -> Text:
        body = render_markdown(src, self._width())
        return (chat_view._step_lines_from_text(body, theme.DEFAULT) if with_dot
                else chat_view._indent_only(body))

    # ── 回调 ──
    def text_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._opened = True
        self._flush_group()
        if self._mode == "reasoning":
            self._close_reasoning()
        self._mode = "text"
        self._text_buf += chunk
        committable, remainder = chat_view._split_committable(self._text_buf)
        if committable.strip():
            block = self._render_md(committable, with_dot=not self._dot)
            (self._commit_step if not self._dot else self._commit)(block)
            self._dot = True
            self._text_buf = remainder
        self._set_active(self._text_buf, "text", self._dot)

    def reasoning_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._opened = True
        self._flush_group()
        self._mode = "reasoning"
        self._reason_buf += chunk
        self._set_active(self._reason_buf, "reasoning", False)

    def _flush_text(self) -> None:
        """工具/收尾前:把当前 text 残块 commit 掉,让工具块从干净行开始。"""
        if self._mode == "reasoning":
            self._close_reasoning()
        self._flush_group()
        buf, self._text_buf = self._text_buf, ""
        if buf.strip():
            block = self._render_md(buf, with_dot=not self._dot)
            (self._commit_step if not self._dot else self._commit)(block)
            self._dot = True
        self._set_active("", "text", self._dot)

    def tool_call(self, name: str, args_preview: str = "",
                  read_only: bool = False) -> None:
        self._opened = True
        self._flush_text()
        self._pending.append((name, args_preview, read_only))

    def tool_result(self, text: str, *, name: str = None, read_only=None,
                    elapsed_sec=None,
                    max_lines: int = 4, max_line_chars: int = 300) -> None:
        """工具结果(Claude Code 风编排,不全量倒出)。

        name/read_only 显式传入时优先(稳健配对,不再只靠 FIFO 倒推 → 修空 ⏺);
        连续同名只读工具(同一回合并行批)合并成 'Read N files'。
        """
        if not text:
            return
        if self._pending:
            fifo_name, preview, fifo_ro = self._pending.pop(0)   # 取 preview(只在 call 时知道)
        else:
            fifo_name, preview, fifo_ro = "", "", False
        rname = name or fifo_name
        ro = fifo_ro if read_only is None else read_only

        stripped = text.lstrip()
        is_err = stripped.startswith("❌") or "拒绝" in stripped
        groupable = ro and not is_err and rname != "write_todo"

        # 不同名 / 不可组 → 先把已攒的组冲掉,保证顺序
        if self._group and (not groupable or rname != self._group_name):
            self._flush_group()

        if groupable:
            self._group_name = rname
            self._group_ro = ro
            self._group.append((preview, text, elapsed_sec))
            # 后面还有同名排队 → 继续攒;否则立即落地(单个也即时,不等 close)
            if not (self._pending and self._pending[0][0] == rname):
                self._flush_group()
            return

        self._commit_single(rname, preview, ro, is_err, text,
                            elapsed_sec=elapsed_sec, max_lines=max_lines,
                            max_line_chars=max_line_chars)

    def _commit_single(self, name: str, preview: str, read_only: bool,
                       is_err: bool, text: str, *, elapsed_sec=None,
                       max_lines: int = 4, max_line_chars: int = 300) -> None:
        """单个工具结果落地(原 tool_result 主体)。"""
        self._dot = False        # 工具块后,下一段正文重新起一个 ⏺ 步(带块间空行)
        # ⏺ 颜色:出错→红;只读→中性;改动类成功→绿。
        color = theme.ERR if is_err else (theme.DEFAULT if read_only else theme.OK)

        # write_todo → 内联 Update Todos 清单
        if name == "write_todo":
            self._commit_step(chat_view._tool_notice_lines("Update Todos", "", theme.DEFAULT))
            self._commit(chat_view._render_update_todos(text))
            return

        self._commit_step(chat_view._tool_notice_lines(name, preview, color))

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

    def _flush_group(self) -> None:
        """把攒着的同类只读工具组落地:单个走普通渲染,多个合并成 'Read N files'。"""
        if not self._group:
            return
        items, name, ro = self._group, self._group_name, self._group_ro
        self._group, self._group_name, self._group_ro = [], None, False
        if len(items) == 1:
            preview, text, elapsed = items[0]
            stripped = text.lstrip()
            is_err = stripped.startswith("❌") or "拒绝" in stripped
            self._commit_single(name, preview, ro, is_err, text, elapsed_sec=elapsed)
            return
        color = theme.DEFAULT if ro else theme.OK
        total = sum(e or 0 for _, _, e in items)
        self._commit_step(chat_view._render_tool_group(
            name, [(p, t) for p, t, _ in items], color,
            elapsed_sec=total if total > 0 else None))
        self._dot = False        # 工具组后,下一段正文重新起一个 ⏺ 步

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
            self._commit_step(chat_view._render_thinking(buf))
            self._dot = False    # 思考块后,正文重新起一个 ⏺ 步
        self._set_active("", "text", self._dot)
