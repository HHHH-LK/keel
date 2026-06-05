"""自定义极简 Markdown → rich.Text 渲染器(贴 Claude Code 干净风)。

设计:
- 返回单个 rich.Text(代码块/表格也烘成带 ANSI 样式的 Text 行),方便上层
  用 _step_lines_from_text 统一加 ⏺/缩进包框。
- 对**残缺** markdown 鲁棒(流式每帧可能喂半成品):未闭合 **/`/*/~~ 按字面;
  未闭合代码围栏当"进行中代码块";半张表按已到行渲染。
- 永不抛:任何异常 → 退回 Text(原文)。
"""
from __future__ import annotations

import re
from typing import List

from rich.markup import escape as _rich_escape
from rich.text import Text

from . import theme

_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE        = re.compile(r"\*\*([^*\n]+?)\*\*")
_ITALIC_RE      = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?!\w)")
_STRIKE_RE      = re.compile(r"~~([^~\n]+?)~~")
_LINK_RE        = re.compile(r"\[([^\]\n]+?)\]\((https?://[^)\s]+)\)")
_HEADER_RE      = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE       = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OLIST_RE       = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_HR_RE          = re.compile(r"^\s*([-*_])\1{2,}\s*$")

_HR_LINE = "─" * 8


def render_inline(text: str) -> Text:
    """只渲 inline(bold/italic/code/strike/link)。其它原文保留。残缺标记按字面。"""
    safe = _rich_escape(text)
    safe = _LINK_RE.sub(rf"[{theme.ACCENT}]\1[/][{theme.DIM}](\2)[/]", safe)
    safe = _INLINE_CODE_RE.sub(r"[dim bold]\1[/]", safe)
    safe = _STRIKE_RE.sub(r"[strike]\1[/]", safe)
    safe = _BOLD_RE.sub(r"[bold]\1[/]", safe)
    safe = _ITALIC_RE.sub(r"[italic]\1[/]", safe)
    try:
        return Text.from_markup(safe)
    except Exception:
        return Text(text)


def _render_header(hashes: str, body: str) -> Text:
    return Text(body.strip(), style=f"bold {theme.ACCENT}")


def _render_hr() -> Text:
    return Text(_HR_LINE, style=theme.DIM)


def _render_quote_block(lines: List[str]) -> Text:
    out = Text()
    for i, ln in enumerate(lines):
        body = ln.lstrip()[1:].lstrip() if ln.lstrip().startswith(">") else ln
        if i:
            out.append("\n")
        out.append("▏ ", style=theme.DIM)
        seg = render_inline(body)
        seg.stylize(theme.DIM)
        out.append_text(seg)
    return out


def _render_list_block(items: List[tuple]) -> Text:
    out = Text()
    for i, (indent, marker, body) in enumerate(items):
        if i:
            out.append("\n")
        out.append(" " * indent)
        out.append(marker + " ")
        out.append_text(render_inline(body))
    return out


def _join(blocks: List[Text]) -> Text:
    out = Text()
    for i, b in enumerate(blocks):
        if i:
            out.append("\n")
        out.append_text(b)
    return out


def render_markdown(text: str, width: int = 80) -> Text:
    try:
        return _render(text, width)
    except Exception:
        return Text(text)


def _render(text: str, width: int) -> Text:
    lines = text.split("\n")
    blocks: List[Text] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _HR_RE.match(line):
            blocks.append(_render_hr()); i += 1; continue
        m = _HEADER_RE.match(line)
        if m:
            blocks.append(_render_header(m.group(1), m.group(2))); i += 1; continue
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(lines[i]); i += 1
            blocks.append(_render_quote_block(buf)); continue
        if _ULIST_RE.match(line) or _OLIST_RE.match(line):
            items = []
            while i < n and (_ULIST_RE.match(lines[i]) or _OLIST_RE.match(lines[i])):
                um = _ULIST_RE.match(lines[i]); om = _OLIST_RE.match(lines[i])
                if om:
                    items.append((len(om.group(1)), om.group(2) + ".", om.group(3)))
                else:
                    items.append((len(um.group(1)), "•", um.group(2)))
                i += 1
            blocks.append(_render_list_block(items)); continue
        if line.strip() == "":
            blocks.append(Text("")); i += 1; continue
        blocks.append(render_inline(line)); i += 1
    return _join(blocks)
