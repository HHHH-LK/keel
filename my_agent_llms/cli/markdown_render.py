"""自定义极简 Markdown → rich.Text 渲染器(贴 Claude Code 干净风)。

设计:
- 返回单个 rich.Text(代码块/表格也烘成带 ANSI 样式的 Text 行),方便上层
  用 _step_lines_from_text 统一加 ⏺/缩进包框。
- 对**残缺** markdown 鲁棒(流式每帧可能喂半成品):未闭合 **/`/*/~~ 按字面;
  未闭合代码围栏当"进行中代码块";半张表按已到行渲染。
- 永不抛:任何异常 → 退回 Text(原文)。
"""
from __future__ import annotations

import io
import re
from typing import List

from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.syntax import Syntax
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
_TABLE_SEP_RE   = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")

_HR_LINE = "─" * 8
_FENCE_RE = re.compile(r"^\s*```(\w+)?\s*$")


def render_inline(text: str) -> Text:
    """只渲 inline(bold/italic/code/strike/link)。其它原文保留。残缺标记按字面。"""
    # 链接先抽成占位符再 escape:否则 rich.markup.escape 会把 [小写字母开头...]
    # 转义成 \[...],导致链接(尤其英文小写文字)整行 markup 破掉。占位符不含
    # [ 或 \,escape 不会动它,后续 inline sub 也不会误伤(占位符无 markdown 字符)。
    links: list[str] = []

    def _stash_link(m):
        # Claude Code 风:链接文字用默认色(不强调),仅 URL 暗色
        links.append(f"{_rich_escape(m.group(1))}"
                     f"[{theme.DIM}]({_rich_escape(m.group(2))})[/]")
        return f"\x00L{len(links) - 1}\x00"

    safe = _LINK_RE.sub(_stash_link, text)
    safe = _rich_escape(safe)
    safe = _INLINE_CODE_RE.sub(r"[dim bold]\1[/]", safe)
    safe = _STRIKE_RE.sub(r"[strike]\1[/]", safe)
    safe = _BOLD_RE.sub(r"[bold]\1[/]", safe)
    safe = _ITALIC_RE.sub(r"[italic]\1[/]", safe)
    for idx, markup in enumerate(links):       # 还原链接 markup
        safe = safe.replace(f"\x00L{idx}\x00", markup)
    try:
        return Text.from_markup(safe)
    except Exception:
        return Text(text)


def _split_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _render_table(rows: List[List[str]], width: int) -> Text:
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    colw = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(ncol)]
    out = Text()
    for ri, row in enumerate(rows):
        if ri:
            out.append("\n")
        for ci, cell in enumerate(row):
            if ci:
                out.append("  ")
            seg = render_inline(cell.ljust(colw[ci]))
            if ri == 0:
                seg.stylize("bold")
            out.append_text(seg)
        if ri == 0:
            out.append("\n")
            out.append("─" * min(width, sum(colw) + 2 * (ncol - 1)), style=theme.DIM)
    return out


def _render_code_block(code: str, lang: str, width: int) -> Text:
    code = code.rstrip("\n")
    try:
        syntax = Syntax(code, lang or "text", theme="ansi_dark",
                        background_color="default", word_wrap=False)
        buf = io.StringIO()
        tmp = Console(file=buf, force_terminal=True, color_system="truecolor",
                      width=max(20, width))
        tmp.print(syntax)
        return Text.from_ansi(buf.getvalue().rstrip("\n"))
    except Exception:
        return Text(code)


def _render_header(hashes: str, body: str) -> Text:
    # Claude Code 风:标题只 bold,不上强调色(去视觉疲劳)
    return Text(body.strip(), style="bold")


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
        # 代码块(``` 围栏;未闭合也当进行中代码块)
        fm = _FENCE_RE.match(line)
        if fm:
            lang = fm.group(1) or "text"
            body = []
            i += 1
            while i < n and not _FENCE_RE.match(lines[i]):
                body.append(lines[i]); i += 1
            if i < n:        # 跳过收尾围栏(未闭合时 i==n,不跳)
                i += 1
            blocks.append(_render_code_block("\n".join(body), lang, width))
            continue
        # 表格(连续以 | 开头的行;跳过 |---| 分隔行)
        if "|" in line and line.strip().startswith("|"):
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip().startswith("|"):
                if not _TABLE_SEP_RE.match(lines[i]):
                    rows.append(_split_row(lines[i]))
                i += 1
            if rows:
                blocks.append(_render_table(rows, width))
            continue
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
