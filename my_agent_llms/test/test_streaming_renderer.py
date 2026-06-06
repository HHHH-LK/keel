"""StreamingAgentRenderer:tty 走 Live markdown,非 tty 退回 raw。"""
import io
from rich.console import Console
from my_agent_llms.cli import chat_view


def _console(tty: bool) -> Console:
    return Console(file=io.StringIO(), force_terminal=tty, width=80)


def test_non_tty_falls_back_to_raw():
    con = _console(False)
    r = chat_view.StreamingAgentRenderer(con)
    r.text_chunk("**粗**")
    r.close()
    out = con.file.getvalue()
    assert "**粗**" in out          # 非 tty:原样,不渲染、不开 Live


def test_tty_renders_markdown_bold_stripped():
    con = _console(True)
    r = chat_view.StreamingAgentRenderer(con)
    r.text_chunk("**粗**")
    r._close_text()                 # 段结束定格
    out = con.file.getvalue()
    assert "粗" in out and "**粗**" not in out   # tty:渲染掉 ** 标记


def test_framed_render_has_dot_prefix():
    con = _console(True)
    r = chat_view.StreamingAgentRenderer(con)
    framed = r._framed_render("你好")
    assert "⏺" in framed.plain and "你好" in framed.plain


from rich.text import Text


def test_tail_cap_keeps_only_last_lines():
    t = Text("\n".join(f"line{i}" for i in range(20)))   # 20 行
    capped = chat_view._tail_cap(t, height=10)            # cap = max(3, 10-6) = 4
    lines = capped.plain.split("\n")
    assert lines == ["line16", "line17", "line18", "line19"]


def test_tail_cap_short_text_unchanged():
    t = Text("a\nb\nc")
    capped = chat_view._tail_cap(t, height=24)            # cap = 18 > 3 行
    assert capped.plain == "a\nb\nc"


def test_tail_cap_floor_is_three():
    t = Text("\n".join(f"l{i}" for i in range(10)))
    capped = chat_view._tail_cap(t, height=1)             # cap = max(3, -5) = 3
    assert capped.plain.split("\n") == ["l7", "l8", "l9"]
