"""StreamingAgentRenderer:tty 走 Live markdown,非 tty 退回 raw。"""
import io
import re
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


def test_close_commits_full_text_even_when_live_capped():
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=8)
    r = chat_view.StreamingAgentRenderer(con)
    long_text = "\n".join(f"row{i}" for i in range(30))
    r.text_chunk(long_text)
    # 流式期间的 live 帧应被截断(不含早期行)
    frame_plain = r._active_frame(long_text).plain
    assert "row0" not in frame_plain, "Live 帧应被尾区截断"
    # close 后全文必须落进 scrollback,且经由 console.print(_framed_render) 路径
    r._close_text()
    out = con.file.getvalue()
    assert "row0" in out    # 早期行从 scrollback commit 找回
    assert "row29" in out   # 尾部也在
    assert "⏺" in out       # _framed_render 前缀,确认走了 console.print 新路径


def test_active_frame_is_capped_during_stream():
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=8)
    r = chat_view.StreamingAgentRenderer(con)
    frame = r._active_frame("\n".join(f"row{i}" for i in range(30)))
    plain = frame.plain
    # height=8 → cap=max(3,8-6)=2;_framed_render 带 ⏺/缩进/markdown,故用宽松断言:
    assert "row29" in plain                    # 尾部保留
    assert "row0" not in plain                 # 早期行被截
    assert len(plain.split("\n")) <= 4         # 高度受限(cap=2,留余量防 markdown 末空行)


def test_reasoning_chunk_commits_dim_thinking_block():
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=40)
    r = chat_view.StreamingAgentRenderer(con)
    r.reasoning_chunk("我先分析一下\n第二行思考\n")
    r.text_chunk("正式回答")     # 切到 text 段,应先 commit reasoning 段
    # 关键(state 级,不依赖 StringIO Live 帧泄漏):text_chunk 必须已收尾 reasoning 段,
    # 否则旧 reasoning Live 不停 + 正文 buf 在 _close_text 被跳过丢失。
    assert r._reason_open is False
    assert r._text_open is True
    r.close()
    out = re.sub(r"\x1b\[[0-9;]*m", "", con.file.getvalue())
    assert "我先分析一下" in out
    assert "正式回答" in out
    assert "✻" in out


def test_split_committable_basic():
    committable, remainder = chat_view._split_committable("p1\n\np2 wip")
    assert committable == "p1"
    assert remainder == "p2 wip"


def test_split_committable_no_block_boundary_holds_everything():
    committable, remainder = chat_view._split_committable("一段没有空行的话")
    assert committable == ""
    assert remainder == "一段没有空行的话"


def test_split_committable_holds_open_code_fence():
    # 代码围栏未闭合 → 整块留残块,fence 内的空行不当提交点
    buf = "前言\n\n```py\n\ncode line"
    committable, remainder = chat_view._split_committable(buf)
    assert committable == "前言"
    assert "```py" in remainder
    assert "code line" in remainder


def test_text_chunk_commits_completed_block_out_of_buffer():
    # 核心:写完的块在 close 之前就 commit 出 buffer(不再等末尾一次性出现)
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=40)
    r = chat_view.StreamingAgentRenderer(con)
    r.text_chunk("第一段已经写完。\n\n第二段进行中")
    assert "第一段" not in r._text_buf          # 首块已落 scrollback
    assert "第二段进行中" in r._text_buf         # 进行中块仍在 live buffer
    assert r._dot_emitted is True               # ⏺ 已在首块发出


def test_progressive_commit_appears_before_close():
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=40)
    r = chat_view.StreamingAgentRenderer(con)
    r.text_chunk("alpha block.\n\n")
    # 还没 close,首块就该已进 scrollback(经 console.print),buffer 已清空首块
    assert "alpha block" not in r._text_buf
    out = re.sub(r"\x1b\[[0-9;]*m", "", con.file.getvalue())
    assert "alpha block" in out


def test_second_block_indents_without_extra_dot():
    # 段内非首块 commit 用 2 空格缩进续行,不再补第二个 ⏺
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=40)
    r = chat_view.StreamingAgentRenderer(con)
    r.text_chunk("blk1\n\nblk2\n\ntail")
    r.close()
    out = re.sub(r"\x1b\[[0-9;]*m", "", con.file.getvalue())
    assert out.count("⏺") == 1                  # 整段只有一个 ⏺ 头
    assert "blk1" in out and "blk2" in out and "tail" in out


def test_reasoning_folds_to_single_line():
    con = Console(file=io.StringIO(), force_terminal=True, width=80, height=40)
    r = chat_view.StreamingAgentRenderer(con)
    r.reasoning_chunk("\n".join(f"思考{i}" for i in range(10)))
    r.close()
    out = re.sub(r"\x1b\[[0-9;]*m", "", con.file.getvalue())
    # 折成 1 行:只留 ✻ + 首行,后续思考行不再整段倒出
    assert "思考0" in out
    assert "思考1" not in out
    assert "思考9" not in out
