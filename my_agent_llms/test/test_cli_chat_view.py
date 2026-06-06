"""chat_view — left-bar role rendering for user input, agent reply, errors."""
import re

from rich.console import Console

from my_agent_llms.cli import chat_view


def _capture(fn, *args, **kwargs) -> str:
    test_console = Console(force_terminal=True, width=80, color_system="truecolor")
    with test_console.capture() as cap:
        fn(test_console, *args, **kwargs)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_render_user_has_header_and_left_bar():
    out = _capture(chat_view.render_user, "hello world")
    assert "you" in out
    assert "┃" in out
    assert "hello world" in out


def test_render_user_multiline_each_line_has_bar():
    out = _capture(chat_view.render_user, "line1\nline2\nline3")
    bar_lines = [line for line in out.splitlines() if "┃" in line]
    assert len(bar_lines) >= 3


def test_render_agent_plain_text_uses_step_dot_and_no_header():
    """Claude Code 风格:⏺ 开头,不再有"伙伴 · time" header。"""
    out = _capture(chat_view.render_agent, "just a plain reply",
                   tools_used=0, elapsed_seconds=1.5)
    assert "伙伴" not in out  # header 已删
    assert "⏺" in out          # 改成 step dot
    assert "just a plain reply" in out


def test_render_agent_does_not_show_tool_count_or_elapsed():
    """render_agent 不再渲染 meta (tools / elapsed) —— 走 Claude Code 极简风。"""
    out = _capture(chat_view.render_agent, "ok",
                   tools_used=3, elapsed_seconds=2.3)
    assert "3 tools" not in out
    assert "2.3s" not in out


def test_render_agent_markdown_renders_list_under_step_dot():
    md = "Here is a list:\n\n- item one\n- item two\n"
    out = _capture(chat_view.render_agent, md, tools_used=0, elapsed_seconds=0.1)
    assert "item one" in out
    assert "item two" in out
    # 整段只起一个 ⏺,续行缩进(没有每行 ┃ bar)
    assert out.count("⏺") == 1
    assert "┃" not in out


def test_render_agent_error_uses_error_marker():
    out = _capture(chat_view.render_agent_error, "boom")
    assert "error" in out
    assert "boom" in out


def test_render_agent_error_message_carries_bar_prefix():
    out = _capture(chat_view.render_agent_error, "connection failed")
    msg_lines = [line for line in out.splitlines() if "connection failed" in line]
    assert msg_lines, "error message should appear in output"
    assert all("┃" in line for line in msg_lines), msg_lines


def test_print_not_ready_hint_mentions_config_key():
    out = _capture(chat_view.print_not_ready_hint)
    assert "not ready" in out.lower() or "missing" in out.lower()
    assert "/config" in out


# ── StreamingAgentRenderer: 工具调用排版 (Claude Code 风格) ──────────

def _capture_renderer(fn) -> str:
    """跑一段用 StreamingAgentRenderer 的逻辑,抓屏并剥 ANSI。"""
    test_console = Console(force_terminal=True, width=80, color_system="truecolor")
    with test_console.capture() as cap:
        fn(test_console)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_tool_notice_multiline_args_align_under_paren():
    """多行 Bash 命令:续行对齐到 'name(' 之后,而非硬缩 2 格。"""
    def run(console):
        r = chat_view.StreamingAgentRenderer(console)
        r.tool_notice("Bash", "cd ~/x\necho hi\npip install foo")
        r.tool_result("done", elapsed_sec=0.1)
        r.close()
    out = _capture_renderer(run)
    lines = out.splitlines()
    head = next(i for i, l in enumerate(lines) if "⏺ Bash(" in l)
    # 续行 'echo hi' 缩进 = len('⏺ ') + len('Bash') + len('(') = 7 空格
    cont = lines[head + 1]
    assert cont.startswith(" " * 7 + "echo hi"), repr(cont)


def test_tool_result_folds_to_four_lines_with_more_marker():
    """结果 >4 行 → 只显示前 4 行 + '… +N lines'。"""
    def run(console):
        r = chat_view.StreamingAgentRenderer(console)
        r.tool_notice("Bash", "pip install x")
        body = "\n".join(f"line{i}" for i in range(10))
        r.tool_result(body, elapsed_sec=0.1)
        r.close()
    out = _capture_renderer(run)
    assert "line0" in out and "line3" in out
    assert "line4" not in out          # 第 5 行起被折叠
    assert "… +6 lines" in out


def test_tool_result_short_output_no_more_marker():
    def run(console):
        r = chat_view.StreamingAgentRenderer(console)
        r.tool_notice("ReadFile", "a.py")
        r.tool_result("✅ ok\nline2", elapsed_sec=0.05)
        r.close()
    out = _capture_renderer(run)
    assert "line2" in out
    assert "more lines" not in out


def test_multi_tool_round_pairs_each_result_with_its_notice():
    """真实回调顺序:Phase A 先全部 tool_notice,Phase C 再全部 tool_result。
    每个结果必须配对到自己的 ⏺(FIFO),不能互相覆盖丢失。"""
    def run(console):
        r = chat_view.StreamingAgentRenderer(console)
        for f in ("a.py", "b.py", "c.py"):           # Phase A:全部 notice
            r.tool_notice("ReadFile", f)
        for f in ("a.py", "b.py", "c.py"):           # Phase C:全部 result
            r.tool_result(f"✅ {f} ok", elapsed_sec=0.05)
        r.close()
    out = _capture_renderer(run)
    # 三个文件名都得作为独立 ⏺ 头出现(不再只剩最后一个 c.py)
    assert "ReadFile(a.py)" in out
    assert "ReadFile(b.py)" in out
    assert "ReadFile(c.py)" in out
    assert out.count("⏺") == 3
    # 结果与各自的 ⏺ 配对
    assert "a.py ok" in out and "b.py ok" in out and "c.py ok" in out


def test_close_flushes_unpaired_notices_as_headers():
    """notice 多于 result(极端:某些工具没上报结果)→ close 时残留的也要打出头,不吞。"""
    def run(console):
        r = chat_view.StreamingAgentRenderer(console)
        r.tool_notice("ReadFile", "a.py")
        r.tool_notice("ReadFile", "b.py")
        r.tool_result("✅ a ok", elapsed_sec=0.05)   # 只报了一个
        r.close()
    out = _capture_renderer(run)
    assert "ReadFile(a.py)" in out
    assert "ReadFile(b.py)" in out


# ── 守卫:钉底 todo 渲染路径已退役(方案 A 单一上滚路径)──────────────
def test_pinned_todo_path_is_retired():
    from my_agent_llms.cli.app import ChatCLI
    assert not hasattr(chat_view, "LiveTurnRenderer"), "LiveTurnRenderer 应已删除"
    assert not hasattr(chat_view, "render_todo_panel"), "render_todo_panel 应已删除"
    assert not hasattr(ChatCLI, "_chat_once_live"), "_chat_once_live 应已删除"
