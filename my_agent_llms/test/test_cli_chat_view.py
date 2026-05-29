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
