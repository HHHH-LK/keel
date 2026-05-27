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


def test_render_agent_plain_text_includes_bar_and_content():
    out = _capture(chat_view.render_agent, "just a plain reply",
                   tools_used=0, elapsed_seconds=1.5)
    assert "伙伴" in out
    assert "┃" in out
    assert "just a plain reply" in out


def test_render_agent_with_tools_shows_tool_count_in_header():
    out = _capture(chat_view.render_agent, "ok",
                   tools_used=3, elapsed_seconds=2.3)
    assert "3 tools" in out


def test_render_agent_no_tools_omits_tool_count():
    out = _capture(chat_view.render_agent, "ok",
                   tools_used=0, elapsed_seconds=1.0)
    assert "tools" not in out


def test_render_agent_markdown_renders_list_and_keeps_bar():
    md = "Here is a list:\n\n- item one\n- item two\n"
    out = _capture(chat_view.render_agent, md, tools_used=0, elapsed_seconds=0.1)
    assert "item one" in out
    assert "item two" in out
    content_lines = [
        line for line in out.splitlines()
        if line.strip() and "伙伴" not in line and "─" not in line
        and "item" in line
    ]
    assert all("┃" in line for line in content_lines), out


def test_render_agent_error_uses_error_marker():
    out = _capture(chat_view.render_agent_error, "boom")
    assert "error" in out
    assert "boom" in out


def test_print_not_ready_hint_mentions_config_key():
    out = _capture(chat_view.print_not_ready_hint)
    assert "not ready" in out.lower() or "missing" in out.lower()
    assert "/config" in out
