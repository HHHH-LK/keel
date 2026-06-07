"""审批:紧凑框(标题+❯选择器+选项)+ 完整改动落 scrollback(可上滑查看)。"""
import re

from my_agent_llms.cli import live_session as ls


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def test_approval_box_lists_three_options():
    raw = _plain(ls._render_approval_box("Write", sel=0, width=70))
    assert "执行一次" in raw
    assert "本会话" in raw
    assert "拒绝" in raw


def test_approval_box_marks_only_selected_option_with_cursor():
    raw = _plain(ls._render_approval_box("Write", sel=1, width=70))
    sel_line = next(l for l in raw.split("\n") if "本会话" in l)
    once_line = next(l for l in raw.split("\n") if "执行一次" in l)
    assert "❯" in sel_line
    assert "❯" not in once_line


def test_approval_box_shows_tool_name():
    raw = _plain(ls._render_approval_box("Edit", sel=0, width=70))
    assert "Edit" in raw


def test_approval_box_stays_compact_without_diff():
    # 框内不再塞 diff —— 无论改动多大,框都紧凑,选项常驻可见。
    raw = _plain(ls._render_approval_box("Write", sel=0, width=70))
    assert len(raw.split("\n")) <= 12


def test_preview_block_keeps_full_content_for_scrollback():
    # 完整改动落 scrollback(可上滑查看),不截断。
    long_diff = "\n".join(f"+ line {i}" for i in range(40))
    blk = ls._render_preview_block("Write", long_diff)
    assert "line 0" in blk.plain
    assert "line 39" in blk.plain          # 末尾内容也在


def test_approval_options_map_to_decisions():
    from my_agent_llms.cli.permission import PermissionDecision
    decisions = [d for d, _label, _hint in ls._APPROVAL_OPTIONS]
    assert decisions == [
        PermissionDecision.ALLOW_ONCE,
        PermissionDecision.ALLOW_ALWAYS,
        PermissionDecision.DENY,
    ]
