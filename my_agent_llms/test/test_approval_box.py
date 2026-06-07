"""审批浮层:圆角框 + ❯ 选择器(从 _demo_approval 并进 live_session)。"""
import re

from my_agent_llms.cli import live_session as ls


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def test_approval_box_lists_three_options():
    raw = _plain(ls._render_approval_box("Write", "写入 LICENSE", sel=0, width=70))
    assert "执行一次" in raw
    assert "本会话" in raw
    assert "拒绝" in raw


def test_approval_box_marks_only_selected_option_with_cursor():
    raw = _plain(ls._render_approval_box("Write", "写入 LICENSE", sel=1, width=70))
    sel_line = next(l for l in raw.split("\n") if "本会话" in l)
    once_line = next(l for l in raw.split("\n") if "执行一次" in l)
    deny_line = next(l for l in raw.split("\n") if "拒绝" in l)
    assert "❯" in sel_line            # 选中项(index 1)带光标
    assert "❯" not in once_line       # 其余不带
    assert "❯" not in deny_line


def test_approval_box_shows_tool_name():
    raw = _plain(ls._render_approval_box("Edit", "改 pyproject.toml", sel=0, width=70))
    assert "Edit" in raw


def test_approval_options_map_to_decisions():
    # 顺序:执行一次 / 本会话总是 / 拒绝 → ALLOW_ONCE / ALLOW_ALWAYS / DENY
    from my_agent_llms.cli.permission import PermissionDecision
    decisions = [d for d, _label, _hint in ls._APPROVAL_OPTIONS]
    assert decisions == [
        PermissionDecision.ALLOW_ONCE,
        PermissionDecision.ALLOW_ALWAYS,
        PermissionDecision.DENY,
    ]
