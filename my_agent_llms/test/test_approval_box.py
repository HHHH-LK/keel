"""审批:框内展示【截断的】完整改动(审核时可见,批准后随浮层消失,不落日志)+ ❯ 选择器。"""
import re

from my_agent_llms.cli import live_session as ls


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def test_approval_box_lists_three_options():
    raw = _plain(ls._render_approval_box("Write", "", 0, 70))
    assert "执行一次" in raw
    assert "本会话" in raw
    assert "拒绝" in raw


def test_approval_box_marks_only_selected_option_with_cursor():
    raw = _plain(ls._render_approval_box("Write", "", 1, 70))
    sel_line = next(l for l in raw.split("\n") if "本会话" in l)
    once_line = next(l for l in raw.split("\n") if "执行一次" in l)
    assert "❯" in sel_line
    assert "❯" not in once_line


def test_approval_box_shows_tool_name():
    assert "Edit" in _plain(ls._render_approval_box("Edit", "", 0, 70))


def test_approval_box_shows_diff_during_review():
    # 审核时框内能看到完整改动(批准后随浮层消失,不落 scrollback)。
    raw = _plain(ls._render_approval_box("Write", "+新增行\n-删除行\n 上下文", 0, 70))
    assert "+新增行" in raw
    assert "-删除行" in raw


def test_approval_box_colors_diff():
    out = ls._render_approval_box("Write", "+加\n-减", 0, 70)
    assert "\x1b[32m" in out      # 新增 → 绿
    assert "\x1b[31m" in out      # 删除 → 红


def test_approval_box_caps_long_diff():
    long_diff = "\n".join(f"+line{i}" for i in range(40))
    raw = _plain(ls._render_approval_box("Write", long_diff, 0, 70, diff_cap=10))
    assert "line0" in raw
    assert "line39" not in raw    # 超出 cap 的不展示(框不被撑爆,选项常驻)
    assert "还有" in raw          # 给出省略提示


def test_approval_options_map_to_decisions():
    from my_agent_llms.cli.permission import PermissionDecision
    decisions = [d for d, _label, _hint in ls._APPROVAL_OPTIONS]
    assert decisions == [
        PermissionDecision.ALLOW_ONCE,
        PermissionDecision.ALLOW_ALWAYS,
        PermissionDecision.DENY,
    ]
