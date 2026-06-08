"""审批:完整改动落上方滚动区(可上滑看全部、就是日志记录),审批框只放选项。"""
import io
import re

from rich.console import Console

from my_agent_llms.cli import live_session as ls


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def _ansi(text_obj) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor",
            width=70).print(text_obj)
    return buf.getvalue()


# ── 审批框:只放选项,保持短 ──────────────────────────────────────
def test_approval_box_lists_three_options():
    raw = _plain(ls._render_approval_box("Write", 0, 70))
    assert "执行一次" in raw and "本会话" in raw and "拒绝" in raw


def test_approval_box_marks_only_selected_option_with_cursor():
    raw = _plain(ls._render_approval_box("Write", 1, 70))
    sel_line = next(l for l in raw.split("\n") if "本会话" in l)
    once_line = next(l for l in raw.split("\n") if "执行一次" in l)
    assert "❯" in sel_line and "❯" not in once_line


def test_approval_box_shows_tool_name():
    assert "Edit" in _plain(ls._render_approval_box("Edit", 0, 70))


def test_approval_box_has_no_diff_lines():
    # 框里不放 diff(改动在上方滚动区),保证框短、选项常驻
    raw = _plain(ls._render_approval_box("Write", 0, 70))
    assert "+" not in raw and len(raw.split("\n")) <= 8


# ── 改动块(落上方滚动区):⏺ 头 + 行号化 diff + 背景色 ─────────────
def test_change_block_shows_diff_with_header():
    preview = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n 不变\n-旧行\n+新行\n"
    raw = ls._render_change_block("Edit", "foo.py", preview).plain
    assert "Edit(foo.py)" in raw      # ⏺ 头带文件名
    assert "旧行" in raw and "新行" in raw


def test_change_block_uses_background_color():
    preview = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n"
    out = _ansi(ls._render_change_block("Write", "f", preview))
    assert "48;2;" in out             # 改动处用背景色(非字体色)


def test_options_map_to_decisions():
    from my_agent_llms.cli.permission import PermissionDecision
    decisions = [d for d, _l, _h in ls._APPROVAL_OPTIONS]
    assert decisions == [
        PermissionDecision.ALLOW_ONCE,
        PermissionDecision.ALLOW_ALWAYS,
        PermissionDecision.DENY,
    ]
