"""cli/permission.py:画 Panel + 单键拿决策。"""
import pytest

from my_agent_llms.cli import permission as perm
from my_agent_llms.cli.permission import PermissionDecision


def test_prompt_permission_returns_allow_once_when_user_presses_y(monkeypatch, capsys):
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.ALLOW_ONCE)
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    result = perm.prompt_permission("EditFile",
                                    {"path": "a.md", "old": "x", "new": "y"},
                                    "--- a.md\n+++ a.md\n@@ -1 +1 @@\n-x\n+y\n")
    assert result is PermissionDecision.ALLOW_ONCE


def test_prompt_permission_returns_deny_when_user_presses_n(monkeypatch):
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.DENY)
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    result = perm.prompt_permission("WriteFile", {"path": "b.md"}, "(新建文件)")
    assert result is PermissionDecision.DENY


def test_prompt_permission_raises_on_non_tty(monkeypatch):
    monkeypatch.setattr(perm, "_is_tty", lambda: False)
    with pytest.raises(perm.TerminalNotInteractiveError):
        perm.prompt_permission("EditFile", {}, "preview")


def test_panel_includes_tool_name_and_preview(monkeypatch, capsys):
    """从 rich console 抓输出,断言 panel 里有工具名和 preview 文本片段。"""
    from rich.console import Console
    from io import StringIO

    captured = StringIO()
    monkeypatch.setattr(perm, "console", Console(file=captured, force_terminal=False, width=100))
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.ALLOW_ONCE)
    monkeypatch.setattr(perm, "_is_tty", lambda: True)

    perm.prompt_permission("EditFile",
                           {"path": "foo.py"},
                           "--- foo.py\n+++ foo.py\n-return None\n+return data\n")
    out = captured.getvalue()
    assert "EditFile" in out
    assert "return None" in out
    assert "return data" in out
    assert "y" in out.lower() and "n" in out.lower()


def _capture_console(monkeypatch):
    from rich.console import Console
    from io import StringIO
    buf = StringIO()
    monkeypatch.setattr(perm, "console", Console(file=buf, force_terminal=False, width=100))
    return buf


def test_decision_echo_when_allowed(monkeypatch):
    buf = _capture_console(monkeypatch)
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.ALLOW_ONCE)
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    perm.prompt_permission("EditFile", {"path": "x"}, "preview text")
    out = buf.getvalue()
    assert "已同意" in out
    assert "已拒绝" not in out


def test_decision_echo_when_rejected(monkeypatch):
    buf = _capture_console(monkeypatch)
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.DENY)
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    perm.prompt_permission("EditFile", {"path": "x"}, "preview text")
    out = buf.getvalue()
    assert "已拒绝" in out
    assert "已同意" not in out
