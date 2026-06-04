"""三态审批枚举 + prompt_permission 返回枚举。"""
from my_agent_llms.cli import permission as perm
from my_agent_llms.cli.permission import PermissionDecision


def test_decision_enum_has_three_states():
    assert {d.name for d in PermissionDecision} == {
        "ALLOW_ONCE", "ALLOW_ALWAYS", "DENY"
    }


def test_prompt_returns_allow_once_on_y(monkeypatch):
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.ALLOW_ONCE)
    assert perm.prompt_permission("Edit", {"path": "a.py"}, "diff") is PermissionDecision.ALLOW_ONCE


def test_prompt_returns_allow_always_on_a(monkeypatch):
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.ALLOW_ALWAYS)
    assert perm.prompt_permission("Edit", {"path": "a.py"}, "diff") is PermissionDecision.ALLOW_ALWAYS


def test_prompt_returns_deny_on_n(monkeypatch):
    monkeypatch.setattr(perm, "_is_tty", lambda: True)
    monkeypatch.setattr(perm, "_read_decision_key", lambda: PermissionDecision.DENY)
    assert perm.prompt_permission("Edit", {"path": "a.py"}, "diff") is PermissionDecision.DENY
