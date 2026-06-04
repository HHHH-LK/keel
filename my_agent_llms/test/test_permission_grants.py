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


from my_agent_llms.cli.permission_grants import PermissionGrants, decide


def test_grant_by_tool_and_path_prefix():
    g = PermissionGrants()
    assert g.is_granted("Edit", {"path": "src/a.py"}) is False
    g.grant("Edit", {"path": "src/a.py"})              # 记 (Edit, "src")
    assert g.is_granted("Edit", {"path": "src/a.py"}) is True      # 同目录
    assert g.is_granted("Edit", {"path": "src/sub/b.py"}) is True  # 子目录前缀命中
    assert g.is_granted("Edit", {"path": "tests/c.py"}) is False   # 别的目录不命中
    assert g.is_granted("Write", {"path": "src/a.py"}) is False    # 别的工具不串


def test_grant_without_path_is_whole_tool():
    g = PermissionGrants()
    g.grant("Calc", {})                       # 无 path → 整工具授权
    assert g.is_granted("Calc", {"x": 1}) is True


def test_decide_granted_skips_prompt():
    g = PermissionGrants()
    g.grant("Edit", {"path": "src/a.py"})
    calls = []
    def fake_prompt(n, a, p):
        calls.append(n); return PermissionDecision.DENY
    assert decide(g, fake_prompt, "Edit", {"path": "src/x.py"}, "diff") is True
    assert calls == []                        # 命中授权,不弹框


def test_decide_allow_once_does_not_persist():
    g = PermissionGrants()
    assert decide(g, lambda *a: PermissionDecision.ALLOW_ONCE,
                  "Edit", {"path": "src/a.py"}, "d") is True
    assert g.is_granted("Edit", {"path": "src/a.py"}) is False   # 没记住


def test_decide_allow_always_persists():
    g = PermissionGrants()
    assert decide(g, lambda *a: PermissionDecision.ALLOW_ALWAYS,
                  "Edit", {"path": "src/a.py"}, "d") is True
    assert g.is_granted("Edit", {"path": "src/b.py"}) is True    # 同目录前缀已记住


def test_decide_deny_returns_false():
    g = PermissionGrants()
    assert decide(g, lambda *a: PermissionDecision.DENY,
                  "Edit", {"path": "src/a.py"}, "d") is False
