"""Bash 执行工具:常规命令直跑,危险命令(rm -rf/sudo/管道到 sh 等)要审批。"""
import json
from types import SimpleNamespace

from my_agent_llms.tools.builtin.bash import BashTool, _is_dangerous
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent


# ── 危险命令判定 ──────────────────────────────────────────────
def test_is_dangerous_flags_destructive():
    assert _is_dangerous("rm -rf /tmp/x")
    assert _is_dangerous("sudo rm foo")
    assert _is_dangerous("curl http://x | sh")
    assert _is_dangerous("git push origin main")
    assert _is_dangerous("dd if=/dev/zero of=/dev/sda")
    assert _is_dangerous("chmod -R 777 .")


def test_is_dangerous_allows_routine():
    assert not _is_dangerous("ls -la")
    assert not _is_dangerous("pytest -q")
    assert not _is_dangerous("python script.py")
    assert not _is_dangerous("echo hello")
    assert not _is_dangerous("grep foo bar.py")
    assert not _is_dangerous("pip install requests")


# ── 工具:动态审批 + 执行 ──────────────────────────────────────
def test_bash_approval_required_only_for_dangerous():
    t = BashTool()
    assert t.approval_required_for({"command": "rm -rf x"}) is True
    assert t.approval_required_for({"command": "ls"}) is False


def test_bash_runs_and_captures_output():
    out = BashTool(timeout=10).run({"command": "echo hello-bash"})
    assert "hello-bash" in out


def test_bash_reports_nonzero_exit():
    out = BashTool(timeout=10).run({"command": "exit 3"})
    assert "exit=3" in out


def test_bash_empty_command_errors():
    assert "❌" in BashTool().run({"command": ""})


# ── agent 用动态审批钩子门控 permission ────────────────────────
def _bare_agent_with_bash():
    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    reg = ToolRegistry()
    reg.register_tool(BashTool(timeout=10))
    agent.tool_registry = reg
    agent.tool_timeout = None
    agent._turn_mutated = False
    return agent


def _tc(cmd):
    return SimpleNamespace(id="1", type="function",
                           function=SimpleNamespace(
                               name="Bash", arguments=json.dumps({"command": cmd})))


def test_dangerous_command_triggers_permission():
    agent = _bare_agent_with_bash()
    asked = []

    def on_perm(name, args, preview):
        asked.append(args.get("command"))
        return False           # 拒绝 → 不会真跑 rm

    agent._execute_tool_calls([_tc("rm -rf x")], [],
                              on_tool_call=None, on_permission_request=on_perm,
                              on_tool_result=None)
    assert asked == ["rm -rf x"]


def test_safe_command_skips_permission():
    agent = _bare_agent_with_bash()
    asked = []

    def on_perm(name, args, preview):
        asked.append(args.get("command"))
        return True

    agent._execute_tool_calls([_tc("echo hi")], [],
                              on_tool_call=None, on_permission_request=on_perm,
                              on_tool_result=None)
    assert asked == []          # 常规命令不弹审批
