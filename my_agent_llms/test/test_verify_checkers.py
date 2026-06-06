"""verify/checkers.py: 每个 checker 的过/不过 + CheckerRunner 汇总。"""
import json
from types import SimpleNamespace

import pytest

from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.checkers import CheckContext, check_one, CheckerRunner


def _ctx(result="", trajectory=None, workspace=None, source=None):
    return CheckContext(result=result, trajectory=trajectory or [],
                        workspace=workspace, source=source)


def test_string_contains_pass_and_fail():
    c = Check(id="c", type="string_contains", params={"s": "hello"})
    assert check_one(c, _ctx(result="say hello world")) is True
    assert check_one(c, _ctx(result="goodbye")) is False


def test_string_absent_pass_and_fail():
    c = Check(id="c", type="string_absent", params={"s": "Traceback"})
    assert check_one(c, _ctx(result="all good")) is True
    assert check_one(c, _ctx(result="Traceback (most recent call last)")) is False


def test_field_equals_reads_json_from_workspace(tmp_path):
    from my_agent_llms.workspace.workspace import Workspace
    (tmp_path / "out.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    ws = Workspace(root=tmp_path)
    c = Check(id="c", type="field_equals",
              params={"path": "out.json", "key": "status", "value": "ok"})
    assert check_one(c, _ctx(workspace=ws)) is True
    c2 = Check(id="c", type="field_equals",
               params={"path": "out.json", "key": "status", "value": "bad"})
    assert check_one(c2, _ctx(workspace=ws)) is False


def test_field_equals_missing_file_is_false(tmp_path):
    from my_agent_llms.workspace.workspace import Workspace
    ws = Workspace(root=tmp_path)
    c = Check(id="c", type="field_equals",
              params={"path": "nope.json", "key": "k", "value": "v"})
    assert check_one(c, _ctx(workspace=ws)) is False


def test_command_ok_exit_code(tmp_path):
    from my_agent_llms.workspace.workspace import Workspace
    ws = Workspace(root=tmp_path)
    ok = Check(id="c", type="command_ok", params={"cmd": "exit 0"})
    bad = Check(id="c", type="command_ok", params={"cmd": "exit 3"})
    assert check_one(ok, _ctx(workspace=ws)) is True
    assert check_one(bad, _ctx(workspace=ws)) is False


def test_tool_called_scans_trajectory():
    traj = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "done"},
    ]
    hit = Check(id="c", type="tool_called", params={"tool": "write_file"})
    miss = Check(id="c", type="tool_called", params={"tool": "read_file"})
    assert check_one(hit, _ctx(trajectory=traj)) is True
    assert check_one(miss, _ctx(trajectory=traj)) is False


def test_judge_parses_pass_fail():
    pass_llm = SimpleNamespace(invoke=lambda msgs, **kw: "PASS: 答案正确")
    fail_llm = SimpleNamespace(invoke=lambda msgs, **kw: "FAIL - 漏了单位")
    c = Check(id="c", type="judge", params={"rubric": "答案带单位"})
    assert check_one(c, _ctx(result="3 米"), llm=pass_llm) is True
    assert check_one(c, _ctx(result="3"), llm=fail_llm) is False


def test_judge_without_llm_is_false():
    c = Check(id="c", type="judge", params={"rubric": "x"})
    assert check_one(c, _ctx(result="y"), llm=None) is False


def test_unknown_type_is_false():
    c = Check(id="c", type="semantic_support", params={"claim": "x"})
    assert check_one(c, _ctx(result="x", source="x")) is False


def test_exception_in_check_is_false():
    # params 缺 "s" → KeyError → 视为未通过,不抛
    c = Check(id="c", type="string_contains", params={})
    assert check_one(c, _ctx(result="anything")) is False


def test_field_equals_non_dict_json_is_false(tmp_path):
    from my_agent_llms.workspace.workspace import Workspace
    (tmp_path / "arr.json").write_text("[1, 2, 3]", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    c = Check(id="c", type="field_equals", params={"path": "arr.json", "key": "k", "value": "v"})
    assert check_one(c, _ctx(workspace=ws)) is False


def test_checker_runner_returns_all_ids():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "x"}),
        Check(id="b", type="string_absent", params={"s": "y"}),
    ])
    runner = CheckerRunner(llm=None)
    results = runner.run(spec, _ctx(result="x present"))
    assert set(results.keys()) == {"a", "b"}
    assert results["a"] is True
    assert results["b"] is True
