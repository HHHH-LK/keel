"""规划层 todo 三件套单元测试。"""
from my_agent_llms.planning.todo import TodoStore, WriteTodoTool, todo_system_message, TODO_HEADING


def test_store_set_and_render():
    s = TodoStore()
    assert s.render() == ""                       # 空 → 空串
    s.set([{"content": "读配置", "status": "completed"},
           {"content": "改端口", "status": "in_progress"}])
    out = s.render()
    assert "## 当前任务清单(进度)" in out
    assert "[x] 读配置" in out
    assert "[~] 改端口" in out


def test_store_drops_empty_content():
    s = TodoStore()
    s.set([{"content": "  ", "status": "pending"}, {"content": "干活", "status": "pending"}])
    assert len(s.items) == 1


def test_store_enforces_single_in_progress():
    """模型若一次标多个 in_progress → 只保留【第一个】,其余降回 pending。
    保证'当前执行步'唯一,高亮才有意义。"""
    s = TodoStore()
    s.set([{"content": "a", "status": "completed"},
           {"content": "b", "status": "in_progress"},
           {"content": "c", "status": "in_progress"},
           {"content": "d", "status": "pending"}])
    assert [it["status"] for it in s.items] == [
        "completed", "in_progress", "pending", "pending"]


def test_write_tool_parses_status_content():
    s = TodoStore()
    tool = WriteTodoTool(s)
    out = tool.run({"todos": ["in_progress|读配置", "pending|改端口"]})
    assert "[~] 读配置" in out and "[ ] 改端口" in out
    assert s.items[0] == {"content": "读配置", "status": "in_progress"}


def test_write_tool_bare_line_is_content():
    s = TodoStore()
    WriteTodoTool(s).run({"todos": ["没有竖线的一条"]})
    assert s.items[0]["content"] == "没有竖线的一条"
    assert s.items[0]["status"] == "pending"


def test_write_tool_side_effect_not_free():
    assert WriteTodoTool(TodoStore()).side_effect_free is False   # 写状态 → 不可并行


def test_system_message_none_when_empty():
    assert todo_system_message(TodoStore()) is None               # 空 → 不注入(短任务零开销)


def test_system_message_present_when_nonempty():
    s = TodoStore(); s.set([{"content": "干活", "status": "pending"}])
    m = todo_system_message(s)
    assert m["role"] == "system" and "干活" in m["content"]


# ── 端到端:每轮注入 ───────────────────────────────────────────
from types import SimpleNamespace
from typing import Any, Dict, List


def _text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def _make_agent(monkeypatch, responses, *, todo_store):
    from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
    from my_agent_llms.tools.registry import ToolRegistry
    from my_agent_llms.core.llm import MyLLM
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile
    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"; llm.model = "stub"; llm.client = SimpleNamespace()
    llm.temperature = 0; llm.max_tokens = 100
    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "t"; agent.llm = llm; agent.tool_registry = ToolRegistry()
    agent.max_steps = 5; agent.last_tool_call_count = 0
    agent.system_prompt = ""; agent.config = None; agent.tool_timeout = None
    agent.workspace = None; agent.enable_verify = False
    agent.spec_generator = None; agent.checker_runner = None; agent.convergence_judge = None
    agent.todo_store = todo_store
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tempfile.mkdtemp()), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp, *, task_turn=False: None
    calls = list(responses)
    captured: List[List[Dict[str, Any]]] = []

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        captured.append([dict(m) for m in messages])
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    agent._captured = captured
    return agent


def test_nonempty_todo_injected_each_turn(monkeypatch):
    s = TodoStore(); s.set([{"content": "干活", "status": "pending"}])
    agent = _make_agent(monkeypatch, [_text_response("done")], todo_store=s)
    agent.run("t")
    assert any(TODO_HEADING in m.get("content", "") for m in agent._captured[0])


def test_empty_todo_not_injected(monkeypatch):
    agent = _make_agent(monkeypatch, [_text_response("done")], todo_store=TodoStore())
    agent.run("t")
    assert not any(TODO_HEADING in m.get("content", "") for m in agent._captured[0])


# ── 结构闸门:禁止"边执行改动边预先打勾" ─────────────────────────
import json
from my_agent_llms.tools.base import Tool
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent


class _FakeEdit(Tool):
    """假的改动类工具:side_effect_free=False,代表 Edit/Bash 这类有副作用动作。"""
    def __init__(self):
        super().__init__("edit", "fake mutating tool")
        self.side_effect_free = False

    def run(self, parameters):
        return "edited"

    def get_parameters(self):
        return []


def _gate_agent(store):
    a = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    reg = ToolRegistry()
    reg.register_tool(WriteTodoTool(store))
    reg.register_tool(_FakeEdit())
    a.tool_registry = reg
    a.todo_store = store
    a.tool_timeout = None
    a._turn_mutated = False
    return a


def _tc(call_id, name, args):
    from types import SimpleNamespace
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _run_batch(agent, calls):
    msgs = []
    agent._execute_tool_calls(
        calls, msgs,
        on_tool_call=None, on_permission_request=None, on_tool_result=None)
    return msgs


def test_gate_vetoes_completion_in_same_turn_as_work():
    """同一轮既 edit 又把步骤标 completed → 否决,store 不被预先打勾。"""
    s = TodoStore()
    s.set([{"content": "step1", "status": "in_progress"},
           {"content": "step2", "status": "pending"}])
    agent = _gate_agent(s)
    msgs = _run_batch(agent, [
        _tc("1", "edit", {"path": "a"}),
        _tc("2", "write_todo", {"todos": ["completed|step1", "in_progress|step2"]}),
    ])
    assert s.items[0]["status"] == "in_progress"          # 没被预先打勾
    todo_msg = next(m for m in msgs if m.get("tool_call_id") == "2")
    assert "⛔" in todo_msg["content"]                     # 回喂了阻止说明


def test_gate_allows_completion_in_todo_only_turn():
    """只调 write_todo(没改动工具)→ completed 正常生效。"""
    s = TodoStore()
    s.set([{"content": "step1", "status": "in_progress"}])
    agent = _gate_agent(s)
    _run_batch(agent, [_tc("1", "write_todo", {"todos": ["completed|step1"]})])
    assert s.items[0]["status"] == "completed"


def test_gate_allows_in_progress_alongside_work():
    """edit + 把下一步标 in_progress(无新增 completed)→ 不否决。"""
    s = TodoStore()
    s.set([{"content": "step1", "status": "completed"},
           {"content": "step2", "status": "pending"}])
    agent = _gate_agent(s)
    _run_batch(agent, [
        _tc("1", "edit", {}),
        # step1 原本就 completed(原样带回,非新增)→ 不触发;step2 转 in_progress
        _tc("2", "write_todo", {"todos": ["completed|step1", "in_progress|step2"]}),
    ])
    assert s.items[0]["status"] == "completed"
    assert s.items[1]["status"] == "in_progress"


# ── 容错:模型常把 todos 传成 dict / JSON 字符串而非 'status|内容' ──
def test_parse_accepts_dict_items():
    from my_agent_llms.planning.todo import parse_todo_lines
    out = parse_todo_lines([
        {"status": "in_progress", "content": "替换中英混杂"},
        {"status": "pending", "content": "加配置表"},
    ])
    assert out == [
        {"content": "替换中英混杂", "status": "in_progress"},
        {"content": "加配置表", "status": "pending"},
    ]


def test_parse_dict_bad_status_falls_to_pending():
    from my_agent_llms.planning.todo import parse_todo_lines
    out = parse_todo_lines([{"status": "doing", "content": "x"}])
    assert out == [{"content": "x", "status": "pending"}]


def test_parse_accepts_json_string_items():
    from my_agent_llms.planning.todo import parse_todo_lines
    out = parse_todo_lines(['{"status": "completed", "content": "跑测试"}'])
    assert out == [{"content": "跑测试", "status": "completed"}]


def test_parse_still_handles_pipe_strings():
    from my_agent_llms.planning.todo import parse_todo_lines
    out = parse_todo_lines(["in_progress|读配置", "改端口"])
    assert out == [
        {"content": "读配置", "status": "in_progress"},
        {"content": "改端口", "status": "pending"},
    ]
