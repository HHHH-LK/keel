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
