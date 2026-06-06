"""build_agent 把 workspace 注入 agent 并默认开启 verify(工具门控保证闲聊不受影响)。"""
import inspect
from my_agent_llms.cli import app


def test_build_agent_injects_workspace():
    src = inspect.getsource(app.build_agent)
    assert "workspace=ws" in src


def test_build_agent_enables_verify():
    src = inspect.getsource(app.build_agent)
    assert "enable_verify=True" in src


def test_build_agent_wires_todo():
    src = inspect.getsource(app.build_agent)
    assert "WriteTodoTool" in src
    assert "todo_store=" in src


def test_build_agent_prompt_mentions_todo():
    src = inspect.getsource(app.build_agent)
    assert "write_todo" in src      # system prompt 引导里提到
