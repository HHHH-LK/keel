"""build_agent 注册了 Grep/Glob,且 system prompt 含探索引导。"""
import inspect
from my_agent_llms.cli import app


def test_search_tools_imported_in_build_agent():
    src = inspect.getsource(app.build_agent)
    assert "GrepTool" in src and "GlobTool" in src
    assert "GrepTool(ws)" in src and "GlobTool(ws)" in src


def test_system_prompt_has_search_guidance():
    src = inspect.getsource(app.build_agent)
    assert "Glob" in src and "Grep" in src
    assert "精读" in src or "定位" in src
