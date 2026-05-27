"""端到端:让真实 LLM 走完 attach → edit → apply → export → apply 完整流程。

跳过条件: 未设置 MY_LLM_API_KEY 环境变量。
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("MY_LLM_API_KEY"),
    reason="需要 MY_LLM_API_KEY 环境变量才能跑 (调用真实 LLM)",
)


def test_full_attach_edit_export_flow(tmp_path):
    from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
    from my_agent_llms.core.llm import MyLLM
    from my_agent_llms.workspace import Workspace
    from my_agent_llms.tools.registry import ToolRegistry
    from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
    from my_agent_llms.tools.builtin.read_file import ReadFile
    from my_agent_llms.tools.builtin.edit_file import EditFile
    from my_agent_llms.tools.builtin.write_file import WriteFile
    from my_agent_llms.tools.builtin.list_dir import ListDir
    from my_agent_llms.tools.builtin.attach_file import AttachFile
    from my_agent_llms.tools.builtin.export_file import ExportFile

    # 准备一个 sandbox 和一份"用户的真实文件"
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    user_file = tmp_path / "user_docs" / "hello.md"
    user_file.parent.mkdir()
    user_file.write_text("Hello, 2024\n")

    ws = Workspace(sandbox)
    store = PendingEditStore()
    registry = ToolRegistry()
    for t in [
        ReadFile(ws),
        EditFile(ws, store),
        WriteFile(ws, store),
        ListDir(ws),
        AttachFile(ws),
        ExportFile(ws, store),
    ]:
        registry.register_tool(t)

    llm = MyLLM(
        api_key=os.environ["MY_LLM_API_KEY"],
        base_url=os.getenv("MY_LLM_BASE_URL"),
        model=os.getenv("MY_LLM_MODEL", "deepseek-chat"),
    )
    agent = MyFunctionCallAgent(
        name="tester",
        llm=llm,
        tool_registry=registry,
        system_prompt=(
            "你有 sandbox 文件工具。流程:\n"
            "1. 外部路径先 AttachFile\n"
            "2. EditFile / WriteFile 是两步: 提案返回 pending_id+diff,用户说'确认'后再 action=apply\n"
            "3. 想写回原位置用 ExportFile (同样两步)\n"
        ),
        max_steps=15,
    )

    # 模拟一次完整对话: 用户连续 2 条消息
    agent.chat(f"把 {user_file} 里的 2024 改成 2025")
    # 上一步 LLM 应该 attach + read + propose edit; 我们模拟"确认"
    agent.chat("确认。然后写回原位置,并再次确认。")

    # 最终断言: 原文件内容已被更新
    assert user_file.read_text() == "Hello, 2025\n", (
        f"原文件未被正确更新,实际: {user_file.read_text()!r}"
    )
