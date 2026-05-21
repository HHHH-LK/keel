"""RecallTool：让 LLM 主动检索长期记忆。"""
from typing import Any, Dict, List

from my_agent_llms.memory.manager import MemoryManager
from my_agent_llms.tools.base import Tool, ToolParameter


class RecallTool(Tool):
    """从长期记忆中按语义检索相关内容。

    Agent 在 system prompt 中需要告知 LLM：「你有长期记忆，可以用 recall 工具
    查询历史」—— 否则 LLM 不会主动调用（OS 类比：进程不知道有虚拟内存就不会触发 page fault）。
    """

    def __init__(self, memory: MemoryManager, top_k: int = 5):
        super().__init__(
            name="recall",
            description="检索长期记忆中与查询语义相关的历史内容。当需要回忆用户之前提到过、"
                        "但当前对话窗口中看不到的信息时使用。",
        )
        self.memory = memory
        self.top_k = top_k

    def run(self, parameters: Dict[str, Any]) -> str:
        query = (
            parameters.get("query")
            or parameters.get("input")
            or parameters.get("q")
            or ""
        )
        query = str(query).strip()
        if not query:
            return "(recall 失败：query 为空)"

        k = int(parameters.get("k", self.top_k) or self.top_k)
        hits = self.memory.recall(query, k=k)
        if not hits:
            return f"(未在长期记忆中找到与「{query}」相关的内容)"

        lines: List[str] = [f"找到 {len(hits)} 条相关记忆："]
        for idx, (item, score) in enumerate(hits, 1):
            preview = item.content if len(item.content) <= 200 else item.content[:200] + "..."
            lines.append(f"{idx}. [{item.role} · score={score:.2f}] {preview}")
        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="检索语义关键词或问题描述",
                required=True,
            ),
            ToolParameter(
                name="k",
                type="integer",
                description="返回最相关的前 k 条",
                required=False,
                default=5,
            ),
        ]
