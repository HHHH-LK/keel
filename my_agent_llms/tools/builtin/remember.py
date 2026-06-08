"""RememberTool：让 LLM 主动把"值得跨会话记住的事实"写进 L0。

与 recall(读) 对称 —— recall 是 page in,remember 是主动落盘。
强模型在对话中判断"这条用户信息很关键、以后还要用"时调用,
建出 source=LLM_REMEMBERED 的 L0 卡片。

注意: LLM 记的卡 user_pinned=False(不是用户显式锁定),
仍受 negate/衰减约束 —— 模型可能判断错,机械层保留纠错权。
"""
from typing import Any, Dict, List

from my_agent_llms.memory.manager import MemoryManager
from my_agent_llms.memory.playbook import L0Source
from my_agent_llms.tools.base import Tool, ToolParameter

# LLM 主动记忆的默认置信度: 比用户显式(1.0)低,留出被衰减的空间
LLM_REMEMBER_CONFIDENCE = 0.8


class RememberTool(Tool):
    """把一条关于用户的核心信息写入长期 L0 记忆。"""

    verify_exempt = True  # 写记忆是内部记账,非代码产物 → 不触发事后 verify

    def __init__(self, memory: MemoryManager):
        super().__init__(
            name="remember",
            description="把一条关于用户的关键信息(身份/偏好/硬约束/长期状态)写入长期记忆,"
                        "跨会话保留。当用户透露了以后对话还需要依据的重要事实时使用。"
                        "不要记临时闲聊或一次性内容。",
        )
        self.memory = memory

    def run(self, parameters: Dict[str, Any]) -> str:
        content = (
            parameters.get("content")
            or parameters.get("input")
            or parameters.get("text")
            or ""
        )
        content = str(content).strip()
        if not content:
            return "(remember 失败：content 为空)"

        scope = str(parameters.get("scope") or "project").strip().lower()
        if scope not in ("user", "project"):
            scope = "project"

        card = self.memory.remember(
            content,
            source=L0Source.LLM_REMEMBERED,
            confidence=LLM_REMEMBER_CONFIDENCE,
            scope=scope,
        )
        return f"已记住（{card.type.value}）：{content}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="要长期记住的一条关键信息(简洁陈述,一条一记)",
                required=True,
            ),
            ToolParameter(
                name="scope",
                type="string",
                description="作用域:'user'=关于你本人的跨项目偏好;'project'=本项目专属(缺省)。不确定填 project。",
                required=False,
                default="project",
            ),
        ]
