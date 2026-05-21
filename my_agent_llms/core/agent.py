""" Agent 基类"""
from abc import ABC, abstractmethod
from typing import Optional, List
from .config import Config
from .llm import MyLLM
from .message import Message
from my_agent_llms.memory import MemoryConfig, MemoryManager


class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: MyLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = MemoryManager(memory_config)

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass

    # ── 历史消息兼容 API ────────────────────────────────────
    # 子类原本直接读写 self._history，这里用 property 把它桥到 MemoryManager。
    @property
    def _history(self) -> List[Message]:
        return [
            Message(content=it.content, role=it.role, metadata=it.metadata)
            for it in self.memory.working.items()
        ]

    def add_message(self, message: Message) -> None:
        self.memory.write(
            message.content,
            role=message.role,
            metadata=message.metadata or {},
        )

    def get_history(self) -> List[Message]:
        return self._history

    def clear_history(self) -> None:
        self.memory.clear()

    def compress_history(self):
        pass

    # ── 每轮结束的统一收尾 ────────────────────────────────
    def _finalize_turn(self, user_input: str, response: str) -> None:
        """子类 run() 末尾调用：写入历史 + 触发热度升降级。

        把"保存 user/assistant 消息"与"调 tick()"打包，避免每个 Agent
        都重复写一遍，也保证不会漏调 tick。
        """
        self.add_message(Message(user_input, "user"))
        self.add_message(Message(response, "assistant"))
        self.memory.tick()

    # ── Memory 相关工具的自动注册 ──────────────────────────
    def _install_memory_tools(self, tool_registry) -> None:
        """带工具的子类在 __init__ 末尾调用：把 RecallTool 装到 registry。

        LLM 通过 registry 的 tool description 路径就能看见 recall —— 不需要
        在 system prompt 里特别说明（CalculatorTool / SearchTool 也是同一机制）。
        """
        if tool_registry is None:
            return
        # 延迟导入避免 core 反向依赖 tools
        from my_agent_llms.tools.builtin.recall import RecallTool

        if "recall" in tool_registry.list_tools():
            return
        tool_registry.register_tool(RecallTool(self.memory))

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
