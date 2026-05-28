""" Agent 基类"""
from abc import ABC, abstractmethod
from typing import List, Optional
from .config import Config
from .hooks import AgentHook, default_hooks
from .llm import MyLLM
from .message import Message
from my_agent_llms.memory import MemoryConfig, MemoryManager


HONESTY_CONTRACT = """## 关于诚实回答的协议

你必须遵守以下规则:

1. **不要凭印象回答用户的历史信息**(偏好、邮箱、之前说过的事)。
   涉及"用户之前/上次/记得/以前"的问题,请先调用 recall 工具检索。
   如果你判断当前 context 缺少必要信息但又不便直接调工具,
   请在回答末尾追加 `[NEEDS_RECALL: 你需要查询的内容]`,
   框架会自动替你检索并让你重新作答。

2. **不知道是被允许的,编造不被允许**。
   找不到来源就说"我没有这方面的记录",不要从字面相似的内容推测。

3. **回答涉及历史事实时,必须引用来源**。
   格式: "根据你之前提到的『...原话...』,..."
   引用不出原文 = 你不知道,应该说不确定。

4. **编造信息会导致用户损失,是严重错误**。
"""


class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: MyLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        memory_config: Optional[MemoryConfig] = None,
        hooks: Optional[List[AgentHook]] = None,
        honesty_contract: bool = True,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = MemoryManager(memory_config)
        self.hooks: List[AgentHook] = hooks if hooks is not None else default_hooks()
        self.honesty_contract = honesty_contract

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

    # ── System Prompt 增强 ─────────────────────────────────
    def _apply_honesty_contract(self, base_prompt: Optional[str]) -> str:
        """把诚实契约附加到 system prompt 后。

        honesty_contract=False 时跳过。base_prompt 为 None 时返回纯契约。
        """
        base = (base_prompt or "").rstrip()
        if not self.honesty_contract:
            return base or ""
        if not base:
            return HONESTY_CONTRACT
        return f"{base}\n\n{HONESTY_CONTRACT}"

    # ── Hook 系统 ─────────────────────────────────────────
    def _run_response_hooks(
        self,
        user_input: str,
        response: str,
        messages: list,
    ) -> str:
        """跑一遍所有 hooks,前一个 hook 的输出是后一个的输入。"""
        for hook in self.hooks:
            new_response = hook.on_response(self, user_input, response, messages)
            if new_response is not None:
                response = new_response
        return response

    # ── 每轮结束的统一收尾 ────────────────────────────────
    def _finalize_turn(self, user_input: str, response: str) -> None:
        """子类 run() 末尾调用：写入历史 + 触发热度升降级。

        把"保存 user/assistant 消息"与"调 tick()"打包，避免每个 Agent
        都重复写一遍，也保证不会漏调 tick。
        """
        self.add_message(Message(user_input, "user"))
        # 空响应不写入：避免 thinking 模型耗尽 max_tokens 时污染历史，
        # 进而让下一轮 LLM 看到自己上轮"空消息"而产生奇怪的道歉行为。
        if response and response.strip():
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
