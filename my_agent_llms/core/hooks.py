"""Agent 响应 Hook 系统。

设计动机:让模型主动声明状态、框架机械响应,职责分离。
- 模型擅长语义判断("我缺什么信息") → 输出明确信号
- 框架擅长稳定执行(检索 + 重调) → 拦截信号并动作

Hook 之间可堆叠,前一个的输出是后一个的输入。
"""
import re
from abc import ABC
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from my_agent_llms.core.agent import Agent


class AgentHook(ABC):
    """Hook 接口 —— 在 Agent 响应阶段切入,可观察、可干预、可替换。"""

    def on_response(
        self,
        agent: "Agent",
        user_input: str,
        response: str,
        messages: list,
    ) -> Optional[str]:
        """检查模型响应。

        Args:
            agent: 当前 Agent 实例(可访问 memory / llm / tool_registry)
            user_input: 本轮用户输入
            response: LLM 输出的原始响应
            messages: 本轮拼装好的完整 messages(给 LLM 的)

        Returns:
            None: 不干预,response 直接进下一步
            str:  替换 response,继续走后续 hooks
        """
        return None


# ──────────────────────────────────────────────────────────────
# 内置 Hook 1: 模型主动声明缺记忆 → 框架自动 recall
# ──────────────────────────────────────────────────────────────

class MemoryRecallHook(AgentHook):
    """检测 [NEEDS_RECALL: query] marker,自动 recall 并重答。

    工作流:
    1. 扫响应里的 marker
    2. 提取 query,调 memory.recall
    3. 把结果注入 system message,重新调 LLM
    4. 命中无果时,把 marker 替换成"已检索无果"提示

    仅在 marker 存在时触发,不存在时零成本通过。
    """

    PATTERN = re.compile(r"\[NEEDS_RECALL:\s*([^\]]+)\]")

    def __init__(self, k: int = 3):
        self.k = k

    def on_response(self, agent, user_input, response, messages):
        m = self.PATTERN.search(response)
        if m is None:
            return None

        query = m.group(1).strip()
        if not query:
            return None

        hits = agent.memory.recall(query, k=self.k)

        if not hits:
            return self.PATTERN.sub(
                "(框架已检索长期记忆,未找到相关内容)",
                response,
            )

        recall_text = "\n".join(
            f"- [{it.role}] {it.content}" for it, _ in hits
        )
        clean_response = self.PATTERN.sub("", response).strip()
        messages_v2 = list(messages) + [
            {
                "role": "assistant",
                "content": clean_response or "(待补充)",
            },
            {
                "role": "system",
                "content": (
                    f"框架已替你检索长期记忆,以下是与「{query}」相关的内容:\n\n"
                    f"{recall_text}\n\n"
                    f"请基于这些信息重新完整回答用户的原始问题。"
                    f"**不要再输出 [NEEDS_RECALL:...] 标记**。"
                ),
            },
        ]
        try:
            return agent.llm.invoke(messages_v2)
        except Exception as exc:
            print(f"⚠️ MemoryRecallHook 重答失败,保留原响应: {exc}")
            return self.PATTERN.sub("", response)


# ──────────────────────────────────────────────────────────────
# 内置 Hook 2: 模型没主动声明但表达了不确定 → 兜底触发
# ──────────────────────────────────────────────────────────────

class UncertaintyFallbackHook(AgentHook):
    """模型没用 marker,但响应里有不确定表达 + 用户问的是历史 → 兜底 recall。

    作为 MemoryRecallHook 的安全网:模型不听话时也能补救一次。
    """

    UNCERTAINTY_PATTERNS = [
        "不记得", "不确定", "不清楚", "印象中", "好像", "可能",
        "我不知道", "没有这个信息", "需要更多上下文",
        "I don't recall", "I'm not sure", "I don't remember",
    ]
    HISTORICAL_HINTS = [
        "之前", "上次", "记得", "你说过", "我以前", "以前",
        "earlier", "previously", "before",
    ]

    def __init__(self, k: int = 3):
        self.k = k

    def on_response(self, agent, user_input, response, messages):
        has_uncertainty = any(p in response for p in self.UNCERTAINTY_PATTERNS)
        user_asks_history = any(h in user_input for h in self.HISTORICAL_HINTS)

        if not (has_uncertainty and user_asks_history):
            return None

        hits = agent.memory.recall(user_input, k=self.k)
        if not hits:
            return None

        recall_text = "\n".join(
            f"- [{it.role}] {it.content}" for it, _ in hits
        )
        messages_v2 = list(messages) + [
            {"role": "assistant", "content": response},
            {
                "role": "system",
                "content": (
                    f"框架注意到你回答里有不确定信号,已替你检索长期记忆,"
                    f"以下是与用户问题相关的内容:\n\n{recall_text}\n\n"
                    f"请基于这些信息修正你的回答。"
                ),
            },
        ]
        try:
            return agent.llm.invoke(messages_v2)
        except Exception as exc:
            print(f"⚠️ UncertaintyFallbackHook 重答失败,保留原响应: {exc}")
            return None


def default_hooks() -> List[AgentHook]:
    """默认 hook 组合:精准 marker 检测 + 不确定词兜底。"""
    return [MemoryRecallHook(), UncertaintyFallbackHook()]
