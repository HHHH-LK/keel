"""记忆冲突检测器 —— 处理"用户偏好/事实随时间变化"的场景。

例: 用户先说"我喜欢 Java",后说"我喜欢 Python"
    → 框架检测到冲突 → 把 Java 记忆标记为被 Python 记忆取代
    → 默认 recall 只返回 Python(最新),但保留 Java 在历史里可追溯

设计原则:
- 不删除旧记忆(可能用于追溯历史 / 对比变化)
- 标记取代关系,recall/assemble 默认过滤已取代项
- 检测策略可插拔: 相似度阈值 / LLM 判断 / 用户自定义
"""
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from my_agent_llms.memory.item import MemoryItem
    from my_agent_llms.memory.manager import MemoryManager


class ConflictDetector(ABC):
    """冲突检测器接口。

    给定一条新写入的记忆,返回它取代了哪些旧记忆的 ID。
    实现可以基于相似度阈值、LLM 判断、规则匹配等。
    """

    @abstractmethod
    def find_superseded(
        self,
        new_item: "MemoryItem",
        manager: "MemoryManager",
    ) -> List[str]:
        """返回应被 new_item 取代的旧记忆 ID 列表。"""


class SimilarityConflictDetector(ConflictDetector):
    """基于语义相似度的冲突检测。

    工作流:
    1. 用 L5 search 找 top-k 最相似的旧记忆
    2. 相似度高于 threshold 的视为冲突
    3. (可选)需要相同的角色 same_role_only=True

    局限: 高相似度 ≠ 真正冲突。
    "我喜欢 Java" 和 "我用过 Java 5 年" 也很相似,但不矛盾。
    准确判断需要 LLMConflictDetector。
    """

    def __init__(
        self,
        threshold: float = 0.75,
        k: int = 3,
        same_role_only: bool = True,
    ):
        self.threshold = threshold
        self.k = k
        self.same_role_only = same_role_only

    def find_superseded(self, new_item, manager) -> List[str]:
        # 从 L5 找相似候选
        candidates = manager.semantic.search(new_item.content, k=self.k + 1)

        superseded: List[str] = []
        for item, score in candidates:
            if item.id == new_item.id:
                continue
            if not item.is_active:
                continue
            if score < self.threshold:
                continue
            if self.same_role_only and item.role != new_item.role:
                continue
            # 时间一定要老于新条目
            if item.created_at >= new_item.created_at:
                continue
            superseded.append(item.id)
        return superseded


class LLMConflictDetector(ConflictDetector):
    """用 LLM 精确判断冲突,准确率高但贵。

    给候选项 + 新项喂给 LLM,问"新项是否取代旧项"。
    适用于关键记忆(用户偏好/决定/事实)。
    """

    DEFAULT_PROMPT = (
        "判断下面的新记忆是否使旧记忆失效(用户偏好/事实变更),不要被相似但不矛盾的内容误导。\n\n"
        "新记忆:\n{new}\n\n"
        "旧记忆:\n{old}\n\n"
        "如果新记忆使旧记忆失效,只输出 YES;否则只输出 NO。"
    )

    def __init__(
        self,
        llm,
        *,
        threshold: float = 0.6,         # 先用相似度粗筛,降低 LLM 调用次数
        k: int = 3,
        same_role_only: bool = True,
        prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.threshold = threshold
        self.k = k
        self.same_role_only = same_role_only
        self.prompt = prompt or self.DEFAULT_PROMPT

    def find_superseded(self, new_item, manager) -> List[str]:
        candidates = manager.semantic.search(new_item.content, k=self.k + 1)
        superseded: List[str] = []

        for item, score in candidates:
            if item.id == new_item.id or not item.is_active:
                continue
            if score < self.threshold:
                continue
            if self.same_role_only and item.role != new_item.role:
                continue
            if item.created_at >= new_item.created_at:
                continue

            # LLM 精判
            prompt = self.prompt.format(new=new_item.content, old=item.content)
            try:
                verdict = self.llm.invoke([{"role": "user", "content": prompt}])
            except Exception as exc:
                print(f"⚠️ LLM 冲突判断失败,退回相似度阈值: {exc}")
                superseded.append(item.id)
                continue

            if verdict and "YES" in verdict.upper():
                superseded.append(item.id)

        return superseded
