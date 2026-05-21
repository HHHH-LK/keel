"""记忆条目的数据结构与元数据。"""
import math
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from my_agent_llms.core.message import MessageRole


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _estimate_tokens(text: str) -> int:
    """轻量 token 估算：中英文混合大约 2~3 char/token，取 3 偏保守。"""
    return max(1, len(text or "") // 3)


class MemoryItem(BaseModel):
    """单条记忆 —— 一个 message 或一段事实卡片。

    所有热度相关字段（access_count、last_access、pinned）在 L1 时期就先建好，
    后续 L2/L5 升降级逻辑可以直接读，不必再迁移数据结构。
    """

    id: str = Field(default_factory=_new_id)
    content: str
    role: MessageRole = "user"

    created_at: datetime = Field(default_factory=datetime.now)
    last_access: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    pinned: bool = False

    token_estimate: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _ctx) -> None:
        if not self.token_estimate:
            self.token_estimate = _estimate_tokens(self.content)

    def touch(self) -> None:
        """标记一次访问 —— 用于热度计算。"""
        self.access_count += 1
        self.last_access = datetime.now()

    def importance(
        self,
        *,
        now: Optional[datetime] = None,
        decay_tau_days: float = 7.0,
        w_access: float = 0.4,
        w_recency: float = 0.4,
        w_explicit: float = 0.2,
    ) -> float:
        """重要性 = 频次 + 时间衰减 + 显式标记，三因子加权。"""
        now = now or datetime.now()
        elapsed_days = max(0.0, (now - self.last_access).total_seconds() / 86400.0)

        access_score = math.log1p(self.access_count) / math.log1p(50)  # 归一到 [0, ~1]
        recency_score = math.exp(-elapsed_days / decay_tau_days)
        explicit_score = 1.0 if self.pinned else 0.0

        return (
            w_access * min(access_score, 1.0)
            + w_recency * recency_score
            + w_explicit * explicit_score
        )

    def to_message_dict(self) -> Dict[str, str]:
        """转 OpenAI 兼容的 message 格式，喂给 LLM。"""
        return {"role": self.role, "content": self.content}
