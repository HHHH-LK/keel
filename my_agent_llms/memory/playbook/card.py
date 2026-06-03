"""L0 卡片数据结构。

PlaybookCard 是 L0 的核心抽象:跨会话保留的"关于用户的核心信息"。
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class L0Type(str, Enum):
    """卡片类型,决定 KG supersede 时的反应规则。"""
    HARD_CONSTRAINT = "hard_constraint"   # 过敏 / 禁忌 / 健康 → 不衰减
    IDENTITY = "identity"                  # 姓名 / 职业 / 亲属 → 慢衰减
    PREFERENCE = "preference"              # 喜好 / 习惯 → 中等衰减
    STATE = "state"                        # 正在做 / 最近 → 快衰减


class L0Lifecycle(str, Enum):
    ACTIVE = "active"           # 在 L0 里,会被注入
    ARCHIVED = "archived"       # 撤下,30 天可激活回 L0
    FORGOTTEN = "forgotten"     # 永久不复活(用户 /forget 或 archive 超期)


class L0Source(str, Enum):
    USER_EXPLICIT = "user_explicit"     # /remember 命令
    SEED_PROMOTED = "seed_promoted"     # 写入瞬间种子分(关键词启发式)晋升
    KG_PROMOTED = "kg_promoted"         # KG detector 抽出事实
    L1_GRADUATED = "l1_graduated"       # L1 长期 pinned 晋升
    LLM_REMEMBERED = "llm_remembered"   # LLM 主动调 remember 工具(强模型)


# 不同 type 在 KG supersede 时的 confidence 扣分幅度
NEGATE_SEVERITY = {
    L0Type.HARD_CONSTRAINT: 0.15,   # 硬约束类降幅最小,需用户显式 /forget
    L0Type.IDENTITY: 0.30,           # 身份变化通常缓慢
    L0Type.PREFERENCE: 0.40,         # 偏好可能改变
    L0Type.STATE: 0.60,              # 状态变化最快(项目结束/学习内容变)
}


class PlaybookCard(BaseModel):
    """L0 持久化卡片:跨会话核心信息。"""

    id: str = Field(default_factory=_new_id)
    content: str

    # ── 必要 tag ──────────────────────────────
    type: L0Type
    lifecycle: L0Lifecycle = L0Lifecycle.ACTIVE

    # ── 数值 ───────────────────────────────────
    confidence: float = 0.8

    # ── 时间戳 ─────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.now)
    last_refresh: datetime = Field(default_factory=datetime.now)
    last_negation: Optional[datetime] = None

    # ── 来源追溯 ───────────────────────────────
    source: L0Source
    source_ref: Optional[str] = None     # 关联 KG 三元组 id / L1 item id

    # ── 用户操作 ───────────────────────────────
    user_pinned: bool = False            # /pin 锁定,永不衰减

    # ── 元数据 ─────────────────────────────────
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── 状态判断 ───────────────────────────────
    def is_active(self) -> bool:
        return self.lifecycle == L0Lifecycle.ACTIVE

    def is_hard_constraint(self) -> bool:
        return self.type == L0Type.HARD_CONSTRAINT

    # ── 演化操作 ───────────────────────────────
    def refresh(self, boost: float = 0.02) -> None:
        """被注入即续命:小幅提升 confidence + 更新时间戳。

        user_pinned 项不动 confidence(已经锁定为 1.0)。
        """
        if not self.user_pinned:
            self.confidence = min(1.0, self.confidence + boost)
        self.last_refresh = datetime.now()

    def negate(self, severity: Optional[float] = None) -> None:
        """被否定/KG supersede:按 type 不同幅度降置信度。

        user_pinned 项不被否定影响。
        severity=None 时按 type 默认值,否则用传入值。
        """
        if self.user_pinned:
            return
        if severity is None:
            severity = NEGATE_SEVERITY.get(self.type, 0.4)
        self.confidence = max(0.0, self.confidence - severity)
        self.last_negation = datetime.now()

    def pin(self) -> None:
        """用户显式锁定。"""
        self.user_pinned = True
        self.confidence = 1.0

    def forget(self) -> None:
        """用户显式忘记。"""
        self.lifecycle = L0Lifecycle.FORGOTTEN
        self.confidence = 0.0

    def archive(self) -> None:
        """confidence 跌破阈值时撤下(可激活回)。"""
        self.lifecycle = L0Lifecycle.ARCHIVED

    def reactivate(self) -> None:
        """archived 项被重新激活。"""
        self.lifecycle = L0Lifecycle.ACTIVE
        self.confidence = max(self.confidence, 0.5)
        self.last_refresh = datetime.now()

    def should_archive(self, threshold: float = 0.3) -> bool:
        """是否应该撤下到 archived(不删除)。"""
        if self.user_pinned:
            return False
        if self.is_hard_constraint():
            return False  # 硬约束需用户显式 /forget
        return self.confidence < threshold


def classify_content_type(content: str) -> L0Type:
    """启发式分类消息内容属于哪类 L0 卡片。

    复用 seed_score 的关键词清单,按优先级:
    hard_constraint > identity > preference > state(默认)
    """
    from my_agent_llms.memory.seed_score import CATEGORY_KEYWORDS

    if any(kw in content for kw in CATEGORY_KEYWORDS["hard_constraint"]["keywords"]):
        return L0Type.HARD_CONSTRAINT
    if any(kw in content for kw in CATEGORY_KEYWORDS["identity"]["keywords"]):
        return L0Type.IDENTITY
    if any(kw in content for kw in CATEGORY_KEYWORDS["preference"]["keywords"]):
        return L0Type.PREFERENCE
    return L0Type.STATE
