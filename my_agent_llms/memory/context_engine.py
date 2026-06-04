"""上下文工程编排层:在 token 预算内决定 LLM 窗口里实际放什么。

记忆层(L0–L5)负责"有什么、多重要"(库存生命周期);
本模块负责"这一次窗口里放什么"(单次组装的预算/去重/排序)。
被丢弃的内容仍在 L4/L5,下一轮可被 recall 回来。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def count_tokens(text: str) -> int:
    """估算文本 token 数。优先 tiktoken,未安装时回退 len//3 启发式。

    回退值对中英混合是保守估计;不影响"永不超预算"铁律,
    因为预算判定与兜底都用同一个 counter。
    """
    if not text:
        return 0
    try:
        import tiktoken  # 可选依赖
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)


# ── 候选片段数据结构 ────────────────────────────────────────
@dataclass
class ContextSegment:
    source: str                     # system|l0-core|l2|kg|l0-bg|recall|l1
    role: str                       # system|user|assistant
    content: str                    # 分组类只存正文行;独立类存完整文本
    priority: float                 # 统一优先级 0..1
    tokens: int                     # 实测 token
    floor: bool                     # True=保底,必进窗口
    order: int                      # 权威性序号(见 ORDER)
    seq: int = 0                    # gather 顺序,稳定排序 tiebreaker
    item_id: Optional[str] = None   # 关联 MemoryItem id(用于跨层去重)


@dataclass
class BudgetReport:
    budget: int
    used: int
    floor_tokens: int
    dropped: List[Tuple[str, int]] = field(default_factory=list)   # (source, tokens)
    deduped: List[Tuple[str, str]] = field(default_factory=list)   # (source, kept_source)


@dataclass
class BuildResult:
    messages: List[Dict[str, str]]
    report: BudgetReport


# ── 相关性打分 ──────────────────────────────────────────────
def bigram_relevance(text: str, query: str) -> float:
    """字符 bigram Jaccard。轻量、无依赖、确定性。"""
    def bigrams(s: str) -> set:
        s = s.strip()
        if len(s) < 2:
            return {s} if s else set()
        return {s[i:i + 2] for i in range(len(s) - 1)}
    a, b = bigrams(text), bigrams(query)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def make_embedding_relevance(provider) -> Callable[[str, str], float]:
    """复用 L5 embedding provider 算余弦相似度;失败回退 bigram。"""
    def fn(text: str, query: str) -> float:
        try:
            import math
            a = list(provider.embed(text))
            b = list(provider.embed(query))
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return max(0.0, dot / (na * nb))
        except Exception:
            return bigram_relevance(text, query)
    return fn
