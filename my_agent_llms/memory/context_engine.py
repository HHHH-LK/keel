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

    回退值对中英混合是保守估计;ContextEngine.build 通过计数感知的预留(HEADING_RESERVE + group 段数)保证渲染后真实 token 不超预算。
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


# ── 渲染常量 ────────────────────────────────────────────────
ORDER = {
    "system": 0, "l0-core": 1, "l2": 2, "kg": 3,
    "l0-bg": 4, "recall": 5, "l1": 6,
}
HEADING_RESERVE = 256  # 给小标题 + role 开销预留的 token

_SUBSTR_MIN_LEN = 4    # 子串包含去重的最短长度阈值(避免误杀过短片段)

# 分组类来源 → 小标题前缀(渲染时归并成一条 message)
GROUP_HEADINGS = {
    "l0-core": ("## 关于用户的核心信息(与当前问题相关 + 硬约束)\n"
                "回答时优先依据此段。\n\n"),
    "l0-bg":   ("## 关于用户的背景信息(可能不直接相关)\n"
                "仅作背景理解,不要直接当作当前问题的依据。\n\n"),
    "kg":      "## 已知事实(来自知识图谱)\n",
    "recall":  "## 与当前问题相关的历史片段\n",
}
GROUP_ROLE = "system"           # 分组类一律 system role
GROUP_SOURCES = set(GROUP_HEADINGS)


class ContextEngine:
    def __init__(self, *, token_counter: Callable[[str], int] = count_tokens,
                 relevance_fn: Optional[Callable[[str, str], float]] = None,
                 dedup: bool = True) -> None:
        self.count_tokens = token_counter
        self.relevance_fn = relevance_fn or bigram_relevance
        self.dedup = dedup

    @staticmethod
    def _normalize(content: str) -> str:
        """归一化用于去重比较:去小标题行、合并空白、小写。"""
        lines = [ln for ln in content.splitlines() if not ln.lstrip().startswith("#")]
        return " ".join(" ".join(lines).split()).lower()

    def _dedup(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        if not self.dedup:
            return list(segments)
        # 权威性:order 小者优先,其次 seq 小者
        ordered = sorted(segments, key=lambda s: (s.order, s.seq))
        kept: List[ContextSegment] = []
        kept_keys: List[Tuple[str, ContextSegment]] = []
        seen_item_ids: Dict[str, ContextSegment] = {}
        for s in ordered:
            key = self._normalize(s.content)
            # item_id 去重
            if s.item_id and s.item_id in seen_item_ids:
                prev = seen_item_ids[s.item_id]
                # L1 原文优先:若新段是 l1 而已留的不是 → 用 l1 替换
                if s.source == "l1" and prev.source != "l1":
                    kept = [k for k in kept if k is not prev]
                    kept_keys = [(k, seg) for k, seg in kept_keys if seg is not prev]
                    kept.append(s)
                    kept_keys.append((key, s))
                    seen_item_ids[s.item_id] = s
                continue
            # 完全相同
            if any(key == k for k, _ in kept_keys):
                continue
            # 子串包含:被已留的更长段包含 → 跳过
            if len(key) >= _SUBSTR_MIN_LEN and any(
                key in k and len(k) > len(key) for k, _ in kept_keys
            ):
                continue
            # 子串包含:新段更长,包含已留的较短段 → 移除已留的较短段
            if len(key) >= _SUBSTR_MIN_LEN:
                superseded = [
                    seg for k, seg in kept_keys
                    if k in key and len(key) > len(k)
                ]
                for sup in superseded:
                    kept = [k for k in kept if k is not sup]
                    kept_keys = [(k, seg) for k, seg in kept_keys if seg is not sup]
            kept.append(s)
            kept_keys.append((key, s))
            if s.item_id:
                seen_item_ids[s.item_id] = s
        return kept

    def _select(self, segments: List[ContextSegment], budget: int
                ) -> Tuple[List[ContextSegment], BudgetReport]:
        chosen = [s for s in segments if s.floor]
        pool = [s for s in segments if not s.floor]
        used = sum(s.tokens for s in chosen)
        floor_tokens = used
        dropped: List[Tuple[str, int]] = []

        # 共享池:按 priority 降序,seq 升序(确定性)贪心填充
        for s in sorted(pool, key=lambda x: (-x.priority, x.seq)):
            if used + s.tokens <= budget:
                chosen.append(s)
                used += s.tokens
            else:
                dropped.append((s.source, s.tokens))

        # 兜底:保底段本身超预算时,从最不权威端(order 大、seq 大)砍,
        # 但永不砍 system(order 0)。
        if used > budget:
            chosen.sort(key=lambda s: (s.order, s.seq))
            i = len(chosen) - 1
            while used > budget and i >= 0:
                s = chosen[i]
                if s.order == 0:        # system 不砍
                    i -= 1
                    continue
                used -= s.tokens
                dropped.append((s.source, s.tokens))
                chosen.pop(i)
                i -= 1

        report = BudgetReport(budget=budget, used=used,
                              floor_tokens=floor_tokens, dropped=dropped)
        return chosen, report

    def _render(self, chosen: List[ContextSegment]) -> List[Dict[str, str]]:
        chosen = sorted(chosen, key=lambda s: (s.order, s.seq))
        messages: List[Dict[str, str]] = []
        i = 0
        n = len(chosen)
        while i < n:
            s = chosen[i]
            if s.source in GROUP_SOURCES:
                # 归并相邻同 source 段成一条 message
                lines = []
                src = s.source
                while i < n and chosen[i].source == src:
                    lines.append(chosen[i].content)
                    i += 1
                messages.append({
                    "role": GROUP_ROLE,
                    "content": GROUP_HEADINGS[src] + "\n".join(lines),
                })
            else:
                messages.append({"role": s.role, "content": s.content})
                i += 1
        return messages

    def build(self, segments: List[ContextSegment], budget: int) -> BuildResult:
        segs = self._dedup(segments)
        # 预算扣减:固定小标题开销 + 与 group 段数量成正比的并接损耗
        # (逐段 token 估算会低估并接后的真实 token,每个 group 段最多差 ~1 token;
        #  计数感知的预留把"永不超预算"从概率性变成确定性。)
        group_count = sum(1 for s in segs if s.source in GROUP_SOURCES)
        reserve = HEADING_RESERVE + group_count
        chosen, report = self._select(segs, max(0, budget - reserve))
        result = BuildResult(messages=self._render(chosen), report=report)
        report.budget = budget
        return result
