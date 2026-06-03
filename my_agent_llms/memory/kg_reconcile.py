"""冷回路 reconcile —— 后台定期巡检,propose→confirm→apply。

定位(对照热路径):热路径写入即时、确定、保守;冷回路慢、可非确定、有确认网。
所以语义判断(CORRECT 自动判定、语义冲突)放这里,只产"提议",规则/置信门确认后
才调确定性的 store 操作处置(correct_relation / supersede_relation / remove_pending)。

当前实现:
- 确定性:pending GC(陈旧 + 低证据的待确认事实清掉)
- LLM(可选,搭骨架待通电):CORRECT 自动判定、语义冲突提议
"""
from datetime import datetime
from typing import Dict, List, Optional


class KGReconciler:
    """KG 冷回路维护器。store 必传;llm 可选(无则只跑确定性 pending GC)。"""

    def __init__(
        self,
        store,
        llm=None,
        *,
        pending_ttl_seconds: int = 7 * 24 * 3600,   # 默认 7 天没再出现就清
        pending_promote_hits: int = 2,
    ):
        self.store = store
        self.llm = llm
        self.pending_ttl_seconds = pending_ttl_seconds
        self.pending_promote_hits = pending_promote_hits

    def reconcile(self, now: Optional[datetime] = None) -> Dict[str, list]:
        """跑一轮冷回路。返回这轮做了什么(可观测)。"""
        now = now or datetime.now()
        report: Dict[str, list] = {
            "dropped_pending": self._gc_pending(now),
        }
        # LLM 部分:有 llm 才跑(CORRECT 自动判定 / 语义冲突),骨架待后续填
        return report

    def _gc_pending(self, now: datetime) -> List[str]:
        """清掉陈旧 + 低证据的 pending:既没攒够证据晋升,又很久没再出现。

        够格晋升(hit >= promote_hits)的不丢 —— 它只是还没被热路径晋升,不是垃圾。
        """
        dropped: List[str] = []
        for entry in self.store.pending_entries():
            if entry["hit_count"] >= self.pending_promote_hits:
                continue
            last_seen = datetime.fromisoformat(entry["last_seen_at"])
            if (now - last_seen).total_seconds() > self.pending_ttl_seconds:
                self.store.remove_pending_key(entry["triple_key"])
                dropped.append(entry["triple_key"])
        return dropped
