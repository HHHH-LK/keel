"""冷回路 reconcile —— 后台定期巡检,propose→confirm→apply。

定位(对照热路径):热路径写入即时、确定、保守;冷回路慢、可非确定、有确认网。
所以语义判断(CORRECT 自动判定、语义冲突)放这里,只产"提议",规则/置信门确认后
才调确定性的 store 操作处置(correct_relation / supersede_relation / remove_pending)。

当前实现:
- 确定性:pending GC(陈旧 + 低证据的待确认事实清掉)
- LLM(可选,搭骨架待通电):CORRECT 自动判定、语义冲突提议
"""
import json
from datetime import datetime
from typing import Dict, List, Optional


_CONFLICT_PROMPT = """下面是关于同一主语的若干事实。找出其中"语义互斥、不可能同时为真"的成对事实
(例如"喜欢咖啡"和"讨厌咖啡")。注意:能同时成立的不算冲突(如"会Java"和"会Python")。

事实(每行 "id: 描述"):
{facts}

输出 JSON 数组,每个冲突对一项:
{{"fact_a_id":"...","fact_b_id":"...","confidence":0~1,"reason":"..."}}
没有冲突就输出 []。只输出 JSON,不要其他文字。"""


def _parse_json_list(raw: str) -> list:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


class KGReconciler:
    """KG 冷回路维护器。store 必传;llm 可选(无则只跑确定性 pending GC)。"""

    def __init__(
        self,
        store,
        llm=None,
        *,
        pending_ttl_seconds: int = 7 * 24 * 3600,   # 默认 7 天没再出现就清
        pending_promote_hits: int = 2,
        conflict_confidence_threshold: float = 0.7,
    ):
        self.store = store
        self.llm = llm
        self.pending_ttl_seconds = pending_ttl_seconds
        self.pending_promote_hits = pending_promote_hits
        self.conflict_confidence_threshold = conflict_confidence_threshold

    def reconcile(self, now: Optional[datetime] = None) -> Dict[str, list]:
        """跑一轮冷回路。返回这轮做了什么(可观测)。"""
        now = now or datetime.now()
        report: Dict[str, list] = {
            "dropped_pending": self._gc_pending(now),
        }
        # LLM 部分:有 llm 才跑(语义冲突提议;CORRECT 自动判定同模式,后续填)
        if self.llm is not None:
            report["conflicts_resolved"] = self._reconcile_semantic_conflicts(now)
        return report

    # ── 语义冲突:propose → confirm → apply ──────────────────
    def _reconcile_semantic_conflicts(self, now: datetime) -> List[str]:
        resolved: List[str] = []
        seen: set = set()
        active = self.store.all_relations(only_active=True)
        by_subject: Dict[str, list] = {}
        for r in active:
            by_subject.setdefault(r.subject_id, []).append(r)

        for rels in by_subject.values():
            if len(rels) < 2:
                continue
            by_id = {r.id: r for r in rels}
            for prop in self._propose_conflicts(rels):
                # confirm:置信门
                if float(prop.get("confidence", 0)) < self.conflict_confidence_threshold:
                    continue
                a = by_id.get(prop.get("fact_a_id"))
                b = by_id.get(prop.get("fact_b_id"))
                if a is None or b is None or a.id == b.id:
                    continue
                loser, winner = self._pick_loser(a, b)
                if loser.id in seen:
                    continue
                seen.add(loser.id)
                # apply:关 loser 的事件区间到 winner 的事件时间,expired_at=now(T')
                self.store.supersede_relation(
                    loser.id, winner.valid_from or now, expired_at=now,
                )
                self.store.log_audit(
                    "supersede", "relation", loser.id,
                    reason=f"冷回路语义冲突: {prop.get('reason', '')}", at_time=now,
                )
                resolved.append(loser.id)
        return resolved

    def _propose_conflicts(self, rels: list) -> list:
        """LLM 提议冲突对。失败/无 llm 返回 []。"""
        facts = "\n".join(f"{r.id}: {self.store.relation_to_nl(r)}" for r in rels)
        try:
            raw = self.llm.invoke([{"role": "user", "content": _CONFLICT_PROMPT.format(facts=facts)}])
        except Exception as exc:
            print(f"⚠️ 冷回路语义冲突 LLM 调用失败: {exc}")
            return []
        return _parse_json_list(raw)

    @staticmethod
    def _pick_loser(a, b):
        """(loser, winner):权威低者输;同权威则事件时间更旧者输(被更新的取代)。"""
        if a.authority != b.authority:
            return (a, b) if a.authority < b.authority else (b, a)
        av = a.valid_from or datetime.min
        bv = b.valid_from or datetime.min
        return (a, b) if av < bv else (b, a)

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
