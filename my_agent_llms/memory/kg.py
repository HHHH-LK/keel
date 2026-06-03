"""轻量级知识图谱(KG)冲突检测 —— extreme 强度档。

核心思想:
- 不存"文本",存"实体-关系-时态" (subject, predicate, object, scope, valid_until)
- 新关系跟旧关系冲突时,旧关系的 valid_until 设为 now(取代但不删)
- 不同 scope 的关系不冲突 → 自然处理"上下文型冲突"
- 时态查询 (valid_until > now) 自动过滤历史

实现选择: 用 SQLite 模拟图,无需 Neo4j。
- 适合:学习/中小规模(万级关系)
- 不适合:生产级大规模,请用 Zep / Graphiti / Neo4j

依赖: LLM 用于实体+关系抽取(extract_relations)。
"""
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from my_agent_llms.memory.conflict import ConflictDetector
from my_agent_llms.memory.embeddings import _tokenize

if TYPE_CHECKING:
    from my_agent_llms.memory.item import MemoryItem
    from my_agent_llms.memory.manager import MemoryManager


# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class Entity:
    id: str
    type: str        # PERSON / TECH / DOMAIN / ITEM / PLACE / TIME / ...
    name: str


@dataclass
class Relation:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    scope: str = ""
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    source_item_id: Optional[str] = None
    confidence: float = 1.0


# ── 混合检索工具:RRF + 余弦 ────────────────────────────────

def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion —— 把多个排序榜单融合成一个。

    每个榜单是一组 id(按相关性降序)。某 id 在某榜单排第 r 位(0-based),
    贡献 1/(k + r) 分;跨榜单累加。同时进多个榜单前列的项得分最高。

    k=60 是 RRF 论文的经验默认值,削弱单榜单头部的绝对优势,
    让"多榜单共识"压过"单榜单极端高分"。

    返回 [(id, fused_score), ...] 按融合分降序。
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── SQLite 存储 ────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS kg_entities (
    id   TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(type, name)
);

CREATE TABLE IF NOT EXISTS kg_relations (
    id              TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object_id       TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT '',
    valid_from      TEXT NOT NULL,
    valid_until     TEXT,
    source_item_id  TEXT,
    confidence      REAL DEFAULT 1.0,
    FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
    FOREIGN KEY (object_id)  REFERENCES kg_entities(id)
);

CREATE INDEX IF NOT EXISTS idx_rel_spo
    ON kg_relations(subject_id, predicate, scope);
CREATE INDEX IF NOT EXISTS idx_rel_valid
    ON kg_relations(valid_until);
"""


class KGStore:
    """轻量级知识图谱存储。

    path=None 时用内存数据库(测试/临时用)。
    传 Path 时持久化到 SQLite,重启可恢复。
    """

    def __init__(self, path: Optional[Path] = None):
        self._db_path = path
        if path is None:
            self.conn = sqlite3.connect(":memory:")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self.conn.commit()

    # ── 实体 ────────────────────────────────────────────────
    def get_or_create_entity(self, type_: str, name: str) -> str:
        row = self.conn.execute(
            "SELECT id FROM kg_entities WHERE type=? AND name=?",
            (type_, name),
        ).fetchone()
        if row:
            return row["id"]
        eid = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO kg_entities(id, type, name) VALUES (?, ?, ?)",
            (eid, type_, name),
        )
        self.conn.commit()
        return eid

    def get_entity(self, eid: str) -> Optional[Entity]:
        row = self.conn.execute(
            "SELECT * FROM kg_entities WHERE id=?", (eid,)
        ).fetchone()
        return Entity(id=row["id"], type=row["type"], name=row["name"]) if row else None

    # ── 关系 ────────────────────────────────────────────────
    def add_relation(self, rel: Relation) -> None:
        self.conn.execute(
            "INSERT INTO kg_relations VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rel.id,
                rel.subject_id,
                rel.predicate,
                rel.object_id,
                rel.scope or "",
                (rel.valid_from or datetime.now()).isoformat(),
                rel.valid_until.isoformat() if rel.valid_until else None,
                rel.source_item_id,
                rel.confidence,
            ),
        )
        self.conn.commit()

    def find_conflicts(
        self,
        subject_id: str,
        predicate: str,
        scope: str,
        exclude_object_id: str,
        at_time: datetime,
    ) -> List[Relation]:
        """查找同 (subject, predicate, scope) 但不同 object 的有效关系。"""
        rows = self.conn.execute(
            """
            SELECT * FROM kg_relations
            WHERE subject_id = ?
              AND predicate = ?
              AND scope = ?
              AND object_id != ?
              AND valid_from <= ?
              AND (valid_until IS NULL OR valid_until > ?)
            """,
            (
                subject_id, predicate, scope, exclude_object_id,
                at_time.isoformat(), at_time.isoformat(),
            ),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def supersede_relation(self, rel_id: str, at_time: datetime) -> None:
        """给一个关系设 valid_until = at_time(让它失效)。"""
        self.conn.execute(
            "UPDATE kg_relations SET valid_until=? WHERE id=?",
            (at_time.isoformat(), rel_id),
        )
        self.conn.commit()

    def find_active_relations_for_entity(
        self,
        entity_name: str,
        at_time: Optional[datetime] = None,
    ) -> List[Relation]:
        """找跟某个实体(按名字) 相关、当前有效的所有关系(作为 subject 或 object)。"""
        at_time = at_time or datetime.now()
        rows = self.conn.execute(
            """
            SELECT r.* FROM kg_relations r
            JOIN kg_entities e ON (r.subject_id = e.id OR r.object_id = e.id)
            WHERE e.name = ?
              AND r.valid_from <= ?
              AND (r.valid_until IS NULL OR r.valid_until > ?)
            """,
            (entity_name, at_time.isoformat(), at_time.isoformat()),
        ).fetchall()
        seen = set()
        out: List[Relation] = []
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(self._row_to_relation(r))
        return out

    def relation_to_nl(self, rel: Relation) -> str:
        """关系 → 自然语言描述,用于喂给 LLM。"""
        subj = self.get_entity(rel.subject_id)
        obj = self.get_entity(rel.object_id)
        if subj is None or obj is None:
            return ""
        scope_part = f" (场景: {rel.scope})" if rel.scope else ""
        return f"{subj.name} {rel.predicate} {obj.name}{scope_part}"

    # ── 可视化导出 ──────────────────────────────────────────
    def to_mermaid(self, include_inactive: bool = True) -> str:
        """导出为 Mermaid 流程图,可直接贴到 Markdown / mermaid.live。

        - 活跃关系用实线 + 标签
        - 失效关系用虚线 + [失效] 标签(可关掉)
        """
        lines = ["graph LR"]
        entity_rows = self.conn.execute("SELECT * FROM kg_entities").fetchall()
        entity_label: dict = {}
        for row in entity_rows:
            safe_id = f"e_{row['id']}"
            label = f"{row['name']}<br/><i>{row['type']}</i>"
            lines.append(f'    {safe_id}["{label}"]')
            entity_label[row["id"]] = safe_id

        for rel in self.all_relations():
            if rel.valid_until is not None and not include_inactive:
                continue
            subj = entity_label.get(rel.subject_id)
            obj = entity_label.get(rel.object_id)
            if subj is None or obj is None:
                continue

            arrow = "-->" if rel.valid_until is None else "-.->"
            label_parts = [rel.predicate]
            if rel.scope:
                label_parts.append(f"@{rel.scope}")
            if rel.valid_until is not None:
                label_parts.append("[失效]")
            label = " ".join(label_parts)
            lines.append(f'    {subj} {arrow}|"{label}"| {obj}')
        return "\n".join(lines)

    def to_dot(self, include_inactive: bool = True) -> str:
        """导出为 GraphViz DOT 格式,可用 `dot -Tpng` 渲染。"""
        lines = ["digraph KG {", '    rankdir=LR;', '    node [shape=box];']
        entity_rows = self.conn.execute("SELECT * FROM kg_entities").fetchall()
        for row in entity_rows:
            lines.append(
                f'    "{row["id"]}" [label="{row["name"]}\\n[{row["type"]}]"];'
            )

        for rel in self.all_relations():
            if rel.valid_until is not None and not include_inactive:
                continue
            style = "solid" if rel.valid_until is None else "dashed"
            color = "black" if rel.valid_until is None else "gray"
            label_parts = [rel.predicate]
            if rel.scope:
                label_parts.append(f"@{rel.scope}")
            if rel.valid_until is not None:
                label_parts.append("[失效]")
            label = " ".join(label_parts)
            lines.append(
                f'    "{rel.subject_id}" -> "{rel.object_id}" '
                f'[label="{label}", style={style}, color={color}];'
            )

        lines.append("}")
        return "\n".join(lines)

    def all_relations(self, only_active: bool = False, at_time: Optional[datetime] = None) -> List[Relation]:
        if only_active:
            at_time = at_time or datetime.now()
            rows = self.conn.execute(
                """SELECT * FROM kg_relations
                   WHERE valid_from <= ?
                   AND (valid_until IS NULL OR valid_until > ?)""",
                (at_time.isoformat(), at_time.isoformat()),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM kg_relations").fetchall()
        return [self._row_to_relation(r) for r in rows]

    def _row_to_relation(self, row) -> Relation:
        return Relation(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            scope=row["scope"] or "",
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            source_item_id=row["source_item_id"],
            confidence=row["confidence"] or 1.0,
        )


# ── LLM 实体+关系抽取 ──────────────────────────────────────

_EXTRACTION_PROMPT_BASE = """从下面文本中抽取实体和关系,输出 JSON 数组。

{context_section}文本: {text}

规则:
1. 实体类型常见: PERSON(人) / TECH(技术/语言/工具) / DOMAIN(领域/场景)
   / ITEM(物品) / PLACE(地点) / TIME(时间)。
2. 关系格式: (subject, predicate, object, scope)
   - predicate 是动词,如"喜欢"、"使用"、"位于"
   - scope 是可选上下文标签,如"工作"、"业余"、"周一"。
   - **如果文本里没明确说,但前面的上下文暗示了场景,请基于上下文推断 scope**。
     比如上下文在聊"工作项目",当前句又说"用 Python",scope 应推断为"工作"。
   - 无任何上下文线索时留空字符串""
3. 用户说自己时,subject_type="PERSON", subject_name="user"

每条输出格式:
{{"subject_type":"...","subject_name":"...","predicate":"...","object_type":"...","object_name":"...","scope":""}}

如果无法抽取,输出 []。**只输出 JSON 数组,不要其他文字、不要 markdown 包裹**。
"""


def _build_extraction_prompt(text: str, context_hint: Optional[str] = None) -> str:
    if context_hint:
        context_section = f"最近对话上下文(用于推断 scope):\n{context_hint}\n\n"
    else:
        context_section = ""
    return _EXTRACTION_PROMPT_BASE.format(
        context_section=context_section,
        text=text,
    )


def _extract_relations_via_llm(llm, text: str, context_hint: Optional[str] = None) -> list:
    prompt = _build_extraction_prompt(text, context_hint)
    try:
        raw = llm.invoke([{"role": "user", "content": prompt}])
    except Exception as exc:
        print(f"⚠️ KG 关系抽取 LLM 调用失败: {exc}")
        return []

    raw = (raw or "").strip()
    # 去掉 markdown 包裹
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        print(f"⚠️ KG JSON 解析失败: {exc}; 原文: {raw[:100]}...")
        return []


# ── 主类:基于 KG 的冲突检测器 ──────────────────────────────

_QUERY_ENTITY_PROMPT = """从下面问题中抽取关键实体名(用户想问的对象),输出 JSON 数组。

问题: {text}

规则:
1. 只输出实体名字符串,不要 type
2. 用户自指时直接输出 "user"
3. 例如"我喜欢什么编程语言" → ["user", "编程语言"]
4. **只输出 JSON 数组,不要其他文字**。
"""


class KnowledgeGraphConflictDetector(ConflictDetector):
    """extreme 强度: 通过知识图谱解决多种类型冲突。

    支持的冲突类型:
    - 替换型: "喜欢 Java" → "喜欢 Python" 同 scope 触发取代
    - 上下文型: "工作用 Java" + "业余 Python" 不同 scope,共存
    - 时效型: 写入时可显式设置 valid_until

    除了冲突检测外,还提供 query_facts(query) 让 recall 路径能查图。
    """

    # 语义路径的候选池上限:活跃关系超过这个数就跳过 embedding(避免每次查询全量嵌入)
    _SEMANTIC_POOL_CAP = 200

    def __init__(
        self,
        llm,
        store: Optional[KGStore] = None,
        context_window: int = 3,
        embedder=None,
    ):
        self.llm = llm
        self.store = store if store is not None else KGStore()
        self.context_window = context_window  # 抽取时参考最近 N 条 L1 消息推断 scope
        self.embedder = embedder              # 有则开语义路径,无则降级为"图遍历+关键词"
        self._rel_emb_cache: Dict[str, List[float]] = {}  # rel_id → 关系串向量(三元组不可变,可缓存)

    def query_facts(self, query: str, max_facts: int = 8) -> List[str]:
        """根据自然语言 query,从 KG 找当前有效的相关事实(混合检索 + RRF)。

        三路 ranker 对"活跃关系"统一候选池打分:
        1. 图遍历:LLM 抽 query 实体 → 精确匹配关系(精度高但实体名抽歪就漏)
        2. 关键词:query 与关系串的 token 重叠(不依赖实体抽取,兜底)
        3. 语义:query 与关系串的 embedding 余弦(有 embedder 才开)

        三路各自产出排序榜单,RRF 融合 → 取前 max_facts 条转自然语言。
        """
        active = self.store.all_relations(only_active=True)
        if not active:
            return []
        nl_map: Dict[str, str] = {}
        for r in active:
            nl = self.store.relation_to_nl(r)
            if nl:
                nl_map[r.id] = nl
        if not nl_map:
            return []

        rankings: List[List[str]] = []
        graph_ids = self._rank_by_graph(query, nl_map)
        if graph_ids:
            rankings.append(graph_ids)
        keyword_ids = self._rank_by_keyword(query, nl_map)
        if keyword_ids:
            rankings.append(keyword_ids)
        semantic_ids = self._rank_by_semantic(query, nl_map)
        if semantic_ids:
            rankings.append(semantic_ids)

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        return [nl_map[rid] for rid, _ in fused[:max_facts] if rid in nl_map]

    def _rank_by_graph(self, query: str, nl_map: Dict[str, str]) -> List[str]:
        """图遍历路径:抽 query 实体 → 找其参与的活跃关系。"""
        entities = self._extract_query_entities(query)
        if not entities:
            return []
        ordered: List[str] = []
        seen: set = set()
        for entity_name in entities:
            for r in self.store.find_active_relations_for_entity(entity_name):
                if r.id in seen or r.id not in nl_map:
                    continue
                seen.add(r.id)
                ordered.append(r.id)
        return ordered

    def _rank_by_keyword(self, query: str, nl_map: Dict[str, str]) -> List[str]:
        """关键词路径:query 与关系串的 token 重叠数,降序。零重叠的丢弃。"""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        scored: List[Tuple[str, int]] = []
        for rid, nl in nl_map.items():
            overlap = len(q_tokens & set(_tokenize(nl)))
            if overlap > 0:
                scored.append((rid, overlap))
        scored.sort(key=lambda kv: -kv[1])
        return [rid for rid, _ in scored]

    def _rank_by_semantic(self, query: str, nl_map: Dict[str, str]) -> List[str]:
        """语义路径:query 与关系串的 embedding 余弦,降序。无 embedder 或池过大则跳过。"""
        if self.embedder is None or len(nl_map) > self._SEMANTIC_POOL_CAP:
            return []
        try:
            q_vec = self.embedder.embed(query)
        except Exception as exc:
            print(f"⚠️ KG 语义检索 embed 失败,跳过该路: {exc}")
            return []
        scored: List[Tuple[str, float]] = []
        for rid, nl in nl_map.items():
            vec = self._rel_emb_cache.get(rid)
            if vec is None:
                try:
                    vec = self.embedder.embed(nl)
                except Exception:
                    continue
                self._rel_emb_cache[rid] = vec
            sim = _cosine(q_vec, vec)
            if sim > 0:
                scored.append((rid, sim))
        scored.sort(key=lambda kv: -kv[1])
        return [rid for rid, _ in scored]

    def _build_context_hint(self, new_item, manager) -> Optional[str]:
        """从 L1 取最近 N 条 (排除自身) 作为抽取时的上下文提示。"""
        if self.context_window <= 0:
            return None
        recent = [
            it for it in manager.working.items()
            if it.id != new_item.id and it.is_active
        ]
        if not recent:
            return None
        # 取最后 N 条
        recent = recent[-self.context_window:]
        return "\n".join(f"- [{it.role}] {it.content}" for it in recent)

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = _QUERY_ENTITY_PROMPT.format(text=query)
        try:
            raw = self.llm.invoke([{"role": "user", "content": prompt}])
        except Exception as exc:
            print(f"⚠️ KG query 实体抽取失败: {exc}")
            return []
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            data = json.loads(raw)
            return [str(x).strip() for x in data if str(x).strip()] if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def find_superseded(self, new_item, manager) -> List[str]:
        """新记忆写入时:抽取关系 → 找冲突 → 让旧关系失效。

        返回被新记忆取代的旧 MemoryItem ID 列表(供上层标记 supersedes 链)。
        抽取时把最近 context_window 条 L1 消息作为上下文喂给 LLM,
        用于推断隐含的 scope("我刚才在聊工作,现在说用 Python" → scope=工作)。
        """
        # 收集最近上下文(排除当前项)
        context_hint = self._build_context_hint(new_item, manager)
        relations_data = _extract_relations_via_llm(
            self.llm, new_item.content, context_hint=context_hint,
        )
        if not relations_data:
            return []

        now = datetime.now()
        superseded_item_ids: set = set()

        for rel_data in relations_data:
            try:
                subject_id = self.store.get_or_create_entity(
                    rel_data["subject_type"], rel_data["subject_name"],
                )
                object_id = self.store.get_or_create_entity(
                    rel_data["object_type"], rel_data["object_name"],
                )
            except (KeyError, sqlite3.Error) as exc:
                print(f"⚠️ KG 实体写入失败,跳过该关系: {exc}")
                continue

            predicate = rel_data.get("predicate", "")
            scope = rel_data.get("scope", "") or ""
            if not predicate:
                continue

            # 查冲突关系
            conflicts = self.store.find_conflicts(
                subject_id=subject_id,
                predicate=predicate,
                scope=scope,
                exclude_object_id=object_id,
                at_time=now,
            )

            # 旧关系失效
            for old_rel in conflicts:
                self.store.supersede_relation(old_rel.id, now)
                if old_rel.source_item_id:
                    superseded_item_ids.add(old_rel.source_item_id)

            # 新关系入库
            new_rel = Relation(
                id=uuid.uuid4().hex[:12],
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                scope=scope,
                valid_from=now,
                source_item_id=new_item.id,
                confidence=1.0,
            )
            self.store.add_relation(new_rel)

        return list(superseded_item_ids)
