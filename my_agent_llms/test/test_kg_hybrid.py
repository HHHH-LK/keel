"""KG 混合检索 + RRF 重排测试。

query_facts 从"只走图遍历(实体精确匹配)"升级为三路混合:
- 图遍历:抽取实体 → 精确匹配关系(精度高但脆)
- 关键词:query 与关系串的 token 重叠(不依赖实体抽取)
- 语义:query 与关系串的 embedding 余弦(可选)
三路结果用 Reciprocal Rank Fusion 融合。

核心收益:实体名抽歪了,关键词/语义仍能兜底召回(修"精确匹配太脆")。
"""
import json
from datetime import datetime, timedelta

from my_agent_llms.memory.embeddings import HashEmbedding
from my_agent_llms.memory.kg import (
    KGStore,
    KnowledgeGraphConflictDetector,
    Relation,
    reciprocal_rank_fusion,
)


class FakeLLM:
    """固定返回一组实体名(模拟 query 实体抽取)。"""

    def __init__(self, entities):
        self._entities = entities

    def invoke(self, messages):
        return json.dumps(self._entities, ensure_ascii=False)


def _store_with_fact(predicate="喜欢", obj_name="Python", subj_name="user"):
    store = KGStore()
    sid = store.get_or_create_entity("PERSON", subj_name)
    oid = store.get_or_create_entity("TECH", obj_name)
    store.add_relation(
        Relation(
            id="r1",
            subject_id=sid,
            predicate=predicate,
            object_id=oid,
            valid_from=datetime.now(),
            source_item_id="item1",
        )
    )
    return store


# ─────────────────────────────────────────────
# RRF 纯函数
# ─────────────────────────────────────────────

def test_rrf_rewards_consensus():
    """同时排在多个榜单前列的项,融合后排第一。"""
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "a"


def test_rrf_unions_all_lists():
    """融合结果是所有榜单的并集。"""
    fused = reciprocal_rank_fusion([["a"], ["b"], ["c"]])
    assert {i for i, _ in fused} == {"a", "b", "c"}


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ─────────────────────────────────────────────
# 混合检索集成
# ─────────────────────────────────────────────

def test_graph_traversal_still_works():
    """实体名精确命中时,图遍历路径照常召回。"""
    store = _store_with_fact()
    detector = KnowledgeGraphConflictDetector(FakeLLM(["user"]), store)
    facts = detector.query_facts("user 喜欢什么")
    assert any("Python" in f for f in facts)


def test_keyword_path_recalls_when_graph_misses():
    """实体抽歪(抽到 KG 里不存在的名字)→ 图遍历空,
    但关键词路径靠 query 与关系串的 token 重叠兜底召回。"""
    store = _store_with_fact()
    # LLM 抽到 "编程语言" —— KG 里没这个实体,图遍历必然为空
    detector = KnowledgeGraphConflictDetector(FakeLLM(["编程语言"]), store)
    facts = detector.query_facts("用户 喜欢 什么编程语言")
    assert any("Python" in f for f in facts), "关键词路径应兜底召回到 Python 事实"


def test_semantic_path_does_not_crash_with_embedder():
    """带 embedder 时三路全开,结果仍正确、不报错。"""
    store = _store_with_fact()
    detector = KnowledgeGraphConflictDetector(
        FakeLLM(["编程语言"]), store, embedder=HashEmbedding(dim=32)
    )
    facts = detector.query_facts("用户 喜欢 Python 吗")
    assert any("Python" in f for f in facts)


def test_manager_wires_embedder_into_kg_detector():
    """MemoryManager(extreme + embedding + llm) 应把 embedder 注入 KG detector,
    让语义路径在生产路径上真正开启。"""
    from my_agent_llms.memory import MemoryConfig, MemoryManager

    embedder = HashEmbedding(dim=32)
    mgr = MemoryManager(
        MemoryConfig(conflict_strength="extreme"),
        embedding=embedder,
        llm=FakeLLM([]),
    )
    assert mgr.conflict_detector.embedder is embedder


def test_superseded_relations_not_recalled():
    """已失效(valid_until 过去)的关系不应被任何路径召回。"""
    store = _store_with_fact()
    # 让 r1 失效
    store.supersede_relation("r1", datetime.now() - timedelta(seconds=1))
    detector = KnowledgeGraphConflictDetector(FakeLLM(["user"]), store)
    facts = detector.query_facts("user 喜欢 Python 吗")
    assert not any("Python" in f for f in facts)
