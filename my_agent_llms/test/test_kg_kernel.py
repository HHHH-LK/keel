"""KG 内核改造测试(Phase 1:冲突键加固)。

覆盖:受控谓词词表 + 基数、实体归一化、scope 归一化、权威闸门、audit。
"""
from datetime import datetime

from my_agent_llms.memory import kg_vocab
from my_agent_llms.memory.kg import KGStore, KnowledgeGraphConflictDetector


def _detector():
    return KnowledgeGraphConflictDetector(llm=None, store=KGStore())


def _rel(subj, pred, obj, scope="", subj_type="PERSON", obj_type="ITEM"):
    return {
        "subject_type": subj_type, "subject_name": subj,
        "predicate": pred,
        "object_type": obj_type, "object_name": obj,
        "scope": scope,
    }


def _active_objects(detector, entity="user"):
    active = detector.store.find_active_relations_for_entity(entity)
    return {detector.store.get_entity(r.object_id).name for r in active}


# ─────────────────────────────────────────────
# Task 1.1: 受控谓词词表 + 归一化
# ─────────────────────────────────────────────

def test_normalize_predicate_maps_synonym_to_canonical():
    """同义谓词归一到同一个 canonical。"""
    canonical, _ = kg_vocab.normalize_predicate("偏好")
    assert canonical == "喜欢"


def test_normalize_predicate_canonical_is_stable():
    """canonical 词归一到自己。"""
    canonical, _ = kg_vocab.normalize_predicate("喜欢")
    assert canonical == "喜欢"


def test_normalize_predicate_single_valued_cardinality():
    """单值谓词(现居地类)基数为 single。"""
    canonical, cardinality = kg_vocab.normalize_predicate("住在")
    assert canonical == "现居地"
    assert cardinality == kg_vocab.CARDINALITY_SINGLE


def test_normalize_predicate_multi_valued_cardinality():
    """多值谓词(过敏)基数为 multi —— 安全关键。"""
    _, cardinality = kg_vocab.normalize_predicate("过敏")
    assert cardinality == kg_vocab.CARDINALITY_MULTI


def test_normalize_predicate_unknown_defaults_to_multi():
    """未知谓词默认 multi —— 宁可漏判不可误杀。"""
    canonical, cardinality = kg_vocab.normalize_predicate("瞎编的动词")
    assert canonical == "瞎编的动词"
    assert cardinality == kg_vocab.CARDINALITY_MULTI


# ─────────────────────────────────────────────
# Task 1.2: 谓词基数 → 多值不 supersede(安全关键)
# ─────────────────────────────────────────────

def test_multi_valued_predicate_coexist_no_supersede():
    """多值谓词(过敏):新值只追加,旧值绝不被误杀。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "过敏", "花生")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "过敏", "牛奶")], source_item_id="i2")
    assert superseded == []                        # 没有任何取代
    assert _active_objects(d) == {"花生", "牛奶"}   # 两个过敏都还在


def test_single_valued_predicate_supersedes_old():
    """单值谓词(现居地):新值取代旧值。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "住在", "北京")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "住在", "上海")], source_item_id="i2")
    assert "i1" in superseded
    assert _active_objects(d) == {"上海"}


def test_synonym_predicate_triggers_conflict():
    """同义谓词归一后能触发冲突:住在/居住 都→现居地。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "住在", "北京")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "居住", "上海")], source_item_id="i2")
    assert "i1" in superseded
    assert _active_objects(d) == {"上海"}
