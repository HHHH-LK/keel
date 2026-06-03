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


# ─────────────────────────────────────────────
# Task 1.3: 实体归一化 + alias(治实体分身)
# ─────────────────────────────────────────────

def test_entity_dedup_case_insensitive():
    """Python / python / PYTHON 归一成同一个实体。"""
    store = KGStore()
    a = store.get_or_create_entity("TECH", "Python")
    b = store.get_or_create_entity("TECH", "python")
    c = store.get_or_create_entity("TECH", "PYTHON")
    assert a == b == c


def test_entity_dedup_whitespace():
    """首尾/多余空白不产生分身。"""
    store = KGStore()
    a = store.get_or_create_entity("TECH", "Python")
    b = store.get_or_create_entity("TECH", "  Python  ")
    assert a == b


def test_distinct_entities_stay_distinct():
    """不同实体仍然分开。"""
    store = KGStore()
    a = store.get_or_create_entity("TECH", "Python")
    b = store.get_or_create_entity("TECH", "Java")
    assert a != b


def test_alias_resolves_to_same_entity():
    """登记别名后,用别名也命中同一实体。"""
    store = KGStore()
    pid = store.get_or_create_entity("TECH", "Python")
    store.add_alias("蟒蛇", pid)
    assert store.get_or_create_entity("TECH", "蟒蛇") == pid


def test_entity_split_no_longer_causes_false_supersede():
    """实体归一后,'Python'/'python' 不再被当成两个值而虚假取代(多值场景共存)。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "喜欢", "Python")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "喜欢", "python")], source_item_id="i2")
    assert superseded == []                 # 同一实体,且'喜欢'多值 → 不取代
    assert _active_objects(d) == {"Python"}  # 归一成一个


# ─────────────────────────────────────────────
# Task 1.4: scope 归一化(治 scope 漂移漏判)
# ─────────────────────────────────────────────

def test_normalize_scope_maps_synonym():
    """同义 scope 归一:上班→工作。"""
    assert kg_vocab.normalize_scope("上班") == "工作"


def test_normalize_scope_empty_stays_empty():
    """空 scope 保持空(无场景约束)。"""
    assert kg_vocab.normalize_scope("") == ""


def test_scope_synonym_triggers_conflict():
    """scope 同义归一后,同场景的单值冲突能触发(工作/上班 视为同场景)。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "主力语言", "Java", scope="工作")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "主力语言", "Go", scope="上班")], source_item_id="i2")
    assert "i1" in superseded               # 工作==上班 同场景 → 取代


def test_different_scope_coexist():
    """不同场景不冲突:工作 vs 业余 共存。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "主力语言", "Java", scope="工作")], source_item_id="i1")
    superseded = d.apply_extracted_relations([_rel("user", "主力语言", "Python", scope="业余")], source_item_id="i2")
    assert superseded == []
    assert _active_objects(d) == {"Java", "Python"}


# ─────────────────────────────────────────────
# Task 1.5: 权威闸门(低不盖高,防抹用户硬约束)
# ─────────────────────────────────────────────

def test_authority_user_explicit_outranks_inferred():
    assert kg_vocab.authority_of("user_explicit") > kg_vocab.authority_of("inferred")


def test_low_authority_cannot_supersede_high():
    """LLM 推断(低权威)不能取代用户显式(高权威)的单值事实。"""
    d = _detector()
    d.apply_extracted_relations(
        [_rel("user", "住在", "北京")], source_item_id="i1", source_type="user_explicit",
    )
    superseded = d.apply_extracted_relations(
        [_rel("user", "住在", "上海")], source_item_id="i2", source_type="inferred",
    )
    assert superseded == []                 # 没取代
    assert "北京" in _active_objects(d)      # 用户显式的硬事实还在


def test_equal_or_higher_authority_can_supersede():
    """同等/更高权威可以正常取代。"""
    d = _detector()
    d.apply_extracted_relations(
        [_rel("user", "住在", "北京")], source_item_id="i1", source_type="user_stated",
    )
    superseded = d.apply_extracted_relations(
        [_rel("user", "住在", "上海")], source_item_id="i2", source_type="user_explicit",
    )
    assert "i1" in superseded
