"""KG 内核改造测试(Phase 1:冲突键加固)。

覆盖:受控谓词词表 + 基数、实体归一化、scope 归一化、权威闸门、audit。
"""
from datetime import datetime

from my_agent_llms.memory import kg_vocab
from my_agent_llms.memory.kg import (
    KGStore,
    KnowledgeGraphConflictDetector,
    is_grounded,
    should_extract,
)


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


# ─────────────────────────────────────────────
# Task 1.6: audit log(supersede 可追溯/可回滚)
# ─────────────────────────────────────────────

def test_supersede_writes_audit():
    """每次 supersede 落一条 audit。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "住在", "北京")], source_item_id="i1")
    d.apply_extracted_relations([_rel("user", "住在", "上海")], source_item_id="i2")
    entries = d.store.audit_entries()
    assert any(e["op"] == "supersede" for e in entries)


def test_no_supersede_audit_for_multivalue():
    """多值谓词只追加、不取代 → 不产生 supersede audit。"""
    d = _detector()
    d.apply_extracted_relations([_rel("user", "过敏", "花生")], source_item_id="i1")
    d.apply_extracted_relations([_rel("user", "过敏", "牛奶")], source_item_id="i2")
    entries = d.store.audit_entries()
    assert not any(e["op"] == "supersede" for e in entries)


# ─────────────────────────────────────────────
# Task 2.3: pending 表 + 证据累积
# ─────────────────────────────────────────────

def test_record_pending_creates_entry():
    store = KGStore()
    hits = store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", source_item_id="i1")
    assert hits == 1
    assert len(store.pending_entries()) == 1


def test_record_pending_accrues_across_synonyms():
    """同一 triple 不同措辞(喜欢/偏好)归一成同 key → 累积而非重复。"""
    store = KGStore()
    store.record_pending(_rel("user", "喜欢", "Rust"), reason="inferred", source_item_id="i1")
    hits = store.record_pending(_rel("user", "偏好", "Rust"), reason="inferred", source_item_id="i2")
    assert hits == 2
    assert len(store.pending_entries()) == 1


def test_remove_pending():
    store = KGStore()
    store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", source_item_id="i1")
    store.remove_pending(_rel("user", "想学", "Rust"))
    assert store.pending_entries() == []


# ─────────────────────────────────────────────
# Task 2.2: grounding 校验
# ─────────────────────────────────────────────

def test_grounded_object_in_text():
    assert is_grounded(_rel("user", "喜欢", "Python"), "我喜欢 Python") is True


def test_ungrounded_object_not_in_text():
    """object 没在原文出现 → 不 grounded(防幻觉:咖啡≠拿铁)。"""
    assert is_grounded(_rel("user", "喜欢", "拿铁"), "我喜欢喝咖啡") is False


def test_user_self_subject_exempt():
    """subject 是 user 自指可豁免,只要 object 落地即可。"""
    assert is_grounded(_rel("user", "住在", "上海"), "我上周搬到上海了") is True


def test_non_user_subject_must_appear():
    """非 user 的 subject 必须在原文出现,否则疑似张冠李戴。"""
    assert is_grounded(_rel("李四", "喜欢", "Python", subj_type="PERSON"), "我喜欢 Python") is False


# ─────────────────────────────────────────────
# Task 2.1: 路由 main vs pending + 接线 source_type
# ─────────────────────────────────────────────

def test_inferred_fact_goes_to_pending():
    """LLM 推断(低权威)不直接进主图,先进 pending 待确认。"""
    d = _detector()
    d.apply_extracted_relations(
        [_rel("user", "喜欢", "Python")], source_item_id="i1", source_type="inferred",
    )
    assert _active_objects(d) == set()
    assert len(d.store.pending_entries()) == 1


def test_user_stated_grounded_fact_goes_to_main():
    """用户陈述 + 原文支撑 → 直写主图。"""
    d = _detector()
    d.apply_extracted_relations(
        [_rel("user", "喜欢", "Python")],
        source_item_id="i1", source_type="user_stated", source_text="我喜欢 Python",
    )
    assert _active_objects(d) == {"Python"}


def test_ungrounded_fact_goes_to_pending():
    """原文不支撑(咖啡≠拿铁)→ 进 pending,不污染主图。"""
    d = _detector()
    d.apply_extracted_relations(
        [_rel("user", "喜欢", "拿铁")],
        source_item_id="i1", source_type="user_stated", source_text="我喜欢喝咖啡",
    )
    assert _active_objects(d) == set()
    assert len(d.store.pending_entries()) == 1


def test_pending_promotes_after_repeated_evidence():
    """推断事实反复独立出现(达阈值)→ 晋升进主图,并从 pending 移除。"""
    d = _detector()  # 默认 pending_promote_hits=2
    d.apply_extracted_relations(
        [_rel("user", "喜欢", "Python")], source_item_id="i1", source_type="inferred",
    )
    assert _active_objects(d) == set()              # 第一次:pending
    d.apply_extracted_relations(
        [_rel("user", "偏好", "Python")], source_item_id="i2", source_type="inferred",
    )
    assert _active_objects(d) == {"Python"}          # 第二次累积到 2 → 晋升
    assert d.store.pending_entries() == []


# ─────────────────────────────────────────────
# Task 2.4: 保守门控(只挡空消息/纯应答,不漏真事实)
# ─────────────────────────────────────────────

def test_should_extract_real_content():
    assert should_extract("我喜欢 Python") is True


def test_should_extract_short_fact_not_dropped():
    """短但有事实的句子不能被门控误杀。"""
    assert should_extract("我住上海") is True


def test_should_skip_pure_acknowledgment():
    assert should_extract("嗯") is False
    assert should_extract("好的") is False
    assert should_extract("OK") is False


def test_should_skip_empty_and_punctuation():
    assert should_extract("") is False
    assert should_extract("   ") is False
    assert should_extract("。。。") is False


# ─────────────────────────────────────────────
# Task 2.5: 复写 = 强化(不重复入库,bump confidence)
# ─────────────────────────────────────────────

def _restate(d, sid):
    d.apply_extracted_relations(
        [_rel("user", "喜欢", "Python")],
        source_item_id=sid, source_type="user_stated", source_text="我喜欢 Python",
    )


def test_restatement_no_duplicate_relation():
    """同一事实复写不产生重复关系行。"""
    d = _detector()
    _restate(d, "i1")
    _restate(d, "i2")
    active = d.store.find_active_relations_for_entity("user")
    pythons = [r for r in active if d.store.get_entity(r.object_id).name == "Python"]
    assert len(pythons) == 1


def test_restatement_bumps_confidence():
    """复写是印证信号 → confidence 提升。"""
    d = _detector()
    _restate(d, "i1")
    before = d.store.find_active_relations_for_entity("user")[0].confidence
    _restate(d, "i2")
    after = d.store.find_active_relations_for_entity("user")[0].confidence
    assert after > before


def test_initial_confidence_by_source():
    """初始 confidence 按来源:用户显式 > 用户陈述。"""
    assert kg_vocab.base_confidence("user_explicit") > kg_vocab.base_confidence("user_stated")


# ─────────────────────────────────────────────
# Task 3.1: bi-temporal —— valid_from 盖事件时间,4 时间戳
# ─────────────────────────────────────────────

def test_valid_from_uses_event_time_not_now():
    """valid_from 盖的是事件时间(Episode 时刻),不是 worker 的 now。"""
    d = _detector()
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    d.apply_extracted_relations(
        [_rel("user", "住在", "北京")],
        source_item_id="i1", source_type="user_stated", source_text="我住北京",
        event_time=t0,
    )
    rel = d.store.find_active_relations_for_entity("user")[0]
    assert rel.valid_from == t0              # 事件时间 T
    assert rel.created_at > rel.valid_from   # 事务时间 T' ≈ now,与事件时间分开


def test_bitemporal_point_in_time_query():
    """能回答'某个时间点什么为真':搬家前查到北京,搬家后查到上海。"""
    d = _detector()
    t0 = datetime(2025, 1, 1)   # 在北京
    t1 = datetime(2025, 6, 1)   # 搬到上海
    d.apply_extracted_relations(
        [_rel("user", "住在", "北京")],
        source_item_id="i1", source_type="user_stated", source_text="我住北京", event_time=t0,
    )
    d.apply_extracted_relations(
        [_rel("user", "住在", "上海")],
        source_item_id="i2", source_type="user_stated", source_text="我搬到上海", event_time=t1,
    )
    now_objs = {
        d.store.get_entity(r.object_id).name
        for r in d.store.find_active_relations_for_entity("user", at_time=datetime(2025, 12, 1))
    }
    past_objs = {
        d.store.get_entity(r.object_id).name
        for r in d.store.find_active_relations_for_entity("user", at_time=datetime(2025, 3, 1))
    }
    assert now_objs == {"上海"}    # 当前
    assert past_objs == {"北京"}    # 搬家前那个时间点
