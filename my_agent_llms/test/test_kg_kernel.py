"""KG 内核改造测试(Phase 1:冲突键加固)。

覆盖:受控谓词词表 + 基数、实体归一化、scope 归一化、权威闸门、audit。
"""
from datetime import datetime

from my_agent_llms.memory import kg_vocab


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
