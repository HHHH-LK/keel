"""跨项目复现台账:triple_key 在几个不同项目出现的去重计数。"""
from my_agent_llms.memory.promotion_ledger import PromotionLedger


def test_record_dedups_same_project():
    led = PromotionLedger(None)             # None → 内存库
    assert led.record("user|like|pg|", "projA") == 1
    assert led.record("user|like|pg|", "projA") == 1   # 同项目重复 → 仍 1
    assert led.project_count("user|like|pg|") == 1


def test_record_counts_distinct_projects():
    led = PromotionLedger(None)
    assert led.record("user|like|pg|", "projA") == 1
    assert led.record("user|like|pg|", "projB") == 2   # 第二个不同项目
    assert led.project_count("user|like|pg|") == 2


def test_different_keys_independent():
    led = PromotionLedger(None)
    led.record("a|b|c|", "projA")
    led.record("x|y|z|", "projA")
    assert led.project_count("a|b|c|") == 1
    assert led.project_count("x|y|z|") == 1


def test_persists_to_disk(tmp_path):
    p = tmp_path / "kg.db"
    led1 = PromotionLedger(p)
    led1.record("a|b|c|", "projA")
    led2 = PromotionLedger(p)               # 重新打开同一文件
    assert led2.record("a|b|c|", "projB") == 2
