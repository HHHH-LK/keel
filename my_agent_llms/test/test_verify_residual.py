"""verify/residual.py: 加权残差聚合。"""
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.residual import residual


def test_all_pass_is_zero():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={}),
        Check(id="b", type="string_absent", params={}),
    ])
    assert residual(spec, {"a": True, "b": True}) == 0.0


def test_weighted_failures_sum():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="x", params={}, weight=2.0),
        Check(id="b", type="y", params={}, weight=3.0),
    ])
    # a 不过 → 2*1*1 = 2;b 过 → 0
    assert residual(spec, {"a": False, "b": True}) == 2.0


def test_confidence_downweights_soft_oracle():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="judge", params={}, weight=1.0, confidence=0.5),
    ])
    assert residual(spec, {"a": False}) == 0.5


def test_hard_oracle_dominates():
    spec = CheckSpec(task="t", checks=[
        Check(id="hard", type="command_ok", params={}, weight=10.0, is_hard_oracle=True),
        Check(id="soft", type="judge", params={}, weight=1.0, confidence=0.3),
    ])
    # 两条都不过:10 + 0.3 = 10.3,hard 压倒性主导
    assert residual(spec, {"hard": False, "soft": False}) == 10.3


def test_missing_result_treated_as_fail():
    spec = CheckSpec(task="t", checks=[Check(id="a", type="x", params={}, weight=4.0)])
    # results 里没有 a 的键 → 当作未通过
    assert residual(spec, {}) == 4.0
