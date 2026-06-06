"""verify/spec.py: Check/CheckSpec 数据类 + SpecGenerator 解析与兜底。"""
from types import SimpleNamespace

from my_agent_llms.verify.spec import Check, CheckSpec, SpecGenerator


def test_check_defaults():
    c = Check(id="c1", type="string_contains", params={"s": "x"})
    assert c.weight == 1.0
    assert c.confidence == 1.0
    assert c.is_hard_oracle is False


def test_checkspec_holds_checks():
    spec = CheckSpec(task="t", checks=[Check(id="c1", type="string_absent", params={"s": "err"})])
    assert spec.task == "t"
    assert spec.checks[0].id == "c1"
