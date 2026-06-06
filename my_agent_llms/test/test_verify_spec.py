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


def _fake_llm(reply: str):
    return SimpleNamespace(invoke=lambda msgs, **kw: reply)


def test_spec_generator_parses_json_checks():
    reply = """这是规格:
```json
{"checks": [
  {"id": "c1", "type": "string_contains", "params": {"s": "结论"}, "weight": 1.0, "confidence": 0.8, "is_hard_oracle": false},
  {"id": "c2", "type": "tool_called", "params": {"tool": "write_file"}, "weight": 10.0, "is_hard_oracle": true}
]}
```"""
    gen = SpecGenerator(_fake_llm(reply))
    spec = gen.generate("写个文件并总结", tools=["write_file", "read_file"])
    assert spec.task == "写个文件并总结"
    assert len(spec.checks) == 2
    assert spec.checks[0].id == "c1"
    assert spec.checks[0].confidence == 0.8
    assert spec.checks[1].is_hard_oracle is True
    assert spec.checks[1].weight == 10.0


def test_spec_generator_falls_back_on_bad_json():
    gen = SpecGenerator(_fake_llm("抱歉我不会输出 JSON"))
    spec = gen.generate("随便", tools=[])
    # 兜底:至少一条 hard-oracle 检查,不抛异常
    assert len(spec.checks) >= 1
    assert all(c.is_hard_oracle for c in spec.checks)


def test_spec_generator_falls_back_on_llm_error():
    def boom(msgs, **kw):
        raise RuntimeError("llm down")
    gen = SpecGenerator(SimpleNamespace(invoke=boom))
    spec = gen.generate("随便", tools=[])
    assert len(spec.checks) >= 1
    assert all(c.is_hard_oracle for c in spec.checks)


def test_spec_generator_skips_malformed_keeps_valid():
    reply = '{"checks": [{"type": "string_contains", "params": {"s": "ok"}}, {"bad": true}]}'
    gen = SpecGenerator(_fake_llm(reply))
    spec = gen.generate("t", tools=[])
    assert len(spec.checks) == 1
    assert spec.checks[0].type == "string_contains"
    assert spec.checks[0].id == "c0"   # id 缺省 → f"c{i}"
