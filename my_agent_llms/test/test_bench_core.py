import json, tempfile
from pathlib import Path
from my_agent_llms.bench.case import BenchCase, load_cases
from my_agent_llms.bench.scorer import score, RunResultLike, Score
from my_agent_llms.bench.report import summarize


def _write_case(d, obj):
    (Path(d) / f"{obj['id']}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_load_cases_fills_id():
    d = tempfile.mkdtemp()
    _write_case(d, {"id": "c1", "task": "做X",
                    "checks": [{"type": "string_contains", "params": {"s": "X"}}]})
    cases = load_cases(d)
    assert len(cases) == 1 and cases[0].id == "c1"
    assert cases[0].checks[0]["id"] == "c0"          # load 自动补 id


def test_score_pass_on_answer_contains():
    case = BenchCase(id="c1", task="t", setup_files={},
                     checks=[{"id": "c0", "type": "string_contains",
                              "params": {"s": "结论"}, "weight": 1.0}])
    rr = RunResultLike(answer="这是结论", trajectory=[], workspace_root=None)
    s = score(case, rr)
    assert s.passed is True and s.residual == 0.0


def test_score_fail_when_missing():
    case = BenchCase(id="c1", task="t", setup_files={},
                     checks=[{"id": "c0", "type": "string_contains",
                              "params": {"s": "结论"}, "weight": 1.0}])
    rr = RunResultLike(answer="没那个词", trajectory=[], workspace_root=None)
    s = score(case, rr)
    assert s.passed is False and s.residual > 0


def test_score_all_skipped_not_pass():
    # 坏命令(SyntaxError)→ SKIP → 全 SKIP → 不算 pass(空验证)
    case = BenchCase(id="c1", task="t", setup_files={},
                     checks=[{"id": "c0", "type": "command_ok",
                              "params": {"cmd": "python3 -c \"a=1; for x in []: pass\""},
                              "weight": 10.0, "is_hard_oracle": True}])
    rr = RunResultLike(answer="x", trajectory=[], workspace_root=None)
    s = score(case, rr)
    assert s.passed is False          # 全 SKIP ≠ 通过


def test_summarize():
    out = summarize([Score("a", True, 0.0, []), Score("b", False, 2.0, ["string_contains"])])
    assert "1/2" in out
    assert "a" in out and "b" in out


def test_run_case_with_mock_factory():
    from my_agent_llms.bench.runner import run_case
    import os
    case = BenchCase(id="c1", task="写 hi", setup_files={"seed.txt": "S"}, checks=[])
    captured = {}

    def factory(ws_root):
        captured["ws"] = ws_root

        class A:
            def run(self, task, **kw):
                return f"done:{task}"
        return A()

    rr = run_case(case, factory)
    assert rr.answer == "done:写 hi"
    assert rr.case_id == "c1"
    assert os.path.exists(os.path.join(captured["ws"], "seed.txt"))   # setup_files 写进隔离 ws
