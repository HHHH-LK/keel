from my_agent_llms.tdd.orchestrator import run_tdd, TddResult
from my_agent_llms.tdd.runner import RunResult, RunOutcome
from my_agent_llms.tdd.test_author import AuthorResult, ProposedTest
from my_agent_llms.tdd.classify import TddDecision


class FakeWorkspace:
    """最小工作区,贴合真实 Workspace API:resolve/resolve_read/root。"""
    def __init__(self, tmp_path):
        self.root = tmp_path

    def resolve(self, relpath):
        return self.root / relpath

    def resolve_read(self, relpath):
        return self.root / relpath


def _author(tests):
    return lambda llm, task, feedback="": AuthorResult(
        tests=[ProposedTest(*t) for t in tests])


def _yes(*a, **k):
    return TddDecision(use_tdd=True, reason="test")


def test_classify_no_degrades(tmp_path):
    ws = FakeWorkspace(tmp_path)
    impl_calls = []
    res = run_tdd(
        llm=None, workspace=ws, task="你好",
        implement_fn=lambda *a, **k: impl_calls.append(1),
        classify_fn=lambda *a, **k: TddDecision(use_tdd=False, reason="闲聊"))
    assert res.degraded is True
    assert impl_calls == []  # 没进 TDD


def test_happy_path_red_then_green(tmp_path):
    ws = FakeWorkspace(tmp_path)
    runs = iter([
        RunResult(RunOutcome.MISSING_IMPL),  # 红门:期望红
        RunResult(RunOutcome.PASS),          # 绿门:转绿
    ])
    res = run_tdd(
        llm=None, workspace=ws, task="写 add",
        implement_fn=lambda *a, **k: None,
        classify_fn=_yes,
        author_fn=_author([("test_add.py", "x")]),
        runner_fn=lambda *a, **k: next(runs))
    assert res.success is True and res.degraded is False


def test_fake_test_bounces_to_author_then_exhausts(tmp_path):
    ws = FakeWorkspace(tmp_path)
    # 红门永远判"假测试"(PASS),author_budget=2 用尽 → 降级
    res = run_tdd(
        llm=None, workspace=ws, task="写 add",
        implement_fn=lambda *a, **k: None,
        classify_fn=_yes,
        author_fn=_author([("test_add.py", "x")]),
        runner_fn=lambda *a, **k: RunResult(RunOutcome.PASS),
        author_budget=2)
    assert res.degraded is True
    assert "test-author" in res.message


def test_impl_never_green_returns_honest_failure(tmp_path):
    ws = FakeWorkspace(tmp_path)
    runs = iter([RunResult(RunOutcome.MISSING_IMPL)] +
                [RunResult(RunOutcome.ASSERT_FAIL, failed=["test_add::test_x"])] * 10)
    res = run_tdd(
        llm=None, workspace=ws, task="写 add",
        implement_fn=lambda *a, **k: None,
        classify_fn=_yes,
        author_fn=_author([("test_add.py", "x")]),
        runner_fn=lambda *a, **k: next(runs),
        impl_budget=3)
    assert res.success is False
    assert "没过" in res.message  # 如实告知,不假装成功


def test_write_failure_degrades(tmp_path):
    """写盘异常(越界/磁盘满)→ 降级走老路,不抛到调用方。"""
    class BoomWorkspace:
        def __init__(self, root): self.root = root
        def resolve(self, relpath): raise RuntimeError("disk full")
        def resolve_read(self, relpath): return self.root / relpath
    res = run_tdd(
        llm=None, workspace=BoomWorkspace(tmp_path), task="写 add",
        implement_fn=lambda *a, **k: None, classify_fn=_yes,
        author_fn=_author([("test_add.py", "x")]),
        runner_fn=lambda *a, **k: RunResult(RunOutcome.PASS))
    assert res.degraded is True and res.success is False


def test_implement_callback_exception_does_not_crash(tmp_path):
    """实现回调每轮都抛 → 不崩,诚实失败(由 impl_budget 兜底)。"""
    ws = FakeWorkspace(tmp_path)
    def boom_impl(task, paths, fb):
        raise RuntimeError("impl boom")
    res = run_tdd(
        llm=None, workspace=ws, task="写 add",
        implement_fn=boom_impl, classify_fn=_yes,
        author_fn=_author([("test_add.py", "x")]),
        runner_fn=lambda *a, **k: RunResult(RunOutcome.MISSING_IMPL),
        impl_budget=2)
    assert res.success is False


def test_implementer_cannot_tamper_tests(tmp_path):
    """隔离不变量:实现方偷改了测试文件 → 即便绿也拒绝。"""
    ws = FakeWorkspace(tmp_path)

    def sneaky_impl(task, test_paths, feedback):
        # 实现方偷偷把测试文件改成永远绿
        (ws.root / "test_add.py").write_text("def test_x():\n    assert True\n")

    runs = iter([
        RunResult(RunOutcome.MISSING_IMPL),  # 红门:期望红
        RunResult(RunOutcome.PASS),          # 绿门:PASS(但测试被改过)
    ])
    res = run_tdd(
        llm=None, workspace=ws, task="写 add",
        implement_fn=sneaky_impl,
        classify_fn=_yes,
        author_fn=_author([("test_add.py", "def test_x():\n    assert add(2,3)==5\n")]),
        runner_fn=lambda *a, **k: next(runs),
        impl_budget=1)
    assert res.success is False
    assert "篡改" in res.message
