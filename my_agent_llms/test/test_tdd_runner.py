"""runner 真跑 pytest,验证三态分类。这是验证 TDD 设计初衷的皇冠测试。"""
from my_agent_llms.tdd.runner import run_pytest, RunOutcome


def _write(d, name, content):
    p = d / name
    p.write_text(content, encoding="utf-8")
    return p


def test_fake_test_passes_without_impl(tmp_path):
    # 🏆 假测试:没实现就 PASS → 必须判 PASS(红门会据此判"假测试")
    _write(tmp_path, "test_fake.py", "def test_x():\n    assert True\n")
    res = run_pytest(str(tmp_path))
    assert res.outcome == RunOutcome.PASS


def test_real_test_missing_impl_is_missing(tmp_path):
    # 🏆 真测试 + 没实现 → ImportError → 必须判 MISSING_IMPL(期望红),不是 BROKEN
    _write(tmp_path, "test_real.py",
           "from mymod import f\n\ndef test_x():\n    assert f(2) == 4\n")
    res = run_pytest(str(tmp_path))
    assert res.outcome == RunOutcome.MISSING_IMPL


def test_real_test_with_correct_impl_passes(tmp_path):
    _write(tmp_path, "mymod.py", "def f(x):\n    return x * 2\n")
    _write(tmp_path, "test_real.py",
           "from mymod import f\n\ndef test_x():\n    assert f(2) == 4\n")
    res = run_pytest(str(tmp_path))
    assert res.outcome == RunOutcome.PASS


def test_assertion_failure_is_assert_fail(tmp_path):
    _write(tmp_path, "mymod.py", "def f(x):\n    return x * 3\n")
    _write(tmp_path, "test_real.py",
           "from mymod import f\n\ndef test_x():\n    assert f(2) == 4\n")
    res = run_pytest(str(tmp_path))
    assert res.outcome == RunOutcome.ASSERT_FAIL


def test_broken_test_syntax_error(tmp_path):
    _write(tmp_path, "test_broken.py", "def test_x(:\n    assert True\n")
    res = run_pytest(str(tmp_path))
    assert res.outcome == RunOutcome.BROKEN
