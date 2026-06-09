"""坏 oracle(命令自身 SyntaxError 等)→ 三态 SKIP,不计入残差,不拿坏题罚 agent。

回归动机:实跑发现 SpecGenerator 会生成 `python3 -c "...; for ..."` 这类单行非法命令,
exit≠0 但根因是命令语法错,不该当成 agent 失败。
"""
from my_agent_llms.verify.checkers import check_one, CheckContext, CheckerRunner
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.residual import residual, effective_count
from my_agent_llms.verify.convergence import ConvergenceJudge, Verdict


# ── check_one 三态 ──────────────────────────────────────────────
def test_command_ok_syntax_error_skips():
    """单行 python -c 里 for 是 SyntaxError → 命令自身坏 → SKIP(None)。"""
    chk = Check(id="c", type="command_ok",
                params={"cmd": 'python3 -c "a=1; for x in []: pass"'})
    assert check_one(chk, CheckContext(result="")) is None


def test_command_ok_assertion_fail_is_false():
    """命令能跑、断言失败(AssertionError)→ 真失败 → False(计入)。"""
    chk = Check(id="c", type="command_ok", params={"cmd": 'python3 -c "assert False"'})
    assert check_one(chk, CheckContext(result="")) is False


def test_command_ok_pass_is_true():
    chk = Check(id="c", type="command_ok", params={"cmd": 'python3 -c "assert True"'})
    assert check_one(chk, CheckContext(result="")) is True


# ── residual 跳过 SKIP ──────────────────────────────────────────
def test_residual_skips_none():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="command_ok", params={}, weight=10.0),
        Check(id="b", type="string_contains", params={"s": "x"}, weight=1.0),
    ])
    assert residual(spec, {"a": None, "b": False}) == 1.0   # a SKIP 不计,b 未过计 1
    assert residual(spec, {"a": None, "b": True}) == 0.0    # a SKIP,b 过 → 0


def test_effective_count_excludes_skipped():
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="command_ok", params={}),
        Check(id="b", type="string_contains", params={"s": "x"}),
    ])
    assert effective_count(spec, {"a": None, "b": True}) == 1
    assert effective_count(spec, {"a": True, "b": False}) == 2
    assert effective_count(spec, {"a": None, "b": None}) == 0


# ── 边界:全 SKIP(空验证)不算收敛 ───────────────────────────────
def test_judge_no_converge_when_all_skipped():
    j = ConvergenceJudge(hard_cap=5, K=99)
    # residual=0 但没有任何有效 check → 不能判 CONVERGED(空验证≠通过)
    assert j.judge(0, 0.0, "fp", [], has_effective=False) != Verdict.CONVERGED
    # 对照:有有效 check 且残差 0 → 正常 CONVERGED
    assert j.judge(0, 0.0, "fp2", []) == Verdict.CONVERGED


# ── 端到端:坏命令 oracle 不阻碍真正满足的好 oracle 收敛 ─────────
def test_runner_mixed_skip_and_pass(tmp_path):
    """一个坏命令(SKIP)+ 一个真过的命令 → 有效残差 0,且 effective_count=1。"""
    spec = CheckSpec(task="t", checks=[
        Check(id="bad", type="command_ok",
              params={"cmd": 'python3 -c "x=1; for i in []: pass"'}, weight=10.0),
        Check(id="good", type="command_ok",
              params={"cmd": 'python3 -c "assert 1==1"'}, weight=10.0),
    ])
    results = CheckerRunner().run(spec, CheckContext(result=""))
    assert results["bad"] is None      # 坏命令 SKIP
    assert results["good"] is True     # 好命令过
    assert residual(spec, results) == 0.0
    assert effective_count(spec, results) == 1


# ── SpecGenerator prompt 明确 string_* 看不到文件,文件内容用 command_ok/field_equals ──
def test_spec_prompt_warns_string_checks_cannot_see_files():
    from my_agent_llms.verify.spec import _SPEC_PROMPT
    assert "文本回答" in _SPEC_PROMPT                  # string_* 语义被澄清
    assert "command_ok 或 field_equals" in _SPEC_PROMPT  # 文件内容的正确选型被点名


def test_spec_prompt_discourages_fabricated_command_ok_for_docs():
    """纯文档/文案任务不该凭空编 command_ok(失败反馈不可行动 → 空转)。"""
    from my_agent_llms.verify.spec import _SPEC_PROMPT
    assert "不可行动" in _SPEC_PROMPT                  # 点名 command_ok 反馈的缺陷
    assert "真实可执行测试" in _SPEC_PROMPT            # 只在有真测试时才用 command_ok
