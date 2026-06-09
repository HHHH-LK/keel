"""不可行动的 command_ok 连续判 False → 降级 SKIP,杜绝空转。

回归动机(见 transcript "答完又自证刷 5 轮"):SpecGenerator 给纯文档任务凭空编了
command_ok 硬 oracle,模型永远满足不了;反馈又是不可行动的通用文案(loop.py:69),
模型只能乱改别的文件让残差忽上忽下,骗过 STUCK,一路磨到 hard_cap。

对策:只对 command_ok(唯一反馈不可行动的硬 oracle)做"连续 N 轮 False → 降级 SKIP",
沿用既有"坏 oracle→SKIP"三态哲学。field_equals/tool_called/string_* 反馈可行动,不降级。
"""
from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.verify.convergence import Verdict, ConvergenceJudge
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.residual import residual, effective_count
from my_agent_llms.verify.stall import stalled_oracle_ids, apply_demotion


def _spec_cmd_and_str():
    return CheckSpec(task="t", checks=[
        Check(id="cmd", type="command_ok", params={"cmd": "false"}, weight=10.0),
        Check(id="str", type="string_contains", params={"s": "x"}, weight=1.0),
    ])


# ── 连续 False 才降级 ────────────────────────────────────────────
def test_one_false_round_not_demoted():
    spec = _spec_cmd_and_str()
    hist = [{"cmd": False, "str": True}]            # 仅 1 轮 → 还没到 stall 阈值
    assert stalled_oracle_ids(spec, hist) == set()


def test_two_consecutive_false_demotes():
    spec = _spec_cmd_and_str()
    hist = [{"cmd": False, "str": True},
            {"cmd": False, "str": True}]            # 连续 2 轮 False → 降级
    assert stalled_oracle_ids(spec, hist) == {"cmd"}


def test_recovered_command_ok_not_demoted():
    spec = _spec_cmd_and_str()
    hist = [{"cmd": False, "str": True},
            {"cmd": True, "str": True}]             # 第二轮过了 → 不降级
    assert stalled_oracle_ids(spec, hist) == set()


# ── 只降 command_ok,可行动的 check 不碰 ──────────────────────────
def test_actionable_check_never_demoted():
    """string_contains 反馈可行动(直接告诉模型缺哪个串),连续 False 也不降级。"""
    spec = _spec_cmd_and_str()
    hist = [{"cmd": True, "str": False},
            {"cmd": True, "str": False}]
    assert stalled_oracle_ids(spec, hist) == set()


# ── 降级后残差归零 → 可收敛(杜绝空转的关键) ────────────────────
def test_demotion_lets_residual_converge():
    spec = _spec_cmd_and_str()
    current = {"cmd": False, "str": True}
    demoted = {"cmd"}
    adjusted = apply_demotion(current, demoted)
    assert adjusted["cmd"] is None                  # 降级为 SKIP
    assert residual(spec, adjusted) == 0.0          # cmd 不再计入 → 残差 0
    assert effective_count(spec, adjusted) == 1     # str 仍是有效 check,非空验证


# ── replan 清 history 后仍降级(用原始 passed 历史,不被清空拖回空转) ──
def test_demotion_survives_history_window():
    """只要原始 passed 历史里末 N 轮 command_ok 都 False 就降级,与 Round 趋势历史无关。"""
    spec = _spec_cmd_and_str()
    hist = [{"cmd": False, "str": False},
            {"cmd": False, "str": True},            # str 抖动恢复,但 cmd 一直 False
            {"cmd": False, "str": True}]
    assert stalled_oracle_ids(spec, hist) == {"cmd"}


# ── 端到端:闸门里一个永远失败的 command_ok,2 轮内降级→收敛,不刷到 hard_cap ──
class _FixedRunner:
    """每轮返回固定结果(模拟不可行动 command_ok 永远 False + 一个能过的 check)。"""
    def __init__(self, results):
        self.results = results

    def run(self, spec, ctx):
        return dict(self.results)


def _gate_agent():
    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.workspace = None
    agent.checker_runner = _FixedRunner({"cmd": False, "str": True})
    agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=2)
    return agent


def test_gate_demotes_and_converges_within_two_rounds():
    agent = _gate_agent()
    spec = _spec_cmd_and_str()
    history, passed_hist = [], []

    # 第 0 轮:command_ok 失败 → 残差 10,还没降级 → CONTINUE(不停)
    g0 = agent._verify_gate("候选答案", [], spec, 0, history, None, passed_hist)
    assert g0["demoted"] == set()
    assert g0["stop"] is False

    # 第 1 轮:command_ok 连续 2 轮 False → 降级 SKIP → 残差归零 → CONVERGED → 停
    g1 = agent._verify_gate("候选答案", [], spec, 1, history, g0["best"], passed_hist)
    assert g1["demoted"] == {"cmd"}
    assert g1["best"].residual == 0.0
    assert g1["stop"] is True       # 2 轮就停,而非磨到 hard_cap=5

