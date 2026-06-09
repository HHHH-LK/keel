"""停滞 oracle 降级:不可行动的 command_ok 连续判 False → 降级为 SKIP。

为什么只降 command_ok:
    它是唯一反馈【不可行动】的硬 oracle —— feedback_from(loop.py) 故意不回灌命令原文
    (防模型自己重跑命令污染对话),结果模型根本不知道哪条命令、为什么挂,无法对症。
    field_equals/tool_called/string_* 的反馈都点名了文件/工具/字符串,模型能改;只有
    command_ok 失败时模型是"睁眼瞎"。一个改不动的 check 不该一直逼着重答 → 连续 N 轮
    False 就判定"该任务下它是坏 oracle",沿用既有"坏 oracle→SKIP"三态哲学,降级不计残差。

权衡:若 command_ok 是真测试(如 pytest)且模型确实 N 轮没修好,会被误降级 → 可能假收敛。
    但(a)command_ok 反馈本就不可行动、重试近乎徒劳;(b)仍返回全程 best、verdict 照常;
    (c)N=2 给了一次真实修正机会。可接受。根治需让 command_ok 反馈可行动(回灌 stderr
    摘要而非命令),那是另一层(闸门反馈层)的事。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from my_agent_llms.verify.spec import CheckSpec

# 连续判 False 多少轮(含当前轮)就降级。2 = 给模型一次真实修正机会后仍 False 才放弃。
DEFAULT_STALL_ROUNDS = 2


def stalled_oracle_ids(spec: CheckSpec, passed_history: List[Dict[str, Optional[bool]]],
                       *, stall_rounds: int = DEFAULT_STALL_ROUNDS) -> Set[str]:
    """从【原始】每轮 passed 历史里挑出该降级的 command_ok id。

    passed_history: 每轮 checker_runner.run 的【原始】结果(末尾是当前轮),不被 replan
        清空 —— 因此即便 Round 趋势历史被 clear,降级判定仍稳定生效,不会被拖回空转。
    返回:末 stall_rounds 轮都【显式 False】的 command_ok id 集合。
    """
    if stall_rounds < 1 or len(passed_history) < stall_rounds:
        return set()
    recent = passed_history[-stall_rounds:]
    demoted: Set[str] = set()
    for c in spec.checks:
        if c.type != "command_ok":
            continue
        if all(r.get(c.id) is False for r in recent):
            demoted.add(c.id)
    return demoted


def apply_demotion(passed: Dict[str, Optional[bool]],
                   demoted: Set[str]) -> Dict[str, Optional[bool]]:
    """把降级 id 在 passed 副本里置 None(SKIP),供 residual/effective_count/feedback 使用。"""
    if not demoted:
        return dict(passed)
    return {k: (None if k in demoted else v) for k, v in passed.items()}
