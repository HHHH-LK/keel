"""收敛判定:三态(CONVERGED/OSCILLATING/STUCK)+ 双上限(soft/hard)+ 确定性指纹。

铁律:收敛 ≠ 正确;看窗口(最近 K 轮)不看相邻两轮,避免被微小抖动骗。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List


class Verdict(Enum):
    CONTINUE = auto()
    CONVERGED = auto()
    OSCILLATING = auto()
    STUCK = auto()
    MAX_STEPS = auto()


@dataclass
class Round:
    residual: float
    fingerprint: str


def fingerprint(result: str, trajectory: List[Dict[str, Any]]) -> str:
    """result 文本 + 工具调用名序列的确定性 hash。"""
    names: List[str] = []
    for msg in trajectory:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
    payload = (result or "") + "\x00" + "|".join(names)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class ConvergenceJudge:
    def __init__(self, *, soft_limit: int = 3, hard_cap: int = 5,
                 K: int = 2, eps: float = 1e-6):
        self.soft_limit = soft_limit
        self.hard_cap = hard_cap
        self.K = K
        self.eps = eps

    def judge(self, round_idx: int, residual: float, fingerprint: str,
              history: List[Round]) -> Verdict:
        # 1. 残差归零 → 收敛(优先级最高)
        if residual <= self.eps:
            return Verdict.CONVERGED
        # 2. 指纹重现 → 震荡(A→B→A 来回改)
        if any(h.fingerprint == fingerprint for h in history):
            return Verdict.OSCILLATING
        # 3. 最近 K 轮残差未严格下降 → 卡住(凑满 K 轮才判)
        window = [h.residual for h in history[-(self.K - 1):]] + [residual] \
            if self.K > 1 else [residual]
        if len(window) >= self.K and (window[0] - window[-1]) <= self.eps:
            return Verdict.STUCK
        # 4. 到硬上限 → 硬停
        if round_idx >= self.hard_cap - 1:
            return Verdict.MAX_STEPS
        # 5. 否则继续
        return Verdict.CONTINUE
