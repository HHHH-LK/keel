"""残差聚合:未通过 check 的加权和。hard oracle 权重应压倒性高于推导性质。

residual == 0 ⟺ 所有 check 通过。
results 里缺失的 check id 一律当作"未通过"(防 KeyError + 漏检查不该判 0)。
"""
from __future__ import annotations

from typing import Dict

from my_agent_llms.verify.spec import CheckSpec


def residual(spec: CheckSpec, results: Dict[str, bool]) -> float:
    return sum(
        c.weight * c.confidence * (0.0 if results.get(c.id, False) else 1.0)
        for c in spec.checks
    )
