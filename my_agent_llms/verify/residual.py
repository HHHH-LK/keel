"""残差聚合:未通过 check 的加权和。hard oracle 权重应压倒性高于推导性质。

三态语义(实跑发现坏 oracle 后引入):
- results[id] is True  → 通过,计入(贡献 0)
- results[id] is False → 未过,计入(贡献 weight*conf)
- results[id] is None  → SKIP(坏 oracle,如命令自身 SyntaxError),【不计入】,不罚 agent
- results 缺失 id      → 当作未过(False),计入(防漏检判 0)
residual == 0 且 effective_count > 0 ⟺ 所有【有效】check 通过。
"""
from __future__ import annotations

from typing import Dict, Optional

from my_agent_llms.verify.spec import CheckSpec


def residual(spec: CheckSpec, results: Dict[str, Optional[bool]]) -> float:
    total = 0.0
    for c in spec.checks:
        v = results.get(c.id, False)   # 缺失 → False(计入);显式 None → SKIP(不计入)
        if v is None:
            continue
        total += c.weight * c.confidence * (0.0 if v else 1.0)
    return total


def effective_count(spec: CheckSpec, results: Dict[str, Optional[bool]]) -> int:
    """有效(非 SKIP)的 check 数。全 SKIP → 0,用于'空验证不算收敛'边界。"""
    return sum(1 for c in spec.checks if results.get(c.id, False) is not None)
