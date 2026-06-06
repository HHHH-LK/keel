"""验证规格:Check / CheckSpec 数据类,以及从任务推导规格的 SpecGenerator。

铁律:规格生成者 ≠ 任务执行者;生成的是"性质(答案必须满足什么)",不是具体答案。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Check:
    id: str
    type: str                      # string_contains|string_absent|field_equals|
                                   # command_ok|tool_called|judge|semantic_support
    params: dict
    weight: float = 1.0
    confidence: float = 1.0        # 规格生成器对该性质的置信度(伪 oracle 降权用)
    is_hard_oracle: bool = False   # True=可执行/解析类真 oracle;False=推导性质/judge


@dataclass
class CheckSpec:
    task: str
    checks: List[Check] = field(default_factory=list)


class SpecGenerator:
    """Placeholder — full implementation lands in Task 5."""
    def __init__(self, llm):
        self.llm = llm
