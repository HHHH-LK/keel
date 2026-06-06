"""判定任务是否走 TDD:用户开关优先,否则模型自报;任何异常 → 不走 TDD(降级)。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TddDecision:
    use_tdd: bool
    reason: str


_PROMPT = """判断下面这个任务是否适合"先写测试再写实现"(TDD)。
只有"能写出可执行测试的代码任务"(写函数/类/模块、修可复现的 bug)才适合。
闲聊、问答、写文案、改配置、纯解释 → 不适合。
只输出 JSON: {{"use_tdd": true/false, "reason": "一句话"}}

任务: {task}"""


def classify(llm, task: str, user_override: Optional[bool] = None) -> TddDecision:
    if user_override is not None:
        return TddDecision(use_tdd=user_override, reason="user override")
    try:
        content = llm.invoke(
            [{"role": "system", "content": _PROMPT.format(task=task)}]) or ""
        data = _parse_json(content)
        return TddDecision(
            use_tdd=bool(data.get("use_tdd", False)),
            reason=str(data.get("reason", "")))
    except Exception as exc:
        logger.warning("TDD classify 失败,降级为不走 TDD: %s", exc)
        return TddDecision(use_tdd=False, reason=f"classify 降级: {exc}")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}
