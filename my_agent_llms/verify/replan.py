"""卡住时的"换思路"重新规划:带'卡在哪'生成新计划,只产计划不产答案。

铁律:计划只作引导(注入 messages),不严格解析步骤;执行仍走 verify,差计划会被挡下。
"""
from __future__ import annotations

_REPLAN_PROMPT = """你是规划助手。之前完成下面任务的尝试【卡住了】——以下验收项反复不通过。
请换一个【整体思路或步骤顺序】重新规划如何完成,不要重复上一种做法,也不要直接给出答案,
只输出简明的分步计划。

# 任务
{task}

# 反复没过的验收项(卡点)
{stuck}
"""


def make_plan(llm, task: str, stuck_feedback: str) -> str:
    """带卡点生成换思路的新计划文本。LLM 无回复时返回空串。"""
    prompt = _REPLAN_PROMPT.format(task=task, stuck=stuck_feedback)
    return llm.invoke([{"role": "system", "content": prompt}]) or ""
