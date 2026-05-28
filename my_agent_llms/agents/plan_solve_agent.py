import re
from typing import Dict, List, Optional, Tuple

from my_agent_llms.core.agent import Agent
from my_agent_llms.core.config import Config
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.core.message import Message
from my_agent_llms.tools import ToolRegistry


class MyPlanSolveAgent(Agent):
    """Plan-and-Solve Agent：先做整体规划，再分步执行并自检。"""

    def __init__(self,
                 name: str,
                 llm: MyLLM,
                 tool_registry: ToolRegistry,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 custom_prompt: Optional[str] = None,
                 max_retries: int = 2,
                 enable_tool_calling: bool = False,
                 **kwargs):
        super().__init__(name, llm, system_prompt, config, **kwargs)
        self.tool_registry = tool_registry
        self.custom_prompt = custom_prompt
        self.max_retries = max_retries
        self.enable_tool_calling = enable_tool_calling

    def run(self, input_text: str, **kwargs) -> str:
        prompts = self.get_system_prompts()

        # 1) 规划阶段：让模型给出分步计划
        plan_system = self._apply_honesty_contract(
            prompts["plan"].format(task=input_text)
        )
        plan_messages = self.memory.assemble_context(plan_system)
        plan_messages.append({"role": "user", "content": input_text})

        plan = self.plan(plan_messages, **kwargs)
        steps = self.parse_plan_steps(plan)

        # 没解析到步骤时，直接基于计划文本走一次最终汇总
        if not steps:
            final = self.llm.invoke([
                {"role": "system", "content": prompts["final"].format(
                    task=input_text, plan=plan, step_results="(无可执行步骤)"
                )}
            ], **kwargs)
            final = self._run_response_hooks(input_text, final, plan_messages)
            self._finalize_turn(input_text, final)
            return final

        # 2) 执行阶段：逐步求解，附带自检与有限重试
        step_results: List[str] = []
        for index, step in enumerate(steps, start=1):
            previous = self._format_previous_results(steps, step_results)
            attempt_result = self.do_plan(
                plan=plan,
                step_index=index,
                step_text=step,
                previous_results=previous,
                input_text=input_text,
                prompts=prompts,
                **kwargs,
            )

            retries = 0
            while retries < self.max_retries:
                is_ok, feedback = self.re_check(
                    step_text=step,
                    step_result=attempt_result,
                    input_text=input_text,
                    prompts=prompts,
                    **kwargs,
                )
                if is_ok:
                    break
                retries += 1
                attempt_result = self.do_plan(
                    plan=plan,
                    step_index=index,
                    step_text=step,
                    previous_results=previous,
                    input_text=input_text,
                    prompts=prompts,
                    feedback=feedback,
                    **kwargs,
                )

            step_results.append(attempt_result)

        # 3) 汇总阶段：根据各步骤结果给出最终答复
        final_answer = self.llm.invoke([
            {"role": "system", "content": prompts["final"].format(
                task=input_text,
                plan=plan,
                step_results=self._format_previous_results(steps, step_results),
            )}
        ], **kwargs)

        final_answer = self._run_response_hooks(input_text, final_answer, plan_messages)
        self._finalize_turn(input_text, final_answer)
        return final_answer

    def plan(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """调用 LLM 生成整体计划。"""
        return self.llm.invoke(messages, **kwargs)

    def do_plan(self,
                plan: str,
                step_index: int,
                step_text: str,
                previous_results: str,
                input_text: str,
                prompts: Dict[str, str],
                feedback: Optional[str] = None,
                **kwargs) -> str:
        """执行计划中的某一步。"""
        solve_prompt = prompts["solve"].format(
            task=input_text,
            plan=plan,
            step_index=step_index,
            step_text=step_text,
            previous_results=previous_results or "(暂无)",
            feedback=feedback or "(无)",
        )
        return self.llm.invoke([{"role": "system", "content": solve_prompt}], **kwargs)

    def re_check(self,
                 step_text: str,
                 step_result: str,
                 input_text: str,
                 prompts: Dict[str, str],
                 **kwargs) -> Tuple[bool, str]:
        """让 LLM 自检步骤结果是否满足要求。"""
        check_prompt = prompts["check"].format(
            task=input_text,
            step_text=step_text,
            step_result=step_result,
        )
        feedback = self.llm.invoke([{"role": "system", "content": check_prompt}], **kwargs)
        return self._is_step_ok(feedback), feedback

    @staticmethod
    def _is_step_ok(feedback: str) -> bool:
        return "通过" in feedback or "无需改进" in feedback

    @staticmethod
    def parse_plan_steps(plan: str) -> List[str]:
        """从计划文本中解析出有序步骤。支持 `1.`、`1)`、`- `、`步骤1:` 等常见格式。"""
        if not plan:
            return []

        steps: List[str] = []
        pattern = re.compile(r"^\s*(?:步骤\s*\d+[:：.\)]|\d+[\.、)]|-)\s*(.+)$")
        for line in plan.splitlines():
            match = pattern.match(line)
            if match:
                content = match.group(1).strip()
                if content:
                    steps.append(content)
        return steps

    def get_plan_step(self, plan: str) -> int:
        return len(self.parse_plan_steps(plan))

    def get_system_prompts(self) -> Dict[str, str]:
        """获取 Plan-and-Solve 各阶段的提示词模板。"""
        DEFAULT_PROMPTS = {
            "plan": """
你是一个善于规划的助手。请针对下面的任务，给出一个清晰、可执行的分步计划。

任务: {task}

要求:
1. 用编号列表（例如 1. 2. 3.）列出每一步骤;
2. 每一步聚焦一个小目标，避免过于宽泛;
3. 不要直接给出最终答案，只输出步骤列表。
""",
            "solve": """
你正在按计划逐步完成任务，请只完成当前这一步。

# 原始任务
{task}

# 整体计划
{plan}

# 之前步骤的结果
{previous_results}

# 当前需要完成的步骤（第 {step_index} 步）
{step_text}

# 上一次自检反馈（如有）
{feedback}

请基于以上信息，给出本步骤的执行结果。只输出当前步骤的结果，不要重复整个计划，也不要给出最终答案。
""",
            "check": """
请判断下面这一步是否已经被令人满意地完成。

# 原始任务
{task}

# 当前步骤
{step_text}

# 步骤执行结果
{step_result}

如果结果已经合格，请直接回答"通过"。
否则请简要指出问题，并给出改进建议。
""",
            "final": """
请基于以下信息，给出对原始任务的最终答复。

# 原始任务
{task}

# 整体计划
{plan}

# 各步骤的执行结果
{step_results}

请整合上述内容，给出一个完整、准确、连贯的答案。
""",
        }

        prompts = DEFAULT_PROMPTS.copy()

        if self.custom_prompt:
            prompts["plan"] = self.custom_prompt

        if self.system_prompt:
            prompts = {
                key: f"{self.system_prompt.strip()}\n\n{prompt.strip()}"
                for key, prompt in prompts.items()
            }

        return prompts

    @staticmethod
    def _format_previous_results(steps: List[str], results: List[str]) -> str:
        if not results:
            return ""
        lines = []
        for idx, result in enumerate(results, start=1):
            step_text = steps[idx - 1] if idx - 1 < len(steps) else ""
            lines.append(f"第{idx}步 - {step_text}\n结果: {result}")
        return "\n\n".join(lines)
