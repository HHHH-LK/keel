from typing import Dict, Optional

from my_agent_llms.core.agent import Agent
from my_agent_llms.core.config import Config
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.core.message import Message
from my_agent_llms.tools import ToolRegistry


class MyReflectionAgent(Agent):
    def __init__(self,
                 name: str,
                 llm: MyLLM,
                 tool_registry: ToolRegistry,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_steps: int = 5,
                 custom_prompt: Optional[str] = None,
                 enable_tool_calling: bool = False,
                 **kwargs):
        super().__init__(name, llm, system_prompt, config, **kwargs)
        self.tool_registry = tool_registry
        self.custom_prompt = custom_prompt
        self.max_steps = max_steps
        self.enable_tool_calling = enable_tool_calling

    def run(self, input_text: str, **kwargs) -> str:
        system_prompts = self.get_system_prompts()

        initial_prompt = self._apply_honesty_contract(
            system_prompts["initial"].format(task=input_text)
        )
        messages = self.memory.assemble_context(initial_prompt)
        messages.append({"role": "user", "content": input_text})

        response = self.llm.invoke(messages, **kwargs)

        for _ in range(self.max_steps):
            feedback = self.llm.invoke([
                {
                    "role": "system",
                    "content": system_prompts["reflect"].format(
                        task=input_text,
                        content=response
                    )
                }
            ], **kwargs)

            if self._is_good_enough(feedback):
                break

            response = self.llm.invoke([
                {
                    "role": "system",
                    "content": system_prompts["refine"].format(
                        task=input_text,
                        last_attempt=response,
                        feedback=feedback
                    )
                }
            ], **kwargs)

        response = self._run_response_hooks(input_text, response, messages)
        self._finalize_turn(input_text, response)
        return response

    @staticmethod
    def _is_good_enough(feedback: str) -> bool:
        return "无需改进" in feedback

    def get_system_prompts(self) -> Dict[str, str]:

        """获取反思智能体使用的提示词模板。"""
        DEFAULT_PROMPTS = {
            "initial": """
        请根据以下要求完成任务:

        任务: {task}

        请提供一个完整、准确的回答。
        """,
            "reflect": """
        请仔细审查以下回答，并找出可能的问题或改进空间:

        # 原始任务:
        {task}

        # 当前回答:
        {content}

        请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
        如果回答已经很好，请回答"无需改进"。
        """,
            "refine": """
        请根据反馈意见改进你的回答:

        # 原始任务:
        {task}

        # 上一轮回答:
        {last_attempt}

        # 反馈意见:
        {feedback}

        请提供一个改进后的回答。
        """
        }

        prompts = DEFAULT_PROMPTS.copy()

        if self.custom_prompt:
            prompts["initial"] = self.custom_prompt

        if self.system_prompt:
            prompts = {
                key: f"{self.system_prompt.strip()}\n\n{prompt.strip()}"
                for key, prompt in prompts.items()
            }

        return prompts
