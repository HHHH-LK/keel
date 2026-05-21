import re
from typing import Optional

from my_agent_llms.core.agent import Agent
from my_agent_llms.core.config import Config
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.core.message import Message
from my_agent_llms.tools.registry import ToolRegistry


class MyReActAgent(Agent):
    def __init__(self,
                 name: str,
                 llm: MyLLM,
                 tool_registry: ToolRegistry,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_steps: int = 5,
                 custom_prompt: Optional[str] = None,
                 enable_tool_calling: bool = False):
        super().__init__(name, llm, system_prompt, config)
        self.name = name
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.config = config
        self.max_steps = max_steps
        self.custom_prompt = custom_prompt
        self.enable_tool_calling = enable_tool_calling
        if self.enable_tool_calling:
            self._install_memory_tools(self.tool_registry)

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        # 通过 MemoryManager 拼装 system + L2 摘要 + L1 历史
        system_prompt = self.get_system_prompt()
        messages = self.memory.assemble_context(system_prompt)
        messages.append({"role": "user", "content": input_text})
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages)
            self._finalize_turn(input_text, response)
            return response

        return self.run_with_tool_use(self, input_text, messages, max_tool_iterations, **kwargs)

    def get_system_prompt(self) -> str | None:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    @staticmethod
    def run_with_tool_use(self, input_text, messages, max_tool_iterations, **kwargs) -> str:
        current_iteration = 0
        final_response = ""
        while current_iteration < max_tool_iterations:
            response = self.llm.think(messages, **kwargs)
            # 解析出response中的工具调用
            tool_lists = self._parse_tool_calls(response)
            tool_response = []
            clean_tool_response = response

            # 判断是否存在工具调用
            if tool_lists:
                # 执行工具调用
                for tool in tool_lists:
                    tool_name = tool['tool_name']
                    tool_parameters = tool['parameters']
                    # 调用工具
                    tool_call = self._execute_tool_call(tool_name, tool_parameters)
                    tool_response.append(tool_call)
                    clean_tool_response = clean_tool_response.replace(tool['original'], "")

                # 添加进会话中
                messages.append({"role": "assistant", "content": clean_tool_response})
                tool_results_text = "\n\n".join(tool_response)
                messages.append(
                    {"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})
                current_iteration += 1
                continue

            # 没有工具调用则直接回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self._finalize_turn(input_text, final_response)
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"

        try:
            # 智能参数解析
            if tool_name == 'calculator':
                # 计算器工具直接传入表达式
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # 其他工具使用智能参数解析
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误:未找到工具 '{tool_name}'"
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数:action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数:key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict
