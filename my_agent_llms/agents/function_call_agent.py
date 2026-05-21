"""基于 OpenAI 原生 function calling 的 Agent。

与 ReActAgent 的区别在于工具调用通道：ReAct 走文本协议
(`[TOOL_CALL:xxx:yyy]` + 正则解析)，本 Agent 走 OpenAI 协议层
的 tools / tool_calls 字段，由模型保证返回合法 JSON 参数，鲁棒性更强。
"""
import json
from typing import Any, Dict, List, Optional, Union

from my_agent_llms.core.agent import Agent
from my_agent_llms.core.config import Config
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.core.message import Message
from my_agent_llms.tools.registry import ToolRegistry


class MyFunctionCallAgent(Agent):
    """使用 OpenAI 原生函数调用机制的 Agent。"""

    def __init__(self,
                 name: str,
                 llm: MyLLM,
                 tool_registry: ToolRegistry,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_steps: int = 5):
        super().__init__(name, llm, system_prompt, config)
        if llm.provider not in MyLLM.OPENAI_COMPATIBLE_PROVIDERS:
            raise ValueError(
                f"FunctionCallAgent 仅支持 OpenAI 兼容 provider，当前为: {llm.provider}"
            )
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self._install_memory_tools(self.tool_registry)

    def run(self,
            input_text: str,
            tool_choice: Union[str, dict] = "auto",
            **kwargs) -> str:
        messages: List[Dict[str, Any]] = list(
            self.memory.assemble_context(self.system_prompt)
        )
        messages.append({"role": "user", "content": input_text})

        tools = self._build_tool_schemas()

        final_response = ""
        for _ in range(self.max_steps):
            response = self._invoke_with_tools(messages, tools, tool_choice, **kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                final_response = self._extract_message_content(message)
                break

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                args = self._parse_function_call_arguments(tc.function.arguments)
                args = self._convert_parameter_types(tool_name, args)
                result = self.tool_registry.execute_tool(tool_name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        if not final_response:
            response = self._invoke_with_tools(messages, tools, "none", **kwargs)
            final_response = self._extract_message_content(response.choices[0].message)

        self._finalize_turn(input_text, final_response)
        print(f"✅ {self.name} 响应完成")
        return final_response

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.tool_registry.to_openai_schemas()

    def _invoke_with_tools(self,
                           messages: List[Dict[str, Any]],
                           tools: List[Dict[str, Any]],
                           tool_choice: Union[str, dict],
                           **kwargs):
        client = getattr(self.llm, "client", None)
        if client is None:
            raise RuntimeError("MyLLM 客户端未初始化，无法执行函数调用。")

        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            request_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **request_kwargs,
        )

    @staticmethod
    def _extract_message_content(message) -> str:
        return message.content or ""

    @staticmethod
    def _parse_function_call_arguments(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"input": raw}

    def _convert_parameter_types(self,
                                 tool_name: str,
                                 args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return args

        type_map = {p.name: p.type for p in tool.get_parameters()}
        return {
            key: self._coerce_value(value, type_map.get(key))
            for key, value in args.items()
        }

    @staticmethod
    def _coerce_value(value: Any, declared_type: Optional[str]) -> Any:
        if declared_type is None or value is None:
            return value
        try:
            if declared_type == "integer" and not isinstance(value, bool):
                return int(value)
            if declared_type == "number":
                return float(value)
            if declared_type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "1", "yes"}
                return bool(value)
            if declared_type == "string" and not isinstance(value, str):
                return str(value)
        except (ValueError, TypeError):
            return value
        return value
