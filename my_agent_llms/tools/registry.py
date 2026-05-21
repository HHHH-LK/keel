from typing import Any, Callable, Optional, Dict

from my_agent_llms.tools.base import Tool


class ToolRegistry:
    """工具注册表：同时支持 Tool 子类对象与轻量函数注册。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        if tool.name in self._tools:
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        if name in self._functions:
            print(f"⚠️ 警告:工具 '{name}' 已存在，将被覆盖。")
        self._functions[name] = {"description": description, "func": func}
        print(f"✅ 工具 '{name}' 已注册。")

    def unregister(self, name: str) -> bool:
        removed = False
        if name in self._tools:
            del self._tools[name]
            removed = True
        if name in self._functions:
            del self._functions[name]
            removed = True
        return removed

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys()) + list(self._functions.keys())

    def execute_tool(self, name: str, params: Any) -> str:
        """执行工具。params 可以是 dict 或 str。"""
        if name in self._functions:
            func = self._functions[name]["func"]
            if isinstance(params, dict):
                arg = params.get("input") or params.get("query") or params.get("expression") or ""
                arg = str(arg)
            else:
                arg = str(params)
            return func(arg)

        if name in self._tools:
            tool = self._tools[name]
            if isinstance(params, dict):
                return tool.run(params)
            # 字符串参数：尝试常见键名传给 Tool.run
            return tool.run({"expression": params, "input": params, "query": params})

        return f"❌ 未找到工具 '{name}'"

    def get_tools_description(self) -> str:
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def to_openai_schemas(self) -> list[Dict[str, Any]]:
        """汇总所有已注册工具的 OpenAI function calling schema。

        Tool 子类调用其自身的 to_openai_schema；轻量函数没有参数声明，
        统一暴露成一个 string 类型的 input 参数。
        """
        schemas: list[Dict[str, Any]] = []

        for tool in self._tools.values():
            schemas.append(tool.to_openai_schema())

        for name, info in self._functions.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "工具输入",
                            },
                        },
                        "required": ["input"],
                    },
                },
            })

        return schemas
