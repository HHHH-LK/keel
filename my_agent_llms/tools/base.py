from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具基类"""

    requires_approval: bool = False
    # 白名单标记:仅纯读/纯计算/无任何写集合的工具才置 True,允许同一轮并行执行。
    # 默认保守 False(碰 DB/文件/进程内状态/外部副作用/对人 IO 一律算有副作用)。
    side_effect_free: bool = False
    # verify 闸门豁免:本工具有副作用,但属于 agent 内部记账(记忆/待办),不是
    # "值得事后验证的产物"。置 True 后即便串行执行也不置 _turn_mutated,避免对
    # "没写代码的闲聊"凭空触发 verify。写文件/改代码/跑命令类工具保持 False。
    verify_exempt: bool = False

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass

    def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
        """生成给审批 UI 看的预览文本。默认 = repr(args)。
        写类工具应覆盖成 diff 字符串。"""
        return repr(parameters)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema。"""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.get_parameters():
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"
            if param.type == "array":
                prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
