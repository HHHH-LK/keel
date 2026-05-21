"""计算器工具：安全地求值算术表达式。"""
import ast
import math
import operator
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"不支持的函数: {node.func.id}")
        return func(*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"未定义的标识符: {node.id}")
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression:
        return "计算表达式不能为空"
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"计算失败:{exc}"


class CalculatorTool(Tool):
    """支持 +-*/、幂、取模、sqrt/log/sin/cos/tan 与 pi/e 常量。"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学计算工具，支持基本运算(+,-,*,/,**,%)与 sqrt/log/sin/cos/tan 等函数。",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = (
            parameters.get("expression")
            or parameters.get("input")
            or parameters.get("expr")
            or ""
        )
        return calculate(str(expression))

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="待求值的算术表达式",
                required=True,
            )
        ]
