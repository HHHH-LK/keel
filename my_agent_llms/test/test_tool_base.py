"""Tool 基类的审批相关默认行为。"""
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter


class _Dummy(Tool):
    def __init__(self):
        super().__init__("dummy", "test")
    def run(self, parameters: Dict[str, Any]) -> str:
        return "ok"
    def get_parameters(self) -> List[ToolParameter]:
        return []


def test_default_requires_approval_is_false():
    t = _Dummy()
    assert t.requires_approval is False


def test_default_preview_returns_repr_of_args():
    t = _Dummy()
    out = t.preview_for_approval({"path": "a.md", "n": 3})
    assert "path" in out and "a.md" in out and "n" in out and "3" in out


def test_subclass_can_override_requires_approval():
    class WriteLike(_Dummy):
        requires_approval = True
    assert WriteLike().requires_approval is True


def test_default_side_effect_free_is_false():
    """默认保守 = 不可并行,除非工具显式声明纯读。"""
    t = _Dummy()
    assert t.side_effect_free is False


def test_subclass_can_override_side_effect_free():
    class ReadLike(_Dummy):
        side_effect_free = True
    assert ReadLike().side_effect_free is True


def test_subclass_can_override_preview():
    class WriteLike(_Dummy):
        def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
            return f"will write to {parameters.get('path')}"
    assert WriteLike().preview_for_approval({"path": "x.txt"}) == "will write to x.txt"
