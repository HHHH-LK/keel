import pytest
from pathlib import Path

from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.edit_file import EditFile


def _make_tool(tmp_path) -> tuple[EditFile, Workspace]:
    ws = Workspace(tmp_path)
    return EditFile(ws), ws


def test_requires_approval_is_true(tmp_path):
    tool, _ = _make_tool(tmp_path)
    assert tool.requires_approval is True


def test_run_writes_to_disk(tmp_path):
    tool, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    out = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    assert out.startswith("✅")
    assert f.read_text() == "hello 2025\n"


def test_run_rejects_missing_match(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello\n")
    out = tool.run({"path": "a.md", "old_string": "xyz", "new_string": "abc"})
    assert out.startswith("❌")
    assert "找不到" in out


def test_run_rejects_multiple_matches(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hi\nhi\n")
    out = tool.run({"path": "a.md", "old_string": "hi", "new_string": "yo"})
    assert out.startswith("❌")


def test_run_rejects_missing_file(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.run({"path": "no.md", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")
    assert "不存在" in out


def test_run_rejects_traversal(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../etc", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")


def test_run_warns_on_no_change(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello\n")
    # old == new
    out = tool.run({"path": "a.md", "old_string": "hello", "new_string": "hello"})
    assert "⚠" in out or "无需" in out


def test_preview_for_approval_returns_unified_diff(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello 2024\n")
    preview = tool.preview_for_approval({
        "path": "a.md", "old_string": "2024", "new_string": "2025"
    })
    assert "---" in preview and "+++" in preview
    assert "-hello 2024" in preview or "-hello 2024\n" in preview
    assert "+hello 2025" in preview or "+hello 2025\n" in preview


def test_preview_for_approval_handles_missing_file(tmp_path):
    """文件不存在时,preview 应给出可读的错误而不是抛 —— Agent 已 catch 但工具自己别炸。"""
    tool, _ = _make_tool(tmp_path)
    preview = tool.preview_for_approval({
        "path": "noexist.md", "old_string": "x", "new_string": "y"
    })
    assert preview and isinstance(preview, str)


def test_run_uses_atomic_write(tmp_path):
    tool, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("x\n")
    tool.run({"path": "a.md", "old_string": "x", "new_string": "y"})
    leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftover == []
