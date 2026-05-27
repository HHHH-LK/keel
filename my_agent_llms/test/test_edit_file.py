import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.edit_file import EditFile


def _make_tool(tmp_path) -> tuple[EditFile, Workspace, PendingEditStore]:
    ws = Workspace(tmp_path)
    store = PendingEditStore()
    return EditFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m, f"no pending_id in: {text}"
    return m.group(1)


def test_propose_creates_pending_with_diff(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello 2024\nbye\n")
    out = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    assert "[待确认]" in out
    assert "2024" in out and "2025" in out
    pid = _pending_id(out)
    pe = store.pop(pid)
    assert pe is not None
    assert pe.kind == "edit"
    assert "2025" in pe.new_content
    assert pe.source_hash is not None  # 原文件存在,有 hash


def test_propose_rejects_missing_match(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello\n")
    out = tool.run({"path": "a.md", "old_string": "xyz", "new_string": "abc"})
    assert out.startswith("❌")
    assert "找不到" in out


def test_propose_rejects_multiple_matches(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hi\nhi\n")
    out = tool.run({"path": "a.md", "old_string": "hi", "new_string": "yo"})
    assert out.startswith("❌")
    assert "多处" in out or "匹配" in out


def test_propose_rejects_missing_file(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "no.md", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")
    assert "不存在" in out


def test_propose_rejects_traversal(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../etc", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")
