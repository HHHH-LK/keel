import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.write_file import WriteFile


def _make_tool(tmp_path):
    ws = Workspace(tmp_path)
    store = PendingEditStore()
    return WriteFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m, f"no pending_id in: {text}"
    return m.group(1)


def test_write_new_file_proposes_with_no_diff(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "new.md", "content": "hello"})
    assert "[待确认]" in out
    assert "新建文件" in out


def test_write_new_file_applies(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    propose = tool.run({"path": "new.md", "content": "hello"})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert (tmp_path / "new.md").read_text() == "hello"


def test_write_overwrite_shows_diff(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("old\n")
    out = tool.run({"path": "a.md", "content": "new\n"})
    assert "[待确认]" in out
    assert "old" in out and "new" in out


def test_write_overwrite_same_content_noop(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("same\n")
    out = tool.run({"path": "a.md", "content": "same\n"})
    assert out.startswith("⚠️")
    assert "相同" in out


def test_write_rejects_traversal(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../bad.md", "content": "x"})
    assert out.startswith("❌")


def test_write_apply_hash_guard(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("v1\n")
    propose = tool.run({"path": "a.md", "content": "v2\n"})
    pid = _pending_id(propose)
    (tmp_path / "a.md").write_text("v1.5\n")  # 外部改
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert (tmp_path / "a.md").read_text() == "v1.5\n"  # 未被覆盖
