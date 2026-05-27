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


def test_apply_writes_to_disk(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    # 注意 pop is destructive; 重新 put 以模拟 store 状态
    # 真实流程里 propose 完后 pid 还在 store 里,apply 时才 pop
    # _pending_id 只解析字符串,没动 store —— 这里其实没问题
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert "已修改" in out
    assert f.read_text() == "hello 2025\n"


def test_cancel_discards(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "cancel"})
    assert "已取消" in out or "丢弃" in out.lower()
    assert f.read_text() == "hello 2024\n"
    # 再次 apply 同一 pid 应失败
    out2 = tool.run({"pending_id": pid, "action": "apply"})
    assert out2.startswith("❌")


def test_apply_unknown_pid(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"pending_id": "deadbeef", "action": "apply"})
    assert out.startswith("❌")
    assert "过期" in out or "不存在" in out


def test_apply_rejects_if_file_changed_externally(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    # 模拟外部编辑
    f.write_text("hello 2024 (touched)\n")
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert "外部修改" in out or "hash" in out.lower()
    # 文件保持外部修改后的内容,没被覆盖
    assert "(touched)" in f.read_text()


def test_apply_uses_atomic_write(tmp_path):
    """apply 落盘后,sandbox 根下不应残留 .tmp 文件。"""
    tool, ws, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("x\n")
    propose = tool.run({"path": "a.md", "old_string": "x", "new_string": "y"})
    pid = _pending_id(propose)
    tool.run({"pending_id": pid, "action": "apply"})
    leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftover == []
