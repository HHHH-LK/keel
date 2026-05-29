import pytest
from pathlib import Path

from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.export_file import ExportFile


def _make_tool(tmp_path) -> tuple[ExportFile, Workspace]:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    ws = Workspace(sandbox)
    return ExportFile(ws), ws


def test_requires_approval_is_true(tmp_path):
    tool, _ = _make_tool(tmp_path)
    assert tool.requires_approval is True


def test_export_to_dest_path_new_file(tmp_path):
    tool, ws = _make_tool(tmp_path)
    (ws.root / "a.md").write_text("hello\n")
    dest = tmp_path / "out" / "a.md"
    out = tool.run({"sandbox_path": "a.md", "dest_path": str(dest)})
    assert out.startswith("✅")
    assert dest.read_text() == "hello\n"


def test_export_to_existing_dest_overwrites(tmp_path):
    tool, ws = _make_tool(tmp_path)
    (ws.root / "a.md").write_text("new\n")
    dest = tmp_path / "out.md"
    dest.write_text("old\n")
    out = tool.run({"sandbox_path": "a.md", "dest_path": str(dest)})
    assert out.startswith("✅")
    assert dest.read_text() == "new\n"


def test_export_missing_sandbox_file(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.run({"sandbox_path": "no.md", "dest_path": str(tmp_path / "x.md")})
    assert out.startswith("❌")
    assert "不存在" in out


def test_export_requires_dest_for_new_sandbox_file(tmp_path):
    """sandbox 内新建(无 origin)且不传 dest_path → 报错。"""
    tool, ws = _make_tool(tmp_path)
    (ws.root / "fresh.md").write_text("x\n")
    out = tool.run({"sandbox_path": "fresh.md"})  # no dest_path
    assert out.startswith("❌")
    assert "dest_path" in out or "源路径" in out


def test_preview_for_existing_dest_shows_diff(tmp_path):
    tool, ws = _make_tool(tmp_path)
    (ws.root / "a.md").write_text("new\n")
    dest = tmp_path / "out.md"
    dest.write_text("old\n")
    preview = tool.preview_for_approval({
        "sandbox_path": "a.md", "dest_path": str(dest)
    })
    assert "---" in preview and "+++" in preview
    assert "-old" in preview and "+new" in preview


def test_preview_for_new_dest_shows_byte_count(tmp_path):
    tool, ws = _make_tool(tmp_path)
    (ws.root / "a.md").write_text("hello\n")
    dest = tmp_path / "out.md"  # 不存在
    preview = tool.preview_for_approval({
        "sandbox_path": "a.md", "dest_path": str(dest)
    })
    assert "新建" in preview or "字节" in preview


def test_export_uses_atomic_write(tmp_path):
    tool, ws = _make_tool(tmp_path)
    (ws.root / "a.md").write_text("x\n")
    dest = tmp_path / "out.md"
    tool.run({"sandbox_path": "a.md", "dest_path": str(dest)})
    leftover = list(dest.parent.glob("*.tmp")) + list(dest.parent.glob(".*.tmp"))
    assert leftover == []
