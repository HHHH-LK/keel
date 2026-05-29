import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.write_file import WriteFile


def _make_tool(tmp_path) -> tuple[WriteFile, Workspace]:
    ws = Workspace(tmp_path)
    return WriteFile(ws), ws


def test_requires_approval_is_true(tmp_path):
    tool, _ = _make_tool(tmp_path)
    assert tool.requires_approval is True


def test_run_creates_new_file(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.run({"path": "b.md", "content": "hello\n"})
    assert out.startswith("✅")
    assert (tmp_path / "b.md").read_text() == "hello\n"


def test_run_overwrites_existing_file(tmp_path):
    tool, _ = _make_tool(tmp_path)
    f = tmp_path / "b.md"
    f.write_text("old\n")
    out = tool.run({"path": "b.md", "content": "new\n"})
    assert out.startswith("✅")
    assert f.read_text() == "new\n"


def test_run_rejects_traversal(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../x", "content": "y"})
    assert out.startswith("❌")


def test_run_rejects_directory_path(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "sub").mkdir()
    out = tool.run({"path": "sub", "content": "x"})
    assert out.startswith("❌")
    assert "目录" in out


def test_run_warns_on_no_change(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "b.md").write_text("same\n")
    out = tool.run({"path": "b.md", "content": "same\n"})
    assert "⚠" in out or "无需" in out


def test_preview_existing_file_returns_diff(tmp_path):
    tool, _ = _make_tool(tmp_path)
    (tmp_path / "b.md").write_text("old\n")
    preview = tool.preview_for_approval({"path": "b.md", "content": "new\n"})
    assert "---" in preview and "+++" in preview
    assert "-old" in preview
    assert "+new" in preview


def test_preview_new_file_shows_full_content_as_diff(tmp_path):
    """新建文件:像 Claude Code 那样,展示要写入的实际内容(全 + 行 diff)。"""
    tool, _ = _make_tool(tmp_path)
    preview = tool.preview_for_approval(
        {"path": "new.md", "content": "line one\nline two\n"}
    )
    assert "new.md" in preview
    assert "---" in preview and "+++" in preview  # unified diff header
    assert "+line one" in preview
    assert "+line two" in preview


def test_run_uses_atomic_write(tmp_path):
    tool, _ = _make_tool(tmp_path)
    tool.run({"path": "b.md", "content": "x"})
    leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftover == []
