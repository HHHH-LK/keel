from pathlib import Path
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.list_dir import ListDir


def _run(ws: Workspace, **params) -> str:
    return ListDir(ws).run(params)


def test_list_empty_sandbox(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws)
    assert "(空)" in out


def test_list_lists_files_with_size(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("12345")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("hi")
    out = _run(ws)
    assert "a.md" in out
    assert "5" in out  # size
    assert "sub/b.md" in out


def test_list_filters_by_pattern(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    out = _run(ws, pattern="*.md")
    assert "a.md" in out
    assert "b.txt" not in out
