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


def test_list_shows_origin_for_attached(tmp_path):
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    ws = Workspace(sandbox)
    src = outside / "origin.md"
    src.write_text("from outside")
    ws.attach(src)
    out = _run(ws)
    assert "origin.md" in out
    assert "←" in out or "→" in out  # origin marker
    assert "outside" in out  # 源路径片段


def test_list_excludes_manifest(tmp_path):
    """MANIFEST.json 是 sandbox 内部状态,不展示给 LLM。"""
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    ws = Workspace(sandbox)
    src = outside / "x.md"
    src.write_text("x")
    ws.attach(src)  # 这会创建 MANIFEST.json
    out = _run(ws)
    assert "MANIFEST.json" not in out
