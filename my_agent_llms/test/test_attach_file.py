from pathlib import Path
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.attach_file import AttachFile


def _run(ws: Workspace, **params) -> str:
    return AttachFile(ws).run(params)


def _setup(tmp_path) -> tuple[Workspace, Path]:
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    return Workspace(sandbox), outside


def test_attach_happy_path(tmp_path):
    ws, outside = _setup(tmp_path)
    src = outside / "report.md"
    src.write_text("hello")
    out = _run(ws, source_path=str(src))
    assert out.startswith("✅")
    assert "report.md" in out
    assert (ws.root / "report.md").read_text() == "hello"
    assert ws.manifest()["report.md"] == str(src.resolve())


def test_attach_missing_source(tmp_path):
    ws, outside = _setup(tmp_path)
    out = _run(ws, source_path=str(outside / "no_such.md"))
    assert out.startswith("❌")
    assert "不存在" in out


def test_attach_deny_source(tmp_path):
    ws, outside = _setup(tmp_path)
    bad = outside / "key.pem"
    bad.write_text("KEY")
    out = _run(ws, source_path=str(bad))
    assert out.startswith("❌")
    assert "黑名单" in out


def test_attach_already_exists(tmp_path):
    ws, outside = _setup(tmp_path)
    (ws.root / "report.md").write_text("x")
    src = outside / "report.md"
    src.write_text("y")
    out = _run(ws, source_path=str(src))
    assert out.startswith("❌")
    assert "已有" in out
