import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.export_file import ExportFile


def _make_tool(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    ws = Workspace(sandbox)
    store = PendingEditStore()
    return ExportFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m
    return m.group(1)


def test_export_attached_uses_manifest_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    src = tmp_path / "report.md"
    src.write_text("v1\n")
    ws.attach(src)
    (ws.root / "report.md").write_text("v2\n")  # 在 sandbox 内改了

    propose = tool.run({"sandbox_path": "report.md"})
    assert "[待确认]" in propose
    assert str(src) in propose  # diff 里能看到目标路径
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert src.read_text() == "v2\n"


def test_export_new_file_requires_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "summary.md").write_text("brand new\n")
    out = tool.run({"sandbox_path": "summary.md"})
    assert out.startswith("❌")
    assert "dest" in out.lower() or "目标" in out


def test_export_new_file_with_explicit_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "summary.md").write_text("brand new\n")
    dest = tmp_path / "out" / "summary.md"
    dest.parent.mkdir()
    propose = tool.run({"sandbox_path": "summary.md", "dest_path": str(dest)})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert dest.read_text() == "brand new\n"


def test_export_rejects_deny_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "k.md").write_text("x\n")
    out = tool.run({"sandbox_path": "k.md", "dest_path": str(tmp_path / "secret.pem")})
    assert out.startswith("❌")
    assert "黑名单" in out


def test_export_apply_hash_guard(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    src = tmp_path / "a.md"
    src.write_text("v1\n")
    ws.attach(src)
    (ws.root / "a.md").write_text("v2\n")

    propose = tool.run({"sandbox_path": "a.md"})
    pid = _pending_id(propose)
    src.write_text("v1.5\n")  # 外部改原文件
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert src.read_text() == "v1.5\n"


def test_export_missing_sandbox_file(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"sandbox_path": "nope.md"})
    assert out.startswith("❌")
    assert "不存在" in out
