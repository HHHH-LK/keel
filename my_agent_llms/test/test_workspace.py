from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def test_explicit_root_is_used_when_exists(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.root == tmp_path.resolve()


def test_explicit_root_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace(tmp_path / "does_not_exist")


def test_none_root_creates_auto_sandbox(monkeypatch, tmp_path):
    # 把 HOME 指到 tmp_path,避免污染用户真实 home
    monkeypatch.setenv("HOME", str(tmp_path))
    ws = Workspace(None)
    assert ws.root.exists()
    # 用 resolve() 抹平 macOS /var → /private/var 的 symlink 差异
    expected_parent = (tmp_path / ".my_agent_llms" / "workspaces").resolve()
    assert ws.root.parent == expected_parent
    # 目录名: YYYYMMDD-HHMMSS-<6 位>
    name = ws.root.name
    assert len(name) == 22
    assert name[8] == "-"
    assert name[15] == "-"


def test_manifest_path_is_under_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.manifest_path == tmp_path.resolve() / "MANIFEST.json"
