"""项目根识别 + 存储目录派生(纯函数)。"""
from pathlib import Path

from my_agent_llms.memory.project_root import (
    resolve_project_root, project_id, project_storage_dir, user_storage_dir,
)


def test_resolve_walks_up_to_git(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    assert resolve_project_root(sub) == repo.resolve()


def test_resolve_falls_back_to_start_when_no_git(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    assert resolve_project_root(d) == d.resolve()


def test_project_id_stable_and_short(tmp_path):
    pid = project_id(tmp_path)
    assert pid == project_id(tmp_path)        # 稳定
    assert len(pid) == 16 and pid.isalnum()   # 16 位 hex


def test_storage_dirs(tmp_path):
    base = tmp_path / "base"
    root = tmp_path / "repo"
    root.mkdir()
    assert project_storage_dir(base, root) == base / "projects" / project_id(root)
    assert user_storage_dir(base) == base / "user"
