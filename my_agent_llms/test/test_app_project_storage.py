"""app.py 按项目隔离 storage_dir,并设置用户层目录。"""
from pathlib import Path

from my_agent_llms.cli import app
from my_agent_llms.memory.project_root import (
    resolve_project_root, project_storage_dir, user_storage_dir, project_id,
)


def test_resolve_storage_dirs_helper(tmp_path):
    base = tmp_path / "base"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    proj_dir, user_dir, pid = app._resolve_storage_dirs(base, repo)
    assert proj_dir == project_storage_dir(base, resolve_project_root(repo))
    assert user_dir == user_storage_dir(base)
    assert pid == project_id(resolve_project_root(repo))
