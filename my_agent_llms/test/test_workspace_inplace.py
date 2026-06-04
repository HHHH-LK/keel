"""就地工作区:root 默认 = 当前目录,而非自动建的空沙箱。"""
import os
from pathlib import Path

import pytest

from my_agent_llms.workspace import Workspace, WorkspaceViolation


def test_root_none_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = Workspace()
    assert ws.root == tmp_path.resolve()


def test_explicit_root_still_honored(tmp_path):
    sub = tmp_path / "proj"
    sub.mkdir()
    ws = Workspace(sub)
    assert ws.root == sub.resolve()


def test_resolve_write_blocks_escape(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation):
        ws.resolve("../outside.txt")


def test_resolve_read_allows_escape(tmp_path):
    outside = tmp_path.parent / "neighbor.txt"
    outside.write_text("hi", encoding="utf-8")
    ws = Workspace(tmp_path)
    p = ws.resolve_read(str(outside))
    assert p == outside.resolve()


def test_resolve_read_blocks_key_suffix(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation):
        ws.resolve_read("secret.key")   # 后缀黑名单对读也生效


def test_resolve_read_blocks_git_dir(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation):
        ws.resolve_read(".git/config")  # 目录段黑名单对读也生效


def test_relative_outside_root_returns_abspath(tmp_path):
    ws = Workspace(tmp_path)
    outside = (tmp_path.parent / "x.txt").resolve()
    assert ws.relative(outside) == str(outside)
