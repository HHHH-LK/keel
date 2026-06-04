"""就地工作区:root 默认 = 当前目录,而非自动建的空沙箱。"""
import os
from pathlib import Path

from my_agent_llms.workspace import Workspace


def test_root_none_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = Workspace()
    assert ws.root == tmp_path.resolve()


def test_explicit_root_still_honored(tmp_path):
    sub = tmp_path / "proj"
    sub.mkdir()
    ws = Workspace(sub)
    assert ws.root == sub.resolve()
