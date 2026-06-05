"""Glob:模式匹配、mtime 倒序、跳 deny 目录、锁根子树、封顶。"""
import os
import time
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.glob import GlobTool


def test_matches_and_mtime_desc(tmp_path):
    (tmp_path / "old.py").write_text("a", encoding="utf-8")
    sub = tmp_path / "src"; sub.mkdir()
    (sub / "new.py").write_text("b", encoding="utf-8")
    old_t = time.time() - 100
    os.utime(tmp_path / "old.py", (old_t, old_t))
    out = GlobTool(Workspace(tmp_path)).run({"pattern": "**/*.py"})
    lines = out.split("\n")
    assert "src/new.py" in lines and "old.py" in lines
    assert lines.index("src/new.py") < lines.index("old.py")


def test_skips_deny_dirs(tmp_path):
    git = tmp_path / ".git"; git.mkdir()
    (git / "x.py").write_text("a", encoding="utf-8")
    (tmp_path / "real.py").write_text("b", encoding="utf-8")
    out = GlobTool(Workspace(tmp_path)).run({"pattern": "**/*.py"})
    assert "real.py" in out and ".git" not in out


def test_no_match(tmp_path):
    assert "无匹配" in GlobTool(Workspace(tmp_path)).run({"pattern": "**/*.zzz"})


def test_out_of_root_rejected(tmp_path):
    assert "❌" in GlobTool(Workspace(tmp_path)).run({"pattern": "*", "path": "../.."})
