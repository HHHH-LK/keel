from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def test_explicit_root_is_used_when_exists(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.root == tmp_path.resolve()


def test_explicit_root_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace(tmp_path / "does_not_exist")


def test_none_root_uses_cwd(monkeypatch, tmp_path):
    # 新语义: root=None 时工作区根 = 当前工作目录
    monkeypatch.chdir(tmp_path)
    ws = Workspace(None)
    assert ws.root == tmp_path.resolve()


def test_resolve_relative_path_in_root(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "report.md").write_text("x")
    p = ws.resolve("report.md")
    assert p == (tmp_path / "report.md").resolve()


def test_resolve_absolute_path_in_root(tmp_path):
    ws = Workspace(tmp_path)
    abs_in = (tmp_path / "sub" / "a.md")
    abs_in.parent.mkdir()
    abs_in.write_text("x")
    p = ws.resolve(str(abs_in))
    assert p == abs_in.resolve()


def test_resolve_allows_not_yet_existing(tmp_path):
    """WriteFile 要写新文件,resolve 必须允许尚未存在的路径。"""
    ws = Workspace(tmp_path)
    p = ws.resolve("new_file.md")
    assert p == (tmp_path / "new_file.md").resolve()
    assert not p.exists()


def test_resolve_blocks_traversal(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("../../etc/passwd")


def test_resolve_blocks_absolute_outside(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("/etc/passwd")


def test_resolve_blocks_symlink_escape(tmp_path):
    ws = Workspace(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("link.txt")


def test_resolve_blocks_broken_symlink_escape(tmp_path):
    """悬空符号链接也必须拦截 —— 否则 LLM 写入会逃逸到外部。"""
    ws = Workspace(tmp_path)
    link = tmp_path / "dangling.txt"
    link.symlink_to(tmp_path.parent / "nonexistent_outside.txt")
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("dangling.txt")


def test_resolve_blocks_deny_dir(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    with pytest.raises(WorkspaceViolation, match="黑名单"):
        ws.resolve(".git/HEAD")


def test_resolve_blocks_deny_suffix(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="黑名单"):
        ws.resolve("server.pem")


def test_relative_strips_root(tmp_path):
    ws = Workspace(tmp_path)
    sub = (tmp_path / "a" / "b.md")
    sub.parent.mkdir()
    sub.write_text("x")
    assert ws.relative(sub.resolve()) == "a/b.md"
