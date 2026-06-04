from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.read_file import ReadFile


def _run(ws: Workspace, **params) -> str:
    return ReadFile(ws).run(params)


def test_read_small_file_returns_numbered_lines(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("hello\nworld\n")
    out = _run(ws, path="a.md")
    assert "1\thello" in out
    assert "2\tworld" in out
    assert out.startswith("Read 2 lines")


def test_read_medium_file_no_paging_by_default(tmp_path):
    """默认上限 2000 行,500 行的文件一次读完(Claude Code Read 一致)。"""
    ws = Workspace(tmp_path)
    big = "\n".join(str(i) for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    out = _run(ws, path="big.txt")
    assert "1\t1" in out
    assert "500\t500" in out
    assert out.startswith("Read 500 lines\n")


def test_read_very_large_file_pages_at_default_limit(tmp_path):
    """文件超过默认 2000 行时,首屏只读前 2000 行;首行带 offset/total 提示。"""
    ws = Workspace(tmp_path)
    huge = "\n".join(str(i) for i in range(1, 2501)) + "\n"
    (tmp_path / "huge.txt").write_text(huge)
    out = _run(ws, path="huge.txt")
    assert "2000\t2000" in out
    assert "2001\t2001" not in out
    assert out.startswith("Read 2000 lines (offset 0, total 2500)")


def test_read_with_offset_and_limit(tmp_path):
    ws = Workspace(tmp_path)
    big = "\n".join(str(i) for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    out = _run(ws, path="big.txt", offset=100, limit=5)
    assert "101\t101" in out
    assert "105\t105" in out
    assert "106\t106" not in out


def test_read_missing_file(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws, path="nope.md")
    assert out.startswith("❌")
    assert "不存在" in out


def test_read_directory_rejected(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "sub").mkdir()
    out = _run(ws, path="sub")
    assert out.startswith("❌")
    assert "目录" in out


def test_read_path_traversal_no_longer_boundary_error(tmp_path):
    """resolve_read 允许读 CWD 外;../../etc/passwd 不再返回越界,
    但 /etc/passwd 在测试机上通常不存在,仍以 ❌ 文件不存在 结束。
    (原行为:严格 resolve 返回 '越界';新行为:宽松 resolve_read 只查黑名单)"""
    ws = Workspace(tmp_path)
    out = _run(ws, path="../../etc/passwd")
    # 路径合法(未命中黑名单);若文件存在则成功读取,不存在则返回不存在错误
    # 关键断言:不再是越界错误
    assert "越界" not in out


def test_read_non_utf8_rejected(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
    out = _run(ws, path="bin.dat")
    assert out.startswith("❌")
    assert "UTF-8" in out
