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
    assert "共 2 行" in out


def test_read_large_file_pages_by_default(tmp_path):
    ws = Workspace(tmp_path)
    big = "\n".join(str(i) for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    out = _run(ws, path="big.txt")
    assert "1\t1" in out
    assert "200\t200" in out
    assert "201\t201" not in out
    assert "共 500 行" in out and "已显示 1-200" in out


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


def test_read_path_traversal_rejected(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws, path="../../etc/passwd")
    assert out.startswith("❌")
    assert "越界" in out


def test_read_non_utf8_rejected(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
    out = _run(ws, path="bin.dat")
    assert out.startswith("❌")
    assert "UTF-8" in out
