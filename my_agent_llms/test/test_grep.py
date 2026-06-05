"""Grep:rg 优先 + Python 兜底,content/files,锁根子树,封顶。"""
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.grep import GrepTool


def _ws(tmp_path):
    (tmp_path / "a.py").write_text("import os\nx = 1\ndef foo():\n    return os\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\nfoo bar\n", encoding="utf-8")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "c.py").write_text("foo = 2\n", encoding="utf-8")
    return Workspace(tmp_path)


def _grep(ws, **params):
    return GrepTool(ws).run(params)


def test_content_mode_has_path_line_text(tmp_path):
    out = _grep(_ws(tmp_path), pattern="foo")
    assert "a.py:3:" in out and "foo" in out
    assert "sub/c.py:1:" in out


def test_files_mode_lists_matching_files(tmp_path):
    out = _grep(_ws(tmp_path), pattern="foo", output_mode="files")
    assert "a.py" in out and "sub/c.py" in out and "b.txt" in out
    assert ":" not in out.replace("sub/", "")


def test_glob_filter(tmp_path):
    out = _grep(_ws(tmp_path), pattern="foo", glob="*.py", output_mode="files")
    assert "a.py" in out and "c.py" in out and "b.txt" not in out


def test_ignore_case(tmp_path):
    out = _grep(_ws(tmp_path), pattern="IMPORT", ignore_case=True)
    assert "import os" in out


def test_no_match(tmp_path):
    assert "无匹配" in _grep(_ws(tmp_path), pattern="zzz_nomatch_zzz")


def test_invalid_regex(tmp_path):
    assert "❌" in _grep(_ws(tmp_path), pattern="(unclosed")


def test_out_of_root_rejected(tmp_path):
    ws = _ws(tmp_path)
    assert "❌" in _grep(ws, pattern="x", path="../..")


def test_python_fallback_when_no_rg(tmp_path, monkeypatch):
    import my_agent_llms.tools.builtin.grep as g
    monkeypatch.setattr(g.shutil, "which", lambda _name: None)
    out = _grep(_ws(tmp_path), pattern="foo")
    assert "a.py:3:" in out


def test_python_fallback_context_no_duplicate_lines(tmp_path, monkeypatch):
    import my_agent_llms.tools.builtin.grep as g
    monkeypatch.setattr(g.shutil, "which", lambda _name: None)
    # 相邻两行都命中 + context=1 → 窗口重叠;不能输出重复行
    (tmp_path / "m.py").write_text("a0\nhit1\nhit2\na3\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    out = g.GrepTool(ws).run({"pattern": "hit", "context": 1, "glob": "m.py"})
    lines = out.split("\n")
    assert len(lines) == len(set(lines))          # 无重复
    assert sum("hit1" in l for l in lines) == 1    # hit1 只出现一次


def test_python_fallback_skips_binary(tmp_path, monkeypatch):
    import my_agent_llms.tools.builtin.grep as g
    monkeypatch.setattr(g.shutil, "which", lambda _name: None)
    (tmp_path / "bin.dat").write_bytes(b"foo\x00\x00foo")
    ws = Workspace(tmp_path)
    out = GrepTool(ws).run({"pattern": "foo", "output_mode": "files"})
    assert "bin.dat" not in out
