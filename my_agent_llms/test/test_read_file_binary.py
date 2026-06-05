"""ReadFile 拒读二进制/图片类文件。"""
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.read_file import ReadFile


def test_rejects_svg_suffix(tmp_path):
    (tmp_path / "d.svg").write_text("<svg></svg>", encoding="utf-8")
    out = ReadFile(Workspace(tmp_path)).run({"path": "d.svg"})
    assert "❌" in out and "二进制" in out


def test_rejects_null_byte_file(tmp_path):
    (tmp_path / "x.dat").write_bytes(b"abc\x00def")
    out = ReadFile(Workspace(tmp_path)).run({"path": "x.dat"})
    assert "❌" in out


def test_still_reads_normal_text(tmp_path):
    (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
    out = ReadFile(Workspace(tmp_path)).run({"path": "ok.py"})
    assert "print(1)" in out and "❌" not in out
