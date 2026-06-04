"""ReadFile 读可越界:能读 CWD 外的文件。"""
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.read_file import ReadFile


def test_read_can_escape_root(tmp_path):
    outside = tmp_path.parent / "outside_read.txt"
    outside.write_text("hello\nworld\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    tool = ReadFile(ws)
    out = tool.run({"path": str(outside)})
    assert "hello" in out and "world" in out
    assert "❌" not in out
