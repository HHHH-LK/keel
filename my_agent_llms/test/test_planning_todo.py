"""规划层 todo 三件套单元测试。"""
from my_agent_llms.planning.todo import TodoStore, WriteTodoTool, todo_system_message, TODO_HEADING


def test_store_set_and_render():
    s = TodoStore()
    assert s.render() == ""                       # 空 → 空串
    s.set([{"content": "读配置", "status": "completed"},
           {"content": "改端口", "status": "in_progress"}])
    out = s.render()
    assert "## 当前任务清单(进度)" in out
    assert "[x] 读配置" in out
    assert "[~] 改端口" in out


def test_store_drops_empty_content():
    s = TodoStore()
    s.set([{"content": "  ", "status": "pending"}, {"content": "干活", "status": "pending"}])
    assert len(s.items) == 1


def test_write_tool_parses_status_content():
    s = TodoStore()
    tool = WriteTodoTool(s)
    out = tool.run({"todos": ["in_progress|读配置", "pending|改端口"]})
    assert "[~] 读配置" in out and "[ ] 改端口" in out
    assert s.items[0] == {"content": "读配置", "status": "in_progress"}


def test_write_tool_bare_line_is_content():
    s = TodoStore()
    WriteTodoTool(s).run({"todos": ["没有竖线的一条"]})
    assert s.items[0]["content"] == "没有竖线的一条"
    assert s.items[0]["status"] == "pending"


def test_write_tool_side_effect_not_free():
    assert WriteTodoTool(TodoStore()).side_effect_free is False   # 写状态 → 不可并行


def test_system_message_none_when_empty():
    assert todo_system_message(TodoStore()) is None               # 空 → 不注入(短任务零开销)


def test_system_message_present_when_nonempty():
    s = TodoStore(); s.set([{"content": "干活", "status": "pending"}])
    m = todo_system_message(s)
    assert m["role"] == "system" and "干活" in m["content"]
