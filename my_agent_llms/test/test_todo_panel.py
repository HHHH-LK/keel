"""固定 todo 面板渲染(chat_view.render_todo_panel)+ live_session 接线。"""
import io

from rich.console import Console

from my_agent_llms.cli import chat_view


def _plain(panel) -> str:
    """无色渲染,纯文本内容断言(ANSI 码不会割断子串)。"""
    buf = io.StringIO()
    Console(file=buf, color_system=None, width=80).print(panel)
    return buf.getvalue()


def _ansi(panel) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor",
            width=60).print(panel)
    return buf.getvalue()


def test_panel_none_when_empty():
    assert chat_view.render_todo_panel([]) is None


def test_panel_marks_three_states():
    items = [{"content": "已完成步", "status": "completed"},
             {"content": "进行步", "status": "in_progress"},
             {"content": "待办步", "status": "pending"}]
    out = _plain(chat_view.render_todo_panel(items))
    assert "✓ 已完成步" in out      # 完成勾
    assert "◐ 进行步" in out        # 当前步标记
    assert "☐ 待办步" in out        # 待办
    assert "1/3" in out             # 进度计数(完成 1 / 共 3)


def test_panel_current_step_highlighted_cyan():
    items = [{"content": "进行步", "status": "in_progress"}]
    out = _ansi(chat_view.render_todo_panel(items))
    # cyan = ANSI 36;当前步必须带 cyan 上色(高亮)
    assert "36" in out


def test_panel_all_done_border_green():
    items = [{"content": "唯一步", "status": "completed"}]
    out = _plain(chat_view.render_todo_panel(items))
    assert "✓ 唯一步" in out and "1/1" in out


# ── live_session 接线:_todo_fragments 空→[],非空→ANSI(含内容) ──
def _session_with(store):
    from types import SimpleNamespace
    from my_agent_llms.cli.live_session import LiveSession
    sess = LiveSession.__new__(LiveSession)
    sess.cli = SimpleNamespace(agent=SimpleNamespace(todo_store=store))
    return sess


def test_todo_fragments_empty_when_no_items():
    from my_agent_llms.planning.todo import TodoStore
    assert _session_with(TodoStore())._todo_fragments() == []


def test_todo_fragments_renders_items():
    from my_agent_llms.planning.todo import TodoStore
    s = TodoStore()
    s.set([{"content": "干活步骤", "status": "in_progress"}])
    frags = _session_with(s)._todo_fragments()
    assert frags != []
    assert "干活步骤" in frags.value          # ANSI.value 含清单内容


def test_has_todos_flag():
    from my_agent_llms.planning.todo import TodoStore
    s = TodoStore()
    assert _session_with(s)._has_todos() is False
    s.set([{"content": "x", "status": "pending"}])
    assert _session_with(s)._has_todos() is True


