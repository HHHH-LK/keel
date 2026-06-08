"""#4 重影根因修复的契约测试:_commit 与 _set_active 必须【按序经同一条 loop 队列】。

真重影/粘连只在 tty 复现、无法自动测;但根因是"提交走 patch_stdout 异步批量代理、
活跃区走即时 invalidate,两者乱序竞争"。这里锁住修复的不变量:两个操作都不在调用线程
直接动终端,而是按调用顺序排进 loop —— commit 先、set_active 后。
"""
import io

from rich.text import Text

from my_agent_llms.cli.live_session import LiveSession


class _FakeLoop:
    def __init__(self):
        self.scheduled = []

    def call_soon_threadsafe(self, fn, *args):
        self.scheduled.append((getattr(fn, "__name__", "fn"), args))


def _bare_session():
    sess = LiveSession.__new__(LiveSession)   # 跳过 __init__,只装契约需要的字段
    sess._loop = _FakeLoop()
    sess.app = None
    sess._real_out = io.StringIO()
    sess._active = ("", "text", False)
    return sess


def test_commit_schedules_through_loop_not_direct_print():
    sess = _bare_session()
    sess._commit(Text("完成块"))
    assert len(sess._loop.scheduled) == 1
    assert sess._loop.scheduled[0][0] == "_emit_scrollback"


def test_commit_then_set_active_preserve_order_on_loop():
    sess = _bare_session()
    sess._commit(Text("完成块"))
    sess._set_active("残块", "text", True)
    names = [n for n, _ in sess._loop.scheduled]
    # 都进了 loop,且 commit 排在 set_active 前面(不会被反超 → 不粘连/不重影)
    assert len(names) == 2
    assert names[0] == "_emit_scrollback"


def test_preview_truncates_long_values():
    from my_agent_llms.cli.live_session import _preview
    p = _preview({"path": "LICENSE", "content": "M" * 200})
    assert "LICENSE" in p
    assert len(p) <= 120          # 不再把整篇文件内容塞进 ⏺ 工具行
    assert "…" in p


def test_preview_collapses_newlines_in_values():
    from my_agent_llms.cli.live_session import _preview
    p = _preview({"content": "第一行\n第二行\n第三行"})
    assert "\n" not in p          # 多行折成单行,不撑高工具行


def test_status_busy_shows_dynamic_activity():
    sess = LiveSession.__new__(LiveSession)
    sess.state = {"busy": True, "spin": 0, "activity": "调用 Read"}
    text = "".join(t for _, t in sess._status_fragments())
    assert "调用 Read" in text          # 随 agent 当前动作变
    assert "esc" in text                # 仍提示可中断


def test_status_busy_falls_back_when_no_activity():
    sess = LiveSession.__new__(LiveSession)
    sess.state = {"busy": True, "spin": 0, "activity": ""}
    text = "".join(t for _, t in sess._status_fragments())
    assert "生成中" in text             # 没有具体动作时回退到"生成中"


def test_status_idle_shows_ready():
    sess = LiveSession.__new__(LiveSession)
    sess.state = {"busy": False, "spin": 0, "activity": ""}
    text = "".join(t for _, t in sess._status_fragments())
    assert "就绪" in text


def test_set_active_defers_state_mutation_to_loop():
    sess = _bare_session()
    sess._set_active("残块", "text", True)
    # 活跃区状态不在调用线程直接改,而是排进 loop(由 loop 线程一并 set+invalidate)
    assert sess._active == ("", "text", False)     # 尚未应用
    assert len(sess._loop.scheduled) == 1


# ── not ready(agent=None)时输入不崩,给友好提示 ──────────────────
from types import SimpleNamespace


def test_run_turn_without_agent_shows_notice_not_crash():
    sess = LiveSession.__new__(LiveSession)
    sess.cli = SimpleNamespace(agent=None)
    sess.state = {"busy": True}
    sess.app = None
    committed = []
    sess._commit = lambda text_obj: committed.append(text_obj)
    sess._run_turn("帮我看看这个项目")          # agent 是 None,不能抛 AttributeError
    assert sess.state["busy"] is False          # 收尾置回空闲
    joined = "".join(getattr(t, "plain", str(t)) for t in committed)
    assert "config" in joined.lower() or "配置" in joined   # 提示去配置


# ── "/" 命令菜单:completer 接进 live 输入框(回归:旧框有/live 漏了)──
def test_input_area_wires_slash_completer():
    from my_agent_llms.cli.completer import SlashCompleter
    sess = LiveSession.__new__(LiveSession)
    ta = sess._make_input_area()
    assert isinstance(ta.completer, SlashCompleter)


def test_build_app_constructs_with_completion_menu():
    sess = LiveSession.__new__(LiveSession)
    sess.state = {"busy": False, "spin": 0, "activity": "", "cwd": "~",
                  "l1_tokens": 0, "sess_in": 0, "sess_out": 0}
    app = sess._build_app(None)            # queue 只在 enter 闭包里用,构建期不碰
    assert app is not None
