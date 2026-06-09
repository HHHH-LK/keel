"""并发常驻 UI(Claude Code 风):底部圆角输入框常驻,agent 在后台线程跑,
输出经 patch_stdout(raw=True) 流进真 scrollback,活跃块在 app 内 diff 渲染。

仅 tty 启用;由 ChatCLI.run 调用(非 tty / MYAI_LEGACY_UI / 异常 → 回退串行)。

渲染分层(= Claude Code 的 Ink <Static> + 底部活跃区):
  - 完成块:ScrollbackRenderer 渲成 Rich Text → _to_ansi → print 进 scrollback(一次,不重画)
  - 正在生成的 thinking/正文:写 state["active"] + app.invalidate(),在 app 内 Window diff 渲染(不抖)

并发桥:
  - agent.run() 是同步阻塞 + 回调式 → 跑在 asyncio.to_thread 的工作线程
  - 回调(工作线程)→ print(线程安全 via patch_stdout 代理) + app.invalidate()(线程安全)
  - esc → key binding 置 cancel 标志 → agent.run(should_cancel=…) 逐 chunk/步级中断
  - 审批 → 工作线程把审批调度回事件循环(call_soon_threadsafe)并阻塞在 Future 上等结果

本阶段审批走临时 run_in_terminal(挂起 app 调现有审批框);正式浮层留 Phase 3。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import os
import shutil
import sys
import time
from functools import partial
from typing import Dict, Optional

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer, Float, FloatContainer, HSplit, VSplit, Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console as RichConsole
from rich.text import Text

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax

from . import chat_view, theme
from .markdown_render import render_markdown
from .permission import PermissionDecision
from .scrollback_renderer import ScrollbackRenderer, _is_error_result

_SPIN = ["✻", "✲", "✳", "✴", "✵", "✶", "✷"]
_APPROVAL_TIMEOUT_S = 300        # 审批 Future 兜底超时(到点 → 安全拒绝,防工作线程永挂)

# 审批选项(决定, 文案, 快捷键提示)—— 顺序即 ❯ 选择器上下导航顺序。
_APPROVAL_OPTIONS = [
    (PermissionDecision.ALLOW_ONCE,   "是,执行一次",        "1 / y / Enter"),
    (PermissionDecision.ALLOW_ALWAYS, "是,本会话不再询问",   "2 / a"),
    (PermissionDecision.DENY,         "否,拒绝",            "3 / n / Esc"),
]


# 会在审批前把【完整改动】落上方滚动区(终端可上滑看全部)的工具(diff 类)
_DIFF_TOOLS = {"Edit", "Write"}


def _looks_like_diff(preview: str) -> bool:
    return bool(preview) and preview.lstrip().startswith("--- ")


def _render_change_block(name: str, path: str, preview: str) -> Text:
    """把完整改动渲成【行号化的紧凑 diff 块】(⏺ 头 + ⎿ +N -M + 改动处),
    审批前落上方滚动区:既给审核看(可上滑看全部),也就是最终日志记录。"""
    added, removed, lines = chat_view._parse_diff_for_display(preview)
    out = Text()
    out.append("⏺ ", style=theme.AGENT)
    out.append(f"{name}({path})" if path else name, style=theme.DEFAULT)
    out.append("\n")
    out.append_text(chat_view._render_tool_diff(f"+{added} -{removed}", lines))
    return out


def _render_diff_tool_error(name: str, result: str) -> Text:
    out = Text()
    out.append("⏺ ", style=theme.ERR)
    out.append(name, style=theme.ERR)
    out.append(f"\n  ⎿  {result}", style=theme.ERR)
    return out


def _render_approval_box(name: str, sel: int, width: int) -> str:
    """审批浮层:只放标题 + ❯ 选择器(完整改动已落上方滚动区,框保持短、选项常驻)。
    返回带 ANSI 的字符串(喂 ptk)。"""
    body = Text()
    body.append("是否执行此操作?(完整改动见上方,可上滑查看)\n", style="bold")
    for i, (_dec, label, hint) in enumerate(_APPROVAL_OPTIONS):
        cur = i == sel
        body.append(" ❯ " if cur else "   ", style="cyan" if cur else "")
        body.append(f"{i + 1}. {label}", style="bold cyan" if cur else "")
        body.append(f"   [{hint}]\n", style=theme.DIM)

    buf = io.StringIO()
    con = RichConsole(file=buf, force_terminal=True, color_system="truecolor",
                      width=width)
    con.print(Panel(body, title=f"  {name}", title_align="left",
                    border_style="grey50"))
    return buf.getvalue().rstrip("\n")


def _width() -> int:
    return max(20, shutil.get_terminal_size((80, 24)).columns - 2)


def _to_ansi(text_obj: Text, w: int) -> str:
    """Rich Text → 带 ANSI 配色的字符串(供 print 进 scrollback / 包成 ANSI 喂 ptk)。"""
    buf = io.StringIO()
    RichConsole(file=buf, force_terminal=True, color_system="truecolor",
                width=w).print(text_obj)
    return buf.getvalue().rstrip("\n")


def _short_cwd() -> str:
    """当前目录,家目录缩成 ~。"""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    return "~" + cwd[len(home):] if cwd.startswith(home) else cwd


def _fmt_tokens(n: int) -> str:
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def _preview(args: Dict) -> str:
    """工具参数 → 'k=v, k=v' 预览(取前 3 个)。

    每个值折成单行 + 截到 40 字 —— 否则像 Write(content=整篇文件) 会把全文倒进
    ⏺ 工具行。完整内容由审批框/diff 渲染负责,这里只给一眼概览。
    """
    try:
        parts = []
        for k, v in list(args.items())[:3]:
            s = " ".join(str(v).split())          # 折叠换行/多空白成单行
            if len(s) > 40:
                s = s[:40] + "…"
            parts.append(f"{k}={s}")
        return ", ".join(parts)
    except Exception:
        return ""


class LiveSession:
    """一个常驻 prompt_toolkit Application;每条用户输入起一轮 agent.run(后台线程)。"""

    def __init__(self, cli):
        self.cli = cli                       # ChatCLI(持 self.agent / self.grants)
        self.state = {"busy": False, "spin": 0, "cancel": False,
                      "activity": "",              # 状态行动态文案(思考/调用工具/撰写)
                      "cwd": _short_cwd(),         # 底部信息栏:当前目录
                      "l1_tokens": 0,              # 当前 L1 上下文长度(每轮末刷新)
                      "sess_in": 0, "sess_out": 0}  # 本会话累计 token
        # 活跃块三元组单次原子赋值(src, mode, dot) —— 工作线程写、loop 线程读,
        # 用单一不可变快照避免多键 update 被 redraw 读到撕裂组合(I1)。
        self._active: tuple = ("", "text", False)
        self._pending_fut: "Optional[concurrent.futures.Future[bool]]" = None
        self.app: Optional[Application] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._real_out = sys.stdout          # 真 stdout(patch_stdout 前捕获,_main 再确认)

    # ── 渲染 sink(交给 ScrollbackRenderer)────────────────────
    # #4 重影根因:提交走 patch_stdout 的异步/批量/独立线程代理(缓冲到换行才刷、
    # 0.2s 批量),活跃区走即时 invalidate —— 二者乱序竞争同一底部变高区域,造成
    # 活跃区塌缩不擦(重影)+ 换行被吞(两轮粘连)。修法:两者都【按调用序排进
    # 事件循环】,提交经 run_in_terminal(擦 app→写 scrollback→重画 app)原子落地,
    # 且写真 stdout 绕开批量代理。
    def _commit(self, text_obj: Text) -> None:
        s = _to_ansi(text_obj, _width())
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._emit_scrollback, s)

    def _emit_scrollback(self, s: str) -> None:
        """在 loop 线程经 run_in_terminal 把一块文本原子写进真 scrollback。"""
        out = self._real_out

        def _write() -> None:
            out.write(s + "\n")
            out.flush()

        try:
            run_in_terminal(_write, in_executor=False)
        except Exception:
            pass

    def _set_active(self, src: str, mode: str, dot: bool) -> None:
        # 与 _commit 同走一条 loop 队列,严格保序(先提交块、后更新残块),
        # 不被代理刷新反超 → 残块不会盖到已提交块上/下方。
        loop = self._loop
        if loop is None:
            self._active = (src, mode, dot)
            return

        def _apply() -> None:
            self._active = (src, mode, dot)      # 单次原子赋值(I1)
            if self.app is not None:
                self.app.invalidate()

        loop.call_soon_threadsafe(_apply)

    # ── app 内 Window 内容回调 ────────────────────────────────
    def _active_fragments(self):
        src, mode, dot = self._active               # 一次读出快照,不会撕裂
        if not src:
            return []
        w = _width()
        if mode == "reasoning":
            framed = chat_view._render_thinking(src)
        else:
            body = render_markdown(src, w)
            framed = (chat_view._step_lines_from_text(body, theme.DEFAULT)
                      if not dot else chat_view._indent_only(body))
        return ANSI(_to_ansi(chat_view._tail_cap(framed, 18), w))

    def _status_fragments(self):
        if self.state["busy"]:
            spin = _SPIN[self.state["spin"] % len(_SPIN)]
            act = self.state.get("activity") or "生成中"   # 随 agent 当前动作动态变
            head = f"{spin} {act}…  (esc 中断)"
        else:
            head = "● 就绪"
        return [("class:status", f"  {head}")]

    # ── 固定 todo 面板:钉在状态行上方,随 store 实时更新 ──────────
    def _todo_store(self):
        return getattr(getattr(self.cli, "agent", None), "todo_store", None)

    def _has_todos(self) -> bool:
        store = self._todo_store()
        return bool(getattr(store, "items", None))

    def _maybe_clear_completed_todos(self) -> bool:
        """清单全完成 → 报一句"完成"并清空(固定面板随之消失);有未完成项则保留。

        关键:这要在【打完最后一个勾的那一刻】就触发,不能只等一轮收尾 —— 否则
        agent 答完后的尾巴(最终总结、verify 自证轮)会让 6/6 面板一直钉着像"卡住"。
        返回是否清空了。"""
        store = self._todo_store()
        items = getattr(store, "items", None)
        if items and all(it.get("status") == "completed" for it in items):
            self._commit(Text("  ⎿  ✓ 任务完成", style=theme.OK))
            store.items = []
            return True
        return False

    def _finalize_todo(self) -> None:
        """一轮收尾兜底:若全完成时没经 write_todo 的即时清空(异常/边界),这里再清一次。"""
        self._maybe_clear_completed_todos()

    def _todo_fragments(self):
        """非空 → 渲成面板的 ANSI;空 → []。filter 已先挡掉空,这里二次兜底。"""
        store = self._todo_store()
        items = getattr(store, "items", None)
        if not items:
            return []
        w = _width()
        panel = chat_view.render_todo_panel(items, width=w)
        if panel is None:
            return []
        return ANSI(_to_ansi(panel, w))

    def _info_bar_fragments(self):
        """输入框下方信息栏:当前目录 · 模型 · 上下文 L1/上限 · 本会话 token。"""
        cfg = getattr(self.cli, "cfg", {}) or {}
        parts = [self.state["cwd"]]
        model = cfg.get("model")
        if model:
            parts.append(model)
        parts.append(f"ctx {self.state['l1_tokens']}/{_fmt_tokens(4000)}")
        parts.append(f"{_fmt_tokens(self.state['sess_in'])}↑ "
                     f"{_fmt_tokens(self.state['sess_out'])}↓")
        return [("class:infobar", "  " + "  ·  ".join(parts))]

    # ── 审批:应用内浮层(无嵌套 app)────────────────────────────
    def _approval_fragments(self):
        """审批浮层内容:圆角框(只放选项)+ ❯ 选择器。空闲返回 []。"""
        appr = self.state.get("approval")
        if not appr:
            return []
        sel = self.state.get("appr_sel", 0)
        return ANSI(_render_approval_box(appr["name"], sel, _width()))

    def _commit_change_review(self, name: str, args: Dict, preview: str) -> bool:
        """diff 类工具:审批【前】就把完整改动落上方滚动区(终端可上滑看全部),
        它同时就是最终日志记录。返回是否落了(供调用方决定收尾文案)。"""
        if not _looks_like_diff(preview):
            return False
        path = str((args or {}).get("path", "")).strip()
        self._commit(_render_change_block(name, path, preview))
        return True

    def _on_permission(self, name: str, args: Dict, preview: str) -> bool:
        """工作线程调用:命中授权台账直接放行;否则在事件循环弹浮层,阻塞等 Future。"""
        try:
            if self.cli.grants.is_granted(name, args):
                self._commit_change_review(name, args, preview)
                return True
        except Exception:
            pass
        if self._loop is None:
            return False
        # 完整改动【审批前】就落上方滚动区:审核时可上滑看全部,且就是日志记录,
        # 审批框只剩选项(短、不和 todo 挤)。批准后不再重复展示(见 _on_tresult)。
        reviewed = self._commit_change_review(name, args, preview)
        fut: "concurrent.futures.Future[bool]" = concurrent.futures.Future()
        self._pending_fut = fut                       # 记录,退出时由 _main 解锁(C1)
        appr = {"fut": fut, "name": name, "args": args, "preview": preview}

        def _show():
            self.state["approval"] = appr
            self.state["appr_sel"] = 0       # 每次新审批从首项(执行一次)起选
            if self.app is not None:
                self.app.invalidate()

        self._loop.call_soon_threadsafe(_show)
        allowed = False
        try:
            allowed = fut.result(timeout=_APPROVAL_TIMEOUT_S)
        except Exception:
            allowed = False
        finally:
            self._pending_fut = None
        if reviewed and not allowed:
            self._commit(Text("  ⎿  (已拒绝,未改动)", style=theme.WARN))
        return allowed

    def _resolve_approval(self, decision: "PermissionDecision") -> None:
        """主循环按键回调:落定审批,唤醒工作线程。"""
        appr = self.state.get("approval")
        if not appr:
            return
        self.state["approval"] = None
        if decision is PermissionDecision.ALLOW_ALWAYS:
            try:
                self.cli.grants.grant(appr["name"], appr["args"])
            except Exception:
                pass
            ok = True
        elif decision is PermissionDecision.ALLOW_ONCE:
            ok = True
        else:
            ok = False
        fut = appr["fut"]
        if not fut.done():
            fut.set_result(ok)
        if self.app is not None:
            self.app.invalidate()

    def _is_read_only(self, name: str) -> bool:
        """工具是否只读(side_effect_free)—— 决定 ⏺ 上色:只读中性,改动类绿/红。"""
        fn = getattr(self.cli.agent, "_tool_is_side_effect_free", None)
        try:
            return bool(fn(name)) if fn else False
        except Exception:
            return False

    def _renders_inline(self, name: str) -> bool:
        """该工具是否在 scrollback 里内联渲染 ⏺。

        否 = 它已在别处展示,日志不再重复:Edit/Write 的 diff 在审批区落过,
        write_todo 的清单在固定任务清单面板常驻 —— 都不内联,免得一份出现两次。"""
        return name not in _DIFF_TOOLS and name != "write_todo"

    def _notify_not_ready(self) -> None:
        """未配置(agent=None)时的友好提示,替代崩 'NoneType'.run。"""
        self._commit(chat_view._continuation_lines(
            "还没配置模型 —— 用 /config key 设置 API Key 后再聊。", theme.WARN))

    # ── 一轮:在后台线程跑 agent.run ──────────────────────────
    def _run_turn(self, user_input: str) -> None:
        # 兜底:agent 没构建成(没配置 key)→ 给提示,别让 agent.run 在 None 上崩。
        if getattr(self.cli, "agent", None) is None:
            self._notify_not_ready()
            self.state["busy"] = False
            if self.app is not None:
                self.app.invalidate()
            return
        r = ScrollbackRenderer(self._commit, self._set_active, _width)
        start = time.monotonic()
        tok = {"in": 0, "out": 0}

        def on_llm_done(elapsed, pt, ct):
            tok["in"] += pt or 0
            tok["out"] += ct or 0

        # 回调包一层:在转给渲染器的同时刷新状态行的动态文案(思考/调用工具/撰写)。
        def _on_text(t):
            self.state["activity"] = "撰写回复"
            r.text_chunk(t)

        def _on_reason(t):
            self.state["activity"] = "思考中"
            r.reasoning_chunk(t)

        def _on_tcall(n, a):
            self.state["activity"] = f"调用 {n}"
            # 已在别处展示的工具不再走 renderer 的内联 ⏺(否则一处出现两次):
            # diff 类(Edit/Write)在审批区;write_todo 在固定任务清单面板。
            if self._renders_inline(n):
                r.tool_call(n, _preview(a), self._is_read_only(n))

        def _on_tresult(n, res, el):
            if n in _DIFF_TOOLS:
                # 成功:diff 块即记录,不重复;只在【出错】时补一行(绕开 renderer FIFO)
                if _is_error_result(res):
                    self._commit(_render_diff_tool_error(n, res))
            elif n == "write_todo":
                # 固定面板已展示;成功时日志不内联(免得一份清单出现两次),
                # 仅在【出错】时补一行,否则用户看不到 todo 没更新成功。
                if _is_error_result(res):
                    self._commit(chat_view._continuation_lines(
                        f"❌ write_todo: {res}", theme.ERR))
                else:
                    # 刚打完最后一个勾 → 立刻清空收尾,别等整轮结束(总结/verify 的
                    # 尾巴会让 6/6 面板一直钉着,看着像"卡住没消失")。
                    self._maybe_clear_completed_todos()
            else:
                r.tool_result(res, name=n, read_only=self._is_read_only(n),
                              elapsed_sec=el)
            self.state["activity"] = "生成中"

        # 自证阶段:本轮只打一次归属头(多轮重试也只标一次,后续核对都归在它下面)。
        verify_announced = {"v": False}

        def _on_verify_phase(_round):
            self.state["activity"] = "自证完成"
            if not verify_announced["v"]:
                r.verify_notice()
                verify_announced["v"] = True

        def _on_verify_start():
            # 候选答案此刻被压制(不显示),闸门正在核对 → 状态行提示"校验中"。
            self.state["activity"] = "校验中"
            if self.app is not None:
                self.app.invalidate()

        try:
            self.cli.agent.run(
                user_input,
                on_text_chunk=_on_text,
                on_reasoning_chunk=_on_reason,
                on_tool_call=_on_tcall,
                on_tool_result=_on_tresult,
                on_permission_request=self._on_permission,
                on_llm_done=on_llm_done,
                on_verify_phase=_on_verify_phase,
                on_verify_start=_on_verify_start,
                should_cancel=lambda: self.state["cancel"],
            )
        except Exception as exc:
            self._commit(chat_view._continuation_lines(f"❌ {exc}", theme.ERR))
        if self.state["cancel"]:
            buf = io.StringIO()
            con = RichConsole(file=buf, force_terminal=True,
                              color_system="truecolor", width=_width())
            chat_view.render_system_notice(con, "warn", "已中断(esc)")
            self._commit(Text.from_ansi(buf.getvalue().rstrip("\n")))
        r.close(tools_used=getattr(self.cli.agent, "last_tool_call_count", 0),
                elapsed_seconds=time.monotonic() - start,
                tokens_in=tok["in"], tokens_out=tok["out"])
        self._finalize_todo()      # 全完成 → 报"完成"并清空,固定面板随之消失
        # 底部信息栏统计:本会话累计 token + 当前 L1 上下文长度(此刻在 worker 线程,
        # agent.run 已结束,顺序读 memory 安全;不在每次重绘时读,避免与生成并发)
        self.state["sess_in"] += tok["in"]
        self.state["sess_out"] += tok["out"]
        try:
            self.state["l1_tokens"] = self.cli.agent.memory.stats().get("l1_tokens", 0)
        except Exception:
            pass
        self.state["activity"] = ""        # 收尾清空动态文案 → 回到 ● 就绪
        self.state["busy"] = False
        if self.app is not None:
            self.app.invalidate()

    async def _worker(self, queue: "asyncio.Queue[str]"):
        try:
            while True:
                line = await queue.get()
                self.state.update(busy=True, cancel=False)
                if self.app is not None:
                    self.app.invalidate()
                await asyncio.to_thread(self._run_turn, line)   # agent 同步阻塞 → 线程
                queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _spinner_loop(self):
        try:
            while True:
                if self.state["busy"]:
                    self.state["spin"] += 1
                    if self.app is not None:
                        self.app.invalidate()
                await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            pass

    def _make_input_area(self):
        """单行输入框,挂 SlashCompleter(打 '/' 弹命令菜单,与旧框一致)。
        不折行(否则中文双宽字符很快撑到行尾,框被折成两行),改横向滚动。"""
        from .completer import SlashCompleter
        return TextArea(multiline=False, wrap_lines=False,
                        completer=SlashCompleter(), complete_while_typing=True,
                        prompt=HTML("<arrow>❯ </arrow>"))

    # ── app 构建 ─────────────────────────────────────────────
    def _build_app(self, queue: "asyncio.Queue[str]"):
        ta = self._make_input_area()
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            # 补全菜单开着时:Enter 接受当前候选 / 关菜单,不提交(与旧框一致)
            buf = ta.buffer
            if buf.complete_state:
                cc = buf.complete_state.current_completion
                if cc is not None:
                    buf.apply_completion(cc)
                else:
                    buf.cancel_completion()
                return
            text = ta.text.strip()
            ta.text = ""
            if not text:
                return
            if text in ("/quit", "/exit"):
                event.app.exit()
                return
            self._emit_scrollback(f"\n❯ {text}")  # 回显用户输入进 scrollback(同一原子路径)
            if text.startswith("/"):
                # slash 命令:挂起 app 调现有 handle_command(临时;交互式命令体验留后续)。
                # run_in_terminal 返回的 future 故意 fire-and-forget(同步命令即可)。
                cmd = text
                run_in_terminal(lambda: self.cli.handle_command(cmd))
                return
            # 未配置 key(agent=None):别排队(否则 worker 在 None 上崩),直接提示
            if getattr(self.cli, "agent", None) is None:
                self._notify_not_ready()
                event.app.invalidate()
                return
            queue.put_nowait(text)               # 普通输入 → 排队给 agent
            event.app.invalidate()

        # ── 审批浮层按键(仅审批弹起时生效,eager 抢在输入框前拦截)──
        appr_on = Condition(lambda: self.state.get("approval") is not None)

        @kb.add("up", filter=appr_on, eager=True)
        def _(event):
            n = len(_APPROVAL_OPTIONS)
            self.state["appr_sel"] = (self.state.get("appr_sel", 0) - 1) % n
            event.app.invalidate()

        @kb.add("down", filter=appr_on, eager=True)
        def _(event):
            n = len(_APPROVAL_OPTIONS)
            self.state["appr_sel"] = (self.state.get("appr_sel", 0) + 1) % n
            event.app.invalidate()

        @kb.add("enter", filter=appr_on, eager=True)
        def _(event):
            self._resolve_approval(
                _APPROVAL_OPTIONS[self.state.get("appr_sel", 0)][0])

        @kb.add("1", filter=appr_on, eager=True)
        @kb.add("y", filter=appr_on, eager=True)
        def _(event):
            self._resolve_approval(PermissionDecision.ALLOW_ONCE)

        @kb.add("2", filter=appr_on, eager=True)
        @kb.add("a", filter=appr_on, eager=True)
        def _(event):
            self._resolve_approval(PermissionDecision.ALLOW_ALWAYS)

        @kb.add("3", filter=appr_on, eager=True)
        @kb.add("n", filter=appr_on, eager=True)
        @kb.add("escape", filter=appr_on, eager=True)
        def _(event):
            self._resolve_approval(PermissionDecision.DENY)

        @kb.add("escape", filter=~appr_on)
        def _(event):
            if self.state["busy"]:
                self.state["cancel"] = True

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            # 退出前置 cancel,让在跑的 agent.run 经 should_cancel 尽快收尾,
            # 否则不可取消的 to_thread 会拖住 asyncio.run 关停(C1)。
            self.state["cancel"] = True
            event.app.exit()

        # 活跃区:空闲时整窗隐藏(否则空内容仍占 1 行/重绘时高度抖动,看着像闪)。
        active = ConditionalContainer(
            content=Window(FormattedTextControl(self._active_fragments),
                           wrap_lines=True, dont_extend_height=True,
                           height=D(min=0, max=12)),
            filter=Condition(lambda: bool(self._active[0])))
        status = Window(FormattedTextControl(self._status_fragments), height=1)
        # 固定 todo 面板:有清单才显示,钉在状态行上方、输入框附近
        todo = ConditionalContainer(
            content=Window(FormattedTextControl(self._todo_fragments),
                           dont_extend_height=True),
            filter=Condition(self._has_todos))
        spacer = Window(height=1)        # 状态行与上方对话之间留一行空隙(别太贴)
        info = Window(FormattedTextControl(self._info_bar_fragments), height=1)
        # 审批浮层:仅审批弹起时显示,在输入框上方
        approval = ConditionalContainer(
            content=Window(FormattedTextControl(self._approval_fragments),
                           dont_extend_height=True),
            filter=appr_on)
        fill = partial(Window, style="class:frame.border")
        top = VSplit([fill(width=1, height=1, char="╭"), fill(char="─"),
                      fill(width=1, height=1, char="╮")], height=1)
        bottom = VSplit([fill(width=1, height=1, char="╰"), fill(char="─"),
                         fill(width=1, height=1, char="╯")], height=1)
        # 钉 height=1:跟 top/bottom 一致,否则两侧 │ 竖条会竖向撑高、把输入框抻成多行。
        middle = VSplit([fill(width=1, char="│"), Window(width=1), ta,
                         Window(width=1), fill(width=1, char="│")], height=1)
        # todo 面板在上(上下文),spacer 隔开,审批浮层紧贴状态行/输入框(动作处),
        # 信息栏在框下方。todo 与审批之间有空隙,不再连成两个框。
        root = HSplit([active, todo, spacer, approval, status,
                       HSplit([top, middle, bottom]), info])
        # 包一层 FloatContainer:打 '/' 时命令补全菜单浮在输入框上方(回归旧框体验)。
        root = FloatContainer(
            content=root,
            floats=[Float(xcursor=True, ycursor=True,
                          content=CompletionsMenu(max_height=8, scroll_offset=1))])
        style = Style.from_dict({"arrow": "magenta", "frame.border": "gray",
                                 "status": "gray", "infobar": "#666666"})
        return Application(layout=Layout(root, focused_element=ta),
                           key_bindings=kb, style=style,
                           full_screen=False, mouse_support=False)

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._real_out = sys.stdout          # patch_stdout 前的真 stdout,提交直写它绕开批量代理
        queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.app = self._build_app(queue)
        with patch_stdout(raw=True):
            spin = asyncio.create_task(self._spinner_loop())
            work = asyncio.create_task(self._worker(queue))
            try:
                await self.app.run_async()
            finally:
                # 退出时若工作线程正阻塞在审批 Future 上,立刻解为 False,
                # 否则它会一直等到 _APPROVAL_TIMEOUT_S 才返回,拖住关停(C1)。
                pend = self._pending_fut
                if pend is not None and not pend.done():
                    pend.set_result(False)
                spin.cancel()
                work.cancel()

    def run(self) -> None:
        self.cli.print_banner()
        asyncio.run(self._main())
        if self.cli.agent is not None:
            try:
                self.cli.agent.memory.close()
            except Exception:
                pass


def run(cli) -> None:
    """入口:ChatCLI.run 在 tty 路径调用。"""
    LiveSession(cli).run()
