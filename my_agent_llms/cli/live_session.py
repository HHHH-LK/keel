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
import time
from functools import partial
from typing import Dict, Optional

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer, HSplit, VSplit, Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console as RichConsole
from rich.text import Text

from rich.panel import Panel
from rich.syntax import Syntax

from . import chat_view, theme
from .markdown_render import render_markdown
from .permission import PermissionDecision, _looks_like_diff
from .scrollback_renderer import ScrollbackRenderer

_SPIN = ["✻", "✲", "✳", "✴", "✵", "✶", "✷"]
_APPROVAL_TIMEOUT_S = 300        # 审批 Future 兜底超时(到点 → 安全拒绝,防工作线程永挂)


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
    """工具参数 → 'k=v, k=v' 预览(取前 3 个)。"""
    try:
        return ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
    except Exception:
        return ""


class LiveSession:
    """一个常驻 prompt_toolkit Application;每条用户输入起一轮 agent.run(后台线程)。"""

    def __init__(self, cli):
        self.cli = cli                       # ChatCLI(持 self.agent / self.grants)
        self.state = {"busy": False, "spin": 0, "cancel": False,
                      "cwd": _short_cwd(),         # 底部信息栏:当前目录
                      "l1_tokens": 0,              # 当前 L1 上下文长度(每轮末刷新)
                      "sess_in": 0, "sess_out": 0}  # 本会话累计 token
        # 活跃块三元组单次原子赋值(src, mode, dot) —— 工作线程写、loop 线程读,
        # 用单一不可变快照避免多键 update 被 redraw 读到撕裂组合(I1)。
        self._active: tuple = ("", "text", False)
        self._pending_fut: "Optional[concurrent.futures.Future[bool]]" = None
        self.app: Optional[Application] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── 渲染 sink(交给 ScrollbackRenderer)────────────────────
    def _commit(self, text_obj: Text) -> None:
        # 工作线程经 patch_stdout(raw) print 进 scrollback。print 异常(并发重绘下
        # 偶发)不得中断当前轮 —— 吞掉,丢这一块好过崩整轮(I2)。
        try:
            print(_to_ansi(text_obj, _width()))
        except Exception:
            pass

    def _set_active(self, src: str, mode: str, dot: bool) -> None:
        self._active = (src, mode, dot)              # 单次原子赋值(I1)
        if self.app is not None:
            self.app.invalidate()

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
            head = f"{_SPIN[self.state['spin'] % len(_SPIN)]} 生成中…  (esc 中断)"
        else:
            head = "● 就绪"
        return [("class:status", f"  {head}")]

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
        """审批浮层内容:工具名 + preview(diff 高亮)+ 1/2/3 提示。空闲返回 []。"""
        appr = self.state.get("approval")
        if not appr:
            return []
        preview = appr["preview"] or ""
        w = _width()
        body = (Syntax(preview, "diff", theme="ansi_dark", background_color="default")
                if _looks_like_diff(preview) else preview)
        buf = io.StringIO()
        con = RichConsole(file=buf, force_terminal=True,
                          color_system="truecolor", width=w)
        con.print(Panel(body, title=f"  {appr['name']}", title_align="left",
                        border_style=theme.AGENT))
        con.print(f"  是否允许?  [{theme.OK}]1.[/]允许一次  "
                  f"[{theme.OK}]2.[/]本会话总是  [{theme.ERR}]3.[/]拒绝   "
                  f"[{theme.DIM}](1/Enter=是 · 3/Esc=否)[/]")
        return ANSI(buf.getvalue().rstrip("\n"))

    def _on_permission(self, name: str, args: Dict, preview: str) -> bool:
        """工作线程调用:命中授权台账直接放行;否则在事件循环弹浮层,阻塞等 Future。"""
        try:
            if self.cli.grants.is_granted(name, args):
                return True
        except Exception:
            pass
        if self._loop is None:
            return False
        fut: "concurrent.futures.Future[bool]" = concurrent.futures.Future()
        self._pending_fut = fut                       # 记录,退出时由 _main 解锁(C1)
        appr = {"fut": fut, "name": name, "args": args, "preview": preview}

        def _show():
            self.state["approval"] = appr
            if self.app is not None:
                self.app.invalidate()

        self._loop.call_soon_threadsafe(_show)
        try:
            return fut.result(timeout=_APPROVAL_TIMEOUT_S)
        except Exception:
            return False
        finally:
            self._pending_fut = None

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

    # ── 一轮:在后台线程跑 agent.run ──────────────────────────
    def _run_turn(self, user_input: str) -> None:
        r = ScrollbackRenderer(self._commit, self._set_active, _width)
        start = time.monotonic()
        tok = {"in": 0, "out": 0}

        def on_llm_done(elapsed, pt, ct):
            tok["in"] += pt or 0
            tok["out"] += ct or 0

        try:
            self.cli.agent.run(
                user_input,
                on_text_chunk=r.text_chunk,
                on_reasoning_chunk=r.reasoning_chunk,
                on_tool_call=lambda n, a: r.tool_call(n, _preview(a),
                                                      self._is_read_only(n)),
                on_tool_result=lambda n, res, el: r.tool_result(
                    res, name=n, read_only=self._is_read_only(n), elapsed_sec=el),
                on_permission_request=self._on_permission,
                on_llm_done=on_llm_done,
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
        # 底部信息栏统计:本会话累计 token + 当前 L1 上下文长度(此刻在 worker 线程,
        # agent.run 已结束,顺序读 memory 安全;不在每次重绘时读,避免与生成并发)
        self.state["sess_in"] += tok["in"]
        self.state["sess_out"] += tok["out"]
        try:
            self.state["l1_tokens"] = self.cli.agent.memory.stats().get("l1_tokens", 0)
        except Exception:
            pass
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

    # ── app 构建 ─────────────────────────────────────────────
    def _build_app(self, queue: "asyncio.Queue[str]"):
        # 单行输入:不折行(否则中文双宽字符很快撑到行尾,框被折成两行),
        # 改为横向滚动,框稳定保持一行高。
        ta = TextArea(multiline=False, wrap_lines=False,
                      prompt=HTML("<arrow>❯ </arrow>"))
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            text = ta.text.strip()
            ta.text = ""
            if not text:
                return
            if text in ("/quit", "/exit"):
                event.app.exit()
                return
            print(f"\n❯ {text}")                 # 回显用户输入进 scrollback
            if text.startswith("/"):
                # slash 命令:挂起 app 调现有 handle_command(临时;交互式命令体验留后续)。
                # run_in_terminal 返回的 future 故意 fire-and-forget(同步命令即可)。
                cmd = text
                run_in_terminal(lambda: self.cli.handle_command(cmd))
                return
            queue.put_nowait(text)               # 普通输入 → 排队给 agent
            event.app.invalidate()

        # ── 审批浮层按键(仅审批弹起时生效,eager 抢在输入框前拦截)──
        appr_on = Condition(lambda: self.state.get("approval") is not None)

        @kb.add("1", filter=appr_on, eager=True)
        @kb.add("y", filter=appr_on, eager=True)
        @kb.add("enter", filter=appr_on, eager=True)
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
        # 审批浮层在最上,生成中状态行在框上方,信息栏在框下方。
        root = HSplit([active, approval, status, HSplit([top, middle, bottom]), info])
        style = Style.from_dict({"arrow": "magenta", "frame.border": "gray",
                                 "status": "gray", "infobar": "#666666"})
        return Application(layout=Layout(root, focused_element=ta),
                           key_bindings=kb, style=style,
                           full_screen=False, mouse_support=False)

    async def _main(self):
        self._loop = asyncio.get_running_loop()
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
