"""底部圆角边框输入盒(Claude Code 风)。

之前用 PromptSession.prompt() 只渲行内 '❯ ',无边框。这里换成一个非全屏
Application:Frame 圆角盒 包 TextArea,等待输入时出现在底部、随输出上滚;
Agent 回复阶段不显示(那时底部是 spinner)。

为什么不用内置 widgets.Frame:它的边框字符写死方角(┌┐└┘),不能传参圆角,
所以照搬其结构、换成 ╭╮╰╯ 自拼一个 _rounded_frame。

Application 自带 load_key_bindings()(emacs 编辑 / 补全 / 导航),与原 PromptSession
同底座,故方向键/退格/补全等行为一致;额外只绑 enter(提交/接受补全)、
c-l(清屏)、c-c(中断)、c-d(空缓冲时 EOF)。
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    Float, FloatContainer, HSplit, VSplit, Window,
)
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from . import theme
from .completer import SlashCompleter

# Note: prompt_toolkit 的 CSS-style 字符串接受 "magenta" 这类基本色名,但不认
# rich 的 "bright_black" —— 所以 theme.DIM 在这里近似为 "gray"。
_STYLE = Style.from_dict({
    "prompt.arrow":     theme.AGENT,        # ❯ 箭头
    "frame.border":     "gray",             # 圆角边框色
    # 补全菜单(slash 菜单)
    "completion-menu":                          "bg:default",
    "completion-menu.completion":               "fg:default",
    "completion-menu.completion.current":       f"bg:{theme.AGENT} fg:black",
    "completion-menu.meta.completion":          "fg:gray",
    "completion-menu.meta.completion.current":  f"bg:{theme.AGENT} fg:black",
})


def _rounded_frame(body):
    """照搬 widgets.Frame 结构,换成圆角字符 ╭╮╰╯ + 左右 1 列内边距。"""
    fill = partial(Window, style="class:frame.border")
    top = VSplit([
        fill(width=1, height=1, char="╭"),
        fill(char="─"),
        fill(width=1, height=1, char="╮"),
    ], height=1)
    bottom = VSplit([
        fill(width=1, height=1, char="╰"),
        fill(char="─"),
        fill(width=1, height=1, char="╯"),
    ], height=1)
    middle = VSplit([
        fill(width=1, char="│"),
        Window(width=1),            # 左内边距
        body,
        Window(width=1),            # 右内边距
        fill(width=1, char="│"),
    ])
    return HSplit([top, middle, bottom])


class BoxPromptSession:
    """PromptSession 的替身:.prompt() 返回一行/多行输入,外观是底部圆角盒。

    复用 SlashCompleter / FileHistory / AutoSuggestFromHistory,接口与原
    PromptSession.prompt(message, multiline=...) 兼容(message 被忽略,盒内自带 ❯)。
    """

    def __init__(self, history_path: Path, clear_screen):
        self._history = FileHistory(str(history_path))
        self._completer = SlashCompleter()
        self._clear = clear_screen

    def prompt(self, message=None, *, multiline: bool = False) -> str:
        ta = TextArea(
            multiline=multiline,
            completer=self._completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            history=self._history,
            wrap_lines=True,
            prompt=HTML("<prompt.arrow>❯ </prompt.arrow>"),
        )
        buf = ta.buffer

        kb = KeyBindings()

        @kb.add("c-l")
        def _(event):
            self._clear()

        @kb.add("c-c")
        def _(event):
            event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d", filter=Condition(lambda: not buf.text))
        def _(event):
            event.app.exit(exception=EOFError)

        if multiline:
            # 多行:Enter 换行(默认),Esc+Enter 提交
            @kb.add("escape", "enter")
            def _(event):
                event.app.exit(result=buf.text)
        else:
            @kb.add("enter")
            def _(event):
                # 补全菜单开着时:Enter 接受当前候选 / 关菜单,不提交
                if buf.complete_state:
                    cc = buf.complete_state.current_completion
                    if cc is not None:
                        buf.apply_completion(cc)
                    else:
                        buf.cancel_completion()
                    return
                event.app.exit(result=buf.text)

        root = FloatContainer(
            content=_rounded_frame(ta),
            floats=[Float(
                xcursor=True, ycursor=True,
                content=CompletionsMenu(max_height=8, scroll_offset=1),
            )],
        )
        app = Application(
            layout=Layout(root, focused_element=ta),
            key_bindings=kb,
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
            # 提交后擦掉圆角框(否则每回合在 scrollback 留一个框)。
            # 擦除后由 app 层 render_user_input 把输入塌成一行 '❯ text' 回显。
            erase_when_done=True,
        )
        text = app.run()
        if text and text.strip():
            self._history.append_string(text)
        return text


def build_session(history_path: Path, clear_screen) -> BoxPromptSession:
    """创建底部圆角盒输入会话。

    Args:
        history_path: 输入历史持久化路径(上/下方向键回溯)。
        clear_screen: 绑到 Ctrl-L 的清屏回调。
    """
    return BoxPromptSession(history_path, clear_screen)
