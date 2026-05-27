"""PromptSession factory — wires SlashCompleter + styles + key bindings."""
from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from . import theme
from .completer import SlashCompleter

_STYLE = Style.from_dict({
    # ❯ arrow + label inside the prompt itself
    "prompt.arrow":     "magenta",
    "prompt.multiline": "gray",
    # Completion menu (slash menu)
    "completion-menu":                          "bg:default",
    "completion-menu.completion":               "fg:default",
    "completion-menu.completion.current":       "bg:magenta fg:black",
    "completion-menu.meta.completion":          "fg:gray",
    "completion-menu.meta.completion.current":  "bg:magenta fg:black",
})


def build_session(history_path: Path, clear_screen) -> PromptSession:
    """Create the shared PromptSession.

    Args:
        history_path: where to persist input history (arrow up / down).
        clear_screen: callable bound to Ctrl-L (clears the terminal).
    """
    kb = KeyBindings()

    @kb.add("c-l")
    def _(event):
        clear_screen()

    return PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=kb,
        style=_STYLE,
        completer=SlashCompleter(),
        complete_while_typing=True,
    )


def prompt_html(multiline: bool) -> HTML:
    """The ❯ markup, optionally with 'multiline ›' label."""
    if multiline:
        return HTML(
            "<prompt.arrow>❯</prompt.arrow> "
            "<prompt.multiline>multiline</prompt.multiline> "
            "<prompt.arrow>›</prompt.arrow> "
        )
    return HTML("<prompt.arrow>❯</prompt.arrow> ")
