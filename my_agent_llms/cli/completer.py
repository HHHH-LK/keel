"""Slash command completer — single source of truth for command metadata.

`SLASH_COMMANDS` is consumed by both this completer (to populate the menu)
and `help_view.render_help` (to render `/help`). Never duplicate this list.
"""
from __future__ import annotations

from typing import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText

# (name, description, group). Order here = display order in menu and /help.
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/help",      "show all commands",                "Basic"),
    ("/quit",      "exit (also /exit, Ctrl+D)",        "Basic"),
    ("/multiline", "toggle multiline input",           "Basic"),
    ("/config",    "configure provider, model, key",   "Config"),
    ("/clear",     "clear context (keeps long-term)",  "Memory"),
    ("/memory",    "show memory stats",                "Memory"),
    ("/recall",    "search long-term memory",          "Memory"),
    ("/remember",  "add a memory card",                "Memory"),
    ("/forget",    "forget a memory card",             "Memory"),
    ("/pin",       "lock a memory card",               "Memory"),
    ("/l0",        "list active L0 cards",             "Memory"),
    ("/restore",   "load recent history from cold",    "Memory"),
    ("/facts",     "query KG facts",                   "Memory"),
    ("/kg",        "export knowledge graph (mermaid)", "Memory"),
]


class SlashCompleter(Completer):
    """Pops up only when the input starts with '/'.

    Matches by case-insensitive substring against command names only (not
    descriptions), preserving the declared order of SLASH_COMMANDS.
    """

    def get_completions(  # type: ignore[override]
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        query = text.lower()
        for name, desc, _group in SLASH_COMMANDS:
            if query not in name.lower():
                continue
            yield Completion(
                text=name,
                start_position=-len(text),
                display=name,
                display_meta=FormattedText([("", desc)]),
            )
