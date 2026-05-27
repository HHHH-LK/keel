"""SlashCompleter — input '/' pops menu, letters filter, non-slash returns []."""
from prompt_toolkit.document import Document

from my_agent_llms.cli.completer import SLASH_COMMANDS, SlashCompleter


def _completions_for(text: str) -> list[str]:
    comp = SlashCompleter()
    doc = Document(text=text, cursor_position=len(text))
    return [c.text for c in comp.get_completions(doc, complete_event=None)]


def test_slash_alone_returns_all_commands_in_declared_order():
    got = _completions_for("/")
    expected = [name for name, _desc, _group in SLASH_COMMANDS]
    assert got == expected


def test_substring_filters_to_re_prefixed_commands():
    got = _completions_for("/re")
    assert "/recall" in got
    assert "/remember" in got
    assert "/restore" in got
    assert "/help" not in got


def test_non_slash_input_returns_empty():
    assert _completions_for("hello") == []
    assert _completions_for("") == []


def test_unknown_substring_returns_empty():
    assert _completions_for("/xxxyyy") == []


def test_completion_carries_description_as_display_meta():
    comp = SlashCompleter()
    doc = Document(text="/help", cursor_position=5)
    completions = list(comp.get_completions(doc, complete_event=None))
    assert len(completions) == 1
    c = completions[0]
    assert c.text == "/help"
    meta_text = "".join(piece[1] for piece in c.display_meta)
    assert "show all commands" in meta_text


def test_start_position_replaces_full_query():
    """When user types '/re' and accepts /recall, /re is replaced — not appended."""
    comp = SlashCompleter()
    doc = Document(text="/re", cursor_position=3)
    completions = list(comp.get_completions(doc, complete_event=None))
    for c in completions:
        assert c.start_position == -3
