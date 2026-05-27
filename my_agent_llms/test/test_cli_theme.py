"""Smoke test: every theme color string must be accepted by rich."""
from rich.console import Console
from rich.text import Text

from my_agent_llms.cli import theme


def test_all_color_constants_render_without_error():
    console = Console(force_terminal=True, width=80)
    colors = [
        theme.YOU, theme.AGENT, theme.ACCENT,
        theme.LOGO_L, theme.LOGO_R, theme.TITLE,
        theme.OK, theme.WARN, theme.ERR,
        theme.DIM, theme.RULE, theme.DEFAULT,
    ]
    for color in colors:
        t = Text("x", style=color)
        with console.capture() as cap:
            console.print(t)
        assert "x" in cap.get()


def test_default_is_empty_string():
    assert theme.DEFAULT == ""
