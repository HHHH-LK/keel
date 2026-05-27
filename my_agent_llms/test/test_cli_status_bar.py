import re
from rich.console import Console
from my_agent_llms.cli import status_bar


def _capture(fn, *args, **kwargs) -> str:
    c = Console(force_terminal=True, width=80, color_system="truecolor")
    with c.capture() as cap:
        fn(c, *args, **kwargs)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_status_bar_ready_shows_model_turn_tokens():
    out = _capture(
        status_bar.render,
        ready=True,
        provider_key="minimax",
        model="Text-01",
        turn=7,
        l1_tokens=1234,
        l1_max_tokens=4000,
        multiline=False,
    )
    assert "minimax" in out
    assert "Text-01" in out
    assert "7" in out
    assert "1234" in out


def test_status_bar_not_ready_mentions_config_key():
    out = _capture(
        status_bar.render,
        ready=False,
        provider_key="",
        model="",
        turn=0,
        l1_tokens=0,
        l1_max_tokens=4000,
        multiline=False,
    )
    assert "not ready" in out
    assert "/config" in out


def test_status_bar_multiline_appends_marker():
    out = _capture(
        status_bar.render,
        ready=True,
        provider_key="minimax",
        model="Text-01",
        turn=1,
        l1_tokens=0,
        l1_max_tokens=4000,
        multiline=True,
    )
    assert "multiline" in out
