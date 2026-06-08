import re
from pathlib import Path

from rich.console import Console

from my_agent_llms.cli import banner


def _capture(fn, *args, **kwargs) -> str:
    test_console = Console(force_terminal=True, width=100, color_system="truecolor")
    with test_console.capture() as cap:
        fn(test_console, *args, **kwargs)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_banner_ready_shows_model_and_tools_and_workspace():
    out = _capture(
        banner.render,
        ready=True,
        provider_key="minimax",
        model="MiniMax-Text-01",
        backend_label="L4 cold: sqlite",
        tool_count=4,
        workspace=Path("/tmp/ws"),
    )
    assert "keel" in out
    assert "minimax" in out
    assert "MiniMax-Text-01" in out
    assert "ready" in out
    assert "4 tools" in out
    assert "/tmp/ws" in out or "ws" in out


def test_banner_not_ready_prompts_for_config_key():
    out = _capture(
        banner.render,
        ready=False,
        provider_key="openai",
        model="",
        backend_label="",
        tool_count=0,
        workspace=None,
    )
    assert "not ready" in out
    assert "/config" in out
