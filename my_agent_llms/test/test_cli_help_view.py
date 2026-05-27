"""help_view — render_help, render_config_show, render_memory_stats, print_*."""
import re

from rich.console import Console

from my_agent_llms.cli import help_view


def _capture(fn, *args, **kwargs) -> str:
    """Run a render fn against a captured rich Console and return stripped text."""
    test_console = Console(force_terminal=True, width=100, color_system="truecolor")
    with test_console.capture() as cap:
        fn(test_console, *args, **kwargs)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_render_help_groups_and_contains_every_command():
    out = _capture(help_view.render_help)
    assert "Basic" in out
    assert "Config" in out
    assert "Memory" in out
    from my_agent_llms.cli.completer import SLASH_COMMANDS
    for name, _desc, _group in SLASH_COMMANDS:
        assert name in out, f"missing {name} in /help output"


def test_render_config_show_masks_api_key():
    cfg = {
        "provider_key": "minimax",
        "provider":     "openai",
        "model":        "Text-01",
        "base_url":     "https://api.minimaxi.com/v1",
        "api_key":      "sk-abcdef1234567890",
        "memory": {
            "cold_backend": "sqlite", "vector_backend": "sqlite",
            "conflict_strength": "fast", "tick_mode": "async",
            "use_embedding": False,
        },
    }
    out = _capture(help_view.render_config_show, cfg, agent_ready=True,
                   config_path="/tmp/cfg.json")
    assert "sk-abc" in out
    assert "7890" in out
    assert "sk-abcdef1234567890" not in out
    assert "minimax" in out
    assert "ready" in out


def test_render_memory_stats_two_columns():
    stats = {"l1_items": 12, "l1_tokens": 1234, "l4_items": 9, "l5_items": 5,
             "l2_tokens": 200}
    out = _capture(help_view.render_memory_stats, stats)
    assert "1234" in out
    assert "L1 tokens" in out or "l1_tokens" in out


def test_print_error_marks_red_and_includes_message():
    out = _capture(lambda c: help_view.print_error(c, "bad thing happened"))
    assert "error" in out
    assert "bad thing happened" in out


def test_print_ok_prints_check_and_message():
    out = _capture(lambda c: help_view.print_ok(c, "all good"))
    assert "all good" in out
