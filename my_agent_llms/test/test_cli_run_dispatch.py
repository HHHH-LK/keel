"""ChatCLI.run 分流:tty 且无 MYAI_LEGACY_UI → 并发 UI;否则串行。"""
from my_agent_llms.cli.app import ChatCLI


def _cli(monkeypatch, tty: bool, legacy: bool):
    cli = ChatCLI.__new__(ChatCLI)            # 跳过 __init__
    cli.agent = object()
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty, raising=False)
    if legacy:
        monkeypatch.setenv("MYAI_LEGACY_UI", "1")
    else:
        monkeypatch.delenv("MYAI_LEGACY_UI", raising=False)
    return cli


def test_tty_uses_live_session(monkeypatch):
    called = {"live": False}
    monkeypatch.setattr("my_agent_llms.cli.live_session.run",
                        lambda cli: called.__setitem__("live", True))
    cli = _cli(monkeypatch, tty=True, legacy=False)
    cli.run()
    assert called["live"] is True


def test_non_tty_uses_serial_loop(monkeypatch):
    called = {"live": False, "serial": False}
    monkeypatch.setattr("my_agent_llms.cli.live_session.run",
                        lambda cli: called.__setitem__("live", True))
    cli = _cli(monkeypatch, tty=False, legacy=False)
    monkeypatch.setattr(cli, "_run_serial", lambda: called.__setitem__("serial", True))
    cli.run()
    assert called["serial"] is True and called["live"] is False


def test_legacy_env_forces_serial(monkeypatch):
    called = {"live": False, "serial": False}
    monkeypatch.setattr("my_agent_llms.cli.live_session.run",
                        lambda cli: called.__setitem__("live", True))
    cli = _cli(monkeypatch, tty=True, legacy=True)
    monkeypatch.setattr(cli, "_run_serial", lambda: called.__setitem__("serial", True))
    cli.run()
    assert called["serial"] is True and called["live"] is False
