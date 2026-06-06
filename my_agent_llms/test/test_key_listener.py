from my_agent_llms.cli.key_listener import EscListener


def test_non_tty_is_noop(monkeypatch):
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with EscListener() as esc:
        assert esc.cancelled is False
    assert esc.cancelled is False


def test_cancelled_property_default_false():
    e = EscListener()
    assert e.cancelled is False


def test_context_manager_returns_self_and_restores(monkeypatch):
    # 非 tty 路径:__enter__ 返回自身,__exit__ 不抛
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    e = EscListener()
    with e as same:
        assert same is e


def test_pause_resume_toggles_state():
    e = EscListener()
    assert e._paused.is_set() is False
    e.pause()
    assert e._paused.is_set() is True
    e.resume()
    assert e._paused.is_set() is False
