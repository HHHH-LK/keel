import time
from pathlib import Path
import pytest
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)


def _make_pe(pid: str = "abc") -> PendingEdit:
    return PendingEdit(
        id=pid,
        kind="edit",
        path=Path("/tmp/x.md"),
        new_content="hello",
        diff_preview="(no diff)",
        source_hash="dummyhash",
        created_at=time.time(),
    )


def test_put_then_pop_returns_original():
    store = PendingEditStore()
    pe = _make_pe()
    store.put(pe)
    got = store.pop("abc")
    assert got is pe


def test_pop_unknown_id_returns_none():
    assert PendingEditStore().pop("nope") is None


def test_pop_is_destructive():
    """同一 pending_id 只能 apply 一次,pop 后再 pop 必须 None。"""
    store = PendingEditStore()
    store.put(_make_pe())
    assert store.pop("abc") is not None
    assert store.pop("abc") is None


def test_discard_removes_entry():
    store = PendingEditStore()
    store.put(_make_pe())
    assert store.discard("abc") is True
    assert store.pop("abc") is None
    assert store.discard("abc") is False  # 已被 discard


def test_ttl_expiry():
    """用 back-date 时间戳,避免 sleep 在慢 CI 上 flaky。"""
    store = PendingEditStore(ttl_seconds=1)
    pe = _make_pe()
    pe.created_at = time.time() - 2.0  # 已过期
    store.put(pe)
    assert store.pop("abc") is None


def test_multiple_pendings_independent():
    store = PendingEditStore()
    store.put(_make_pe("a"))
    store.put(_make_pe("b"))
    assert store.pop("a") is not None
    assert store.pop("b") is not None
