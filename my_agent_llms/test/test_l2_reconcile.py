"""L2 分层托管测试 —— 解"摘要陈旧与当前状态冲突"。

设计:LLM 管语义编辑(reconciler),机械层当护栏。
双扳机:
- 冲突触发: write 检测到 supersede → 立即校正 L2
- 定期反思: 每 N 轮主动复查 L2 是否过期

reconciler 可注入,测试用确定性 stub;真实 LLM 版另有降级兜底。
"""
from my_agent_llms.memory import MemoryConfig, MemoryManager
from my_agent_llms.memory.summary import SummaryMemory


# ─────────────────────────────────────────────
# SummaryMemory.reconcile 单元
# ─────────────────────────────────────────────

def test_reconcile_updates_summary_via_reconciler():
    captured = {}

    def stub_reconciler(current: str, signal: str) -> str:
        captured["current"] = current
        captured["signal"] = signal
        return "用户正在学 Python(已从 Java 切换)"

    mem = SummaryMemory(reconciler=stub_reconciler)
    # 先有一段旧摘要
    mem._set_summary_text("用户正在学 Java")
    result = mem.reconcile("状态变更: 学Java → 学Python")

    assert result is not None
    assert "Python" in result.content
    assert captured["signal"] == "状态变更: 学Java → 学Python"
    assert "Java" in captured["current"]


def test_reconcile_noop_without_reconciler():
    """没注入 reconciler → 不动摘要(降级)。"""
    mem = SummaryMemory()  # 无 reconciler
    mem._set_summary_text("用户正在学 Java")
    result = mem.reconcile("状态变更")
    assert result.content == "用户正在学 Java"


def test_reconcile_noop_without_summary():
    """还没有摘要时,reconcile 无事可做。"""
    mem = SummaryMemory(reconciler=lambda c, s: "x")
    assert mem.reconcile("任何信号") is None


def test_reconcile_size_capped():
    """reconciler 吐超长内容 → 机械层截断兜底。"""
    mem = SummaryMemory(reconciler=lambda c, s: "长" * 10000, max_tokens=100)
    mem._set_summary_text("旧")
    result = mem.reconcile("signal")
    assert len(result.content) <= 100 * 3  # max_tokens * 3 char/token


def test_reconcile_failure_keeps_old_summary():
    """reconciler 抛异常 → 保留旧摘要,不污染。"""
    def boom(current: str, signal: str) -> str:
        raise RuntimeError("LLM 挂了")

    mem = SummaryMemory(reconciler=boom)
    mem._set_summary_text("用户正在学 Java")
    result = mem.reconcile("signal")
    assert result.content == "用户正在学 Java"


# ─────────────────────────────────────────────
# 集成: 冲突扳机
# ─────────────────────────────────────────────

def test_conflict_triggers_reconcile():
    """write 检测到 supersede 时,联动校正 L2。"""
    calls = []

    def stub_reconciler(current: str, signal: str) -> str:
        calls.append(signal)
        return current + " [已校正]"

    cfg = MemoryConfig(conflict_strength="fast", conflict_threshold=0.1)
    mgr = MemoryManager(cfg, reconciler=stub_reconciler)
    # 先垫一段摘要,让 reconcile 有东西可改
    mgr.summary._set_summary_text("用户正在学 Java,掌握很多 Java 知识")

    # 写两条高相似内容,触发 supersede(阈值压到 0.1 易触发)
    mgr.write("我正在学 Java 编程", role="user")
    mgr.write("我正在学 Java 编程语言", role="user")

    assert len(calls) >= 1  # 至少触发了一次校正


# ─────────────────────────────────────────────
# 集成: 反思扳机
# ─────────────────────────────────────────────

def test_reflection_triggers_every_n_turns():
    """每 l2_reflect_every_n_turns 轮主动校正一次。"""
    calls = []

    def stub_reconciler(current: str, signal: str) -> str:
        calls.append(signal)
        return current

    cfg = MemoryConfig(l2_reflect_every_n_turns=3, conflict_strength="off")
    mgr = MemoryManager(cfg, reconciler=stub_reconciler)
    mgr.summary._set_summary_text("一些摘要内容")
    mgr.write("普通对话内容", role="user")

    # 跑 6 轮 tick,反思应在第 3、6 轮触发(2 次)
    for _ in range(6):
        mgr.tick()

    assert len(calls) == 2


def test_reflection_off_when_zero():
    calls = []
    cfg = MemoryConfig(l2_reflect_every_n_turns=0, conflict_strength="off")
    mgr = MemoryManager(cfg, reconciler=lambda c, s: calls.append(s) or c)
    mgr.summary._set_summary_text("摘要")
    for _ in range(10):
        mgr.tick()
    assert len(calls) == 0


if __name__ == "__main__":
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"  ✓ {f.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {f.__name__}: {exc}")
    print(f"\n{passed}/{len(funcs)} passed")
