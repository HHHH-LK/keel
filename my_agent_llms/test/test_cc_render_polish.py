"""Claude Code 风排版收口:思考折 1 行、工具名稳健配对、LS 摘要、连续同类工具合并。"""
from rich.text import Text

from my_agent_llms.cli import chat_view
from my_agent_llms.cli.scrollback_renderer import ScrollbackRenderer


def _make():
    commits: list[Text] = []
    actives: list[tuple] = []
    r = ScrollbackRenderer(
        commit=lambda t: commits.append(t),
        set_active=lambda src, mode, dot: actives.append((src, mode, dot)),
        width=lambda: 76,
    )
    return r, commits, actives


# ── #3 思考折成 1 行 ──────────────────────────────────────────
def test_thinking_commits_single_line():
    t = chat_view._render_thinking("第一行想法\n第二行\n第三行\n第四行")
    # 折成 1 行:✻ + 首行,无 "… +N 行(思考)" 的多行展开
    assert t.plain == "✻ 第一行想法"


# ── #1 工具名稳健配对(不依赖 FIFO,显式传名即用)────────────────
def test_tool_result_honors_explicit_name_without_prior_call():
    r, commits, actives = _make()
    # 没有先 tool_call(FIFO 空)→ 仍按显式 name 渲染,⏺ 不空
    r.tool_result("内容若干行", name="Read", read_only=True, elapsed_sec=0.01)
    text = "\n".join(c.plain for c in commits)
    assert "Read" in text


# ── #5 LS 走 per-type 摘要,不再泛型倒出 ───────────────────────
def test_ls_result_gets_entry_summary():
    r, commits, actives = _make()
    r.tool_call("LS", "path=.", read_only=True)
    r.tool_result("a.py  10\nb.py  20\nc.py  30", name="LS", elapsed_sec=0.01)
    text = "\n".join(c.plain for c in commits)
    assert "3 items" in text
    # 不该把三行原始内容全倒出来
    assert "b.py  20" not in text


# ── #2 连续同类只读工具合并成一行计数 ──────────────────────────
def test_consecutive_readonly_same_tool_groups_into_count_line():
    r, commits, actives = _make()
    # 模拟 Phase A 全部入队、Phase C 全部出结果(并行批)
    for f in ("a.py", "b.py", "c.py"):
        r.tool_call("Read", f"path={f}", read_only=True)
    for f in ("a.py", "b.py", "c.py"):
        r.tool_result("10 行", name="Read", read_only=True, elapsed_sec=0.01)
    r.close()
    text = "\n".join(c.plain for c in commits)
    assert "Read 3 files" in text
    assert "a.py" in text and "c.py" in text


def test_single_readonly_tool_still_commits_immediately():
    r, commits, actives = _make()
    r.tool_call("Read", "path=solo.py", read_only=True)
    r.tool_result("3 行", name="Read", read_only=True, elapsed_sec=0.01)
    # 没有更多同名排队 → 立即提交,不等 close
    text = "\n".join(c.plain for c in commits)
    assert "Read" in text
    assert "Read 1 files" not in text   # 单个不走分组计数文案


def test_blank_line_separates_consecutive_blocks():
    r, commits, actives = _make()
    r.tool_call("recall", "query=x")
    r.tool_result("找到 5 条", name="recall")     # 第 1 块:无前导空行
    r.reasoning_chunk("想一下下")
    r.text_chunk("正式回答")                       # 切正文 → 先 commit 思考块,再正文
    r.close()
    plains = [c.plain for c in commits]
    assert plains[0] != ""                         # 首块不加前导空行
    assert "" in plains                            # 块间有空行分隔


def test_blank_line_precedes_thinking_after_a_tool():
    r, commits, actives = _make()
    r.tool_call("recall", "query=x")
    r.tool_result("找到 5 条", name="recall")
    r.reasoning_chunk("想一下下")
    r.text_chunk("答")
    plains = [c.plain for c in commits]
    idx = next(i for i, p in enumerate(plains) if p.startswith("✻"))
    assert plains[idx - 1] == ""                   # 思考块前正好一行空行


def test_text_after_tool_starts_new_step_with_blank_and_dot():
    r, commits, actives = _make()
    r.text_chunk("第一段\n\n")                      # 首个正文步(⏺)
    r.tool_call("Read", "path=x", read_only=True)
    r.tool_result("内容", name="Read", read_only=True)
    r.text_chunk("工具之后的话")                     # 工具后的正文 → 新步
    r.close()
    plains = [c.plain for c in commits]
    idx = next(i for i, p in enumerate(plains) if "工具之后的话" in p)
    assert plains[idx].startswith("⏺")              # 自带 ⏺(不是缩进续行)
    assert plains[idx - 1] == ""                    # 前有空行分隔


def test_group_flushes_when_different_tool_arrives():
    r, commits, actives = _make()
    r.tool_call("Read", "path=a.py", read_only=True)
    r.tool_call("Read", "path=b.py", read_only=True)
    r.tool_call("Grep", "pattern=foo", read_only=True)
    r.tool_result("x", name="Read", read_only=True)
    r.tool_result("y", name="Read", read_only=True)
    r.tool_result("2 matches", name="Grep", read_only=True)
    r.close()
    text = "\n".join(c.plain for c in commits)
    assert "Read 2 files" in text
    assert "Grep" in text
