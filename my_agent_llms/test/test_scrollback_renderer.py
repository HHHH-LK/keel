"""ScrollbackRenderer:agent 回调 → 完成块(commit)/ 残块(set_active)。"""
from rich.text import Text
from my_agent_llms.cli.scrollback_renderer import ScrollbackRenderer


def _make():
    commits: list[Text] = []
    actives: list[tuple] = []          # (src, mode, dot)
    r = ScrollbackRenderer(
        commit=lambda t: commits.append(t),
        set_active=lambda src, mode, dot: actives.append((src, mode, dot)),
        width=lambda: 76,
    )
    return r, commits, actives


def test_text_commits_completed_block_keeps_remainder_active():
    r, commits, actives = _make()
    r.text_chunk("第一段。\n\n第二段进行中")
    # 完成块(第一段)经 commit 落 scrollback;残块进 active
    assert any("第一段" in c.plain for c in commits)
    assert not any("第二段" in c.plain for c in commits)
    assert actives[-1][0] == "第二段进行中"
    assert actives[-1][1] == "text"


def test_close_reasoning_commits_thinking_block_on_text_chunk():
    # reasoning 段在切到 text 时折叠 commit;_mode 复位为 text
    r, commits, actives = _make()
    r._mode = "reasoning"
    r._reason_buf = "我先想一下"
    r.text_chunk("正式回答")
    assert any("我先想一下" in c.plain for c in commits)   # _close_reasoning 已 commit
    assert r._mode == "text"
    assert actives[-1][1] == "text"


def test_reasoning_streams_active_then_commits_folded_on_switch():
    r, commits, actives = _make()
    r.reasoning_chunk("我先想\n第二行\n第三行\n第四行")
    # 流式期间残块进 active,mode=reasoning
    assert actives[-1][1] == "reasoning"
    assert "我先想" in actives[-1][0]
    r.text_chunk("正式回答")          # 切到正文 → 思考块折叠 commit
    assert any("我先想" in c.plain and "✻" in c.plain for c in commits)
    assert actives[-1][1] == "text"
