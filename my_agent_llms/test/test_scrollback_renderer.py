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
