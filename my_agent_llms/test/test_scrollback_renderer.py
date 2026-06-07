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


def test_close_commits_remaining_block_and_meta():
    r, commits, actives = _make()
    r.text_chunk("一段没有空行的话")          # 无块边界 → 残块留 active
    assert not any("一段没有空行" in c.plain for c in commits)
    r.close(tools_used=2, elapsed_seconds=2.4, tokens_in=100, tokens_out=50)
    # 残块在 close 时 commit;meta 行带 tools/elapsed/token
    assert any("一段没有空行" in c.plain for c in commits)
    assert any("2 tools" in c.plain and "100↑ 50↓" in c.plain for c in commits)
    assert actives[-1][0] == ""               # 收尾清空活跃区


def test_close_without_output_is_noop():
    r, commits, actives = _make()
    r.close()
    assert commits == []


def test_tool_call_and_result_commit_lines():
    r, commits, actives = _make()
    r.text_chunk("准备读文件。")            # 先有正文残块
    r.tool_call("ReadFile", "path=config.py")
    r.tool_result("✅ 读取成功", elapsed_sec=0.2)
    plains = "\n".join(c.plain for c in commits)
    assert "准备读文件" in plains            # 工具前正文残块已被收尾 commit
    assert "ReadFile(path=config.py)" in plains
    assert "✅ 读取成功" in plains


def test_tool_success_dot_is_green():
    # 工具成功(普通结果,非 ✅ 开头)→ ⏺ 应绿色(像 Claude Code 默认完成=绿)
    from my_agent_llms.cli import theme
    r, commits, actives = _make()
    r.tool_call("EditFile", "config.py")
    r.tool_result("已写入 3 行")
    notice = commits[0]                     # ⏺ EditFile(...)
    assert any(theme.OK in str(s.style) for s in notice.spans)


def test_tool_error_dot_is_red():
    # 工具出错(agent 用 ❌ 包裹)→ ⏺ 应红色
    from my_agent_llms.cli import theme
    r, commits, actives = _make()
    r.tool_call("EditFile", "config.py")
    r.tool_result("❌ 工具 'EditFile' 执行异常: boom")
    notice = commits[0]
    assert any(theme.ERR in str(s.style) for s in notice.spans)
