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


def test_tool_readonly_dot_is_neutral():
    # 只读工具(side_effect_free)成功 → ⏺ 中性(不绿不红)
    from my_agent_llms.cli import theme
    r, commits, actives = _make()
    r.tool_call("ReadFile", "path=x", read_only=True)
    r.tool_result("文件内容若干行")
    notice = commits[0]
    assert not any(theme.OK in str(s.style) or theme.ERR in str(s.style)
                   for s in notice.spans)


def test_tool_mutating_success_dot_is_green():
    # 改动类工具(非只读)成功 → 绿
    from my_agent_llms.cli import theme
    r, commits, actives = _make()
    r.tool_call("EditFile", "config.py", read_only=False)
    r.tool_result("已写入 3 行")
    notice = commits[0]
    assert any(theme.OK in str(s.style) for s in notice.spans)


def test_tool_result_folds_long_output():
    # 长结果折叠到前几行 + "… +N lines",不全量倒出(Claude Code 风,防混乱)
    r, commits, actives = _make()
    r.tool_call("Bash", "ls", read_only=False)
    r.tool_result("\n".join(f"line{i}" for i in range(20)))
    body = commits[1].plain
    assert "line0" in body
    assert "line19" not in body          # 尾部折掉
    assert "+16 lines" in body           # 20-4=16 行被折


def test_tool_result_read_summary_not_raw():
    # Read 工具 → "Read N lines" 摘要,而非倒文件内容
    r, commits, actives = _make()
    r.tool_call("Read", "path=x", read_only=True)
    r.tool_result("aaa\nbbb\nccc")
    body = commits[1].plain
    assert "Read 3 lines" in body
    assert "bbb" not in body


def test_write_todo_renders_update_todos():
    r, commits, actives = _make()
    r.tool_call("write_todo", "")
    r.tool_result("## Todos\n[x] 读配置\n[ ] 写代码")
    out = "\n".join(c.plain for c in commits)
    assert "Update Todos" in out
    assert "☑" in out and "☐" in out


def test_multi_tool_round_fifo_pairs_names_and_readonly():
    # agent:Phase A 全部 tool_call 入队,Phase C 全部 tool_result。
    # 单槽会丢名/丢 read_only;必须 FIFO 队列。
    r, commits, actives = _make()
    r.tool_call("Read", "a.py", read_only=True)     # Phase A
    r.tool_call("Bash", "ls", read_only=False)
    r.tool_result("文件内容")                        # Phase C → 配 Read
    r.tool_result("✅ done")                          # → 配 Bash
    notices = [c.plain for c in commits if c.plain.startswith("⏺")]
    joined = "\n".join(c.plain for c in commits)
    assert "Read(a.py)" in joined                     # 第一个名字在
    assert "Bash(ls)" in joined                       # 第二个名字没丢
    assert all(n.strip() != "⏺" for n in notices)     # 没有空 ⏺


# ── 错误判定:只认开头标记,不在正文里全局搜 '拒绝'(否则读到含'拒绝'的文件会误判变红)──
def test_is_error_result_precise():
    from my_agent_llms.cli.scrollback_renderer import _is_error_result
    assert _is_error_result("❌ 写入失败") is True
    assert _is_error_result("用户拒绝了对 Edit 的调用") is True
    # Read 一个内容里含"拒绝"二字的文件 → 不是错误,不该变红
    assert _is_error_result("# README\n用户可以拒绝该操作,框会提示。\n...") is False
    assert _is_error_result("Read 190 lines") is False


def test_result_dot_color_not_red_on_content_with_reject_word():
    from my_agent_llms.cli.chat_view import _result_dot_color
    from my_agent_llms.cli import theme
    assert _result_dot_color("内容里提到用户可以拒绝操作") == theme.DEFAULT   # 不变红
    assert _result_dot_color("❌ 出错了") == theme.ERR
    assert _result_dot_color("用户拒绝了对 Edit 的调用") == theme.ERR
