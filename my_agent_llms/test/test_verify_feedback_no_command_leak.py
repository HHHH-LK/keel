"""B: verify 反馈不再回灌 command_ok 的命令原文。

根因:门内已用 subprocess 跑过 command_ok;若把命令字符串塞回反馈,模型会用自己的
Bash/工具把同一条命令再跑一遍来"自证",污染对话(见 transcript 里答完又冒出 git diff)。
反馈应只给方向(去修产物、别重跑检查),不暴露可执行命令。
"""
from my_agent_llms.verify.loop import feedback_from
from my_agent_llms.verify.spec import Check, CheckSpec


def test_command_ok_feedback_omits_raw_command():
    cmd = "git --no-pager diff --quiet README.md; test $? -ne 0"
    spec = CheckSpec(task="改 README", checks=[
        Check(id="c1", type="command_ok", params={"cmd": cmd}, is_hard_oracle=True),
    ])
    fb = feedback_from(spec, {"c1": False})
    assert fb is not None
    assert cmd not in fb                 # 命令原文不得出现
    assert "不要重新运行" in fb          # 明确叫模型别再跑检查命令
    assert "git" not in fb               # 连命令里的可执行片段也不泄露


def test_command_ok_failures_dedup_to_one_line():
    """多条 command_ok 都没过时,泛化提示会重复 —— feedback_from 应去重,不堆同一句。"""
    spec = CheckSpec(task="t", checks=[
        Check(id="c1", type="command_ok", params={"cmd": "cmd-a"}, is_hard_oracle=True),
        Check(id="c2", type="command_ok", params={"cmd": "cmd-b"}, is_hard_oracle=True),
    ])
    fb = feedback_from(spec, {"c1": False, "c2": False})
    # 头部一行 + 去重后仅一条 command_ok 提示
    body_lines = [ln for ln in fb.splitlines() if ln.startswith("- ")]
    assert len(body_lines) == 1


def test_other_check_types_still_describe_specifically():
    """只动 command_ok:string_contains 等仍要给出具体内容,别被误伤成泛化。"""
    spec = CheckSpec(task="t", checks=[
        Check(id="c1", type="string_contains", params={"s": "结论"}),
    ])
    fb = feedback_from(spec, {"c1": False})
    assert "结论" in fb
