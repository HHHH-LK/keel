"""测试 MyFunctionCallAgent.run 的 should_cancel 行为。

- 步级取消:每次循环顶部检查,True 时不调用 _invoke_with_tools 直接 break
- 正常路径:不传 should_cancel 时行为与原来完全一致
"""
import types

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent


def _bare_agent(monkeypatch):
    a = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    a.name = "test"
    a.system_prompt = None
    a.max_steps = 5
    a.enable_verify = False
    a.replan_budget = 0
    a.todo_store = None
    a.last_tool_call_count = 0
    a._in_tdd = False
    a.enable_tdd = False

    class _Mem:
        def assemble_context(self, sp, query=None):
            return []

    a.memory = _Mem()
    monkeypatch.setattr(a, "_apply_honesty_contract", lambda p: p or "", raising=False)
    monkeypatch.setattr(a, "_build_tool_schemas", lambda: [], raising=False)
    monkeypatch.setattr(a, "_run_response_hooks", lambda i, r, m: r, raising=False)
    monkeypatch.setattr(a, "_finalize_turn", lambda i, r, **kw: None, raising=False)
    # _maybe_run_tdd: TDD は使わない
    monkeypatch.setattr(a, "_maybe_run_tdd", lambda text: None, raising=False)
    return a


def test_run_cancels_before_first_step(monkeypatch):
    """should_cancel() が True なら _invoke_with_tools は呼ばれない。"""
    a = _bare_agent(monkeypatch)

    def fake_invoke(*args, **kwargs):
        raise AssertionError("不应被调用:已取消")

    monkeypatch.setattr(a, "_invoke_with_tools", fake_invoke)
    out = a.run("hi", should_cancel=lambda: True)
    assert "中断" in out


def test_run_normal_path_unaffected_when_no_cancel(monkeypatch):
    """不传 should_cancel 时,正常路径 content 原样返回。"""
    a = _bare_agent(monkeypatch)

    msg = types.SimpleNamespace(content="正常回答", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None,
    )
    monkeypatch.setattr(a, "_invoke_with_tools", lambda *args, **kw: resp)
    out = a.run("hi")  # 不传 should_cancel
    assert out == "正常回答"


def test_run_cancel_stops_after_tool_round(monkeypatch):
    """取消信号在第二步 (工具执行后) 到来时,不继续调用 invoke。"""
    a = _bare_agent(monkeypatch)
    call_count = [0]

    # 第一次返回含 tool_calls 的响应,第二次不会被调用(cancel=True 从第2步开始生效)
    tc = types.SimpleNamespace(
        id="call_1",
        function=types.SimpleNamespace(name="dummy", arguments="{}"),
    )
    first_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(
                content="",
                tool_calls=[tc],
                reasoning_content=None,
            ),
            finish_reason="tool_calls",
        )],
        usage=None,
    )

    def fake_invoke(*args, **kwargs):
        call_count[0] += 1
        return first_resp

    monkeypatch.setattr(a, "_invoke_with_tools", fake_invoke)

    # 第一步不取消,第二步取消
    step = [0]

    def cancel_from_step2():
        step[0] += 1
        return step[0] > 1

    # dummy tool:直接返回
    monkeypatch.setattr(
        a,
        "_execute_tool_calls",
        lambda tc_list, msgs, **kw: 1,
        raising=False,
    )
    monkeypatch.setattr(a, "_refresh_todo_injection", lambda msgs: None, raising=False)

    out = a.run("go", should_cancel=cancel_from_step2)
    # 只调用过一次 invoke(第二步取消,不再调用第二次)
    assert call_count[0] == 1
    assert "中断" in out


def test_should_cancel_not_in_request_kwargs(monkeypatch):
    """should_cancel は run(**kwargs) 側の **kwargs に入らない → OpenAI クライアントに漏れない。

    _invoke_with_tools 自体は should_cancel を明示的な名前付き引数として受け取るが、
    それは request_kwargs (= dict(kwargs)) には含まれず OpenAI SDK へ渡されない。
    このテストでは run(**kwargs) に余分なキーが入らないことを確認する。
    """
    a = _bare_agent(monkeypatch)
    captured_extra_kwargs = {}  # on_text_chunk/on_reasoning_chunk/should_cancel 以外の **kwargs

    msg = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None,
    )

    def fake_invoke(messages, tools, tool_choice, *,
                    on_text_chunk=None, on_reasoning_chunk=None,
                    should_cancel=None, **extra):
        # extra は run(**kwargs) から流れてくる余剰 kwargs;should_cancel はここに来てはいけない
        captured_extra_kwargs.update(extra)
        return resp

    monkeypatch.setattr(a, "_invoke_with_tools", fake_invoke)
    a.run("hi", should_cancel=lambda: False)
    # should_cancel は run(**kwargs) 経由の extra に紛れ込んではいけない
    assert "should_cancel" not in captured_extra_kwargs
