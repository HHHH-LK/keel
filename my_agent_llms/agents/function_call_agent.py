
"""基于 OpenAI 原生 function calling 的 Agent。

与 ReActAgent 的区别在于工具调用通道：ReAct 走文本协议
(`[TOOL_CALL:xxx:yyy]` + 正则解析)，本 Agent 走 OpenAI 协议层
的 tools / tool_calls 字段，由模型保证返回合法 JSON 参数，鲁棒性更强。
"""
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from my_agent_llms.core.agent import Agent
from my_agent_llms.core.config import Config
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.core.message import Message
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.verify.replan import make_plan
from my_agent_llms.planning.todo import todo_system_message, TODO_HEADING

# 结构化触发 todo 的阈值:本轮连续改动达这么多次还没列清单 → 提醒模型用 write_todo
TODO_NUDGE_THRESHOLD = 3

logger = logging.getLogger(__name__)

# 流式回调签名:
# - on_text_chunk(text): 模型每吐一段可见 content 时调用一次
# - on_reasoning_chunk(text): 思考模型 (MiMo/DeepSeek-R1/Qwen-thinking 等)
#   每吐一段 reasoning_content 时调用一次。UI 可以借此期间保留 "thinking..."
#   spinner,避免在思考阶段被错误地认为"卡住了"
# - on_tool_call(name, args_dict): 工具调用即将执行时调用一次
# - on_permission_request(name, args_dict, preview) → bool: 需审批工具执行前询问用户
# - on_tool_result(name, result_str, elapsed_sec): 工具执行完成后立即调用,带耗时
# - on_llm_done(elapsed_sec, prompt_tokens, completion_tokens): 每次 LLM invoke
#   结束后调用,带耗时 + token 用量 (provider 不支持 usage 时 tokens 为 None)
TextChunkCallback = Callable[[str], None]
ReasoningChunkCallback = Callable[[str], None]
ToolCallCallback = Callable[[str, Dict[str, Any]], None]
PermissionCallback = Callable[[str, Dict[str, Any], str], bool]
ToolResultCallback = Callable[[str, str, float], None]
LLMDoneCallback = Callable[[float, Optional[int], Optional[int]], None]
# - on_verify_phase(round_idx): 进入一轮"自证重试"前调用(模型已答、门判未达标、
#   即将注入反馈再跑一轮)。UI 借此给随后的核对动作打一个区别于普通工具调用的归属头,
#   不再让 verify 触发的 Grep/Bash 混在主对话里像"答完又莫名其妙翻工具"。
VerifyPhaseCallback = Callable[[int], None]


class MyFunctionCallAgent(Agent):
    """使用 OpenAI 原生函数调用机制的 Agent。"""

    def __init__(self,
                 name: str,
                 llm: MyLLM,
                 tool_registry: ToolRegistry,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_steps: int = 5,
                 tool_timeout: Optional[float] = None,
                 workspace=None,
                 enable_verify: bool = False,
                 replan_budget: int = 1,
                 todo_store=None,
                 spec_generator=None,
                 convergence_judge=None,
                 enable_tdd: bool = False,
                 **kwargs):
        super().__init__(name, llm, system_prompt, config, **kwargs)
        if llm.provider not in MyLLM.OPENAI_COMPATIBLE_PROVIDERS:
            raise ValueError(
                f"FunctionCallAgent 仅支持 OpenAI 兼容 provider，当前为: {llm.provider}"
            )
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        # 单工具执行超时(秒)。None = 不限。超时只"放弃等待"并回喂文案,
        # 不强杀线程(Python 线程杀不掉;后台线程会继续跑到自己结束)。
        self.tool_timeout = tool_timeout
        # 文件类任务的产物边界:注入后 verify 的 field_equals/command_ok 硬 oracle 才能读回校验。
        self.workspace = workspace
        self.last_tool_call_count = 0  # chat 层读取用作 meta
        self._install_memory_tools(self.tool_registry)
        # ── 在线验证-重试(默认关闭,开关隔离,不破坏旧行为)──
        self.replan_budget = replan_budget
        self.todo_store = todo_store
        self.enable_verify = enable_verify
        # ── TDD 模式(默认关闭,开关隔离)──
        self.enable_tdd = enable_tdd
        self._in_tdd = False   # 重入保护:_run_tdd 内部再调 run() 写实现时不再触发 TDD
        if enable_verify:
            from my_agent_llms.verify import (
                SpecGenerator, CheckerRunner, ConvergenceJudge)
            self.spec_generator = spec_generator or SpecGenerator(llm)
            self.convergence_judge = convergence_judge or ConvergenceJudge()
            self.checker_runner = CheckerRunner(llm=llm)
        else:
            self.spec_generator = None
            self.convergence_judge = None
            self.checker_runner = None

    def run(self,
            input_text: str,
            tool_choice: Union[str, dict] = "auto",
            on_text_chunk: Optional[TextChunkCallback] = None,
            on_tool_call: Optional[ToolCallCallback] = None,
            on_permission_request: Optional[PermissionCallback] = None,
            on_tool_result: Optional[ToolResultCallback] = None,
            on_llm_done: Optional[LLMDoneCallback] = None,
            on_reasoning_chunk: Optional[ReasoningChunkCallback] = None,
            on_verify_phase: Optional[VerifyPhaseCallback] = None,
            on_verify_start: Optional[Callable[[], None]] = None,
            should_cancel: Optional[Callable[[], bool]] = None,
            **kwargs) -> str:
        """运行一轮。on_text_chunk/on_tool_call 不传 → 同步阻塞行为，传 → 流式回调。

        should_cancel: 可选的取消信号函数。每次循环顶部 + 流式逐 chunk 均检查。
        返回 True 时立即中断,返回 "(已被用户中断 esc)" 标记字符串。
        """
        _cancel = should_cancel or (lambda: False)
        _tdd = self._maybe_run_tdd(input_text)
        if _tdd is not None:
            return _tdd
        system_prompt = self._apply_honesty_contract(self.system_prompt)
        # query=input_text 让 memory 做 L0 query-aware 加权 + 被动 recall
        messages: List[Dict[str, Any]] = list(
            self.memory.assemble_context(system_prompt, query=input_text)
        )
        messages.append({"role": "user", "content": input_text})

        tools = self._build_tool_schemas()
        tool_call_count = 0
        self._turn_mutated = False    # 本轮是否动过有副作用工具 → 决定 verify 闸门是否开
        self._turn_mutation_count = 0  # 本轮"产物型"改动次数 → 结构化触发 todo 提醒
        self._todo_nudged = False      # 本轮是否已提醒过用 write_todo(只提醒一次)

        final_response = ""
        # 验证-重试 gate 的循环外状态(spec 首次进 gate 时惰性生成一次,之后不变)
        _verify_spec = None
        _verify_round = 0
        _verify_history: list = []
        # 原始每轮 passed 历史:用于停滞 oracle 降级判定。**故意不随 replan 清空**
        # (_verify_history 会被清),否则不可行动的 command_ok 又会把循环拖回空转。
        _verify_passed_history: list = []
        _verify_best = None
        # ── C: 验证未通过前不把候选答案显示给用户 ──
        # enable_verify 且本轮有改动时,候选最终答案先缓冲、不流式;闸门判通过/止损后再
        # 一次性 emit。被拒候选(重答/replan 那几轮)直接丢弃,用户永远看不到。
        # 带 tool_calls 的那次响应里的文本是【叙述/预说明】(不是候选答案)→ 照常 flush 给用户。
        _text_buf: list = []
        _suppress = [False]            # 可变盒:闭包里改
        _answer_shown_live = [False]   # 最终答案是否已流式给过用户 → 末尾不再重复 emit

        def _gated_text(t):
            if _suppress[0]:
                _text_buf.append(t)
            elif on_text_chunk is not None:
                on_text_chunk(t)

        def _flush_buf_live():
            """把缓冲文本真正吐给用户(用于带 tool_calls 的叙述/预说明)。"""
            if _text_buf and on_text_chunk is not None:
                try:
                    on_text_chunk("".join(_text_buf))
                except Exception:
                    logger.exception("on_text_chunk 回调异常,忽略")
            _text_buf.clear()

        # 默认 0:__new__ 建的测试 agent 无此属性 → 不 replan,保持旧行为
        _replan_budget = getattr(self, "replan_budget", 0)
        # 注意(Phase 1 已知限制):验证重试轮与工具轮共享 self.max_steps 预算。
        # 工具用得多时验证轮会被挤压,可能到不了 convergence_judge.hard_cap。Phase 2 再拆独立预算。
        for _ in range(self.max_steps):
            if _cancel():
                final_response = final_response or "(已被用户中断 esc)"
                break
            self._refresh_todo_injection(messages)
            self._maybe_nudge_todo(messages)      # 改够多次还没列清单 → 提醒用 write_todo
            # 本轮是否压制候选文本 = 验证将会跑(开关开 且 已动过改动)。压制 ⟺ 会验证。
            _text_buf.clear()
            _suppress[0] = self._should_run_verify(self._turn_mutated)
            t_llm = time.monotonic()
            response = self._invoke_with_tools(
                messages, tools, tool_choice,
                on_text_chunk=_gated_text,
                on_reasoning_chunk=on_reasoning_chunk,
                should_cancel=_cancel,
                **kwargs,
            )
            llm_elapsed = time.monotonic() - t_llm
            # 把 usage (provider 可能没给) 拿出来,通过回调上报
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            if on_llm_done is not None:
                try:
                    on_llm_done(llm_elapsed, pt, ct)
                except Exception:
                    logger.exception("on_llm_done 回调异常,忽略")

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                candidate = self._extract_message_content(message)
                # 工具门控:本轮没动过【有副作用】工具(闲聊/纯问答/只读探索)→ 跳过验证,
                # 零开销返回首答。只读任务不该被凭空生成 spec、逼模型凑关键词/重答。
                if not self._should_run_verify(self._turn_mutated):
                    final_response = candidate
                    _answer_shown_live[0] = True   # 未压制 → 已流式给用户,末尾不再 emit
                    break
                # ── 验证-重试 gate(硬插入,不靠模型自觉)──
                # 候选已被 _gated_text 缓冲(未显示)。先告知 UI"校验中",再跑闸门。
                self._emit_verify_start(on_verify_start)
                if _verify_spec is None:
                    _verify_spec = self.spec_generator.generate(
                        input_text, tools=self.tool_registry.list_tools())
                gate = self._verify_gate(
                    candidate, messages, _verify_spec,
                    _verify_round, _verify_history, _verify_best,
                    _verify_passed_history)
                if gate["demoted"]:
                    logger.info("verify: 停滞 command_ok 已降级为 SKIP(不可行动,杜绝空转): %s",
                                sorted(gate["demoted"]))
                _verify_best = gate["best"]
                _verify_round += 1
                # STUCK/OSCILLATING(原地打转)且还有预算 → 换思路重新规划,而非放弃
                if gate["needs_replan"] and _replan_budget > 0:
                    _replan_budget -= 1
                    self._emit_verify_phase(on_verify_phase, _verify_round)
                    plan = self._make_plan(input_text, gate["feedback"])
                    messages.append({"role": "user", "content":
                        f"⚠️ 之前的做法卡住了。换个思路,按以下新计划重做:\n{plan}"})
                    _verify_history.clear()   # 换思路 → 残差趋势重新起算,旧 STUCK 历史不再拖累
                    continue
                if gate["stop"]:
                    final_response = gate["best"].result
                    break
                self._emit_verify_phase(on_verify_phase, _verify_round)
                messages.append({"role": "user", "content": gate["feedback"]})
                continue

            # 带 tool_calls → 缓冲的是叙述/预说明,不是候选答案 → flush 给用户。
            _flush_buf_live()
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            tool_call_count += self._execute_tool_calls(
                tool_calls, messages,
                on_tool_call=on_tool_call,
                on_permission_request=on_permission_request,
                on_tool_result=on_tool_result,
            )

        # 验证开启且主循环耗尽预算却没 break:返回全程最优的已验证候选,
        # 而不是再做一次"无验证"的兜底调用(否则丢掉 best、违背"始终返回 best")。
        if not final_response and getattr(self, "enable_verify", False) and _verify_best is not None:
            final_response = _verify_best.result

        if not final_response:
            t_llm = time.monotonic()
            response = self._invoke_with_tools(
                messages, tools, "none",
                on_text_chunk=on_text_chunk,
                on_reasoning_chunk=on_reasoning_chunk,
                should_cancel=_cancel,
                **kwargs,
            )
            llm_elapsed = time.monotonic() - t_llm
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            if on_llm_done is not None:
                try:
                    on_llm_done(llm_elapsed, pt, ct)
                except Exception:
                    logger.exception("on_llm_done 回调异常,忽略")
            final_response = self._extract_message_content(response.choices[0].message)
            _answer_shown_live[0] = True   # 兜底调用走真 on_text_chunk 直接流式了

        if not final_response:
            final_response = (
                "(模型本轮未产生文本响应。可能是 max_tokens 在思考阶段就用尽了，"
                "或本轮触发了模型的纯思考通道。可尝试加大 LLM_MAX_TOKENS、换非 thinking 模型，"
                "或换个表述重试。)"
            )
            logger.warning("FunctionCallAgent: 主循环与 fallback 都未拿到文本响应")
            if on_text_chunk is not None:
                # 占位文本也得让用户看到（前面流式可能一字未出）
                try:
                    on_text_chunk(final_response)
                    _answer_shown_live[0] = True
                except Exception:
                    logger.exception("on_text_chunk 回调异常,忽略")

        final_response = self._run_response_hooks(input_text, final_response, messages)
        # C: 候选被压制(未流式)且最终答案还没显示过 → 此刻一次性 emit 已验证的最终答案。
        if final_response and not _answer_shown_live[0] and on_text_chunk is not None:
            try:
                on_text_chunk(final_response)
            except Exception:
                logger.exception("on_text_chunk 回调异常,忽略")
        # TDD 子运行(_in_tdd)期间不在这里写 memory:由顶层 _run_tdd 统一补记一次,
        # 避免把实现阶段的子提示词("请写实现…")当成用户输入污染记忆。
        if not getattr(self, "_in_tdd", False):
            self._finalize_turn(input_text, final_response, task_turn=tool_call_count > 0)
        self.last_tool_call_count = tool_call_count
        logger.debug(f"{self.name} 响应完成")
        return final_response

    @staticmethod
    def _emit_verify_start(cb) -> None:
        """安全触发 on_verify_start:每轮进闸门前调用,UI 借此显示'校验中'。"""
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.exception("on_verify_start 回调异常,忽略")

    @staticmethod
    def _emit_verify_phase(cb, round_idx: int) -> None:
        """安全触发 on_verify_phase 回调:无回调则 no-op,回调异常不冒泡毁整轮。"""
        if cb is None:
            return
        try:
            cb(round_idx)
        except Exception:
            logger.exception("on_verify_phase 回调异常,忽略")

    def _verify_gate(self, candidate, messages, spec, round_idx, history, best,
                     passed_history=None):
        """对候选答案跑一轮验证,更新 best/history,返回是否止损 + 反馈文案。

        passed_history: run() 持有的【原始】每轮 passed 累积列表(不随 replan 清空)。
            据此把连续 N 轮判 False 的不可行动 command_ok 降级为 SKIP,杜绝空转。
        """
        from my_agent_llms.verify import CheckContext, residual, fingerprint, Verdict
        from my_agent_llms.verify.residual import effective_count
        from my_agent_llms.verify.loop import feedback_from
        from my_agent_llms.verify.convergence import Round
        from my_agent_llms.verify.stall import stalled_oracle_ids, apply_demotion

        ctx = CheckContext(
            result=candidate, trajectory=messages,
            workspace=getattr(self, "workspace", None))
        raw_passed = self.checker_runner.run(spec, ctx)
        # 停滞 oracle 降级:先把本轮原始结果入历史,再据末 N 轮挑出该降级的 command_ok。
        if passed_history is not None:
            passed_history.append(dict(raw_passed))
            demoted = stalled_oracle_ids(spec, passed_history)
        else:
            demoted = set()
        passed = apply_demotion(raw_passed, demoted)   # 降级后的视图,供残差/收敛/反馈用

        res = residual(spec, passed)
        if best is None or res < best.residual:   # 严格小于 → 平局保留更早那轮
            best = SimpleNamespace(residual=res, result=candidate, passed=passed)
        fp = fingerprint(candidate, messages)
        verdict = self.convergence_judge.judge(
            round_idx, res, fp, history,
            has_effective=effective_count(spec, passed) > 0)
        history.append(Round(residual=res, fingerprint=fp))
        stop = verdict != Verdict.CONTINUE
        needs_replan = verdict in (Verdict.STUCK, Verdict.OSCILLATING)
        return {"best": best, "stop": stop, "needs_replan": needs_replan,
                "demoted": demoted,
                "feedback": feedback_from(spec, passed) or "请继续完善答案。"}

    def _make_plan(self, task: str, stuck_feedback: str) -> str:
        """卡住时换思路重新规划(薄包装 verify.replan.make_plan,便于测试 monkeypatch)。"""
        return make_plan(self.llm, task, stuck_feedback)

    def _maybe_run_tdd(self, input_text: str):
        """run() 顶部调用。接管返回最终字符串;不接管返回 None(走老路)。"""
        if not getattr(self, "enable_tdd", False) or getattr(self, "_in_tdd", False):
            return None
        if not self._tdd_should_run(input_text):
            return None
        return self._run_tdd(input_text)

    def _tdd_should_run(self, input_text: str) -> bool:
        from my_agent_llms.tdd import classify
        return classify(self.llm, input_text).use_tdd

    def _run_tdd(self, input_text: str) -> str:
        # 注意(Phase 1 已知限制):TDD 路径不透传 run() 的流式回调
        # (on_text_chunk/on_tool_call 等),CLI 在 TDD 期间无过程输出,只在结束拿最终串。
        # 透传回调留作后续(与流式渲染一并做)。
        from my_agent_llms.tdd import run_tdd
        self._in_tdd = True
        saved_verify = self.enable_verify
        self.enable_verify = False   # TDD 自带红/绿门,关掉事后 verify 避免双重验证
        try:
            result = run_tdd(
                llm=self.llm, workspace=self.workspace, task=input_text,
                implement_fn=self._tdd_implement,
                # 已由 _tdd_should_run 确认走 TDD,免得 orchestrator 再 classify 一次(省一次 LLM 调用)
                user_override=True)
            if result.degraded:
                # 降级:回普通工具循环跑一遍(_in_tdd=True 防再触发 TDD)。
                # 先恢复 verify,让降级回退仍享有事后验证保险。
                self.enable_verify = saved_verify
                message = self.run(input_text)
            else:
                message = result.message
        finally:
            self._in_tdd = False
            self.enable_verify = saved_verify
        # 顶层补记一次 memory:原始任务 ↔ 最终结果(嵌套子 run 已被抑制 finalize)。
        self._finalize_turn(input_text, message, task_turn=True)
        return message

    def _tdd_implement(self, task: str, test_paths, feedback: str) -> None:
        """实现回调:让主 agent 用工具循环写实现去满足测试(不许改测试)。"""
        prompt = (f"请写实现,让这些测试通过:{', '.join(test_paths)}。"
                  f"先用 Read 读测试了解要求。**不要修改测试文件**,只写实现代码。")
        if feedback:
            prompt += f"\n上一轮:{feedback}"
        self.run(prompt)  # _in_tdd=True 保证不再触发 TDD;enable_verify 已临时关闭

    def _refresh_todo_injection(self, messages):
        """每轮 invoke 前就地刷新 todo 注入:删上一份、加当前(非空才加)。短任务零开销。"""
        store = getattr(self, "todo_store", None)
        if store is None:
            return
        messages[:] = [m for m in messages
                       if not (m.get("role") == "system"
                               and TODO_HEADING in (m.get("content") or ""))]
        msg = todo_system_message(store)
        if msg:
            messages.append(msg)

    def _maybe_nudge_todo(self, messages):
        """结构化触发:本轮连续改动达阈值却没列清单 → 注入一次提醒,促模型用 write_todo。

        todo 没有判别器(全靠模型自觉),弱模型常对大任务不列清单。这里用"实际改动
        次数"这个结构信号兜底:改够多次还没计划 → 提醒。注入一次,空清单才提醒。"""
        store = getattr(self, "todo_store", None)
        if store is None or getattr(self, "_todo_nudged", False):
            return
        if getattr(store, "items", None):           # 已有清单 → 不打扰
            return
        if getattr(self, "_turn_mutation_count", 0) < TODO_NUDGE_THRESHOLD:
            return
        messages.append({"role": "system", "content":
            "提示:你已经连续做了多处改动,但还没建任务清单。这看起来是个多步任务——"
            "请马上调用 write_todo 把(剩余)步骤列成分步计划,并在每完成一步后立刻"
            "更新状态,让用户能跟踪进度。"})
        self._todo_nudged = True

    def _should_run_verify(self, mutated: bool) -> bool:
        """verify-retry 闸门是否该开:开关开 且 本轮动过【有副作用】工具。

        纯只读探索/问答(Read/LS/Glob/Grep)不验证 —— 否则 SpecGenerator 会给开放
        问答凭空编 spec(string_contains/tool_called),反馈回灌逼模型凑关键词、重答
        整篇(见 test_verify_gate_trigger)。"""
        return bool(getattr(self, "enable_verify", False)) and mutated

    def _tool_is_side_effect_free(self, name: str) -> bool:
        """白名单判定:仅 Tool.side_effect_free=True 的才允许并行。
        轻量函数工具(无 Tool 对象)与未标记的一律按有副作用处理 → 串行。"""
        tool = self.tool_registry.get_tool(name)
        return bool(getattr(tool, "side_effect_free", False))

    def _tool_is_verify_exempt(self, name: str) -> bool:
        """verify 闸门豁免:记账型工具(记忆/待办)有副作用但非代码产物,不触发事后 verify。"""
        tool = self.tool_registry.get_tool(name)
        return bool(getattr(tool, "verify_exempt", False))

    @staticmethod
    def _tool_timeout_message(name: str, timeout: float) -> str:
        return (f"⏱️ 工具 '{name}' 执行超时(>{timeout}s),已放弃等待;"
                f"线程可能仍在后台运行,其结果将被忽略。")

    def _run_single_tool(self, name: str, args: Any, timeout: Optional[float] = None):
        """执行单个工具,返回 (result, elapsed_sec)。

        timeout 非 None 时,超过即放弃等待并返回超时文案。底层线程不被强杀
        (shutdown(wait=False)),会在后台继续跑完,但其返回值被忽略。
        """
        t = time.monotonic()
        if timeout is None:
            result = self.tool_registry.execute_tool(name, args)
            return result, time.monotonic() - t

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(self.tool_registry.execute_tool, name, args)
        try:
            result = fut.result(timeout=timeout)
        except FuturesTimeout:
            result = self._tool_timeout_message(name, timeout)
        finally:
            ex.shutdown(wait=False)
        return result, time.monotonic() - t

    def _execute_tool_calls(self,
                            tool_calls,
                            messages: List[Dict[str, Any]],
                            *,
                            on_tool_call: Optional[ToolCallCallback],
                            on_permission_request: Optional[PermissionCallback],
                            on_tool_result: Optional[ToolResultCallback]) -> int:
        """同一轮多个 tool_call 的三阶段调度,返回"尝试次数"(含被拒绝)。

        A. 串行(原顺序):parse/类型转换/on_tool_call/审批,产出执行计划。
        B. 执行:side_effect_free 工具并行(线程池),其余按原顺序串行。
        C. 串行(原顺序):结果按 tool_call_id 原顺序回填 messages + on_tool_result 上报。

        审批是对人 IO,必须在 A 阶段串行、按序进行;并行只发生在 B 阶段且仅限白名单工具。
        """
        # ── Phase A:串行产出计划 ──
        plans: List[Dict[str, Any]] = []
        attempts = 0
        for tc in tool_calls:
            name = tc.function.name
            args = self._parse_function_call_arguments(tc.function.arguments)
            args = self._convert_parameter_types(name, args)
            if on_tool_call is not None:
                try:
                    on_tool_call(name, args)
                except Exception:
                    logger.exception("on_tool_call 回调异常,忽略不影响主流程")

            plan = {"tc": tc, "name": name, "args": args,
                    "result": None, "elapsed": 0.0, "execute": True, "report": True}

            tool_obj = self.tool_registry.get_tool(name)
            # 审批判定:工具若提供 approval_required_for(args) 则按命令内容动态决定
            # (如 Bash 只对危险命令弹审批);否则回退静态 requires_approval。
            _checker = getattr(tool_obj, "approval_required_for", None)
            try:
                _needs_approval = (bool(_checker(args)) if _checker is not None
                                   else bool(getattr(tool_obj, "requires_approval", False)))
            except Exception:
                logger.exception("approval_required_for 异常,回退 requires_approval")
                _needs_approval = bool(getattr(tool_obj, "requires_approval", False))
            if (tool_obj is not None
                    and _needs_approval
                    and on_permission_request is not None):
                try:
                    preview = tool_obj.preview_for_approval(args)
                except Exception:
                    logger.exception("preview_for_approval 异常,降级为 repr")
                    preview = repr(args)
                try:
                    allowed = on_permission_request(name, args, preview)
                except Exception:
                    logger.exception("on_permission_request 异常,默认拒绝")
                    allowed = False
                if not allowed:
                    plan["result"] = f"用户拒绝了对 {name} 的调用"
                    plan["execute"] = False
                    plan["report"] = False  # 被拒绝不算一次真正执行,不上报 on_tool_result
                    attempts += 1
            plans.append(plan)

        # ── 结构闸门:否决"边执行改动边预先打勾"(看到结果前不许标 completed)──
        self._veto_premature_completions(plans)

        # ── Phase B:执行。白名单并行,其余按原顺序串行 ──
        timeout = getattr(self, "tool_timeout", None)
        to_exec = [p for p in plans if p["execute"]]
        serial = [p for p in to_exec if not self._tool_is_side_effect_free(p["name"])]
        parallel = [p for p in to_exec if self._tool_is_side_effect_free(p["name"])]
        # verify 闸门:只有"产物型"副作用工具(写文件/改代码/跑命令)才算动过改动;
        # 记账型(记忆/待办)虽串行执行,但 verify_exempt → 不触发事后验证。
        _mutating = [p for p in serial if not self._tool_is_verify_exempt(p["name"])]
        if _mutating:
            self._turn_mutated = True
            # 累计本轮产物型改动次数(供结构化 todo 提醒判定)
            self._turn_mutation_count = getattr(self, "_turn_mutation_count", 0) + len(_mutating)

        for p in serial:
            p["result"], p["elapsed"] = self._run_single_tool(p["name"], p["args"], timeout)
            attempts += 1

        if parallel:
            from concurrent.futures import (
                ThreadPoolExecutor, as_completed, wait as futures_wait)
            ex = ThreadPoolExecutor(max_workers=min(len(parallel), 8))
            t0 = time.monotonic()
            futs = {ex.submit(self.tool_registry.execute_tool, p["name"], p["args"]): p
                    for p in parallel}
            try:
                if timeout is None:
                    for fut in as_completed(futs):
                        p = futs[fut]
                        p["result"] = fut.result()
                        p["elapsed"] = time.monotonic() - t0
                else:
                    # 整批共享一个 deadline(它们同时起跑);未完成的回喂超时文案,
                    # 不强杀,不等待——慢工具不阻塞已完成的快工具。
                    done, _not_done = futures_wait(futs, timeout=timeout)
                    for fut, p in futs.items():
                        if fut in done:
                            try:
                                p["result"] = fut.result()
                            except Exception as e:
                                p["result"] = f"❌ 工具 '{p['name']}' 执行异常: {e}"
                        else:
                            p["result"] = self._tool_timeout_message(p["name"], timeout)
                        p["elapsed"] = time.monotonic() - t0
            finally:
                ex.shutdown(wait=False)
            attempts += len(parallel)

        # ── Phase C:按原顺序回填 + 上报 ──
        for p in plans:
            messages.append({
                "role": "tool",
                "tool_call_id": p["tc"].id,
                "content": str(p["result"]),
            })
            if p["report"] and on_tool_result is not None:
                try:
                    on_tool_result(p["name"], str(p["result"]), p["elapsed"])
                except Exception:
                    logger.exception("on_tool_result 回调异常,忽略不影响主流程")

        return attempts

    def _veto_premature_completions(self, plans: List[Dict[str, Any]]) -> None:
        """结构闸门:同一轮里既执行【改动类】工具、又想把某步标 completed → 否决该 write_todo。

        根因:completed 全靠模型自报、与真实执行结果零绑定;模型可在看到工具结果前
        就在同一轮预先打勾。这里硬性拆开「做」与「打勾」:有改动工具在跑时,write_todo
        里【新增】的 completed 一律否决(原样带回的已完成项不算),回喂纠正文案让模型下一轮
        看到结果后再单独标。只读工具(side_effect_free)不算改动,不触发本闸门。"""
        store = getattr(self, "todo_store", None)
        if store is None:
            return
        has_work = any(
            p["execute"] and p["name"] != "write_todo"
            and not self._tool_is_side_effect_free(p["name"])
            for p in plans)
        if not has_work:
            return
        from my_agent_llms.planning.todo import parse_todo_lines
        prev = {it["content"]: it["status"] for it in store.items}
        for p in plans:
            if p["name"] != "write_todo" or not p["execute"]:
                continue
            incoming = parse_todo_lines((p["args"] or {}).get("todos"))
            new_done = [it["content"] for it in incoming
                        if it["status"] == "completed"
                        and prev.get(it["content"]) != "completed"]
            if new_done:
                p["execute"] = False
                p["result"] = (
                    "⛔ 本轮你正在执行改动类动作,不能同时把步骤标 completed:"
                    + "、".join(new_done)
                    + "。请先看到这些动作的工具结果、确认这步真的成功了,下一轮再单独调 "
                    "write_todo 标 completed(现在可以只把下一步标 in_progress,但别预先打勾)。")

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.tool_registry.to_openai_schemas()

    def _invoke_with_tools(self,
                           messages: List[Dict[str, Any]],
                           tools: List[Dict[str, Any]],
                           tool_choice: Union[str, dict],
                           on_text_chunk: Optional[TextChunkCallback] = None,
                           on_reasoning_chunk: Optional[ReasoningChunkCallback] = None,
                           should_cancel: Optional[Callable[[], bool]] = None,
                           **kwargs):
        client = getattr(self.llm, "client", None)
        if client is None:
            raise RuntimeError("MyLLM 客户端未初始化，无法执行函数调用。")

        # NOTE: should_cancel is an explicit param and must NOT be in request_kwargs.
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            request_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        base_request: Dict[str, Any] = dict(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **request_kwargs,
        )

        response = self._stream_chat_completion(
            client, base_request, on_text_chunk, on_reasoning_chunk, should_cancel=should_cancel)

        # 自救：撞 max_tokens 上限、content 又是空（thinking 模型把预算吃在
        # reasoning 阶段最常见的失败形态）→ 把预算翻倍重试一次。
        choice = response.choices[0]
        message = choice.message
        empty_content = not (getattr(message, "content", None) or "").strip()
        no_tool_calls = not getattr(message, "tool_calls", None)
        if choice.finish_reason == "length" and empty_content and no_tool_calls:
            current_budget = request_kwargs.get("max_tokens") or self.llm.max_tokens or 8192
            bumped_request = dict(base_request)
            bumped_request["max_tokens"] = current_budget * 2
            logger.warning(
                "finish_reason=length 且响应为空，max_tokens %s→%s 重试",
                current_budget, bumped_request["max_tokens"],
            )
            response = self._stream_chat_completion(
                client, bumped_request, on_text_chunk, on_reasoning_chunk, should_cancel=should_cancel)

        return response

    @staticmethod
    def _stream_chat_completion(
        client,
        base_request: Dict[str, Any],
        on_text_chunk: Optional[TextChunkCallback],
        on_reasoning_chunk: Optional[ReasoningChunkCallback] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        """开 stream=True 调 chat.completions, 累积出与非流式等价的 response 对象。

        - 文本 content chunk → 实时回调 on_text_chunk 并累积
        - reasoning_content chunk → 实时回调 on_reasoning_chunk (用于 UI 显示
          "thinking..." 指示);同时累积作为 content 为空时的兜底
        - tool_calls chunk → 按 index 累加 arguments JSON 片段
        - finish_reason 取最后一个非空值
        - should_cancel: 每个 chunk 前检查;True 时中断流式读取(累积到当前为止)

        返回的 SimpleNamespace 跟原生 ChatCompletion 同形:
        response.choices[0].message.{content, tool_calls, reasoning_content}
        response.choices[0].finish_reason
        """
        stream_request = dict(base_request)
        stream_request["stream"] = True
        # 让兼容的 provider (OpenAI/DeepSeek/MiMo/...) 在最后一帧带上 usage
        stream_request["stream_options"] = {"include_usage": True}

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        # tool_calls 按 index 累积: {index: {id, name, arguments(str)}}
        tool_acc: Dict[int, Dict[str, str]] = {}
        finish_reason: Optional[str] = None
        usage = None  # 最后一帧的 usage,部分 provider 可能不返回

        stream = client.chat.completions.create(**stream_request)
        for chunk in stream:
            if should_cancel is not None and should_cancel():
                break
            # usage 帧 (有的 provider 把它放在 choices 为空的最后一帧)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            delta = choice.delta
            content = getattr(delta, "content", None) or ""
            if content:
                content_parts.append(content)
                if on_text_chunk is not None:
                    try:
                        on_text_chunk(content)
                    except Exception:
                        logger.exception("on_text_chunk 回调异常,忽略不影响流式累积")

            reasoning = getattr(delta, "reasoning_content", None) or ""
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning_chunk is not None:
                    try:
                        on_reasoning_chunk(reasoning)
                    except Exception:
                        logger.exception("on_reasoning_chunk 回调异常,忽略不影响流式累积")

            for tc_chunk in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc_chunk, "index", 0) or 0
                entry = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                tc_id = getattr(tc_chunk, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                fn = getattr(tc_chunk, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        entry["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        entry["arguments"] += fn_args

        # 组装等价于非流式 message 的对象
        tool_calls_list = None
        if tool_acc:
            tool_calls_list = [
                SimpleNamespace(
                    id=tool_acc[i]["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=tool_acc[i]["name"],
                        arguments=tool_acc[i]["arguments"],
                    ),
                )
                for i in sorted(tool_acc)
            ]

        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)
        message = SimpleNamespace(
            # 与 OpenAI SDK 对齐: 空字符串保留为 None 区分"没生成"vs"生成了空串"
            content=full_content if full_content else None,
            tool_calls=tool_calls_list,
            reasoning_content=full_reasoning if full_reasoning else None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    @staticmethod
    def _extract_message_content(message) -> str:
        content = (message.content or "").strip()
        if content:
            return content
        # thinking 模型 (MiMo / Qwen3-thinking / DeepSeek-R1 等) 会把答案
        # 放在 reasoning_content，content 可能为空——降级取它，避免静默丢响应。
        reasoning = (getattr(message, "reasoning_content", "") or "").strip()
        if reasoning:
            logger.warning("content 为空，降级使用 reasoning_content（可能是 max_tokens 不足或模型走了纯思考通道）")
            return reasoning
        return ""

    @staticmethod
    def _parse_function_call_arguments(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"input": raw}

    def _convert_parameter_types(self,
                                 tool_name: str,
                                 args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return args

        type_map = {p.name: p.type for p in tool.get_parameters()}
        return {
            key: self._coerce_value(value, type_map.get(key))
            for key, value in args.items()
        }

    @staticmethod
    def _coerce_value(value: Any, declared_type: Optional[str]) -> Any:
        if declared_type is None or value is None:
            return value
        try:
            if declared_type == "integer" and not isinstance(value, bool):
                return int(value)
            if declared_type == "number":
                return float(value)
            if declared_type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "1", "yes"}
                return bool(value)
            if declared_type == "string" and not isinstance(value, str):
                return str(value)
        except (ValueError, TypeError):
            return value
        return value
