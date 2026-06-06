"""检查器库:每个 check 类型 = 一个核对函数。字符串匹配只是最弱的一种。

铁律(spec §2.6):能客观就别主观——
现成工具/执行 > 结构化解析 > 轨迹查询 > embedding 语义 > LLM judge。
任何核对异常一律视为"未通过"(返回 False),绝不让验证本身崩溃。
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from my_agent_llms.verify.spec import Check, CheckSpec

logger = logging.getLogger(__name__)

# 命令自身坏了(不是 agent 失败)的 stderr 标志:这类 check 判 SKIP(None),不计入残差。
_BROKEN_ORACLE_MARKERS = (
    "SyntaxError", "IndentationError", "NameError",
    "ModuleNotFoundError", "ImportError", "command not found",
)


@dataclass
class CheckContext:
    """一次验证能看到的全部:产出、轨迹、环境。"""
    result: str = ""                       # agent 最终文本产出
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # messages
    workspace: Any = None                  # Workspace | None(文件类任务的产物)
    source: Optional[str] = None           # 对照类任务的参考原文


def _iter_tool_calls(trajectory: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """遍历轨迹里所有 assistant 发起的 tool_calls。"""
    for msg in trajectory:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                yield fn


def check_one(check: Check, ctx: CheckContext, *, llm=None) -> Optional[bool]:
    """按类型分发核对。True=过 / False=未过 / None=SKIP(坏 oracle)。任何异常 → False。"""
    try:
        t = check.type
        p = check.params
        if t == "string_contains":
            return p["s"] in ctx.result
        if t == "string_absent":
            return p["s"] not in ctx.result
        if t == "field_equals":
            if ctx.workspace is None:
                return False
            path = ctx.workspace.resolve_read(p["path"])
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                logger.warning("field_equals: %s 解析出的根不是 JSON 对象,视为未通过", p["path"])
                return False
            return obj.get(p["key"]) == p["value"]
        if t == "command_ok":
            cwd = getattr(ctx.workspace, "root", None) if ctx.workspace else None
            # 注意:cmd 来自 LLM 生成的 spec,shell=True 存在注入风险。
            # 这是 hard-oracle 跑真实检查命令的有意取舍;workspace.root 限定 cwd 但不限定命令本身。
            proc = subprocess.run(
                p["cmd"], shell=True, cwd=str(cwd) if cwd else None,
                capture_output=True, timeout=p.get("timeout", 30),
            )
            if proc.returncode == 0:
                return True
            # 区分"命令自身坏了"(SKIP)vs"断言失败"(真未过)。坏 oracle 不该罚 agent。
            stderr = (proc.stderr or b"").decode("utf-8", "replace")
            if any(m in stderr for m in _BROKEN_ORACLE_MARKERS):
                last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
                logger.warning("command_ok 检查命令自身坏了(SKIP,不计入残差): %s | %s",
                               p["cmd"][:80], last)
                return None
            return False
        if t == "tool_called":
            want = p["tool"]
            return any(fn.get("name") == want for fn in _iter_tool_calls(ctx.trajectory))
        if t == "judge":
            if llm is None:
                return False
            verdict = _llm_judge(llm, p["rubric"], ctx)
            return verdict
        if t == "semantic_support":
            # Phase 2 实现(embedding 对照),本阶段不支持。
            logger.warning("semantic_support 尚未实现(Phase 2),视为未通过")
            return False
        logger.warning("未知 check 类型 %r,视为未通过", t)
        return False
    except Exception:
        logger.exception("check %s (type=%s) 核对异常,视为未通过", check.id, check.type)
        return False


def _llm_judge(llm, rubric: str, ctx: CheckContext) -> bool:
    """最弱兜底:独立 LLM 调用,只回 PASS/FAIL。解析首词。"""
    prompt = (
        "你是一个独立的验收员。只根据给定标准判断产出是否通过,"
        "不要补充答案、不要解释过多。\n"
        "第一行必须只输出 PASS 或 FAIL。\n\n"
        f"# 验收标准\n{rubric}\n\n"
        f"# 待验收产出\n{ctx.result}\n"
    )
    reply = llm.invoke([{"role": "system", "content": prompt}]) or ""
    head = reply.strip().upper()
    return head.startswith("PASS")


class CheckerRunner:
    """用 checks[] 逐条核对 ctx,返回 {check_id -> True|False|None}。
    None = SKIP(坏 oracle,如命令自身 SyntaxError),不计入残差。保证每个 id 都有键。"""

    def __init__(self, *, llm=None):
        self.llm = llm

    def run(self, spec: CheckSpec, ctx: CheckContext) -> Dict[str, Optional[bool]]:
        return {c.id: check_one(c, ctx, llm=self.llm) for c in spec.checks}
