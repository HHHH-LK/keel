"""验证规格:Check / CheckSpec 数据类,以及从任务推导规格的 SpecGenerator。

铁律:规格生成者 ≠ 任务执行者;生成的是"性质(答案必须满足什么)",不是具体答案。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Check:
    id: str
    type: str                      # string_contains|string_absent|field_equals|
                                   # command_ok|tool_called|judge|semantic_support
    params: dict
    weight: float = 1.0
    confidence: float = 1.0        # 规格生成器对该性质的置信度(伪 oracle 降权用)
    is_hard_oracle: bool = False   # True=可执行/解析类真 oracle;False=推导性质/judge


@dataclass
class CheckSpec:
    task: str
    checks: List[Check] = field(default_factory=list)


_SPEC_PROMPT = """你是验收规格生成器,**不是任务执行者**。
给定一个任务,你只产出"答案必须满足的性质/约束",绝不产出具体答案。

优先级(尽量用靠前的):
1. command_ok    —— 跑命令看 exit code(可执行真 oracle),params={{"cmd": "..."}}
2. field_equals  —— 解析产物文件取键比对,params={{"path","key","value"}}
3. tool_called   —— 任务要求必须用过某工具,params={{"tool": "..."}}
4. string_contains / string_absent —— **agent 文本回答**(对话输出,看不到文件)必须含/不含某串,params={{"s": "..."}}
5. judge         —— 实在无法客观核对时的兜底,params={{"rubric": "..."}}

command_ok 命令必须单行可执行;若用 python3 -c,**只能写单个表达式或分号连接的简单语句,
禁止 for/if/while 等复合语句**(单行 `; for` 是 SyntaxError)。需要遍历校验时改用单表达式,如:
python3 -c "import json;d=json.load(open('f.json'));assert all(x['k'].startswith('v') for x in d['items'])"

⚠️ string_*/judge 只能看到 agent 的【文本回答】,看不到文件系统;**任何对文件内容/存在性的检查,
必须用 command_ok 或 field_equals**(它们能读文件),绝不要用 string_contains 去验文件内容。

⚠️ command_ok 的失败反馈【不可行动】(只说"没通过",不告诉模型哪条命令、为什么)。所以
**只在有真实可执行测试时才用 command_ok**(如跑 pytest、编译、校验 JSON 可解析)。纯文档/
文案/README/注释类任务【没有真测试】,不要凭空编 command_ok(那会让模型对着看不见的命令空转);
此类任务的"文件是否存在/某字段是否对"请用 field_equals,其余约束用 judge 或 string_*(验文本回答)。

可用工具: {tools}

只输出一个 JSON 对象,形如:
{{"checks": [
  {{"id": "c1", "type": "string_contains", "params": {{"s": "..."}},
   "weight": 1.0, "confidence": 0.8, "is_hard_oracle": false}}
]}}
hard oracle(command_ok/field_equals/tool_called)请置 is_hard_oracle=true 且 weight 给高(如 10);
推导性质(string_*/judge)按你的把握给 confidence(0~1)。

任务:
{task}
"""


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 回复里抽出第一个 JSON 对象。容忍 ```json fenced``` 包裹。"""
    if not text:
        return None
    # ```json fence``` 标记不含花括号,故正则从首个 { 到末个 } 自然跳过它
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _hard_oracle_fallback(task: str) -> "CheckSpec":
    """SpecGenerator 失败时的兜底:只跑 hard oracle,宁缺毋错。"""
    return CheckSpec(task=task, checks=[
        Check(id="no_traceback", type="string_absent",
              params={"s": "Traceback"}, weight=10.0, is_hard_oracle=True),
    ])


class SpecGenerator:
    """独立 LLM 调用,从任务 T 推导 CheckSpec。只产性质,不产答案。"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, task: str, *, tools: List[str]) -> "CheckSpec":
        try:
            prompt = _SPEC_PROMPT.format(task=task, tools=", ".join(tools) or "(无)")
            reply = self.llm.invoke([{"role": "system", "content": prompt}])
        except Exception:
            logger.exception("SpecGenerator LLM 调用失败,退化为 hard-oracle-only")
            return _hard_oracle_fallback(task)

        obj = _extract_json(reply)
        raw_checks = (obj or {}).get("checks") if isinstance(obj, dict) else None
        if not raw_checks:
            logger.warning("SpecGenerator 解析失败/产出空,退化为 hard-oracle-only")
            return _hard_oracle_fallback(task)

        checks: List[Check] = []
        for i, rc in enumerate(raw_checks):
            try:
                checks.append(Check(
                    id=str(rc.get("id") or f"c{i}"),
                    type=str(rc["type"]),
                    params=dict(rc.get("params") or {}),
                    weight=float(rc.get("weight", 1.0)),
                    confidence=float(rc.get("confidence", 1.0)),
                    is_hard_oracle=bool(rc.get("is_hard_oracle", False)),
                ))
            except Exception:
                logger.warning("跳过非法 check 定义: %r", rc)
        if not checks:
            return _hard_oracle_fallback(task)
        return CheckSpec(task=task, checks=checks)
