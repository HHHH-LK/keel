"""独立出题方:只产出测试文件内容(path+content),绝不写实现。

设计:返回"建议测试文件"列表,由 orchestrator 负责写盘(过审批)+ 记哈希。
这样出题与写盘分离,出题方易测、隔离清晰。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ProposedTest:
    relpath: str
    content: str


@dataclass
class AuthorResult:
    tests: List[ProposedTest] = field(default_factory=list)
    note: str = ""


_PROMPT = """你是**出题方**,只写测试,不要写实现。
给定任务,产出 pytest 测试文件——测试要真正断言目标行为,
让它在实现缺失/错误时失败(这样才是有效测试)。
测试里的 import 就是接口契约(告诉实现方要提供哪些函数/类)。

{feedback_block}只输出 JSON:
{{"tests": [{{"relpath": "test_xxx.py", "content": "<完整测试文件内容>"}}]}}

任务: {task}"""


def author_tests(llm, task: str, feedback: str = "") -> AuthorResult:
    fb = f"上一轮反馈(必须改进):{feedback}\n\n" if feedback else ""
    prompt = _PROMPT.format(task=task, feedback_block=fb)
    try:
        content = llm.invoke([{"role": "system", "content": prompt}]) or ""
        data = _parse_json(content)
        tests = [ProposedTest(relpath=t["relpath"], content=t["content"])
                 for t in data.get("tests", [])
                 if t.get("relpath") and t.get("content")]
        return AuthorResult(tests=tests)
    except Exception as exc:
        logger.warning("test_author 解析失败: %s", exc)
        return AuthorResult(tests=[], note=f"解析失败: {exc}")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}
