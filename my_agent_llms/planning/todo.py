"""规划层:todo 三件套(状态 + 写工具 + 每轮注入)。

灵魂是"每轮注入"——光有工具,模型几轮后会忘;todo_system_message 让计划持续在场。
短任务模型不调 write_todo → store 空 → 不注入 → 零开销。
"""
from __future__ import annotations

import json
from typing import Optional

from my_agent_llms.tools.base import Tool, ToolParameter

_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_VALID_STATUS = {"pending", "in_progress", "completed"}
TODO_HEADING = "## 当前任务清单(进度)"


def _norm_status(status) -> str:
    s = str(status or "").strip()
    return s if s in _VALID_STATUS else "pending"


def _item_from_dict(d) -> Optional[dict]:
    c = str(d.get("content", "")).strip()
    return {"content": c, "status": _norm_status(d.get("status"))} if c else None


def parse_todo_lines(raw):
    """解析成 [{content, status}](空内容丢弃)。容忍模型常用的多种格式:
      - 'status|内容' 字符串(原设计)
      - {'status':.., 'content':..} 字典对象(模型常直接传结构化对象)
      - 上述字典的 JSON 字符串
    单源真相:WriteTodoTool 落库与 agent 结构闸门共用,避免两处解析跑偏。"""
    items = []
    for line in raw or []:
        # 1) 直接是字典对象
        if isinstance(line, dict):
            it = _item_from_dict(line)
            if it:
                items.append(it)
            continue
        text = str(line).strip()
        # 2) 看着像 JSON 字典 → 试解析(模型偶尔把对象当字符串塞进来)
        if text.startswith("{"):
            try:
                d = json.loads(text)
                if isinstance(d, dict):
                    it = _item_from_dict(d)
                    if it:
                        items.append(it)
                    continue
            except Exception:
                pass
        # 3) 'status|内容' / 纯内容
        s, sep, c = text.partition("|")
        if sep and c.strip():
            items.append({"content": c.strip(), "status": _norm_status(s)})
        elif s.strip():
            items.append({"content": s.strip(), "status": "pending"})
    return items


class TodoStore:
    """进程内计划状态:每项 {content, status}。整体覆盖式更新。"""

    def __init__(self):
        self.items = []

    def set(self, items):
        norm = []
        seen_in_progress = False
        for it in items:
            c = str(it.get("content", "")).strip()
            if not c:
                continue
            status = it.get("status", "pending")
            # 单一"当前步":一次出现多个 in_progress 只认第一个,其余降回 pending,
            # 保证高亮的"正在执行步"唯一。
            if status == "in_progress":
                if seen_in_progress:
                    status = "pending"
                else:
                    seen_in_progress = True
            norm.append({"content": c, "status": status})
        self.items = norm

    def render(self) -> str:
        if not self.items:
            return ""
        lines = [TODO_HEADING]
        lines += [f"{_MARK.get(i['status'], '[ ]')} {i['content']}" for i in self.items]
        return "\n".join(lines)


class WriteTodoTool(Tool):
    """模型用它记录/更新分步计划。每项编码 'status|content'(规避 array-items 限制)。"""

    verify_exempt = True  # 列计划是内部记账,非代码产物 → 不触发事后 verify

    def __init__(self, store: TodoStore):
        super().__init__(
            "write_todo",
            "复杂多步任务:记录分步计划并更新进度(简单任务别用)。"
            "todos 是字符串数组,每项 'status|内容',"
            "status ∈ pending/in_progress/completed,如 'in_progress|读取配置'。")
        self.store = store
        self.side_effect_free = False

    def run(self, parameters):
        self.store.set(parse_todo_lines(parameters.get("todos")))
        return self.store.render() or "(清单已清空)"

    def get_parameters(self):
        return [ToolParameter(
            name="todos", type="array",
            description="完整清单,每项 'status|内容'(status: pending/in_progress/completed)")]


def todo_system_message(store: Optional[TodoStore]) -> Optional[dict]:
    """渲染当前清单为 system 注入消息;空清单返回 None(短任务零注入)。"""
    if store is None:
        return None
    text = store.render()
    if not text:
        return None
    return {"role": "system", "content": text + (
        "\n规则:每做完一步,**立刻**再调一次 write_todo,把那一步的 status 改成 "
        "completed,并把下一步置为 in_progress —— 然后才继续。不要攒到最后才一次性更新;"
        "用户要靠这个看到实时打勾。")}
