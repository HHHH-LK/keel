"""L2 滚动摘要：被 L1 驱逐的消息压缩成一段摘要塞回 context。"""
from typing import Callable, List, Optional

from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.item import MemoryItem


# summarizer 签名：(待压缩消息列表, 原有摘要文本) -> 新摘要文本
Summarizer = Callable[[List[MemoryItem], str], str]

# reconciler 签名：(当前摘要文本, 校正信号) -> 校正后的摘要文本
# 用于 L2 分层托管:LLM 看懂"状态变了"后主动改写摘要,机械层只做护栏。
SummaryReconciler = Callable[[str, str], str]


class LLMReconciler:
    """把 MyLLM 包成 SummaryReconciler —— L2 的语义编辑器。

    与 LLMSummarizer(压缩新内容)不同,reconciler 负责"纠错":
    拿当前摘要 + 一个变更/反思信号,让模型重写摘要,
    区分"稳定事实(累加保留)"和"当前状态(以新为准)"。
    """

    DEFAULT_PROMPT = (
        "你在维护一份『对话梗概』—— 记录我们聊过什么、进展到哪,而不是用户档案。\n"
        "请根据下面的『校正信号』更新『当前摘要』。\n"
        "规则:\n"
        "1. 这是叙事梗概,不要在此记录离散事实(姓名/过敏/偏好/身份等)——"
        "那是 L0 的职责,会单独维护;此处只保留对话脉络。\n"
        "2. 当前状态/进展如与信号矛盾,以信号中的新信息为准。\n"
        "3. 只在有明确矛盾证据时才修改或删除既有内容;"
        "某条内容仅仅'最近没被提到'不等于过期,不得据此删除。\n"
        "4. 长度控制在 {max_tokens} token 内,去重,不要编造信号里没有的信息。\n\n"
        "当前摘要:\n{current}\n\n校正信号:\n{signal}\n\n"
        "请直接输出更新后的摘要正文,不要前缀。"
    )

    def __init__(self, llm, *, max_tokens: int = 400, prompt: Optional[str] = None):
        self.llm = llm
        self.max_tokens = max_tokens
        self.prompt = prompt or self.DEFAULT_PROMPT

    def __call__(self, current: str, signal: str) -> str:
        prompt = self.prompt.format(
            current=current, signal=signal, max_tokens=self.max_tokens
        )
        result = self.llm.invoke([{"role": "user", "content": prompt}])
        return (result or "").strip() or current


def _default_summarizer(batch: List[MemoryItem], previous: str) -> str:
    """没注入 LLM 时的兜底：直接把内容拼起来。"""
    bullets = [f"- [{it.role}] {it.content}" for it in batch]
    fresh = "\n".join(bullets)
    if previous:
        return f"{previous}\n\n[新增片段]\n{fresh}"
    return f"[早期对话片段]\n{fresh}"


class LLMSummarizer:
    """把任意 MyLLM 包成 Summarizer 调用。

    用法：
        manager = MemoryManager(
            config,
            summarizer=LLMSummarizer(llm, max_tokens=400),
        )

    工作模式：
    - 首次：把 batch 给 LLM 让它写一段精炼总结
    - 后续：把"原摘要 + 新增 batch"一起给 LLM，要求保留旧要点、合并新信息
    """

    DEFAULT_INITIAL_PROMPT = (
        "请把下面这段对话片段总结成不超过 {max_tokens} 个 token 的精炼摘要，"
        "保留关键事实、用户偏好、决定、待办；丢弃寒暄和重复内容。\n\n"
        "对话片段：\n{batch}\n\n"
        "请直接输出摘要正文，不要前缀。"
    )

    DEFAULT_MERGE_PROMPT = (
        "你正在维护一份滚动的对话摘要。请把"
        "「新增片段」融合进「现有摘要」，保留两边的关键事实，整体长度控制在 {max_tokens} token 内，"
        "去除重复、矛盾时以更新的为准。\n\n"
        "现有摘要：\n{previous}\n\n"
        "新增片段：\n{batch}\n\n"
        "请直接输出新摘要正文，不要前缀。"
    )

    def __init__(
        self,
        llm,
        *,
        max_tokens: int = 400,
        initial_prompt: Optional[str] = None,
        merge_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.initial_prompt = initial_prompt or self.DEFAULT_INITIAL_PROMPT
        self.merge_prompt = merge_prompt or self.DEFAULT_MERGE_PROMPT

    def __call__(self, batch: List[MemoryItem], previous: str) -> str:
        batch_text = "\n".join(f"[{it.role}] {it.content}" for it in batch)

        if previous:
            prompt = self.merge_prompt.format(
                previous=previous,
                batch=batch_text,
                max_tokens=self.max_tokens,
            )
        else:
            prompt = self.initial_prompt.format(
                batch=batch_text,
                max_tokens=self.max_tokens,
            )

        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
        except Exception as exc:
            # LLM 调用失败时退到拼接兜底，不能让摘要错误影响对话
            print(f"⚠️ LLMSummarizer 调用失败，降级到拼接: {exc}")
            return _default_summarizer(batch, previous)

        return (result or "").strip() or _default_summarizer(batch, previous)


class SummaryMemory(MemoryTier):
    """L2 —— 摘要层。

    内部状态：
    - _summary_item：滚动摘要（一条 MemoryItem，role=system）
    - _buffer：尚未触发压缩的待压消息
    - flush_threshold：缓冲达到这个量就调 summarizer
    """

    name = "L2"

    def __init__(
        self,
        flush_threshold: int = 4,
        summarizer: Optional[Summarizer] = None,
        max_tokens: int = 1000,
        reconciler: Optional[SummaryReconciler] = None,
    ):
        self.flush_threshold = flush_threshold
        self.summarizer = summarizer or _default_summarizer
        self.max_tokens = max_tokens
        self.reconciler = reconciler

        self._summary_item: Optional[MemoryItem] = None
        self._buffer: List[MemoryItem] = []

    def _cap_text(self, text: str) -> str:
        """按 max_tokens(3 char/token 估算)截断尾部。"""
        max_chars = self.max_tokens * 3
        return text[-max_chars:] if len(text) > max_chars else text

    def _set_summary_text(self, text: str) -> MemoryItem:
        """直接设置/替换摘要文本(reconcile 与测试用)。"""
        text = self._cap_text(text)
        if self._summary_item is None:
            self._summary_item = MemoryItem(
                content=text, role="system", metadata={"kind": "summary"},
            )
        else:
            self._summary_item.content = text
            self._summary_item.token_estimate = max(1, len(text) // 3)
        return self._summary_item

    def reconcile(self, signal: str) -> Optional[MemoryItem]:
        """L2 分层托管:让 reconciler 根据信号校正摘要,机械层兜底。

        - 没有摘要 → 无事可做,返回 None。
        - 没有 reconciler → 不动(降级),返回原摘要。
        - reconciler 抛异常 / 吐空 → 保留旧摘要,不污染。
        - 输出超 max_tokens → 截断。
        """
        if self._summary_item is None:
            return None
        if self.reconciler is None:
            return self._summary_item
        try:
            new_text = self.reconciler(self._summary_item.content, signal)
        except Exception as exc:
            print(f"⚠️ L2 reconcile 失败,保留旧摘要: {exc}")
            return self._summary_item
        new_text = (new_text or "").strip()
        if not new_text:
            return self._summary_item
        return self._set_summary_text(new_text)

    # ── MemoryTier 接口 ─────────────────────────────────────
    def add(self, item: MemoryItem) -> None:
        """L2 不是被 LLM 直接写入的，这里把"被驱逐的消息"作为输入。"""
        self._buffer.append(item)
        if len(self._buffer) >= self.flush_threshold:
            self.flush()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        if self._summary_item and self._summary_item.id == item_id:
            return self._summary_item
        for it in self._buffer:
            if it.id == item_id:
                return it
        return None

    def items(self) -> List[MemoryItem]:
        out: List[MemoryItem] = []
        if self._summary_item:
            out.append(self._summary_item)
        return out

    # ── 摘要专属 API ────────────────────────────────────────
    def flush(self) -> Optional[MemoryItem]:
        """把缓冲压缩为/合并到摘要。返回当前摘要项。"""
        if not self._buffer:
            return self._summary_item

        previous = self._summary_item.content if self._summary_item else ""
        new_text = self.summarizer(self._buffer, previous)

        # 超过 max_tokens（按 3 char/token 估算）则截断尾部
        max_chars = self.max_tokens * 3
        if len(new_text) > max_chars:
            new_text = new_text[-max_chars:]

        if self._summary_item is None:
            self._summary_item = MemoryItem(
                content=new_text,
                role="system",
                metadata={"kind": "summary"},
            )
        else:
            self._summary_item.content = new_text
            self._summary_item.token_estimate = max(1, len(new_text) // 3)

        self._buffer.clear()
        return self._summary_item

    def current_summary(self) -> Optional[MemoryItem]:
        return self._summary_item

    def clear(self) -> None:
        self._summary_item = None
        self._buffer.clear()
