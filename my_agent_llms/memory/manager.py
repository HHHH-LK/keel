"""MemoryManager —— 统一调度各层，Agent 持有一个。

支持自由配置：
- embedding：传 EmbeddingProvider 实例或裸 callable
- 冷存储：JSONL / SQLite / 关闭
- 向量库：内存 / SQLite（重启可恢复）
- tick 调度：同步 / 异步 / 关 + 节流
"""
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from my_agent_llms.core.message import MessageRole
from my_agent_llms.memory.backends.base import ColdBackend, VectorBackend
from my_agent_llms.memory.backends.inmemory import InMemoryVectorBackend
from my_agent_llms.memory.backends.jsonl import JSONLColdBackend
from my_agent_llms.memory.backends.sqlite import (
    SQLiteColdBackend,
    SQLiteVectorBackend,
)
from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.cold import ColdStorage
from my_agent_llms.memory.config import MemoryConfig
from my_agent_llms.memory.conflict import (
    ConflictDetector,
    LLMConflictDetector,
    SimilarityConflictDetector,
)
from my_agent_llms.memory.embeddings import (
    EmbeddingProvider,
    coerce_embedding,
)
from my_agent_llms.memory.item import MemoryItem
from my_agent_llms.memory.playbook import (
    L0Lifecycle,
    L0Source,
    L0Type,
    PlaybookCard,
    PlaybookStore,
    classify_content_type,
)
from my_agent_llms.memory.seed_score import (
    AUTO_PIN_THRESHOLD,
    boost_with_kg_feedback,
    evaluate_prior_score,
    should_auto_pin,
)
from my_agent_llms.memory.recall_buffer import RecallBuffer
from my_agent_llms.memory.semantic import SemanticIndex
from my_agent_llms.memory.summary import (
    LLMReconciler,
    Summarizer,
    SummaryMemory,
    SummaryReconciler,
)
from my_agent_llms.memory.working import WorkingMemory


class MemoryManager:
    """记忆系统对外统一入口。

    构造时可注入：
    - embedding: EmbeddingProvider 或 callable（None 表示走 TF-IDF）
    - cold_backend / vector_backend: 直接传 backend 实例，覆盖 config 中的字符串配置
    - summarizer: L2 摘要器
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        embedding: Union[None, EmbeddingProvider, Callable[[str], Sequence[float]]] = None,
        cold_backend: Optional[ColdBackend] = None,
        vector_backend: Optional[VectorBackend] = None,
        summarizer: Optional[Summarizer] = None,
        summary_flush_threshold: int = 4,
        reconciler: Optional[SummaryReconciler] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        llm=None,
    ):
        self.config = config or MemoryConfig()
        self.embedding = coerce_embedding(embedding)

        self.working = WorkingMemory(self.config)
        self.cold = ColdStorage(cold_backend or self._build_cold_backend())
        self.semantic = SemanticIndex(
            vector_backend or self._build_vector_backend()
        )
        # L2 reconciler: 显式优先,否则有 llm 时自动构造 LLMReconciler
        if reconciler is None and llm is not None:
            reconciler = LLMReconciler(llm, max_tokens=self.config.l2_max_tokens)
        self.summary = SummaryMemory(
            flush_threshold=summary_flush_threshold,
            summarizer=summarizer,
            max_tokens=self.config.l2_max_tokens,
            reconciler=reconciler,
        )
        # L0 跨会话核心记忆
        self.playbook = PlaybookStore(self.config.playbook_path())
        # L3 检索缓冲(纯内存,不持久化)
        self.recall_buffer = RecallBuffer(self.config)
        # 冲突检测器: 显式传入优先,否则按 config.conflict_strength 自动构造
        self.conflict_detector = (
            conflict_detector
            if conflict_detector is not None
            else self._build_conflict_detector(
                self.config.conflict_strength,
                llm=llm,
            )
        )

        # KG 冷回路:仅当冲突检测器是知识图谱型(extreme)时才建。
        # 先传 llm=None → 只跑确定性 pending GC;语义冲突等有可用 llm 再开。
        self._kg_reconciler = self._build_kg_reconciler()

        self.tiers: Dict[str, MemoryTier] = {
            self.working.name: self.working,
            self.summary.name: self.summary,
            self.cold.name: self.cold,
            self.semantic.name: self.semantic,
        }

        # ── tick 异步化所需状态 ────────────────────────────
        # 用 RLock(可重入):tick 内部可能调 _cascade_evict_from_l1,
        # 后者如果再次需要锁不会自死锁
        self._lock = threading.RLock()
        self._turn_counter = 0
        self._last_reflect_turn = 0       # 上次 L2 反思发生的轮次
        self._last_reconcile_turn = 0     # 上次 KG 冷回路发生的轮次
        self._pending_tick: Optional[Future] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.config.tick_mode == "async":
            self._executor = ThreadPoolExecutor(
                max_workers=1,                          # 串行执行,不要并发 tick
                thread_name_prefix="memory-tick",
            )

    # ── 后端工厂（按 config 字符串选择） ────────────────────
    def _build_cold_backend(self) -> Optional[ColdBackend]:
        path = self.config.cold_path()
        if path is None:
            return None
        backend_type = self.config.cold_backend
        if backend_type == "jsonl":
            return JSONLColdBackend(path)
        if backend_type == "sqlite":
            return SQLiteColdBackend(path)
        return None

    def _build_vector_backend(self) -> VectorBackend:
        backend_type = self.config.vector_backend
        if backend_type == "sqlite":
            path = self.config.vector_path()
            if path is None:
                raise ValueError("vector_backend='sqlite' 需要同时设置 storage_dir")
            return SQLiteVectorBackend(path, embedder=self.embedding)
        return InMemoryVectorBackend(embedder=self.embedding)

    def _build_conflict_detector(
        self,
        strength: str,
        llm=None,
    ) -> Optional[ConflictDetector]:
        """根据 conflict_strength 自动构造检测器。
        优雅降级: extreme 需要 llm,没有就降到 accurate;accurate 没 llm 就降到 fast。
        """
        threshold = self.config.conflict_threshold

        if strength == "off":
            return None

        if strength == "fast":
            return SimilarityConflictDetector(threshold=threshold)

        if strength == "accurate":
            if llm is None:
                print("⚠️ conflict_strength='accurate' 需要传 llm,降级到 fast")
                return SimilarityConflictDetector(threshold=threshold)
            return LLMConflictDetector(llm, threshold=threshold)

        if strength == "extreme":
            if llm is None:
                print("⚠️ conflict_strength='extreme' 需要传 llm,降级到 fast")
                return SimilarityConflictDetector(threshold=threshold)
            # 延迟导入避免循环
            from my_agent_llms.memory.kg import KGStore, KnowledgeGraphConflictDetector
            kg_path = None
            if self.config.storage_dir is not None:
                kg_path = self.config.storage_dir / "kg.db"
            store = KGStore(kg_path)
            # 把 embedder 传进去,让 query_facts 的语义路径在生产路径上开启
            return KnowledgeGraphConflictDetector(llm, store, embedder=self.embedding)

        return None

    def _build_kg_reconciler(self):
        """仅当冲突检测器是 KG 型时,基于它的 store 建冷回路 reconciler。

        先传 llm=None → 只跑确定性 pending GC(语义冲突那段需可用 llm,后续再开)。
        """
        from my_agent_llms.memory.kg import KnowledgeGraphConflictDetector
        if not isinstance(self.conflict_detector, KnowledgeGraphConflictDetector):
            return None
        from my_agent_llms.memory.kg_reconcile import KGReconciler
        return KGReconciler(self.conflict_detector.store, llm=None)

    # ── L0 公开 API ────────────────────────────────────────
    def remember(
        self,
        content: str,
        *,
        type: Optional[L0Type] = None,
        source: L0Source = L0Source.USER_EXPLICIT,
        source_ref: Optional[str] = None,
        confidence: float = 1.0,
    ) -> PlaybookCard:
        """显式添加一张 L0 卡片(/remember 命令的底层)。

        type 不传则启发式分类。user_explicit 来源默认 confidence=1.0。
        """
        card = PlaybookCard(
            content=content,
            type=type or classify_content_type(content),
            source=source,
            source_ref=source_ref,
            confidence=confidence,
            user_pinned=(source == L0Source.USER_EXPLICIT),
        )
        self.playbook.add(card)
        return card

    def forget(self, card_id: str) -> bool:
        """显式忘记一张 L0 卡片(/forget 命令)。返回是否成功。"""
        card = self.playbook.get(card_id)
        if card is None:
            return False
        card.forget()
        self.playbook.update(card)
        return True

    def pin_card(self, card_id: str) -> bool:
        """显式锁定一张 L0 卡片,永不衰减(/pin 命令)。"""
        card = self.playbook.get(card_id)
        if card is None:
            return False
        card.pin()
        self.playbook.update(card)
        return True

    def list_l0(self, lifecycle: L0Lifecycle = L0Lifecycle.ACTIVE) -> List[PlaybookCard]:
        """列出指定生命周期的 L0 卡片(/l0 命令用)。"""
        if lifecycle == L0Lifecycle.ACTIVE:
            return self.playbook.all_active()
        return self.playbook.all_with_lifecycle(lifecycle)

    # ── 写入路径 ────────────────────────────────────────────
    def write(
        self,
        content: str,
        role: MessageRole = "user",
        *,
        pinned: bool = False,
        metadata: Optional[Dict] = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content,
            role=role,
            pinned=pinned,
            metadata=metadata or {},
        )
        # ★ 种子分: 启发式打分,关键消息一进 L1 就被 pin,不依赖 LLM
        item.prior_score = evaluate_prior_score(content, role)
        if not item.pinned and should_auto_pin(item.prior_score):
            item.pinned = True

        # 加锁:tick 可能正在后台改 working/semantic 的 pinned 字段或追加项
        with self._lock:
            self.working.add(item)
            self.semantic.add(item)
            self._cascade_evict_from_l1()
            # 检测冲突 —— 用新写入项去查找它取代了哪些旧项
            # (副作用: KG 反哺可能提升 item.prior_score,L0 旧卡可能被 negate)
            if self.conflict_detector is not None:
                self._apply_conflict_detection(item)
            # ★ L0 晋升: 高分用户消息晋升为 L0 卡片(跨会话保留)
            self._maybe_promote_to_l0(item)
        return item

    def _maybe_promote_to_l0(self, item: MemoryItem) -> None:
        """高 prior_score 的用户消息晋升为 L0 卡片。

        触发条件:
        - prior_score >= AUTO_PIN_THRESHOLD
        - role == 'user' (assistant 自述不晋升)
        - 内容未在 L0 中重复(简单去重)
        """
        if item.role != "user":
            return
        if item.prior_score < AUTO_PIN_THRESHOLD:
            return

        # 简单去重: 同内容已存在则跳过
        for existing in self.playbook.all_active():
            if existing.content == item.content:
                return

        card = PlaybookCard(
            content=item.content,
            type=classify_content_type(item.content),
            source=L0Source.SEED_PROMOTED,
            source_ref=item.id,
            confidence=min(1.0, 0.7 + 0.3 * item.prior_score),
        )
        self.playbook.add(card)

    def _maybe_graduate_to_l0(self) -> List[str]:
        """实绩毕业: L1 里 pinned 且反复被访问的用户消息晋升为 L0 卡片。

        与 _maybe_promote_to_l0 的区别 ——
        - 写入晋升 = "预测": 靠关键词种子分,一进 L1 就判定重要。
        - 实绩毕业 = "证据": 靠 pinned + access_count,久经考验才毕业。
          覆盖那些没命中关键词、但被反复召回证明了价值的项。

        触发条件:
        - role == 'user'(assistant 自述不毕业)
        - item.pinned 且 access_count >= l0_graduate_min_hits
        - 该 item 尚未产生过任何 L0 卡(source_ref 去重,避免重复毕业)
        - 内容未在活跃 L0 中重复
        """
        graduated: List[str] = []
        threshold = self.config.l0_graduate_min_hits
        # 候选通常只有少数几条 pinned 项,但 all_active 每次查库,先过滤再查
        candidates = [
            item for item in self.working.items()
            if item.role == "user"
            and item.is_active
            and item.pinned
            and item.access_count >= threshold
        ]
        if not candidates:
            return graduated
        active_contents = {c.content for c in self.playbook.all_active()}
        for item in candidates:
            # source_ref 去重: 查所有生命周期(含 archived/forgotten)。
            # 故意不只查 active —— forgotten 卡也要挡住重新毕业,
            # 否则用户 /forget 后,仍在 L1 的高热项会被一次次复活,违背用户意图。
            # (被 supersede 的项已被上面的 is_active 过滤,不会走到这。)
            if self.playbook.find_by_source_ref(item.id):
                continue
            # 内容去重: 不同 item 但同文本已在 L0
            if item.content in active_contents:
                continue
            card = PlaybookCard(
                content=item.content,
                type=classify_content_type(item.content),
                source=L0Source.L1_GRADUATED,
                source_ref=item.id,
                confidence=0.8,
            )
            self.playbook.add(card)
            active_contents.add(item.content)
            graduated.append(item.id)
        return graduated

    def _apply_conflict_detection(self, new_item: MemoryItem) -> None:
        """对新写入项做冲突检测,标记被取代的旧项。

        副作用 —— KG 反哺:
        如果 detector 真的找到了 supersede 候选,说明 new_item 含有"有事实价值"
        的信息(不是普通闲聊),累加 KG_FEEDBACK_BOOST 到 prior_score。
        加分后若达到 auto_pin 阈值,立即 pin。
        """
        try:
            superseded_ids = self.conflict_detector.find_superseded(new_item, self)
        except Exception as exc:
            print(f"⚠️ 冲突检测失败,跳过: {exc}")
            return

        # ★ KG 反哺: detector 找到 supersede 候选 = 信息密度高 → 加分
        if superseded_ids:
            new_item.prior_score = boost_with_kg_feedback(new_item.prior_score)
            if not new_item.pinned and should_auto_pin(new_item.prior_score):
                new_item.pinned = True

        superseded_contents: List[str] = []
        for old_id in superseded_ids:
            old = self.semantic.get(old_id) or self.working.get(old_id)
            if old is None or not old.is_active:
                continue
            old.superseded_by = new_item.id
            new_item.supersedes.append(old_id)
            superseded_contents.append(old.content)
            # ★ L0 反哺: 如果该旧项有对应的 L0 卡片,按 type 不同幅度 negate
            self._negate_l0_cards_for(old_id)

        # ★ L2 冲突扳机: 现实变了 → 立即校正摘要(分层托管)
        if superseded_contents:
            self._reconcile_summary_on_conflict(new_item, superseded_contents)

    def _reconcile_summary_on_conflict(
        self, new_item: MemoryItem, old_contents: List[str]
    ) -> None:
        """supersede 发生时,联动校正 L2 摘要 —— 把"旧→新"的变更交给 reconciler。"""
        if self.summary.current_summary() is None:
            return
        old_joined = " | ".join(old_contents)
        signal = (
            f"状态变更: 旧信息「{old_joined}」已被新信息「{new_item.content}」取代。"
            "请更新摘要 —— 稳定事实保留,当前状态以新信息为准。"
        )
        self.summary.reconcile(signal)

    def _reflect_on_summary(self) -> None:
        """L2 定期反思: 对照最近对话主动复查摘要,修正过期/矛盾内容。"""
        if self.summary.current_summary() is None:
            return
        recent = [it.content for it in self.working.items()[-5:] if it.is_active]
        if not recent:
            return
        signal = (
            "定期复查: 对照下面的最近对话,修正摘要中已过期或自相矛盾的内容,"
            "稳定事实保留。最近对话:\n" + "\n".join(f"- {c}" for c in recent)
        )
        self.summary.reconcile(signal)

    def _negate_l0_cards_for(self, item_id: str) -> None:
        """某条 memory item 被 supersede 时,联动 negate 它产生的 L0 卡片。

        卡片按 type 不同幅度降 confidence:
        - state: -0.6 (项目结束/学习内容变,立即撤下)
        - preference: -0.4 (偏好变化)
        - identity: -0.3 (身份变化通常缓慢)
        - hard_constraint: -0.15 (健康类需用户显式 /forget)
        撤下后 KG 标记 archived(我们不直接动 KG,KG 的 supersede 链已经独立维护)。
        """
        for card in self.playbook.find_by_source_ref(item_id):
            if card.lifecycle != L0Lifecycle.ACTIVE:
                continue
            card.negate()
            if card.should_archive():
                card.archive()
            self.playbook.update(card)

    def _cascade_evict_from_l1(self) -> None:
        evicted = self.working.evict()
        for it in evicted:
            self.cold.add(it)
            self.summary.add(it)

    # ── 读取路径 ────────────────────────────────────────────
    def assemble_context(
        self,
        system_prompt: Optional[str] = None,
        *,
        query: Optional[str] = None,
        kg_query: Optional[str] = None,
        passive_recall_k: int = 5,
    ) -> List[Dict[str, str]]:
        """组装喂给 LLM 的 messages。

        注入顺序(按权威性递减):
        1. system_prompt          — 你的人设
        2. L0 核心信息             — 跨会话核心(query-aware 加权)
        3. L2 全局摘要             — 长期画像
        4. KG facts (默认注入)     — 结构化事实
        5. L0 背景信息             — 不直接相关的 L0 卡片
        6. 被动 recall 命中         — query 相关的历史片段
        7. L1 最近对话             — 现场上下文

        query: 当前用户提问,用于 L0 加权 + 被动 recall。
               为 None 时跳过 L0 加权和被动 recall(纯 L0 全量 + 不召回)。
        kg_query: 向后兼容老参数,等同于 query。
        """
        messages: List[Dict[str, str]] = []
        effective_query = query or kg_query

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # ── L0 注入(query-aware 加权,分核心段/背景段) ──
        l0_core_text, l0_bg_text = self._compose_l0_segments(effective_query)
        if l0_core_text:
            messages.append({
                "role": "system",
                "content": (
                    "## 关于用户的核心信息(与当前问题相关 + 硬约束)\n"
                    "回答时优先依据此段。\n\n" + l0_core_text
                ),
            })

        # ── L2 全局摘要(背景叙事,最低权威) ──
        summary = self.summary.current_summary()
        if summary is not None:
            messages.append({
                "role": "system",
                "content": (
                    "## 对话梗概(背景叙事)\n"
                    "仅用于理解上下文脉络。关于用户的具体事实"
                    "(身份/偏好/约束/当前状态)以上文 L0 核心信息与 KG 事实为准;"
                    "本段与之冲突时一律以 L0/KG 为准。\n\n"
                    + summary.content
                ),
            })

        # ── KG facts (默认注入) ──
        if effective_query:
            facts = self.recall_facts(effective_query)
            if facts:
                fact_text = "\n".join(f"- {f}" for f in facts)
                messages.append({
                    "role": "system",
                    "content": f"## 已知事实(来自知识图谱)\n{fact_text}",
                })

        # ── L0 背景信息 ──
        if l0_bg_text:
            messages.append({
                "role": "system",
                "content": (
                    "## 关于用户的背景信息(可能不直接相关)\n"
                    "仅作背景理解,不要直接当作当前问题的依据。\n\n" + l0_bg_text
                ),
            })

        # ── 被动 recall: 系统主动从 L5 召回相关原文 ──
        if effective_query and passive_recall_k > 0:
            recall_text = self._compose_passive_recall(effective_query, passive_recall_k)
            if recall_text:
                messages.append({
                    "role": "system",
                    "content": f"## 与当前问题相关的历史片段\n{recall_text}",
                })

        # ── L1 最近对话原文 ──
        for it in self.working.items():
            if not it.is_active:
                continue  # 跳过已被取代的旧记忆
            messages.append(it.to_message_dict())
        return messages

    def _compose_l0_segments(
        self, query: Optional[str]
    ) -> Tuple[str, str]:
        """生成 L0 注入的核心段和背景段文本。

        query-aware 加权规则:
        - hard_constraint 永远进核心段(无视相关性)
        - effective_confidence = static_confidence + 0.3 * relevance
        - effective ≥ 0.5 → 核心段
        - 0.2 ≤ effective < 0.5 → 背景段
        - effective < 0.2 且非硬约束 → 省略

        无 query 时退化为:hard 进核心,其他进背景(按 confidence)。
        """
        active_cards = self.playbook.all_active()
        if not active_cards:
            return "", ""

        # 算相关性
        relevance_map: Dict[str, float] = {}
        if query and active_cards:
            try:
                hits = self.semantic.search(query, k=len(active_cards) * 2)
                # 这里 semantic.search 命中的是 MemoryItem 不是 PlaybookCard
                # 我们不直接用相似度,而是简单的"content 子串匹配"作 query-aware 近似
                # (避免给 L0 卡片独立 embedding 的复杂度,这是 v2 优化)
            except Exception:
                pass
            for card in active_cards:
                relevance_map[card.id] = self._cheap_relevance(card.content, query)
        else:
            for card in active_cards:
                relevance_map[card.id] = 0.0

        core_lines: List[str] = []
        bg_lines: List[str] = []
        for card in active_cards:
            relevance = relevance_map.get(card.id, 0.0)
            effective = card.confidence + 0.3 * relevance

            # hard_constraint 永远进核心
            if card.is_hard_constraint():
                core_lines.append(self._format_l0_line(card))
                card.refresh()
                self.playbook.update(card)
                continue

            if effective >= 0.5:
                core_lines.append(self._format_l0_line(card))
                card.refresh()
                self.playbook.update(card)
            elif effective >= 0.2:
                bg_lines.append(self._format_l0_line(card))
                # 背景段也算被引用,但 refresh 幅度小
                card.refresh(boost=0.005)
                self.playbook.update(card)
            # else: 省略,不注入

        return "\n".join(core_lines), "\n".join(bg_lines)

    @staticmethod
    def _format_l0_line(card: PlaybookCard) -> str:
        """格式化一行 L0 内容,带 type tag 让 LLM 知道权威性。"""
        type_tag = {
            L0Type.HARD_CONSTRAINT: "硬约束",
            L0Type.IDENTITY: "身份",
            L0Type.PREFERENCE: "偏好",
            L0Type.STATE: "状态",
        }.get(card.type, "")
        prefix = f"- [{type_tag}] " if type_tag else "- "
        return f"{prefix}{card.content}"

    @staticmethod
    def _cheap_relevance(card_content: str, query: str) -> float:
        """轻量相关性: 字符 bigram Jaccard。

        避免给 L0 卡片单独算 embedding 的复杂度。准确度 < 向量但足够 query-aware。
        """
        def bigrams(s: str) -> set:
            s = s.strip()
            if len(s) < 2:
                return {s} if s else set()
            return {s[i:i + 2] for i in range(len(s) - 1)}

        a = bigrams(card_content)
        b = bigrams(query)
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _compose_passive_recall(self, query: str, k: int) -> str:
        """被动 recall:系统主动调 L5,命中登记进 L3 缓冲,再从 L3 注入。

        与旧版区别:命中不再"用完即弃",而是写入 L3 台账累计热度。
        反复命中的项由 tick 的 _maybe_promote_from_l3 晋升进 L1。
        """
        # recall() 走向量检索(可能 I/O),放锁外;只有 L3 缓冲的读写需要加锁,
        # 因为异步 tick 的 _maybe_promote_from_l3 会并发改同一个 buffer。
        try:
            hits = self.recall(query, k=k)
        except Exception as exc:
            print(f"⚠️ 被动 recall 失败,跳过: {exc}")
            hits = []
        with self._lock:
            self.recall_buffer.evict_expired(self._turn_counter)
            for item, score in hits:
                # 高分且未 pinned → 立即 pin(用户 query 相关 = 这条该被保护)
                if not item.pinned and score >= 0.6:
                    item.pinned = True
                self.recall_buffer.record_hit(item.id, score, self._turn_counter)
            return self._compose_l3_injection()

    def _compose_l3_injection(self) -> str:
        """把当前 L3 台账渲染成注入文本(原文从 L5 取)。

        跳过已在 L1 的项 —— 它们已由 L1 原文段注入,避免重复
        (例如被 importance 召回路径加回 L1 但还没从 L3 清掉的项)。
        """
        l1_ids = {it.id for it in self.working.items()}
        lines: List[str] = []
        for entry in self.recall_buffer.entries():
            if entry.item_id in l1_ids:
                continue
            item = self.semantic.get(entry.item_id)
            if item is None or not item.is_active:
                continue
            preview = item.content[:120]
            lines.append(f"- [{item.role}] {preview}")
        return "\n".join(lines)

    # ── 检索 ────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        exclude_working: bool = True,
        include_superseded: bool = False,
    ) -> List[Tuple[MemoryItem, float]]:
        """语义检索。命中的项自动 touch(热度+1)。

        - exclude_working=True 排除已在 L1 的项(避免 context 重复)
        - include_superseded=False 排除已被新记忆取代的旧版本(默认)
        """
        # 多取候选,过滤后仍能保证返回 k 条
        results = self.semantic.search(query, k=max(k * 3, k))
        if not include_superseded:
            results = [(it, s) for it, s in results if it.is_active]
        if exclude_working:
            l1_ids = {it.id for it in self.working.items()}
            results = [(it, s) for it, s in results if it.id not in l1_ids]
        results = results[:k]
        for item, _ in results:
            item.touch()
        return results

    def restore_from_cold(self, n: int = 10) -> int:
        """从 L4 加载最近 n 条历史原文回 L1,用于重启后恢复对话现场。

        - 跳过已存在于 L1 的 id (避免重复)
        - 跳过已被取代的旧版本 (is_active=False)
        - 不重建 L5: 默认 vector_backend='sqlite' 已自带持久化
        - 不主动触发 evict: 把控场交给下一次 write 的级联逻辑,
          避免在恢复瞬间把刚加回来的项又写一遍 L4/L2

        返回实际加载的条数。
        """
        if self.cold.backend is None:
            return 0
        with self._lock:
            all_items = list(self.cold.items())
            if not all_items:
                return 0
            all_items.sort(key=lambda x: x.created_at)
            existing_ids = {it.id for it in self.working.items()}
            loaded = 0
            for item in all_items[-n:]:
                if item.id in existing_ids or not item.is_active:
                    continue
                self.working.add(item)
                loaded += 1
            return loaded

    def export_kg_graph(self, format: str = "mermaid", include_inactive: bool = True) -> str:
        """导出 KG 为 Mermaid 或 DOT 字符串。

        format: "mermaid" | "dot"
        其他强度下返回友好提示。
        """
        from my_agent_llms.memory.kg import KnowledgeGraphConflictDetector

        if not isinstance(self.conflict_detector, KnowledgeGraphConflictDetector):
            return f"# 当前 conflict_strength 不是 'extreme',无 KG 可导出"

        store = self.conflict_detector.store
        if format == "mermaid":
            return store.to_mermaid(include_inactive=include_inactive)
        if format == "dot":
            return store.to_dot(include_inactive=include_inactive)
        raise ValueError(f"未知格式: {format!r},支持 'mermaid' / 'dot'")

    def recall_facts(self, query: str, max_facts: int = 8) -> List[str]:
        """从 KG 拿当前活跃事实(extreme 强度时才有用)。

        返回自然语言的事实串,用于把"用户当前状态"喂给 LLM。
        其他强度下返回空列表。
        """
        from my_agent_llms.memory.kg import KnowledgeGraphConflictDetector

        if not isinstance(self.conflict_detector, KnowledgeGraphConflictDetector):
            return []
        return self.conflict_detector.query_facts(query, max_facts=max_facts)

    def history_chain(self, item_id: str) -> List[MemoryItem]:
        """追溯一条记忆被取代过的完整链条。

        从 item_id 出发,沿 supersedes 字段递归找回所有前身,
        最新的在前。useful for debugging or showing user "你的偏好演变"。
        """
        chain: List[MemoryItem] = []
        cur = self.semantic.get(item_id)
        while cur is not None:
            chain.append(cur)
            if not cur.supersedes:
                break
            cur = self.semantic.get(cur.supersedes[0])
        return chain

    # ── 热度 / 维护 ────────────────────────────────────────
    def record_access(self, item_id: str) -> Optional[MemoryItem]:
        for tier in self.tiers.values():
            try:
                item = tier.get(item_id)
            except NotImplementedError:
                continue
            if item is not None:
                item.touch()
                return item
        return None

    def pin(self, item_id: str) -> bool:
        item = self.working.get(item_id)
        if item is None:
            return False
        item.pinned = True
        return True

    def unpin(self, item_id: str) -> bool:
        item = self.working.get(item_id)
        if item is None:
            return False
        item.pinned = False
        return True

    def tick(self) -> Dict[str, List[str]]:
        """tick 调度入口 —— 按 config 决定同步/异步/跳过/节流。

        返回字典里:
        - "skipped": 节流跳过 / 已有 tick 在跑
        - "submitted": 异步提交了,本次不返回结果
        - promoted/demoted/recalled: 同步执行的结果
        """
        self._turn_counter += 1
        cfg = self.config

        if cfg.tick_mode == "off":
            return {"skipped": "off"}

        # 节流:不到 N 轮直接返回
        if self._turn_counter % cfg.tick_every_n_turns != 0:
            return {"skipped": "throttled"}

        if cfg.tick_mode == "sync":
            return self._tick_locked()

        # async 模式 —— 提交到后台
        if self._pending_tick is not None and not self._pending_tick.done():
            # 上次的还在跑,跳过这次(避免堆积)
            return {"skipped": "in_flight"}

        if self._executor is None:  # 防御性兜底
            return self._tick_locked()

        self._pending_tick = self._executor.submit(self._tick_safe)
        return {"submitted": True}

    def _tick_safe(self) -> Dict[str, List[str]]:
        """后台线程的入口 —— 捕获异常防止 Future 静默吞掉错误。"""
        try:
            return self._tick_locked()
        except Exception as exc:
            print(f"⚠️ 后台 tick 失败: {exc}")
            return {"error": str(exc)}

    def _tick_locked(self) -> Dict[str, List[str]]:
        """加锁版本的 tick —— 同步/异步都走这条。"""
        with self._lock:
            return self._tick_impl()

    def _tick_impl(self) -> Dict[str, List[str]]:
        """tick 的实际计算逻辑,假定调用方已持锁。"""
        promoted: List[str] = []
        demoted: List[str] = []
        recalled: List[str] = []

        cfg = self.config
        kw = dict(
            decay_tau_days=cfg.decay_tau_days,
            w_access=cfg.w_access,
            w_recency=cfg.w_recency,
            w_explicit=cfg.w_explicit,
            w_prior=cfg.w_prior,
        )

        # 取快照,避免遍历期间被改
        l1_snapshot = list(self.working.items())
        for item in l1_snapshot:
            score = item.importance(**kw)
            score_without_pin = score - (cfg.w_explicit if item.pinned else 0.0)
            if (
                not item.pinned
                and score >= cfg.promote_threshold
                and item.access_count >= cfg.promote_min_hits
            ):
                item.pinned = True
                promoted.append(item.id)
            elif item.pinned and score_without_pin < cfg.demote_threshold:
                item.pinned = False
                demoted.append(item.id)

        l1_ids = {it.id for it in self.working.items()}
        for item in list(self.semantic.items()):
            if item.id in l1_ids:
                continue
            score = item.importance(**kw)
            if (
                score >= cfg.promote_threshold
                and item.access_count >= cfg.promote_min_hits
            ):
                item.pinned = True
                self.working.add(item)
                recalled.append(item.id)

        # L3 → L1: 反复命中的检索项晋升回现场(高阈值)
        l3_promoted = self._maybe_promote_from_l3()

        if recalled or l3_promoted:
            self._cascade_evict_from_l1()

        # 实绩毕业: 久经考验的 L1 项晋升 L0(跨会话保留)
        graduated = self._maybe_graduate_to_l0()

        # L2 定期反思扳机: 距上次反思满 N 轮就触发。
        # 用"距上次"而非 _turn_counter % N —— 后者会被 tick_every_n_turns
        # 节流吞掉(只在 _tick_impl 实际执行的轮里判断,可能错过整除点)。
        reflect_n = self.config.l2_reflect_every_n_turns
        if reflect_n > 0 and self._turn_counter - self._last_reflect_turn >= reflect_n:
            self._last_reflect_turn = self._turn_counter
            self._reflect_on_summary()

        # KG 冷回路扳机: 同样用"距上次满 N 轮"触发(避免被 tick 节流吞掉整除点)
        recon_n = self.config.kg_reconcile_every_n_turns
        if (
            self._kg_reconciler is not None
            and recon_n > 0
            and self._turn_counter - self._last_reconcile_turn >= recon_n
        ):
            self._last_reconcile_turn = self._turn_counter
            try:
                self._kg_reconciler.reconcile()
            except Exception as exc:
                print(f"⚠️ KG 冷回路失败,跳过: {exc}")

        return {
            "promoted": promoted,
            "demoted": demoted,
            "recalled": recalled,
            "l3_promoted": l3_promoted,
            "graduated": graduated,
        }

    def _maybe_promote_from_l3(self) -> List[str]:
        """L3 缓冲里反复命中的项晋升回 L1。

        与 L5→L1 召回(上面那段)的区别:
        - L5→L1 看 importance(综合热度),覆盖所有 L5 项。
        - L3→L1 看"被动 recall 跨轮命中次数",只针对当前正反复用到的检索项。
          这些是"借来的参考资料反复要用 = 已重新参与对话",该恢复完整原文。

        晋升即从 L3 移除,避免和 L1 重复注入。
        """
        promoted: List[str] = []
        l1_ids = {it.id for it in self.working.items()}
        candidates = self.recall_buffer.promotable(
            min_hits=self.config.l3_promote_min_hits,
            min_score=self.config.l3_promote_min_score,
        )
        for entry in candidates:
            item = self.semantic.get(entry.item_id)
            # 真身没了 / 已被取代 / 已在 L1 → 直接从台账撤掉
            if item is None or not item.is_active or item.id in l1_ids:
                self.recall_buffer.remove(entry.item_id)
                continue
            item.pinned = True
            self.working.add(item)
            self.recall_buffer.remove(entry.item_id)
            promoted.append(item.id)
        return promoted

    # ── 异步 tick 的辅助 API ───────────────────────────────
    def wait_for_tick(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """阻塞等待最近一次异步 tick 完成 —— 测试或优雅关闭时用。"""
        if self._pending_tick is None:
            return None
        try:
            return self._pending_tick.result(timeout=timeout)
        except Exception as exc:
            print(f"⚠️ 等待 tick 失败: {exc}")
            return None

    def close(self) -> None:
        """关闭后台线程池 —— 进程退出前调用。"""
        if self._executor is None:
            return
        try:
            self.wait_for_tick(timeout=5)
        finally:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ── 工具方法 ───────────────────────────────────────────
    def clear(self) -> None:
        self.working = WorkingMemory(self.config)
        self.tiers[self.working.name] = self.working
        self.summary.clear()
        self.recall_buffer = RecallBuffer(self.config)
        self._turn_counter = 0
        self._last_reflect_turn = 0
        self._last_reconcile_turn = 0

    def stats(self) -> Dict[str, int]:
        summary = self.summary.current_summary()
        return {
            "l1_items": len(self.working.items()),
            "l1_tokens": self.working.total_tokens(),
            "l2_tokens": summary.token_estimate if summary else 0,
            "l3_entries": len(self.recall_buffer),
            "l4_items": self.cold.count(),
            "l5_items": len(self.semantic.items()),
        }
