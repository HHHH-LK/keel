"""上下文工程包:与具体记忆层解耦的窗口编排组件。

依赖方向单向:memory 依赖 context,context 不依赖 memory。
"""
from .engine import (
    BudgetReport,
    BuildResult,
    ContextEngine,
    ContextSegment,
    bigram_relevance,
    count_tokens,
    make_embedding_relevance,
)

__all__ = [
    "BudgetReport",
    "BuildResult",
    "ContextEngine",
    "ContextSegment",
    "bigram_relevance",
    "count_tokens",
    "make_embedding_relevance",
]
