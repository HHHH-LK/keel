"""上下文工程编排层:在 token 预算内决定 LLM 窗口里实际放什么。

记忆层(L0–L5)负责"有什么、多重要"(库存生命周期);
本模块负责"这一次窗口里放什么"(单次组装的预算/去重/排序)。
被丢弃的内容仍在 L4/L5,下一轮可被 recall 回来。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def count_tokens(text: str) -> int:
    """估算文本 token 数。优先 tiktoken,未安装时回退 len//3 启发式。

    回退值对中英混合是保守估计;不影响"永不超预算"铁律,
    因为预算判定与兜底都用同一个 counter。
    """
    if not text:
        return 0
    try:
        import tiktoken  # 可选依赖
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)
