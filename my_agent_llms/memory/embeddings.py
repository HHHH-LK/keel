"""Embedding 提供商抽象。

设计原则：
- `EmbeddingProvider` 是抽象接口，向量库只依赖它
- `OpenAIEmbedding` 用于生产（兼容所有 OpenAI 兼容 API：OpenAI / Aliyun / Zhipu 等）
- `HashEmbedding` 用于测试和无网络环境（确定性、无外部依赖）
- 用户也可以传任意 `Callable[[str], Sequence[float]]`，框架内部统一包成 `_CallableEmbedding`
"""
import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


class EmbeddingProvider(ABC):
    """文本 → 向量。向量维度由 `dim` 暴露，向量库据此预分配存储。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class HashEmbedding(EmbeddingProvider):
    """基于词哈希的确定性 embedding，零依赖、可重复。

    不是真正的语义向量，**只用于测试 / 离线开发**。语义相近的文本
    通常不会有高余弦相似度。
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for tok in _tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            for i in range(self._dim):
                vec[i] += (digest[i % len(digest)] / 255.0) - 0.5
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedding(EmbeddingProvider):
    """走 OpenAI 兼容 API 的 embedding 调用。

    支持复用 `MyLLM.client`，或自建 OpenAI client。
    常用模型：
    - OpenAI 官方:           text-embedding-3-small (1536) / text-embedding-3-large (3072)
    - 阿里云 (DashScope):    text-embedding-v2 (1536) / text-embedding-v3 (1024)
    - 智谱:                  embedding-3 (1024)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        client=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if client is None:
            from openai import OpenAI

            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)

        self.client = client
        self.model = model
        self._dim = dim

    @classmethod
    def from_llm(cls, llm, model: str = "text-embedding-3-small", dim: int = 1536) -> "OpenAIEmbedding":
        """复用一个已配置好的 MyLLM 的 OpenAI client（共享 base_url / api_key）。"""
        client = getattr(llm, "client", None)
        if client is None:
            raise ValueError("MyLLM 没有可用的 OpenAI client（provider 必须是 OpenAI 兼容）。")
        return cls(model=model, dim=dim, client=client)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        text = text or " "
        resp = self.client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # OpenAI embedding API 单次支持批量，比逐条快很多
        clean = [t or " " for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=clean)
        return [list(d.embedding) for d in resp.data]


class _CallableEmbedding(EmbeddingProvider):
    """把用户传入的裸 callable 包装成 EmbeddingProvider。"""

    def __init__(self, func: Callable[[str], Sequence[float]], dim: int):
        self._func = func
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        return list(self._func(text))


def coerce_embedding(
    obj: Union[None, EmbeddingProvider, Callable[[str], Sequence[float]]],
    *,
    dim_hint: Optional[int] = None,
) -> Optional[EmbeddingProvider]:
    """把用户可能传的多种形态统一成 EmbeddingProvider。"""
    if obj is None:
        return None
    if isinstance(obj, EmbeddingProvider):
        return obj
    if callable(obj):
        if dim_hint is None:
            # 探测一次
            probe = list(obj("hello"))
            dim_hint = len(probe)
        return _CallableEmbedding(obj, dim=dim_hint)
    raise TypeError(f"无法识别的 embedding 类型: {type(obj)}")
