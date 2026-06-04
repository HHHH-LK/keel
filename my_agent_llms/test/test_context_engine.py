from my_agent_llms.memory.context_engine import count_tokens


def test_count_tokens_positive_and_monotonic():
    # 空串 0;非空 >= 1;更长文本 token 不减
    assert count_tokens("") == 0
    assert count_tokens("hi") >= 1
    short = count_tokens("hello")
    long = count_tokens("hello hello hello hello hello")
    assert long >= short


from my_agent_llms.memory.context_engine import (
    ContextSegment, bigram_relevance, make_embedding_relevance,
)


def test_bigram_relevance_basic():
    assert bigram_relevance("用户对花生过敏", "花生") > 0.0
    assert bigram_relevance("完全无关", "xyz") == 0.0
    assert bigram_relevance("", "q") == 0.0


def test_embedding_relevance_falls_back_on_error():
    class BadProvider:
        def embed(self, text):
            raise RuntimeError("no backend")
    fn = make_embedding_relevance(BadProvider())
    # provider 抛错 → 自动回退 bigram,不抛异常
    assert fn("花生过敏", "花生") > 0.0


def test_embedding_relevance_cosine():
    # 假 provider:相同文本向量相同 → 余弦=1;正交 → 0
    vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    class FakeProvider:
        def embed(self, text):
            return vecs[text]
    fn = make_embedding_relevance(FakeProvider())
    assert abs(fn("a", "a") - 1.0) < 1e-6
    assert abs(fn("a", "b") - 0.0) < 1e-6


def test_segment_defaults():
    seg = ContextSegment(source="l1", role="user", content="hi",
                         priority=0.5, tokens=1, floor=False, order=6)
    assert seg.seq == 0
    assert seg.item_id is None
