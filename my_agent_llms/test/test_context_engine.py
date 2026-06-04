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


from my_agent_llms.memory.context_engine import ContextEngine


def _seg(source, content, *, priority=0.5, tokens=None, floor=False,
         order=6, seq=0, item_id=None, role="system"):
    return ContextSegment(
        source=source, role=role, content=content, priority=priority,
        tokens=tokens if tokens is not None else max(1, len(content) // 3),
        floor=floor, order=order, seq=seq, item_id=item_id,
    )


def test_dedup_exact_keeps_more_authoritative():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("l1", "用户对花生过敏", order=6, seq=1),
        _seg("l0-core", "用户对花生过敏", order=1, seq=0),
    ]
    kept = eng._dedup(segs)
    sources = {s.source for s in kept}
    assert "l0-core" in sources       # 权威的留下
    assert "l1" not in sources        # 重复的被删


def test_dedup_substring_drops_contained():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("l0-core", "喜欢猫", order=1, seq=0),
        _seg("recall", "用户说他非常喜欢猫还有狗", order=5, seq=1),
    ]
    kept = eng._dedup(segs)
    # "喜欢猫" 被更长的 recall 完全包含 → 删短的 l0-core
    assert any(s.source == "recall" for s in kept)
    assert all(s.source != "l0-core" for s in kept)


def test_dedup_item_id_prefers_l1():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("recall", "片段预览", order=5, seq=1, item_id="x1"),
        _seg("l1", "完整原文不同字面", order=6, seq=0, item_id="x1"),
    ]
    kept = eng._dedup(segs)
    assert any(s.source == "l1" for s in kept)
    assert all(s.source != "recall" for s in kept)


def test_dedup_disabled_keeps_all():
    eng = ContextEngine(dedup=False)
    segs = [_seg("l1", "同样的话", seq=0), _seg("l0-core", "同样的话", order=1, seq=1)]
    assert len(eng._dedup(segs)) == 2
