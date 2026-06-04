from my_agent_llms.memory.context_engine import count_tokens


def test_count_tokens_positive_and_monotonic():
    # 空串 0;非空 >= 1;更长文本 token 不减
    assert count_tokens("") == 0
    assert count_tokens("hi") >= 1
    short = count_tokens("hello")
    long = count_tokens("hello hello hello hello hello")
    assert long >= short
