"""UserLayer:用户层 KG 读写。"""
from my_agent_llms.memory.user_layer import UserLayer


def _rel(obj="Postgres"):
    return {"subject_type": "PERSON", "subject_name": "user",
            "predicate": "喜欢", "object_type": "TECH", "object_name": obj}


def test_ingest_then_query():
    ul = UserLayer(storage_dir=None, llm=None, embedder=None)   # 内存模式
    ul.ingest_confirmed_fact(_rel("Postgres"))
    facts = ul.query_facts("用户 喜欢 什么")
    assert any("Postgres" in f for f in facts)


def test_query_empty_when_no_facts():
    ul = UserLayer(storage_dir=None, llm=None, embedder=None)
    assert ul.query_facts("任何") == []
