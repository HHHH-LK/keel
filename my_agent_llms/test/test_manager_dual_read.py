"""MemoryManager 双层读取:用户层事实并入 recall_facts,项目优先+去重。"""
from my_agent_llms.memory import MemoryConfig, MemoryManager
from my_agent_llms.memory.embeddings import HashEmbedding


class FakeLLM:
    """最小 LLM stub:invoke 返回空列表 JSON,供 KG 实体抽取降级用。"""
    def __init__(self, *a, **k): pass

    def invoke(self, messages):
        return "[]"


def _rel(obj):
    return {"subject_type": "PERSON", "subject_name": "user",
            "predicate": "喜欢", "object_type": "TECH", "object_name": obj}


def test_no_user_layer_when_unset():
    mgr = MemoryManager(MemoryConfig(conflict_strength="extreme"), llm=FakeLLM())
    assert mgr.user_layer is None


def test_user_layer_built_when_set(tmp_path):
    mgr = MemoryManager(
        MemoryConfig(conflict_strength="extreme", user_storage_dir=tmp_path),
        llm=FakeLLM(),
    )
    assert mgr.user_layer is not None


def test_recall_facts_merges_both_layers(tmp_path):
    mgr = MemoryManager(
        MemoryConfig(conflict_strength="extreme", user_storage_dir=tmp_path),
        llm=FakeLLM(),
        embedding=HashEmbedding(dim=32),
    )
    mgr.conflict_detector.apply_confirmed_relation(_rel("Postgres"))
    mgr.user_layer.ingest_confirmed_fact(_rel("Vim"))
    facts = mgr.recall_facts("用户 喜欢 什么", max_facts=8)
    joined = " ".join(facts)
    assert "Postgres" in joined and "Vim" in joined
