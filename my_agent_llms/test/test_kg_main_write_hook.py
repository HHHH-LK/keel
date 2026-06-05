"""KG detector:写主图钩子 + 直写主图入口。"""
from my_agent_llms.memory.kg import KGStore, KnowledgeGraphConflictDetector


def _rel(subj="user", pred="喜欢", obj="Postgres"):
    return {"subject_type": "PERSON", "subject_name": subj,
            "predicate": pred, "object_type": "TECH", "object_name": obj}


def test_apply_confirmed_writes_to_main():
    det = KnowledgeGraphConflictDetector(llm=None, store=KGStore())
    det.apply_confirmed_relation(_rel())
    facts = det.store.all_relations(only_active=True)
    assert len(facts) == 1


def test_on_main_write_hook_fires():
    det = KnowledgeGraphConflictDetector(llm=None, store=KGStore())
    seen = []
    det.on_main_write = lambda rd: seen.append(rd.get("object_name"))
    det.apply_confirmed_relation(_rel(obj="Rust"))
    assert seen == ["Rust"]
