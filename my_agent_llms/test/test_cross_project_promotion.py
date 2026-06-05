"""跨项目提升:同一事实在 ≥N 个不同项目写主图 → 进用户层 KG。"""
from my_agent_llms.memory import MemoryConfig, MemoryManager


class FakeLLM:
    def __init__(self, *a, **k): pass
    def invoke(self, messages): return "[]"


def _rel(obj="Postgres"):
    return {"subject_type": "PERSON", "subject_name": "user",
            "predicate": "喜欢", "object_type": "TECH", "object_name": obj}


def _mgr(user_dir, project_id):
    return MemoryManager(
        MemoryConfig(conflict_strength="extreme", user_storage_dir=user_dir,
                     project_id=project_id, user_promote_min_projects=2),
        llm=FakeLLM(),
    )


def test_promote_after_two_projects(tmp_path):
    user_dir = tmp_path / "user"
    mgrA = _mgr(user_dir, "projA")
    mgrA.conflict_detector.apply_confirmed_relation(_rel())
    assert all("Postgres" not in f for f in mgrA.user_layer.query_facts("用户 喜欢 什么"))
    mgrB = _mgr(user_dir, "projB")
    mgrB.conflict_detector.apply_confirmed_relation(_rel())
    facts = mgrB.user_layer.query_facts("用户 喜欢 什么")
    assert any("Postgres" in f for f in facts)


def test_same_project_twice_not_promoted(tmp_path):
    user_dir = tmp_path / "user"
    mgr = _mgr(user_dir, "projA")
    mgr.conflict_detector.apply_confirmed_relation(_rel())
    mgr.conflict_detector.apply_confirmed_relation(_rel())
    assert all("Postgres" not in f for f in mgr.user_layer.query_facts("用户 喜欢 什么"))
