"""MemoryConfig 用户层字段与路径派生。"""
from pathlib import Path

from my_agent_llms.memory.config import MemoryConfig


def test_user_paths_none_when_unset():
    cfg = MemoryConfig()
    assert cfg.user_storage_dir is None
    assert cfg.user_kg_path() is None
    assert cfg.user_vector_path() is None
    assert cfg.user_playbook_path() is None


def test_user_paths_derived(tmp_path):
    cfg = MemoryConfig(user_storage_dir=tmp_path, vector_backend="sqlite")
    assert cfg.user_kg_path() == tmp_path / "kg.db"
    assert cfg.user_vector_path() == tmp_path / "memory.db"
    assert cfg.user_playbook_path() == tmp_path / "memory.db"


def test_user_promote_default_is_two():
    assert MemoryConfig().user_promote_min_projects == 2
