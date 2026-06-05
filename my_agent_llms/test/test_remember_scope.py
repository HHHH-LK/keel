"""remember 的 scope 路由:user→用户层 playbook,缺省→项目层。"""
from my_agent_llms.memory import MemoryConfig, MemoryManager


def test_default_scope_writes_project(tmp_path):
    mgr = MemoryManager(MemoryConfig(storage_dir=tmp_path, user_storage_dir=tmp_path / "u"))
    mgr.remember("项目用 Postgres")
    assert any("Postgres" in c.content for c in mgr.playbook.all_active())


def test_user_scope_writes_user_layer(tmp_path):
    mgr = MemoryManager(MemoryConfig(storage_dir=tmp_path, user_storage_dir=tmp_path / "u"))
    mgr.remember("我喜欢 4 空格缩进", scope="user")
    assert any("缩进" in c.content for c in mgr.user_layer.playbook.all_active())


def test_user_scope_degrades_without_user_layer(tmp_path):
    mgr = MemoryManager(MemoryConfig(storage_dir=tmp_path))   # 无用户层
    card = mgr.remember("随便", scope="user")                 # 不报错,落项目层
    assert card is not None
