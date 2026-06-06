"""task_turn 信号端到端压低任务指令分,阻止其固化为 L0。"""
from my_agent_llms.memory import MemoryManager, MemoryConfig


def test_task_turn_suppresses_directive_promotion():
    mgr = MemoryManager(MemoryConfig())
    # 自指+祈使本会算 hard_constraint(0.5);叠加 task_turn(-0.25) → 0.25 < 0.4,不固化
    mgr.write("我必须更新这个文件", role="user", task_turn=True)
    assert mgr.playbook.count_active() == 0


def test_non_task_turn_user_fact_still_promotes():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我对花生过敏", role="user", task_turn=False)
    assert mgr.playbook.count_active() == 1


def test_write_task_turn_defaults_false():
    # 不传 task_turn → 旧行为:真约束照常固化
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我对花生过敏", role="user")
    assert mgr.playbook.count_active() == 1
