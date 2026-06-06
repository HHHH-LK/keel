"""TDD 模式:独立出题(test-author)+ 硬红门 + 实现转绿,与事后 verify 并存。"""
from my_agent_llms.tdd.orchestrator import run_tdd, TddResult
from my_agent_llms.tdd.classify import classify, TddDecision
from my_agent_llms.tdd.runner import run_pytest, RunResult, RunOutcome
from my_agent_llms.tdd.gates import red_gate, green_gate, RedVerdict, GreenVerdict

__all__ = [
    "run_tdd", "TddResult", "classify", "TddDecision",
    "run_pytest", "RunResult", "RunOutcome",
    "red_gate", "green_gate", "RedVerdict", "GreenVerdict",
]
