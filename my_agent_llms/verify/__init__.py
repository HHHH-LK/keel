"""verify —— 在线验证-重试机制(Online Verify-Retry Loop)。"""
from my_agent_llms.verify.spec import Check, CheckSpec, SpecGenerator
from my_agent_llms.verify.checkers import CheckContext, CheckerRunner, check_one
from my_agent_llms.verify.residual import residual
from my_agent_llms.verify.convergence import (
    Verdict, ConvergenceJudge, Round, fingerprint,
)
from my_agent_llms.verify.loop import VerifyResult, VerifyRetryLoop, Executor
from my_agent_llms.verify.stall import stalled_oracle_ids, apply_demotion

__all__ = [
    "Check", "CheckSpec", "SpecGenerator",
    "CheckContext", "CheckerRunner", "check_one",
    "residual",
    "Verdict", "ConvergenceJudge", "Round", "fingerprint",
    "VerifyResult", "VerifyRetryLoop", "Executor",
    "stalled_oracle_ids", "apply_demotion",
]
