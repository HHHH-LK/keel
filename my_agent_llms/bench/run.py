"""离线评估入口:python -m my_agent_llms.bench.run [cases_dir]"""
from __future__ import annotations

import sys
from pathlib import Path

from my_agent_llms.bench.case import load_cases
from my_agent_llms.bench.runner import run_case
from my_agent_llms.bench.scorer import score
from my_agent_llms.bench.report import summarize


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cases_dir = argv[0] if argv else str(Path(__file__).parent / "cases")
    from my_agent_llms.bench.default_factory import build_factory
    factory = build_factory()
    cases = load_cases(cases_dir)
    scores = []
    for c in cases:
        print(f"[run] {c.id} ...")
        rr = run_case(c, factory)
        s = score(c, rr)
        scores.append(s)
        print(f"  {'✓' if s.passed else '✗'} residual={s.residual:.1f}")
    print("\n" + summarize(scores))


if __name__ == "__main__":
    main()
