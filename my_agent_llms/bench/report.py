"""汇总报告:success rate + 每 case ✓/✗ + 失败项。"""
from __future__ import annotations


def summarize(scores) -> str:
    total = len(scores)
    ok = sum(1 for s in scores if s.passed)
    pct = (100 * ok // total) if total else 0
    lines = [f"通过 {ok}/{total} ({pct}%)", "-" * 32]
    for s in scores:
        mark = "✓" if s.passed else "✗"
        tail = "" if s.passed else f"  未过: {', '.join(s.failed) or 'residual>0'}"
        lines.append(f"  {mark} {s.case_id}  residual={s.residual:.1f}{tail}")
    return "\n".join(lines)
