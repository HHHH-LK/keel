"""最小离线评估集:守护 promote 判定 + type 判定的准确率。
调 seed_score 的权重/阈值后跑它,确认没有顾此失彼。"""
import json
from pathlib import Path

from my_agent_llms.memory.seed_score import evaluate_prior_score, should_auto_pin
from my_agent_llms.memory.playbook.card import classify_content_type

DATA = Path(__file__).parent / "data" / "seed_eval.jsonl"


def _load():
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_eval_set_promotion_accuracy():
    rows = _load()
    wrong = []
    for r in rows:
        score = evaluate_prior_score(r["content"], r["role"], task_turn=r["task_turn"])
        promote = r["role"] == "user" and should_auto_pin(score)
        if promote != r["expect_promote"]:
            wrong.append((r["content"], r["expect_promote"], promote, round(score, 3)))
    assert not wrong, f"promote 判定错误: {wrong}"


def test_eval_set_type_accuracy():
    rows = _load()
    wrong = []
    for r in rows:
        got = classify_content_type(r["content"]).value
        if got != r["expect_type"]:
            wrong.append((r["content"], r["expect_type"], got))
    assert not wrong, f"type 判定错误: {wrong}"
