"""种子分 + KG 反哺 测试。"""
from my_agent_llms.memory.seed_score import (
    AUTO_PIN_THRESHOLD,
    KG_FEEDBACK_BOOST,
    TASK_TURN_PENALTY,
    boost_with_kg_feedback,
    evaluate_prior_score,
    is_hard_constraint_content,
    should_auto_pin,
)


# ─────────────────────────────────────────────
# 主分: 关键词命中各档
# ─────────────────────────────────────────────

def test_hard_constraint_triggers_high_score():
    score = evaluate_prior_score("我对花生过敏")
    assert score >= 0.5, f"hard_constraint 应 ≥ 0.5, 实际 {score}"
    assert should_auto_pin(score)


def test_identity_score():
    score = evaluate_prior_score("我叫张三,是一名软件工程师")
    # identity 0.40,不短不问句 → 0.40
    assert 0.35 <= score <= 0.45
    assert should_auto_pin(score)


def test_preference_score():
    score = evaluate_prior_score("我喜欢喝美式咖啡")
    # preference 0.30,长度 > 10,无调整
    assert 0.25 <= score <= 0.35
    assert not should_auto_pin(score)  # 0.30 < 0.40


def test_decision_score():
    score = evaluate_prior_score("我决定下周开始学 Java")
    # decision 0.25
    assert 0.2 <= score <= 0.3


def test_state_score():
    score = evaluate_prior_score("我最近在做一个 AI 项目")
    # state 0.20
    assert 0.15 <= score <= 0.25


def test_no_keyword_zero_score():
    score = evaluate_prior_score("今天天气挺好的")
    assert score == 0.0


# ─────────────────────────────────────────────
# 主分: 取最高档(不累加)
# ─────────────────────────────────────────────

def test_multiple_categories_take_max():
    """含 hard(0.5) + decision(0.25) + preference(0.3),应取 0.5。"""
    score = evaluate_prior_score("我决定避开花生过敏源,因为我对花生过敏")
    assert score >= 0.5
    assert score <= 1.0  # 不超过 1.0(累加会 > 1)


# ─────────────────────────────────────────────
# 调整分: 短消息 / 问句 / assistant
# ─────────────────────────────────────────────

def test_short_message_penalty():
    score = evaluate_prior_score("好的")
    # 无关键词 → base=0,短消息 → -0.20,clamp 后 0
    assert score == 0.0


def test_short_with_keyword_no_penalty():
    """命中关键词的短消息不受短消息扣分影响(中文表达密度高)。"""
    # "过敏" 2 字符 → 命中 hard,base=0.5,base>0 → 不扣短消息
    assert evaluate_prior_score("过敏") >= 0.5
    # "我叫张三" 4 字符 → 命中 identity 0.40,不扣
    assert evaluate_prior_score("我叫张三") >= 0.35


def test_question_penalty():
    score = evaluate_prior_score("我喜欢什么咖啡?")
    # preference 0.30 - 问句 0.15 = 0.15
    assert 0.1 <= score <= 0.2


def test_question_chinese_punct():
    score = evaluate_prior_score("我喜欢什么咖啡？")  # 全角问号
    assert 0.1 <= score <= 0.2


def test_assistant_role_penalty():
    score = evaluate_prior_score("我决定为你推荐", role="assistant")
    # decision 0.25 - assistant 0.10 = 0.15
    assert 0.1 <= score <= 0.2


# ─────────────────────────────────────────────
# Clamp 边界
# ─────────────────────────────────────────────

def test_score_clamp_lower():
    """所有扣分加起来不应该 < 0。"""
    score = evaluate_prior_score("?", role="assistant")  # 短 + 问句 + assistant
    assert score >= 0.0


def test_score_clamp_upper():
    """种子分本身 ≤ 0.5(hard 上限),不会超 1.0。"""
    score = evaluate_prior_score("过敏过敏过敏我必须千万不要")
    assert score <= 1.0


def test_empty_content():
    assert evaluate_prior_score("") == 0.0
    assert evaluate_prior_score("   ") == 0.0


# ─────────────────────────────────────────────
# Auto pin 阈值
# ─────────────────────────────────────────────

def test_auto_pin_threshold():
    assert should_auto_pin(0.5) is True
    assert should_auto_pin(0.4) is True   # 等于阈值
    assert should_auto_pin(0.39) is False
    assert should_auto_pin(0.0) is False


# ─────────────────────────────────────────────
# KG 反哺
# ─────────────────────────────────────────────

def test_kg_feedback_boost():
    # preference 类种子分 0.30,KG 反哺 +0.3 → 0.60,触发 auto_pin
    seed = 0.30
    boosted = boost_with_kg_feedback(seed)
    assert abs(boosted - (seed + KG_FEEDBACK_BOOST)) < 1e-6
    assert should_auto_pin(boosted)


def test_kg_feedback_clamp():
    """加 KG 反哺也不会超 1.0。"""
    boosted = boost_with_kg_feedback(0.9)
    assert boosted <= 1.0
    assert boosted >= 0.9


def test_kg_feedback_combined_scenario():
    """完整场景: 用户说'我对花生过敏' → 种子分 0.5 + KG 反哺 → 0.8。"""
    seed = evaluate_prior_score("我对花生过敏")
    boosted = boost_with_kg_feedback(seed)
    assert boosted >= 0.7  # 0.5 + 0.3 = 0.8
    assert should_auto_pin(boosted)


# ─────────────────────────────────────────────
# 真实场景验证
# ─────────────────────────────────────────────

def test_realistic_scenarios():
    scenarios = [
        # (content, role, expected_min, expected_max, should_pin)
        ("我对花生过敏",                  "user",      0.50, 0.55, True),
        ("我叫张三",                      "user",      0.20, 0.45, True),  # 短消息但 identity
        ("我喜欢喝美式咖啡",              "user",      0.25, 0.35, False),
        ("我决定下周开始学 Java",         "user",      0.20, 0.30, False),
        ("我最近在做一个 AI 项目",        "user",      0.15, 0.25, False),
        ("今天天气挺好的",                "user",      0.00, 0.05, False),
        ("好的",                          "user",      0.00, 0.05, False),
        ("ok 谢谢",                       "user",      0.00, 0.05, False),
        ("我喜欢什么咖啡?",               "user",      0.10, 0.20, False),
    ]

    for content, role, lo, hi, should_pin in scenarios:
        s = evaluate_prior_score(content, role)
        assert lo <= s <= hi, f'"{content}" 期望 [{lo}, {hi}],实际 {s:.3f}'
        assert should_auto_pin(s) == should_pin, (
            f'"{content}" 期望 pin={should_pin}, prior={s:.3f}'
        )


def test_user_fact_is_hard_constraint():
    assert is_hard_constraint_content("我对花生过敏")
    assert is_hard_constraint_content("不能吃海鲜")          # 用户事实词,无需自指


def test_generic_imperative_needs_self_reference():
    assert is_hard_constraint_content("我必须每天吃药")        # 自指 → 算
    assert not is_hard_constraint_content("回答里必须包含编程语言")  # 任务指令 → 不算
    assert not is_hard_constraint_content("文件必须包含字段 status")  # 任务指令 → 不算


def test_task_directive_scores_low():
    s = evaluate_prior_score("回答里必须包含编程语言")
    assert s < 0.4, f"任务指令应低于 0.4,实际 {s}"


def test_task_turn_penalty_applied():
    base = evaluate_prior_score("我对花生过敏", task_turn=False)
    penalized = evaluate_prior_score("我对花生过敏", task_turn=True)
    assert abs((base - penalized) - abs(TASK_TURN_PENALTY)) < 1e-9


def test_directive_nouns_do_not_suppress_self_description():
    # 含产物名词但是用户自述 → 不被压低
    assert evaluate_prior_score("我喜欢生成艺术") >= 0.30   # preference 保住
    assert evaluate_prior_score("我是文件管理员") >= 0.40   # identity 保住


def test_self_intro_compound_not_hard_constraint():
    # "自我介绍必须简洁" 里的"自我"不是自指 → 不算 hard_constraint
    assert not is_hard_constraint_content("自我介绍必须简洁")


if __name__ == "__main__":
    # 当作脚本跑,看每条场景的实际打分
    test_cases = [
        "我对花生过敏",
        "我叫张三",
        "我是一名 Python 工程师",
        "我喜欢喝美式咖啡",
        "我决定下周开始学 Java",
        "我最近在做一个 AI 项目",
        "今天天气挺好的",
        "好的",
        "我喜欢什么咖啡?",
        "我决定避开花生过敏源",
    ]
    print(f"AUTO_PIN_THRESHOLD = {AUTO_PIN_THRESHOLD}")
    print(f"KG_FEEDBACK_BOOST  = {KG_FEEDBACK_BOOST}")
    print()
    print(f"{'content':<35} {'seed':>6} {'kg_boost':>10} {'pin?':>6}")
    print("-" * 65)
    for c in test_cases:
        s = evaluate_prior_score(c)
        b = boost_with_kg_feedback(s)
        pin = "✓" if should_auto_pin(s) else " "
        pin_kg = "✓" if should_auto_pin(b) else " "
        print(f'{c:<35} {s:>6.3f} {b:>10.3f}  {pin}/{pin_kg}')

    # 运行 pytest 风格的断言
    print()
    print("=" * 65)
    print("running asserts...")
    test_hard_constraint_triggers_high_score()
    test_identity_score()
    test_preference_score()
    test_decision_score()
    test_state_score()
    test_no_keyword_zero_score()
    test_multiple_categories_take_max()
    test_short_message_penalty()
    test_short_with_keyword_no_penalty()
    test_question_penalty()
    test_question_chinese_punct()
    test_assistant_role_penalty()
    test_score_clamp_lower()
    test_score_clamp_upper()
    test_empty_content()
    test_auto_pin_threshold()
    test_kg_feedback_boost()
    test_kg_feedback_clamp()
    test_kg_feedback_combined_scenario()
    test_realistic_scenarios()
    print("all passed")
