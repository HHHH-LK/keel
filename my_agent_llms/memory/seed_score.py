"""种子分 —— 写入时的启发式重要性预判。

设计原则:
- 预定义关键词(零 LLM 依赖,跨 provider 100% 稳定)
- 主分取最高档(不累加),调整分小幅修正
- 范围 [0, 1],参与 importance 公式的 prior_score 分量
- 不追求精准,corner case 交给 KG detector 兜底纠正

与 KG 反哺协同:
- 种子分 = 字符串特征粗筛(本模块)
- KG 反哺 = detector 抽出事实/检测冲突时累加 KG_FEEDBACK_BOOST
- 最终 prior_score = clamp(种子分 + KG 反哺, 0, 1)
"""
from typing import Dict, List


# ─────────────────────────────────────────────────────────
# 主分: 按 type 分档,取最高档(不累加)
# 关键词命中即归入该档
# ─────────────────────────────────────────────────────────
CATEGORY_KEYWORDS: Dict[str, Dict] = {
    "hard_constraint": {
        "score": 0.50,
        "keywords": [
            "过敏", "禁忌", "不能吃", "不能喝",
            "必须", "千万不要", "严禁", "忌讳",
            "不可以", "绝对不",
        ],
    },
    "identity": {
        "score": 0.40,
        "keywords": [
            "我叫", "我是", "我的工作", "我的职业",
            "我住在", "我家", "我老婆", "我老公",
            "我儿子", "我女儿", "我妈", "我爸",
            "我女朋友", "我男朋友",
        ],
    },
    "preference": {
        "score": 0.30,
        "keywords": [
            "我喜欢", "我不喜欢", "我偏好", "我习惯",
            "我讨厌", "我中意", "我对", "我特别",
        ],
    },
    "decision": {
        "score": 0.25,
        "keywords": [
            "我决定", "我打算", "我准备", "我要",
            "我会", "我想",
        ],
    },
    "state": {
        "score": 0.20,
        "keywords": [
            "我在做", "我正在", "我最近", "我目前",
            "我这个月", "我这周", "我今天",
        ],
    },
}


# ─────────────────────────────────────────────────────────
# 调整规则: 在主分上做小幅修正(可累加)
# ─────────────────────────────────────────────────────────
SHORT_MESSAGE_THRESHOLD = 10
SHORT_MESSAGE_PENALTY = -0.20
QUESTION_PENALTY = -0.15
ASSISTANT_ROLE_PENALTY = -0.10


# ─────────────────────────────────────────────────────────
# 阈值常量
# ─────────────────────────────────────────────────────────
# prior_score >= 此值时,写入瞬间直接 pinned(跳过 access_count + tick 等待期)
AUTO_PIN_THRESHOLD = 0.4

# KG detector 抽出事实/检测到冲突时,给原 item 的 prior_score 加分
KG_FEEDBACK_BOOST = 0.3


def evaluate_prior_score(content: str, role: str = "user") -> float:
    """根据消息内容打种子分。

    Step 1: 主分 = max(命中的所有档位的 score)
    Step 2: 调整分 = sum(命中的调整规则)
            短消息扣分(hard_constraint 类豁免)
            问句扣分
            assistant 自述扣分
    Step 3: clamp 到 [0, 1]

    Args:
        content: 消息文本
        role: 消息角色 (user/assistant/system 等)

    Returns:
        prior_score, 范围 [0.0, 1.0]
    """
    if not content:
        return 0.0

    content = content.strip()

    # Step 1: 主分(取最高档,不累加)
    base = 0.0
    for category, conf in CATEGORY_KEYWORDS.items():
        if any(kw in content for kw in conf["keywords"]):
            if conf["score"] > base:
                base = conf["score"]

    # Step 2: 调整分
    adj = 0.0
    # 短消息扣分:只在"没命中任何关键词"时生效。
    # 中文表达密度高(如"我喜欢喝美式咖啡"才 8 字),
    # 命中关键词的短消息不应该被惩罚。
    if len(content) < SHORT_MESSAGE_THRESHOLD and base == 0:
        adj += SHORT_MESSAGE_PENALTY
    # 问句扣分(问题不是事实陈述)
    if content.endswith(("?", "？")):
        adj += QUESTION_PENALTY
    # assistant 自述扣分(避免 LLM 自己说的内容被高估)
    if role == "assistant":
        adj += ASSISTANT_ROLE_PENALTY

    # Step 3: clamp
    return max(0.0, min(1.0, base + adj))


def should_auto_pin(prior_score: float) -> bool:
    """是否触发立即 pin(跳过 access_count + tick 等待期)。

    用于种子分够高的项一进 L1 就受 pinned 保护,
    不会被 cascade_evict 踢出。
    """
    return prior_score >= AUTO_PIN_THRESHOLD


def boost_with_kg_feedback(prior_score: float) -> float:
    """KG detector 反哺: 累加 KG_FEEDBACK_BOOST 后 clamp。

    触发场景: detector 在 new_item 上抽出新事实/检测到冲突,
    说明 new_item 含有"有事实价值"的信息,值得加分。
    """
    return max(0.0, min(1.0, prior_score + KG_FEEDBACK_BOOST))
