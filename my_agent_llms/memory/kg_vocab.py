"""受控词表:谓词(带基数)+ scope。

为什么受控:KG 的冲突检测按"字段精确匹配"判定。如果谓词/scope 让 LLM 自由
发挥,同义词("喜欢"/"偏好")会让冲突悄悄漏判,跑几百轮后图越长越脏。
归一到 canonical 后,冲突键才稳。

谓词基数(cardinality)是安全关键:
- single(单值/functional):一个主语只能有一个当前值(现居地、主力语言)
  → 新值取代旧值
- multi(多值/set):一个主语可以有很多(过敏、会的语言)
  → 新值只追加,绝不取代

未知谓词默认 multi —— 宁可漏判(两条共存)也不可误杀(把"对花生过敏"删掉)。
"""
from typing import Dict, Tuple

CARDINALITY_SINGLE = "single"
CARDINALITY_MULTI = "multi"

# canonical -> (cardinality, [synonyms])
_PREDICATES: Dict[str, Tuple[str, list]] = {
    # ── 单值(functional):只能有一个当前值 ──
    "现居地": (CARDINALITY_SINGLE, ["住在", "居住", "居住于", "现居", "住"]),
    "现任雇主": (CARDINALITY_SINGLE, ["就职于", "工作于", "任职于", "雇主", "供职于"]),
    "主力语言": (CARDINALITY_SINGLE, ["主用语言", "主要语言"]),
    "婚姻状态": (CARDINALITY_SINGLE, ["婚姻"]),
    "当前项目": (CARDINALITY_SINGLE, ["在做的项目", "手头项目"]),
    "母语": (CARDINALITY_SINGLE, ["第一语言"]),
    "出生地": (CARDINALITY_SINGLE, ["出生于", "生于"]),
    "职业": (CARDINALITY_SINGLE, ["职位", "工作是"]),
    # ── 多值(set):可以有很多 ──
    "喜欢": (CARDINALITY_MULTI, ["偏好", "爱用", "喜爱", "钟爱", "喜好"]),
    "使用": (CARDINALITY_MULTI, ["用", "采用"]),
    "会": (CARDINALITY_MULTI, ["掌握", "会用", "懂"]),
    "去过": (CARDINALITY_MULTI, ["到过", "去了"]),
    "拥有": (CARDINALITY_MULTI, ["有", "持有"]),
    "过敏": (CARDINALITY_MULTI, ["过敏于"]),
    "兴趣": (CARDINALITY_MULTI, ["爱好", "感兴趣"]),
}

# 反向表:synonym/canonical -> (canonical, cardinality)
_REVERSE: Dict[str, Tuple[str, str]] = {}
for _canon, (_card, _syns) in _PREDICATES.items():
    _REVERSE[_canon] = (_canon, _card)
    for _s in _syns:
        _REVERSE[_s] = (_canon, _card)


def normalize_predicate(raw: str) -> Tuple[str, str]:
    """原始谓词 → (canonical, cardinality)。

    未知谓词原样返回,基数默认 multi(安全:不会误杀已有事实)。
    """
    key = (raw or "").strip()
    if key in _REVERSE:
        return _REVERSE[key]
    return key, CARDINALITY_MULTI


# ── scope(场景)受控词表 ────────────────────────────────────
# scope 同样是 LLM 自由推断的,不归一会"工作"/"上班"漂移 → 同场景冲突漏判。
# canonical -> [synonyms]
_SCOPES: Dict[str, list] = {
    "工作": ["上班", "工作场景", "职场", "公司", "办公"],
    "业余": ["私下", "个人", "闲暇", "生活", "日常"],
    "学习": ["学校", "上学", "学业"],
}

_SCOPE_REVERSE: Dict[str, str] = {}
for _canon, _syns in _SCOPES.items():
    _SCOPE_REVERSE[_canon] = _canon
    for _s in _syns:
        _SCOPE_REVERSE[_s] = _canon


def normalize_scope(raw: str) -> str:
    """原始 scope → canonical。空保持空;未知原样返回(去空白)。"""
    key = (raw or "").strip()
    if not key:
        return ""
    return _SCOPE_REVERSE.get(key, key)


# ── 来源权威等级 ────────────────────────────────────────────
# 谁能取代谁:低权威的事实不能 supersede 高权威的事实(防 LLM 推断抹掉用户硬约束)。
_AUTHORITY: Dict[str, int] = {
    "user_explicit": 3,   # 用户显式声明 / /remember
    "user_stated": 2,     # 用户对话中陈述
    "tool": 1,            # 工具/外部结果
    "inferred": 0,        # LLM 推断 / assistant 自述
}
DEFAULT_SOURCE_TYPE = "user_stated"


def authority_of(source_type: str) -> int:
    """source_type → 权威等级整数。未知来源按最低(0)处理。"""
    return _AUTHORITY.get(source_type or "", 0)
