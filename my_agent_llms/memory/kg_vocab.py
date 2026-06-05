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
#
# 通用智能体策略:重点 curate 单值(它们才 supersede,且有界);
# 多值列了一组常见的,但未知谓词本就默认 multi,漏列无害。
_PREDICATES: Dict[str, Tuple[str, list]] = {
    # ── 单值(functional):一个当前值,会随时间变 → 新值取代旧值 ──
    # 环境 / 技术栈核心
    "主力语言": (CARDINALITY_SINGLE, ["主用语言", "主要语言", "主要用", "主要写", "现在用", "改用"]),
    "操作系统": (CARDINALITY_SINGLE, ["系统", "用的系统", "os", "开发环境系统"]),
    "shell": (CARDINALITY_SINGLE, ["终端", "命令行"]),
    "时区": (CARDINALITY_SINGLE, ["timezone", "所在时区"]),
    # 项目 / 焦点
    "当前项目": (CARDINALITY_SINGLE, ["在做的项目", "手头项目", "正在做的项目", "现在做的项目"]),
    "当前重点": (CARDINALITY_SINGLE, ["当前焦点", "现在的重点", "眼下重点", "这阶段重点"]),
    # 身份
    "现居地": (CARDINALITY_SINGLE, ["住在", "居住", "居住于", "现居", "住", "所在城市"]),
    "现任雇主": (CARDINALITY_SINGLE, ["就职于", "工作于", "任职于", "雇主", "供职于"]),
    "职业": (CARDINALITY_SINGLE, ["职位", "工作是", "角色是", "身份是"]),
    "经验水平": (CARDINALITY_SINGLE, ["资历", "级别", "seniority", "经验"]),
    "婚姻状态": (CARDINALITY_SINGLE, ["婚姻"]),
    "母语": (CARDINALITY_SINGLE, ["第一语言"]),
    "出生地": (CARDINALITY_SINGLE, ["出生于", "生于"]),

    # ── 多值(set):能同时有很多 → 只追加,绝不取代 ──
    "使用": (CARDINALITY_MULTI, ["用", "采用", "在用", "用的是"]),          # 语言/框架/库/工具/编辑器
    "会": (CARDINALITY_MULTI, ["掌握", "会用", "懂", "熟悉"]),
    "喜欢": (CARDINALITY_MULTI, ["偏好", "爱用", "喜爱", "钟爱", "喜好"]),     # 代码风格/约定/库
    "讨厌": (CARDINALITY_MULTI, ["不喜欢", "厌恶", "反感", "避免", "不想用", "不愿用"]),
    "习惯": (CARDINALITY_MULTI, ["约定", "坚持", "规矩", "准则", "总是", "一贯"]),  # 个人开发约定(总用TDD…)
    "维护": (CARDINALITY_MULTI, ["维护着", "负责", "在维护"]),               # repo / 项目
    "关注": (CARDINALITY_MULTI, ["在意", "看重", "重视"]),                  # 领域 / 技术方向
    "兴趣": (CARDINALITY_MULTI, ["爱好", "感兴趣"]),
    "去过": (CARDINALITY_MULTI, ["到过", "去了"]),
    "拥有": (CARDINALITY_MULTI, ["有", "持有"]),
    "过敏": (CARDINALITY_MULTI, ["过敏于"]),
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
    "业余": ["私下", "个人", "闲暇", "生活", "日常", "个人项目"],
    "学习": ["学校", "上学", "学业"],
    "开源": ["开源项目", "oss", "开源贡献"],
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
    "user_promoted": 2,   # 跨项目复现提升:反复印证,等同 user_stated
    "tool": 1,            # 工具/外部结果
    "inferred": 0,        # LLM 推断 / assistant 自述
}
DEFAULT_SOURCE_TYPE = "user_stated"


def authority_of(source_type: str) -> int:
    """source_type → 权威等级整数。未知来源按最低(0)处理。"""
    return _AUTHORITY.get(source_type or "", 0)


# 初始置信度:越权威的来源,事实初始 confidence 越高(复写印证再往上 bump)
_BASE_CONFIDENCE: Dict[str, float] = {
    "user_explicit": 1.0,
    "user_stated": 0.9,
    "user_promoted": 0.9,  # 跨项目复现提升:等同 user_stated
    "tool": 0.8,
    "inferred": 0.7,
}


def base_confidence(source_type: str) -> float:
    """source_type → 初始 confidence。未知来源按 0.7。"""
    return _BASE_CONFIDENCE.get(source_type or "", 0.7)
