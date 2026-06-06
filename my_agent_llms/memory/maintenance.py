"""一次性记忆维护:清理被旧规则误固化的 seed hard_constraint 卡。

判据:source=seed_promoted 的 hard_constraint,但在新规则下 classify_content_type
已不再判为 hard_constraint(即"非自指的祈使指令")→ forget。
非 seed 来源不动(用户显式/KG/实绩毕业更可信)。
"""
from my_agent_llms.memory.playbook.card import L0Source, L0Type, classify_content_type


def purge_misclassified_seed_cards(store) -> int:
    """返回被 forget 的卡数量。"""
    purged = 0
    for card in list(store.all_active()):
        if card.source != L0Source.SEED_PROMOTED:
            continue
        if card.type != L0Type.HARD_CONSTRAINT:
            continue
        if classify_content_type(card.content) != L0Type.HARD_CONSTRAINT:
            card.forget()
            store.add(card)   # upsert by id — persists the forgotten lifecycle
            purged += 1
    return purged
