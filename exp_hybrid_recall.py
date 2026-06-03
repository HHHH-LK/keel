"""对比实验(离线·机制隔离):KG 检索 graph-only vs graph+keyword(RRF)。

背景:.env 的 LLM_API_KEY 已失效(401),真实 LLM 实体抽取 + 真实 embedding
都跑不了。所以这里把唯一变量"实体抽取成功率 p"显式扫一遍,隔离机制收益,
避免挑一个对新方法有利的噪声水平。

设定:
- 一组事实(subject 统一存为英文 "user" —— 这是中文场景最常见的 canonical 不匹配)
- 一组自然语言追问。每条 query 都含目标词,但实体抽取不一定能挑成实体。
- p = 抽取把 query 里的目标实体正确识别的概率。p<1 模拟真实抽取失败。
- graph-only 依赖抽取;keyword 用整条 query、不依赖抽取 → 扫 p 隔离 keyword 兜底价值。

指标:hit@3(目标对象名是否出现在召回的前 3 条事实里),按 query 平均,
p<1 的随机情形对多个随机种子取平均。

注:语义路径需真 embedding 端点,离线 HashEmbedding 非语义、不计入头条。
"""
import random
from datetime import datetime

from my_agent_llms.memory.kg import (
    KGStore,
    KnowledgeGraphConflictDetector,
    Relation,
)

# (predicate, object_type, object_name)
FACTS = [
    ("使用", "TECH", "Python"),
    ("使用", "TECH", "VSCode"),
    ("住在", "PLACE", "上海"),
    ("职业是", "ROLE", "后端工程师"),
    ("养了", "ITEM", "猫"),
    ("喜欢喝", "ITEM", "美式咖啡"),
    ("在读", "ITEM", "三体"),
    ("使用", "TECH", "MacBook"),
    ("过敏", "ITEM", "花生"),
    ("母语是", "ITEM", "中文"),
]

# A 组:token-present —— 目标词字面出现在 query 里(keyword 能兜底)
# (query, 目标对象名, 抽取成功时返回的 canonical 实体)
QUERIES_A = [
    ("平时用 Python 多吗", "Python", "Python"),
    ("编辑器还是 VSCode 吧", "VSCode", "VSCode"),
    ("现在还住在上海吗", "上海", "上海"),
    ("你是做后端工程师的对吧", "后端工程师", "后端工程师"),
    ("家里那只猫还好吗", "猫", "猫"),
    ("还是喝美式咖啡吗", "美式咖啡", "美式咖啡"),
    ("三体读完了吗", "三体", "三体"),
    ("还在用 MacBook 吗", "MacBook", "MacBook"),
    ("对花生还过敏吗", "花生", "花生"),
    ("母语是中文对吧", "中文", "中文"),
]

# B 组:paraphrase —— 改述,与事实串无字面 token 重叠(keyword 无能为力,需语义)
QUERIES_B = [
    ("我主要写后端那门蛇形语言", "Python", "Python"),
    ("我现在常驻的那座城市", "上海", "上海"),
    ("我每天早上离不开那一杯", "美式咖啡", "美式咖啡"),
    ("我最近在追的那部科幻小说", "三体", "三体"),
    ("家里那个毛茸茸的小动物", "猫", "猫"),
]


def build_store():
    store = KGStore()
    uid = store.get_or_create_entity("PERSON", "user")
    for predicate, otype, oname in FACTS:
        oid = store.get_or_create_entity(otype, oname)
        store.add_relation(
            Relation(
                id=f"r_{oname}",
                subject_id=uid,
                predicate=predicate,
                object_id=oid,
                valid_from=datetime.now(),
                source_item_id=f"item_{oname}",
            )
        )
    return store


def graph_only_facts(detector, query, max_facts=3):
    """复刻旧 query_facts:只走图遍历路径。"""
    active = detector.store.all_relations(only_active=True)
    nl_map = {r.id: detector.store.relation_to_nl(r) for r in active}
    ids = detector._rank_by_graph(query, nl_map)
    return [nl_map[i] for i in ids[:max_facts] if i in nl_map]


def hit(facts, target):
    return any(target in f for f in facts)


def run(queries, p, seed):
    store = build_store()
    detector = KnowledgeGraphConflictDetector(llm=None, store=store)
    rng = random.Random(seed)

    graph_hits = 0
    hybrid_hits = 0
    for query, target, canonical in queries:
        # 模拟实体抽取:成功(prob p)→ 返回 canonical 实体;失败 → 返回非匹配词
        if rng.random() < p:
            extracted = [canonical]
        else:
            extracted = ["你"]  # 非匹配:KG 里没有 "你" 这个实体
        detector._extract_query_entities = lambda q, _e=extracted: _e

        g = graph_only_facts(detector, query)
        h = detector.query_facts(query, max_facts=3)
        graph_hits += hit(g, target)
        hybrid_hits += hit(h, target)

    n = len(queries)
    return graph_hits / n, hybrid_hits / n


def report(name, queries, note=""):
    seeds = list(range(20))
    print(f"\n【{name}】{len(queries)} 条查询 · 每个 p 取 {len(seeds)} 个种子平均  {note}")
    print(f"{'抽取成功率 p':>10} | {'graph-only':>11} | {'graph+keyword':>14} | {'提升':>8}")
    print("-" * 56)
    for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        gs, hs = [], []
        for s in seeds:
            g, h = run(queries, p, s)
            gs.append(g)
            hs.append(h)
        g_avg = sum(gs) / len(gs)
        h_avg = sum(hs) / len(hs)
        print(f"{p:>10.1f} | {g_avg:>10.1%} | {h_avg:>13.1%} | {h_avg - g_avg:>+7.1%}")


def main():
    print(f"事实库 {len(FACTS)} 条 | hit@3 | subject 统一存为 'user'(模拟中文 canonical 不匹配)")
    report("A 组 token-present", QUERIES_A, "目标词字面出现 → keyword 可兜底")
    report("B 组 paraphrase", QUERIES_B, "改述无字面重叠 → keyword 无能为力,需语义(离线测不了)")


if __name__ == "__main__":
    main()
