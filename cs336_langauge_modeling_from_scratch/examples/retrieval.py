"""BM25 retrieval, reciprocal-rank fusion and ranking metrics on an original fixed corpus.

No learned embeddings, ANN index, or language-generation call is used here.
"""
from collections import Counter
import json
import math
import operator
import re


def terms(text):
    # ASCII words/numbers and individual CJK characters; an explicit toy convention.
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        documents = list(documents)
        if not documents or len({key for key, _ in documents}) != len(documents) or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("unique nonempty corpus, positive k1 and b in [0,1] required")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(terms(text)) for _, text in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths)
        if self.average == 0:
            raise ValueError("corpus must contain searchable terms")
        self.df = Counter(term for c in self.counts for term in c)

    def search(self, query, k=3):
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        result, n = [], len(self.documents)
        query_terms = set(terms(query))
        for index, counts in enumerate(self.counts):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log1p((n - df + .5) / (df + .5))
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                score += idf * tf * (self.k1 + 1) / denominator
            if score > 0:
                result.append((self.documents[index][0], score))
        return sorted(result, key=lambda x: (-x[1], x[0]))[:k]


def reciprocal_rank_fusion(rankings, k=60):
    """Merge ranked ID lists by sum of 1/(k + rank); uses ranks only, so score scales never mix."""
    if not math.isfinite(k) or k < 0:
        raise ValueError('finite nonnegative RRF rank offset required')
    scores = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc] += 1 / (k + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def ranking_metrics(ranked_ids, relevant, k):
    relevant = set(relevant)
    if type(k) is not int or k < 1 or not relevant:
        raise ValueError("positive k and nonempty relevant set required")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    relevant = set(relevant)
    first = next((i + 1 for i, doc in enumerate(ranked) if doc in relevant), None)
    return dict(recall=len(set(ranked) & relevant) / len(relevant), mrr=0.0 if first is None else 1 / first)


def ndcg_at_k(ranked_ids, gains, k):
    """Graded relevance: DCG = sum_i (2^g_i - 1) / log2(i + 1) over the top k, divided by the ideal DCG.

    gains maps document id -> finite nonnegative relevance grade (missing ids count as 0). Unlike recall@k,
    a highly relevant document ranked first is worth more than a marginal one ranked first.
    """
    try:
        k = operator.index(k)
        gains = {doc: float(grade) for doc, grade in gains.items()}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("integer k and finite numeric relevance grades required") from error
    if k < 1 or not gains or any(not math.isfinite(g) or g < 0 for g in gains.values()) or max(gains.values()) == 0:
        raise ValueError("positive integer k and finite nonnegative grades with a positive maximum required")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    maximum = max(gains.values())
    def gain(grade):
        # Divide numerator and denominator by 2**maximum; expm1 preserves tiny positive grades.
        return 2.0 ** (grade - maximum) * -math.expm1(-grade * math.log(2))
    dcg = math.fsum(gain(gains.get(doc, 0)) / math.log2(i + 2) for i, doc in enumerate(ranked))
    ideal = sorted(gains.values(), reverse=True)[:k]
    return dcg / math.fsum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal))


def demo():
    corpus = [("rope", "RoPE 使用旋转位置编码，让 query 和 key 的内积与相对位置有关。"),
              ("kv", "KV cache 缓存历史键值，避免逐 token 生成时重复计算过去前缀。"),
              ("adam", "AdamW 保存一阶和二阶动量，并解耦权重衰减。"),
              ("data", "训练数据要过滤、去重，并保留来源和版本。")]
    index = BM25(corpus)
    queries = [("RoPE 相对位置", {"rope"}), ("KV cache 历史键值", {"kv"}), ("AdamW 动量", {"adam"}),
               # A paraphrase with no shared terms: lexical retrieval returns nothing.
               ("如何表示词序", {"rope"})]
    results = []
    for query, relevant in queries:
        ranked = index.search(query)
        results.append(dict(query=query, ranked=ranked, metrics=ranking_metrics([x[0] for x in ranked] or ["none"], relevant, 3)))
    fused = reciprocal_rank_fusion([["kv", "rope", "data"], ["rope", "adam"]])
    return dict(scope="fixed four-document lexical retrieval demo; not an end-to-end RAG evaluation",
                queries=results, rrf_example=fused)


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
