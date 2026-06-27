"""Ranking metrics for local validation against a golden set.

Implements the composite from the submission spec:

  composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

`relevance` maps candidate_id -> graded relevance (tier). "Relevant" for
P@K/MAP means tier >= 3, per the spec's P@10 definition.
"""

from __future__ import annotations

import math


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: list[str], relevance: dict[str, float], k: int) -> float:
    gains = [relevance.get(cid, 0.0) for cid in ranking]
    ideal = sorted(relevance.values(), reverse=True)
    denom = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / denom if denom > 0 else 0.0


def precision_at_k(
    ranking: list[str], relevance: dict[str, float], k: int, threshold: float = 3.0
) -> float:
    hits = sum(1 for cid in ranking[:k] if relevance.get(cid, 0.0) >= threshold)
    return hits / k if k else 0.0


def average_precision(
    ranking: list[str], relevance: dict[str, float], threshold: float = 3.0
) -> float:
    relevant_total = sum(1 for v in relevance.values() if v >= threshold)
    if relevant_total == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, cid in enumerate(ranking, start=1):
        if relevance.get(cid, 0.0) >= threshold:
            hits += 1
            ap += hits / i
    return ap / min(relevant_total, len(ranking))


def composite(ranking: list[str], relevance: dict[str, float]) -> dict[str, float]:
    scores = {
        "ndcg@10": ndcg_at_k(ranking, relevance, 10),
        "ndcg@50": ndcg_at_k(ranking, relevance, 50),
        "map": average_precision(ranking, relevance),
        "p@10": precision_at_k(ranking, relevance, 10),
        "p@5": precision_at_k(ranking, relevance, 5),
    }
    scores["composite"] = (
        0.50 * scores["ndcg@10"]
        + 0.30 * scores["ndcg@50"]
        + 0.15 * scores["map"]
        + 0.05 * scores["p@10"]
    )
    return scores
