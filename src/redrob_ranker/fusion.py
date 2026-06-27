"""Stage 6: fusion, ranking and deterministic tie-breaks.

  final = trust_multiplier x avail_mult x (0.75*core + 0.25*behavioral)

Technical substance leads; behavioral is a strong secondary; trust and
availability scale the whole thing. Ties break by behavioral score, then
candidate_id ascending, so the run is fully deterministic (the spec requires
unique ranks even on tied scores).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float


def fuse(
    core: np.ndarray,
    behavioral: np.ndarray,
    avail_mult: np.ndarray,
    trust_mult: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    f = cfg["fusion"]
    return trust_mult * avail_mult * (
        float(f["core_weight"]) * core + float(f["behavioral_weight"]) * behavioral
    )


def rank_top_k(
    candidate_ids: list[str],
    final: np.ndarray,
    behavioral: np.ndarray,
    cfg: dict,
) -> list[RankedCandidate]:
    """Sort descending with deterministic tie-breaks and emit the top K with
    non-increasing rounded scores (a spec auto-reject we will not trip on)."""
    top_k = int(cfg["output"]["top_k"])
    decimals = int(cfg["output"]["score_decimals"])

    order = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-final[i], -behavioral[i], candidate_ids[i]),
    )[:top_k]

    ranked: list[RankedCandidate] = []
    prev_score = None
    for position, idx in enumerate(order, start=1):
        score = round(float(final[idx]), decimals)
        if prev_score is not None and score > prev_score:
            score = prev_score  # guard against rounding inversions
        prev_score = score
        ranked.append(
            RankedCandidate(candidate_id=candidate_ids[idx], rank=position, score=score)
        )
    return ranked
