"""Stage 2: semantic retrieval shortlist.

One matrix product over the surviving pool, aggregate across JD facets, keep
the top-K. This is what catches the plain-language Tier-5 ("built a product
recommendation engine" lands near "shipped an end-to-end recommender" in
vector space with zero shared keywords) while staying trivially cheap.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def facet_similarity(emb: np.ndarray, jd_vecs: np.ndarray, agg: str = "max") -> np.ndarray:
    """Cosine similarity of each candidate vector to the JD facets, aggregated."""
    sims = emb @ jd_vecs.T  # (N, F); embeddings are L2-normalized
    if agg == "mean":
        return sims.mean(axis=1)
    return sims.max(axis=1)


def shortlist_indices(sim: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k similarities (descending), deterministic."""
    k = min(k, sim.shape[0])
    # argpartition for O(N), then exact sort of the slice; stable order via
    # (-sim, index) so equal sims break by index — fully deterministic.
    part = np.argpartition(-sim, k - 1)[:k]
    order = np.lexsort((part, -sim[part]))
    return part[order]


def calibrate(sim: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rescale raw cosine sims into 0-1 for use inside weighted sums."""
    return np.clip((sim - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
