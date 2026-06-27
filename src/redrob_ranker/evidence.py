"""Evidence-document construction and embedding pooling (Stage 0).

We encode the JD as INTENT (facet sentences), and each candidate as EVIDENCE:
every career role encoded separately, pooled with duration x recency weights,
plus a modestly weighted headline+summary vector. The skills array is
deliberately excluded — it is the planted trap; skills act only as weak
corroboration in the feature layer.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np

from .features import effective_duration_months, recency_weight
from .models import Candidate

logger = logging.getLogger(__name__)


def role_texts(candidate: Candidate) -> list[str]:
    return [r.text for r in candidate.career_history if r.text.strip()]


def role_weights(candidate: Candidate, ref_date: dt.date, cfg: dict) -> np.ndarray:
    """Duration x recency weight per role (normalized to sum 1)."""
    ev = cfg["evidence"]
    weights = []
    for role in candidate.career_history:
        if not role.text.strip():
            continue
        months = max(effective_duration_months(role, ref_date), 1)
        rec = recency_weight(
            role, ref_date, float(ev["recency_half_life_years"]), float(ev["min_role_weight"])
        )
        weights.append(months * rec)
    arr = np.asarray(weights, dtype=np.float32)
    if arr.size == 0 or arr.sum() == 0:
        return arr
    return arr / arr.sum()


def pool_candidate_vector(
    role_vecs: np.ndarray,
    weights: np.ndarray,
    profile_vec: np.ndarray | None,
    profile_weight: float,
) -> np.ndarray:
    """Weighted pool of role vectors + profile vector, re-normalized to unit length."""
    if role_vecs.size == 0:
        pooled = profile_vec if profile_vec is not None else np.zeros(384, np.float32)
    else:
        pooled = (weights[:, None] * role_vecs).sum(axis=0)
        if profile_vec is not None:
            pooled = (1.0 - profile_weight) * pooled + profile_weight * profile_vec
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32)


def semantic_role_features(
    role_vecs: np.ndarray,
    weights: np.ndarray,
    profile_vec: np.ndarray | None,
    jd_vecs: np.ndarray,
    core_facet_indices: list[int],
) -> dict[str, float]:
    """Scalar semantic features stored in the artifact (JD is fixed, so these
    are computable offline):

      role_max_sim   — best single role vs any core facet (peak evidence)
      role_wmean_sim — duration/recency-weighted mean of per-role best-facet
                       sims (sustained evidence)
      profile_sim    — headline+summary vs core facets (claims, low trust)
    """
    core = jd_vecs[core_facet_indices]  # (F, d)
    if role_vecs.size == 0:
        role_max = role_wmean = 0.0
    else:
        sims = role_vecs @ core.T            # (R, F)
        best_per_role = sims.max(axis=1)     # (R,)
        role_max = float(best_per_role.max())
        role_wmean = float((weights * best_per_role).sum())
    profile_sim = float((profile_vec @ core.T).max()) if profile_vec is not None else 0.0
    return {
        "role_max_sim": role_max,
        "role_wmean_sim": role_wmean,
        "profile_sim": profile_sim,
    }
