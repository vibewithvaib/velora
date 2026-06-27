"""Stages 1-7: the online ranking pipeline (<= 5 min, CPU, no network).

Loads precomputed arrays, applies hard gates, retrieves a semantic shortlist,
scores it with pure NumPy, fuses, ranks, generates grounded reasoning and
writes the validated submission CSV. The heaviest operation is one (N, 384) x
(384, F) matrix product — seconds, not minutes.

This module imports neither torch nor sentence-transformers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .artifacts import Artifacts, file_sha1, load_artifacts
from .behavioral import score_behavioral
from .core_fit import score_core_fit
from .fusion import fuse, rank_top_k
from .reasoning import build_concerns, generate_reasoning
from .retrieval import facet_similarity, shortlist_indices
from .trust import trust_multiplier
from .validator import validate_submission
from .writer import write_submission

logger = logging.getLogger(__name__)

# Factors eligible to lead a reasoning line, with their score arrays' names.
_FACTOR_KEYS = ["technical", "experience", "shipping", "availability", "credibility", "logistics"]


def _trust_multipliers(mismatches: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Vectorized trust tier mapping; also returns tier name per row."""
    mult = np.empty(mismatches.shape, dtype=np.float64)
    tiers: list[str] = []
    for i, count in enumerate(mismatches):
        m, tier = trust_multiplier(int(count), cfg)
        mult[i] = m
        tiers.append(tier)
    return mult, tiers


def run_rank(
    candidates_path: str | Path,
    cfg: dict,
    artifacts_base: str | Path | None = None,
    out_path: str | Path = "submission.csv",
) -> Path:
    t0 = time.time()
    artifacts_base = Path(artifacts_base or cfg["artifacts_dir"])
    input_sha = file_sha1(candidates_path)
    art: Artifacts = load_artifacts(artifacts_base, input_sha)
    n = len(art.ids)
    logger.info("Loaded artifacts: %d candidates (%.2fs)", n, time.time() - t0)

    # ---- Stage 1: hard gates (kill switch, not a penalty) -----------------
    alive = ~art.honeypot & ~art.disqualified
    n_alive = int(alive.sum())
    logger.info(
        "Stage 1 gates: %d alive / %d (removed %d honeypots, %d disqualified)",
        n_alive, n, int(art.honeypot.sum()), int(art.disqualified.sum()),
    )
    alive_idx = np.flatnonzero(alive)

    # ---- Stage 2: semantic retrieval shortlist ------------------------------
    r = cfg["retrieval"]
    sim = facet_similarity(art.emb[alive_idx], art.jd_vecs, agg=r["facet_agg"])
    local_short = shortlist_indices(sim, int(r["shortlist_size"]))
    short_idx = alive_idx[local_short]  # global row indices of the shortlist
    logger.info("Stage 2 retrieval: shortlist %d (%.2fs)", len(short_idx), time.time() - t0)

    feats = {name: arr[short_idx] for name, arr in art.features.items()}
    short_ids = [art.ids[i] for i in short_idx]

    # ---- Stages 3-5: scoring -------------------------------------------------
    core = score_core_fit(feats, cfg)
    beh = score_behavioral(feats, cfg)
    trust_mult, trust_tiers = _trust_multipliers(art.trust_mismatches[short_idx], cfg)

    # ---- Stage 6: fusion + ranking -------------------------------------------
    final = fuse(core.core, beh.behavioral, beh.avail_mult, trust_mult, cfg)
    ranked = rank_top_k(short_ids, final, beh.behavioral, cfg)
    logger.info("Stage 6 fusion: top %d selected (%.2fs)", len(ranked), time.time() - t0)

    # ---- Stage 7: grounded reasoning ------------------------------------------
    pos = {cid: i for i, cid in enumerate(short_ids)}
    factor_arrays = {
        "technical": core.technical,
        "experience": core.experience,
        "shipping": core.shipping,
        "availability": beh.availability,
        "credibility": beh.credibility,
        "logistics": core.logistics,
    }
    rows = []
    for item in ranked:
        i = pos[item.candidate_id]
        factor_ranking = sorted(
            _FACTOR_KEYS, key=lambda k: -float(factor_arrays[k][i])
        )
        facts = art.facts[item.candidate_id]
        cand_reasons = art.reasons.get(item.candidate_id, {})
        concerns = build_concerns(
            facts, trust_tiers[i], cand_reasons.get("trust", [])
        )
        reasoning = generate_reasoning(
            item.candidate_id,
            facts,
            factor_ranking,
            concerns,
            item.rank,
            top_k=int(cfg["output"]["top_k"]),
        )
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "rank": item.rank,
                "score": f"{item.score:.{int(cfg['output']['score_decimals'])}f}",
                "reasoning": reasoning,
            }
        )

    out = write_submission(rows, out_path)

    # ---- final self-validation against the spec --------------------------------
    errors = validate_submission(out, known_ids=set(art.ids), top_k=int(cfg["output"]["top_k"]))
    if errors:
        for err in errors:
            logger.error("VALIDATION: %s", err)
        raise RuntimeError(f"Submission failed self-validation with {len(errors)} errors")
    logger.info("Submission validated OK. Total rank time %.2fs", time.time() - t0)
    return out
