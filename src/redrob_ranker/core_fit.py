"""Stage 3: core fit scoring (technical-gated), vectorized over the shortlist.

  core_raw = 0.45*T + 0.20*S + 0.20*E + 0.05*C + 0.10*L
  core     = core_raw * clamp(T / tau, 0, 1) ** beta   # must-have GATE
  core    += capped bonus (nice-to-haves break ties, never rescue)

Weak technical substance is multiplicatively crushed — a candidate cannot buy
back the JD's must-haves with bonus skills or behavioral shine.

All inputs are numpy arrays from the feature table; `cfg` is config.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

FeatureTable = dict[str, np.ndarray]


def piecewise(x: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """Piecewise-linear curve through config-declared (x, y) points."""
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    return np.interp(x, xs, ys)


def calibrate(sim: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((sim - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


@dataclass
class CoreFitResult:
    core: np.ndarray
    technical: np.ndarray
    shipping: np.ndarray
    experience: np.ndarray
    culture: np.ndarray
    logistics: np.ndarray
    gate: np.ndarray
    bonus: np.ndarray


def _logistics_core(feats: FeatureTable, cfg: dict) -> np.ndarray:
    """Location + notice + work-mode fit (the JD's logistics section)."""
    city_scores = cfg["logistics"]["city_scores"]
    tier_lookup = np.asarray(
        [
            city_scores["primary"],
            city_scores["welcome"],
            city_scores["india_relocate"],
            city_scores["india_fixed"],
            city_scores["abroad"],
        ]
    )
    city = tier_lookup[feats["city_tier_code"].astype(int).clip(0, 4)]

    bands = cfg["behavioral"]["availability"]["notice_bands"]
    notice = np.full_like(city, bands[-1][1])
    for limit, score in reversed(bands):
        notice = np.where(feats["notice_days"] <= limit, score, notice)

    mode_scores = cfg["logistics"]["work_mode_scores"]
    mode_lookup = np.asarray(
        [mode_scores["onsite"], mode_scores["hybrid"], mode_scores["remote"], mode_scores["flexible"]]
    )
    mode = mode_lookup[feats["work_mode_code"].astype(int).clip(0, 3)]

    return 0.60 * city + 0.25 * notice + 0.15 * mode


def score_core_fit(feats: FeatureTable, cfg: dict) -> CoreFitResult:
    cf = cfg["core_fit"]
    cal = cfg["retrieval"]["sim_calibration"]

    # --- T: technical fit — the spine -------------------------------------
    sem_cfg = cf["semantic_blend"]
    sem = sem_cfg["wmean"] * calibrate(
        feats["role_wmean_sim"], cal["lo"], cal["hi"]
    ) + sem_cfg["best_role"] * calibrate(feats["role_max_sim"], cal["lo"], cal["hi"])

    # Lexical built-it evidence: retrieval/ranking/search work in real roles,
    # with product-company context weighted highest. sqrt softens the fact
    # that these are fractions of weighted role-mass.
    evidence = np.sqrt(
        np.clip(0.7 * feats["retrieval_kw_product"] + 0.3 * feats["retrieval_kw"], 0, 1)
    )

    ai_years = feats["ai_product_months"] / 12.0
    tenure = piecewise(ai_years, cf["applied_ai_years_curve"]["points"])

    tb = cf["technical_blend"]
    technical = (
        tb["semantic"] * sem + tb["evidence_keywords"] * evidence + tb["applied_ai_tenure"] * tenure
    )

    # --- S: shipping — language of production and scale --------------------
    shipping = 0.75 * np.sqrt(np.clip(feats["production_kw"], 0, 1)) + 0.25 * feats[
        "evaluation_kw"
    ]

    # --- E: experience — smooth curve peaking at the JD's 5-9 band ---------
    career_years = feats["career_months"] / 12.0
    years = np.where(career_years > 0, career_years, feats["yoe_stated"])
    eb = cf["experience_blend"]
    experience = eb["total_years"] * piecewise(
        years, cf["experience_curve"]["points"]
    ) + eb["applied_ai_years"] * piecewise(ai_years, cf["applied_ai_years_curve"]["points"])

    # --- C: culture (soft, low weight) / L: logistics ----------------------
    culture = np.clip(feats["culture_kw"], 0, 1)
    logistics = _logistics_core(feats, cfg)

    w = cf["weights"]
    core_raw = (
        w["technical"] * technical
        + w["shipping"] * shipping
        + w["experience"] * experience
        + w["culture"] * culture
        + w["logistics"] * logistics
    )

    # --- must-have gate + capped bonus -------------------------------------
    gate = np.clip(technical / float(cf["gate"]["tau_tech"]), 0.0, 1.0) ** float(
        cf["gate"]["beta"]
    )
    bonus = np.minimum(feats["bonus_count"] * float(cf["bonus_per_hit"]), float(cf["bonus_cap"]))
    core = core_raw * gate + bonus

    return CoreFitResult(
        core=core,
        technical=technical,
        shipping=shipping,
        experience=experience,
        culture=culture,
        logistics=logistics,
        gate=gate,
        bonus=bonus,
    )
