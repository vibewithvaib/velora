"""Stage 4: behavioral & hireability scoring from the 23 Redrob signals.

Six signal groups -> one behavioral score, and availability ALSO becomes a
multiplier on the final score (the JD: a perfect-on-paper candidate who hasn't
logged in for six months and answers 5% of recruiters is not actually
available — down-weight them).

Missing-value sentinels (github_activity_score = -1, offer_acceptance_rate =
-1, absent salary) mean UNKNOWN and map to neutral, never to worst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

FeatureTable = dict[str, np.ndarray]


def log_cap(x: np.ndarray, cap: float) -> np.ndarray:
    """log1p scale clipped at a config cap, so one viral profile can't dominate."""
    return np.clip(np.log1p(np.clip(x, 0, None)) / np.log1p(cap), 0.0, 1.0)


def _neutral_if_missing(x: np.ndarray, scale: float = 100.0, neutral: float = 0.5) -> np.ndarray:
    """-1 sentinel -> neutral; otherwise rescale to 0-1."""
    return np.where(x < 0, neutral, np.clip(x / scale, 0.0, 1.0))


@dataclass
class BehavioralResult:
    behavioral: np.ndarray
    availability: np.ndarray
    avail_mult: np.ndarray
    credibility: np.ndarray
    market: np.ndarray
    professionalism: np.ndarray
    logistics_compat: np.ndarray
    activity: np.ndarray


def availability_score(feats: FeatureTable, cfg: dict) -> np.ndarray:
    av = cfg["behavioral"]["availability"]
    w = av["weights"]

    recency = 0.5 ** (feats["days_since_active"] / float(av["last_active_half_life_days"]))
    response_time = 1.0 / (1.0 + feats["response_time_h"] / float(av["response_time_scale_hours"]))

    notice = np.full_like(recency, av["notice_bands"][-1][1])
    for limit, score in reversed(av["notice_bands"]):
        notice = np.where(feats["notice_days"] <= limit, score, notice)

    return (
        w["open_to_work"] * feats["open_to_work"]
        + w["recency"] * recency
        + w["response_rate"] * np.clip(feats["response_rate"], 0, 1)
        + w["response_time"] * response_time
        + w["notice_period"] * notice
        + w["interview_completion"] * np.clip(feats["interview_rate"], 0, 1)
    )


def _salary_fit(feats: FeatureTable, cfg: dict) -> np.ndarray:
    """1.0 when expectations sit inside the role's sane band; degrades as the
    expected range exceeds it; neutral when unstated."""
    lo, hi = cfg["behavioral"]["salary_band_lpa"]
    smin, smax = feats["salary_min"], feats["salary_max"]
    width = np.maximum(smax - smin, 1.0)
    over = np.clip((hi - smin) / width, 0.0, 1.0)  # fraction of range below cap
    fit = np.where(smax <= hi, 1.0, over)
    missing = (smin < 0) | (smax <= 0)
    return np.where(missing, float(cfg["behavioral"]["salary_missing_score"]), fit)


def score_behavioral(feats: FeatureTable, cfg: dict) -> BehavioralResult:
    b = cfg["behavioral"]
    caps = b["caps"]

    availability = availability_score(feats, cfg)

    credibility = (
        0.50 * _neutral_if_missing(feats["assess_jd_mean"])
        + 0.30 * _neutral_if_missing(feats["github_score"])
        + 0.20 * log_cap(feats["endorsements"], caps["endorsements_received"])
    )

    market = (
        log_cap(feats["views_30d"], caps["profile_views_received_30d"])
        + log_cap(feats["search_30d"], caps["search_appearance_30d"])
        + log_cap(feats["saved_30d"], caps["saved_by_recruiters_30d"])
    ) / 3.0

    professionalism = (
        0.40 * np.clip(feats["completeness"] / 100.0, 0, 1)
        + 0.15 * feats["verified_email"]
        + 0.15 * feats["verified_phone"]
        + 0.15 * feats["linkedin"]
        + 0.15 * log_cap(feats["connections"], caps["connection_count"])
    )

    mode_scores = cfg["logistics"]["work_mode_scores"]
    mode_lookup = np.asarray(
        [mode_scores["onsite"], mode_scores["hybrid"], mode_scores["remote"], mode_scores["flexible"]]
    )
    logistics_compat = (
        0.35 * mode_lookup[feats["work_mode_code"].astype(int).clip(0, 3)]
        + 0.20 * feats["relocate"]
        + 0.45 * _salary_fit(feats, cfg)
    )

    # Activity: applying is a warmth signal; spray-and-pray is not.
    act = b["activity"]
    apps = feats["apps_30d"]
    ramp_up = np.clip(apps / max(act["ideal_lo"], 1), 0, 1)
    decay = 1.0 - np.clip(
        (apps - act["ideal_hi"]) / max(act["spam_hi"] - act["ideal_hi"], 1), 0, 1
    ) * (1.0 - act["spam_score"])
    activity = np.where(
        apps <= 0,
        act["zero_apps_score"],
        np.where(apps <= act["ideal_hi"], np.maximum(ramp_up, 0.6), decay),
    )

    w = b["weights"]
    behavioral = (
        w["availability"] * availability
        + w["credibility"] * credibility
        + w["market"] * market
        + w["professionalism"] * professionalism
        + w["logistics_compat"] * logistics_compat
        + w["activity"] * activity
    )

    floor = float(b["availability_mult_floor"])
    avail_mult = floor + (1.0 - floor) * np.clip(availability, 0, 1)

    return BehavioralResult(
        behavioral=behavioral,
        availability=availability,
        avail_mult=avail_mult,
        credibility=credibility,
        market=market,
        professionalism=professionalism,
        logistics_compat=logistics_compat,
        activity=activity,
    )
