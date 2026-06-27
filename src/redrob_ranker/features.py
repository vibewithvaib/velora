"""Feature derivation from candidate profiles (Stage 0, offline).

Everything here is deterministic structured processing — no embeddings, no
LLMs. The outputs are flat numeric features consumed by the online scoring
stages, plus role-classification helpers reused by the gates and trust checks.

Design rule: career history (what they DID) outweighs profile summary and the
skills array (what they CLAIM).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

from .lexicons import Lexicons
from .models import Candidate, Role

logger = logging.getLogger(__name__)

WORK_MODE_CODES = {"onsite": 0, "hybrid": 1, "remote": 2, "flexible": 3}
CITY_TIER_CODES = {"primary": 0, "welcome": 1, "india_relocate": 2, "india_fixed": 3, "abroad": 4}


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleTags:
    """Per-role boolean classification used across features, gates and trust."""

    is_ai: bool
    is_research: bool
    is_consulting: bool
    is_product: bool
    is_coding: bool
    is_llm_wrapper_flavored: bool
    has_production_language: bool
    has_retrieval_evidence: bool


def classify_role(role: Role, lex: Lexicons) -> RoleTags:
    title = role.title.lower()
    text = role.text.lower()
    company = role.company.lower()
    industry = role.industry.lower()

    is_research = (
        lex.present("research_titles", title)
        or lex.present("research_orgs", company)
    )
    is_consulting = (
        lex.present("consulting_firms", company)
        or lex.present("consulting_industries", industry)
    )
    is_product = not is_research and not is_consulting

    is_non_coding_title = lex.present("non_coding_titles", title) or lex.present(
        "non_engineering_titles", title
    )
    has_hands_on = lex.present("hands_on_markers", text)
    # A "Director" who's clearly still building counts as coding; a pure
    # architect/manager role with no hands-on markers does not.
    is_coding = has_hands_on or not is_non_coding_title
    if lex.present("non_engineering_titles", title):
        is_coding = False  # marketing/sales/HR never count as engineering roles

    return RoleTags(
        is_ai=lex.present("ai_ml_terms", text),
        is_research=is_research,
        is_consulting=is_consulting,
        is_product=is_product,
        is_coding=is_coding,
        is_llm_wrapper_flavored=lex.present("llm_wrapper_terms", text),
        has_production_language=lex.present("production_language", text),
        has_retrieval_evidence=lex.present("retrieval_evidence", text),
    )


# ---------------------------------------------------------------------------
# Date / duration math
# ---------------------------------------------------------------------------

def months_between(start: Optional[dt.date], end: Optional[dt.date]) -> Optional[int]:
    if start is None or end is None:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month)


def effective_duration_months(role: Role, ref_date: dt.date) -> int:
    """Best estimate of a role's duration, preferring date math over the
    stated duration_months (which honeypots fabricate)."""
    end = role.end_date or (ref_date if role.is_current else None)
    span = months_between(role.start_date, end)
    if span is not None and span >= 0:
        return span
    return max(role.duration_months, 0)


def union_months(roles: list[Role], ref_date: dt.date) -> int:
    """Total months covered by the union of role intervals (overlaps merged),
    so parallel roles don't double count."""
    intervals: list[tuple[dt.date, dt.date]] = []
    for role in roles:
        start = role.start_date
        end = role.end_date or (ref_date if role.is_current else None)
        if start is None or end is None or end < start:
            continue
        intervals.append((start, min(end, ref_date)))
    if not intervals:
        return sum(max(r.duration_months, 0) for r in roles)
    intervals.sort()
    merged_months = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_months += months_between(cur_start, cur_end) or 0
            cur_start, cur_end = start, end
    merged_months += months_between(cur_start, cur_end) or 0
    return merged_months


def recency_weight(role: Role, ref_date: dt.date, half_life_years: float, floor: float) -> float:
    """Exponential decay on years since the role ended (current roles = 1.0)."""
    if role.is_current or role.end_date is None:
        return 1.0
    years_ago = max((ref_date - role.end_date).days / 365.25, 0.0)
    return max(0.5 ** (years_ago / half_life_years), floor)


def months_since_last_coding_role(
    roles: list[Role], tags: list[RoleTags], ref_date: dt.date
) -> float:
    """Months since the candidate last held a hands-on engineering role.

    0.0 when a current role is hands-on; large when they've moved into
    pure architecture/management (a JD disqualifier past 18 months).
    """
    best: Optional[float] = None
    for role, tag in zip(roles, tags):
        if not tag.is_coding:
            continue
        if role.is_current:
            return 0.0
        end = role.end_date
        if end is None:
            continue
        months = max((ref_date - end).days / 30.44, 0.0)
        best = months if best is None else min(best, months)
    if best is None:
        return 999.0  # never held a coding role
    return best


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

def _kw_density(count: int, saturation: int) -> float:
    """Squash a keyword count into 0-1 with diminishing returns."""
    return min(count / max(saturation, 1), 1.0)


def city_tier(candidate: Candidate, lex: Lexicons) -> str:
    location = candidate.location.lower()
    country = candidate.country.lower()
    in_india = "india" in country
    if lex.present("cities_primary", location):
        return "primary"
    if in_india and lex.present("cities_welcome", location):
        return "welcome"
    if in_india:
        return "india_relocate" if candidate.signals.willing_to_relocate else "india_fixed"
    return "abroad"


def derive_features(
    candidate: Candidate,
    tags: list[RoleTags],
    lex: Lexicons,
    ref_date: dt.date,
    cfg: dict,
) -> dict[str, float]:
    """Flat numeric feature dict for one candidate. All values are floats so
    they stack into numpy arrays in the artifact store."""
    roles = candidate.career_history
    sig = candidate.signals
    half_life = float(cfg["evidence"]["recency_half_life_years"])
    floor = float(cfg["evidence"]["min_role_weight"])

    durations = [effective_duration_months(r, ref_date) for r in roles]
    weights = [
        d * recency_weight(r, ref_date, half_life, floor)
        for r, d in zip(roles, durations)
    ]
    total_w = sum(weights) or 1.0

    career_months = union_months(roles, ref_date)
    ai_months = sum(d for d, t in zip(durations, tags) if t.is_ai)
    ai_product_months = sum(
        d for d, t in zip(durations, tags) if t.is_ai and t.is_product
    )
    product_months = sum(d for d, t in zip(durations, tags) if t.is_product)
    consulting_months = sum(d for d, t in zip(durations, tags) if t.is_consulting)
    research_months = sum(d for d, t in zip(durations, tags) if t.is_research)

    career_text = candidate.career_text.lower()
    profile_text = candidate.profile_text.lower()

    # Recency+duration weighted presence of production / retrieval evidence:
    # a 3-year recent lead role counts far more than a 2014 internship.
    prod_w = sum(w for w, t in zip(weights, tags) if t.has_production_language) / total_w
    retr_w = sum(w for w, t in zip(weights, tags) if t.has_retrieval_evidence) / total_w
    retr_product_w = (
        sum(
            w
            for w, t in zip(weights, tags)
            if t.has_retrieval_evidence and t.is_product
        )
        / total_w
    )

    completed = [
        (r, d) for r, d in zip(roles, durations) if not r.is_current and d > 0
    ]
    short_stints = sum(1 for _, d in completed if d < 18)
    avg_tenure = (
        sum(d for _, d in completed) / len(completed) if completed else float(
            durations[0] if durations else 0
        )
    )

    # Pre-LLM-era ML production work: an AI-flavored coding role that STARTED
    # before the cutoff rescues the LangChain-wrapper disqualifier (per JD).
    cutoff_year = int(cfg["gates"]["pre_llm_cutoff_year"])
    pre_llm_ml = any(
        t.is_ai
        and t.is_coding
        and r.start_date is not None
        and r.start_date.year < cutoff_year
        for r, t in zip(roles, tags)
    )

    # JD-relevant assessment scores (credibility, not claims).
    matched_scores = [
        score
        for name, score in sig.skill_assessment_scores.items()
        if lex.present("jd_skill_keys", name.lower())
    ]
    assess_mean = sum(matched_scores) / len(matched_scores) if matched_scores else -1.0

    days_since_active = (
        (ref_date - sig.last_active_date).days if sig.last_active_date else 9999
    )

    return {
        # --- career structure ---
        "yoe_stated": candidate.years_of_experience,
        "career_months": float(career_months),
        "ai_months": float(ai_months),
        "ai_product_months": float(ai_product_months),
        "product_months": float(product_months),
        "consulting_months": float(consulting_months),
        "research_months": float(research_months),
        "n_roles": float(len(roles)),
        "avg_tenure_months": float(avg_tenure),
        "short_stint_count": float(short_stints),
        "months_since_code": months_since_last_coding_role(roles, tags, ref_date),
        "pre_llm_ml": float(pre_llm_ml),
        # --- text evidence (career history; recency/duration weighted) ---
        "production_kw": prod_w,
        "retrieval_kw": retr_w,
        "retrieval_kw_product": retr_product_w,
        "evaluation_kw": _kw_density(lex.count("evaluation_language", career_text), 3),
        "culture_kw": _kw_density(
            lex.count("culture_markers", career_text + " " + profile_text), 4
        ),
        "nlp_ir_kw": _kw_density(lex.count("nlp_ir_terms", career_text), 4),
        "cv_kw": _kw_density(lex.count("cv_speech_robotics", career_text), 4),
        "llm_wrapper_kw": _kw_density(lex.count("llm_wrapper_terms", career_text), 3),
        "bonus_count": float(len(lex.bonus_hits(career_text))),
        # --- behavioral signals (raw; normalized online so weights stay tunable) ---
        "completeness": sig.profile_completeness_score,
        "days_since_active": float(days_since_active),
        "open_to_work": float(sig.open_to_work_flag),
        "views_30d": float(sig.profile_views_received_30d),
        "apps_30d": float(sig.applications_submitted_30d),
        "response_rate": sig.recruiter_response_rate,
        "response_time_h": sig.avg_response_time_hours,
        "assess_jd_mean": assess_mean,
        "connections": float(sig.connection_count),
        "endorsements": float(sig.endorsements_received),
        "notice_days": float(sig.notice_period_days),
        "salary_min": sig.expected_salary_min,
        "salary_max": sig.expected_salary_max,
        "work_mode_code": float(WORK_MODE_CODES.get(sig.preferred_work_mode, 3)),
        "relocate": float(sig.willing_to_relocate),
        "github_score": sig.github_activity_score,
        "search_30d": float(sig.search_appearance_30d),
        "saved_30d": float(sig.saved_by_recruiters_30d),
        "interview_rate": sig.interview_completion_rate,
        "offer_rate": sig.offer_acceptance_rate,
        "verified_email": float(sig.verified_email),
        "verified_phone": float(sig.verified_phone),
        "linkedin": float(sig.linkedin_connected),
        # --- logistics ---
        "city_tier_code": float(CITY_TIER_CODES[city_tier(candidate, lex)]),
    }
