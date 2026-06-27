"""Stage 1 hard gates: honeypot detection + JD disqualifiers.

Honeypots are removed only on PROVABLE impossibilities (high precision on
purpose — a false kill removes a real candidate). Anything merely odd flows to
the Stage-5 trust multiplier instead.

Disqualifiers implement exactly what the JD says it applies, including the
rescue clauses ("unless prior product-company experience", "unless substantial
pre-LLM-era ML production work").
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from .features import RoleTags, effective_duration_months, months_between, union_months
from .lexicons import Lexicons
from .models import Candidate

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    honeypot: bool = False
    disqualified: bool = False
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Honeypots — impossible profiles
# ---------------------------------------------------------------------------

def check_honeypot(candidate: Candidate, ref_date: dt.date, cfg: dict) -> list[str]:
    """Return a list of impossibility reasons (empty list == not a honeypot)."""
    hp = cfg["gates"]["honeypot"]
    reasons: list[str] = []
    future_limit = ref_date + dt.timedelta(days=int(hp["future_date_slack_days"]))

    for i, role in enumerate(candidate.career_history):
        start, end = role.start_date, role.end_date
        if start is not None and start > future_limit:
            reasons.append(f"role[{i}] starts in the future ({start})")
        if end is not None and end > future_limit:
            reasons.append(f"role[{i}] ends in the future ({end})")
        if start is not None and end is not None:
            span = months_between(start, end)
            if span is not None and span < 0:
                reasons.append(f"role[{i}] ends before it starts")
            elif span is not None and role.duration_months > 0:
                # stated duration wildly exceeding what the dates allow
                diff = role.duration_months - span
                if diff > int(hp["duration_mismatch_months"]) and role.duration_months > span * float(
                    hp["duration_mismatch_ratio"]
                ):
                    reasons.append(
                        f"role[{i}] claims {role.duration_months}mo but dates allow {span}mo"
                    )

    # Career arithmetic vs stated years of experience.
    stated_months = candidate.years_of_experience * 12
    covered = union_months(candidate.career_history, ref_date)
    slack = int(hp["career_vs_stated_slack_months"])
    if covered > 0 and stated_months > covered + int(hp["span_vs_stated_slack_months"]):
        # claims far more experience than their entire dated history can hold
        # (the classic "8 years of experience, first role 3 years ago")
        earliest = min(
            (r.start_date for r in candidate.career_history if r.start_date),
            default=None,
        )
        if earliest is not None:
            total_span = months_between(earliest, ref_date) or 0
            if stated_months > total_span + slack:
                reasons.append(
                    f"claims {candidate.years_of_experience:.0f}y experience but career "
                    f"span is only {total_span / 12:.1f}y"
                )

    # Expert proficiency with zero months used, in bulk.
    expert_zero = sum(
        1
        for s in candidate.skills
        if s.proficiency == "expert" and s.duration_months == 0
    )
    if expert_zero >= int(hp["expert_zero_month_min_count"]):
        reasons.append(f"{expert_zero} 'expert' skills with 0 months of use")

    # Platform-signal impossibility: active before signing up.
    sig = candidate.signals
    if (
        sig.signup_date is not None
        and sig.last_active_date is not None
        and (sig.signup_date - sig.last_active_date).days
        > int(hp["active_before_signup_days"])
    ):
        reasons.append("last_active_date precedes signup_date")

    return reasons


# ---------------------------------------------------------------------------
# JD disqualifiers
# ---------------------------------------------------------------------------

def check_disqualifiers(
    candidate: Candidate,
    tags: list[RoleTags],
    features: dict[str, float],
    lex: Lexicons,
    cfg: dict,
) -> list[str]:
    """Return JD-disqualifier reasons (empty list == passes the gate)."""
    gates_cfg = cfg["gates"]
    reasons: list[str] = []
    roles = candidate.career_history

    if not roles:
        return ["no career history"]

    # 1. Research-only career with no production deployment ("we will not move forward").
    if all(t.is_research for t in tags) and features["production_kw"] == 0.0:
        reasons.append("research-only career, no production deployment")

    # 2. Consulting-only career — rescued by ANY product-company role (JD).
    if all(t.is_consulting for t in tags):
        reasons.append("entire career at IT-services/consulting firms")

    # 3. LangChain-wrapper-only AI experience — rescued by pre-LLM-era ML work.
    ai_months = features["ai_months"]
    if (
        features["llm_wrapper_kw"] > 0
        and ai_months <= float(gates_cfg["langchain_only_max_ai_months"])
        and not features["pre_llm_ml"]
        and features["retrieval_kw_product"] == 0.0
    ):
        reasons.append(
            "AI experience is <12 months of LLM-API wrapper work with no "
            "pre-LLM ML production history"
        )

    # 4. Stopped coding: no hands-on production role in the last 18 months.
    if features["months_since_code"] > float(gates_cfg["stopped_coding_months"]):
        reasons.append(
            f"no hands-on coding role in {features['months_since_code']:.0f} months"
        )

    # 5. Wrong domain: CV/speech/robotics specialist without NLP/IR exposure.
    if features["cv_kw"] > 0.5 and features["nlp_ir_kw"] == 0.0:
        reasons.append("primary expertise is CV/speech/robotics with no NLP/IR exposure")

    # 6. Non-engineering current role (the "Marketing Manager with RAG skills"
    # stuffer): not strictly a JD bullet, but title-vs-skills mismatch is the
    # trap the JD describes; the technical gate also crushes these.
    if lex.present("non_engineering_titles", candidate.current_title.lower()):
        if all(not t.is_coding for t in tags):
            reasons.append(
                f"current title '{candidate.current_title}' is non-engineering "
                "and no engineering role in history"
            )

    return reasons


def run_gates(
    candidate: Candidate,
    tags: list[RoleTags],
    features: dict[str, float],
    lex: Lexicons,
    ref_date: dt.date,
    cfg: dict,
) -> GateResult:
    result = GateResult()
    hp_reasons = check_honeypot(candidate, ref_date, cfg)
    if hp_reasons:
        result.honeypot = True
        result.reasons.extend(f"honeypot: {r}" for r in hp_reasons)
    dq_reasons = check_disqualifiers(candidate, tags, features, lex, cfg)
    if dq_reasons:
        result.disqualified = True
        result.reasons.extend(f"disqualified: {r}" for r in dq_reasons)
    return result
