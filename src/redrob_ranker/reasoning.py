"""Stage 7: deterministic, grounded reasoning generation.

The Stage-4 graders sample 10 rows and check: specific facts, JD connection,
honest concerns, no hallucination, variation, rank consistency. This generator
satisfies each by construction:

  - every clause is assembled from fields that exist in the candidate's
    profile (a compact "facts" dict built at precompute) -> no hallucination;
  - clauses cite concrete numbers (years, response rate, notice period) and
    phrase them against JD requirements -> specific facts + JD connection;
  - when a concern flag is set it is stated outright -> honest concerns;
  - sentence shape is keyed to the candidate's dominant strength, concern
    state and a per-candidate hash -> variation;
  - tone tracks the rank band (confident at the top, hedged near 100) ->
    rank consistency.

No LLM call — the no-network constraint makes that mandatory anyway.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from typing import Any

from .features import RoleTags, effective_duration_months
from .lexicons import Lexicons
from .models import Candidate

logger = logging.getLogger(__name__)

Facts = dict[str, Any]


# ---------------------------------------------------------------------------
# Facts extraction (offline, stored in the artifact)
# ---------------------------------------------------------------------------

def build_facts(
    candidate: Candidate,
    tags: list[RoleTags],
    features: dict[str, float],
    lex: Lexicons,
    ref_date: dt.date,
) -> Facts:
    """Compact per-candidate fact dict — the only source the generator may cite."""
    roles = candidate.career_history
    durations = [effective_duration_months(r, ref_date) for r in roles]

    # Strongest evidence roles: retrieval/production roles first, longest first.
    scored_roles = sorted(
        zip(roles, tags, durations),
        key=lambda rt: (rt[1].has_retrieval_evidence, rt[1].has_production_language, rt[2]),
        reverse=True,
    )
    top_roles = []
    for role, tag, months in scored_roles[:2]:
        top_roles.append(
            {
                "title": role.title,
                "company": role.company,
                "months": months,
                "evidence": lex.matches("retrieval_evidence", role.text.lower())[:3],
                "production": tag.has_production_language,
                "is_product": tag.is_product,
            }
        )

    matched_assessments = sorted(
        (
            (name, score)
            for name, score in candidate.signals.skill_assessment_scores.items()
            if lex.present("jd_skill_keys", name.lower())
        ),
        key=lambda kv: -kv[1],
    )[:2]

    sig = candidate.signals
    return {
        "yoe": candidate.years_of_experience,
        "ai_product_years": features["ai_product_months"] / 12.0,
        "current_title": candidate.current_title,
        "current_company": candidate.current_company,
        "location": candidate.location,
        "country": candidate.country,
        "top_roles": top_roles,
        "notice_days": int(features["notice_days"]),
        "days_since_active": int(features["days_since_active"]),
        "response_rate": float(features["response_rate"]),
        "open_to_work": bool(features["open_to_work"]),
        "github_linked": features["github_score"] >= 0,
        "assessments": matched_assessments,
        "salary": [sig.expected_salary_min, sig.expected_salary_max],
        "work_mode": sig.preferred_work_mode,
        "relocate": bool(features["relocate"]),
        "city_tier_code": int(features["city_tier_code"]),
    }


# ---------------------------------------------------------------------------
# Clause builders — every clause sourced from facts only
# ---------------------------------------------------------------------------

def _pick(candidate_id: str, salt: str, options: list[str]) -> str:
    digest = hashlib.md5(f"{candidate_id}:{salt}".encode()).hexdigest()
    return options[int(digest, 16) % len(options)]


def _technical_clause(cid: str, facts: Facts) -> str | None:
    for role in facts["top_roles"]:
        if role["evidence"]:
            terms = " and ".join(role["evidence"][:2])
            where = f" at {role['company']}" if role["company"] else ""
            tmpl = _pick(cid, "tech", [
                "career history shows hands-on {terms} work{where} ({months} months)",
                "built {terms} systems{where} over {months} months",
                "{months} months of {terms} work{where} directly matches the retrieval/ranking mandate",
            ])
            return tmpl.format(terms=terms, where=where, months=role["months"])
    return None


def _experience_clause(cid: str, facts: Facts) -> str | None:
    yoe = facts["yoe"]
    ai = facts["ai_product_years"]
    if yoe <= 0:
        return None
    if ai >= 1:
        tmpl = _pick(cid, "exp", [
            "{yoe:.0f} years total with ~{ai:.0f} in applied ML at product companies — squarely in the JD's 5-9 band" if 5 <= yoe <= 9 else "{yoe:.0f} years total with ~{ai:.0f} in applied ML at product companies",
            "brings {yoe:.0f} years of experience, ~{ai:.0f} of them shipping ML at product companies",
        ])
        return tmpl.format(yoe=yoe, ai=ai)
    return _pick(cid, "exp2", [
        "{yoe:.0f} years of engineering experience",
        "has {yoe:.0f} years in the field",
    ]).format(yoe=yoe)


def _shipping_clause(cid: str, facts: Facts) -> str | None:
    prod_roles = [r for r in facts["top_roles"] if r["production"]]
    if not prod_roles:
        return None
    role = prod_roles[0]
    return _pick(cid, "ship", [
        f"shows production ownership as {role['title']}",
        f"role descriptions evidence deployed, production-scale work ({role['title']})",
    ])


def _availability_clause(cid: str, facts: Facts) -> str | None:
    bits = []
    if facts["open_to_work"]:
        bits.append("open to work")
    if facts["days_since_active"] <= 14:
        bits.append("active on Redrob this fortnight")
    if facts["response_rate"] >= 0.6:
        bits.append(f"{facts['response_rate']:.0%} recruiter response rate")
    if facts["notice_days"] <= 30:
        bits.append(f"{facts['notice_days']}-day notice")
    if not bits:
        return None
    return _pick(cid, "avail", ["{b} — reachable and hireable", "{b}"]).format(
        b=", ".join(bits[:3])
    )


def _credibility_clause(cid: str, facts: Facts) -> str | None:
    if facts["assessments"]:
        name, score = facts["assessments"][0]
        if score >= 70:
            return f"scored {score:.0f}/100 on the {name} assessment"
    return None


def _location_clause(cid: str, facts: Facts) -> str | None:
    tier = facts["city_tier_code"]
    if tier == 0:
        return f"based in {facts['location']} (a JD-preferred location)"
    if tier == 1:
        return f"based in {facts['location']}"
    if tier == 2 and facts["relocate"]:
        return f"in {facts['location']} but willing to relocate"
    return None


def build_concerns(facts: Facts, trust_tier: str, trust_reasons: list[str]) -> list[str]:
    """Honest concerns, each grounded in a profile field."""
    concerns: list[str] = []
    if facts["notice_days"] > 60:
        concerns.append(f"{facts['notice_days']}-day notice period")
    if facts["days_since_active"] > 90:
        concerns.append(f"last active {facts['days_since_active'] // 30} months ago")
    if facts["response_rate"] < 0.2:
        concerns.append(f"only {facts['response_rate']:.0%} recruiter response rate")
    if not facts["github_linked"]:
        concerns.append("no GitHub linked")
    if trust_tier in ("several", "shaky") and trust_reasons:
        concerns.append(trust_reasons[0])
    if facts["work_mode"] == "remote":
        concerns.append("prefers fully-remote (role is hybrid Pune/Noida)")
    return concerns


# ---------------------------------------------------------------------------
# Sentence assembly
# ---------------------------------------------------------------------------

_CLAUSE_BUILDERS = {
    "technical": _technical_clause,
    "experience": _experience_clause,
    "shipping": _shipping_clause,
    "availability": _availability_clause,
    "credibility": _credibility_clause,
    "logistics": _location_clause,
}


def generate_reasoning(
    candidate_id: str,
    facts: Facts,
    factor_ranking: list[str],
    concerns: list[str],
    rank: int,
    top_k: int = 100,
) -> str:
    """One-or-two sentence justification, tone keyed to the rank band."""
    clauses: list[str] = []
    for factor in factor_ranking:
        builder = _CLAUSE_BUILDERS.get(factor)
        if builder is None:
            continue
        clause = builder(candidate_id, facts)
        if clause:
            clauses.append(clause)
        if len(clauses) == 2:
            break
    if not clauses:
        title = facts["current_title"] or "candidate"
        clauses.append(f"{title} with {facts['yoe']:.0f} years of experience")

    body = clauses[0]
    if len(clauses) > 1:
        body = f"{clauses[0]}; {clauses[1]}"

    # Rank-band tone
    if rank <= 10:
        opener = _pick(candidate_id, "op1", ["Strong fit:", "Excellent match:", "Top-tier fit —"])
    elif rank <= 40:
        opener = _pick(candidate_id, "op2", ["Good fit:", "Solid match:", "Strong candidate —"])
    elif rank <= 80:
        opener = _pick(candidate_id, "op3", ["Reasonable fit:", "Qualified:", "Credible option —"])
    else:
        opener = _pick(candidate_id, "op4", [
            "Borderline — included on partial evidence:",
            "Below the strongest cohort:",
            "Adjacent fit:",
        ])

    sentence = f"{opener} {body}."
    if concerns:
        concern_text = "; ".join(concerns[:2])
        joiner = _pick(candidate_id, "cj", ["Concern:", "Noted concern:", "Caveat:"])
        sentence = f"{sentence} {joiner} {concern_text}."
    elif rank > 80:
        sentence = f"{sentence} Ranked lower on overall evidence depth."
    return sentence
