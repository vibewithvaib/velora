"""Stage 5: trust / consistency multiplier.

Honeypots are already gone (Stage 1 removes impossibilities). This module
catches the SOFTER cases — claims the candidate's own history can't back up.
We discount rather than delete, because these may be real people with sloppy
profiles. Every downgrade carries a reason so we can defend it later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .features import RoleTags
from .lexicons import Lexicons
from .models import Candidate

logger = logging.getLogger(__name__)

# Self-claimed AI skill names that need corroborating career evidence.
_AI_SKILL_HINTS = (
    "machine learning", "deep learning", "nlp", "llm", "rag", "langchain",
    "pytorch", "tensorflow", "embedding", "vector", "retrieval", "ranking",
    "recommendation", "transformer", "generative ai", "genai",
)

_SENIOR_TITLE_HINTS = ("senior", "staff", "principal", "lead", "head", "director", "vp")


@dataclass
class TrustResult:
    mismatch_count: int = 0
    reasons: list[str] = field(default_factory=list)


def assess_trust(
    candidate: Candidate,
    tags: list[RoleTags],
    features: dict[str, float],
    lex: Lexicons,
) -> TrustResult:
    """Count cross-corner contradictions: career history vs skills vs summary
    vs current title. More contradictions => lower trust tier."""
    result = TrustResult()
    career_text = candidate.career_text.lower()

    def flag(reason: str) -> None:
        result.mismatch_count += 1
        result.reasons.append(reason)

    # 1. Advanced/expert AI skill claims with zero supporting career evidence.
    claimed_ai = [
        s.name
        for s in candidate.skills
        if s.proficiency in ("advanced", "expert")
        and any(h in s.name.lower() for h in _AI_SKILL_HINTS)
    ]
    if claimed_ai and features["ai_months"] == 0 and features["retrieval_kw"] == 0.0:
        flag(
            f"claims advanced AI skills ({', '.join(claimed_ai[:3])}) with no AI "
            "work in any role"
        )

    # 2. Current title vs most recent role title disagree.
    current_roles = [r for r in candidate.career_history if r.is_current]
    if current_roles and candidate.current_title:
        role_title = current_roles[0].title.lower()
        prof_title = candidate.current_title.lower()
        if prof_title not in role_title and role_title not in prof_title:
            overlap = set(prof_title.split()) & set(role_title.split())
            if not overlap:
                flag(
                    f"profile title '{candidate.current_title}' doesn't match "
                    f"current role '{current_roles[0].title}'"
                )

    # 3. Seniority inflation: Senior/Staff/Principal title on a junior history.
    title = candidate.current_title.lower()
    if any(h in title for h in _SENIOR_TITLE_HINTS) and candidate.years_of_experience < 3:
        flag(
            f"'{candidate.current_title}' title with only "
            f"{candidate.years_of_experience:.0f}y total experience"
        )

    # 4. Stated years vs dated career disagree (below the honeypot bar).
    career_years = features["career_months"] / 12.0
    if career_years > 0 and candidate.years_of_experience > career_years + 2.0:
        flag(
            f"states {candidate.years_of_experience:.0f}y experience; dated "
            f"history covers {career_years:.1f}y"
        )

    # 5. Skill durations exceed the whole career.
    if candidate.skills and features["career_months"] > 0:
        max_skill_months = max((s.duration_months or 0) for s in candidate.skills)
        if max_skill_months > features["career_months"] + 24:
            flag("a skill claims more months of use than the entire career")

    # 6. Scattered expert-with-zero-months below the honeypot threshold.
    expert_zero = sum(
        1
        for s in candidate.skills
        if s.proficiency == "expert" and s.duration_months == 0
    )
    if 1 <= expert_zero < 3:
        flag(f"{expert_zero} 'expert' skill(s) with 0 months of use")

    # 7. Summary brags about retrieval/ranking systems that no role mentions.
    summary = candidate.profile_text.lower()
    if lex.present("retrieval_evidence", summary) and not lex.present(
        "retrieval_evidence", career_text
    ):
        flag("summary claims retrieval/ranking work absent from all role descriptions")

    return result


def trust_multiplier(mismatch_count: int, cfg: dict) -> tuple[float, str]:
    """Map a mismatch count to (multiplier, tier_name)."""
    t = cfg["trust"]
    if mismatch_count >= int(t["shaky_at"]):
        return float(t["multipliers"]["shaky"]), "shaky"
    if mismatch_count >= int(t["several_at"]):
        return float(t["multipliers"]["several"]), "several"
    if mismatch_count >= int(t["minor_at"]):
        return float(t["multipliers"]["minor"]), "minor"
    return float(t["multipliers"]["clean"]), "clean"
