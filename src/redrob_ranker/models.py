"""Typed candidate data model with defensive parsing.

The dataset is synthetic and adversarial (honeypots, stuffers), so the parser
must never crash on odd values — it normalizes them and lets the gates and
trust checks judge them.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y")


def parse_date(value: Any) -> Optional[dt.date]:
    """Parse a date string defensively; return None when unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


@dataclass
class Role:
    company: str
    title: str
    start_date: Optional[dt.date]
    end_date: Optional[dt.date]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Role":
        return cls(
            company=_as_str(raw.get("company")),
            title=_as_str(raw.get("title")),
            start_date=parse_date(raw.get("start_date")),
            end_date=parse_date(raw.get("end_date")),
            duration_months=_as_int(raw.get("duration_months")),
            is_current=bool(raw.get("is_current", False)),
            industry=_as_str(raw.get("industry")),
            company_size=_as_str(raw.get("company_size")),
            description=_as_str(raw.get("description")),
        )

    @property
    def text(self) -> str:
        """Role text used for evidence embedding and lexicon matching."""
        return f"{self.title}. {self.description}".strip()


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    tier: str = "unknown"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Education":
        return cls(
            institution=_as_str(raw.get("institution")),
            degree=_as_str(raw.get("degree")),
            field_of_study=_as_str(raw.get("field_of_study")),
            start_year=_as_int(raw.get("start_year")),
            end_year=_as_int(raw.get("end_year")),
            tier=_as_str(raw.get("tier"), "unknown"),
        )


@dataclass
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int]  # None when the field is absent

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Skill":
        duration = raw.get("duration_months")
        return cls(
            name=_as_str(raw.get("name")),
            proficiency=_as_str(raw.get("proficiency")).lower(),
            endorsements=_as_int(raw.get("endorsements")),
            duration_months=_as_int(duration) if duration is not None else None,
        )


@dataclass
class Signals:
    """The 23 Redrob behavioral signals. -1 sentinels mean UNKNOWN, not worst."""

    profile_completeness_score: float = 0.0
    signup_date: Optional[dt.date] = None
    last_active_date: Optional[dt.date] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_min: float = -1.0
    expected_salary_max: float = -1.0
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Signals":
        salary = raw.get("expected_salary_range_inr_lpa") or {}
        assessments = raw.get("skill_assessment_scores") or {}
        return cls(
            profile_completeness_score=_as_float(raw.get("profile_completeness_score")),
            signup_date=parse_date(raw.get("signup_date")),
            last_active_date=parse_date(raw.get("last_active_date")),
            open_to_work_flag=bool(raw.get("open_to_work_flag", False)),
            profile_views_received_30d=_as_int(raw.get("profile_views_received_30d")),
            applications_submitted_30d=_as_int(raw.get("applications_submitted_30d")),
            recruiter_response_rate=_as_float(raw.get("recruiter_response_rate")),
            avg_response_time_hours=_as_float(raw.get("avg_response_time_hours")),
            skill_assessment_scores={
                str(k): _as_float(v) for k, v in assessments.items()
            },
            connection_count=_as_int(raw.get("connection_count")),
            endorsements_received=_as_int(raw.get("endorsements_received")),
            notice_period_days=_as_int(raw.get("notice_period_days")),
            expected_salary_min=_as_float(salary.get("min"), -1.0),
            expected_salary_max=_as_float(salary.get("max"), -1.0),
            preferred_work_mode=_as_str(raw.get("preferred_work_mode"), "flexible").lower(),
            willing_to_relocate=bool(raw.get("willing_to_relocate", False)),
            github_activity_score=_as_float(raw.get("github_activity_score"), -1.0),
            search_appearance_30d=_as_int(raw.get("search_appearance_30d")),
            saved_by_recruiters_30d=_as_int(raw.get("saved_by_recruiters_30d")),
            interview_completion_rate=_as_float(raw.get("interview_completion_rate")),
            offer_acceptance_rate=_as_float(raw.get("offer_acceptance_rate"), -1.0),
            verified_email=bool(raw.get("verified_email", False)),
            verified_phone=bool(raw.get("verified_phone", False)),
            linkedin_connected=bool(raw.get("linkedin_connected", False)),
        )


@dataclass
class Candidate:
    candidate_id: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    career_history: list[Role]
    education: list[Education]
    skills: list[Skill]
    signals: Signals

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Candidate":
        profile = raw.get("profile") or {}
        return cls(
            candidate_id=_as_str(raw.get("candidate_id")),
            headline=_as_str(profile.get("headline")),
            summary=_as_str(profile.get("summary")),
            location=_as_str(profile.get("location")),
            country=_as_str(profile.get("country")),
            years_of_experience=_as_float(profile.get("years_of_experience")),
            current_title=_as_str(profile.get("current_title")),
            current_company=_as_str(profile.get("current_company")),
            current_company_size=_as_str(profile.get("current_company_size")),
            current_industry=_as_str(profile.get("current_industry")),
            career_history=[Role.from_dict(r) for r in raw.get("career_history") or []],
            education=[Education.from_dict(e) for e in raw.get("education") or []],
            skills=[Skill.from_dict(s) for s in raw.get("skills") or []],
            signals=Signals.from_dict(raw.get("redrob_signals") or {}),
        )

    @property
    def profile_text(self) -> str:
        """Headline + summary; candidate CLAIMS (weighted below career EVIDENCE)."""
        return f"{self.headline}. {self.summary}".strip()

    @property
    def career_text(self) -> str:
        """All role texts joined; the evidence we trust most."""
        return " ".join(role.text for role in self.career_history)
