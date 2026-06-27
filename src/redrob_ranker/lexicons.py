"""Compiled keyword lexicons.

All evidence keywords live in config/lexicons.yaml; this module compiles them
into word-boundary regexes once and exposes cheap counting/matching helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


def _compile_terms(terms: list[str]) -> re.Pattern[str]:
    """Compile a list of phrases into a single alternation regex.

    Word boundaries on both sides so 'ml' does not match 'html'. Terms with
    special characters (c++, a/b) are escaped.
    """
    escaped = sorted((re.escape(t.lower()) for t in terms), key=len, reverse=True)
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern)


@dataclass(frozen=True)
class Lexicons:
    """Holds one compiled matcher per lexicon category."""

    patterns: dict
    bonus_patterns: dict
    raw: dict

    def count(self, category: str, text: str) -> int:
        """Number of matches of a category's terms in text (lowercased)."""
        return len(self.patterns[category].findall(text))

    def present(self, category: str, text: str) -> bool:
        return bool(self.patterns[category].search(text))

    def matches(self, category: str, text: str) -> list[str]:
        """Distinct matched terms, in order of first appearance."""
        seen: list[str] = []
        for match in self.patterns[category].finditer(text):
            term = match.group(0)
            if term not in seen:
                seen.append(term)
        return seen

    def bonus_hits(self, text: str) -> list[str]:
        """Names of nice-to-have bonus categories present in text."""
        return [name for name, pat in self.bonus_patterns.items() if pat.search(text)]


@lru_cache(maxsize=4)
def load_lexicons(path: str) -> Lexicons:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    patterns: dict[str, re.Pattern[str]] = {}
    bonus_patterns: dict[str, re.Pattern[str]] = {}
    for key, value in raw.items():
        if key == "bonus_categories":
            for cat, terms in value.items():
                bonus_patterns[cat] = _compile_terms(terms)
        elif key == "cities":
            for tier, names in value.items():
                patterns[f"cities_{tier}"] = _compile_terms(names)
        elif isinstance(value, list):
            patterns[key] = _compile_terms(value)
    return Lexicons(patterns=patterns, bonus_patterns=bonus_patterns, raw=raw)
