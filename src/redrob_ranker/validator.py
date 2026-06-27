"""Submission validator — mirrors every auto-reject rule in spec section 3/6.

Run before every upload; the server-side validator rejects without scoring on
any violation, and submissions are capped at three.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CANDIDATE_ID_RE = re.compile(r"^CAND_\d{7}$")


def validate_submission(
    path: str | Path,
    known_ids: set[str] | None = None,
    top_k: int = 100,
) -> list[str]:
    """Return a list of violations (empty list == valid submission)."""
    path = Path(path)
    errors: list[str] = []

    if path.suffix.lower() != ".csv":
        errors.append(f"file extension is '{path.suffix}', must be .csv")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except UnicodeDecodeError:
        return ["file is not valid UTF-8"]

    if not rows:
        return ["file is empty"]

    header, data = rows[0], rows[1:]
    if header[:4] != ["candidate_id", "rank", "score", "reasoning"]:
        errors.append(
            f"header must be candidate_id,rank,score,reasoning — got {header}"
        )

    if len(data) != top_k:
        errors.append(f"expected exactly {top_k} data rows, found {len(data)}")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    prev_score: float | None = None
    score_values: set[float] = set()

    for i, row in enumerate(data, start=2):
        if len(row) < 3:
            errors.append(f"line {i}: fewer than 3 columns")
            continue
        cid, rank_s, score_s = row[0], row[1], row[2]

        if not CANDIDATE_ID_RE.match(cid):
            errors.append(f"line {i}: candidate_id '{cid}' doesn't match CAND_XXXXXXX")
        if cid in seen_ids:
            errors.append(f"line {i}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid)
        if known_ids is not None and cid not in known_ids:
            errors.append(f"line {i}: candidate_id '{cid}' not in candidates.jsonl")

        try:
            rank = int(rank_s)
        except ValueError:
            errors.append(f"line {i}: rank '{rank_s}' is not an integer")
            continue
        if rank in seen_ranks:
            errors.append(f"line {i}: duplicate rank {rank}")
        seen_ranks.add(rank)

        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"line {i}: score '{score_s}' is not a float")
            continue
        score_values.add(score)
        if prev_score is not None and score > prev_score:
            errors.append(
                f"line {i}: score {score} increases over previous {prev_score} "
                "(must be non-increasing with rank)"
            )
        prev_score = score

    expected_ranks = set(range(1, top_k + 1))
    if seen_ranks and seen_ranks != expected_ranks:
        missing = sorted(expected_ranks - seen_ranks)[:5]
        extra = sorted(seen_ranks - expected_ranks)[:5]
        if missing:
            errors.append(f"missing ranks (first few): {missing}")
        if extra:
            errors.append(f"unexpected ranks (first few): {extra} (ranks must be 1..{top_k})")

    if len(score_values) == 1 and len(data) > 1:
        errors.append("all scores identical — model isn't differentiating")

    empty_reasoning = sum(1 for row in data if len(row) < 4 or not row[3].strip())
    if empty_reasoning:
        errors.append(
            f"{empty_reasoning} rows have empty reasoning (optional but penalized at Stage 4)"
        )

    return errors
