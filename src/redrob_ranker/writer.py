"""Submission CSV writer (spec section 2)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission(
    rows: list[dict],
    out_path: str | Path,
) -> Path:
    """Write the submission CSV: candidate_id,rank,score,reasoning (UTF-8)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in HEADER})
    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return out_path
