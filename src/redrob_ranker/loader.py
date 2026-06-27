"""Streaming JSONL ingestion.

The candidate pool is ~100k records; we stream line-by-line and never hold the
raw text of the full file in memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from .models import Candidate

logger = logging.getLogger(__name__)


def iter_candidates(path: str | Path) -> Iterator[Candidate]:
    """Yield Candidate objects from a JSONL file, skipping malformed lines."""
    path = Path(path)
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                candidate = Candidate.from_dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                skipped += 1
                logger.warning("Skipping malformed line %d: %s", line_no, exc)
                continue
            if not candidate.candidate_id:
                skipped += 1
                logger.warning("Skipping line %d: missing candidate_id", line_no)
                continue
            yield candidate
    if skipped:
        logger.warning("Skipped %d malformed records in %s", skipped, path)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
