#!/usr/bin/env python3
"""Online ranking (Stages 1-7). Produces the submission CSV.

This is the Stage-3 reproduction command:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Budget: <= 5 min wall-clock, <= 16 GB RAM, CPU only, no network. This script
imports neither torch nor sentence-transformers; it consumes precomputed
artifacts. If artifacts for the given file are missing (e.g. the <=100
candidate sandbox sample), precompute runs automatically first — on a small
sample that finishes in seconds and stays inside the budget.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.artifacts import artifact_dir, file_sha1  # noqa: E402
from redrob_ranker.loader import load_config  # noqa: E402
from redrob_ranker.rank_pipeline import run_rank  # noqa: E402

logger = logging.getLogger("rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--lexicons", default="config/lexicons.yaml")
    parser.add_argument("--artifacts", default=None, help="Artifacts base dir (default from config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    base = Path(args.artifacts or cfg["artifacts_dir"])

    if not (artifact_dir(base, file_sha1(args.candidates)) / "meta.json").exists():
        logger.warning(
            "No artifacts found for this candidates file — running precompute "
            "first. For the full pool, run precompute.py separately (it may "
            "exceed the 5-minute ranking budget; the spec allows that)."
        )
        from redrob_ranker.precompute_pipeline import run_precompute

        run_precompute(args.candidates, cfg, args.lexicons, base)

    out = run_rank(args.candidates, cfg, artifacts_base=base, out_path=args.out)
    print(f"Submission written to: {out}")


if __name__ == "__main__":
    main()
