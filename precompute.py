#!/usr/bin/env python3
"""Offline precompute (Stage 0). Run once per candidates file; no time budget.

Usage:
    python precompute.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.loader import load_config  # noqa: E402
from redrob_ranker.precompute_pipeline import run_precompute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
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
    out = run_precompute(args.candidates, cfg, args.lexicons, args.artifacts)
    print(f"Artifacts written to: {out}")


if __name__ == "__main__":
    main()
