"""Sandbox demo app (HuggingFace Spaces / Streamlit Cloud).

Accepts a small candidates.jsonl sample (<=100 candidates), runs the full
pipeline end-to-end (precompute + rank — fast at this scale) and offers the
ranked CSV for download. This satisfies the spec's sandbox requirement; full
100k reproduction happens via rank.py in the organizers' own sandbox.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.loader import load_config  # noqa: E402
from redrob_ranker.precompute_pipeline import run_precompute  # noqa: E402
from redrob_ranker.rank_pipeline import run_rank  # noqa: E402

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent

st.title("Redrob Candidate Ranker — sandbox")
st.caption(
    "Upload a candidates.jsonl sample (≤100 candidates). The system runs "
    "precompute + rank end-to-end and returns the ranked CSV."
)

uploaded = st.file_uploader("candidates.jsonl", type=["jsonl", "json", "txt"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        candidates = tmp_path / "candidates.jsonl"
        candidates.write_bytes(uploaded.getvalue())

        n_lines = sum(1 for line in candidates.read_text(encoding="utf-8").splitlines() if line.strip())
        if n_lines > 100:
            st.warning(f"{n_lines} candidates uploaded; sandbox is sized for ≤100. Proceeding anyway.")

        cfg = load_config(ROOT / "config" / "config.yaml")
        cfg["output"]["top_k"] = min(100, n_lines)
        lexicons = ROOT / "config" / "lexicons.yaml"
        artifacts = tmp_path / "artifacts"

        with st.spinner("Precomputing embeddings and features..."):
            run_precompute(candidates, cfg, lexicons, artifacts)
        with st.spinner("Ranking..."):
            out = run_rank(candidates, cfg, artifacts_base=artifacts, out_path=tmp_path / "submission.csv")

        csv_bytes = Path(out).read_bytes()
        st.success(f"Ranked {min(100, n_lines)} candidates.")
        st.download_button("Download submission.csv", csv_bytes, file_name="submission.csv", mime="text/csv")

        import csv as _csv
        import io

        rows = list(_csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
        st.dataframe(rows, use_container_width=True)
