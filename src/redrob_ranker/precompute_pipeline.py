"""Stage 0: offline precompute pipeline.

Runs once per candidates file, with no time budget. Produces every expensive
artifact the 5-minute online rank step needs:

  - JD facet embeddings (intent, not keywords)
  - per-candidate pooled evidence embeddings (roles encoded separately,
    duration x recency weighted; skills list deliberately excluded)
  - scalar semantic features (JD is fixed, so role-vs-facet sims are offline)
  - derived numeric features (career math + 23 raw behavioral signals)
  - hard-gate flags (honeypot / disqualified) with logged reasons
  - trust mismatch counts with logged reasons
  - facts dicts for grounded reasoning generation

Memory stays bounded: candidates stream in chunks; role texts are encoded
chunk-by-chunk and only the pooled (N, 384) matrix is retained.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import Artifacts, file_sha1, save_artifacts
from .encoder import Encoder
from .evidence import pool_candidate_vector, role_texts, role_weights, semantic_role_features
from .features import classify_role, derive_features
from .gates import run_gates
from .lexicons import load_lexicons
from .loader import iter_candidates
from .models import Candidate, parse_date
from .reasoning import build_facts
from .trust import assess_trust, trust_multiplier

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512


def derive_reference_date(candidates_path: str | Path) -> dt.date:
    """Deterministic dataset 'today': max(last_active_date) over the file."""
    best: dt.date | None = None
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = parse_date((raw.get("redrob_signals") or {}).get("last_active_date"))
            if date is not None and (best is None or date > best):
                best = date
    return best or dt.date.today()


def _chunks(iterable, size: int):
    chunk: list[Candidate] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_precompute(
    candidates_path: str | Path,
    cfg: dict,
    lexicons_path: str | Path,
    artifacts_base: str | Path | None = None,
) -> Path:
    started = time.time()
    candidates_path = Path(candidates_path)
    artifacts_base = Path(artifacts_base or cfg["artifacts_dir"])
    lex = load_lexicons(str(lexicons_path))

    input_sha = file_sha1(candidates_path)
    logger.info("Input %s (sha1 %s)", candidates_path, input_sha[:12])

    ref_date = (
        parse_date(cfg.get("reference_date"))
        if cfg.get("reference_date")
        else derive_reference_date(candidates_path)
    )
    assert ref_date is not None
    logger.info("Reference date: %s", ref_date)

    model_cfg = cfg["model"]
    encoder = Encoder(
        model_name=model_cfg["name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        batch_size=int(model_cfg["batch_size"]),
    )

    jd_facets: list[str] = cfg["jd_facets"]
    jd_vecs = encoder.encode(jd_facets)
    core_idx: list[int] = cfg["core_facet_indices"]
    profile_weight = float(cfg["evidence"]["profile_text_weight"])

    ids: list[str] = []
    emb_rows: list[np.ndarray] = []
    feature_rows: list[dict[str, float]] = []
    honeypot_flags: list[bool] = []
    dq_flags: list[bool] = []
    mismatch_counts: list[int] = []
    reasons: dict[str, dict] = {}
    facts: dict[str, dict] = {}

    n_processed = 0
    for chunk in _chunks(iter_candidates(candidates_path), CHUNK_SIZE):
        # ---- batch-encode all texts in this chunk -------------------------
        texts: list[str] = []
        spans: list[tuple[int, int, bool]] = []  # (start, n_roles, has_profile)
        for cand in chunk:
            roles = role_texts(cand)
            profile = cand.profile_text.strip()
            spans.append((len(texts), len(roles), bool(profile)))
            texts.extend(roles)
            if profile:
                texts.append(profile)
        vecs = encoder.encode(texts) if texts else np.zeros((0, encoder.dim), np.float32)

        # ---- per-candidate processing --------------------------------------
        for cand, (start, n_roles, has_profile) in zip(chunk, spans):
            role_vecs = vecs[start : start + n_roles]
            profile_vec = vecs[start + n_roles] if has_profile else None
            weights = role_weights(cand, ref_date, cfg)

            tags = [classify_role(r, lex) for r in cand.career_history]
            feats = derive_features(cand, tags, lex, ref_date, cfg)
            feats.update(
                semantic_role_features(role_vecs, weights, profile_vec, jd_vecs, core_idx)
            )

            gate = run_gates(cand, tags, feats, lex, ref_date, cfg)
            trust = assess_trust(cand, tags, feats, lex)

            ids.append(cand.candidate_id)
            emb_rows.append(
                pool_candidate_vector(role_vecs, weights, profile_vec, profile_weight)
            )
            feature_rows.append(feats)
            honeypot_flags.append(gate.honeypot)
            dq_flags.append(gate.disqualified)
            mismatch_counts.append(trust.mismatch_count)
            if gate.reasons or trust.reasons:
                reasons[cand.candidate_id] = {
                    "gate": gate.reasons,
                    "trust": trust.reasons,
                }
            facts[cand.candidate_id] = build_facts(cand, tags, feats, lex, ref_date)

        n_processed += len(chunk)
        if n_processed % 5120 == 0:
            logger.info("Processed %d candidates...", n_processed)

    if not ids:
        raise ValueError(f"No valid candidates found in {candidates_path}")

    feature_names = sorted(feature_rows[0].keys())
    features = {
        name: np.asarray([row[name] for row in feature_rows], dtype=np.float32)
        for name in feature_names
    }

    n_honeypots = int(np.sum(honeypot_flags))
    n_dq = int(np.sum(dq_flags))
    logger.info(
        "Precompute done: %d candidates, %d honeypots flagged, %d disqualified (%.1fs)",
        len(ids), n_honeypots, n_dq, time.time() - started,
    )

    data = Artifacts(
        meta={
            "version": __version__,
            "input_sha": input_sha,
            "input_file": str(candidates_path),
            "n_candidates": len(ids),
            "n_honeypots": n_honeypots,
            "n_disqualified": n_dq,
            "model": model_cfg["name"],
            "dim": int(jd_vecs.shape[1]),
            "reference_date": ref_date.isoformat(),
            "feature_names": feature_names,
            "jd_facets": jd_facets,
        },
        ids=ids,
        emb=np.vstack(emb_rows).astype(np.float32),
        jd_vecs=jd_vecs,
        features=features,
        honeypot=np.asarray(honeypot_flags, dtype=bool),
        disqualified=np.asarray(dq_flags, dtype=bool),
        trust_mismatches=np.asarray(mismatch_counts, dtype=np.int32),
        reasons=reasons,
        facts=facts,
    )
    return save_artifacts(artifacts_base, data)
