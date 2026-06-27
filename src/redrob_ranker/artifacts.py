"""Artifact store for precomputed state.

Artifacts are keyed by the SHA-1 of the input candidates file, so ranking
always runs against state derived from exactly the file it was given — a
stale-artifact mismatch is detected, never silently used.

Layout: artifacts/<sha12>/
  meta.json      — input hash, model, reference date, counts, versions
  ids.json       — candidate_id list (row order for all arrays)
  emb.npy        — pooled evidence embeddings, float32 (N, d), L2-normalized
  jd_vecs.npy    — JD facet embeddings, float32 (F, d)
  features.npz   — one float32 array per feature name
  flags.npz      — honeypot / disqualified / trust_mismatches
  reasons.json.gz— per-candidate gate/trust reasons (auditability)
  facts.json.gz  — per-candidate facts for reasoning generation
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def file_sha1(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Artifacts:
    meta: dict
    ids: list[str]
    emb: np.ndarray
    jd_vecs: np.ndarray
    features: dict[str, np.ndarray]
    honeypot: np.ndarray
    disqualified: np.ndarray
    trust_mismatches: np.ndarray
    reasons: dict[str, dict]
    facts: dict[str, dict]


def artifact_dir(base_dir: str | Path, input_sha: str) -> Path:
    return Path(base_dir) / input_sha[:12]


def save_artifacts(base_dir: str | Path, data: Artifacts) -> Path:
    out = artifact_dir(base_dir, data.meta["input_sha"])
    out.mkdir(parents=True, exist_ok=True)

    (out / "meta.json").write_text(json.dumps(data.meta, indent=2), encoding="utf-8")
    (out / "ids.json").write_text(json.dumps(data.ids), encoding="utf-8")
    np.save(out / "emb.npy", data.emb)
    np.save(out / "jd_vecs.npy", data.jd_vecs)
    np.savez_compressed(out / "features.npz", **data.features)
    np.savez_compressed(
        out / "flags.npz",
        honeypot=data.honeypot,
        disqualified=data.disqualified,
        trust_mismatches=data.trust_mismatches,
    )
    with gzip.open(out / "reasons.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(data.reasons, handle)
    with gzip.open(out / "facts.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(data.facts, handle)

    logger.info("Saved artifacts to %s", out)
    return out


def load_artifacts(base_dir: str | Path, input_sha: str) -> Artifacts:
    src = artifact_dir(base_dir, input_sha)
    if not (src / "meta.json").exists():
        raise FileNotFoundError(
            f"No precomputed artifacts at {src}. Run precompute.py first "
            "(or rank.py will run it automatically)."
        )
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if meta["input_sha"] != input_sha:
        raise ValueError(
            f"Artifact hash mismatch: artifacts built for {meta['input_sha'][:12]}, "
            f"input file is {input_sha[:12]}. Re-run precompute."
        )
    ids = json.loads((src / "ids.json").read_text(encoding="utf-8"))
    features_npz = np.load(src / "features.npz")
    flags = np.load(src / "flags.npz")
    with gzip.open(src / "reasons.json.gz", "rt", encoding="utf-8") as handle:
        reasons = json.load(handle)
    with gzip.open(src / "facts.json.gz", "rt", encoding="utf-8") as handle:
        facts = json.load(handle)
    return Artifacts(
        meta=meta,
        ids=ids,
        emb=np.load(src / "emb.npy"),
        jd_vecs=np.load(src / "jd_vecs.npy"),
        features={name: features_npz[name] for name in features_npz.files},
        honeypot=flags["honeypot"],
        disqualified=flags["disqualified"],
        trust_mismatches=flags["trust_mismatches"],
        reasons=reasons,
        facts=facts,
    )
