"""Sentence-encoder wrapper. Imported ONLY by the precompute pipeline.

The online ranking step never touches this module (or torch) — it consumes
precomputed float32 arrays, which is what keeps the 5-minute CPU budget safe.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Encoder:
    """Thin wrapper around sentence-transformers with L2-normalized output."""

    def __init__(self, model_name: str, max_seq_length: int, batch_size: int) -> None:
        # Lazy import so the rank step can run without torch installed.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading encoder %s", model_name)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """Encode texts to L2-normalized float32 vectors (N, dim)."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=progress,
        )
        return np.asarray(vectors, dtype=np.float32)
