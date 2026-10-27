"""Lazy, CPU-only sentence embeddings. Importing this module never downloads a model."""

from functools import lru_cache
import os
from typing import Protocol

import numpy as np

from maintainai.database import PROJECT_ROOT

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_DIMENSION = 384
# Include the document pooling format so an algorithm change invalidates old indexes.
MODEL_KEY = f"{MODEL_ID}@{MODEL_REVISION}:field-mean-v1"


class EmbeddingError(RuntimeError):
    """The local model could not be loaded or returned unusable embeddings."""


class TextEmbedder(Protocol):
    model_key: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@lru_cache(maxsize=2)
def _load_model(cache_folder: str, offline: bool):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cpu", cache_folder=cache_folder,
        local_files_only=offline, trust_remote_code=False, token=False,
        model_kwargs={"use_safetensors": True},
    )


class MiniLMEmbedder:
    model_key = MODEL_KEY
    dimension = MODEL_DIMENSION

    def __init__(self, *, offline: bool | None = None):
        self.offline = offline

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        cache_folder = str(PROJECT_ROOT / ".cache" / "sentence-transformers")
        offline = self.offline if self.offline is not None else os.environ.get("MAINTAINAI_MODEL_OFFLINE") == "1"
        try:
            model = _load_model(cache_folder, offline)
            return model.encode(
                texts, batch_size=32, device="cpu", convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                "Could not load or run the local embedding model. Install requirements.txt. "
                "The first rebuild needs internet access to download all-MiniLM-L6-v2. "
                "For offline use, cache the model first and set MAINTAINAI_MODEL_OFFLINE=1."
            ) from exc
