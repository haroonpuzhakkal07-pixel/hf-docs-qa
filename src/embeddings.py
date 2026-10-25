"""Pinned embedding model and checked, normalized encoding."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Reuse the pinned model, loading only from the local cache."""
    return SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )


def encode_texts(
    texts: list[str],
    *,
    prompt: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode complete inputs in their original order, or fail."""
    if not texts:
        raise ValueError("No texts supplied")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a nonempty string")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    model = get_embedding_model()
    limit = model.max_seq_length
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"Invalid encoder sequence limit: {limit!r}")

    # Count the exact encoder input, including prompt and special tokens.
    for start in range(0, len(texts), batch_size):
        features = model.preprocess(
            texts[start:start + batch_size],
            prompt=prompt,
            processing_kwargs={"text": {"truncation": False}},
        )
        lengths = features["attention_mask"].sum(dim=1).tolist()
        for offset, length in enumerate(lengths):
            if length > limit:
                raise ValueError(
                    f"Input row {start + offset} has {length} tokens; "
                    f"encoder limit is {limit}. Text was not truncated."
                )

    vectors = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        precision="float32",
        convert_to_numpy=True,
        show_progress_bar=False,
        processing_kwargs={"text": {"truncation": False}},
    )
    expected_shape = (len(texts), model.get_embedding_dimension())
    if vectors.shape != expected_shape:
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain finite float32 values")
    if not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Embeddings are not unit normalized")

    return np.ascontiguousarray(vectors)