"""Pinned cross-encoder scoring without input truncation."""

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from src.hybrid import hybrid_search

from src.config import (
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_REVISION,
)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Reuse the pinned CPU model from the local cache."""
    model = CrossEncoder(
        RERANKER_MODEL,
        revision=RERANKER_REVISION,
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
        max_length=RERANKER_MAX_LENGTH,
        model_kwargs={"dtype": torch.float32},
        activation_fn=torch.nn.Identity(),
    )
    if (
        model.max_seq_length != RERANKER_MAX_LENGTH
        or model.config.max_position_embeddings != RERANKER_MAX_LENGTH
        or model.num_labels != 1
        or next(model.parameters()).dtype != torch.float32
    ):
        raise ValueError("Unexpected reranker configuration")
    return model


def score_pairs(
    query: str,
    passages: list[str],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Return raw relevance scores in passage order, or fail."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if not passages or any(
        not isinstance(text, str) or not text.strip()
        for text in passages
    ):
        raise ValueError("passages must contain nonempty strings")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    model = get_reranker()
    pairs = [(query, passage) for passage in passages]

    # Check every complete pair before performing any inference.
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        features = model.preprocess(
            batch,
            prompt="",
            processing_kwargs={"text": {"truncation": False}},
        )
        lengths = features["attention_mask"].sum(dim=1).tolist()
        if len(lengths) != len(batch):
            raise ValueError("Unexpected pair token-count shape")
        for offset, length in enumerate(lengths):
            if length > RERANKER_MAX_LENGTH:
                raise ValueError(
                    f"Pair row {start + offset} has {length} tokens; "
                    f"reranker limit is {RERANKER_MAX_LENGTH}. "
                    "Text was not truncated."
                )

    scores = model.predict(
        pairs,
        prompt="",
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        apply_softmax=False,
        processing_kwargs={"text": {"truncation": False}},
    )
    if not isinstance(scores, np.ndarray) or scores.shape != (len(pairs),):
        raise ValueError("Unexpected reranker score shape")
    if scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError("Reranker scores must contain finite float32 values")
    return scores


RERANK_CANDIDATES = 20


def reranked_search(query: str, index_name: str, k: int = 5) -> list[dict]:
    """Rerank the top 20 hybrid chunks, preserving complete records."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")
    if type(k) is not int or not 1 <= k <= RERANK_CANDIDATES:
        raise ValueError(f"k must be an integer from 1 to {RERANK_CANDIDATES}")

    candidates = hybrid_search(query, index_name, k=RERANK_CANDIDATES)
    if not candidates:
        return []

    scores = score_pairs(query, [row["text"] for row in candidates])
    ordered = sorted(
        range(len(candidates)),
        key=lambda position: (-float(scores[position]), position),
    )

    return [
        {
            **candidates[position],
            "hybrid_score": candidates[position]["score"],
            "hybrid_rank": position + 1,
            "score": float(scores[position]),
        }
        for position in ordered[:k]
    ]