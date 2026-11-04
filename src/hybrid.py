"""Hybrid retrieval using reciprocal rank fusion."""
from functools import lru_cache


RRF_CONSTANT = 60
CANDIDATE_DEPTH = 50


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict]],
    k: int = 5,
    rrf_constant: int = RRF_CONSTANT,
) -> list[dict]:
    """Fuse named chunk rankings, preserving complete chunk metadata.

    Each chunk contributes once per ranking. Equal fused scores are
    ordered by chunk ID. Returned rrf_ranks records contributing ranks.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(rrf_constant) is not int or rrf_constant <= 0:
        raise ValueError("rrf_constant must be a positive integer")

    records = {}
    scores = {}
    ranks = {}

    for name, results in rankings.items():
        seen = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Each result needs a nonempty chunk_id")
            if chunk_id in seen:
                raise ValueError(f"Duplicate chunk ID within {name} results")
            seen.add(chunk_id)

            record = {
                key: value
                for key, value in result.items()
                if key != "score"
            }
            if chunk_id in records and records[chunk_id] != record:
                raise ValueError(f"Conflicting chunk records: {chunk_id}")

            records[chunk_id] = record
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
            )
            ranks.setdefault(chunk_id, {})[name] = rank

    ordered = sorted(
        records,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:k]

    return [
        {
            **records[chunk_id],
            "score": scores[chunk_id],
            "rrf_ranks": ranks[chunk_id],
        }
        for chunk_id in ordered
    ]


@lru_cache(maxsize=2)
def check_hybrid_indexes(index_name: str) -> None:
    """Verify both indexes and require identical complete chunk mappings."""
    from src.bm25 import load_bm25_index
    from src.retrieve import load_index

    _, dense_records = load_index(index_name)
    _, bm25_records = load_bm25_index(index_name)
    if dense_records != bm25_records:
        raise ValueError("Dense and BM25 chunk mappings differ")


def hybrid_search(query: str, index_name: str, k: int = 5) -> list[dict]:
    """Fuse up to 50 dense and 50 BM25 candidates."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")

    check_hybrid_indexes(index_name)

    from src.bm25 import bm25_search
    from src.retrieve import dense_search

    return reciprocal_rank_fusion(
        {
            "dense": dense_search(query, index_name, k=CANDIDATE_DEPTH),
            "bm25": bm25_search(query, index_name, k=CANDIDATE_DEPTH),
        },
        k=k,
    )