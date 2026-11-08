"""Answer generation and passage citation mapping."""

import re
from time import perf_counter

from src.bm25 import bm25_search
from src.llm import complete
from src.prompting import SYSTEM_PROMPT, build_prompt
from src.hybrid import hybrid_search
from src.rerank import reranked_search
from src.retrieve import dense_search


INPUT_TOKEN_BUDGET = 3000

CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question concisely. "
    "If you are unsure, say you don't know."
)

class CompletionUnavailable(RuntimeError):
    """The answer pipeline could not obtain a completion."""

def _cited_chunk_ids(text: str, passage_chunk_ids: list[str]) -> list[str]:
    """Map numeric citations to chunk IDs in first-citation order."""
    chunk_by_number = {
        str(number): chunk_id
        for number, chunk_id in enumerate(passage_chunk_ids, start=1)
    }
    cited = {}
    for group in re.findall(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]", text):
        for number in group.split(","):
            chunk_id = chunk_by_number.get(number.strip())
            if chunk_id is not None:
                cited[chunk_id] = None
    return list(cited)


def answer(question: str, config: str) -> dict:
    """Generate an answer using the selected retrieval configuration."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")

    if config == "closed_book":
        chunks = []
        retrieval_ms = 0.0
        system = CLOSED_BOOK_SYSTEM_PROMPT
        prompt = question
    else:
        retrievers = {
            "dense_fixed": (dense_search, "fixed"),
            "dense_heading": (dense_search, "heading"),
            "bm25_heading": (bm25_search, "heading"),
            "hybrid_heading": (hybrid_search, "heading"),
            "hybrid_rerank_heading": (reranked_search, "heading"),
        }
        if config not in retrievers:
            raise ValueError(f"unknown answer config: {config!r}")

        search, index_name = retrievers[config]
        started = perf_counter()
        chunks = search(question, index_name=index_name, k=5)
        retrieval_ms = (perf_counter() - started) * 1000
        system = SYSTEM_PROMPT
        prompt = build_prompt(question, chunks)

    retrieved_ids = [chunk["chunk_id"] for chunk in chunks]
    try:
        result = complete(
            system,
            prompt,
            temperature=0,
            max_tokens=500,
            input_token_budget=INPUT_TOKEN_BUDGET,
        )
    except RuntimeError as exc:
        raise CompletionUnavailable(str(exc)) from exc
    text = result["text"]
    return {
        "answer": text,
        "cited_chunk_ids": _cited_chunk_ids(text, retrieved_ids),
        "retrieved_chunk_ids": retrieved_ids,
        "retrieval_ms": retrieval_ms,
        "llm_ms": result["latency_ms"],
        "original_llm_ms": result["original_latency_ms"],
        "requested_model": result["model"],
        "cache_key": result["cache_key"],
        "token_usage": {
            name: result[name]
            for name in ("tokens_in", "tokens_out", "thinking_tokens", "total_tokens")
        },
        "api_usage": result["api_usage"],
        "preflight_tokens": result["preflight_tokens"],
        "cache_hit": result["cache_hit"],
        "model_version": result["model_version"],
    }