"""Answer generation and passage citation mapping."""

import re
from time import perf_counter

from src.bm25 import bm25_search
from src.llm import complete
from src.prompting import SYSTEM_PROMPT, build_prompt


INPUT_TOKEN_BUDGET = 3000

CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question concisely. "
    "If you are unsure, say you don't know."
)

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
    """Generate one answer using BM25 passages or the closed-book baseline."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")

    if config == "bm25_heading":
        started = perf_counter()
        chunks = bm25_search(question, index_name="heading", k=5)
        retrieval_ms = (perf_counter() - started) * 1000
        system = SYSTEM_PROMPT
        prompt = build_prompt(question, chunks)
    elif config == "closed_book":
        chunks = []
        retrieval_ms = 0.0
        system = CLOSED_BOOK_SYSTEM_PROMPT
        prompt = question
    else:
        raise ValueError(f"unknown answer config: {config!r}")

    retrieved_ids = [chunk["chunk_id"] for chunk in chunks]
    result = complete(
        system,
        prompt,
        temperature=0,
        max_tokens=500,
        input_token_budget=INPUT_TOKEN_BUDGET,
    )
    text = result["text"]
    return {
        "answer": text,
        "cited_chunk_ids": _cited_chunk_ids(text, retrieved_ids),
        "retrieved_chunk_ids": retrieved_ids,
        "retrieval_ms": retrieval_ms,
        "llm_ms": result["latency_ms"],
        "token_usage": {
            name: result[name]
            for name in ("tokens_in", "tokens_out", "thinking_tokens", "total_tokens")
        },
        "api_usage": result["api_usage"],
        "preflight_tokens": result["preflight_tokens"],
        "cache_hit": result["cache_hit"],
        "model_version": result["model_version"],
    }