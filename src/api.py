"""HTTP service for Transformers documentation QA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from src.embeddings import get_embedding_model
from src.generate import CompletionUnavailable, answer
from src.hybrid import check_hybrid_indexes
from src.rerank import get_reranker
from src.retrieve import load_index


class HealthResponse(BaseModel):
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    check_hybrid_indexes("heading")
    _, chunks = load_index("heading")
    app.state.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    get_embedding_model()
    get_reranker()
    yield


app = FastAPI(title="Transformers Docs QA", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    passage: int
    path: str
    heading: str
    snippet: str


class TokenUsage(BaseModel):
    tokens_in: int
    tokens_out: int
    thinking_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: float
    llm_ms: float
    token_usage: TokenUsage


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    try:
        result = answer(body.question, "hybrid_rerank_heading")
    except CompletionUnavailable:
        raise HTTPException(
            status_code=503, detail="Answer generation is temporarily unavailable."
        ) from None

    passage_numbers = {
        chunk_id: number
        for number, chunk_id in enumerate(result["retrieved_chunk_ids"], start=1)
    }
    sources = []
    for chunk_id in result["cited_chunk_ids"]:
        chunk = request.app.state.chunks_by_id[chunk_id]
        sources.append(
            Source(
                passage=passage_numbers[chunk_id],
                path=chunk["path"],
                heading=chunk["heading_path"],
                snippet=" ".join(chunk["source_text"].split())[:240],
            )
        )

    return AskResponse(
        answer=result["answer"],
        sources=sources,
        retrieval_ms=result["retrieval_ms"],
        llm_ms=result["llm_ms"],
        token_usage=TokenUsage(**result["token_usage"]),
    )
