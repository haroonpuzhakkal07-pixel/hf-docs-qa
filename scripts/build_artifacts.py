"""Prepare documentation QA artifacts during the Docker build."""

import subprocess

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

from src.bm25 import write_bm25_index
from src.chunking_heading import write_heading_dataset
from src.config import (
    CORPUS_COMMIT,
    CORPUS_REPO,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RAW_DIR,
    REPO_ROOT,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_REVISION,
)
from src.index import build_index
from src.ingest import write_documents


def fetch_corpus() -> None:
    if RAW_DIR.exists():
        raise FileExistsError(f"Corpus checkout already exists: {RAW_DIR}")

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(RAW_DIR)], check=True)
    for args in (
        ["remote", "add", "origin", CORPUS_REPO],
        ["sparse-checkout", "set", "docs/source/en"],
        ["fetch", "--depth", "1", "origin", CORPUS_COMMIT],
        ["checkout", "--detach", CORPUS_COMMIT],
    ):
        subprocess.run(["git", "-C", str(RAW_DIR), *args], check=True)


def prefetch_models() -> None:
    AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        use_fast=True,
    )
    SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        device="cpu",
        trust_remote_code=False,
    )
    CrossEncoder(
        RERANKER_MODEL,
        revision=RERANKER_REVISION,
        device="cpu",
        max_length=RERANKER_MAX_LENGTH,
        trust_remote_code=False,
    )


def build_heading_indexes() -> None:
    processed = REPO_ROOT / "data" / "processed"
    documents = processed / "docs.jsonl"

    write_documents(documents)
    write_heading_dataset(documents, processed / "chunks_heading.jsonl")
    build_index("heading")
    write_bm25_index("heading")


if __name__ == "__main__":
    fetch_corpus()
    prefetch_models()
    build_heading_indexes()
