"""Token-based chunking that preserves original source text."""

from functools import lru_cache

from transformers import AutoTokenizer

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    MAX_CHUNK_TOKENS,
    CORPUS_COMMIT,
    REPO_ROOT
)


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the pinned tokenizer once, using the local cache."""
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Character offsets require a fast tokenizer.")
    return tokenizer


def count_tokens(text: str) -> int:
    """Count the passage as it will be tokenized, including special tokens."""
    return len(
        get_tokenizer().encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    )


def chunk_fixed(
    text: str,
    size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[dict]:
    """Create overlapping token windows using exact source-text slices."""
    tokenizer = get_tokenizer()
    special = tokenizer.num_special_tokens_to_add(pair=False)

    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if size + special > MAX_CHUNK_TOKENS:
        raise ValueError("size plus special tokens exceeds the safety limit")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks = []
    start_token = 0

    while start_token < len(offsets):
        end_token = min(start_token + size, len(offsets))

        # Include leading whitespace in the first chunk and trailing
        # whitespace before the next token in each chunk.
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = (
            len(text)
            if end_token == len(offsets)
            else offsets[end_token][0]
        )

        passage = text[start_char:end_char]
        token_count = count_tokens(passage)

        if not passage.strip():
            raise ValueError("Chunk contains no non-whitespace content")
        if token_count > MAX_CHUNK_TOKENS:
            raise ValueError(
                f"Chunk has {token_count} tokens; "
                f"limit is {MAX_CHUNK_TOKENS}"
            )

        chunks.append(
            {
                "text": passage,
                "token_count": token_count,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        if end_token == len(offsets):
            break

        start_token = end_token - overlap

    return chunks


def build_fixed_chunks(documents: list[dict]) -> list[dict]:
    """Attach source metadata to fixed-size chunks."""
    records = []
    seen_doc_ids = set()

    for document in documents:
        doc_id = document["doc_id"]
        path = document["path"]
        text = document["text"]

        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        seen_doc_ids.add(doc_id)

        chunks = chunk_fixed(text)
        if not chunks:
            raise ValueError(f"Document produced no chunks: {path}")

        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{doc_id}::fixed::{index:05d}",
                    "doc_id": doc_id,
                    "path": path,
                    "heading_path": "",
                    **chunk,
                }
            )

    return records


def write_fixed_dataset(source: Path, output: Path) -> dict:
    """Write chunks and provenance without overwriting existing files."""
    manifest_path = output.with_suffix(".manifest.json")

    for target in (output, manifest_path):
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")

    source_bytes = source.read_bytes()
    documents = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not documents:
        raise ValueError("Input dataset contains no documents")

    records = build_fixed_chunks(documents)

    if len({record["chunk_id"] for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs")

    output_bytes = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )

    counts = [record["token_count"] for record in records]
    manifest = {
        "schema_version": 1,
        "chunker": "fixed",
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "size": FIXED_CHUNK_SIZE,
            "overlap": FIXED_CHUNK_OVERLAP,
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "token_count_includes_special_tokens": True,
        },
        "corpus_commit": CORPUS_COMMIT,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "code_sha256": {
            name: hashlib.sha256(
                (REPO_ROOT / "src" / name).read_bytes()
            ).hexdigest()
            for name in ("config.py", "chunking.py")
        },
        "packages": {
            name: version(name)
            for name in ("transformers", "tokenizers")
        },
        "statistics": {
            "documents": len(documents),
            "chunks": len(records),
            "average_tokens": mean(counts),
            "minimum_tokens": min(counts),
            "maximum_tokens": max(counts),
        },
    }

    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("xb") as stream:
        stream.write(output_bytes)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)

    return manifest


if __name__ == "__main__":
    source = REPO_ROOT / "data" / "processed" / "docs.jsonl"
    output = REPO_ROOT / "data" / "processed" / "chunks_fixed.jsonl"

    manifest = write_fixed_dataset(source, output)
    stats = manifest["statistics"]

    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Average tokens: {stats['average_tokens']:.2f}")
    print(f"Token range: {stats['minimum_tokens']}–{stats['maximum_tokens']}")
    print(f"Saved: {output}")
    print(f"Manifest: {output.with_suffix('.manifest.json')}")
    print(f"Git working tree had uncommitted changes: {manifest['git_dirty']}")