"""Build dense indexes with preserved chunk mappings and provenance."""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version

import faiss
import numpy as np
import torch

from src.config import (
    CORPUS_COMMIT,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    REPO_ROOT,
)
from src.embeddings import QUERY_PREFIX, encode_texts, get_embedding_model


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_index(chunker: str, *, batch_size: int = 64) -> dict:
    """Build one index, refusing any existing output directory."""
    if chunker not in ("fixed", "heading"):
        raise ValueError("chunker must be fixed or heading")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source = REPO_ROOT / "data" / "processed" / f"chunks_{chunker}.jsonl"
    output = REPO_ROOT / "data" / "index" / chunker
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    source_bytes = source.read_bytes()
    source_manifest_bytes = source.with_suffix(".manifest.json").read_bytes()
    source_manifest = json.loads(source_manifest_bytes)

    if sha256(source_bytes) != source_manifest["output_sha256"]:
        raise ValueError("Chunk dataset hash does not match its manifest")
    if source_manifest["chunker"] != chunker:
        raise ValueError("Chunker does not match its manifest")
    if source_manifest["corpus_commit"] != CORPUS_COMMIT:
        raise ValueError("Corpus commit does not match project configuration")

    config = source_manifest["config"]
    if (
        config["embedding_model"] != EMBEDDING_MODEL
        or config["embedding_revision"] != EMBEDDING_REVISION
    ):
        raise ValueError("Chunk tokenizer pin does not match the embedding pin")

    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Chunk dataset is empty")
    if len(records) != source_manifest["statistics"]["chunks"]:
        raise ValueError("Chunk count does not match its manifest")
    ids = [record["chunk_id"] for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Invalid chunk ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs")

    np.random.seed(0)
    torch.manual_seed(0)
    print(f"Encoding {len(records)} {chunker} passages...", flush=True)
    vectors = encode_texts(
        [record["text"] for record in records],
        batch_size=batch_size,
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != len(records):
        raise ValueError("Index row count does not match chunk count")
    index_bytes = faiss.serialize_index(index).tobytes()

    def git_output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True
        ).strip()

    manifest = {
        "schema_version": 1,
        "chunker": chunker,
        "corpus_commit": CORPUS_COMMIT,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output(
            "status", "--porcelain", "--untracked-files=normal"
        )),
        "embedding": {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "device": "cpu",
            "backend": "torch",
            "precision": "float32",
            "normalize_embeddings": True,
            "passage_prompt": "",
            "query_prompt": QUERY_PREFIX,
            "batch_size": batch_size,
            "max_seq_length": get_embedding_model().max_seq_length,
            "dimension": vectors.shape[1],
            "seed": 0,
            "torch_threads": torch.get_num_threads(),
        },
        "index": {
            "type": "IndexFlatIP",
            "metric": "inner_product",
            "rows": index.ntotal,
            "mapping": "zero-based index row equals chunks.jsonl record order",
        },
        "input": {
            "path": source.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(source_bytes),
            "manifest_sha256": sha256(source_manifest_bytes),
            "manifest": source_manifest,
        },
        "artifacts": {
            "index.faiss": sha256(index_bytes),
            "chunks.jsonl": sha256(source_bytes),
        },
        "code_sha256": {
            name: sha256((REPO_ROOT / "src" / name).read_bytes())
            for name in ("config.py", "embeddings.py", "index.py")
        },
        "packages": {
            name: version(name)
            for name in (
                "sentence-transformers", "faiss-cpu", "torch",
                "transformers", "tokenizers", "huggingface-hub", "numpy",
            )
        },
    }

    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    # Reserve a new directory. A manifest is written last.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("index.faiss", index_bytes),
        ("chunks.jsonl", source_bytes),
        ("manifest.json", manifest_bytes),
    ):
        with (output / name).open("xb") as stream:
            stream.write(data)

    print(f"Saved {index.ntotal} rows to {output}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunker", choices=("fixed", "heading"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_index(args.chunker, batch_size=args.batch_size)


if __name__ == "__main__":
    main()