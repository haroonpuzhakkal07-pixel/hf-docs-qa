"""BM25 keyword retrieval over preserved chunk records."""

import re
from rank_bm25 import BM25Okapi
import hashlib
import json
import pickle
import platform
import subprocess
from importlib.metadata import version
from functools import lru_cache
import numpy as np
import argparse

from src.config import CORPUS_COMMIT, REPO_ROOT


TOKEN_PATTERN = r"[a-z0-9_]+"

BM25_SETTINGS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}

SEARCH_SETTINGS = {
    "candidate_filter": "at least one query token occurs in chunk",
    "tie_break": "ascending saved row",
}


def tokenize(text: str) -> list[str]:
    """Lowercase and extract ASCII word tokens, preserving underscores.

    Keep repeated terms. Apply no stemming or stopword removal.
    This representation is separate from the original chunk text.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    return re.findall(TOKEN_PATTERN, text.lower())


def build_bm25(records: list[dict]) -> tuple[BM25Okapi, list[list[str]]]:
    """Build BM25 from complete chunk texts in their original row order."""
    if not records:
        raise ValueError("Chunk dataset is empty")

    ids = [record["chunk_id"] for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Invalid chunk ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs")

    tokenized = [tokenize(record["text"]) for record in records]
    if not any(tokenized):
        raise ValueError("Chunk dataset has no searchable tokens")

    # Keep empty token lists so row numbers still match the chunk records.
    model = BM25Okapi(tokenized, **BM25_SETTINGS)
    return model, tokenized


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bm25_index(index_name: str) -> dict:
    """Save tokenized chunks and provenance in a new BM25 directory."""
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    source = (
        REPO_ROOT / "data" / "processed" / f"chunks_{index_name}.jsonl"
    )
    output = REPO_ROOT / "data" / "index" / "bm25" / index_name
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    source_bytes = source.read_bytes()
    source_manifest_bytes = source.with_suffix(".manifest.json").read_bytes()
    source_manifest = json.loads(source_manifest_bytes)

    if source_manifest["schema_version"] != 1:
        raise ValueError("Unsupported chunk manifest schema")
    if source_manifest["chunker"] != index_name:
        raise ValueError("Chunker does not match its manifest")
    if source_manifest["corpus_commit"] != CORPUS_COMMIT:
        raise ValueError("Chunk corpus does not match the project pin")
    if sha256(source_bytes) != source_manifest["output_sha256"]:
        raise ValueError("Chunk dataset hash does not match its manifest")

    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
    ]
    if len(records) != source_manifest["statistics"]["chunks"]:
        raise ValueError("Chunk count does not match its manifest")

    _, tokenized = build_bm25(records)
    token_bytes = pickle.dumps(tokenized, protocol=4)

    def git_output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True,
        ).strip()

    manifest = {
        "schema_version": 1,
        "retriever": "bm25",
        "chunker": index_name,
        "corpus_commit": CORPUS_COMMIT,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output(
            "status", "--porcelain", "--untracked-files=all",
        )),
        "tokenizer": {
            "pattern": TOKEN_PATTERN,
            "lowercase": True,
            "stemming": False,
            "remove_stopwords": False,
        },
        "bm25": {"class": "BM25Okapi", **BM25_SETTINGS},
        "search": dict(SEARCH_SETTINGS),
        "index": {
            "rows": len(records),
            "mapping": "token list row equals chunks.jsonl record order",
            "pickle_protocol": 4,
        },
        "statistics": {
            "tokens": sum(len(tokens) for tokens in tokenized),
            "tokenless_chunks": sum(not tokens for tokens in tokenized),
        },
        "input": {
            "path": source.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(source_bytes),
            "manifest_sha256": sha256(source_manifest_bytes),
            "manifest": source_manifest,
        },
        "artifacts": {
            "chunks.jsonl": sha256(source_bytes),
            "tokens.pkl": sha256(token_bytes),
        },
        "implementation_sha256": {
            name: sha256((REPO_ROOT / name).read_bytes())
            for name in ("src/bm25.py", "src/config.py")
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("rank-bm25", "numpy")
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")

    # Reserve a new directory. Write the completion manifest last.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("chunks.jsonl", source_bytes),
        ("tokens.pkl", token_bytes),
        ("manifest.json", manifest_bytes),
    ):
        with (output / name).open("xb") as stream:
            stream.write(data)

    return manifest


@lru_cache(maxsize=2)
def load_bm25_index(index_name: str):
    """Load locally generated artifacts after hash and alignment checks."""
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    folder = REPO_ROOT / "data" / "index" / "bm25" / index_name
    manifest = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported BM25 manifest schema")
    if manifest["retriever"] != "bm25" or manifest["chunker"] != index_name:
        raise ValueError("BM25 manifest does not match requested index")
    if manifest["corpus_commit"] != CORPUS_COMMIT:
        raise ValueError("BM25 corpus does not match the project pin")
    if manifest["tokenizer"] != {
        "pattern": TOKEN_PATTERN,
        "lowercase": True,
        "stemming": False,
        "remove_stopwords": False,
    }:
        raise ValueError("Incompatible BM25 tokenizer")
    if manifest["bm25"] != {"class": "BM25Okapi", **BM25_SETTINGS}:
        raise ValueError("Incompatible BM25 settings")
    if manifest["search"] != SEARCH_SETTINGS:
        raise ValueError("Incompatible BM25 search policy")
    if manifest["packages"]["rank-bm25"] != version("rank-bm25"):
        raise ValueError("Installed rank-bm25 version differs from manifest")

    artifacts = {
        name: (folder / name).read_bytes()
        for name in ("chunks.jsonl", "tokens.pkl")
    }
    for name, data in artifacts.items():
        if sha256(data) != manifest["artifacts"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")

    if sha256(artifacts["chunks.jsonl"]) != manifest["input"]["sha256"]:
        raise ValueError("Chunk mapping does not match the recorded input")

    records = [
        json.loads(line)
        for line in artifacts["chunks.jsonl"].decode("utf-8").splitlines()
    ]
    if len(records) != manifest["index"]["rows"]:
        raise ValueError("BM25 row count does not match its manifest")

    # Pickle inputs are restricted to our locally generated artifacts.
    tokenized = pickle.loads(artifacts["tokens.pkl"])
    model, expected_tokens = build_bm25(records)
    if tokenized != expected_tokens:
        raise ValueError("Saved tokens do not match chunk text and row order")

    return model, records


def bm25_search(query: str, index_name: str, k: int = 5) -> list[dict]:
    """Return complete scored chunk records with lexical overlap."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    model, records = load_bm25_index(index_name)
    unique_tokens = set(query_tokens)
    candidates = [
        row
        for row, frequencies in enumerate(model.doc_freqs)
        if unique_tokens.intersection(frequencies)
    ]
    if not candidates:
        return []

    scores = model.get_scores(query_tokens)
    if scores.shape != (len(records),) or not np.isfinite(scores).all():
        raise ValueError("BM25 returned invalid scores")

    ranked_rows = sorted(
        candidates,
        key=lambda row: (-float(scores[row]), row),
    )[:k]
    return [
        {**records[row], "score": float(scores[row])}
        for row in ranked_rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="Build a new BM25 index")
    build_parser.add_argument(
        "--index", choices=("fixed", "heading"), required=True,
    )

    search_parser = commands.add_parser("search", help="Search a saved index")
    search_parser.add_argument(
        "--index", choices=("fixed", "heading"), required=True,
    )
    search_parser.add_argument("--k", type=int, default=5)
    search_parser.add_argument("query", nargs="+")

    args = parser.parse_args()
    if args.command == "build":
        manifest = write_bm25_index(args.index)
        print(
            f"Saved {manifest['index']['rows']} rows "
            f"to data/index/bm25/{args.index}"
        )
        print(json.dumps(manifest["statistics"], indent=2))
        return

    results = bm25_search(" ".join(args.query), args.index, k=args.k)
    if not results:
        print("No chunks share a query token.")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result['score']:.6f}  "
            f"{result['path']}  chunk={result['chunk_id']}"
        )
        print(result["text"])
        print()


if __name__ == "__main__":
    main()