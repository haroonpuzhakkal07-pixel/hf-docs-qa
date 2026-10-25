"""Dense search over saved, verified chunk indexes."""

import argparse
import hashlib
import json
from functools import lru_cache

import faiss

from src.config import (
    CORPUS_COMMIT,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    REPO_ROOT,
)
from src.embeddings import QUERY_PREFIX, encode_texts


@lru_cache(maxsize=2)
def load_index(index_name: str):
    """Load each index once per process after checking its saved artifacts."""
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    folder = REPO_ROOT / "data" / "index" / index_name
    manifest = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported index manifest schema")
    if manifest["chunker"] != index_name:
        raise ValueError("Index chunker does not match requested configuration")
    if manifest["corpus_commit"] != CORPUS_COMMIT:
        raise ValueError("Index corpus does not match project configuration")

    settings = manifest["embedding"]
    expected = {
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "normalize_embeddings": True,
        "precision": "float32",
        "passage_prompt": "",
        "query_prompt": QUERY_PREFIX,
    }
    for name, value in expected.items():
        if settings[name] != value:
            raise ValueError(f"Incompatible index embedding setting: {name}")

    # Check both files before deserializing the index.
    for name in ("index.faiss", "chunks.jsonl"):
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != manifest["artifacts"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")

    mapping_bytes = (folder / "chunks.jsonl").read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != manifest["input"]["sha256"]:
        raise ValueError("Chunk mapping does not match the recorded input")

    records = [
        json.loads(line)
        for line in mapping_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Index chunk mapping is empty")
    if len({record["chunk_id"] for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs in index mapping")

    index = faiss.read_index(str(folder / "index.faiss"))
    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("Expected an IndexFlatIP index")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Expected inner-product search")
    if index.ntotal != len(records) or index.ntotal != manifest["index"]["rows"]:
        raise ValueError("Index and mapping row counts differ")
    if index.d != settings["dimension"]:
        raise ValueError("Index and manifest dimensions differ")

    return index, records


def dense_search(query: str, index_name: str, k: int = 5) -> list[dict]:
    """Return scored chunk records in descending similarity order."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    index, records = load_index(index_name)
    query_vector = encode_texts([query], prompt=QUERY_PREFIX)
    if query_vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension differs from index")

    # FAISS returns -1 for missing neighbors if k exceeds the index size.
    scores, rows = index.search(query_vector, min(k, index.ntotal))
    results = []
    for score, row in zip(scores[0], rows[0]):
        if not 0 <= row < len(records):
            raise ValueError(f"Invalid FAISS result row: {row}")
        results.append({**records[int(row)], "score": float(score)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("fixed", "heading"), required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()

    results = dense_search(" ".join(args.query), args.index, args.k)
    for rank, result in enumerate(results, start=1):
        snippet = result["text"][:150].replace("\n", " ").replace("\r", " ")
        heading = result["heading_path"] or "(no heading)"
        print(f"{rank}. score={result['score']:.6f}  {result['path']}")
        print(f"   heading: {heading}")
        print(f"   chunk: {result['chunk_id']}")
        print(f"   {snippet}")
        print()


if __name__ == "__main__":
    main()