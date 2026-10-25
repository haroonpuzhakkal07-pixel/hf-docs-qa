import hashlib
import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from src import embeddings
from src import index as indexing
from src.config import CORPUS_COMMIT, EMBEDDING_MODEL, EMBEDDING_REVISION


@pytest.fixture(params=["fixed", "heading"])
def index_case(tmp_path, monkeypatch, request):
    chunker = request.param
    original_root = indexing.REPO_ROOT

    source_dir = tmp_path / "data" / "processed"
    source_dir.mkdir(parents=True)
    code_dir = tmp_path / "src"
    code_dir.mkdir()
    for name in ("config.py", "embeddings.py", "index.py"):
        (code_dir / name).write_bytes(
            (original_root / "src" / name).read_bytes()
        )

    records = []
    for row, text in enumerate(("Short passage.", "A longer café passage.")):
        record = {
            "chunk_id": f"example.md::{chunker}::{row:05d}",
            "doc_id": "example.md",
            "path": "example.md",
            "heading_path": "",
            "text": text,
            "token_count": 10,
            "start_char": row * 100,
            "end_char": row * 100 + len(text),
        }
        if chunker == "heading":
            record.update(
                heading_path="Example > Usage",
                heading_paths=["Example > Usage"],
                source_text=text,
                text=f"Example > Usage\n\n{text}",
                code_fragment=False,
                fragment_fence="",
            )
        records.append(record)

    # These are writer fixtures, not regenerated chunker outputs.
    raw = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")
    source = source_dir / f"chunks_{chunker}.jsonl"
    source.write_bytes(raw)
    source.with_suffix(".manifest.json").write_text(
        json.dumps({
            "chunker": chunker,
            "corpus_commit": CORPUS_COMMIT,
            "config": {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_revision": EMBEDDING_REVISION,
            },
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "statistics": {"chunks": len(records)},
        }),
        encoding="utf-8",
    )

    vectors = np.eye(2, dtype=np.float32)
    calls = []

    def fake_encode(texts, *, batch_size):
        calls.append((texts, batch_size))
        return vectors.copy()

    def fake_git(arguments, **kwargs):
        if arguments == ["git", "rev-parse", "HEAD"]:
            return "test-commit\n"
        if arguments == [
            "git", "status", "--porcelain", "--untracked-files=normal"
        ]:
            return "?? src/index.py\n"
        raise AssertionError(f"Unexpected Git command: {arguments}")

    monkeypatch.setattr(indexing, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(indexing, "encode_texts", fake_encode)
    monkeypatch.setattr(
        indexing, "get_embedding_model",
        lambda: SimpleNamespace(max_seq_length=512),
    )
    monkeypatch.setattr(indexing.subprocess, "check_output", fake_git)

    return SimpleNamespace(
        chunker=chunker,
        source=source,
        output=tmp_path / "data" / "index" / chunker,
        records=records,
        raw=raw,
        vectors=vectors,
        calls=calls,
    )


def test_saved_index_preserves_rows_metadata_and_provenance(index_case):
    case = index_case
    manifest = indexing.build_index(case.chunker, batch_size=2)

    assert case.calls == [([r["text"] for r in case.records], 2)]
    assert (case.output / "chunks.jsonl").read_bytes() == case.raw
    saved_records = [
        json.loads(line)
        for line in (case.output / "chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert saved_records == case.records

    saved_index = faiss.read_index(str(case.output / "index.faiss"))
    assert saved_index.ntotal == 2
    assert saved_index.d == 2
    assert saved_index.metric_type == faiss.METRIC_INNER_PRODUCT

    scores, rows = saved_index.search(case.vectors, 1)
    assert rows[:, 0].tolist() == [0, 1]
    np.testing.assert_allclose(scores[:, 0], 1.0)
    for row in range(2):
        np.testing.assert_array_equal(
            saved_index.reconstruct(row), case.vectors[row]
        )
        assert saved_records[rows[row, 0]] == case.records[row]

    saved_manifest = json.loads(
        (case.output / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest == manifest
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256(
            (case.output / name).read_bytes()
        ).hexdigest() == digest
    assert manifest["input"]["sha256"] == hashlib.sha256(case.raw).hexdigest()
    assert manifest["git_commit"] == "test-commit"
    assert manifest["git_dirty"] is True
    assert manifest["embedding"]["revision"] == EMBEDDING_REVISION
    assert manifest["embedding"]["passage_prompt"] == ""
    assert manifest["embedding"]["query_prompt"] == embeddings.QUERY_PREFIX


def test_existing_directory_is_preserved_without_encoding(index_case):
    case = index_case
    case.output.mkdir(parents=True)
    sentinel = case.output / "keep.txt"
    sentinel.write_bytes(b"existing output")

    with pytest.raises(FileExistsError):
        indexing.build_index(case.chunker)

    assert sentinel.read_bytes() == b"existing output"
    assert list(case.output.iterdir()) == [sentinel]
    assert case.calls == []


def test_changed_dataset_is_rejected_before_encoding(index_case):
    case = index_case
    case.source.write_bytes(case.raw + b"\n")

    with pytest.raises(ValueError, match="hash"):
        indexing.build_index(case.chunker)

    assert case.calls == []
    assert not case.output.exists()


@pytest.mark.parametrize("prompt", ["", embeddings.QUERY_PREFIX])
def test_overlong_input_is_rejected_before_inference(monkeypatch, prompt):
    model = embeddings.get_embedding_model()
    # A passage exceeds the limit directly; a query exceeds it via its prefix.
    words = model.max_seq_length if not prompt else model.max_seq_length - 2
    text = "hello " * words
    full_ids = model.tokenizer(
        prompt + text, truncation=False, add_special_tokens=True
    )["input_ids"]
    assert len(full_ids) > model.max_seq_length

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Inference must not run for an overlong input")

    monkeypatch.setattr(model, "encode", unexpected_encode)
    with pytest.raises(ValueError, match="Text was not truncated"):
        embeddings.encode_texts([text], prompt=prompt)


@pytest.fixture
def saved_search(index_case, monkeypatch):
    from src import retrieve

    case = index_case
    indexing.build_index(case.chunker)
    retrieve.load_index.cache_clear()
    monkeypatch.setattr(retrieve, "REPO_ROOT", case.output.parents[2])
    query_calls = []

    def fake_query_encode(texts, *, prompt):
        query_calls.append((texts, prompt))
        return np.array([[0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(retrieve, "encode_texts", fake_query_encode)
    try:
        yield retrieve, case, query_calls
    finally:
        retrieve.load_index.cache_clear()


def test_search_uses_query_prefix_and_returns_complete_records(saved_search):
    retrieve, case, calls = saved_search

    results = retrieve.dense_search("memory query", case.chunker, k=1)

    assert calls == [(["memory query"], embeddings.QUERY_PREFIX)]
    assert results == [{**case.records[1], "score": 1.0}]

    # Asking for more neighbors than exist must not introduce row -1.
    results = retrieve.dense_search("memory query", case.chunker, k=99)
    assert len(results) == 2
    assert results == [
        {**case.records[1], "score": 1.0},
        {**case.records[0], "score": 0.0},
    ]


@pytest.mark.parametrize("artifact", ["index.faiss", "chunks.jsonl"])
def test_search_rejects_changed_artifacts(saved_search, artifact):
    retrieve, case, calls = saved_search
    path = case.output / artifact
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        retrieve.dense_search("memory query", case.chunker)

    assert calls == []


def test_search_rejects_incompatible_model_revision(saved_search):
    retrieve, case, calls = saved_search
    path = case.output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["embedding"]["revision"] = "different-revision"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding setting: revision"):
        retrieve.dense_search("memory query", case.chunker)

    assert calls == []


@pytest.mark.parametrize(
    "query, k",
    [(" \n ", 5), ("memory query", 0), ("memory query", -1)],
)
def test_search_rejects_invalid_requests_before_encoding(
    saved_search, query, k
):
    retrieve, case, calls = saved_search

    with pytest.raises(ValueError):
        retrieve.dense_search(query, case.chunker, k)

    assert calls == []