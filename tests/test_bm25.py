import pytest
from copy import deepcopy
import json
import pickle

from src import bm25
from src.bm25 import BM25_SETTINGS, build_bm25, tokenize

from types import SimpleNamespace

import numpy as np


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "AutoModel.from_pretrained('MODEL_ID')",
            ["automodel", "from_pretrained", "model_id"],
        ),
        (
            "Qwen3-30B-A3B uses 8-bit weights.",
            ["qwen3", "30b", "a3b", "uses", "8", "bit", "weights"],
        ),
        (
            "device_map='auto'; torch.float16",
            ["device_map", "auto", "torch", "float16"],
        ),
        ("cache CACHE cache", ["cache", "cache", "cache"]),
        ("How do I load a model?", ["how", "do", "i", "load", "a", "model"]),
        ("", []),
        (" \n\t...!? ", []),
    ],
)
def test_tokenization_policy(text, expected):
    assert tokenize(text) == expected


@pytest.mark.parametrize("text", [None, 123, ["from_pretrained"]])
def test_rejects_non_string_input(text):
    with pytest.raises(ValueError, match="text must be a string"):
        tokenize(text)


def test_build_preserves_complete_records_and_token_row_alignment():
    records = [
        {
            "chunk_id": "example:0",
            "path": "example.md",
            "text": "AutoModel.from_pretrained('MODEL_ID')",
        },
        {
            "chunk_id": "example:1",
            "path": "example.md",
            "text": "cache cache",
        },
        {
            "chunk_id": "punctuation:0",
            "path": "punctuation.md",
            "text": "...",
        },
    ]
    original = deepcopy(records)

    model, tokenized = build_bm25(records)
    scores = model.get_scores(tokenize("from_pretrained"))

    assert records == original
    assert tokenized == [
        ["automodel", "from_pretrained", "model_id"],
        ["cache", "cache"],
        [],
    ]
    assert model.corpus_size == len(records)
    assert model.doc_len == [3, 2, 0]
    assert scores[0] > scores[1]
    assert scores[1] == scores[2] == 0
    for name, value in BM25_SETTINGS.items():
        assert getattr(model, name) == value


@pytest.mark.parametrize(
    "records, message",
    [
        ([], "Chunk dataset is empty"),
        (
            [
                {"chunk_id": "duplicate", "text": "first"},
                {"chunk_id": "duplicate", "text": "second"},
            ],
            "Duplicate chunk IDs",
        ),
        ([{"chunk_id": "", "text": "content"}], "Invalid chunk ID"),
        ([{"chunk_id": 123, "text": "content"}], "Invalid chunk ID"),
        (
            [{"chunk_id": "example:0", "text": "..."}],
            "no searchable tokens",
        ),
    ],
)
def test_build_rejects_unusable_input(records, message):
    with pytest.raises(ValueError, match=message):
        build_bm25(records)


@pytest.fixture(params=["fixed", "heading"])
def bm25_case(tmp_path, monkeypatch, request):
    index_name = request.param
    original_root = bm25.REPO_ROOT
    records = [
        {
            "chunk_id": "example:0",
            "path": "example.md",
            "text": "from_pretrained('café')\n",
        },
        {
            "chunk_id": "example:1",
            "path": "example.md",
            "text": "cache cache",
        },
        {
            "chunk_id": "punctuation:0",
            "path": "punctuation.md",
            "text": "...",
        },
    ]
    source = tmp_path / "data" / "processed" / f"chunks_{index_name}.jsonl"
    source.parent.mkdir(parents=True)
    source_bytes = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")
    source.write_bytes(source_bytes)
    source.with_suffix(".manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "chunker": index_name,
            "corpus_commit": bm25.CORPUS_COMMIT,
            "output_sha256": bm25.sha256(source_bytes),
            "statistics": {"chunks": len(records)},
        }),
        encoding="utf-8",
    )

    code_dir = tmp_path / "src"
    code_dir.mkdir()
    for name in ("bm25.py", "config.py"):
        (code_dir / name).write_bytes(
            (original_root / "src" / name).read_bytes()
        )

    monkeypatch.setattr(bm25, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        bm25.subprocess,
        "check_output",
        lambda arguments, **kwargs: (
            "fixture-commit\n"
            if arguments[1:] == ["rev-parse", "HEAD"]
            else ""
        ),
    )
    output = tmp_path / "data" / "index" / "bm25" / index_name
    bm25.load_bm25_index.cache_clear()
    yield index_name, source, output, records, source_bytes
    bm25.load_bm25_index.cache_clear()


def test_writer_preserves_mapping_and_serializes_aligned_tokens(bm25_case):
    index_name, source, output, records, original_bytes = bm25_case

    manifest = bm25.write_bm25_index(index_name)
    saved_manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    tokenized = pickle.loads((output / "tokens.pkl").read_bytes())

    assert saved_manifest == manifest
    assert source.read_bytes() == original_bytes
    assert (output / "chunks.jsonl").read_bytes() == original_bytes
    assert tokenized == [tokenize(record["text"]) for record in records]
    assert tokenized[-1] == []
    assert manifest["index"]["rows"] == len(records)
    assert manifest["statistics"]["tokenless_chunks"] == 1
    assert manifest["input"]["sha256"] == bm25.sha256(original_bytes)
    assert manifest["input"]["manifest_sha256"] == bm25.sha256(
        source.with_suffix(".manifest.json").read_bytes()
    )
    for name, digest in manifest["artifacts"].items():
        assert bm25.sha256((output / name).read_bytes()) == digest
    assert {path.name for path in output.iterdir()} == {
        "chunks.jsonl", "tokens.pkl", "manifest.json",
    }


@pytest.mark.parametrize("complete", [False, True])
def test_writer_refuses_complete_or_partial_existing_directory(
    bm25_case, complete,
):
    index_name, source, output, _, original_bytes = bm25_case
    if complete:
        bm25.write_bm25_index(index_name)
    else:
        output.mkdir(parents=True)
        (output / "partial.txt").write_bytes(b"Preserve this file.\n")

    before = {
        path.name: path.read_bytes()
        for path in output.iterdir()
    }

    with pytest.raises(FileExistsError, match="already exists"):
        bm25.write_bm25_index(index_name)

    assert {
        path.name: path.read_bytes()
        for path in output.iterdir()
    } == before
    assert source.read_bytes() == original_bytes


def test_writer_rejects_changed_source_before_creating_output(bm25_case):
    index_name, source, output, _, original_bytes = bm25_case
    source.write_bytes(original_bytes + b"\n")

    with pytest.raises(ValueError, match="hash does not match"):
        bm25.write_bm25_index(index_name)

    assert not output.exists()


def test_search_returns_complete_matching_record(bm25_case):
    index_name, _, _, records, _ = bm25_case
    bm25.write_bm25_index(index_name)

    results = bm25.bm25_search("FROM_PRETRAINED", index_name, k=10)

    assert len(results) == 1
    assert {key: value for key, value in results[0].items()
            if key != "score"} == records[0]
    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] > 0


@pytest.mark.parametrize("query", ["...", "unknown_identifier_zzz"])
def test_search_returns_no_results_without_token_overlap(bm25_case, query):
    index_name, _, _, _, _ = bm25_case
    bm25.write_bm25_index(index_name)

    assert bm25.bm25_search(query, index_name) == []


def test_equal_scores_keep_row_order_and_exclude_nonmatching_rows(monkeypatch):
    records = [
        {"chunk_id": "first", "path": "first.md", "text": "cache"},
        {"chunk_id": "second", "path": "second.md", "text": "cache"},
        {"chunk_id": "other", "path": "other.md", "text": "other"},
    ]
    model = SimpleNamespace(
        doc_freqs=[{"cache": 1}, {"cache": 1}, {"other": 1}],
        get_scores=lambda tokens: np.array([0.0, 0.0, 99.0]),
    )
    monkeypatch.setattr(
        bm25, "load_bm25_index", lambda name: (model, records),
    )

    results = bm25.bm25_search("cache", "fixed", k=10)

    assert [record["chunk_id"] for record in results] == ["first", "second"]
    assert bm25.bm25_search("cache", "fixed", k=1) == results[:1]


@pytest.mark.parametrize("artifact", ["chunks.jsonl", "tokens.pkl"])
def test_loader_checks_hashes_before_deserializing(
    bm25_case, monkeypatch, artifact,
):
    index_name, _, output, _, _ = bm25_case
    bm25.write_bm25_index(index_name)
    (output / artifact).write_bytes(b"Changed artifact.")

    def unexpected_deserialization(data):
        pytest.fail("Must reject changed artifacts before reading pickle")

    monkeypatch.setattr(bm25.pickle, "loads", unexpected_deserialization)

    with pytest.raises(ValueError, match=f"Artifact hash mismatch: {artifact}"):
        bm25.load_bm25_index(index_name)


def test_loader_rejects_reordered_tokens_even_with_updated_hash(bm25_case):
    index_name, _, output, _, _ = bm25_case
    bm25.write_bm25_index(index_name)
    token_path = output / "tokens.pkl"
    tokenized = pickle.loads(token_path.read_bytes())
    changed_bytes = pickle.dumps(list(reversed(tokenized)), protocol=4)
    token_path.write_bytes(changed_bytes)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["tokens.pkl"] = bm25.sha256(changed_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved tokens do not match"):
        bm25.load_bm25_index(index_name)


@pytest.mark.parametrize(
    "query, index_name, k",
    [
        (None, "fixed", 5),
        ("", "fixed", 5),
        ("   ", "fixed", 5),
        ("cache", "fixed", 0),
        ("cache", "fixed", -1),
        ("cache", "fixed", True),
        ("cache", "fixed", 1.5),
        ("cache", "invalid", 5),
    ],
)
def test_search_rejects_invalid_arguments_before_loading(
    monkeypatch, query, index_name, k,
):
    def unexpected_load(name):
        pytest.fail("Invalid arguments must fail before loading")

    monkeypatch.setattr(bm25, "load_bm25_index", unexpected_load)

    with pytest.raises(ValueError):
        bm25.bm25_search(query, index_name, k)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("corpus_commit", "wrong-commit", "corpus"),
        ("tokenizer", {}, "tokenizer"),
        ("bm25", {}, "settings"),
    ],
)
def test_loader_rejects_incompatible_manifest(
    bm25_case, field, value, message,
):
    index_name, _, output, _, _ = bm25_case
    bm25.write_bm25_index(index_name)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        bm25.load_bm25_index(index_name)