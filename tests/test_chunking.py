import pytest

from src.chunking import chunk_fixed, count_tokens, get_tokenizer
from src.config import MAX_CHUNK_TOKENS

import hashlib
import json

from src.chunking import write_fixed_dataset


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_or_whitespace_input_produces_no_chunks(text):
    assert chunk_fixed(text) == []


@pytest.mark.parametrize(
    "size, overlap",
    [
        (0, 0),
        (-1, 0),
        (300, -1),
        (300, 300),
        (300, 301),
        (MAX_CHUNK_TOKENS, 0),
    ],
)
def test_rejects_invalid_window_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed("Example text.", size=size, overlap=overlap)


@pytest.mark.parametrize(
    "length, expected_chunks",
    [
        (1, 1),
        (300, 1),
        (301, 2),
        (550, 2),
        (551, 3),
    ],
)
def test_window_boundaries_and_token_overlap(length, expected_chunks):
    tokenizer = get_tokenizer()
    text = " ".join(["hello"] * length)
    source_ids = tokenizer.encode(text, add_special_tokens=False)

    assert len(source_ids) == length

    chunks = chunk_fixed(text)
    assert len(chunks) == expected_chunks

    for index, chunk in enumerate(chunks):
        start = index * 250
        expected_ids = source_ids[start : start + 300]
        actual_ids = tokenizer.encode(
            chunk["text"], add_special_tokens=False
        )

        assert actual_ids == expected_ids
        assert chunk["token_count"] == len(actual_ids) + 2
        assert chunk["token_count"] <= MAX_CHUNK_TOKENS


def test_preserves_original_text_and_complete_coverage():
    text = (
        "  # Example\n\n"
        "Unicode: café 中文. Keep <image> and `inline_code`.\n\n"
        "```python\n"
        "    print('hello')\n"
        "```\n\n"
        "| Setting | Value |\n"
        "| --- | --- |\n"
        "| size | 128 |\n\n"
    ) * 40

    chunks = chunk_fixed(text)
    assert len(chunks) > 1
    assert chunks[0]["start_char"] == 0

    covered_until = 0

    for chunk in chunks:
        start = chunk["start_char"]
        end = chunk["end_char"]

        assert 0 <= start < end <= len(text)
        assert start <= covered_until
        assert end > covered_until
        assert chunk["text"] == text[start:end]
        assert chunk["text"].strip()
        assert chunk["token_count"] == count_tokens(chunk["text"])
        assert chunk["token_count"] <= MAX_CHUNK_TOKENS

        covered_until = end

    assert covered_until == len(text)


def test_zero_overlap_preserves_source_without_gaps():
    text = "  " + "hello " * 700 + "\n"
    chunks = chunk_fixed(text, overlap=0)

    assert "".join(chunk["text"] for chunk in chunks) == text


def test_short_document_remains_unchanged():
    text = "  # Small document\n\nUseful content: café 中文.\n\n"
    chunks = chunk_fixed(text)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(text)


def test_repeated_runs_produce_identical_chunks():
    text = "hello world " * 500
    assert chunk_fixed(text) == chunk_fixed(text)


def test_chunk_records_preserve_document_metadata():
    from src.chunking import build_fixed_chunks

    documents = [
        {
            "doc_id": "first.md",
            "path": "first.md",
            "title": "First",
            "text": "# First\n\nUseful content.",
        },
        {
            "doc_id": "second.md",
            "path": "second.md",
            "title": "Second",
            "text": "# Second\n\nDifferent content.",
        },
    ]

    records = build_fixed_chunks(documents)

    assert len(records) == 2

    for document, record in zip(documents, records):
        assert record["chunk_id"] == f"{document['doc_id']}::fixed::00000"
        assert record["doc_id"] == document["doc_id"]
        assert record["path"] == document["path"]
        assert record["heading_path"] == ""
        assert record["text"] == document["text"]
        assert record["start_char"] == 0
        assert record["end_char"] == len(document["text"])

    assert records == build_fixed_chunks(documents)


def test_chunk_records_reject_duplicate_documents_and_empty_text():
    from src.chunking import build_fixed_chunks

    document = {
        "doc_id": "example.md",
        "path": "example.md",
        "text": "Useful content.",
    }

    with pytest.raises(ValueError, match="Duplicate document ID"):
        build_fixed_chunks([document, document])

    with pytest.raises(ValueError, match="produced no chunks"):
        build_fixed_chunks([{**document, "text": " \n\t "}])


@pytest.fixture
def document_file(tmp_path):
    documents = [
        {
            "doc_id": "first.md",
            "path": "first.md",
            "title": "First",
            "text": "# First\n\nUnicode: café 中文.",
        },
        {
            "doc_id": "second.md",
            "path": "second.md",
            "title": "Second",
            "text": "# Second\n\nDifferent useful content.",
        },
    ]
    source = tmp_path / "docs.jsonl"
    source.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents),
        encoding="utf-8",
    )
    return source


def test_writer_saves_chunks_and_matching_manifest(document_file, tmp_path):
    output = tmp_path / "processed" / "chunks_fixed.jsonl"
    manifest = write_fixed_dataset(document_file, output)

    output_bytes = output.read_bytes()
    records = [
        json.loads(line)
        for line in output_bytes.decode("utf-8").splitlines()
    ]
    saved_manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved_manifest == manifest
    assert manifest["input_sha256"] == hashlib.sha256(
        document_file.read_bytes()
    ).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(output_bytes).hexdigest()
    assert manifest["statistics"]["documents"] == 2
    assert manifest["statistics"]["chunks"] == len(records) == 2
    assert manifest["config"]["token_count_includes_special_tokens"] is True
    assert isinstance(manifest["git_dirty"], bool)

    assert {record["doc_id"] for record in records} == {"first.md", "second.md"}
    assert "café 中文" in records[0]["text"]


@pytest.mark.parametrize("existing", ["dataset", "manifest"])
def test_writer_refuses_existing_outputs(document_file, tmp_path, existing):
    output = tmp_path / "chunks_fixed.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    occupied = output if existing == "dataset" else manifest_path
    other = manifest_path if existing == "dataset" else output

    occupied.write_bytes(b"existing content")

    with pytest.raises(FileExistsError):
        write_fixed_dataset(document_file, output)

    assert occupied.read_bytes() == b"existing content"
    assert not other.exists()


def test_writer_rejects_empty_dataset_without_creating_outputs(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text("", encoding="utf-8")
    output = tmp_path / "chunks_fixed.jsonl"

    with pytest.raises(ValueError, match="no documents"):
        write_fixed_dataset(source, output)

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()