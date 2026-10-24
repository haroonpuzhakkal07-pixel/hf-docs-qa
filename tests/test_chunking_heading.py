from src.chunking import count_tokens
from src.chunking_heading import split_to_token_limit

import pytest

import hashlib
import json

from src.chunking_heading import write_heading_dataset


def test_oversized_line_preserves_source_and_counts_heading_prefix():
    # One oversized line, with surrounding whitespace and Unicode.
    source = (
        "  ```py\n"
        "text = '" + "hello café 中文 " * 250 + "'\n"
        "```\n\n"
    )
    prefix = "Summarization > Inspect the dataset\n\n"

    assert count_tokens(source.splitlines()[1]) > 400

    records = split_to_token_limit(source, prefix=prefix)

    assert len(records) > 1
    assert "".join(record["source_text"] for record in records) == source

    cursor = 0
    for record in records:
        assert record["start_char"] == cursor
        assert record["end_char"] > cursor
        assert record["source_text"] == source[
            record["start_char"]:record["end_char"]
        ]
        assert record["text"] == prefix + record["source_text"]
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= 400
        cursor = record["end_char"]

    assert cursor == len(source)


@pytest.mark.parametrize("overlap", [0, 30])
def test_heading_chunker_preserves_fitting_code_and_bounds_large_fragments(
    overlap,
):
    from src.chunking_heading import chunk_by_heading
    from src.markdown_structure import find_fenced_blocks

    small_code = "    ```python\n    print('hello')\n    ```\n"
    large_code = (
        "``````markdown\n"
        "```python\n"
        "value = '" + "hello café 中文 " * 250 + "'\n"
        "```\n"
        "~~~~\n"
        "``````\n"
    )
    text = (
        "# Guide\n\n"
        "## Small\n\n" + small_code +
        "\n## Large\n\n" + large_code
    )

    records = chunk_by_heading(text, overlap=overlap)

    assert any(small_code in record["source_text"] for record in records)

    fragments = [record for record in records if record["code_fragment"]]
    assert len(fragments) > 1

    cursor = 0
    recovered = []
    for record in records:
        start, end = record["start_char"], record["end_char"]

        assert 0 <= start <= cursor < end
        assert record["source_text"] == text[start:end]
        assert record["text"].startswith(record["heading_path"] + "\n\n")
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= 400

        # Recover only the newly covered portion of this source slice.
        recovered.append(record["source_text"][cursor - start:])
        cursor = end

    assert cursor == len(text)
    assert "".join(recovered) == text

    if overlap:
        assert any(
            current["start_char"] < previous["end_char"]
            for previous, current in zip(fragments, fragments[1:])
        )

    for fragment in fragments:
        [outer] = find_fenced_blocks(fragment["text"])
        assert outer["closed"] is True
        assert outer["marker"] == fragment["fragment_fence"]
        assert len(outer["marker"]) > 4


@pytest.mark.parametrize("indent", ["", "    "])
def test_oversized_multiline_content_prefers_complete_lines(indent):
    source = (indent + "value = '" + "hello " * 35 + "'\n") * 20

    records = split_to_token_limit(
        source,
        prefix="Guide > Example\n\n",
    )

    assert len(records) > 1
    assert "".join(record["source_text"] for record in records) == source
    assert all(record["source_text"].endswith("\n") for record in records)
    assert all(record["token_count"] <= 400 for record in records)

    cursor = 0
    for record in records:
        assert record["start_char"] == cursor
        assert record["source_text"] == source[
            record["start_char"]:record["end_char"]
        ]
        cursor = record["end_char"]

    assert cursor == len(source)


def test_overlap_repeats_context_without_gaps_or_stalled_progress():
    source = "  " + "hello world " * 600 + "\n"
    prefix = "Guide > Long example\n\n"

    records = split_to_token_limit(
        source,
        prefix=prefix,
        overlap=30,
    )

    assert len(records) > 1
    covered_until = 0

    for index, record in enumerate(records):
        start, end = record["start_char"], record["end_char"]

        assert start <= covered_until < end
        if index:
            assert start < covered_until

        assert record["source_text"] == source[start:end]
        assert record["text"] == prefix + source[start:end]
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= 400
        covered_until = end

    assert records[0]["start_char"] == 0
    assert covered_until == len(source)


@pytest.mark.parametrize(
    "max_tokens, expected_chunks",
    [(100, 2), (400, 1)],
)
def test_small_chunks_merge_with_all_headings_only_when_budget_allows(
    max_tokens, expected_chunks,
):
    from src.chunking_heading import chunk_by_heading

    text = (
        "# Guide\n\n"
        "## Loading\n\n" + "hello " * 45 +
        "\n\n## Running\n\n" + "hello " * 45 + "\n"
    )

    records = chunk_by_heading(
        text,
        max_tokens=max_tokens,
        min_tokens=80,
        overlap=0,
    )

    assert len(records) == expected_chunks
    assert "".join(record["source_text"] for record in records) == text

    paths = list(dict.fromkeys(
        path
        for record in records
        for path in record["heading_paths"]
    ))
    assert paths == ["Guide", "Guide > Loading", "Guide > Running"]

    for record in records:
        assert record["source_text"] == text[
            record["start_char"]:record["end_char"]
        ]
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= max_tokens


def test_packed_paragraphs_overlap_without_gaps_or_exceeding_limit():
    from src.chunking_heading import chunk_by_heading

    text = "# Guide\n\n" + ("hello " * 170 + "\n\n") * 6

    records = chunk_by_heading(text, min_tokens=0, overlap=30)

    assert len(records) > 1
    assert any(
        current["start_char"] < previous["end_char"]
        for previous, current in zip(records, records[1:])
    )

    cursor = 0
    recovered = []
    for record in records:
        start, end = record["start_char"], record["end_char"]

        assert 0 <= start <= cursor < end
        assert record["source_text"] == text[start:end]
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= 400

        recovered.append(record["source_text"][cursor - start:])
        cursor = end

    assert cursor == len(text)
    assert "".join(recovered) == text


def test_separator_after_large_code_belongs_to_next_section():
    from src.chunking_heading import chunk_by_heading
    from src.markdown_structure import find_fenced_blocks

    text = (
        "# Guide\n\n"
        "## Example\n\n"
        "```python\n" + "hello " * 500 +
        "\n```\n\n"
        "## Resources\n\nUseful reference material.\n"
    )

    records = chunk_by_heading(text)
    [source_fence] = find_fenced_blocks(text)

    assert all(record["source_text"].strip() for record in records)
    assert any(
        record["source_text"].startswith("\n## Resources")
        for record in records
    )

    cursor = 0
    recovered = []
    for record in records:
        start, end = record["start_char"], record["end_char"]
        assert 0 <= start <= cursor < end
        assert record["source_text"] == text[start:end]
        assert record["token_count"] == count_tokens(record["text"])
        assert record["token_count"] <= 400

        if record["code_fragment"]:
            assert source_fence["start_char"] <= start
            assert end <= source_fence["end_char"]

        recovered.append(record["source_text"][cursor - start:])
        cursor = end

    assert cursor == len(text)
    assert "".join(recovered) == text


def test_heading_writer_saves_metadata_and_matching_provenance(tmp_path):
    source = tmp_path / "docs.jsonl"
    document = {
        "doc_id": "example.md",
        "path": "example.md",
        "title": "Example",
        "text": "# Example\n\nUseful content: café 中文.\n",
    }
    source.write_text(
        json.dumps(document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks_heading.jsonl"

    manifest = write_heading_dataset(source, output)
    saved_bytes = output.read_bytes()
    [record] = [
        json.loads(line)
        for line in saved_bytes.decode("utf-8").splitlines()
    ]

    assert record["chunk_id"] == "example.md::heading::00000"
    assert record["doc_id"] == record["path"] == "example.md"
    assert record["source_text"] == document["text"]
    assert record["token_count"] == count_tokens(record["text"])
    assert manifest["input_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(saved_bytes).hexdigest()
    assert manifest["statistics"]["documents"] == 1
    assert manifest["statistics"]["chunks"] == 1
    assert json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    ) == manifest


@pytest.mark.parametrize("existing", ["dataset", "manifest"])
def test_heading_writer_refuses_either_existing_output(tmp_path, existing):
    output = tmp_path / "chunks_heading.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    occupied = output if existing == "dataset" else manifest_path
    other = manifest_path if existing == "dataset" else output
    occupied.write_bytes(b"existing content")

    with pytest.raises(FileExistsError):
        write_heading_dataset(tmp_path / "unused.jsonl", output)

    assert occupied.read_bytes() == b"existing content"
    assert not other.exists()