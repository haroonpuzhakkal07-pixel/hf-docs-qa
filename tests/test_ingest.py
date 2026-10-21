import json

import pytest

from src import ingest


@pytest.mark.parametrize(
    "opening",
    [
        "<!-- Copyright",
        "<!---Copyright",
        "<!---\nCopyright",
        "<-- Copyright",
        "<!--\n\nLicensed under",
    ],
)
def test_removes_leading_license_variants(opening):
    raw = f"{opening} example license\n-->\n\n# Example\n\nUseful text."
    assert ingest.clean_source(raw) == "# Example\n\nUseful text."


@pytest.mark.parametrize("closing", ["---", "..."])
def test_removes_front_matter_and_license(closing):
    license_header = "<!-- Copyright example -->\n"
    front_matter = f"---\ntitle: Example\n{closing}\n"
    body = "# Example\n\nUseful text."

    for prefix in (
        license_header + front_matter,
        front_matter + license_header,
    ):
        assert ingest.clean_source(prefix + body) == body


def test_preserves_answer_bearing_content():
    body = (
        "# Example\n\n"
        "This model is licensed under Apache 2.0.\n\n"
        "```python\n"
        "# A code comment\n"
        "print('hello')\n"
        "```\n\n"
        "| Setting | Value |\n"
        "| --- | --- |\n"
        "| size | 128 |\n\n"
        "> [!TIP]\n"
        "> Keep this guidance."
    )

    assert ingest.clean_source("<!-- Copyright example -->\n" + body) == body


def test_rejects_unclosed_leading_metadata():
    with pytest.raises(ValueError, match="Unclosed leading license"):
        ingest.clean_source("<!-- Copyright example")

    with pytest.raises(ValueError, match="Unclosed leading YAML"):
        ingest.clean_source("---\ntitle: Example")


def test_title_ignores_comments_and_nested_code_fences():
    text = (
        "<!-- # Hidden heading -->\n"
        "``````markdown\n"
        "```python\n"
        "# Code heading\n"
        "```\n"
        "# Still inside the outer fence\n"
        "``````\n"
        "# Actual title [[api-reference]]\n"
    )

    assert ingest.extract_title(text, "example.md") == "Actual title"


def test_explicit_title_fallback():
    assert (
        ingest.extract_title("## Overview\n\nModel details.", "model_doc/qwen3_next.md")
        == "Qwen3-Next"
    )

    with pytest.raises(ValueError, match="No H1"):
        ingest.extract_title("## Overview", "unknown.md")


def test_loader_rejects_empty_cleaned_page(tmp_path, monkeypatch):
    (tmp_path / "empty.md").write_text(
        "<!-- Copyright example -->", encoding="utf-8"
    )
    monkeypatch.setattr(ingest, "DOCS_ROOT", tmp_path)
    monkeypatch.setattr(ingest, "INCLUDE", ["empty.md"])

    with pytest.raises(ValueError, match="Empty document"):
        ingest.load_documents()


def test_jsonl_writer_preserves_content_and_refuses_overwrite(tmp_path, monkeypatch):
    documents = [
        {
            "doc_id": "example.md",
            "path": "example.md",
            "title": "Example",
            "text": "# Example\n\nUnicode: café 中文",
        }
    ]
    monkeypatch.setattr(ingest, "load_documents", lambda: documents)
    output = tmp_path / "processed" / "docs.jsonl"

    assert ingest.write_documents(output) == 1
    original = output.read_bytes()
    saved = [
        json.loads(line)
        for line in original.decode("utf-8").splitlines()
    ]
    assert saved == documents

    with pytest.raises(FileExistsError):
        ingest.write_documents(output)

    assert output.read_bytes() == original


def test_selected_corpus_has_content_and_no_leading_license():
    documents = ingest.load_documents()

    assert len(documents) == len(ingest.INCLUDE)
    assert {doc["path"] for doc in documents} == set(ingest.INCLUDE)

    for doc in documents:
        assert doc["title"].strip(), doc["path"]
        assert doc["text"].strip(), doc["path"]
        assert not ingest.LICENSE_START.match(
            doc["text"].lstrip()
        ), f"Leading license remains: {doc['path']}"


def test_tab_cleanup_preserves_labels_and_contents():
    source = (
        '<hfoptions id="backend">\n'
        '<hfoption id="FSDP">\n'
        "Use fsdp_config.\n"
        "</hfoption>\n"
        '<hfoption id="DeepSpeed">\n'
        "Use deepspeed.\n"
        "</hfoption>\n"
        "</hfoptions>\n"
    )

    assert ingest.clean_tab_markup(source) == (
        "Options: backend\n"
        "Option: FSDP\n"
        "Use fsdp_config.\n"
        "Option: DeepSpeed\n"
        "Use deepspeed.\n"
    )


def test_tab_cleanup_preserves_code_and_unknown_markup():
    source = (
        "``````markdown\n"
        "```html\n"
        '<hfoption id="literal example">\n'
        "</hfoption>\n"
        "```\n"
        "``````\n"
        '`<hfoption id="inline example">`\n'
        '<hfoption custom="unknown">\n'
        "<image>\n"
        "<think>\n"
    )

    assert ingest.clean_tab_markup(source) == source