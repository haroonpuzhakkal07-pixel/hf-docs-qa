"""Tests for src/ingest.py.

Part 1 uses small inline pages and always runs (also in CI).
Part 2 checks the real cleaned corpus in data/processed/docs.jsonl and is
skipped when that file does not exist (run `python -m src.ingest` first).
"""
import json
import re
import tempfile
from pathlib import Path

import pytest

from src import config, ingest

LICENSE = """<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

-->
"""


def clean(raw):
    return ingest.clean_markdown(raw)[0]


# ---------------------------------------------------------------------------
# Part 1: unit tests on inline pages
# ---------------------------------------------------------------------------
def test_license_comment_is_removed_and_text_starts_at_title():
    out = clean(LICENSE + "\n# Quicktour\n\nHello.\n")
    assert out == "# Quicktour\n\nHello."


def test_page_without_license_is_unchanged():
    assert clean("# Tracing\n\nText.\n") == "# Tracing\n\nText."


def test_non_license_leading_comment_is_removed():
    raw = "<!--Note that this file is in Markdown but contains doc-builder syntax.\n-->\n\n# Community\n"
    assert clean(raw) == "# Community"


def test_extra_comments_removed_outside_code_but_kept_inside_code():
    raw = (
        "# T\n\nIntro.\n\n<!-- TODO -->\n<!-- multi\nline -->\n\n"
        "```html\n<!-- keep me -->\n<div>x</div>\n```\n"
    )
    out = clean(raw)
    assert "TODO" not in out and "multi" not in out
    assert "<!-- keep me -->\n<div>x</div>" in out


def test_code_fence_content_is_byte_for_byte_unchanged():
    code = '```py\n[[autodoc]] X\n[a](b)\n<div>\n> [!NOTE]\nx = "![i](j)"\n```'
    out = clean("# T\n\n" + code + "\n")
    assert code in out


def test_date_line_moves_under_title():
    raw = (
        LICENSE
        + "\n*This model was published in HF papers on 2018-10-11 and contributed to "
        "Hugging Face Transformers on 2020-11-16.*\n\n"
        '<div style="float: right;">\n    <img alt="x" src="https://a/b.png">\n</div>\n\n'
        "# BERT\n\n[BERT](https://x.org/p) is a model.\n"
    )
    out = clean(raw)
    lines = out.split("\n")
    assert lines[0] == "# BERT"
    assert lines[1] == ""
    assert lines[2].startswith("*This model was published")
    assert lines[3] == ""
    assert lines[4] == "BERT is a model."
    assert "<div" not in out and "<img" not in out


def test_date_line_after_title_is_left_alone():
    raw = "# BERT\n\ntext\n\n*This model was published on 2018-10-11.*\n"
    assert clean(raw) == "# BERT\n\ntext\n\n*This model was published on 2018-10-11.*"


def test_autodoc_and_option_lines_removed_next_section_kept():
    raw = (
        "# T\n\n## BertConfig\n\n[[autodoc]] BertConfig\n\n"
        "## BertModel\n\n[[autodoc]] BertModel\n    - forward\n    - all\n\n## Next\n\nKept text.\n"
    )
    out = clean(raw)
    assert "autodoc" not in out and "forward" not in out and "- all" not in out
    assert "## BertModel" in out and "Kept text." in out


def test_other_directives_removed():
    out = clean("# T\n\n[[open-in-colab]]\n\nText with [[nested]] stays.\n")
    assert "open-in-colab" not in out
    assert "Text with [[nested]] stays." in out


def test_hfoptions_become_option_labels_and_code_survives():
    raw = (
        '# T\n\n<hfoptions id="usage">\n<hfoption id="Pipeline">\n\n```py\nx = 1\n```\n\n</hfoption>\n'
        '<hfoption id="curl (stream)">\n\ncurl text\n\n</hfoption>\n</hfoptions>\n'
    )
    out = clean(raw)
    assert "<hfoption" not in out and "</hfoption" not in out
    assert "Option: Pipeline" in out and "Option: curl (stream)" in out
    assert "```py\nx = 1\n```" in out


def test_callouts_become_plain_prefixes():
    raw = "# T\n\n> [!WARNING]\n> Be careful.\n> Really.\n\n> [!TIP] Inline tip.\n\n> A normal quote.\n"
    out = clean(raw)
    assert "[!" not in out
    assert "Warning: Be careful.\nReally." in out
    assert "Tip: Inline tip." in out
    assert "> A normal quote." in out


def test_links_images_and_crossrefs():
    raw = (
        "# T\n\nSee [the guide](./tasks/qa) and [site](https://x.org/a_(b)) and [`~PreTrainedModel.generate`] "
        "and [`Trainer`].\n\n![alt](https://x/y.png)\n\n"
        "[![badge](https://img/b.svg)](https://arxiv.org/abs/1)\n\nCode `[a](b)` stays.\n"
    )
    out = clean(raw)
    assert "See the guide and site and `PreTrainedModel.generate` and `Trainer`." in out
    assert "![" not in out and "img/b.svg" not in out and "arxiv.org" not in out
    assert "Code `[a](b)` stays." in out


def test_html_table_text_is_kept_and_tokens_in_prose_are_not_eaten():
    raw = (
        "# T\n\n<table>\n<tr><th>Name</th><th>Desc</th></tr>\n"
        "<tr><td><code>sdpa</code></td><td>Uses<br>torch</td></tr>\n</table>\n\nA <pad> token.\n"
    )
    out = clean(raw)
    assert "| Name | Desc |" in out
    assert "| sdpa | Uses torch |" in out
    assert "<table" not in out and "<tr" not in out
    assert "<pad>" in out


def test_markdown_tables_are_kept():
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert table in clean("# T\n\n" + table + "\n")


def test_heading_anchor_removed():
    assert clean("# T\n\n## Done[[done]]\n") == "# T\n\n## Done"


def test_blank_line_runs_collapse():
    assert clean("# T\n\n\n\n\ntext\n\n\n") == "# T\n\ntext"


def test_title_prefers_h1_outside_fences_then_other_heading_then_filename():
    t, s = ingest.extract_title("```bash\n# not a title\n```\n\n# Real\n", "fb")
    assert (t, s) == ("Real", "h1")
    t, s = ingest.extract_title("```bash\n# not a title\n```\n\n## Overview\n", "fb")
    assert (t, s) == ("Overview", "other_heading")
    t, s = ingest.extract_title("just text\n", "fb")
    assert (t, s) == ("fb", "filename")


def test_fences_balanced_detects_unclosed_fence():
    assert ingest.fences_balanced("```py\nx\n```\n")
    assert not ingest.fences_balanced("```py\nx\n")
    assert ingest.fences_balanced("````md\n```py\nx\n```\n````\n")


def test_tip_tags_become_prefixes():
    raw = "# T\n\n<Tip>\n\nBe careful with `x`.\n\n</Tip>\n\n<Tip warning={true}>\n\nDanger.\n\n</Tip>\n"
    out = clean(raw)
    assert "<Tip" not in out and "</Tip" not in out
    assert "Tip: Be careful with `x`." in out
    assert "Warning: Danger." in out


def test_tip_around_a_code_fence_keeps_the_fence_intact():
    raw = "# T\n\n<Tip>\n\nRun:\n\n```bash\npip install x\n```\n\n</Tip>\n\nAfter.\n"
    out = clean(raw)
    assert "```bash\npip install x\n```" in out
    assert "<Tip" not in out and "</Tip" not in out
    assert ingest.fences_balanced(out)


def test_youtube_tags_removed():
    out = clean('# T\n\n<Youtube id="abc"/>\n\nText.\n')
    assert "Youtube" not in out and "Text." in out


def test_empty_link_keeps_its_text():
    assert "Gaudi supports torch.compile." in clean("# T\n\nGaudi supports [torch.compile]().\n")


def test_html_headings_become_markdown_headings_and_code_tags_are_dropped():
    out = clean("# T\n\n<h3>Sub title</h3>\n\n<code>foo</code> and <attr> stay\n")
    assert "### Sub title" in out
    assert "foo and <attr> stay" in out


def test_hfoptions_blocks_are_counted():
    raw = '# T\n\n<hfoptions id="a">\n<hfoption id="x">\n\nt\n\n</hfoption>\n<hfoption id="y">\n\nu\n\n</hfoption>\n</hfoptions>\n'
    touched = ingest.clean_markdown(raw)[1]
    assert touched["hfoptions_block"] == 1 and touched["hfoption"] == 2


def test_page_without_h1_gets_a_title_from_its_file_name_and_date_line_follows_it():
    raw = (
        LICENSE
        + "*This model was published in HF papers on 2025-10-30 and contributed to "
        "Hugging Face Transformers on 2026-09-04.*\n\n## Overview\n\nKimi Linear is a model.\n"
    )
    out, touched = ingest.clean_markdown(raw, fallback_title=ingest.title_from_stem("kimi_linear"))
    lines = out.split("\n")
    assert lines[0] == "# Kimi Linear"
    assert lines[2].startswith("*This model was published")
    assert lines[4] == "## Overview"
    assert touched["title_added"] == 1


def test_title_is_not_added_when_an_h1_exists_or_only_inside_a_code_fence_counts_as_none():
    out, touched = ingest.clean_markdown("# Real\n\ntext\n", fallback_title="Fallback")
    assert out.startswith("# Real") and "title_added" not in touched
    out, touched = ingest.clean_markdown("```bash\n# comment\n```\n\ntext\n", fallback_title="Fallback")
    assert out.startswith("# Fallback\n\n```bash")


def test_title_from_stem():
    assert ingest.title_from_stem("kimi_linear") == "Kimi Linear"
    assert ingest.title_from_stem("add-new-model") == "Add New Model"


def test_pipe_entity_decoded_in_prose_but_not_in_inline_code():
    out = clean("# T\n\n<table><tr><td><code>paged&#124;sdpa</code></td><td>x</td></tr></table>\n\nKeep `a&#124;b`.\n")
    assert "| paged|sdpa | x |" in out
    assert "`a&#124;b`" in out


def test_build_records_on_a_temp_tree():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tasks").mkdir()
        (root / "a.md").write_text(LICENSE + "\n# Alpha\n\ntext\n", encoding="utf-8", newline="\n")
        (root / "tasks" / "b.md").write_text("# Beta\n\nunicode: \u00e9\u4e2d\n", encoding="utf-8", newline="\n")
        records, meta = ingest.build_records(root, ["tasks/b.md", "a.md"])
        assert [r["path"] for r in records] == ["a.md", "tasks/b.md"]
        assert [r["doc_id"] for r in records] == ["a", "tasks/b"]
        assert [r["title"] for r in records] == ["Alpha", "Beta"]
        out = root / "out" / "docs.jsonl"
        ingest.write_jsonl(records, out)
        back = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert back == records
        assert set(back[0]) == {"doc_id", "path", "title", "text"}


# ---------------------------------------------------------------------------
# Part 2: checks on the real cleaned corpus
# ---------------------------------------------------------------------------
def load_docs():
    if not ingest.DOCS_JSONL.exists():
        pytest.skip("data/processed/docs.jsonl not found; run: python -m src.ingest")
    with open(ingest.DOCS_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def offenders(docs, predicate):
    return [d["path"] for d in docs if predicate(d)]


def test_corpus_page_count_matches_include():
    docs = load_docs()
    assert len(docs) == len(config.INCLUDE)
    assert sorted(d["path"] for d in docs) == sorted(config.INCLUDE)


def test_corpus_ids_are_unique_and_use_forward_slashes():
    docs = load_docs()
    assert len({d["doc_id"] for d in docs}) == len(docs)
    assert not offenders(docs, lambda d: "\\" in d["path"] or "\\" in d["doc_id"])
    assert not offenders(docs, lambda d: not d["path"].endswith(".md"))


def test_corpus_no_page_is_empty_and_every_page_has_a_title():
    docs = load_docs()
    assert not offenders(docs, lambda d: not d["text"].strip())
    assert not offenders(docs, lambda d: not d["title"].strip())


def test_corpus_has_no_license_text_left():
    docs = load_docs()
    needles = [
        "Apache License, Version 2.0",
        "WITHOUT WARRANTIES OR CONDITIONS",
        "Copyright 20",
        "All rights reserved",
        "contains specific syntax for our doc-builder",
    ]
    # A leaked header would sit at the very top. Pages that talk about licenses
    # later on (for example modeling_rules.md) are real content.
    bad = offenders(docs, lambda d: any(n in d["text"][:1000] for n in needles))
    assert not bad, "license text still at the top of: %s" % bad[:10]


def test_corpus_has_no_leftover_markup():
    docs = load_docs()
    checks = {
        "html comment": re.compile(r"<!--"),
        "autodoc": re.compile(r"\[\[autodoc\]\]"),
        "directive line": re.compile(r"^\s*\[\[[A-Za-z][\w-]*\]\]\s*$", re.MULTILINE),
        "hfoption tag": re.compile(r"</?hfoptions?\b"),
        "callout marker": re.compile(r"^>\s*\[!", re.MULTILINE),
        "markdown image": re.compile(r"!\[[^\]]*\]\("),
        "markdown link": re.compile(r"\[[^\]]*\]\([^)]*\)"),
        "html layout tag": re.compile(r"</?(?:div|span|img|table|tr|td|th)\b", re.IGNORECASE),
        "tip tag": re.compile(r"</?Tip\b", re.IGNORECASE),
        "youtube tag": re.compile(r"</?Youtube\b", re.IGNORECASE),
        "html heading": re.compile(r"</?h[1-6]\b", re.IGNORECASE),
    }
    problems = []
    for d in docs:
        prose = ingest.prose_only(d["text"])
        for name, rx in checks.items():
            m = rx.search(prose)
            if m:
                s = max(m.start() - 30, 0)
                problems.append("%s: %s: %r" % (d["path"], name, prose[s : m.end() + 30]))
    assert not problems, "\n".join(problems[:10])


def test_corpus_code_fences_are_balanced():
    docs = load_docs()
    bad = offenders(docs, lambda d: not ingest.fences_balanced(d["text"]))
    assert not bad, "unclosed code fence in: %s" % bad[:10]


def test_corpus_date_line_sits_under_the_title():
    docs = load_docs()
    with_date = []
    wrong = []
    for d in docs:
        lines = d["text"].split("\n")
        idx = [i for i, ln in enumerate(lines) if ingest.DATE_LINE_RE.match(ln)]
        if not idx:
            continue
        with_date.append(d["path"])
        if not (lines[0].startswith("# ") and idx[0] == 2):
            wrong.append(d["path"])
    assert len(with_date) >= 18, "expected at least 18 pages with a date line, got %d" % len(with_date)
    assert not wrong, "date line not directly under the title in: %s" % wrong[:10]


def test_corpus_every_page_starts_with_its_title():
    docs = load_docs()
    bad = offenders(docs, lambda d: not d["text"].startswith("# "))
    assert not bad, "page does not start with a '# ' title: %s" % bad[:10]
    bad = offenders(docs, lambda d: d["text"].split("\n")[0] != "# " + d["title"])
    assert not bad, "title does not match the first line in: %s" % bad[:10]


def test_corpus_pipe_entity_is_decoded():
    docs = load_docs()
    bad = offenders(docs, lambda d: "&#124;" in ingest.prose_only(d["text"]))
    assert not bad, "&#124; still in prose of: %s" % bad[:10]
