import pytest

from src.markdown_structure import find_fenced_blocks


@pytest.mark.parametrize("indent", ["", "  ", "   ", "    ", "     ", "\t"])
def test_indented_fences_preserve_exact_source_span(indent):
    block = f"{indent}```python\n{indent}print('hello')\n{indent}```\n"
    text = "Before\n" + block + "After\n"

    [found] = find_fenced_blocks(text)

    assert text[found["start_char"]:found["end_char"]] == block
    assert found["marker"] == "```"
    assert found["info"] == "python"
    assert found["closed"] is True


def test_shorter_different_and_nonclosing_markers_stay_inside():
    text = (
        "``````text\n"
        "```python\n"
        "# Not a document heading\n"
        "```\n"
        "~~~~~~\n"
        "``````not-a-closer\n"
        "```````\n"
    )

    [found] = find_fenced_blocks(text)

    assert found["start_char"] == 0
    assert found["end_char"] == len(text)
    assert found["closed"] is True


def test_comments_are_ignored_but_comment_syntax_in_code_is_literal():
    text = (
        "<!--\n"
        "```hidden\n"
        "-->\n"
        "~~~html\n"
        "<!-- literal code content\n"
        "~~~\n"
    )

    [found] = find_fenced_blocks(text)

    assert text[found["start_char"]:found["end_char"]] == (
        "~~~html\n<!-- literal code content\n~~~\n"
    )
    assert found["closed"] is True


def test_unclosed_fence_extends_to_end_without_dropping_content():
    text = "Before\n```python\nprint('hello')"

    [found] = find_fenced_blocks(text)

    assert text[found["start_char"]:found["end_char"]] == (
        "```python\nprint('hello')"
    )
    assert found["closed"] is False


def test_sections_preserve_source_and_follow_visible_heading_hierarchy():
    from src.markdown_structure import find_heading_sections

    text = (
        "Preamble with useful context.\n\n"
        "# Guide [[guide]]\n"
        "Introduction.\n"
        "````text\n"
        "```python\n"
        "## Not a section\n"
        "```\n"
        "<!-- literal code comment\n"
        "````\n"
        "<!--\n"
        "## Hidden heading\n"
        "-->\n"
        "### Usage ###\n"
        "Example.\n"
        "#### Detail\n"
        "Keep this in Usage.\n"
        "## Reference\n"
        "API details.\n"
        "# Other\n"
        "Final content.\n\n"
    )

    sections = find_heading_sections(text)

    assert [section["heading_path"] for section in sections] == [
        "",
        "Guide",
        "Guide > Usage",
        "Guide > Reference",
        "Other",
    ]
    assert sections[0]["start_char"] == 0
    assert sections[-1]["end_char"] == len(text)
    for previous, current in zip(sections, sections[1:]):
        assert previous["end_char"] == current["start_char"]
    assert "".join(
        text[section["start_char"]:section["end_char"]]
        for section in sections
    ) == text


def test_unclosed_fence_prevents_false_section_boundaries():
    from src.markdown_structure import find_heading_sections

    text = "# Guide\n\n    ```python\n## Still code\nprint('hello')"

    assert find_heading_sections(text) == [{
        "start_char": 0,
        "end_char": len(text),
        "heading_path": "Guide",
    }]


def test_content_blocks_preserve_source_and_ignore_blank_lines_inside_fences():
    from src.markdown_structure import find_content_blocks

    prose = "First paragraph.\n\nSecond paragraph.\n\n"
    fence = (
        "    ````markdown\n"
        "```python\n"
        "print('hello')\n\n"
        "# Still inside the outer fence\n"
        "```\n"
        "    ````\n"
    )
    text = prose + fence + "\nFinal paragraph.\n\n"

    blocks = find_content_blocks(text)

    fenced = [block for block in blocks if block["kind"] == "fence"]
    assert len(fenced) == 1
    assert text[
        fenced[0]["start_char"]:fenced[0]["end_char"]
    ] == fence

    assert [text[b["start_char"]:b["end_char"]] for b in blocks[:2]] == [
        "First paragraph.\n\n",
        "Second paragraph.\n\n",
    ]

    cursor = 0
    for block in blocks:
        assert block["start_char"] == cursor
        assert block["end_char"] > cursor
        cursor = block["end_char"]

    assert cursor == len(text)
    assert "".join(
        text[b["start_char"]:b["end_char"]] for b in blocks
    ) == text