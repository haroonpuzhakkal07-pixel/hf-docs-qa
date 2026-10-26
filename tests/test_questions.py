import json

import pytest

from eval.check_questions import check_questions


@pytest.fixture
def question_case(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "example.md").write_text(
        "# Example\n\n"
        "## Usage\n\n"
        "A supported fact.\n\n"
        "```markdown\n"
        "## Fenced heading\n"
        "```\n\n"
        "<!--\n"
        "## Commented heading\n"
        "-->\n",
        encoding="utf-8",
    )
    (docs / "excluded.md").write_text(
        "# Excluded\n", encoding="utf-8"
    )
    path = tmp_path / "questions.jsonl"

    def validate(records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return check_questions(
            path,
            docs_root=docs,
            included_paths={"example.md", "missing.md"},
        )

    return path, validate


def question(number):
    return {
        "id": f"q{number:03d}",
        "question": f"What is fixture fact number {number}?",
        "type": "lookup",
        "answerable": True,
        "gold_answer": "A supported fact.",
        "gold_sources": [
            {"path": "example.md", "heading": "Example > Usage"}
        ],
        "split": "test" if number % 4 == 0 else "dev",
    }


def test_accepts_valid_records_with_every_fourth_question_held_out(question_case):
    _, validate = question_case
    records = [question(number) for number in range(1, 5)]

    assert validate(records) == records


@pytest.mark.parametrize(
    "heading",
    ["Example > Fenced heading", "Example > Commented heading"],
)
def test_rejects_headings_inside_code_or_comments(question_case, heading):
    _, validate = question_case
    record = question(1)
    record["gold_sources"][0]["heading"] = heading

    with pytest.raises(ValueError, match="Line 1: heading not found"):
        validate([record])


@pytest.mark.parametrize(
    "path, message",
    [
        ("excluded.md", "not in the selected corpus"),
        ("missing.md", "source file missing"),
    ],
)
def test_requires_both_selected_and_existing_source(question_case, path, message):
    _, validate = question_case
    record = question(1)
    record["gold_sources"][0]["path"] = path

    with pytest.raises(ValueError, match=message):
        validate([record])


def test_rejects_incorrect_fourth_question_split(question_case):
    _, validate = question_case
    records = [question(number) for number in range(1, 5)]
    records[3]["split"] = "dev"

    with pytest.raises(ValueError, match="Line 4: split must be 'test'"):
        validate(records)


def test_rejects_duplicate_questions_despite_case_and_spacing(question_case):
    _, validate = question_case
    first, second = question(1), question(2)
    second["question"] = "  " + first["question"].upper().replace(" ", "  ")

    with pytest.raises(ValueError, match="Line 2: duplicate question"):
        validate([first, second])


def test_accepts_unanswerable_question_without_gold_sources(question_case):
    _, validate = question_case
    record = question(1)
    record.update(
        type="unanswerable",
        answerable=False,
        gold_answer="",
        gold_sources=[],
    )

    assert validate([record]) == [record]


def test_reports_invalid_json_with_line_number(question_case):
    path, _ = question_case
    path.write_text(json.dumps(question(1)) + "\nnot JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Line 2: invalid JSON"):
        check_questions(
            path,
            docs_root=path.parent / "docs",
            included_paths={"example.md"},
        )


def test_rejects_multi_step_with_only_one_source(question_case):
    _, validate = question_case
    record = question(1)
    record["type"] = "multi_step"

    with pytest.raises(ValueError, match="at least two distinct source references"):
        validate([record])


def test_accepts_multi_step_with_two_valid_source_sections(question_case):
    _, validate = question_case
    record = question(1)
    record["type"] = "multi_step"
    record["gold_sources"].append(
        {"path": "example.md", "heading": "Example"}
    )

    assert validate([record]) == [record]