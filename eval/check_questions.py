"""Validate evaluation records and their references to the selected corpus."""

import json
import subprocess
from collections import Counter
from pathlib import Path

from src.config import CORPUS_COMMIT, DOCS_ROOT, INCLUDE, RAW_DIR, REPO_ROOT
from src.ingest import clean_source
from src.markdown_structure import find_heading_sections


REQUIRED_FIELDS = {
    "id", "question", "type", "answerable",
    "gold_answer", "gold_sources", "split",
}
QUESTION_TYPES = {"lookup", "multi_step", "new_or_changed", "unanswerable"}


def check_questions(
    path: Path,
    *,
    docs_root: Path = DOCS_ROOT,
    included_paths=None,
) -> list[dict]:
    """Validate labels without running retrieval or generation."""
    selected = set(INCLUDE if included_paths is None else included_paths)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Question dataset is empty")

    records = []
    seen_questions = set()
    heading_cache = {}

    for number, line in enumerate(lines, start=1):
        label = f"Line {number}"

        def require(condition, message):
            if not condition:
                raise ValueError(f"{label}: {message}")

        require(bool(line.strip()), "blank lines are not allowed")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}: invalid JSON: {error.msg}") from error

        require(isinstance(record, dict), "record must be an object")
        require(
            REQUIRED_FIELDS <= record.keys(),
            f"missing fields: {sorted(REQUIRED_FIELDS - record.keys())}",
        )
        require(
            record["id"] == f"q{number:03d}",
            "IDs must follow authoring order: q001, q002, ...",
        )
        expected_split = "test" if number % 4 == 0 else "dev"
        require(
            record["split"] == expected_split,
            f"split must be {expected_split!r} for this authoring position",
        )

        question = record["question"]
        require(
            isinstance(question, str) and bool(question.strip()),
            "question must be a nonempty string",
        )
        normalized = " ".join(question.casefold().split())
        require(normalized not in seen_questions, "duplicate question")
        seen_questions.add(normalized)

        kind = record["type"]
        require(
            isinstance(kind, str) and kind in QUESTION_TYPES,
            "invalid question type",
        )
        require(type(record["answerable"]) is bool, "answerable must be boolean")
        require(
            record["answerable"] == (kind != "unanswerable"),
            "answerable disagrees with question type",
        )

        answer = record["gold_answer"]
        require(isinstance(answer, str), "gold_answer must be a string")
        if record["answerable"]:
            require(bool(answer.strip()), "answerable question needs a gold answer")

        sources = record["gold_sources"]
        require(isinstance(sources, list), "gold_sources must be a list")
        require(
            bool(sources) == record["answerable"],
            "answerable questions need sources; unanswerable ones need none",
        )
        seen_sources = set()

        for source in sources:
            require(isinstance(source, dict), "source must be an object")
            require(
                set(source) == {"path", "heading"},
                "source must contain exactly path and heading",
            )
            relative_path = source["path"]
            heading = source["heading"]
            require(
                isinstance(relative_path, str) and relative_path in selected,
                f"source path is not in the selected corpus: {relative_path!r}",
            )
            require(
                isinstance(heading, str) and bool(heading.strip()),
                "source heading must be a nonempty string",
            )
            pair = (relative_path, heading)
            require(pair not in seen_sources, "duplicate source reference")
            seen_sources.add(pair)

            source_path = docs_root / relative_path
            require(source_path.is_file(), f"source file missing: {relative_path}")
            if relative_path not in heading_cache:
                text = clean_source(source_path.read_text(encoding="utf-8"))
                heading_cache[relative_path] = {
                    section["heading_path"]
                    for section in find_heading_sections(text)
                    if section["heading_path"]
                }
            require(
                heading in heading_cache[relative_path],
                f"heading not found in {relative_path}: {heading!r}",
            )

        records.append(record)

    return records


def main() -> None:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(RAW_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != CORPUS_COMMIT:
        raise ValueError("Downloaded corpus commit does not match the project pin")

    records = check_questions(REPO_ROOT / "eval" / "questions.jsonl")
    print(f"Validated {len(records)} questions.")
    print("Types:", dict(Counter(record["type"] for record in records)))
    print("Splits:", dict(Counter(record["split"] for record in records)))
    print("Corpus pin, selected source paths and heading references: OK")


if __name__ == "__main__":
    main()