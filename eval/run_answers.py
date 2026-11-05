"""Generate one configuration of answers on the validated DEV split."""

import argparse
import json
from collections import Counter
from time import sleep

from eval.check_questions import check_questions
from src.config import REPO_ROOT
from src.generate import answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=("closed_book", "bm25_heading"))
    args = parser.parse_args()

    eval_dir = REPO_ROOT / "eval"
    output = eval_dir / "results" / "answers" / f"{args.config}.jsonl"
    if output.exists():
        raise SystemExit(f"Result already exists: {output}")

    records = check_questions(eval_dir / "questions.jsonl")
    dev = [record for record in records if record["split"] == "dev"]
    counts = Counter(record["answerable"] for record in dev)
    if counts != {True: 36, False: 9}:
        raise SystemExit(f"Unexpected DEV counts: {dict(counts)}")

    rows = []
    for position, record in enumerate(dev, start=1):
        question_id = record["id"]
        try:
            result = answer(record["question"], args.config)
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"{args.config} {question_id}: {error}") from None

        rows.append({
            "id": question_id,
            "question": record["question"],
            "split": record["split"],
            "answerable": record["answerable"],
            "config": args.config,
            **result,
        })
        source = "cache" if result["cache_hit"] else "fresh"
        print(f"{position}/{len(dev)} {question_id} ({source})", flush=True)
        if not result["cache_hit"] and position < len(dev):
            sleep(10)

    jsonl = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(jsonl)

    hits = sum(row["cache_hit"] for row in rows)
    original_total = sum(row["token_usage"]["total_tokens"] for row in rows)
    api_total = sum(row["api_usage"]["total_tokens"] for row in rows)
    print(f"Saved {len(rows)} DEV answers to {output.relative_to(REPO_ROOT)}")
    print(f"Cache hits: {hits}/{len(rows)}")
    print(f"Original response tokens: {original_total:,}")
    print(f"Current API response tokens: {api_total:,}")


if __name__ == "__main__":
    main()