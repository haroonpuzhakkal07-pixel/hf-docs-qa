"""Load verified DEV answers and complete passages for judging."""

import hashlib
import json
import argparse
from time import sleep
from collections import Counter

from eval.check_questions import check_questions
from eval.run_answers import GROUNDING_INDEXES
from src.config import REPO_ROOT
from src.evaluate import collect_combined_provenance, collect_provenance
from src import judge
from src.prompting import REFUSAL


def load_judge_inputs(config):
    if config != "closed_book" and config not in GROUNDING_INDEXES:
        raise ValueError(f"Unknown configuration: {config!r}")

    questions_path = REPO_ROOT / "eval" / "questions.jsonl"
    dev = [
        question for question in check_questions(questions_path)
        if question["split"] == "dev"
    ]
    if Counter(question["answerable"] for question in dev) != {True: 36, False: 9}:
        raise ValueError("Unexpected DEV counts")

    provenance = {}
    chunks_by_id = {}
    if config in GROUNDING_INDEXES:
        chunker, retriever = GROUNDING_INDEXES[config]
        if retriever in ("hybrid", "reranked"):
            snapshot = collect_combined_provenance(
                chunker, questions_path, retriever=retriever,
            )
            indexes = tuple(snapshot["indexes"].values())
        else:
            snapshot = collect_provenance(
                chunker, questions_path, retriever=retriever,
            )
            indexes = (snapshot["index"],)

        provenance = {
            index["path"]: {
                "manifest_sha256": index["manifest_sha256"],
                "chunks_sha256": index["artifact_sha256"]["chunks.jsonl"],
            }
            for index in indexes
        }

        # Combined provenance verifies both indexes share the same mapping.
        index = indexes[0]
        mapping = (REPO_ROOT / index["path"] / "chunks.jsonl").read_bytes()
        digest = hashlib.sha256(mapping).hexdigest()
        if (
            digest != provenance[index["path"]]["chunks_sha256"]
            or digest != index["manifest"]["input"]["sha256"]
        ):
            raise ValueError(f"{config}: chunk mapping hash mismatch")

        chunks = [json.loads(line) for line in mapping.splitlines()]
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        if not chunks or len(chunks_by_id) != len(chunks):
            raise ValueError(f"{config}: empty mapping or duplicate chunk IDs")

    answer_path = REPO_ROOT / "eval" / "results" / "answers" / f"{config}.jsonl"
    with answer_path.open(encoding="utf-8") as stream:
        saved = [json.loads(line) for line in stream]

    if len(saved) != len(dev):
        raise ValueError(f"{config}: expected {len(dev)} saved DEV answers")

    inputs = []
    for question, row in zip(dev, saved, strict=True):
        label = f"{config} {question['id']}"
        fields = ("id", "question", "split", "answerable")
        if row["config"] != config or any(row[key] != question[key] for key in fields):
            raise ValueError(f"{label}: saved answer does not match the DEV record")

        if config not in ("closed_book", "bm25_heading"):
            if row["index_provenance"] != provenance:
                raise ValueError(f"{label}: saved index provenance mismatch")

        ids = row["retrieved_chunk_ids"]
        expected_count = 0 if config == "closed_book" else 5
        if (
            len(ids) != expected_count
            or len(set(ids)) != expected_count
            or any(chunk_id not in chunks_by_id for chunk_id in ids)
        ):
            raise ValueError(f"{label}: invalid saved retrieval IDs")

        passages = [chunks_by_id[chunk_id] for chunk_id in ids]
        inputs.append((question, row, passages))

    return inputs


def score_answer(question, row, passages, *, on_attempt=None):
    grounded = row["config"] != "closed_book"
    abstention = REFUSAL if grounded else "I don't know."
    abstained = row["answer"] == abstention

    correctness = None
    if question["answerable"]:
        correctness = 0
        if not abstained:
            correctness = judge.judge_correctness(
                question["question"],
                question["gold_answer"],
                row["answer"],
                on_attempt=on_attempt,
            )["correct"]

    faithfulness = None
    if grounded and not abstained:
        faithfulness = judge.judge_faithfulness(
            question["question"],
            passages,
            row["answer"],
            on_attempt=on_attempt,
        )["faithful"]

    return {
        "abstained": abstained,
        "correctness": correctness,
        "faithfulness": faithfulness,
    }


def main():
    parser = argparse.ArgumentParser(description="Judge one saved DEV configuration.")
    parser.add_argument("config", choices=("closed_book", *GROUNDING_INDEXES))
    args = parser.parse_args()

    output = (
        REPO_ROOT / "eval" / "results" / "judges" / f"{args.config}.jsonl"
    )
    if output.exists():
        raise SystemExit(f"Result already exists: {output}")

    inputs = load_judge_inputs(args.config)
    rows = []
    for position, (question, answer_row, passages) in enumerate(inputs, start=1):
        question_id = question["id"]
        attempts = []

        def record_attempt(result):
            attempts.append({
                "requested_model": result["model"],
                "model_version": result["model_version"],
                "cache_key": result["cache_key"],
                "cache_hit": result["cache_hit"],
                "preflight_tokens": result["preflight_tokens"],
                "token_usage": {
                    name: result[name]
                    for name in (
                        "tokens_in", "tokens_out", "thinking_tokens", "total_tokens",
                    )
                },
                "api_usage": result["api_usage"],
            })
            if not result["cache_hit"]:
                sleep(10)

        try:
            scores = score_answer(
                question, answer_row, passages, on_attempt=record_attempt,
            )
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"{args.config} {question_id}: {error}") from None

        rows.append({
            "id": question_id,
            "split": "dev",
            "config": args.config,
            **scores,
            "attempts": attempts,
        })
        fresh = sum(not attempt["cache_hit"] for attempt in attempts)
        print(
            f"{position}/{len(inputs)} {question_id} "
            f"({fresh} fresh, {len(attempts) - fresh} cached)",
            flush=True,
        )

    jsonl = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(jsonl)

    attempts = [attempt for row in rows for attempt in row["attempts"]]
    fresh = sum(not attempt["cache_hit"] for attempt in attempts)
    print(f"Saved {len(rows)} DEV scores to {output.relative_to(REPO_ROOT)}")
    print(f"Completed attempts this invocation: {fresh} fresh, {len(attempts) - fresh} cached")


if __name__ == "__main__":
    main()