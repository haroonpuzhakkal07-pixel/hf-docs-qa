"""Measure dense heading recall@5 on the CI DEV fixture."""
import argparse
import hashlib
import json
from pathlib import Path

from src.config import CORPUS_COMMIT, EMBEDDING_MODEL, EMBEDDING_REVISION
from src.evaluate import eval_retrieval
FIXTURE = Path(__file__).with_name("ci_dev_questions.jsonl")

def measure_recall(questions_path: Path) -> dict:
    # Normalize checkout line endings before hashing the fixture.
    fixture = questions_path.read_text(encoding="utf-8")
    questions = [json.loads(line) for line in fixture.splitlines()]
    if not questions or any(
        row["split"] != "dev" or row["answerable"] is not True
        for row in questions
    ):
        raise ValueError("Expected a nonempty answerable DEV fixture")
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in the DEV fixture")

    from src.retrieve import dense_search

    report = eval_retrieval(
        lambda query, *, k: dense_search(query, "heading", k=k),
        questions,
        ks=(5,),
    )
    return {
        "config": {
            "fixture_sha256": hashlib.sha256(fixture.encode("utf-8")).hexdigest(),
            "split": "dev",
            "answerable_only": True,
            "retriever": "dense",
            "index": "heading",
            "retrieval_depth": 5,
            "metric": "fraction of questions with any gold-page hit in the first 5 chunks",
            "corpus_commit": CORPUS_COMMIT,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
        },
        "n_questions": report["n_questions"],
        "hits_at_5": report["hit_counts"]["5"],
        "recall_at_5": report["recall"]["5"],
    }


def check_recall(baseline: dict, current: dict) -> None:
    if baseline["config"] != current["config"]:
        raise ValueError("Baseline and smoke evaluation configurations differ")

    count = baseline["n_questions"]
    if type(count) is not int or count <= 0 or count != current["n_questions"]:
        raise ValueError("Baseline and smoke evaluation question counts differ or are invalid")

    baseline_hits = baseline["hits_at_5"]
    current_hits = current["hits_at_5"]
    for hits in (baseline_hits, current_hits):
        if type(hits) is not int or not 0 <= hits <= count:
            raise ValueError("Invalid recall@5 hit count")

    lost_hits = baseline_hits - current_hits
    drop_points = 100 * lost_hits / count
    comparison = (
        f"recall@5: baseline {baseline_hits}/{count}, "
        f"current {current_hits}/{count}, drop {drop_points:.2f} percentage points"
    )
    if 100 * lost_hits > 3 * count:
        raise SystemExit(f"Recall gate failed: {comparison} (limit 3 points)")
    print(f"Recall gate passed: {comparison}")
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=FIXTURE)
    parser.add_argument("--baseline", type=Path, help="Check recall against this baseline")
    args = parser.parse_args()

    current = measure_recall(args.questions)
    print(json.dumps(current, indent=2))
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        check_recall(baseline, current)


if __name__ == "__main__":
    main()