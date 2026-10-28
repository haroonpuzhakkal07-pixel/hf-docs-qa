"""Page-level retrieval evaluation."""

import argparse
import random
import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import platform
import subprocess
from importlib.metadata import version

from src.config import CORPUS_COMMIT, RAW_DIR, REPO_ROOT

DEFAULT_KS = (1, 3, 5, 10)


def score_retrieval(
    ranked_paths: list[str],
    gold_paths: set[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Score one question at chunk ranks, matching any gold source page.

    Only ranks through max(ks) contribute. Repeated pages retain their
    original chunk ranks. A page match does not establish answer evidence.
    """
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if not gold_paths:
        raise ValueError("gold_paths must not be empty")

    cutoff = max(ks)
    first_hit_rank = next(
        (
            rank
            for rank, path in enumerate(ranked_paths[:cutoff], start=1)
            if path in gold_paths
        ),
        None,
    )

    return {
        "hits": {
            str(k): first_hit_rank is not None and first_hit_rank <= k
            for k in ks
        },
        "first_hit_rank": first_hit_rank,
        "reciprocal_rank": (
            1.0 / first_hit_rank if first_hit_rank is not None else 0.0
        ),
    }


def eval_retrieval(retriever, questions, ks=DEFAULT_KS) -> dict:
    """Evaluate answerable dev questions using retriever(query, k=...).

    Questions must already have passed dataset validation.
    Recall is the share of selected questions with any gold-page hit.
    MRR is cut off at max(ks). Both denominators include misses.
    """
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")

    selected = [
        question
        for question in questions
        if question["split"] == "dev" and question["answerable"] is True
    ]
    if not selected:
        raise ValueError("No answerable dev questions to evaluate")

    cutoff = max(ks)
    per_question = []

    for question in selected:
        gold_paths = {source["path"] for source in question["gold_sources"]}
        if not gold_paths:
            raise ValueError(f"{question['id']}: missing gold source pages")

        retrieved = list(retriever(question["question"], k=cutoff))
        if len(retrieved) > cutoff:
            raise ValueError("Retriever returned more results than requested")

        metrics = score_retrieval(
            [chunk["path"] for chunk in retrieved],
            gold_paths,
            ks,
        )
        per_question.append({
            "id": question["id"],
            "question": question["question"],
            "type": question["type"],
            "gold_paths": sorted(gold_paths),
            "retrieved": retrieved,
            **metrics,
        })

    count = len(per_question)
    hit_counts = {
        str(k): sum(row["hits"][str(k)] for row in per_question)
        for k in ks
    }

    return {
        "split": "dev",
        "answerable_only": True,
        "ks": list(ks),
        "retrieval_depth": cutoff,
        "n_questions": count,
        "hit_counts": hit_counts,
        "recall": {
            key: hits / count
            for key, hits in hit_counts.items()
        },
        f"mrr@{cutoff}": sum(
            row["reciprocal_rank"] for row in per_question
        ) / count,
        "per_question": per_question,
    }


def save_retrieval_run(
    output: Path,
    report: dict,
    *,
    config: dict,
    provenance: dict,
) -> None:
    """Save a new run, preserving any existing result file."""
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "provenance": provenance,
        "metric_definitions": {
            "hit": "Retrieved chunk path matches any gold source page.",
            "recall@k": (
                "Questions with a hit within k chunk ranks / n_questions."
            ),
            "mrr": (
                "Mean reciprocal first-hit chunk rank through retrieval_depth; "
                "misses contribute zero; denominator is n_questions."
            ),
            "limitation": (
                "Page hits do not establish answering evidence or coverage "
                "of every source needed by a multi-step question."
            ),
        },
        "evaluation": report,
    }

    # Serialize before creating anything. Reject non-finite JSON numbers.
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)


def collect_provenance(index_name: str, questions_path: Path) -> dict:
    """Snapshot evaluation inputs without modifying them."""
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def git_output(folder: Path, *arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=folder, text=True,
        ).strip()

    corpus_commit = git_output(RAW_DIR, "rev-parse", "HEAD")
    corpus_status = git_output(
        RAW_DIR, "status", "--porcelain", "--untracked-files=all",
    )
    if corpus_commit != CORPUS_COMMIT:
        raise ValueError("Downloaded corpus does not match the project pin")
    if corpus_status:
        raise ValueError("Downloaded corpus has local changes")

    folder = REPO_ROOT / "data" / "index" / index_name
    manifest_bytes = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    if manifest["chunker"] != index_name:
        raise ValueError("Index manifest has the wrong chunker")
    if manifest["corpus_commit"] != corpus_commit:
        raise ValueError("Index corpus does not match the downloaded corpus")

    artifact_hashes = {
        name: file_hash(folder / name)
        for name in ("index.faiss", "chunks.jsonl")
    }
    for name, digest in artifact_hashes.items():
        if digest != manifest["artifacts"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")

    project_status = git_output(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=all",
    )
    implementation_files = (
        "src/evaluate.py",
        "src/retrieve.py",
        "src/embeddings.py",
        "src/config.py",
        "src/ingest.py",
        "src/markdown_structure.py",
        "eval/check_questions.py",
        "requirements.txt",
    )

    return {
        "dataset": {
            "path": questions_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_hash(questions_path),
        },
        "corpus": {
            "pinned_commit": CORPUS_COMMIT,
            "actual_commit": corpus_commit,
            "dirty": bool(corpus_status),
        },
        "index": {
            "path": folder.relative_to(REPO_ROOT).as_posix(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "artifact_sha256": artifact_hashes,
            "manifest": manifest,
        },
        "git": {
            "commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
            "dirty": bool(project_status),
            "status_porcelain": project_status,
        },
        "implementation_sha256": {
            name: file_hash(REPO_ROOT / name)
            for name in implementation_files
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "sentence-transformers", "faiss-cpu", "torch",
                    "transformers", "tokenizers", "huggingface-hub", "numpy",
                )
            },
        },
    }


def run_dense_evaluation(index_name: str) -> dict:
    """Evaluate one existing dense index and save a protected dev run."""
    if index_name not in ("fixed", "heading"):
        raise ValueError("index_name must be fixed or heading")

    output = (
        REPO_ROOT / "eval" / "results" / "retrieval"
        / f"dense_{index_name}_dev.json"
    )
    if output.exists():
        raise FileExistsError(f"Result already exists: {output}")

    questions_path = REPO_ROOT / "eval" / "questions.jsonl"
    provenance = collect_provenance(index_name, questions_path)

    from eval.check_questions import check_questions

    # Label validation may inspect both splits; retrieval filters to dev.
    questions = check_questions(questions_path)

    import faiss
    import numpy as np
    import torch

    from src.retrieve import dense_search, load_index

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Ensure this run loads the current verified artifacts.
    load_index.cache_clear()
    load_index(index_name)

    config = {
        "name": f"dense_{index_name}_dev",
        "retriever": "dense",
        "index": index_name,
        "split": "dev",
        "answerable_only": True,
        "ks": list(DEFAULT_KS),
        "retrieval_depth": max(DEFAULT_KS),
        "query_batch_size": 1,
        "seed": 0,
        "torch_threads": torch.get_num_threads(),
        "faiss_threads": faiss.omp_get_max_threads(),
    }

    print(
        f"Evaluating dense+{index_name} on answerable dev questions...",
        flush=True,
    )
    report = eval_retrieval(
        lambda query, *, k: dense_search(query, index_name, k=k),
        questions,
    )

    if collect_provenance(index_name, questions_path) != provenance:
        raise ValueError("Evaluation inputs or Git state changed during the run")

    save_retrieval_run(
        output, report, config=config, provenance=provenance,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", choices=("fixed", "heading"), required=True,
    )
    args = parser.parse_args()

    report = run_dense_evaluation(args.index)
    count = report["n_questions"]

    print(f"Evaluated {count} answerable dev questions.")
    for k in report["ks"]:
        key = str(k)
        print(
            f"recall@{k}: {report['hit_counts'][key]}/{count} "
            f"({report['recall'][key]:.4f})"
        )
    print(f"MRR@10: {report['mrr@10']:.4f}")
    print(f"Saved eval/results/retrieval/dense_{args.index}_dev.json")


if __name__ == "__main__":
    main()