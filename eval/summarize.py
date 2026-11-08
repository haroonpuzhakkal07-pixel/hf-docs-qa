"""Summarize saved DEV answers and judge scores."""

import csv
import json
from statistics import mean, quantiles

from eval.run_answers import GROUNDING_INDEXES
from eval.run_judges import load_judge_inputs
from src.config import REPO_ROOT


def metric_counts(values):
    applicable = [value for value in values if value is not None]
    return sum(applicable), len(applicable)


def summarize(config, question_type, rows):
    latencies = [row["latency_ms"] for row in rows]
    percentiles = quantiles(latencies, n=100, method="inclusive")
    summary = {
        "config": config,
        "question_type": question_type,
        "queries": len(rows),
    }
    for metric in (
        "recall_at_5", "correctness", "faithfulness",
        "refusal_accuracy", "over_refusal",
    ):
        numerator, denominator = metric_counts(row[metric] for row in rows)
        summary[f"{metric}_num"] = numerator if denominator else None
        summary[f"{metric}_den"] = denominator

    summary.update(
        p50_latency_ms=round(percentiles[49], 2),
        p95_latency_ms=round(percentiles[94], 2),
        mean_tokens=round(mean(row["tokens"] for row in rows), 2),
        cost_per_query_usd=0.0,
        saved_generation_cache_hits=sum(row["cache_hit"] for row in rows),
    )
    return summary


def build_summary():
    results = REPO_ROOT / "eval" / "results"
    summaries = []

    for config in ("closed_book", *GROUNDING_INDEXES):
        inputs = load_judge_inputs(config)
        with (results / "judges" / f"{config}.jsonl").open(encoding="utf-8") as stream:
            scores = [json.loads(line) for line in stream]

        rows = []
        for (question, answer, passages), score in zip(inputs, scores, strict=True):
            if (score["config"], score["id"], score["split"]) != (
                config, question["id"], "dev",
            ):
                raise ValueError(f"{config} {question['id']}: score row mismatch")

            answerable = question["answerable"]
            recall = None
            if answerable and config != "closed_book":
                gold_paths = {source["path"] for source in question["gold_sources"]}
                recall = any(passage["path"] in gold_paths for passage in passages[:5])

            rows.append({
                "type": question["type"],
                "recall_at_5": recall,
                "correctness": score["correctness"],
                "faithfulness": score["faithfulness"],
                "refusal_accuracy": None if answerable else score["abstained"],
                "over_refusal": score["abstained"] if answerable else None,
                "latency_ms": answer["retrieval_ms"] + answer["llm_ms"],
                "tokens": answer["token_usage"]["total_tokens"],
                "cache_hit": answer["cache_hit"],
            })

        summaries.append(summarize(config, "overall", rows))
        for question_type in ("lookup", "multi_step", "new_or_changed", "unanswerable"):
            subset = [row for row in rows if row["type"] == question_type]
            summaries.append(summarize(config, question_type, subset))

    return summaries


METRIC_COLUMNS = (
    ("recall_at_5", "Recall@5"),
    ("correctness", "Correctness"),
    ("faithfulness", "Faithfulness"),
    ("refusal_accuracy", "Refusal accuracy"),
    ("over_refusal", "Over-refusal"),
)


def markdown_table(rows):
    headers = [
        "Configuration", "Questions",
        *(label for _, label in METRIC_COLUMNS),
        "p50 ms", "p95 ms", "Mean tokens", "USD/query", "Cache hits",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * (len(headers) - 1))]) + " |",
    ]

    for row in rows:
        cells = [f"`{row['config']}`", str(row["queries"])]
        for metric, _ in METRIC_COLUMNS:
            denominator = row[f"{metric}_den"]
            cells.append(
                f"{row[f'{metric}_num']}/{denominator}" if denominator else "N/A"
            )
        cells.extend([
            f"{row['p50_latency_ms']:,.2f}",
            f"{row['p95_latency_ms']:,.2f}",
            f"{row['mean_tokens']:,.2f}",
            f"${row['cost_per_query_usd']:.2f}",
            f"{row['saved_generation_cache_hits']}/{row['queries']}",
        ])
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_markdown(rows):
    sections = [
        "# Day 20 DEV comparison",
        "Each configuration uses the same 45 DEV questions. Grounded retrieval uses k=5.",
        "## Overall",
        markdown_table([row for row in rows if row["question_type"] == "overall"]),
        "## By question type",
    ]
    for question_type, label in (
        ("lookup", "Lookup"),
        ("multi_step", "Multi-step"),
        ("new_or_changed", "New or changed"),
        ("unanswerable", "Unanswerable"),
    ):
        sections.extend([
            f"### {label}",
            markdown_table([row for row in rows if row["question_type"] == question_type]),
        ])

    sections.extend([
        "## Metric definitions",
        "- Recall@5 counts answerable grounded questions with at least one gold "
        "source page among the saved top five passages. It is inapplicable to closed-book.",
        "- Correctness includes all answerable questions. Exact abstentions score zero.",
        "- Faithfulness includes all grounded answers other than exact refusals. "
        "It is inapplicable to closed-book.",
        "- Refusal accuracy uses unanswerable questions. Over-refusal uses answerable "
        "questions. Both use the configuration's exact abstention string.",
        "- N/A means the metric has no eligible rows. The CSV uses an empty numerator "
        "and a zero denominator for these cells.",
        "## Timing, tokens, and cost",
        "- Latency is saved retrieval time plus saved LLM call time, using inclusive "
        "percentiles. Cache hits include lookup time. Saved rows span different runs "
        "and quota windows, so these figures describe recorded runs.",
        "- Mean tokens use original generation usage, including input, output, and "
        "thinking tokens. Cached answers retain the original completion's token counts.",
        "- Generation cost per query is $0 on the billing-disabled Free tier. Judge "
        "requests are evaluation overhead. Fresh completion attempts also make a "
        "separate token-count request.",
        "- Cache-hit counts describe the invocation that saved each file. Requests "
        "completed before a restart can appear as cache hits in the final file. "
        "Saved API-usage fields cover successful completions in the saving invocation.",
    ])
    return "\n\n".join(sections) + "\n"


def main():
    results = REPO_ROOT / "eval" / "results"
    csv_path = results / "summary.csv"
    markdown_path = results / "summary.md"

    for path in (csv_path, markdown_path):
        if path.exists():
            raise SystemExit(f"Result already exists: {path}")

    rows = build_summary()
    markdown = render_markdown(rows)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with markdown_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)

    print(f"Saved {len(rows)} rows to {csv_path.relative_to(REPO_ROOT)}")
    print(f"Saved overall and per-type tables to {markdown_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
