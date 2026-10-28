import json
import pytest
from unittest.mock import Mock

from src import evaluate as evaluation
from src.evaluate import eval_retrieval, save_retrieval_run, score_retrieval


@pytest.mark.parametrize("rank", [1, 3, 5, 10, 11, None])
def test_hit_boundaries_and_reciprocal_rank_cutoff(rank):
    paths = ["other.md"] * 11
    if rank is not None:
        paths[rank - 1] = "gold.md"

    result = score_retrieval(paths, {"gold.md"})
    expected_rank = rank if rank is not None and rank <= 10 else None

    assert result["first_hit_rank"] == expected_rank
    assert result["hits"] == {
        str(k): expected_rank is not None and expected_rank <= k
        for k in (1, 3, 5, 10)
    }
    assert result["reciprocal_rank"] == pytest.approx(
        1.0 / expected_rank if expected_rank is not None else 0.0
    )


def test_any_gold_page_matches_without_collapsing_chunk_ranks():
    result = score_retrieval(
        ["other.md", "other.md", "second_gold.md", "first_gold.md"],
        {"first_gold.md", "second_gold.md"},
    )

    assert result["first_hit_rank"] == 3
    assert result["hits"] == {
        "1": False, "3": True, "5": True, "10": True
    }
    assert result["reciprocal_rank"] == pytest.approx(1 / 3)


def test_empty_results_are_a_miss():
    result = score_retrieval([], {"gold.md"})

    assert not any(result["hits"].values())
    assert result["first_hit_rank"] is None
    assert result["reciprocal_rank"] == 0.0


@pytest.mark.parametrize("ks", [(), (0,), (-1,), (True,)])
def test_rejects_invalid_cutoffs(ks):
    with pytest.raises(ValueError, match="positive integers"):
        score_retrieval(["gold.md"], {"gold.md"}, ks)


def test_rejects_missing_gold_pages():
    with pytest.raises(ValueError, match="gold_paths"):
        score_retrieval(["other.md"], set())


def evaluation_question(number, *, split="dev", answerable=True):
    return {
        "id": f"q{number:03d}",
        "question": f"Question {number}",
        "type": "lookup" if answerable else "unanswerable",
        "split": split,
        "answerable": answerable,
        "gold_sources": [{"path": "gold.md"}] if answerable else [],
    }


def test_filters_before_retrieval_and_includes_misses_in_denominator():
    questions = [
        evaluation_question(1),
        evaluation_question(2),
        evaluation_question(3),
        evaluation_question(4, split="test"),
        evaluation_question(5, answerable=False),
    ]
    responses = {
        "Question 1": [{"path": "gold.md"}],
        "Question 2": [
            {"path": "other.md"},
            {"path": "other.md"},
            {"path": "gold.md"},
        ],
        "Question 3": [],
    }
    calls = []

    def retriever(query, *, k):
        calls.append((query, k))
        return responses[query]

    result = eval_retrieval(retriever, questions)

    assert calls == [
        ("Question 1", 10),
        ("Question 2", 10),
        ("Question 3", 10),
    ]
    assert result["n_questions"] == 3
    assert result["hit_counts"] == {"1": 1, "3": 2, "5": 2, "10": 2}
    assert result["recall"] == pytest.approx({
        "1": 1 / 3, "3": 2 / 3, "5": 2 / 3, "10": 2 / 3,
    })
    assert result["mrr@10"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert [row["id"] for row in result["per_question"]] == [
        "q001", "q002", "q003",
    ]
    assert result["per_question"][1]["retrieved"] == responses["Question 2"]


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [evaluation_question(4, split="test")],
        [evaluation_question(5, answerable=False)],
    ],
)
def test_empty_eligible_set_fails_without_calling_retriever(questions):
    def retriever(query, *, k):
        pytest.fail("Retriever must not be called")

    with pytest.raises(ValueError, match="No answerable dev questions"):
        eval_retrieval(retriever, questions)


def test_custom_cutoff_controls_search_depth_and_mrr_label():
    calls = []

    def retriever(query, *, k):
        calls.append(k)
        return [{"path": "other.md"}, {"path": "gold.md"}]

    result = eval_retrieval(
        retriever,
        [evaluation_question(1)],
        ks=(1, 3),
    )

    assert calls == [3]
    assert result["retrieval_depth"] == 3
    assert result["hit_counts"] == {"1": 0, "3": 1}
    assert result["mrr@3"] == pytest.approx(0.5)
    assert "mrr@10" not in result


def test_saved_run_preserves_report_configuration_and_provenance(tmp_path):
    chunk = {
        "chunk_id": "example:0",
        "path": "gold.md",
        "text": "Complete source text.\nSecond line.",
        "score": 0.75,
    }
    report = eval_retrieval(
        lambda query, *, k: [chunk],
        [evaluation_question(1)],
    )
    config = {"retriever": "dense", "index": "fixed"}
    provenance = {
        "dataset": {"sha256": "fixture-dataset-hash"},
        "corpus": {"commit": "fixture-corpus-commit"},
        "index": {"manifest_sha256": "fixture-manifest-hash"},
        "git": {"commit": "fixture-project-commit", "dirty": True},
    }
    output = tmp_path / "retrieval" / "dense_fixed_dev.json"

    save_retrieval_run(
        output, report, config=config, provenance=provenance,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["schema_version"] == 1
    assert saved["created_at_utc"].endswith("+00:00")
    assert saved["config"] == config
    assert saved["provenance"] == provenance
    assert saved["evaluation"] == report
    assert saved["evaluation"]["per_question"][0]["retrieved"] == [chunk]


def test_saved_run_refuses_to_overwrite_existing_file(tmp_path):
    output = tmp_path / "existing.json"
    original = b"Existing baseline result.\n"
    output.write_bytes(original)

    with pytest.raises(FileExistsError):
        save_retrieval_run(
            output, {}, config={}, provenance={},
        )

    assert output.read_bytes() == original


def test_invalid_json_numbers_fail_before_creating_output(tmp_path):
    output = tmp_path / "new-directory" / "result.json"

    with pytest.raises(ValueError):
        save_retrieval_run(
            output, {"score": float("nan")}, config={}, provenance={},
        )

    assert not output.parent.exists()


@pytest.fixture(params=["fixed", "heading"])
def dense_runner_case(tmp_path, monkeypatch, request):
    from eval import check_questions as question_validation
    from src import retrieve

    index_name = request.param
    output = (
        tmp_path / "eval" / "results" / "retrieval"
        / f"dense_{index_name}_dev.json"
    )
    questions = [
        evaluation_question(1),
        evaluation_question(4, split="test"),
        evaluation_question(5, answerable=False),
    ]
    calls = []
    loader = Mock()

    def fake_search(query, selected_index, *, k):
        calls.append((query, selected_index, k))
        return [{
            "chunk_id": "fixture:0",
            "path": "gold.md",
            "text": "Complete fixture passage.",
            "score": 0.75,
        }]

    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation, "collect_provenance",
        Mock(return_value={"dataset": {"sha256": "fixture-hash"}}),
    )
    monkeypatch.setattr(
        question_validation, "check_questions",
        Mock(return_value=questions),
    )
    monkeypatch.setattr(retrieve, "load_index", loader)
    monkeypatch.setattr(retrieve, "dense_search", fake_search)

    return index_name, output, calls, loader


def test_dense_runner_saves_only_answerable_dev_results(dense_runner_case):
    index_name, output, calls, loader = dense_runner_case

    report = evaluation.run_dense_evaluation(index_name)
    saved = json.loads(output.read_text(encoding="utf-8"))

    loader.cache_clear.assert_called_once_with()
    loader.assert_called_once_with(index_name)
    assert calls == [("Question 1", index_name, 10)]
    assert report["n_questions"] == 1
    assert report["mrr@10"] == 1.0
    assert saved["evaluation"] == report
    assert saved["config"]["index"] == index_name
    assert saved["config"]["split"] == "dev"
    assert saved["config"]["retrieval_depth"] == 10
    assert saved["provenance"]["dataset"]["sha256"] == "fixture-hash"
    assert [row["id"] for row in report["per_question"]] == ["q001"]


def test_dense_runner_rejects_provenance_changes(
    dense_runner_case, monkeypatch,
):
    index_name, output, _, _ = dense_runner_case
    monkeypatch.setattr(
        evaluation,
        "collect_provenance",
        Mock(side_effect=[
            {"dataset": {"sha256": "before"}},
            {"dataset": {"sha256": "after"}},
        ]),
    )

    with pytest.raises(ValueError, match="changed during the run"):
        evaluation.run_dense_evaluation(index_name)

    assert not output.exists()
    assert not output.parent.exists()


def test_dense_runner_refuses_existing_result_before_retrieval(
    dense_runner_case, monkeypatch,
):
    index_name, output, calls, loader = dense_runner_case
    output.parent.mkdir(parents=True)
    original = b"Preserved baseline result.\n"
    output.write_bytes(original)
    monkeypatch.setattr(
        evaluation,
        "collect_provenance",
        Mock(side_effect=AssertionError("Must reject existing output first")),
    )

    with pytest.raises(FileExistsError, match="Result already exists"):
        evaluation.run_dense_evaluation(index_name)

    loader.assert_not_called()
    assert calls == []
    assert output.read_bytes() == original