from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src import rerank


def fake_model(length_batches, scores):
    model = Mock()
    model.preprocess.side_effect = [
        {"attention_mask": torch.tensor([
            [1] * length + [0] * (max(lengths) - length)
            for length in lengths
        ])}
        for lengths in length_batches
    ]
    model.predict.return_value = scores
    return model


def test_complete_pairs_and_raw_scores_preserve_input_order(monkeypatch):
    query = "  Full query?  "
    passages = ["First complete passage.", "Second complete passage."]
    expected = np.array([-2.5, 4.0], dtype=np.float32)
    model = fake_model([[512, 9]], expected)
    monkeypatch.setattr(rerank, "get_reranker", lambda: model)

    actual = rerank.score_pairs(query, passages)

    np.testing.assert_array_equal(actual, expected)
    pairs = [(query, text) for text in passages]
    model.preprocess.assert_called_once_with(
        pairs,
        prompt="",
        processing_kwargs={"text": {"truncation": False}},
    )
    model.predict.assert_called_once_with(
        pairs,
        prompt="",
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        apply_softmax=False,
        processing_kwargs={"text": {"truncation": False}},
    )


def test_later_overflow_rejects_all_inference(monkeypatch):
    model = fake_model(
        [[20], [513]],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    monkeypatch.setattr(rerank, "get_reranker", lambda: model)

    with pytest.raises(ValueError, match=r"Pair row 1 has 513 tokens"):
        rerank.score_pairs("Query", ["First", "Second"], batch_size=1)

    assert model.preprocess.call_count == 2
    model.predict.assert_not_called()


@pytest.mark.parametrize(
    "query,passages,batch_size",
    [
        (" ", ["Passage"], 8),
        (None, ["Passage"], 8),
        ("Query", [], 8),
        ("Query", [""], 8),
        ("Query", [None], 8),
        ("Query", ["Passage"], 0),
        ("Query", ["Passage"], True),
    ],
)
def test_invalid_inputs_fail_before_loading(
    monkeypatch, query, passages, batch_size
):
    loader = Mock()
    monkeypatch.setattr(rerank, "get_reranker", loader)

    with pytest.raises(ValueError):
        rerank.score_pairs(query, passages, batch_size=batch_size)

    loader.assert_not_called()


@pytest.mark.parametrize(
    "scores",
    [
        np.array([[1.0]], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([1.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_invalid_model_scores_are_rejected(monkeypatch, scores):
    model = fake_model([[10]], scores)
    monkeypatch.setattr(rerank, "get_reranker", lambda: model)

    with pytest.raises(ValueError, match="score"):
        rerank.score_pairs("Query", ["Passage"])


from copy import deepcopy


@pytest.mark.parametrize("index_name", ["fixed", "heading"])
def test_reranked_search_preserves_records_and_hybrid_tie_order(
    monkeypatch, index_name
):
    candidates = [
        {
            "chunk_id": chunk_id,
            "path": f"{chunk_id}.md",
            "text": f"Complete passage {chunk_id}.",
            "heading_path": "Example > Details",
            "score": 0.03 - position * 0.001,
            "rrf_ranks": {"dense": position + 1},
        }
        for position, chunk_id in enumerate(["z", "a", "b"])
    ]
    original = deepcopy(candidates)
    search = Mock(return_value=candidates)
    scorer = Mock(return_value=np.array([-1, 4, 4], dtype=np.float32))
    monkeypatch.setattr(rerank, "hybrid_search", search)
    monkeypatch.setattr(rerank, "score_pairs", scorer)

    results = rerank.reranked_search(" Full query ", index_name, k=2)

    search.assert_called_once_with(" Full query ", index_name, k=20)
    scorer.assert_called_once_with(
        " Full query ", [row["text"] for row in candidates]
    )
    assert [row["chunk_id"] for row in results] == ["a", "b"]
    assert [row["hybrid_rank"] for row in results] == [2, 3]
    for result, position in zip(results, [1, 2]):
        assert result == {
            **original[position],
            "hybrid_score": original[position]["score"],
            "hybrid_rank": position + 1,
            "score": 4.0,
        }
    assert candidates == original


@pytest.mark.parametrize("requested_k", [5, 10])
def test_reranked_search_scores_all_twenty_before_selecting(
    monkeypatch, requested_k
):
    candidates = [
        {"chunk_id": str(i), "text": f"Full passage {i}", "score": 0.01}
        for i in range(20)
    ]
    monkeypatch.setattr(
        rerank, "hybrid_search", Mock(return_value=candidates)
    )
    scorer = Mock(return_value=np.arange(20, dtype=np.float32))
    monkeypatch.setattr(rerank, "score_pairs", scorer)

    if requested_k == 5:
        results = rerank.reranked_search("Query", "heading")
    else:
        results = rerank.reranked_search("Query", "heading", k=requested_k)

    assert len(results) == requested_k
    assert [row["chunk_id"] for row in results] == [
        str(i) for i in range(19, 19 - requested_k, -1)
    ]
    assert len(scorer.call_args.args[1]) == 20


def test_empty_candidates_skip_scoring(monkeypatch):
    monkeypatch.setattr(rerank, "hybrid_search", Mock(return_value=[]))
    scorer = Mock()
    monkeypatch.setattr(rerank, "score_pairs", scorer)

    assert rerank.reranked_search("Query", "fixed") == []
    scorer.assert_not_called()


@pytest.mark.parametrize(
    "query,index_name,k",
    [
        ("", "fixed", 5),
        (None, "fixed", 5),
        ("Query", "unknown", 5),
        ("Query", "fixed", 0),
        ("Query", "fixed", 21),
        ("Query", "fixed", True),
        ("Query", "fixed", 1.5),
    ],
)
def test_invalid_search_arguments_fail_before_retrieval(
    monkeypatch, query, index_name, k
):
    search = Mock()
    monkeypatch.setattr(rerank, "hybrid_search", search)

    with pytest.raises(ValueError):
        rerank.reranked_search(query, index_name, k=k)

    search.assert_not_called()