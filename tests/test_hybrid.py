from copy import deepcopy

import pytest

from src.hybrid import reciprocal_rank_fusion

from unittest.mock import Mock

from src import hybrid

def fusion_chunk(chunk_id, score=0.0):
    return {
        "chunk_id": chunk_id,
        "path": f"{chunk_id}.md",
        "heading_path": "Fixture",
        "text": f"Complete passage for {chunk_id}.",
        "score": score,
    }


def test_fusion_uses_ranks_preserves_records_and_breaks_ties_by_id():
    rankings = {
        "dense": [fusion_chunk("a", 999), fusion_chunk("b", -10)],
        "bm25": [fusion_chunk("c", 1_000_000), fusion_chunk("b", -200)],
    }
    original = deepcopy(rankings)

    results = reciprocal_rank_fusion(rankings, k=10)

    assert [row["chunk_id"] for row in results] == ["b", "a", "c"]
    assert results[0]["score"] == pytest.approx(2 / 62)
    assert results[1]["score"] == pytest.approx(1 / 61)
    assert results[2]["score"] == pytest.approx(1 / 61)
    assert results[0]["rrf_ranks"] == {"dense": 2, "bm25": 2}
    assert {
        key: value for key, value in results[0].items()
        if key not in ("score", "rrf_ranks")
    } == {
        key: value for key, value in rankings["dense"][1].items()
        if key != "score"
    }
    assert reciprocal_rank_fusion(rankings, k=1) == results[:1]
    assert rankings == original


@pytest.mark.parametrize("rankings", [{}, {"dense": [], "bm25": []}])
def test_empty_rankings_return_no_results(rankings):
    assert reciprocal_rank_fusion(rankings) == []


def test_duplicate_chunk_within_one_method_is_rejected():
    rankings = {"dense": [fusion_chunk("a"), fusion_chunk("a")]}

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        reciprocal_rank_fusion(rankings)


def test_same_chunk_id_with_different_content_is_rejected():
    changed = fusion_chunk("a")
    changed["text"] = "Different source content."

    with pytest.raises(ValueError, match="Conflicting chunk records"):
        reciprocal_rank_fusion({
            "dense": [fusion_chunk("a")],
            "bm25": [changed],
        })


@pytest.mark.parametrize(
    "k, constant",
    [(0, 60), (-1, 60), (True, 60), (5, 0), (5, -1), (5, True)],
)
def test_invalid_parameters_are_rejected(k, constant):
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank_fusion({}, k=k, rrf_constant=constant)


@pytest.mark.parametrize("index_name", ["fixed", "heading"])
def test_hybrid_requests_50_from_each_method_and_preserves_query(
    monkeypatch, index_name,
):
    from src import bm25, retrieve

    query = "  A complete question?\nSecond line.  "
    calls = []
    check = Mock()

    def fake_dense(text, selected_index, *, k):
        calls.append(("dense", text, selected_index, k))
        return [fusion_chunk("a", 100), fusion_chunk("b", 50)]

    def fake_bm25(text, selected_index, *, k):
        calls.append(("bm25", text, selected_index, k))
        return [fusion_chunk("b", 2), fusion_chunk("c", 1)]

    monkeypatch.setattr(hybrid, "check_hybrid_indexes", check)
    monkeypatch.setattr(retrieve, "dense_search", fake_dense)
    monkeypatch.setattr(bm25, "bm25_search", fake_bm25)

    results = hybrid.hybrid_search(query, index_name, k=2)

    check.assert_called_once_with(index_name)
    assert calls == [
        ("dense", query, index_name, 50),
        ("bm25", query, index_name, 50),
    ]
    assert [row["chunk_id"] for row in results] == ["b", "a"]
    assert results[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert results[0]["rrf_ranks"] == {"dense": 2, "bm25": 1}


@pytest.mark.parametrize("index_name", ["fixed", "heading"])
@pytest.mark.parametrize("matches", [False, True])
def test_hybrid_mapping_check_rejects_changes_and_caches_success(
    monkeypatch, index_name, matches,
):
    from src import bm25, retrieve

    dense_records = [
        {key: value for key, value in fusion_chunk("a").items()
         if key != "score"}
    ]
    bm25_records = deepcopy(dense_records)
    if not matches:
        bm25_records[0]["text"] = "Changed source passage."

    dense_loader = Mock(return_value=(None, dense_records))
    bm25_loader = Mock(return_value=(None, bm25_records))
    monkeypatch.setattr(retrieve, "load_index", dense_loader)
    monkeypatch.setattr(bm25, "load_bm25_index", bm25_loader)

    hybrid.check_hybrid_indexes.cache_clear()
    try:
        if matches:
            hybrid.check_hybrid_indexes(index_name)
            hybrid.check_hybrid_indexes(index_name)
        else:
            with pytest.raises(ValueError, match="chunk mappings differ"):
                hybrid.check_hybrid_indexes(index_name)

        dense_loader.assert_called_once_with(index_name)
        bm25_loader.assert_called_once_with(index_name)
    finally:
        hybrid.check_hybrid_indexes.cache_clear()


@pytest.mark.parametrize(
    "query, index_name, k",
    [
        (None, "fixed", 5),
        ("", "fixed", 5),
        ("   ", "fixed", 5),
        ("query", "invalid", 5),
        ("query", "fixed", 0),
        ("query", "fixed", -1),
        ("query", "fixed", True),
        ("query", "fixed", 1.5),
    ],
)
def test_hybrid_rejects_invalid_arguments_before_loading(
    monkeypatch, query, index_name, k,
):
    def unexpected_check(name):
        pytest.fail("Invalid arguments must fail before index loading")

    monkeypatch.setattr(hybrid, "check_hybrid_indexes", unexpected_check)

    with pytest.raises(ValueError):
        hybrid.hybrid_search(query, index_name, k)