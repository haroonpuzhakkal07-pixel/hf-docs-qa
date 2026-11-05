import pytest

from src import generate
from src.generate import _cited_chunk_ids
from src.prompting import REFUSAL, SYSTEM_PROMPT, build_prompt


@pytest.mark.parametrize(
    ("text", "passage_chunk_ids", "expected"),
    [
        ("See [2], then [1, 2].", ["chunk-a", "chunk-b"], ["chunk-b", "chunk-a"]),
        ("See [0, 1, 3] and [note 2].", ["chunk-a", "chunk-b"], ["chunk-a"]),
        (REFUSAL, ["chunk-a"], []),
        ("See [1, 2].", [], []),
    ],
)
def test_cited_chunk_ids(text, passage_chunk_ids, expected):
    assert _cited_chunk_ids(text, passage_chunk_ids) == expected

@pytest.mark.parametrize("config", ["bm25_heading", "closed_book"])
def test_answer_config_wires_retrieval_prompt_and_usage(monkeypatch, config):
    question = "How do I configure this?"
    chunks = [
        {"chunk_id": "a", "path": "a.md", "heading_path": "A", "text": "First passage."},
        {"chunk_id": "b", "path": "b.md", "heading_path": "B", "text": "Second passage."},
    ]
    retrieval_calls = []
    usage = {
        "tokens_in": 100,
        "tokens_out": 8,
        "thinking_tokens": 2,
        "total_tokens": 110,
    }
    api_usage = dict.fromkeys(usage, 0)

    def fake_search(query, *, index_name, k):
        retrieval_calls.append((query, index_name, k))
        return chunks

    def fake_complete(system, prompt, *, temperature, max_tokens, input_token_budget):
        if config == "bm25_heading":
            assert system == SYSTEM_PROMPT
            assert prompt == build_prompt(question, chunks)
        else:
            assert system == generate.CLOSED_BOOK_SYSTEM_PROMPT
            assert prompt == question
        assert (temperature, max_tokens, input_token_budget) == (0, 500, 3000)
        return {
            **usage,
            "text": "Use the configuration in [2, 1].",
            "latency_ms": 4.0,
            "api_usage": api_usage,
            "preflight_tokens": 101,
            "cache_hit": True,
            "model_version": "test-model",
        }

    monkeypatch.setattr(generate, "bm25_search", fake_search)
    monkeypatch.setattr(generate, "complete", fake_complete)

    result = generate.answer(question, config)

    if config == "bm25_heading":
        assert retrieval_calls == [(question, "heading", 5)]
        assert result["retrieved_chunk_ids"] == ["a", "b"]
        assert result["cited_chunk_ids"] == ["b", "a"]
        assert result["retrieval_ms"] >= 0.0
    else:
        assert retrieval_calls == []
        assert result["retrieved_chunk_ids"] == []
        assert result["cited_chunk_ids"] == []
        assert result["retrieval_ms"] == 0.0

    assert result["answer"] == "Use the configuration in [2, 1]."
    assert result["llm_ms"] == 4.0
    assert result["token_usage"] == usage
    assert result["api_usage"] == api_usage
    assert result["preflight_tokens"] == 101
    assert result["cache_hit"] is True
    assert result["model_version"] == "test-model"