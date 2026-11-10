from fastapi.testclient import TestClient

from src import api, generate


def test_ask_returns_cited_source_with_stubbed_completion(monkeypatch):
    chunks = [
        {
            "chunk_id": "amber:0",
            "path": "amber-loader.md",
            "heading_path": "Amber Loader",
            "text": "Load a synthetic adapter.",
            "source_text": "Load a synthetic adapter.",
        },
        {
            "chunk_id": "quartz:0",
            "path": "quartz-batching.md",
            "heading_path": "Quartz Batching",
            "text": "Group examples by length.",
            "source_text": "Group examples\n  by length.",
        },
    ]
    answer_text = "Use `quartz_batching` to group examples by length [2]."
    usage = {
        "tokens_in": 100,
        "tokens_out": 10,
        "thinking_tokens": 0,
        "total_tokens": 110,
    }

    monkeypatch.setattr(api, "check_hybrid_indexes", lambda _: None)
    monkeypatch.setattr(api, "load_index", lambda _: (None, chunks))
    monkeypatch.setattr(api, "get_embedding_model", lambda: None)
    monkeypatch.setattr(api, "get_reranker", lambda: None)

    def fake_search(question, *, index_name, k):
        assert (question, index_name, k) == (
            "How do I group examples by length?",
            "heading",
            5,
        )
        return chunks

    monkeypatch.setattr(generate, "reranked_search", fake_search)
    monkeypatch.setattr(
        generate,
        "complete",
        lambda *args, **kwargs: {
            **usage,
            "text": answer_text,
            "latency_ms": 4.0,
            "original_latency_ms": 4.0,
            "model": "stub-model",
            "cache_key": "stub-key",
            "api_usage": dict.fromkeys(usage, 0),
            "preflight_tokens": 100,
            "cache_hit": False,
            "model_version": "stub-version",
        },
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/ask", json={"question": "How do I group examples by length?"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == answer_text
    assert body["sources"] == [
        {
            "passage": 2,
            "path": "quartz-batching.md",
            "heading": "Quartz Batching",
            "snippet": "Group examples by length.",
        }
    ]
    assert body["retrieval_ms"] >= 0
    assert body["llm_ms"] == 4.0
    assert body["token_usage"] == usage