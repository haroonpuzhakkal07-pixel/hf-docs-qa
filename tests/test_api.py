from fastapi.testclient import TestClient

from src import api, generate


def test_ask_returns_cited_source_with_stubbed_completion(monkeypatch):
    monkeypatch.delenv("DOCS_RAG_RETRIEVAL_ONLY", raising=False)
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

    search_calls = []

    def fake_search(question, *, index_name, k):
        search_calls.append(question)
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

    payload = {"question": "How do I group examples by length?"}
    monkeypatch.delenv("DOCS_RAG_DEMO_KEY", raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/ask", json=payload).status_code == 503

        monkeypatch.setenv("DOCS_RAG_DEMO_KEY", "synthetic-demo-key")
        assert client.post("/ask", json=payload).status_code == 401
        assert client.post(
            "/ask", json=payload, headers={"X-Demo-Key": "incorrect-key"}
        ).status_code == 401
        assert search_calls == []

        response = client.post(
            "/ask", json=payload, headers={"X-Demo-Key": "synthetic-demo-key"}
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


def test_retrieval_only_returns_sources_without_generation(monkeypatch):
    monkeypatch.setenv("DOCS_RAG_RETRIEVAL_ONLY", "1")
    monkeypatch.setenv("DOCS_RAG_DEMO_KEY", "synthetic-demo-key")

    def unexpected_answer(*args, **kwargs):
        raise AssertionError("Retrieval-only mode must bypass generation")

    def fake_search(question, *, index_name, k):
        assert (question, index_name, k) == ("How do I load an adapter?", "heading", 3)
        return [{
            "path": "amber-loader.md",
            "heading_path": "Amber Loader",
            "source_text": "Load a\n  synthetic adapter.",
        }]

    monkeypatch.setattr(api, "answer", unexpected_answer)
    monkeypatch.setattr(api, "reranked_search", fake_search)

    response = TestClient(api.app).post(
        "/ask",
        json={"question": "How do I load an adapter?"},
        headers={"X-Demo-Key": "synthetic-demo-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Retrieval-only demo.")
    assert body["sources"] == [{
        "passage": 1,
        "path": "amber-loader.md",
        "heading": "Amber Loader",
        "snippet": "Load a synthetic adapter.",
    }]
    assert body["retrieval_ms"] >= 0
    assert body["llm_ms"] == 0
    assert body["token_usage"] == dict.fromkeys(
        ("tokens_in", "tokens_out", "thinking_tokens", "total_tokens"), 0
    )
