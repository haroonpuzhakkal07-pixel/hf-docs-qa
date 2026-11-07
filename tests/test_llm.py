import hashlib
import json
from collections import deque

import httpx
import pytest

from src import judge, llm


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    calls = []
    responses = deque()
    real_client = httpx.Client

    def handler(request):
        calls.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": json.loads(request.content),
        })
        if not responses:
            pytest.fail("Unexpected HTTP request")
        status, body = responses.popleft()
        return httpx.Response(status, json=body)

    def client(**kwargs):
        return real_client(
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    monkeypatch.setattr(llm.httpx, "Client", client)
    monkeypatch.setattr(
        llm, "dotenv_values",
        lambda path: {"GEMINI_API_KEY": "synthetic-key"},
    )
    return calls, responses


def successful_response():
    return {
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [
                {"text": "Hidden reasoning", "thought": True},
                {"text": "First line.\n"},
                {"text": "Second line."},
            ]},
        }],
        "usageMetadata": {
            "promptTokenCount": 30,
            "candidatesTokenCount": 8,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 40,
        },
        "modelVersion": "synthetic-model-version",
    }


def test_complete_preserves_inputs_usage_and_timing(api, monkeypatch):
    calls, responses = api
    responses.extend([
        (200, {"totalTokens": 30}),
        (200, successful_response()),
    ])
    clock = iter([0.0, 0.1, 1.0, 1.25, 1.5, 2.0])
    monkeypatch.setattr(llm, "perf_counter", lambda: next(clock))

    system = "  Full system instruction.\nSecond line.  "
    prompt = "  Full prompt.\n```python\nprint('hello')\n```  "
    result = llm.complete(system, prompt)

    assert len(calls) == 2
    assert calls[0]["url"].endswith(":countTokens")
    assert calls[1]["url"].endswith(":generateContent")
    generation = calls[1]["body"]
    assert generation["systemInstruction"]["parts"] == [{"text": system}]
    assert generation["contents"] == [
        {"role": "user", "parts": [{"text": prompt}]}
    ]
    assert calls[0]["body"]["generateContentRequest"] == {
        "model": f"models/{llm.MODEL}",
        **generation,
    }
    assert generation["generationConfig"] == {
        "temperature": 0,
        "maxOutputTokens": 500,
        "candidateCount": 1,
        "thinkingConfig": {"thinkingLevel": "MINIMAL"},
    }
    assert calls[1]["headers"]["x-goog-api-key"] == "synthetic-key"
    assert "synthetic-key" not in calls[1]["url"]
    assert result["text"] == "First line.\nSecond line."
    assert result["tokens_in"] == 30
    assert result["tokens_out"] == 8
    assert result["thinking_tokens"] == 2
    assert result["total_tokens"] == 40
    assert result["preflight_tokens"] == 30
    assert result["model_version"] == "synthetic-model-version"
    assert result["generation_ms"] == pytest.approx(250)
    assert result["latency_ms"] == pytest.approx(2000)
    assert result["original_latency_ms"] == pytest.approx(1400)
    assert result["cache_hit"] is False


def test_overflow_stops_before_generation(api):
    calls, responses = api
    responses.append((200, {
        "totalTokens": llm.INPUT_TOKEN_LIMIT + 1,
    }))

    with pytest.raises(ValueError, match="Text was not truncated"):
        llm.complete("System", "Complete prompt")

    assert len(calls) == 1
    assert calls[0]["url"].endswith(":countTokens")


def test_exact_input_limit_is_allowed(api):
    calls, responses = api
    responses.extend([
        (200, {"totalTokens": llm.INPUT_TOKEN_LIMIT}),
        (200, successful_response()),
    ])

    result = llm.complete("System", "Prompt")

    assert len(calls) == 2
    assert result["preflight_tokens"] == llm.INPUT_TOKEN_LIMIT


@pytest.mark.parametrize("overrides", [
    {"system": ""},
    {"prompt": " \n"},
    {"temperature": True},
    {"temperature": float("nan")},
    {"temperature": -0.1},
    {"temperature": 2.1},
    {"max_tokens": True},
    {"max_tokens": 0},
    {"max_tokens": llm.OUTPUT_TOKEN_LIMIT + 1},
])
def test_invalid_arguments_make_no_requests(api, overrides):
    calls, _ = api
    arguments = {"system": "System", "prompt": "Prompt", **overrides}

    with pytest.raises(ValueError):
        llm.complete(**arguments)

    assert calls == []


@pytest.mark.parametrize("finish_reason", ["MAX_TOKENS", "SAFETY", None])
def test_partial_or_blocked_output_is_rejected(api, finish_reason):
    calls, responses = api
    data = successful_response()
    data["candidates"][0]["finishReason"] = finish_reason
    responses.extend([
        (200, {"totalTokens": 30}),
        (200, data),
    ])

    with pytest.raises(RuntimeError, match="partial output was rejected"):
        llm.complete("System", "Prompt")

    assert len(calls) == 2


def test_rate_limit_is_not_retried_or_exposed(api):
    calls, responses = api
    responses.extend([
        (200, {"totalTokens": 30}),
        (429, {"error": {"message": "sensitive server details"}}),
    ])

    with pytest.raises(RuntimeError, match="HTTP 429") as error:
        llm.complete("System", "Prompt")

    assert len(calls) == 2
    assert "sensitive server details" not in str(error.value)
    assert "synthetic-key" not in str(error.value)


def test_missing_usage_is_rejected(api):
    _, responses = api
    data = successful_response()
    del data["usageMetadata"]
    responses.extend([
        (200, {"totalTokens": 30}),
        (200, data),
    ])

    with pytest.raises(RuntimeError, match="no usage metadata"):
        llm.complete("System", "Prompt")

def queue_success(responses):
    responses.extend([
        (200, {"totalTokens": 30}),
        (200, successful_response()),
    ])


def test_day17_text_cache_is_reused(api, monkeypatch, tmp_path):
    calls, responses = api
    old_source = tmp_path / "day17_llm.py"
    old_source.write_bytes(b"synthetic Day 17 implementation")
    old_hash = hashlib.sha256(old_source.read_bytes()).hexdigest()
    with monkeypatch.context() as old:
        old.setattr(llm, "__file__", str(old_source))
        queue_success(responses)
        original = llm.complete("System", "Prompt")

    monkeypatch.setattr(llm, "DAY17_IMPLEMENTATION_SHA256", old_hash)
    path = tmp_path / "data" / "cache" / "llm" / f"{original['cache_key']}.json"
    original_bytes = path.read_bytes()
    hit = llm.complete("System", "Prompt")

    assert hit["cache_hit"] is True
    assert hit["cache_key"] == original["cache_key"]
    assert path.read_bytes() == original_bytes
    assert len(calls) == 2


def test_judge_json_retry_bypasses_cached_malformed_reply(api):
    calls, responses = api
    prompt = json.dumps({
        "question": "Question?", "gold_answer": "Gold", "model_answer": "Answer",
    })
    queue_success(responses)
    malformed = llm.complete(
        judge.CORRECTNESS_SYSTEM, prompt,
        model=judge.JUDGE_MODEL, json_output=True,
    )
    valid = successful_response()
    valid["candidates"][0]["content"]["parts"] = [
        {"text": '{"correct": 0, "reason": "Omits the gold fact."}'},
    ]
    responses.extend([(200, {"totalTokens": 30}), (200, valid)])

    assert judge.judge_correctness("Question?", "Gold", "Answer") == {
        "correct": 0, "reason": "Omits the gold fact.",
    }
    assert len(calls) == 4
    assert llm.complete(
        judge.CORRECTNESS_SYSTEM, prompt,
        model=judge.JUDGE_MODEL, json_output=True,
    )["cache_key"] == malformed["cache_key"]
    assert len(calls) == 4
    for preflight, generation in (calls[:2], calls[2:]):
        endpoint = f"/models/{judge.JUDGE_MODEL}"
        assert preflight["url"].endswith(f"{endpoint}:countTokens")
        assert generation["url"].endswith(f"{endpoint}:generateContent")
        assert preflight["body"]["generateContentRequest"] == {
            "model": f"models/{judge.JUDGE_MODEL}", **generation["body"],
        }
        assert generation["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_cache_hit_works_without_credentials_or_http(api, monkeypatch):
    calls, responses = api
    queue_success(responses)
    system = "  Complete system.\nSecond line.  "
    prompt = "  Complete prompt.\nThird line.  "

    first = llm.complete(system, prompt)
    files = list((llm.REPO_ROOT / "data/cache/llm").glob("*.json"))
    assert len(files) == 1
    original_bytes = files[0].read_bytes()
    saved = json.loads(original_bytes)
    assert saved["request"]["system"] == system
    assert saved["request"]["prompt"] == prompt
    assert saved["result"]["text"] == first["text"]
    assert b"synthetic-key" not in original_bytes

    def forbidden(*args, **kwargs):
        pytest.fail("Cache hit must not read credentials or create HTTP client")

    monkeypatch.setattr(llm, "dotenv_values", forbidden)
    monkeypatch.setattr(llm.httpx, "Client", forbidden)
    second = llm.complete(system, prompt)

    assert len(calls) == 2
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["cache_key"] == first["cache_key"]
    assert second["text"] == first["text"]
    assert second["model_version"] == first["model_version"]
    assert second["tokens_in"] == first["tokens_in"]
    assert second["tokens_out"] == first["tokens_out"]
    assert second["generation_ms"] == 0
    assert second["original_generation_ms"] == first["original_generation_ms"]
    assert second["original_latency_ms"] == first["original_latency_ms"]
    assert second["api_usage"] == dict.fromkeys(first["api_usage"], 0)
    assert first["api_usage"] == {
        "tokens_in": 30, "tokens_out": 8,
        "thinking_tokens": 2, "total_tokens": 40,
    }
    assert files[0].read_bytes() == original_bytes


@pytest.mark.parametrize("overrides", [
    {"system": "Different system"},
    {"prompt": "Different prompt"},
    {"prompt": "Prompt "},
    {"temperature": 0.5},
    {"max_tokens": 501},
])
def test_changed_request_preserves_old_entry_and_creates_new_one(api, overrides):
    _, responses = api
    queue_success(responses)
    first = llm.complete("System", "Prompt")
    folder = llm.REPO_ROOT / "data/cache/llm"
    old_path = folder / f"{first['cache_key']}.json"
    old_bytes = old_path.read_bytes()

    queue_success(responses)
    second = llm.complete(**{
        "system": "System", "prompt": "Prompt", **overrides,
    })

    assert second["cache_hit"] is False
    assert second["cache_key"] != first["cache_key"]
    assert old_path.read_bytes() == old_bytes
    assert len(list(folder.glob("*.json"))) == 2


@pytest.mark.parametrize("setting,value", [
    ("MODEL", "synthetic-other-model"),
    ("THINKING_LEVEL", "LOW"),
    ("API_ROOT", "https://synthetic.example/v1beta"),
])
def test_changed_configuration_does_not_reuse_old_response(
    api, monkeypatch, setting, value
):
    _, responses = api
    queue_success(responses)
    first = llm.complete("System", "Prompt")

    monkeypatch.setattr(llm, setting, value)
    queue_success(responses)
    second = llm.complete("System", "Prompt")

    assert second["cache_hit"] is False
    assert second["cache_key"] != first["cache_key"]


def test_changed_implementation_does_not_reuse_old_response(
    api, monkeypatch, tmp_path
):
    _, responses = api
    implementation = tmp_path / "synthetic_implementation.py"
    implementation.write_text("# version one\n", encoding="utf-8")
    monkeypatch.setattr(llm, "__file__", str(implementation))

    queue_success(responses)
    first = llm.complete("System", "Prompt")

    implementation.write_text("# version two\n", encoding="utf-8")
    queue_success(responses)
    second = llm.complete("System", "Prompt")

    assert second["cache_hit"] is False
    assert second["cache_key"] != first["cache_key"]


@pytest.mark.parametrize("damage", ["json", "request", "text", "tokens"])
def test_corrupt_entry_is_preserved_without_http_fallback(api, damage):
    calls, responses = api
    queue_success(responses)
    first = llm.complete("System", "Prompt")
    path = llm.REPO_ROOT / "data/cache/llm" / f"{first['cache_key']}.json"

    saved = json.loads(path.read_text(encoding="utf-8"))
    if damage == "json":
        corrupt = "{invalid JSON"
    else:
        if damage == "request":
            saved["request"]["prompt"] = "Changed stored prompt"
        elif damage == "text":
            saved["result"]["text"] = ""
        else:
            saved["result"]["tokens_in"] = -1
        corrupt = json.dumps(saved)
    path.write_text(corrupt, encoding="utf-8")
    corrupt_bytes = path.read_bytes()

    with pytest.raises(RuntimeError):
        llm.complete("System", "Prompt")

    assert len(calls) == 2
    assert path.read_bytes() == corrupt_bytes


def test_failed_generation_creates_no_cache(api):
    _, responses = api
    responses.extend([
        (200, {"totalTokens": 30}),
        (429, {"error": {"message": "Synthetic quota failure"}}),
    ])

    with pytest.raises(RuntimeError, match="HTTP 429"):
        llm.complete("System", "Prompt")

    assert not (llm.REPO_ROOT / "data/cache").exists()


def test_input_budget_stops_before_generation(api):
    calls, responses = api
    responses.append((200, {"totalTokens": 3001}))

    with pytest.raises(ValueError, match="3001 tokens; limit is 3000"):
        llm.complete("System", "Complete prompt", input_token_budget=3000)

    assert len(calls) == 1
    assert calls[0]["url"].endswith(":countTokens")
    assert not (llm.REPO_ROOT / "data/cache/llm").exists()


def test_cached_response_must_fit_requested_budget(api):
    calls, responses = api
    responses.extend([
        (200, {"totalTokens": 3000}),
        (200, successful_response()),
    ])
    first = llm.complete("System", "Complete prompt", input_token_budget=3000)
    assert first["preflight_tokens"] == 3000

    with pytest.raises(ValueError, match="3000 tokens; limit is 2999"):
        llm.complete("System", "Complete prompt", input_token_budget=2999)

    cached = llm.complete("System", "Complete prompt", input_token_budget=3001)
    assert cached["cache_hit"] is True
    assert cached["cache_key"] == first["cache_key"]
    assert len(calls) == 2
