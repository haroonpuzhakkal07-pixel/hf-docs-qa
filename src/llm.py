"""Gemini text completion with complete-input checks."""

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import dotenv_values

from src.config import REPO_ROOT


MODEL = "gemini-3.5-flash-lite"
INPUT_TOKEN_LIMIT = 1_048_576
OUTPUT_TOKEN_LIMIT = 65_536
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
THINKING_LEVEL = "MINIMAL"


def _post(client, method, payload, *, model):
    """Return a JSON object without exposing credentials in errors."""
    try:
        response = client.post(
            f"{API_ROOT}/models/{model}:{method}",
            json=payload,
        )
    except httpx.RequestError:
        raise RuntimeError(
            f"Gemini {method}: connection failed or timed out; "
            "no automatic retry was attempted."
        ) from None

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini {method}: HTTP {response.status_code}; "
            "no automatic retry was attempted."
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Gemini returned invalid JSON.") from None

    if not isinstance(data, dict):
        raise RuntimeError("Gemini returned an unexpected JSON structure.")
    return data


def _token_count(data, name):
    value = data.get(name)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Missing or invalid token count: {name}")
    return value


def _validate_inputs(system, prompt, temperature, max_tokens):
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must be a nonempty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be a finite number from 0 to 2")
    if type(max_tokens) is not int or not 1 <= max_tokens <= OUTPUT_TOKEN_LIMIT:
        raise ValueError(
            f"max_tokens must be an integer from 1 to {OUTPUT_TOKEN_LIMIT}"
        )


def _check_input_limit(tokens, limit):
    if tokens > limit:
        raise ValueError(
            f"Complete input has {tokens} tokens; "
            f"limit is {limit}. Text was not truncated."
        )


def _complete_uncached(
    system, prompt, temperature, max_tokens, *, input_limit, model, json_output
):
    """Return a complete text response, usage and elapsed milliseconds.

    Reads the key from the runtime environment, with a repository .env fallback.
    Does not truncate inputs, retry requests or cache responses.
    """
    _validate_inputs(system, prompt, temperature, max_tokens)

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = dotenv_values(REPO_ROOT / ".env").get("GEMINI_API_KEY")
    if not key or key == "replace_with_your_api_key":
        raise RuntimeError(
            "Set GEMINI_API_KEY in the runtime environment or the repository .env file."
        )

    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
            "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
        },
    }

    if json_output:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    started = perf_counter()
    with httpx.Client(
        headers={"x-goog-api-key": key},
        timeout=60.0,
        follow_redirects=False,
    ) as client:
        counted = _post(
            client,
            "countTokens",
            {
                "generateContentRequest": {
                    "model": f"models/{model}",
                    **payload,
                }
            },
            model=model,
        )
        input_tokens = _token_count(counted, "totalTokens")
        _check_input_limit(input_tokens, input_limit)

        generation_started = perf_counter()
        data = _post(client, "generateContent", payload, model=model)
        generation_ms = (perf_counter() - generation_started) * 1000

    candidates = data.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("Gemini did not return exactly one candidate.")

    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError(
            "Gemini did not finish normally; partial output was rejected."
        )

    parts = candidate.get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        raise RuntimeError("Gemini returned invalid response parts.")

    texts = []
    for part in parts:
        if not isinstance(part, dict):
            raise RuntimeError("Gemini returned an invalid response part.")
        if part.get("thought") is True:
            continue
        if not isinstance(part.get("text"), str):
            raise RuntimeError("Gemini returned unexpected non-text output.")
        texts.append(part["text"])

    text = "".join(texts)
    if not text.strip():
        raise RuntimeError("Gemini returned no answer text.")

    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        raise RuntimeError("Gemini returned no usage metadata.")

    model_version = data.get("modelVersion")
    if not isinstance(model_version, str) or not model_version:
        raise RuntimeError("Gemini returned no model version.")

    return {
        "text": text,
        "model": model,
        "model_version": model_version,
        "tokens_in": _token_count(usage, "promptTokenCount"),
        "tokens_out": _token_count(usage, "candidatesTokenCount"),
        "thinking_tokens": _token_count(
            {"count": usage.get("thoughtsTokenCount", 0)}, "count"
        ),
        "total_tokens": _token_count(usage, "totalTokenCount"),
        "preflight_tokens": input_tokens,
        "generation_ms": generation_ms,
        "latency_ms": (perf_counter() - started) * 1000,
        "finish_reason": candidate["finishReason"],
    }

CACHE_SCHEMA_VERSION = 1
DAY17_IMPLEMENTATION_SHA256 = (
    "ee68aa92c73c1080ab023c05c237c5cca0569e6660e610fcd643fe1dda91b171"
)


def _validate_cached_result(result, *, model):
    """Reject incomplete or malformed cache entries."""
    if not isinstance(result, dict):
        raise RuntimeError("Invalid cached completion.")
    if result.get("model") != model or result.get("finish_reason") != "STOP":
        raise RuntimeError("Incompatible cached completion.")
    for name in ("text", "model_version"):
        value = result.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Invalid cached completion field: {name}")
    for name in (
        "tokens_in", "tokens_out", "thinking_tokens",
        "total_tokens", "preflight_tokens",
    ):
        _token_count(result, name)
    for name in ("generation_ms", "latency_ms"):
        value = result.get(name)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(f"Invalid cached timing: {name}")


def _cache_path(request):
    encoded = json.dumps(
        request, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    return REPO_ROOT / "data" / "cache" / "llm" / f"{key}.json"


def complete(
    system, prompt, temperature=0, max_tokens=500, *,
    input_token_budget=None, model=None, json_output=False,
):
    """Return a completion, reusing a successful saved response when present.

    Top-level token counts describe the original response.
    api_usage describes usage for this call and is zero on cache hits.
    latency_ms measures this call, including cache access.
    Original response timings are retained separately.

    input_token_budget caps the complete input, including on cache hits.
    model defaults to the generator model. json_output requests JSON mode.
    """
    model = MODEL if model is None else model
    _validate_inputs(system, prompt, temperature, max_tokens)
    input_limit = INPUT_TOKEN_LIMIT
    if input_token_budget is not None:
        if (
            type(input_token_budget) is not int
            or not 1 <= input_token_budget <= INPUT_TOKEN_LIMIT
        ):
            raise ValueError(
                f"input_token_budget must be an integer from 1 to {INPUT_TOKEN_LIMIT}"
            )
        input_limit = input_token_budget
    started = perf_counter()

    request = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "api_root": API_ROOT,
        "model": model,
        "system": system,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "thinking_level": THINKING_LEVEL,
        "candidate_count": 1,
        "input_token_limit": INPUT_TOKEN_LIMIT,
        "output_token_limit": OUTPUT_TOKEN_LIMIT,
    }
    if json_output:
        request["response_mime_type"] = "application/json"

    cache_path = _cache_path(request)
    candidates = [request]
    if model == "gemini-3.5-flash-lite" and not json_output:
        # Day 17 used the same payload and response checks for text requests.
        candidates.append({**request, "implementation_sha256": DAY17_IMPLEMENTATION_SHA256})

    for candidate in candidates:
        path = _cache_path(candidate)
        try:
            saved_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        cache_path, request = path, candidate
        cache_hit = True
        break
    else:
        cache_hit = False

    if cache_hit:
        try:
            saved = json.loads(saved_text)
        except ValueError:
            raise RuntimeError(
                f"Invalid cache JSON: {cache_path.name}; entry preserved."
            ) from None
        if (
            not isinstance(saved, dict)
            or saved.get("schema_version") != CACHE_SCHEMA_VERSION
            or saved.get("request") != request
        ):
            raise RuntimeError(
                f"Incompatible cache entry: {cache_path.name}; entry preserved."
            )
        result = saved.get("result")
        _validate_cached_result(result, model=model)
        _check_input_limit(result["preflight_tokens"], input_limit)

    if not cache_hit:
        result = _complete_uncached(
            system, prompt, temperature, max_tokens,
            input_limit=input_limit, model=model, json_output=json_output,
        )
        _validate_cached_result(result, model=model)
        saved = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "request": request,
            "result": result,
        }
        serialized = json.dumps(
            saved, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)

    usage_names = ("tokens_in", "tokens_out", "thinking_tokens", "total_tokens")
    return {
        **result,
        "cache_hit": cache_hit,
        "cache_key": cache_path.stem,
        "original_latency_ms": result["latency_ms"],
        "original_generation_ms": result["generation_ms"],
        "generation_ms": 0.0 if cache_hit else result["generation_ms"],
        "latency_ms": (perf_counter() - started) * 1000,
        "api_usage": {
            name: 0 if cache_hit else result[name]
            for name in usage_names
        },
    }
