import pytest
import json

from src import judge
from src.judge import parse_correctness, parse_faithfulness



@pytest.mark.parametrize(
    ("parse", "text", "expected"),
    [
        (parse_correctness, '{"correct": 1, "reason": "States the gold fact."}',
         {"correct": 1, "reason": "States the gold fact."}),
        (parse_faithfulness, '{"faithful": 1, "unsupported_claims": []}',
         {"faithful": 1, "unsupported_claims": []}),
        (parse_faithfulness, '{"faithful": 0, "unsupported_claims": ["Uses FP8."]}',
         {"faithful": 0, "unsupported_claims": ["Uses FP8."]}),
    ],
)
def test_parse_valid_verdict(parse, text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"correct": 1, "reason": "Matches."}\n```',
        '[]',
        '{"correct": 1}',
        '{"correct": true, "reason": "Matches."}',
        '{"correct": 1.0, "reason": "Matches."}',
        '{"correct": 2, "reason": "Matches."}',
        '{"correct": 0, "reason": " "}',
    ],
)
def test_reject_invalid_correctness(text):
    with pytest.raises(ValueError):
        parse_correctness(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"faithful": 0, "unsupported_claims": "Uses FP8."}',
        '{"faithful": 0, "unsupported_claims": [" "]}',
        '{"faithful": 0, "unsupported_claims": []}',
        '{"faithful": 1, "unsupported_claims": ["Uses FP8."]}',
    ],
)
def test_reject_invalid_faithfulness(text):
    with pytest.raises(ValueError):
        parse_faithfulness(text)


def test_parse_retry_uses_a_distinct_request(monkeypatch):
    calls = []
    cache = {}
    malformed = '{"correct": true, "reason": "Matches."}'
    valid = '{"correct": 1, "reason": "Matches."}'

    def fake_complete(system, prompt, **options):
        key = (system, prompt)
        calls.append(key)
        assert options == {
            "temperature": 0,
            "max_tokens": 500,
            "input_token_budget": 4000,
            "model": "gemini-3.5-flash",
            "json_output": True,
        }
        if key not in cache:
            cache[key] = valid if system != judge.CORRECTNESS_SYSTEM else malformed
        return {"text": cache[key]}

    monkeypatch.setattr(judge.llm, "complete", fake_complete)

    expected = {"correct": 1, "reason": "Matches."}
    assert judge.judge_correctness("Question?", "Gold", "Answer") == expected
    assert judge.judge_correctness("Question?", "Gold", "Answer") == expected
    first, retry, cached_first, cached_retry = calls
    assert first[0] != retry[0]
    assert first[1] == retry[1]
    assert (first, retry) == (cached_first, cached_retry)
    assert cache[first] == malformed


def test_second_invalid_reply_raises_without_another_retry(monkeypatch):
    calls = []

    def fake_complete(system, prompt, **options):
        calls.append(system)
        return {"text": "invalid JSON"}

    monkeypatch.setattr(judge.llm, "complete", fake_complete)

    with pytest.raises(ValueError):
        judge.judge_correctness("Question?", "Gold", "Answer")

    assert len(calls) == 2


def test_faithfulness_preserves_complete_ranked_passages(monkeypatch):
    chunks = [
        {"path": "b.md", "heading_path": "B", "text": "First\n\n```py\nx = 1\n```\n"},
        {"path": "a.md", "heading_path": "A", "text": "Second paragraph.\n"},
    ]
    calls = []

    def fake_complete(system, prompt, **options):
        calls.append(json.loads(prompt))
        return {"text": '{"faithful": 1, "unsupported_claims": []}'}

    monkeypatch.setattr(judge.llm, "complete", fake_complete)

    result = judge.judge_faithfulness("Question?", chunks, "Answer [1, 2]")
    assert result == {"faithful": 1, "unsupported_claims": []}
    assert calls == [
        {
            "question_and_passages": (
                "Question:\nQuestion?\n\nDocumentation passages:\n"
                "[1]\nPath: b.md\nHeading: B\nText:\nFirst\n\n```py\nx = 1\n```\n"
                "\n\n[2]\nPath: a.md\nHeading: A\nText:\nSecond paragraph.\n"
            ),
            "model_answer": "Answer [1, 2]",
        }
    ]
