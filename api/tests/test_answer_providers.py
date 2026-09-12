"""Answer providers.

No live call has ever been made against Anthropic from this project, so these
tests cover the half that is ours: the request we build, how we read the
response, and how every upstream failure maps onto our error types. A fake
client stands in for the SDK's transport - the SDK's own behaviour is not what
is under test here.
"""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.config import settings
from app.errors import UpstreamError
from app.providers.anthropic_answers import SYSTEM_PROMPT, AnthropicAnswerProvider
from app.providers.echo_answers import EchoAnswerProvider

PASSAGES = [
    {"citation": 1, "source_label": "Note: Aurora", "content": "Rollback: revert the flag."},
    {"citation": 2, "source_label": "Runbook (https://e.com)", "content": "Then page the SRE."},
]


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _response(blocks, *, stop_reason="end_turn", **extra):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        model="claude-opus-5",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        **extra,
    )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    return AnthropicAnswerProvider()


def _install(provider, handler):
    """Swap the SDK client for a fake whose messages.create runs `handler`."""
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return handler(**kwargs)

    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return captured


# ---------- the request we build ----------


async def test_sends_the_grounding_prompt_and_numbered_context(provider):
    captured = _install(provider, lambda **_: _response([_text_block("Revert the flag [1].")]))

    await provider.generate(question="How do I roll back?", passages=PASSAGES)

    assert captured["system"] == SYSTEM_PROMPT
    assert captured["model"] == settings.anthropic_model

    sent = captured["messages"][0]["content"]
    assert "[1] Note: Aurora" in sent
    assert "[2] Runbook (https://e.com)" in sent
    assert "How do I roll back?" in sent
    # The model must never be handed history it could answer from instead.
    assert len(captured["messages"]) == 1


async def test_passes_the_configured_effort_and_token_ceiling(provider, monkeypatch):
    monkeypatch.setattr(settings, "answer_effort", "medium")
    monkeypatch.setattr(settings, "answer_max_tokens", 1234)
    captured = _install(provider, lambda **_: _response([_text_block("ok")]))

    await provider.generate(question="q?", passages=PASSAGES)

    assert captured["output_config"] == {"effort": "medium"}
    assert captured["max_tokens"] == 1234


# ---------- how we read the response ----------


async def test_joins_text_blocks_and_ignores_non_text(provider):
    _install(
        provider,
        lambda **_: _response(
            [
                SimpleNamespace(type="thinking", thinking=""),
                _text_block("First part [1]."),
                _text_block("Second part [2]."),
            ]
        ),
    )

    answer = await provider.generate(question="q?", passages=PASSAGES)

    assert answer.text == "First part [1].\nSecond part [2]."
    assert answer.truncated is False
    assert answer.usage == {"model": "claude-opus-5", "input_tokens": 120, "output_tokens": 30}


async def test_hitting_the_output_cap_is_reported_not_hidden(provider):
    _install(provider, lambda **_: _response([_text_block("cut off mid-")], stop_reason="max_tokens"))

    answer = await provider.generate(question="q?", passages=PASSAGES)

    assert answer.truncated is True


async def test_a_refusal_surfaces_as_422_with_its_category(provider):
    """A policy decline arrives as HTTP 200 with empty content - checked before reading it."""
    _install(
        provider,
        lambda **_: _response(
            [], stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")
        ),
    )

    with pytest.raises(UpstreamError) as excinfo:
        await provider.generate(question="q?", passages=PASSAGES)

    assert excinfo.value.status == 422
    assert excinfo.value.code == "answer_refused"
    assert excinfo.value.details == {"category": "cyber"}


async def test_a_refusal_without_stop_details_does_not_crash(provider):
    _install(provider, lambda **_: _response([], stop_reason="refusal"))

    with pytest.raises(UpstreamError) as excinfo:
        await provider.generate(question="q?", passages=PASSAGES)
    assert excinfo.value.details == {"category": None}


# ---------- how upstream failures map ----------


def _raise(exception):
    def handler(**_):
        raise exception

    return handler


def _sdk_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("upstream said no", response=response, body=None)


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (_sdk_error(anthropic.RateLimitError, 429), 429, "answer_rate_limited"),
        (_sdk_error(anthropic.AuthenticationError, 401), 502, "answer_unauthorized"),
    ],
)
async def test_sdk_errors_map_to_our_error_types(provider, exception, expected_status, expected_code):
    _install(provider, _raise(exception))

    with pytest.raises(UpstreamError) as excinfo:
        await provider.generate(question="q?", passages=PASSAGES)

    assert excinfo.value.status == expected_status
    assert excinfo.value.code == expected_code


async def test_a_timeout_maps_to_504(provider):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    _install(provider, _raise(anthropic.APITimeoutError(request=request)))

    with pytest.raises(UpstreamError) as excinfo:
        await provider.generate(question="q?", passages=PASSAGES)

    assert excinfo.value.status == 504
    assert excinfo.value.code == "answer_timeout"


async def test_an_unclassified_api_error_still_becomes_an_upstream_error(provider):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    _install(provider, _raise(anthropic.APIConnectionError(request=request)))

    with pytest.raises(UpstreamError) as excinfo:
        await provider.generate(question="q?", passages=PASSAGES)
    assert excinfo.value.code == "answer_failed"


async def test_missing_credentials_fail_at_construction_not_at_query_time(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    with pytest.raises(RuntimeError) as excinfo:
        AnthropicAnswerProvider()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


# ---------- the keyless provider ----------


async def test_echo_returns_the_top_passage_with_an_explicit_notice():
    answer = await EchoAnswerProvider().generate(question="How do I roll back?", passages=PASSAGES)

    assert "Rollback: revert the flag." in answer.text
    assert "[1] Note: Aurora" in answer.text
    assert "ANTHROPIC_API_KEY" in answer.text, "must say why it is not a real answer"
    assert answer.usage["model"] == "none"
