import json
from unittest.mock import Mock

import pytest
import requests

from maintainai.llm import MAX_RESPONSE_BYTES, ChatCompletionsProvider, LLMConfig, ProviderError


class FakeResponse:
    def __init__(self, body=None, status=200):
        self.status_code = status
        self.body = body if isinstance(body, bytes) else json.dumps(body if body is not None else {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"synthetic":"test output"}'}}],
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.body


class FakeSession:
    def __init__(self, response=None, error=None):
        self.post = Mock(return_value=response or FakeResponse(), side_effect=error)
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def config():
    # Deliberately nonfunctional fixture values, never real account credentials.
    return LLMConfig("https://provider.invalid/v1/chat/completions", "unit-test-model", "unit-test-secret")


def install_transport(monkeypatch, session):
    monkeypatch.setattr("maintainai.llm.requests.Session", lambda: session)


def test_request_is_single_bounded_and_has_no_tools(config, monkeypatch):
    session = FakeSession()
    install_transport(monkeypatch, session)
    messages = [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "Synthetic fault"}]
    assert ChatCompletionsProvider(config).generate(messages) == '{"synthetic":"test output"}'
    session.post.assert_called_once()
    kwargs = session.post.call_args.kwargs
    assert session.post.call_args.args == (config.endpoint,)
    assert kwargs["headers"]["Authorization"] == "Bearer " + config.api_key
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (5, 30)
    assert kwargs["stream"] is True
    assert kwargs["json"] == {"model": config.model, "messages": messages, "stream": False,
                               "max_tokens": 1000, "response_format": {"type": "json_object"}}
    assert "tools" not in kwargs["json"]
    assert session.trust_env is False
    assert config.api_key not in repr(config)


@pytest.mark.parametrize("status,code", [(429, "rate_limit"), (401, "authentication"), (403, "authentication"),
                                        (402, "quota"), (503, "unavailable"), (400, "request"), (302, "request")])
def test_http_errors_are_sanitized_and_not_retried(config, monkeypatch, status, code):
    session = FakeSession(FakeResponse({"error": "do not expose unit-test-secret or provider body"}, status))
    install_transport(monkeypatch, session)
    with pytest.raises(ProviderError) as caught:
        ChatCompletionsProvider(config).generate([])
    assert caught.value.code == code
    assert config.api_key not in str(caught.value)
    assert "provider body" not in str(caught.value)
    session.post.assert_called_once()


@pytest.mark.parametrize("error,code", [(requests.Timeout("Sensitive details"), "timeout"),
                                       (requests.ConnectionError("Sensitive details"), "unavailable")])
def test_transport_errors_are_sanitized(config, monkeypatch, error, code):
    session = FakeSession(error=error)
    install_transport(monkeypatch, session)
    with pytest.raises(ProviderError) as caught:
        ChatCompletionsProvider(config).generate([])
    assert caught.value.code == code
    assert "Sensitive details" not in str(caught.value)
    session.post.assert_called_once()


@pytest.mark.parametrize("body", [b"not JSON", {"choices": []}, {"choices": [{"message": "wrong type"}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"message": {"content": "{}", "tool_calls": [{"name": "operate_robot"}]}}]},
    {"choices": [{"message": {"content": "{}", "refusal": "Declined"}}]},
    b"x" * (MAX_RESPONSE_BYTES + 1)],
    ids=["non-json", "empty-choices", "wrong-message-type", "null-content", "truncated", "tool-call", "refusal", "oversized"])
def test_bad_provider_envelopes_fail_gracefully(config, monkeypatch, body):
    install_transport(monkeypatch, FakeSession(FakeResponse(body)))
    with pytest.raises(ProviderError) as caught:
        ChatCompletionsProvider(config).generate([])
    assert caught.value.code == "invalid_response"


def test_configuration_requires_all_credentials(monkeypatch):
    monkeypatch.setenv("MAINTAINAI_LLM_MODEL", "unit-test-model")
    with pytest.raises(ProviderError, match="requires"):
        LLMConfig.from_environment()


def test_configuration_and_compatibility_options_come_from_environment(monkeypatch):
    for name, value in {"ENDPOINT": "http://127.0.0.1:9999/v1/chat/completions", "MODEL": "chosen-test-model",
                        "API_KEY": "nonfunctional-test-value", "TIMEOUT_SECONDS": "15", "MAX_TOKENS": "500",
                        "JSON_MODE": "0", "TOKEN_PARAMETER": "max_completion_tokens"}.items():
        monkeypatch.setenv("MAINTAINAI_LLM_" + name, value)
    config = LLMConfig.from_environment()
    assert config.model == "chosen-test-model"
    assert config.timeout_seconds == 15
    assert config.max_tokens == 500
    assert config.json_mode is False
    session = FakeSession()
    install_transport(monkeypatch, session)
    ChatCompletionsProvider(config).generate([])
    payload = session.post.call_args.kwargs["json"]
    assert payload["max_completion_tokens"] == 500
    assert "max_tokens" not in payload and "response_format" not in payload


@pytest.mark.parametrize("endpoint", ["http://cloud.invalid/v1/chat/completions", "https://user:pass@cloud.invalid/v1/chat/completions",
                                      "https://cloud.invalid/v1/chat/completions?key=example", "file:///tmp/chat/completions",
                                      "https://cloud.invalid/v1/other"])
def test_unsafe_or_incompatible_endpoints_are_rejected(endpoint):
    with pytest.raises(ProviderError, match="configuration"):
        LLMConfig(endpoint, "unit-test-model", "unit-test-secret")


@pytest.mark.parametrize("name,value", [("TIMEOUT_SECONDS", "nan"), ("TIMEOUT_SECONDS", "999"),
                                        ("MAX_TOKENS", "nope"), ("MAX_TOKENS", "99999"),
                                        ("JSON_MODE", "maybe"), ("TOKEN_PARAMETER", "other")])
def test_invalid_environment_options_fail_cleanly(monkeypatch, name, value):
    for key, item in {"ENDPOINT": "https://provider.invalid/v1/chat/completions", "MODEL": "unit-test-model", "API_KEY": "unit-test-secret"}.items():
        monkeypatch.setenv("MAINTAINAI_LLM_" + key, item)
    monkeypatch.setenv("MAINTAINAI_LLM_" + name, value)
    with pytest.raises(ProviderError):
        LLMConfig.from_environment()
