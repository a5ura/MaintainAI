"""Optional chat-completions transport. No SDK, default model, or automatic retry."""

from dataclasses import dataclass, field
import json
import math
import os
import re
from typing import Protocol
from urllib.parse import urlsplit

import requests

from maintainai.database import PROJECT_ROOT

MAX_RESPONSE_BYTES = 128 * 1024
_LLM_ENV_NAMES = frozenset("MAINTAINAI_LLM_" + name for name in (
    "ENDPOINT", "MODEL", "API_KEY", "TIMEOUT_SECONDS", "MAX_TOKENS", "TOKEN_PARAMETER", "JSON_MODE",
))


class ProviderError(RuntimeError):
    """Sanitized provider failure; never include request headers or response bodies."""

    def __init__(self, code: str, message: str, *, missing_variables: tuple[str, ...] = ()):
        self.code = code
        self.missing_variables = missing_variables
        super().__init__(message)


class LLMProvider(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


def _local_llm_settings() -> dict[str, str]:
    """Read literal, single-line project-root .env settings without changing os.environ.

Support comments, optional export prefixes, and matching single/double quotes.
Backslashes and dollar signs stay literal; no interpolation, escape processing,
multiline values, shell execution, or logging of file contents occurs.
"""
    try:
        content = (PROJECT_ROOT / ".env").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise ProviderError("configuration", "Could not read the project-root .env file. Check file permissions and UTF-8 encoding.") from None
    settings = {}
    for number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name, value = name.strip(), raw_value.strip()
        if not separator:
            raise ProviderError("configuration", f"Invalid project-root .env syntax at line {number}. Use one NAME=VALUE per line.")
        if name not in _LLM_ENV_NAMES:
            continue
        if value.startswith(("'", '"')):
            closing = value.find(value[0], 1)
            if closing < 0 or (value[closing + 1:].strip() and not value[closing + 1:].strip().startswith("#")):
                raise ProviderError("configuration", f"Invalid project-root .env quoting at line {number}. Use matching quotes around a single-line value.")
            value = value[1:closing]
        else:
            # A comment requires preceding whitespace; embedded # remains literal.
            value = re.split(r"\s+#", raw_value, maxsplit=1)[0].strip()
        settings[name] = value
    return settings


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 30
    max_tokens: int = 1000
    json_mode: bool = True
    token_parameter: str = "max_tokens"

    def __post_init__(self):
        try:
            url = urlsplit(self.endpoint)
            allowed = url.scheme == "https" or (
                url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1"))
            if (not allowed or not url.hostname or url.username or url.password or url.query or url.fragment
                    or not url.path.endswith("/chat/completions")):
                raise ValueError("Invalid endpoint")
            if not self.model.strip() or len(self.model) > 200:
                raise ValueError("Invalid model")
            if not self.api_key.strip() or any(char in self.api_key for char in "\r\n"):
                raise ValueError("Invalid credentials")
            if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
                raise ValueError("Invalid timeout")
            if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 4096:
                raise ValueError("Invalid output limit")
            if self.token_parameter not in ("max_tokens", "max_completion_tokens"):
                raise ValueError("Invalid token parameter")
        except (ValueError, TypeError, AttributeError):
            raise ProviderError("configuration", "Invalid LLM configuration. Check endpoint, model, API key, timeout, and token settings. Use HTTPS, or HTTP on loopback only; no URL credentials or query parameters.") from None

    @classmethod
    def from_environment(cls):
        # Process values, including explicit empty values, override the optional
        # local file. Keep file settings out of the global process environment.
        environment = _local_llm_settings()
        environment.update({name: os.environ[name] for name in _LLM_ENV_NAMES if name in os.environ})
        values = {name: environment.get(f"MAINTAINAI_LLM_{name}", "").strip() for name in ("ENDPOINT", "MODEL", "API_KEY")}
        missing = tuple(f"MAINTAINAI_LLM_{name}" for name, value in values.items() if not value)
        if missing:
            raise ProviderError("configuration", "Cloud mode requires MAINTAINAI_LLM_ENDPOINT, MAINTAINAI_LLM_MODEL, and MAINTAINAI_LLM_API_KEY. Retrieval-only mode needs no LLM credentials.",
                                missing_variables=missing)
        try:
            timeout = float(environment.get("MAINTAINAI_LLM_TIMEOUT_SECONDS", "30"))
            tokens = int(environment.get("MAINTAINAI_LLM_MAX_TOKENS", "1000"))
            json_mode = environment.get("MAINTAINAI_LLM_JSON_MODE", "1")
            if json_mode not in ("0", "1"):
                raise ValueError("Invalid JSON mode")
        except ValueError:
            raise ProviderError("configuration", "Invalid LLM timeout, output-token limit, or JSON-mode setting.") from None
        return cls(values["ENDPOINT"], values["MODEL"], values["API_KEY"], timeout, tokens, json_mode == "1",
                   environment.get("MAINTAINAI_LLM_TOKEN_PARAMETER", "max_tokens"))


@dataclass(frozen=True)
class LLMConfigStatus:
    """Safe display data only; never retain a config object or credential details."""

    status: str
    model: str | None = None
    missing_variables: tuple[str, ...] = ()
    message: str = ""


def get_llm_config_status() -> LLMConfigStatus:
    """Validate merged local settings without initializing or contacting a provider."""
    try:
        config = LLMConfig.from_environment()
    except ProviderError as exc:
        if exc.missing_variables:
            return LLMConfigStatus("Not configured", missing_variables=exc.missing_variables,
                                   message="Set the missing variables in the project-root .env file or process environment. Process environment values take precedence.")
        # Status rendering must never surface exception text, raw settings, or
        # credentials, including text from a future configuration error source.
        return LLMConfigStatus("Invalid configuration", message=(
            "Check project-root .env syntax, UTF-8 encoding and file permissions, then review the "
            "endpoint, model, credentials, timeout and token settings against README.md. "
            "Process environment values take precedence over .env."
        ))
    # A credential accidentally pasted into the model field must not be displayed.
    model = config.model if config.api_key not in config.model else "(hidden: credential found in model field)"
    return LLMConfigStatus("Configured", model=model,
                           message="Provider connectivity, credential validity and model availability are not verified.")


class ChatCompletionsProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, messages: list[dict[str, str]]) -> str:
        config = self.config
        payload = {"model": config.model, "messages": messages, "stream": False,
                   config.token_parameter: config.max_tokens}
        if config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            with requests.Session() as session:
                # Avoid inheriting netrc credentials or silently routing prompts through environment proxies.
                session.trust_env = False
                with session.post(config.endpoint, json=payload,
                                  headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                                  timeout=(min(5, config.timeout_seconds), config.timeout_seconds),
                                  allow_redirects=False, stream=True) as response:
                    status = response.status_code
                    if status == 429:
                        raise ProviderError("rate_limit", "Provider rate limit reached. Wait and retry manually, or use retrieval-only mode.")
                    if status in (401, 403):
                        raise ProviderError("authentication", "Provider authentication failed. Check the configured API key and endpoint permissions.")
                    if status == 402:
                        raise ProviderError("quota", "Provider billing or quota is unavailable. Use retrieval-only mode or review your provider account.")
                    if status >= 500:
                        raise ProviderError("unavailable", "The inference provider is unavailable. Retry later or use retrieval-only mode.")
                    if status != 200:
                        raise ProviderError("request", "Provider rejected the request or redirected it. Check the final endpoint, model, token parameter, and JSON-mode support.")
                    chunks, size = [], 0
                    for chunk in response.iter_content(chunk_size=4096):
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ProviderError("invalid_response", "Provider response exceeded the local size limit.")
                        chunks.append(chunk)
                    body = json.loads(b"".join(chunks))
                    choice = body["choices"][0]
                    message = choice["message"]
                    if (choice.get("finish_reason") not in (None, "stop") or message.get("refusal")
                            or message.get("tool_calls") or message.get("function_call")):
                        raise ProviderError("invalid_response", "Provider returned an incomplete, refused, or unsupported response.")
                    content = message["content"]
                    if not isinstance(content, str) or not content.strip() or len(content) > 32000:
                        raise ValueError("Invalid completion content")
                    return content
        except requests.Timeout:
            raise ProviderError("timeout", "Provider request timed out. Retry manually or use retrieval-only mode.") from None
        except requests.RequestException:
            raise ProviderError("unavailable", "Could not reach the configured inference provider. Check service availability and TLS configuration.") from None
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, UnicodeError, RecursionError):
            raise ProviderError("invalid_response", "Provider returned an invalid chat-completions response.") from None


def provider_from_environment() -> LLMProvider:
    return ChatCompletionsProvider(LLMConfig.from_environment())


class DeterministicMockProvider:
    """Test double only; no HTTP, model, credentials, or inferred diagnosis."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        context = json.loads(messages[-1]["content"])
        issue_id = context["untrusted_records"][0]["issue_id"]
        return json.dumps({
            "issue_summary": "Synthetic mock interpretation of the reported problem.",
            "possible_causes": [{"text": "A similar historical failure pattern may be relevant.", "issue_ids": [issue_id]}],
            "recommended_diagnostic_checks": [{"text": "Have qualified personnel review existing fault logs and the cited record.", "issue_ids": [issue_id]}],
            "suggested_next_steps": [{"text": "Have qualified personnel confirm applicability using approved procedures.", "issue_ids": [issue_id]}],
            "evidence_references": [issue_id],
            "uncertainty_or_missing_information": ["Deterministic mock only; current equipment conditions have not been verified."],
        })
