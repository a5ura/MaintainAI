"""Local configuration tests use only synthetic .env files under pytest's temp root."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from maintainai import llm
from maintainai.rag import CLAIM_FIELDS, generate_grounded_analysis

PREFIX = "MAINTAINAI_LLM_"
SETTINGS = {
    "ENDPOINT": "https://provider.invalid/v1/chat/completions",
    "MODEL": "synthetic-local-model",
    "API_KEY": "synthetic-fixture-credential",
    "TIMEOUT_SECONDS": "30", "MAX_TOKENS": "1000",
    "TOKEN_PARAMETER": "max_tokens", "JSON_MODE": "1",
}


def write_settings(tmp_path, **overrides):
    values = {**SETTINGS, **overrides}
    target = tmp_path / ".env"
    target.write_text("\n".join(PREFIX + name + "=" + value for name, value in values.items()) + "\n",
                      encoding="utf-8")
    return target


def test_project_dotenv_loads_all_settings_without_exporting_environment(tmp_path):
    write_settings(tmp_path)
    before = {name: os.environ.get(name) for name in llm._LLM_ENV_NAMES}
    config = llm.LLMConfig.from_environment()
    assert config.endpoint == SETTINGS["ENDPOINT"]
    assert config.model == SETTINGS["MODEL"]
    assert config.api_key == SETTINGS["API_KEY"]
    assert config.timeout_seconds == 30
    assert config.max_tokens == 1000
    assert config.token_parameter == "max_tokens" and config.json_mode is True
    assert {name: os.environ.get(name) for name in llm._LLM_ENV_NAMES} == before
    assert config.api_key not in repr(config)


@pytest.mark.parametrize("name,value,attribute,expected", [
    ("ENDPOINT", "https://process.invalid/v1/chat/completions", "endpoint", "https://process.invalid/v1/chat/completions"),
    ("MODEL", "process-model", "model", "process-model"),
    ("API_KEY", "synthetic-process-credential", "api_key", "synthetic-process-credential"),
    ("TIMEOUT_SECONDS", "45", "timeout_seconds", 45),
    ("MAX_TOKENS", "512", "max_tokens", 512),
    ("TOKEN_PARAMETER", "max_completion_tokens", "token_parameter", "max_completion_tokens"),
    ("JSON_MODE", "0", "json_mode", False),
])
def test_every_process_setting_takes_precedence(tmp_path, monkeypatch, name, value, attribute, expected):
    write_settings(tmp_path)
    monkeypatch.setenv(PREFIX + name, value)
    assert getattr(llm.LLMConfig.from_environment(), attribute) == expected


def test_process_value_overrides_even_an_invalid_file_value(tmp_path, monkeypatch):
    write_settings(tmp_path, ENDPOINT="http://remote.invalid/wrong")
    monkeypatch.setenv(PREFIX + "ENDPOINT", SETTINGS["ENDPOINT"])
    assert llm.LLMConfig.from_environment().endpoint == SETTINGS["ENDPOINT"]


def test_explicit_empty_process_key_does_not_reactivate_file_credential(tmp_path, monkeypatch):
    write_settings(tmp_path)
    monkeypatch.setenv(PREFIX + "API_KEY", "")
    with pytest.raises(llm.ProviderError, match="requires"):
        llm.LLMConfig.from_environment()


def test_missing_dotenv_preserves_environment_only_configuration(monkeypatch):
    for name in ("ENDPOINT", "MODEL", "API_KEY"):
        monkeypatch.setenv(PREFIX + name, SETTINGS[name])
    config = llm.LLMConfig.from_environment()
    assert config.model == SETTINGS["MODEL"]
    assert config.timeout_seconds == 30 and config.max_tokens == 1000
    assert config.token_parameter == "max_tokens" and config.json_mode is True


def test_missing_file_and_missing_process_values_raise_sanitized_configuration_error():
    with pytest.raises(llm.ProviderError, match="requires") as caught:
        llm.LLMConfig.from_environment()
    assert caught.value.code == "configuration"


@pytest.mark.parametrize("key", ["", "   # local comment", "''"])
def test_blank_file_key_does_not_enable_cloud(tmp_path, key):
    write_settings(tmp_path, API_KEY=key)
    with pytest.raises(llm.ProviderError, match="requires"):
        llm.LLMConfig.from_environment()


@pytest.mark.parametrize("name,value", [
    ("ENDPOINT", "https://provider.invalid/v1/"),
    ("ENDPOINT", "https://provider.invalid/v1/chat/completions?key=synthetic-sensitive-marker"),
    ("MODEL", "x" * 201),
    ("TIMEOUT_SECONDS", "nan"), ("TIMEOUT_SECONDS", "121"), ("TIMEOUT_SECONDS", "nonnumeric"),
    ("MAX_TOKENS", "127"), ("MAX_TOKENS", "4097"), ("MAX_TOKENS", "1000.5"),
    ("TOKEN_PARAMETER", "other"), ("JSON_MODE", "maybe"),
])
def test_existing_validation_still_rejects_bad_file_values_without_secret_output(
    tmp_path, name, value, capsys, caplog,
):
    write_settings(tmp_path, **{name: value})
    with pytest.raises(llm.ProviderError) as caught:
        llm.LLMConfig.from_environment()
    assert caught.value.code == "configuration"
    output = str(caught.value) + repr(caught.value) + caplog.text + str(capsys.readouterr())
    assert SETTINGS["API_KEY"] not in output
    assert "synthetic-sensitive-marker" not in output


def test_bom_crlf_quotes_export_and_comments_are_literal(tmp_path):
    lines = ["# synthetic configuration", "",
             f'export {PREFIX}ENDPOINT="{SETTINGS["ENDPOINT"]}" # endpoint comment',
             f"{PREFIX}MODEL='{SETTINGS['MODEL']}'",
             PREFIX + r"API_KEY='synthetic#value=$UNEXPANDED\path=literal' # comment",
             PREFIX + "MAX_TOKENS=512\t# numeric comment",
             PREFIX + "JSON_MODE=0"]
    (tmp_path / ".env").write_bytes(("\ufeff" + "\r\n".join(lines)).encode("utf-8"))
    config = llm.LLMConfig.from_environment()
    assert config.api_key == r"synthetic#value=$UNEXPANDED\path=literal"
    assert config.model == SETTINGS["MODEL"]
    assert config.max_tokens == 512 and config.json_mode is False


def test_unquoted_hash_without_whitespace_is_not_a_comment(tmp_path):
    write_settings(tmp_path, API_KEY="synthetic#literal=value")
    assert llm.LLMConfig.from_environment().api_key == "synthetic#literal=value"


def test_project_root_is_used_regardless_of_cwd(tmp_path, monkeypatch):
    write_settings(tmp_path)
    other = tmp_path / "different-working-directory"
    other.mkdir()
    write_settings(other, API_KEY="synthetic-wrong-directory-value", MODEL="wrong-directory-model")
    monkeypatch.chdir(other)
    config = llm.LLMConfig.from_environment()
    assert config.api_key == SETTINGS["API_KEY"]
    assert config.model == SETTINGS["MODEL"]


def test_edits_are_read_again_without_sticky_process_exports(tmp_path):
    write_settings(tmp_path)
    assert llm.LLMConfig.from_environment().model == SETTINGS["MODEL"]
    write_settings(tmp_path, MODEL="edited-local-model")
    assert llm.LLMConfig.from_environment().model == "edited-local-model"
    assert PREFIX + "MODEL" not in os.environ


def test_unrelated_dotenv_variables_do_not_affect_process_environment(tmp_path, monkeypatch):
    path = write_settings(tmp_path)
    monkeypatch.delenv("SYNTHETIC_UNRELATED_SETTING", raising=False)
    with path.open("a", encoding="utf-8") as file:
        file.write("SYNTHETIC_UNRELATED_SETTING=ignored\n")
    llm.LLMConfig.from_environment()
    assert "SYNTHETIC_UNRELATED_SETTING" not in os.environ


@pytest.mark.parametrize("line", ["synthetic-sensitive-marker-without-equals", PREFIX + "API_KEY='synthetic-sensitive-marker",
                                 PREFIX + 'API_KEY="synthetic-sensitive-marker" trailing-content'])
def test_malformed_lines_never_echo_credentials(tmp_path, line, caplog, capsys):
    (tmp_path / ".env").write_text(line, encoding="utf-8")
    with pytest.raises(llm.ProviderError, match="line 1") as caught:
        llm.LLMConfig.from_environment()
    output = str(caught.value) + repr(caught.value) + caplog.text + str(capsys.readouterr())
    assert "synthetic-sensitive-marker" not in output


def test_invalid_utf8_has_a_sanitized_read_error(tmp_path, caplog, capsys):
    (tmp_path / ".env").write_bytes(b"\xffsynthetic-sensitive-marker")
    with pytest.raises(llm.ProviderError, match="UTF-8") as caught:
        llm.LLMConfig.from_environment()
    assert "synthetic-sensitive-marker" not in str(caught.value) + caplog.text + str(capsys.readouterr())


def test_unreadable_file_has_a_sanitized_read_error(tmp_path, monkeypatch, caplog, capsys):
    target = write_settings(tmp_path)
    original = Path.read_text
    def unreadable(path, *args, **kwargs):
        if path == target:
            raise PermissionError("synthetic-sensitive-marker")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(llm.ProviderError, match="permissions") as caught:
        llm.LLMConfig.from_environment()
    assert "synthetic-sensitive-marker" not in str(caught.value) + caplog.text + str(capsys.readouterr())


def test_configuration_loading_never_initializes_transport_or_network(tmp_path, monkeypatch):
    write_settings(tmp_path)
    constructor = Mock(side_effect=AssertionError("Provider initialization is not configuration loading"))
    session = Mock(side_effect=AssertionError("No network session during configuration loading"))
    monkeypatch.setattr(llm, "ChatCompletionsProvider", constructor)
    monkeypatch.setattr(llm.requests, "Session", session)
    llm.LLMConfig.from_environment()
    constructor.assert_not_called()
    session.assert_not_called()


def test_local_only_rag_never_reads_dotenv_even_when_it_exists(tmp_path, monkeypatch):
    write_settings(tmp_path)
    loader = Mock(side_effect=AssertionError("No local configuration read without authorization"))
    monkeypatch.setattr(llm, "_local_llm_settings", loader)
    fallback = {"local_evidence": True}
    assert generate_grounded_analysis([], [{"id": 1}], fallback) == (fallback, False, None)
    loader.assert_not_called()


def test_invalid_dotenv_preserves_existing_local_rag_fallback(tmp_path):
    write_settings(tmp_path, TIMEOUT_SECONDS="999")
    fallback = {"local_evidence": True}
    analysis, generated, notice = generate_grounded_analysis([], [{"id": 1}], fallback, use_cloud=True)
    assert analysis == fallback and not generated
    assert "retrieval-only" in notice
    assert SETTINGS["API_KEY"] not in notice


def test_authorized_rag_uses_file_configuration_and_existing_provider_api(tmp_path, monkeypatch):
    write_settings(tmp_path)
    output = {"issue_summary": "Synthetic mock interpretation.",
              **{field: [] for field in CLAIM_FIELDS}, "evidence_references": [],
              "uncertainty_or_missing_information": ["Synthetic test; current conditions are unverified."]}
    provider = Mock(spec=llm.LLMProvider)
    provider.generate.return_value = json.dumps(output)
    constructor = Mock(return_value=provider)
    monkeypatch.setattr(llm, "ChatCompletionsProvider", constructor)
    messages = [{"role": "user", "content": "Synthetic context"}]
    analysis, generated, notice = generate_grounded_analysis(messages, [{"id": 1}], {}, use_cloud=True)
    assert generated and analysis == output and notice is None
    config = constructor.call_args.args[0]
    assert config.api_key == SETTINGS["API_KEY"]
    constructor.assert_called_once()
    provider.generate.assert_called_once_with(messages)
