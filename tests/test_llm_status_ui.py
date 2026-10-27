"""Local-only status tests; navigation uses stub pages and never opens SQLite."""

from dataclasses import asdict
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from maintainai import llm, ui

PREFIX = "MAINTAINAI_LLM_"
SECRET = "SYNTHETIC_PREFIX_sensitive_credential_SYNTHETIC_SUFFIX"
SETTINGS = {
    "ENDPOINT": "https://provider.invalid/v1/chat/completions",
    "MODEL": "synthetic-status-model",
    "API_KEY": SECRET,
}


def sidebar_test_app():
    from maintainai.ui import render

    # Sidebar status and navigation do not need a database. Stub page handlers
    # below isolate their unrelated domain operations from this UI verification.
    render(object())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(tmp_path / "unused.db"))
    handlers = {}
    for name in ("home_page", "dashboard", "equipment_page", "create_issue_page", "issues_page", "history_page",
                 "import_page", "similar_issues_page", "troubleshooting_page", "equipment_logs_page", "fault_analysis_page",
                 "guided_troubleshooting_page"):
        def page(store, page_name=name):
            st.text("Synthetic page: " + page_name)
        handlers[name] = Mock(side_effect=page)
        monkeypatch.setattr(ui, name, handlers[name])
    guards = {}
    for name in ("provider_from_environment", "ChatCompletionsProvider"):
        guard = Mock(side_effect=AssertionError("Status must not initialize a provider"))
        monkeypatch.setattr(llm, name, guard)
        guards[name] = guard
    guards["network"] = Mock(side_effect=AssertionError("Status must not create an HTTP session"))
    monkeypatch.setattr(llm.requests, "Session", guards["network"])
    return SimpleNamespace(run=lambda: AppTest.from_function(sidebar_test_app, default_timeout=15).run(),
                           guards=guards, handlers=handlers)


def set_process(monkeypatch, **overrides):
    for name, value in {**SETTINGS, **overrides}.items():
        monkeypatch.setenv(PREFIX + name, value)


def write_local(tmp_path, **overrides):
    (tmp_path / ".env").write_text("\n".join(PREFIX + name + "=" + value
        for name, value in {**SETTINGS, **overrides}.items()) + "\n", encoding="utf-8")


def visible_sidebar(app):
    return "\n".join(str(item.value) for elements in (
        app.sidebar.header, app.sidebar.subheader, app.sidebar.caption, app.sidebar.text,
        app.sidebar.code, app.sidebar.info, app.sidebar.warning, app.sidebar.success,
    ) for item in elements)


def assert_private(text):
    assert SECRET not in text
    assert "SYNTHETIC_PREFIX" not in text
    assert "SYNTHETIC_SUFFIX" not in text
    assert "key length" not in text.lower()


def assert_no_calls(harness):
    for guard in harness.guards.values():
        guard.assert_not_called()


def test_missing_settings_list_only_required_variable_names_on_startup(harness):
    app = harness.run()
    assert not app.exception
    assert [item.value for item in app.sidebar.subheader] == ["Cloud AI Configuration"]
    assert app.sidebar.info[0].value == "Not configured"
    assert app.sidebar.text[0].value == "Missing: MAINTAINAI_LLM_ENDPOINT, MAINTAINAI_LLM_MODEL, MAINTAINAI_LLM_API_KEY"
    text = visible_sidebar(app)
    assert "project-root .env" in text and "process environment" in text
    assert "explicit consent" in text
    assert_private(text)
    assert_no_calls(harness)


def test_partial_configuration_displays_only_missing_names_not_present_values(harness, monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", SETTINGS["ENDPOINT"])
    monkeypatch.setenv(PREFIX + "API_KEY", SECRET)
    app = harness.run()
    assert app.sidebar.info[0].value == "Not configured"
    assert app.sidebar.text[0].value == "Missing: MAINTAINAI_LLM_MODEL"
    assert SETTINGS["ENDPOINT"] not in visible_sidebar(app)
    assert_private(visible_sidebar(app))
    assert_no_calls(harness)


def test_configured_status_displays_model_and_only_local_validation(harness, monkeypatch):
    set_process(monkeypatch)
    app = harness.run()
    assert not app.exception
    assert app.sidebar.success[0].value == "Configured"
    assert app.sidebar.text[0].value == "Model: synthetic-status-model"
    text = visible_sidebar(app)
    assert "model availability are not verified" in text
    assert "Provider connectivity" in text and "no provider request" in text
    assert SETTINGS["ENDPOINT"] not in text
    assert_private(text)
    assert_no_calls(harness)


@pytest.mark.parametrize("name,value", [
    ("ENDPOINT", "https://provider.invalid/v1/"),
    ("ENDPOINT", "https://provider.invalid/v1/chat/completions?key=" + SECRET),
    ("ENDPOINT", "https://user:" + SECRET + "@provider.invalid/v1/chat/completions"),
    ("MODEL", "x" * 201), ("API_KEY", "synthetic\ncredential"),
    ("TIMEOUT_SECONDS", "nan"), ("MAX_TOKENS", "99999"),
    ("TOKEN_PARAMETER", "other"), ("JSON_MODE", "maybe"),
])
def test_invalid_configuration_has_safe_actionable_feedback(harness, monkeypatch, caplog, capsys, name, value):
    set_process(monkeypatch, **{name: value})
    app = harness.run()
    assert not app.exception
    assert app.sidebar.warning[0].value == "Invalid configuration"
    text = visible_sidebar(app)
    assert "README.md" in text and ".env" in text and "Process environment" in text
    assert "Model:" not in text
    assert SETTINGS["ENDPOINT"] not in text
    assert_private(text + caplog.text + str(capsys.readouterr()))
    assert_no_calls(harness)


def test_configured_model_cannot_echo_an_accidentally_pasted_key(harness, monkeypatch):
    set_process(monkeypatch, MODEL="prefix-" + SECRET + "-suffix")
    app = harness.run()
    assert app.sidebar.success[0].value == "Configured"
    assert "hidden: credential found in model field" in visible_sidebar(app)
    assert_private(visible_sidebar(app))
    assert_no_calls(harness)


def test_status_return_value_has_no_credential_details(monkeypatch):
    set_process(monkeypatch)
    status = llm.get_llm_config_status()
    data = asdict(status)
    assert set(data) == {"status", "model", "missing_variables", "message"}
    assert data["status"] == "Configured"
    assert_private(repr(status) + json.dumps(data))


def test_status_reuses_existing_configuration_validation(monkeypatch):
    set_process(monkeypatch)
    validator = Mock(wraps=llm.LLMConfig.from_environment)
    monkeypatch.setattr(llm.LLMConfig, "from_environment", validator)
    assert llm.get_llm_config_status().status == "Configured"
    validator.assert_called_once_with()


def test_missing_name_metadata_comes_from_the_existing_required_check(monkeypatch):
    monkeypatch.setenv(PREFIX + "MODEL", SETTINGS["MODEL"])
    with pytest.raises(llm.ProviderError) as caught:
        llm.LLMConfig.from_environment()
    assert caught.value.code == "configuration"
    assert caught.value.missing_variables == (PREFIX + "ENDPOINT", PREFIX + "API_KEY")
    assert llm.get_llm_config_status().missing_variables == caught.value.missing_variables


def test_even_future_configuration_exception_text_is_not_rendered(harness, monkeypatch):
    monkeypatch.setattr(llm.LLMConfig, "from_environment", Mock(side_effect=llm.ProviderError("configuration", SECRET)))
    app = harness.run()
    assert not app.exception
    assert app.sidebar.warning[0].value == "Invalid configuration"
    assert_private(visible_sidebar(app))


def test_project_dotenv_and_process_precedence_are_used_by_sidebar(harness, tmp_path, monkeypatch):
    write_local(tmp_path)
    monkeypatch.setenv(PREFIX + "MODEL", "synthetic-process-model")
    app = harness.run()
    assert app.sidebar.success[0].value == "Configured"
    assert app.sidebar.text[0].value == "Model: synthetic-process-model"
    monkeypatch.setenv(PREFIX + "API_KEY", "")
    app.run()
    assert not app.exception
    assert app.sidebar.info[0].value == "Not configured"
    assert app.sidebar.text[0].value == "Missing: MAINTAINAI_LLM_API_KEY"
    assert_private(visible_sidebar(app))
    assert_no_calls(harness)


def test_malformed_local_file_is_invalid_without_exposing_its_contents(harness, tmp_path, caplog, capsys):
    (tmp_path / ".env").write_text(PREFIX + "API_KEY='" + SECRET, encoding="utf-8")
    app = harness.run()
    assert not app.exception
    assert app.sidebar.warning[0].value == "Invalid configuration"
    assert_private(visible_sidebar(app) + caplog.text + str(capsys.readouterr()))
    assert_no_calls(harness)


@pytest.mark.parametrize("configured", [False, True])
def test_sidebar_status_remains_visible_and_navigation_still_dispatches(harness, monkeypatch, configured):
    if configured:
        set_process(monkeypatch)
    app = harness.run()
    labels = [getattr(item, "value", None) for item in app.sidebar]
    assert labels.index("Cloud AI Configuration") > labels.index("Run this loader again safely; existing demo records are preserved.")
    for page in ui.PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert app.header[0].value == page
        assert app.sidebar.subheader[0].value == "Cloud AI Configuration"
        assert any(item.value.startswith("Synthetic page:") for item in app.text)
        assert_private(visible_sidebar(app))
    assert all(handler.called for handler in harness.handlers.values())
    assert_no_calls(harness)


def test_status_is_refreshed_on_rerun_without_caching_configuration(harness, monkeypatch):
    app = harness.run()
    assert app.sidebar.info[0].value == "Not configured"
    set_process(monkeypatch)
    app.run()
    assert app.sidebar.success[0].value == "Configured"
    monkeypatch.setenv(PREFIX + "MAX_TOKENS", "99999")
    app.run()
    assert app.sidebar.warning[0].value == "Invalid configuration"
    assert_no_calls(harness)
