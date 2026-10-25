from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.llm import DeterministicMockProvider
from maintainai.seed import seed_demo
from maintainai.semantic_search import SemanticSearch
from maintainai.service import MaintenanceStore


def find(elements, label):
    return next(element for element in elements if element.label == label)


def setup_app(tmp_path, monkeypatch, fake_embedder, *, populated=True):
    path = tmp_path / "rag_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    with connect(path) as connection:
        initialize(connection)
        if populated:
            seed_demo(connection)
            SemanticSearch(MaintenanceStore(connection), fake_embedder).rebuild()
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.radio, "Navigate").set_value("AI Troubleshooting Assistant").run()
    return app, path


def test_warning_is_visible_even_without_history(tmp_path, monkeypatch, fake_embedder):
    app, _ = setup_app(tmp_path, monkeypatch, fake_embedder, populated=False)
    assert not app.exception
    assert any("qualified human verification" in warning.value for warning in app.warning)
    assert any("No maintenance history" in info.value for info in app.info)
    assert fake_embedder.calls == []


def test_retrieval_only_page_displays_facts_and_all_structured_sections(tmp_path, monkeypatch, fake_embedder):
    app, path = setup_app(tmp_path, monkeypatch, fake_embedder)
    with connect(path) as connection:
        before = MaintenanceStore(connection).list_issues()
    assert find(app.radio, "Assistant mode").value == "Retrieval only (no cloud request)"
    find(app.text_area, "Describe the equipment problem").input("Vacuum gripper loses suction")
    find(app.button, "Get troubleshooting guidance").click().run()
    assert not app.exception
    assert len(app.expander) == 5
    titles = {item.value for item in app.subheader}
    assert {"Issue summary", "Possible causes", "Recommended diagnostic checks", "Suggested next steps",
            "Evidence references", "Uncertainty or missing information"} <= titles
    assert any("no AI generation" in item.value for item in app.caption)
    assert any("No equipment or ticket changes" in item.value for item in app.caption)
    with connect(path) as connection:
        assert MaintenanceStore(connection).list_issues() == before


def test_cloud_opt_in_mock_generation_and_equipment_filter(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr("maintainai.rag.provider_from_environment", lambda: DeterministicMockProvider())
    app, path = setup_app(tmp_path, monkeypatch, fake_embedder)
    find(app.radio, "Assistant mode").set_value("Cloud-assisted RAG").run()
    assert find(app.button, "Get troubleshooting guidance").disabled
    find(app.checkbox, "Allow sending this question and retrieved excerpts to the configured provider").check().run()
    assert not find(app.button, "Get troubleshooting guidance").disabled
    with connect(path) as connection:
        gantry_id = next(row["id"] for row in MaintenanceStore(connection).list_equipment() if row["asset_tag"] == "DEMO-ZAB-001")
    find(app.selectbox, "Equipment (optional)").set_value(gantry_id)
    find(app.text_area, "Describe the equipment problem").input("Zaber gantry homing fault")
    find(app.button, "Get troubleshooting guidance").click().run()
    assert not app.exception
    assert len(app.expander) == 4
    assert all("Zaber gantry" in item.label for item in app.expander)
    assert any("AI-generated hypotheses" in item.value for item in app.caption)
    assert any("Historical evidence: Issue #" in item.value for item in app.caption)


def test_missing_cloud_configuration_displays_retrieval_fallback(tmp_path, monkeypatch, fake_embedder):
    app, _ = setup_app(tmp_path, monkeypatch, fake_embedder)
    find(app.radio, "Assistant mode").set_value("Cloud-assisted RAG").run()
    find(app.checkbox, "Allow sending this question and retrieved excerpts to the configured provider").check().run()
    find(app.text_area, "Describe the equipment problem").input("Conveyor belt slips")
    find(app.button, "Get troubleshooting guidance").click().run()
    assert not app.exception
    assert any("MAINTAINAI_LLM_API_KEY" in item.value for item in app.warning)
    assert len(app.expander) == 5
    assert any("no AI generation" in item.value for item in app.caption)


def test_stale_index_disables_assistant_submission(tmp_path, monkeypatch, fake_embedder):
    app, path = setup_app(tmp_path, monkeypatch, fake_embedder)
    with connect(path) as connection:
        store = MaintenanceStore(connection)
        store.update_issue(store.list_issues()[0]["id"], status="Closed", root_cause="Synthetic user edit")
    app.run()
    assert not app.exception
    assert find(app.button, "Get troubleshooting guidance").disabled
    assert any("rebuild the search index" in item.value for item in app.warning)
