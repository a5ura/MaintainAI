from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect
from maintainai.embeddings import EmbeddingError
from maintainai.service import MaintenanceStore


def find(elements, label):
    return next(element for element in elements if element.label == label)


def test_empty_search_page_does_not_load_model(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.radio, "Navigate").set_value("Similar Issues").run()
    assert not app.exception
    assert any("No maintenance issues" in info.value for info in app.info)
    assert fake_embedder.calls == []


def test_rebuild_search_and_stale_result_blocking_through_ui(tmp_path, monkeypatch, fake_embedder):
    path = tmp_path / "semantic_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.button, "Load synthetic demo data").click().run()
    find(app.radio, "Navigate").set_value("Similar Issues").run()
    assert not app.exception
    assert find(app.button, "Find similar issues").disabled
    assert fake_embedder.calls == []
    find(app.button, "Rebuild / refresh search index").click().run()
    assert not app.exception
    assert not find(app.button, "Find similar issues").disabled
    find(app.text_area, "Maintenance problem description").input("vacuum gripper suction seal leak")
    find(app.button, "Find similar issues").click().run()
    assert not app.exception
    assert len([item for item in app.subheader if "Issue #" in item.value]) == 5
    assert any("Original record: Issues" in item.value for item in app.caption)
    assert any("Vacuum gripper" in item.value for item in app.text)
    assert any("Root cause" in item.value for item in app.markdown)
    # Edit a ticket through the existing v0.1 issue form, then revisit search.
    find(app.radio, "Navigate").set_value("Issues").run()
    find(app.text_area, "Root cause").input("Synthetic updated finding")
    find(app.button, "Save issue updates").click().run()
    calls = len(fake_embedder.calls)
    find(app.radio, "Navigate").set_value("Similar Issues").run()
    assert not app.exception
    assert find(app.button, "Find similar issues").disabled
    assert any("Stale results are hidden" in item.value for item in app.warning)
    assert not any("Issue #" in item.value for item in app.subheader)
    assert len(fake_embedder.calls) == calls
    with connect(path) as connection:
        assert MaintenanceStore(connection).dashboard()["total_issues"] == 20
    find(app.button, "Rebuild / refresh search index").click().run()
    assert not app.exception
    assert not find(app.button, "Find similar issues").disabled


def test_model_failure_is_displayed_without_runtime_exception(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(tmp_path / "model_error.db"))
    fake_embedder.encode = Mock(side_effect=EmbeddingError("Model unavailable: simulated offline download failure"))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.button, "Load synthetic demo data").click().run()
    find(app.radio, "Navigate").set_value("Similar Issues").run()
    find(app.button, "Rebuild / refresh search index").click().run()
    assert not app.exception
    assert "Model unavailable" in app.error[0].value
