from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.seed import seed_demo
from maintainai.service import MaintenanceStore
from maintainai.ui import PAGES


def find(elements, label):
    return next(element for element in elements if element.label == label)


def test_empty_dashboard_and_demo_loader(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["0", "0", "0", "0"]
    find(app.button, "Load synthetic demo data").click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["5", "20", "3", "17"]
    find(app.button, "Load synthetic demo data").click().run()
    assert [metric.value for metric in app.metric] == ["5", "20", "3", "17"]
    for page in PAGES:
        find(app.radio, "Navigate").set_value(page).run()
        assert not app.exception, page


def test_create_and_close_issue_through_ui(tmp_path, monkeypatch):
    path = tmp_path / "ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    with connect(path) as connection:
        initialize(connection)
        seed_demo(connection)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.radio, "Navigate").set_value("Create issue").run()
    find(app.text_area, "Description *").input("Synthetic UI test: sensor alarm")
    find(app.text_input, "Reporter *").input("Demo tester")
    find(app.selectbox, "Priority").set_value("High")
    find(app.button, "Create issue").click().run()
    assert not app.exception
    assert app.success
    find(app.radio, "Navigate").set_value("Issues").run()
    find(app.selectbox, "Issue status").set_value("Closed")
    find(app.text_area, "Troubleshooting notes").input("Inspected synthetic sensor")
    find(app.text_area, "Root cause").input("Synthetic loose connector")
    find(app.text_area, "Corrective action").input("Secured demo connector")
    find(app.button, "Save issue updates").click().run()
    assert not app.exception
    with connect(path) as connection:
        records = MaintenanceStore(connection).list_issues(search="Synthetic UI test")
    assert len(records) == 1
    assert records[0]["status"] == "Closed"
    assert records[0]["root_cause"] == "Synthetic loose connector"


def test_blank_issue_form_displays_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(tmp_path / "validation.db"))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.button, "Load synthetic demo data").click().run()
    find(app.radio, "Navigate").set_value("Create issue").run()
    find(app.button, "Create issue").click().run()
    assert not app.exception
    assert "description is required" in app.error[0].value


def test_invalid_database_parent_displays_error(tmp_path, monkeypatch):
    parent_file = tmp_path / "not_a_folder"
    parent_file.write_text("Existing file", encoding="utf-8")
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(parent_file / "app.db"))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert "database could not be read or written" in app.error[0].value


def test_equipment_registration_and_condition_update(tmp_path, monkeypatch):
    path = tmp_path / "equipment_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.radio, "Navigate").set_value("Equipment").run()
    find(app.text_input, "Asset tag *").input("AUDIT-001")
    find(app.text_input, "Name *").input("Synthetic audit conveyor")
    find(app.text_input, "Category *").input("Material handling")
    find(app.button, "Register equipment").click().run()
    assert not app.exception
    find(app.selectbox, "New condition").set_value("Maintenance")
    find(app.button, "Save condition").click().run()
    assert not app.exception
    with connect(path) as connection:
        equipment = MaintenanceStore(connection).list_equipment()
    assert len(equipment) == 1
    assert equipment[0]["asset_tag"] == "AUDIT-001"
    assert equipment[0]["operational_status"] == "Maintenance"
