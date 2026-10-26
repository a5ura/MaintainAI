from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.log_import import import_logs_csv
from maintainai.service import LOG_SEVERITIES, MaintenanceStore

HEADER = "equipment_asset_tag,timestamp,severity,message,event_code,source\n"
VALID_CSV = (HEADER +
             "LOG-A,2026-10-25T11:30:00+02:00,ERROR,Synthetic pressure fault,VAC-LOW,Demo PLC\n" +
             "LOG-B,2026-10-25T10:00:00Z,INFO,Synthetic conveyor started,START,Demo controller\n").encode("utf-8")


def find(elements, label):
    return next(element for element in elements if element.label == label)


def setup_app(tmp_path, monkeypatch, *, with_equipment=True, with_logs=False):
    path = tmp_path / "log_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    ids = []
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        if with_equipment:
            for tag, name in (("LOG-A", "Synthetic robot"), ("LOG-B", "Synthetic conveyor"),
                              ("LOG-C", "Synthetic vision station")):
                ids.append(store.create_equipment(asset_tag=tag, name=name, category="Synthetic test equipment"))
        if with_logs:
            store.import_log_records([
                {"equipment_id": ids[0], "timestamp": "2026-10-25T12:00:00Z", "severity": "INFO",
                 "event_code": "START", "message": "Synthetic robot started", "source": "Demo PLC"},
                {"equipment_id": ids[0], "timestamp": "2026-10-25T09:00:00-04:00", "severity": "ERROR",
                 "event_code": "VAC-LOW", "message": "Synthetic vacuum fault", "source": "Demo PLC"},
                {"equipment_id": ids[1], "timestamp": "2026-10-25T14:00:00Z", "severity": "ERROR",
                 "event_code": "BELT", "message": "Synthetic conveyor fault", "source": "Demo controller"},
            ])
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert "Equipment Logs" in find(app.radio, "Navigate").options
    find(app.radio, "Navigate").set_value("Equipment Logs").run()
    return app, path, ids


def read_logs(path):
    with connect(path) as connection:
        return MaintenanceStore(connection).list_logs()


def upload(app, content=VALID_CSV):
    find(app.file_uploader, "Choose equipment logs CSV").set_value(("synthetic_logs.csv", content, "text/csv")).run()


def test_log_page_renders_empty_workspace_and_all_filters(tmp_path, monkeypatch):
    app, path, _ = setup_app(tmp_path, monkeypatch, with_equipment=False)
    assert not app.exception
    assert any(item.value == "Equipment Logs" for item in app.header)
    assert any("No equipment registered" in item.value for item in app.info)
    assert any("No equipment logs yet" in item.value for item in app.info)
    assert find(app.selectbox, "Equipment filter").options == ["All equipment"]
    assert find(app.selectbox, "Severity filter").options == ["All severities", *LOG_SEVERITIES]
    assert find(app.button, "Import logs").disabled
    assert find(app.file_uploader, "Choose equipment logs CSV").allowed_type == [".csv"]
    assert not app.dataframe
    assert read_logs(path) == []


def test_log_page_with_equipment_but_no_logs_is_informative(tmp_path, monkeypatch):
    app, path, _ = setup_app(tmp_path, monkeypatch)
    assert not app.exception
    assert any("No equipment logs yet" in item.value for item in app.info)
    assert find(app.button, "Import logs").disabled
    assert len(find(app.selectbox, "Equipment filter").options) == 4
    assert any("timezone" in item.value for item in app.markdown)
    assert any("UTC" in item.value for item in app.caption)
    assert read_logs(path) == []


def test_log_table_shows_required_columns_utc_and_newest_events_first(tmp_path, monkeypatch):
    app, _, _ = setup_app(tmp_path, monkeypatch, with_logs=True)
    assert not app.exception
    assert len(app.dataframe) == 1
    frame = app.dataframe[0].value
    assert list(frame.columns) == ["Timestamp (UTC)", "Asset Tag", "Equipment Name", "Severity", "Event Code", "Message", "Source"]
    assert list(frame["Timestamp (UTC)"]) == ["2026-10-25T14:00:00.000000+00:00",
                                              "2026-10-25T13:00:00.000000+00:00",
                                              "2026-10-25T12:00:00.000000+00:00"]
    assert frame.iloc[0]["Asset Tag"] == "LOG-B"
    assert frame.iloc[0]["Equipment Name"] == "Synthetic conveyor"
    assert frame.iloc[0]["Severity"] == "ERROR"
    assert frame.iloc[0]["Event Code"] == "BELT"
    assert frame.iloc[0]["Message"] == "Synthetic conveyor fault"
    assert frame.iloc[0]["Source"] == "Demo controller"


def test_import_success_count_and_immediate_display_use_real_importer(tmp_path, monkeypatch):
    app, path, _ = setup_app(tmp_path, monkeypatch)
    with connect(path) as connection:
        before_equipment = MaintenanceStore(connection).list_equipment()
    upload(app)
    assert not find(app.button, "Import logs").disabled
    find(app.button, "Import logs").click().run()
    assert not app.exception
    assert any(item.value == "Imported 2 equipment log record(s)." for item in app.success)
    assert len(app.dataframe[0].value) == 2
    assert len(read_logs(path)) == 2
    with connect(path) as connection:
        store = MaintenanceStore(connection)
        assert store.list_equipment() == before_equipment
        assert store.list_issues() == []


@pytest.mark.parametrize("bad_row,error", [
    ("UNKNOWN,2026-10-25T10:00:00Z,ERROR,Synthetic fault,E,Demo\n", "Row 3: Unknown equipment_asset_tag"),
    ("LOG-A,2026-10-25T10:00:00,ERROR,Synthetic fault,E,Demo\n", "Row 3: timestamp"),
    ("LOG-A,2026-10-25T10:00:00Z,DEBUG,Synthetic fault,E,Demo\n", "Row 3: severity"),
    ("LOG-A,2026-10-25T10:00:00Z,ERROR\n", "Row 3: Column count"),
])
def test_import_failure_displays_row_errors_and_preserves_stored_logs(tmp_path, monkeypatch, bad_row, error):
    app, path, _ = setup_app(tmp_path, monkeypatch, with_logs=True)
    before = read_logs(path)
    valid_first = "LOG-A,2026-10-25T10:00:00Z,INFO,Synthetic valid first event,E,Demo\n"
    upload(app, (HEADER + valid_first + bad_row).encode("utf-8"))
    find(app.button, "Import logs").click().run()
    assert not app.exception
    assert any(item.value == "Log import rejected. No logs were added." for item in app.error)
    assert any(error in item.value for item in app.text)
    assert not app.success
    assert read_logs(path) == before
    assert len(app.dataframe[0].value) == len(before)


def test_equipment_and_severity_filters_include_all_and_empty_matches(tmp_path, monkeypatch):
    app, _, ids = setup_app(tmp_path, monkeypatch, with_logs=True)
    assert len(app.dataframe[0].value) == 3
    find(app.selectbox, "Equipment filter").set_value(ids[0]).run()
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[0].value["Asset Tag"]) == {"LOG-A"}
    find(app.selectbox, "Severity filter").set_value("ERROR").run()
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[0].value.iloc[0]["Message"] == "Synthetic vacuum fault"
    find(app.selectbox, "Equipment filter").set_value(None).run()
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[0].value["Severity"]) == {"ERROR"}
    find(app.selectbox, "Severity filter").set_value(None).run()
    assert len(app.dataframe[0].value) == 3
    find(app.selectbox, "Equipment filter").set_value(ids[2]).run()
    assert not app.exception
    assert not app.dataframe
    assert any("No logs match the selected filters" in item.value for item in app.info)
    find(app.selectbox, "Equipment filter").set_value(None).run()
    assert len(app.dataframe[0].value) == 3
    find(app.selectbox, "Severity filter").set_value("CRITICAL").run()
    assert not app.dataframe
    assert any("No logs match" in item.value for item in app.info)


def test_upload_and_reruns_do_not_import_without_explicit_button(tmp_path, monkeypatch):
    importer = Mock(wraps=import_logs_csv)
    monkeypatch.setattr("maintainai.log_ui.import_logs_csv", importer)
    app, path, _ = setup_app(tmp_path, monkeypatch)
    upload(app)
    app.run()
    find(app.selectbox, "Severity filter").set_value("ERROR").run()
    importer.assert_not_called()
    assert read_logs(path) == []
    find(app.button, "Import logs").click().run()
    importer.assert_called_once()
    assert len(read_logs(path)) == 2
    app.run()
    find(app.selectbox, "Severity filter").set_value(None).run()
    assert find(app.file_uploader, "Choose equipment logs CSV").value is not None
    assert importer.call_count == 1
    assert len(read_logs(path)) == 2
    find(app.radio, "Navigate").set_value("Equipment").run()
    find(app.radio, "Navigate").set_value("Equipment Logs").run()
    assert not app.exception
    assert importer.call_count == 1
    assert len(read_logs(path)) == 2
