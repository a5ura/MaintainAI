from datetime import datetime, timezone
import sqlite3
from unittest.mock import Mock

import pytest

from maintainai.csv_import import CSVImportError
from maintainai.database import connect
from maintainai.log_import import MAX_BYTES, MAX_ROWS, import_logs_csv
from maintainai.service import LOG_SEVERITIES, MaintenanceStore, ValidationError

HEADER = "equipment_asset_tag,timestamp,severity,message\n"
VALID_ROW = "TEST-001,2026-10-25T09:30:00-04:00,INFO,Synthetic equipment event\n"


@pytest.mark.parametrize("as_bytes", [False, True], ids=["text", "bytes"])
def test_import_success_is_persistent_and_uses_one_batch(store, equipment_id, db_path, monkeypatch, as_bytes):
    before_equipment, before_issues = store.list_equipment(), store.list_issues()
    batch = Mock(wraps=store.import_log_records)
    monkeypatch.setattr(store, "import_log_records", batch)
    content = HEADER + "".join(VALID_ROW.replace(",INFO,", f",{severity},") for severity in LOG_SEVERITIES)
    assert import_logs_csv(store, content.encode("utf-8") if as_bytes else content) == 4
    batch.assert_called_once()
    submitted = batch.call_args.args[0]
    assert len(submitted) == 4
    assert {record["equipment_id"] for record in submitted} == {equipment_id}
    assert all(record["timestamp"] == "2026-10-25T13:30:00.000000+00:00" for record in submitted)
    assert all("equipment_asset_tag" not in record for record in submitted)
    with connect(db_path) as connection:
        reopened = MaintenanceStore(connection)
        logs = reopened.list_logs()
        assert len(logs) == 4
        assert {row["severity"] for row in logs} == set(LOG_SEVERITIES)
        assert all(row["event_code"] == row["source"] == "" for row in logs)
        assert all(row["asset_tag"] == "TEST-001" and row["equipment_name"] == "Test robot" for row in logs)
        assert all(datetime.fromisoformat(row["imported_at"]).utcoffset().total_seconds() == 0 for row in logs)
        assert reopened.list_equipment() == before_equipment
        assert reopened.list_issues() == before_issues


def test_bom_utf8_crlf_optional_fields_and_quoted_messages(store, equipment_id):
    content = ('\ufeffequipment_asset_tag,timestamp,severity,message,event_code,source\r\n'
               '  teSt-001  , 2026-10-25T09:30:00Z , WARNING ,"Synthetic: café, pressure ""low""", VAC-LOW , Demo PLC \r\n')
    assert import_logs_csv(store, content.encode("utf-8")) == 1
    row = store.list_logs()[0]
    assert row["equipment_id"] == equipment_id
    assert row["message"] == 'Synthetic: café, pressure "low"'
    assert row["event_code"] == "VAC-LOW"
    assert row["source"] == "Demo PLC"
    assert row["severity"] == "WARNING"


def test_text_bom_and_blank_optional_cells(store, equipment_id):
    content = "\ufeff" + HEADER.rstrip() + ",event_code,source\n" + VALID_ROW.rstrip() + ",  ,\n"
    assert import_logs_csv(store, content) == 1
    assert store.list_logs()[0]["event_code"] == store.list_logs()[0]["source"] == ""


def test_multiple_assets_and_ascii_only_matching(store):
    first = store.create_equipment(asset_tag="STRASSE", name="Synthetic first", category="Test")
    second = store.create_equipment(asset_tag="Stra\u00dfe", name="Synthetic second", category="Test")
    content = HEADER + "strasse,2026-10-25T09:30:00Z,INFO,Synthetic first event\nStra\u00dfe,2026-10-25T09:30:00Z,ERROR,Synthetic second event\n"
    assert import_logs_csv(store, content) == 2
    assert [row["message"] for row in store.list_logs(first)] == ["Synthetic first event"]
    assert [row["message"] for row in store.list_logs(second)] == ["Synthetic second event"]


def test_unknown_equipment_has_row_error_and_is_never_created(store, equipment_id, monkeypatch):
    before = store.list_equipment()
    batch = Mock(wraps=store.import_log_records)
    monkeypatch.setattr(store, "import_log_records", batch)
    content = HEADER + VALID_ROW + VALID_ROW.replace("TEST-001", "NOT-REGISTERED")
    with pytest.raises(CSVImportError, match="Row 3: Unknown equipment_asset_tag 'NOT-REGISTERED'"):
        import_logs_csv(store, content)
    batch.assert_not_called()
    assert store.list_logs() == []
    assert store.list_equipment() == before


@pytest.mark.parametrize("value", ["", "  ", "yesterday", "2026-10-25", "2026-10-25T09:30:00",
                                  "2026-02-30T09:30:00Z", "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_bad_timestamps_reject_the_whole_file_with_row_numbers(store, equipment_id, value):
    content = HEADER + VALID_ROW + f"TEST-001,{value},INFO,Synthetic invalid event\n"
    with pytest.raises(CSVImportError, match="Row 3: timestamp"):
        import_logs_csv(store, content)
    assert store.list_logs() == []


@pytest.mark.parametrize("value", ["", "DEBUG", "warning"])
def test_bad_severity_is_a_row_error(store, equipment_id, value):
    with pytest.raises(CSVImportError, match="Row 2: severity must be one of"):
        import_logs_csv(store, HEADER + VALID_ROW.replace(",INFO,", f",{value},"))
    assert store.list_logs() == []


@pytest.mark.parametrize("field,error", [("equipment_asset_tag", "equipment_asset_tag is required"),
                                       ("message", "message is required")])
def test_blank_required_cells_are_rejected(store, equipment_id, field, error):
    values = {"equipment_asset_tag": "TEST-001", "timestamp": "2026-10-25T09:30:00Z",
              "severity": "INFO", "message": "Synthetic event"}
    values[field] = "  "
    with pytest.raises(CSVImportError, match="Row 2: " + error):
        import_logs_csv(store, HEADER + ",".join(values.values()) + "\n")


@pytest.mark.parametrize("field,limit", [("message", 10000), ("event_code", 200), ("source", 200)])
def test_field_limits_use_log_validation_and_row_errors(store, equipment_id, field, limit):
    headers = HEADER.rstrip() + ",event_code,source\n"
    values = {"equipment_asset_tag": "TEST-001", "timestamp": "2026-10-25T09:30:00Z", "severity": "INFO",
              "message": "Synthetic event", "event_code": "", "source": ""}
    values[field] = "x" * (limit + 1)
    with pytest.raises(CSVImportError, match=f"Row 2: {field} must be at most {limit}"):
        import_logs_csv(store, headers + ",".join(values.values()) + "\n")
    assert store.list_logs() == []


@pytest.mark.parametrize("content,error", [
    pytest.param("", "empty", id="empty-text"),
    pytest.param(b"", "empty", id="empty-bytes"),
    pytest.param("\ufeff\n\n", "empty", id="bom-blank"),
    pytest.param(HEADER, "no data rows", id="header-only"),
    pytest.param(HEADER + "\n\n", "no data rows", id="blank-rows-only"),
    pytest.param("equipment_asset_tag,severity,message\n", "Missing required columns: timestamp", id="missing-header"),
    pytest.param(HEADER.rstrip() + ",severity\n", "duplicate column names", id="duplicate-header"),
    pytest.param(HEADER.rstrip() + ",typo\n", "Unknown columns: typo", id="unknown-header"),
    pytest.param(HEADER.rstrip() + ",imported_at\n", "Unknown columns: imported_at", id="import-time-not-user-input"),
    pytest.param(HEADER.rstrip() + ",equipment_id\n", "Unknown columns: equipment_id", id="no-id-bypass"),
    pytest.param(HEADER.rstrip() + ",\n", "Unknown columns: <empty>", id="empty-header-name"),
    pytest.param(HEADER.replace("timestamp", "Timestamp"), "Missing required columns: timestamp", id="headers-case-sensitive"),
    pytest.param(HEADER + "TEST-001,2026-10-25T09:30:00Z,INFO\n", "Row 2: Column count", id="missing-cell"),
    pytest.param(HEADER + VALID_ROW.rstrip() + ",extra\n", "Row 2: Column count", id="extra-cell"),
    pytest.param(HEADER + 'TEST-001,2026-10-25T09:30:00Z,INFO,"unclosed\n', "Malformed CSV near line", id="unclosed-quote"),
    pytest.param(HEADER + 'TEST-001,2026-10-25T09:30:00Z,INFO,"message"extra\n', "Malformed CSV near line", id="bad-quote-suffix"),
    pytest.param(HEADER + VALID_ROW.replace("Synthetic", "\x00Synthetic"), "null characters", id="null-in-message"),
    pytest.param(HEADER.replace("timestamp", "time\x00stamp"), "null characters", id="null-in-header"),
    pytest.param(b"\xff", "UTF-8", id="invalid-utf8"),
    pytest.param((HEADER + VALID_ROW).encode("utf-16"), "UTF-8", id="utf16-is-not-utf8"),
    pytest.param("\ud800", "UTF-8", id="unencodable-text"),
])
def test_invalid_files_make_no_changes(store, equipment_id, monkeypatch, content, error):
    batch = Mock(wraps=store.import_log_records)
    monkeypatch.setattr(store, "import_log_records", batch)
    with pytest.raises(CSVImportError, match=error):
        import_logs_csv(store, content)
    batch.assert_not_called()
    assert store.list_logs() == []


@pytest.mark.parametrize("kind", ["bytes", "text", "multibyte-text"])
def test_real_five_mib_limit_is_measured_in_utf8_bytes(store, kind):
    assert MAX_BYTES == 5 * 1024 * 1024
    content = "x" * (MAX_BYTES + 1) if kind != "multibyte-text" else "é" * (MAX_BYTES // 2 + 1)
    if kind == "bytes":
        content = content.encode("utf-8")
    with pytest.raises(CSVImportError, match="5 MiB size limit"):
        import_logs_csv(store, content)
    assert store.list_logs() == []


def test_exact_byte_limit_is_allowed_and_one_more_byte_is_rejected(store, equipment_id, monkeypatch):
    content = HEADER + VALID_ROW
    monkeypatch.setattr("maintainai.log_import.MAX_BYTES", len(content.encode("utf-8")))
    assert import_logs_csv(store, content) == 1
    with pytest.raises(CSVImportError, match="size limit"):
        import_logs_csv(store, content + "\n")
    assert len(store.list_logs()) == 1


@pytest.mark.parametrize("row_count", [10000, 10001])
def test_actual_ten_thousand_row_boundary(store, equipment_id, monkeypatch, row_count):
    assert MAX_ROWS == 10000
    batch = Mock(wraps=store.import_log_records)
    monkeypatch.setattr(store, "import_log_records", batch)
    content = HEADER + VALID_ROW * row_count
    if row_count == 10000:
        assert import_logs_csv(store, content) == 10000
        batch.assert_called_once()
        assert len(batch.call_args.args[0]) == 10000
    else:
        with pytest.raises(CSVImportError, match="10,000 row limit"):
            import_logs_csv(store, content)
        batch.assert_not_called()
    count = store.connection.execute("SELECT COUNT(*) FROM equipment_logs").fetchone()[0]
    assert count == (10000 if row_count == 10000 else 0)


def test_errors_use_logical_row_numbers_even_after_multiline_message(store, equipment_id):
    multiline = 'TEST-001,2026-10-25T09:30:00Z,INFO,"Synthetic first line\nSynthetic second line"\n'
    content = HEADER + multiline + VALID_ROW.replace("TEST-001", "UNKNOWN") + VALID_ROW.replace(",INFO,", ",DEBUG,")
    with pytest.raises(CSVImportError) as caught:
        import_logs_csv(store, content)
    assert len(caught.value.errors) == 2
    assert caught.value.errors[0].startswith("Row 3: Unknown equipment_asset_tag")
    assert caught.value.errors[1].startswith("Row 4: severity")
    assert store.list_logs() == []


def test_validation_stops_after_twenty_five_errors_without_writing(store, equipment_id):
    with pytest.raises(CSVImportError) as caught:
        import_logs_csv(store, HEADER + VALID_ROW.replace(",INFO,", ",DEBUG,") * 26)
    assert len(caught.value.errors) == 26
    assert caught.value.errors[-1].startswith("Stopped after 25 errors")
    assert store.list_logs() == []


def test_database_failure_rolls_back_all_new_rows_and_preserves_existing_logs(store, equipment_id, db_path, monkeypatch):
    store.create_log(equipment_id=equipment_id, timestamp="2026-10-25T00:00:00Z", severity="INFO", message="Synthetic existing log")
    before = store.list_logs()
    equipment_before = store.list_equipment()
    # Only this isolated test connection has this temporary failure-injection trigger.
    store.connection.executescript("""CREATE TEMP TRIGGER reject_synthetic_import
        BEFORE INSERT ON equipment_logs WHEN NEW.event_code = 'REJECT'
        BEGIN SELECT RAISE(ABORT, 'Synthetic import failure'); END;""")
    batch = Mock(wraps=store.import_log_records)
    monkeypatch.setattr(store, "import_log_records", batch)
    content = HEADER.rstrip() + ",event_code\n" + VALID_ROW.rstrip() + ",OK\n" + VALID_ROW.rstrip() + ",REJECT\n"
    with pytest.raises(CSVImportError, match="No logs were imported") as caught:
        import_logs_csv(store, content)
    batch.assert_called_once()
    assert isinstance(caught.value.__cause__, ValidationError)
    assert isinstance(caught.value.__cause__.__cause__, sqlite3.IntegrityError)
    assert not store.connection.in_transaction
    with connect(db_path) as connection:
        reopened = MaintenanceStore(connection)
        assert reopened.list_logs() == before
        assert reopened.list_equipment() == equipment_before


def test_valid_quoted_multiline_message_is_preserved(store, equipment_id):
    message = "Synthetic first line\nSynthetic second line"
    assert import_logs_csv(store, HEADER + f'TEST-001,2026-10-25T09:30:00Z,INFO,"{message}"\n') == 1
    assert store.list_logs()[0]["message"] == message
