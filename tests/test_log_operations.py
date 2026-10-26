from copy import deepcopy
import sqlite3

import pytest

from maintainai.database import connect
from maintainai.service import LOG_SEVERITIES, MaintenanceStore, ValidationError


def log_record(equipment_id, **overrides):
    record = {"equipment_id": equipment_id, "timestamp": "2026-10-25T09:30:00-04:00",
              "severity": "INFO", "message": "Synthetic demo: equipment event"}
    record.update(overrides)
    return record


@pytest.mark.parametrize("severity", LOG_SEVERITIES)
def test_create_log_stores_normalized_fields_and_equipment_details(store, equipment_id, db_path, monkeypatch, severity):
    imported_at = "2026-10-25T15:00:00.000000+00:00"
    monkeypatch.setattr("maintainai.service.utc_now", lambda: imported_at)
    record = log_record(equipment_id, severity=severity, message="  Synthetic event  ",
                        event_code="  VAC-LOW  ", source="  Demo PLC  ", imported_at="ignored input")
    issue_before = store.list_issues()
    equipment_before = store.list_equipment()
    log_id = store.create_log(**record)
    assert type(log_id) is int and log_id > 0
    with connect(db_path) as connection:
        reopened = MaintenanceStore(connection)
        assert reopened.list_logs() == [{
            "id": log_id, "equipment_id": equipment_id, "timestamp": "2026-10-25T13:30:00.000000+00:00",
            "severity": severity, "message": "Synthetic event", "event_code": "VAC-LOW", "source": "Demo PLC",
            "imported_at": imported_at, "asset_tag": "TEST-001", "equipment_name": "Test robot",
        }]
        assert reopened.list_issues() == issue_before
        assert reopened.list_equipment() == equipment_before


def test_batch_import_returns_ids_in_input_order_and_preserves_input(store, equipment_id):
    second = store.create_equipment(asset_tag="TEST-002", name="Synthetic conveyor", category="Transport")
    records = [log_record(equipment_id, message="Synthetic first", timestamp="2026-10-25T12:00:00Z"),
               log_record(second, message="Synthetic second", severity="ERROR", timestamp="2026-10-25T11:00:00Z")]
    before = deepcopy(records)
    ids = store.import_log_records(records)
    assert len(ids) == 2 and len(set(ids)) == 2
    by_id = {row["id"]: row for row in store.list_logs()}
    assert [by_id[log_id]["message"] for log_id in ids] == [record["message"] for record in records]
    assert by_id[ids[1]]["equipment_name"] == "Synthetic conveyor"
    assert by_id[ids[0]]["event_code"] == by_id[ids[0]]["source"] == ""
    assert records == before
    assert not store.connection.in_transaction


def test_log_order_is_event_timestamp_then_descending_id(store, equipment_id):
    ids = store.import_log_records([
        log_record(equipment_id, message="Synthetic noon first", timestamp="2026-10-25T12:00:00Z"),
        log_record(equipment_id, message="Synthetic earlier event", timestamp="2026-10-25T10:00:00Z"),
        log_record(equipment_id, message="Synthetic noon later ID", timestamp="2026-10-25T08:00:00-04:00"),
        log_record(equipment_id, message="Synthetic oldest event imported last", timestamp="2026-10-25T09:00:00Z"),
    ])
    assert [row["id"] for row in store.list_logs()] == [ids[2], ids[0], ids[1], ids[3]]


def test_log_filters_work_independently_and_together(store, equipment_id):
    second = store.create_equipment(asset_tag="TEST-002", name="Synthetic conveyor", category="Transport")
    ids = store.import_log_records([
        log_record(equipment_id, severity="INFO"), log_record(equipment_id, severity="ERROR"),
        log_record(second, severity="ERROR"), log_record(second, severity="WARNING"),
        log_record(equipment_id, severity="CRITICAL"),
    ])
    assert len(store.list_logs()) == 5
    assert {row["id"] for row in store.list_logs(equipment_id=equipment_id)} == {ids[0], ids[1], ids[4]}
    assert {row["id"] for row in store.list_logs(severity="ERROR")} == {ids[1], ids[2]}
    assert [row["id"] for row in store.list_logs(equipment_id, "ERROR")] == [ids[1]]
    assert store.list_logs(second, "INFO") == []
    for severity in LOG_SEVERITIES:
        assert all(row["severity"] == severity for row in store.list_logs(severity=severity))


@pytest.mark.parametrize("equipment_id", [True, False, 0, -1, 1.0, "1", [], {}])
def test_log_equipment_filter_rejects_invalid_types_and_values(store, equipment_id):
    with pytest.raises(ValidationError, match="^equipment_id must be a positive integer"):
        store.list_logs(equipment_id=equipment_id)


def test_log_equipment_filter_rejects_unregistered_equipment(store):
    with pytest.raises(ValidationError, match="Equipment 999999 does not exist"):
        store.list_logs(equipment_id=999999)


@pytest.mark.parametrize("severity", ["", "DEBUG", "error", "ERROR ", True, [], "ERROR' OR 1=1 --"])
def test_log_severity_filter_is_validated(store, severity):
    with pytest.raises(ValidationError, match="^severity must be one of"):
        store.list_logs(severity=severity)


def test_log_empty_import_and_queries_are_safe(store, equipment_id):
    assert store.list_logs() == []
    assert store.list_logs(equipment_id, "INFO") == []
    assert store.import_log_records([]) == []
    existing = store.create_log(**log_record(equipment_id))
    assert store.import_log_records([]) == []
    assert [row["id"] for row in store.list_logs()] == [existing]


@pytest.mark.parametrize("field,value", [("severity", "DEBUG"), ("timestamp", ""),
                                         ("equipment_id", True), ("message", " ")])
def test_invalid_batch_record_is_rejected_before_any_insert(store, equipment_id, field, value):
    store.create_log(**log_record(equipment_id, message="Synthetic existing event"))
    before = store.list_logs()
    statements = []
    store.connection.set_trace_callback(statements.append)
    invalid = log_record(equipment_id)
    invalid[field] = value
    try:
        with pytest.raises(ValidationError, match=f"^{field}\\b"):
            store.import_log_records([log_record(equipment_id), invalid])
    finally:
        store.connection.set_trace_callback(None)
    assert not any(statement.lstrip().upper().startswith("INSERT INTO EQUIPMENT_LOGS") for statement in statements)
    assert store.list_logs() == before


def test_all_equipment_references_are_checked_before_first_insert(store, equipment_id):
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValidationError, match="Equipment 999999 does not exist"):
            store.import_log_records([log_record(equipment_id), log_record(999999)])
    finally:
        store.connection.set_trace_callback(None)
    assert not any(statement.lstrip().upper().startswith("INSERT INTO EQUIPMENT_LOGS") for statement in statements)
    assert store.list_logs() == []


def test_single_log_rejects_missing_equipment(store):
    with pytest.raises(ValidationError, match="does not exist"):
        store.create_log(**log_record(999999))
    assert store.list_logs() == []


def test_database_integrity_failure_rolls_back_entire_batch_and_preserves_prior_data(store, equipment_id, db_path):
    existing = store.create_log(**log_record(equipment_id, message="Synthetic existing event"))
    before = store.list_logs()
    equipment_before = store.list_equipment()
    # This trigger exists only in this isolated test connection, not in the application schema.
    store.connection.executescript("""CREATE TEMP TRIGGER reject_synthetic_log
        BEFORE INSERT ON equipment_logs WHEN NEW.event_code = 'REJECT'
        BEGIN SELECT RAISE(ABORT, 'Synthetic mid-batch constraint failure'); END;""")
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValidationError, match="No logs were imported") as caught:
            store.import_log_records([
                log_record(equipment_id, message="Synthetic first insert must roll back"),
                log_record(equipment_id, event_code="REJECT"),
            ])
    finally:
        store.connection.set_trace_callback(None)
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    assert "Verify that referenced equipment" in str(caught.value)
    assert any(statement.strip().upper() == "ROLLBACK" for statement in statements)
    assert not store.connection.in_transaction
    with connect(db_path) as connection:
        reopened = MaintenanceStore(connection)
        assert reopened.list_logs() == before
        assert reopened.list_equipment() == equipment_before
    assert store.create_log(**log_record(equipment_id, message="Synthetic retry after rollback")) > existing
    assert len(store.list_logs()) == 2


def test_log_insert_treats_quotes_and_sql_as_literal_data(store, equipment_id):
    message = "Synthetic demo: operator's text '); DROP TABLE equipment; --"
    log_id = store.create_log(**log_record(equipment_id, message=message, event_code="E'42", source="Demo 'PLC'"))
    row = store.list_logs()[0]
    assert row["id"] == log_id and row["message"] == message
    assert row["event_code"] == "E'42" and row["source"] == "Demo 'PLC'"
    assert store.get_equipment(equipment_id)["name"] == "Test robot"
