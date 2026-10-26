from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from maintainai.service import LOG_SEVERITIES, ValidationError, timestamp, validate_log


@pytest.fixture
def log_record():
    return {
        "equipment_id": 1,
        "timestamp": "2026-10-25T09:30:00-04:00",
        "severity": "WARNING",
        "message": "Synthetic demo: vacuum pressure dropped during a test cycle.",
    }


def test_log_severities_match_supported_values():
    assert LOG_SEVERITIES == ('INFO', 'WARNING', 'ERROR', 'CRITICAL')


@pytest.mark.parametrize("severity", LOG_SEVERITIES)
def test_log_valid_records_have_normalized_values_and_generated_import_time(log_record, monkeypatch, severity):
    now = "2026-10-25T15:45:00.123456+00:00"
    clock = Mock(return_value=now)
    monkeypatch.setattr("maintainai.service.utc_now", clock)
    log_record.update(severity=severity, message="  Synthetic event  ", event_code="  VAC-LOW  ", source="  Demo PLC  ",
                      imported_at="Untrusted user-supplied import time", unrelated="Not a log field")
    before = deepcopy(log_record)
    assert validate_log(log_record) == {
        "equipment_id": 1,
        "timestamp": "2026-10-25T13:30:00.000000+00:00",
        "severity": severity,
        "message": "Synthetic event",
        "event_code": "VAC-LOW",
        "source": "Demo PLC",
        "imported_at": now,
    }
    clock.assert_called_once_with()
    assert log_record == before


def test_log_optional_fields_default_to_empty_and_import_time_is_utc(log_record):
    before = datetime.now(timezone.utc)
    result = validate_log(log_record)
    after = datetime.now(timezone.utc)
    assert result["event_code"] == result["source"] == ""
    imported = datetime.fromisoformat(result["imported_at"])
    assert imported.utcoffset().total_seconds() == 0
    assert before <= imported <= after


@pytest.mark.parametrize("field", ["event_code", "source"])
def test_log_blank_and_null_optional_text_is_empty(log_record, field):
    for value in (None, "", "  "):
        log_record[field] = value
        assert validate_log(log_record)[field] == ""


@pytest.mark.parametrize("field", ["equipment_id", "timestamp", "severity", "message"])
def test_log_required_fields_cannot_be_omitted(log_record, monkeypatch, field):
    clock = Mock(side_effect=AssertionError("Invalid records must not obtain a default timestamp"))
    monkeypatch.setattr("maintainai.service.utc_now", clock)
    del log_record[field]
    with pytest.raises(ValidationError, match=f"^{field}\\b"):
        validate_log(log_record)
    clock.assert_not_called()


@pytest.mark.parametrize("value", [None, 0, -1, True, False, 1.0, 1.5, "1", [], {}])
def test_log_equipment_id_requires_a_positive_integer(log_record, value):
    log_record["equipment_id"] = value
    with pytest.raises(ValidationError, match="^equipment_id must be a positive integer"):
        validate_log(log_record)


@pytest.mark.parametrize("value", [None, "", "  ", 123, True, [], "not-a-date", "2026-10-25",
    "2026-10-25T09:30:00", "2026-02-30T09:30:00Z", "2026-10-25T09:30:00+25:00",
    "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_log_invalid_event_timestamps_are_field_specific_and_never_default(log_record, monkeypatch, value):
    clock = Mock(side_effect=AssertionError("Missing/invalid event times must not default to now"))
    monkeypatch.setattr("maintainai.service.utc_now", clock)
    log_record["timestamp"] = value
    with pytest.raises(ValidationError, match="^timestamp\\b") as caught:
        validate_log(log_record)
    assert "created_at" not in str(caught.value)
    clock.assert_not_called()


@pytest.mark.parametrize("value,expected", [
    ("2026-10-25T09:30:00Z", "2026-10-25T09:30:00.000000+00:00"),
    ("2026-10-25T09:30:00.123456+00:00", "2026-10-25T09:30:00.123456+00:00"),
    ("2026-10-25T00:30:00+05:30", "2026-10-24T19:00:00.000000+00:00"),
    ("  2026-10-25T23:30:00-02:00  ", "2026-10-26T01:30:00.000000+00:00"),
])
def test_log_timestamp_normalizes_offsets_and_preserves_precision(log_record, value, expected):
    log_record["timestamp"] = value
    assert validate_log(log_record)["timestamp"] == expected


@pytest.mark.parametrize("value", [None, "", "DEBUG", "warning", " WARNING ", True, []])
def test_log_invalid_severities_are_rejected(log_record, value):
    log_record["severity"] = value
    with pytest.raises(ValidationError, match="^severity must be one of: INFO, WARNING, ERROR, CRITICAL"):
        validate_log(log_record)


@pytest.mark.parametrize("value", [None, "", "  ", "\n\t"])
def test_log_message_must_not_be_empty(log_record, value):
    log_record["message"] = value
    with pytest.raises(ValidationError, match="^message is required"):
        validate_log(log_record)


@pytest.mark.parametrize("field", ["message", "event_code", "source"])
@pytest.mark.parametrize("value", [123, True, []])
def test_log_text_fields_reject_nontext_values(log_record, field, value):
    log_record[field] = value
    with pytest.raises(ValidationError, match=f"^{field} must be text"):
        validate_log(log_record)


@pytest.mark.parametrize("field,limit", [("message", 10000), ("event_code", 200), ("source", 200)])
def test_log_text_length_boundaries(log_record, field, limit):
    log_record[field] = "x" * limit
    assert validate_log(log_record)[field] == log_record[field]
    log_record[field] += "x"
    with pytest.raises(ValidationError, match=f"^{field} must be at most {limit} characters"):
        validate_log(log_record)


def test_log_timestamp_helper_keeps_existing_created_at_error_convention():
    with pytest.raises(ValidationError, match="^created_at must be an ISO 8601 timestamp"):
        timestamp("not-a-date")
