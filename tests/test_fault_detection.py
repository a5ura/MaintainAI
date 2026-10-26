from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import permutations

import pytest

from maintainai.fault_detection import detect_fault_patterns

BASE_TIME = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


def log(log_id, seconds=0, *, equipment_id=1, severity="ERROR", event_code="SYNTHETIC-FAULT", **overrides):
    record = {"id": log_id, "equipment_id": equipment_id,
              "timestamp": (BASE_TIME + timedelta(seconds=seconds)).isoformat(),
              "severity": severity, "event_code": event_code, "message": "Synthetic demo event",
              "source": "Demo controller", "imported_at": "2026-10-25T20:00:00Z",
              "asset_tag": f"DEMO-{equipment_id}", "equipment_name": "Synthetic equipment"}
    record.update(overrides)
    return record


def patterns(logs, pattern_type):
    return [finding for finding in detect_fault_patterns(logs) if finding["pattern_type"] == pattern_type]


@pytest.mark.parametrize("logs", [[], [log(1)], [log(1, severity="WARNING")],
                                 [log(1, severity="INFO"), log(2, 10, severity="INFO")]])
def test_empty_or_insufficient_evidence_produces_no_findings(logs):
    assert detect_fault_patterns(logs) == []


@pytest.mark.parametrize("first,second,expected", [("ERROR", "ERROR", "ERROR"), ("ERROR", "CRITICAL", "CRITICAL"),
                                                  ("CRITICAL", "ERROR", "CRITICAL"), ("CRITICAL", "CRITICAL", "CRITICAL")])
def test_repeated_failures_have_structured_observed_evidence(first, second, expected):
    findings = detect_fault_patterns([log(1, severity=first), log(2, 60, severity=second)])
    assert len(findings) == 1
    finding = findings[0]
    assert set(finding) == {"equipment_id", "pattern_type", "severity", "start_timestamp", "end_timestamp",
                             "log_ids", "event_codes", "explanation"}
    assert finding["pattern_type"] == "repeated_failures"
    assert finding["equipment_id"] == 1
    assert finding["severity"] == expected
    assert finding["log_ids"] == [1, 2]
    assert finding["event_codes"] == ["SYNTHETIC-FAULT"]
    assert finding["start_timestamp"] == "2026-10-25T12:00:00.000000+00:00"
    assert finding["end_timestamp"] == "2026-10-25T12:01:00.000000+00:00"
    assert "2 ERROR/CRITICAL events" in finding["explanation"]
    assert "not a confirmed root cause" in finding["explanation"]


@pytest.mark.parametrize("seconds,expected", [(0, 1), (300, 1), (300.000001, 0)])
def test_repeated_failure_window_boundary_is_inclusive(seconds, expected):
    assert len(detect_fault_patterns([log(1), log(2, seconds)])) == expected


def test_rolling_window_crosses_fixed_five_minute_buckets():
    assert len(detect_fault_patterns([log(1, 299), log(2, 301)])) == 1


def test_repeated_overlap_is_one_episode_without_claiming_one_five_minute_span():
    records = [log(1), log(2, 240), log(3, 480, severity="CRITICAL"), log(4, 781), log(5, 800)]
    findings = detect_fault_patterns(records)
    assert [finding["log_ids"] for finding in findings] == [[1, 2, 3], [4, 5]]
    assert findings[0]["end_timestamp"] == "2026-10-25T12:08:00.000000+00:00"
    assert findings[0]["severity"] == "CRITICAL"
    assert "Adjacent failures are at most 5 minutes apart" in findings[0]["explanation"]
    assert "one episode" in findings[0]["explanation"]


@pytest.mark.parametrize("second_code", ["DIFFERENT-FAULT", "synthetic-fault"])
def test_different_or_differently_cased_codes_are_not_repeated_failures(second_code):
    assert detect_fault_patterns([log(1), log(2, 20, event_code=second_code)]) == []


def test_missing_event_codes_are_grouped_as_unknown_without_inferred_causes():
    records = [log(1, event_code=""), log(2, 20, event_code="")]
    del records[0]["event_code"]
    finding = detect_fault_patterns(records)[0]
    assert finding["event_codes"] == [""]
    assert "unknown code" in finding["explanation"]
    assert "not a confirmed root cause" in finding["explanation"]


@pytest.mark.parametrize("severity", ["ERROR", "CRITICAL"])
@pytest.mark.parametrize("seconds,expected", [(0, 0), (0.000001, 1), (120, 1), (120.000001, 0)])
def test_warning_to_failure_requires_strict_order_and_inclusive_two_minutes(severity, seconds, expected):
    records = [log(1, severity="WARNING", event_code="SYNTHETIC-WARN"), log(2, seconds, severity=severity)]
    assert len(patterns(records, "warning_to_error")) == expected


def test_warning_to_error_does_not_require_the_same_event_code():
    finding = detect_fault_patterns([log(1, severity="WARNING", event_code="WARN-A"),
                                     log(2, 20, severity="CRITICAL", event_code="FAULT-B")])[0]
    assert finding["pattern_type"] == "warning_to_error"
    assert finding["severity"] == "CRITICAL"
    assert finding["log_ids"] == [1, 2]
    assert finding["event_codes"] == ["FAULT-B", "WARN-A"]
    assert "1 WARNING event(s) and 1 ERROR/CRITICAL event(s)" in finding["explanation"]
    assert "not a confirmed root cause" in finding["explanation"]


def test_error_before_warning_is_not_an_escalation_even_if_input_is_reversed():
    assert detect_fault_patterns([log(2, 10, severity="WARNING"), log(1)]) == []


def test_info_logs_do_not_break_matches_and_are_not_evidence():
    records = [log(1, severity="WARNING"), log(2, 10, severity="INFO"), log(3, 20), log(4, 30)]
    findings = detect_fault_patterns(records)
    assert {finding["pattern_type"] for finding in findings} == {"repeated_failures", "warning_to_error"}
    assert all(2 not in finding["log_ids"] for finding in findings)
    assert patterns(records, "warning_to_error")[0]["log_ids"] == [1, 3, 4]


def test_overlapping_warning_matches_are_merged_without_duplicate_evidence():
    records = [log(1, severity="WARNING"), log(2, 60, severity="WARNING"), log(3, 90), log(4, 150),
               log(5, 180, severity="WARNING"), log(6, 250)]
    findings = patterns(records, "warning_to_error")
    assert [finding["log_ids"] for finding in findings] == [[1, 2, 3, 4], [5, 6]]
    assert len({log_id for finding in findings for log_id in finding["log_ids"]}) == 6


def test_coalesced_warning_episode_can_span_longer_than_two_minutes():
    records = [log(1, severity="WARNING"), log(2, 50, severity="WARNING"), log(3, 60),
               log(4, 150, severity="WARNING"), log(5, 160), log(6, 260)]
    finding = patterns(records, "warning_to_error")[0]
    assert finding["log_ids"] == [1, 2, 3, 4, 5, 6]
    assert finding["end_timestamp"] == "2026-10-25T12:04:20.000000+00:00"
    # Every cited event must actually participate in at least one qualifying pair.
    by_id = {record["id"]: record for record in records}
    for log_id in finding["log_ids"]:
        current = by_id[log_id]
        current_time = datetime.fromisoformat(current["timestamp"])
        assert any(
            0 < (datetime.fromisoformat(other["timestamp"]) - current_time).total_seconds() <= 120
            if current["severity"] == "WARNING" else
            0 < (current_time - datetime.fromisoformat(other["timestamp"])).total_seconds() <= 120
            for other in records if (other["severity"] == "WARNING") != (current["severity"] == "WARNING")
        )


def test_touching_warning_windows_without_shared_events_remain_separate():
    records = [log(1, severity="WARNING"), log(2, 120), log(3, 120, severity="WARNING"), log(4, 240)]
    assert [finding["log_ids"] for finding in patterns(records, "warning_to_error")] == [[1, 2], [3, 4]]


@pytest.mark.parametrize("records", [[log(1), log(2, 20, equipment_id=2)],
                                    [log(1, severity="WARNING"), log(2, 20, equipment_id=2)]])
def test_different_equipment_is_never_correlated(records):
    assert detect_fault_patterns(records) == []


def test_multiple_equipment_findings_have_isolated_evidence():
    records = [log(1), log(2, 20, equipment_id=2, severity="WARNING"),
               log(3, 30), log(4, 40, equipment_id=2)]
    findings = detect_fault_patterns(records)
    assert [(finding["equipment_id"], finding["log_ids"]) for finding in findings] == [(1, [1, 3]), (2, [2, 4])]
    by_id = {record["id"]: record for record in records}
    assert all(all(by_id[log_id]["equipment_id"] == finding["equipment_id"] for log_id in finding["log_ids"])
               for finding in findings)


def test_utc_normalization_handles_different_offsets_and_midnight():
    records = [log(1, severity="WARNING", timestamp="2026-10-25T23:59:30-04:00"),
               log(2, timestamp="2026-10-26T09:30:30+05:30")]
    finding = detect_fault_patterns(records)[0]
    assert finding["start_timestamp"] == "2026-10-26T03:59:30.000000+00:00"
    assert finding["end_timestamp"] == "2026-10-26T04:00:30.000000+00:00"
    assert finding["pattern_type"] == "warning_to_error"


def test_out_of_order_input_is_deterministic_and_not_modified():
    records = [log(8, 100), log(3, 0, severity="WARNING"), log(9, 100),
               log(2, 5, equipment_id=2), log(1, 10, equipment_id=2)]
    before = deepcopy(records)
    expected = detect_fault_patterns(records)
    assert expected
    for ordering in permutations(records):
        assert detect_fault_patterns(list(ordering)) == expected
    assert records == before
    for finding in expected:
        ids = finding["log_ids"]
        if 8 in ids and 9 in ids:
            assert ids.index(8) < ids.index(9)


def test_duplicate_input_ids_do_not_count_as_repeated_events_or_duplicate_findings():
    first, second = log(1), log(2, 30)
    assert detect_fault_patterns([first, deepcopy(first)]) == []
    expected = detect_fault_patterns([first, second])
    assert detect_fault_patterns([first, second, deepcopy(first), deepcopy(second)]) == expected


def test_duplicate_ids_with_equivalent_timezone_representations_are_one_event():
    first = log(1, timestamp="2026-10-25T12:00:00Z")
    equivalent = log(1, timestamp="2026-10-25T08:00:00-04:00")
    assert detect_fault_patterns([first, equivalent]) == []


@pytest.mark.parametrize("field,value", [("equipment_id", 2), ("severity", "CRITICAL"),
                                        ("event_code", "OTHER"), ("timestamp", "2026-10-25T12:00:01Z")])
def test_conflicting_duplicate_ids_are_rejected(field, value):
    first, conflicting = log(1), log(1)
    conflicting[field] = value
    with pytest.raises(ValueError, match="Conflicting entries for log ID 1"):
        detect_fault_patterns([first, conflicting])


@pytest.mark.parametrize("value", [None, "", "not-a-date", "2026-10-25", "2026-10-25T12:00:00",
                                  "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_invalid_or_naive_timestamps_are_never_assumed_to_be_utc(value):
    with pytest.raises(ValueError, match="timestamp.*timezone"):
        detect_fault_patterns([log(1, timestamp=value)])


@pytest.mark.parametrize("field,value", [("id", True), ("id", 0), ("equipment_id", False), ("equipment_id", -1),
                                        ("severity", "DEBUG"), ("severity", []), ("event_code", None)])
def test_invalid_detection_fields_raise_clear_errors(field, value):
    record = log(1)
    record[field] = value
    with pytest.raises(ValueError, match=field):
        detect_fault_patterns([record])
