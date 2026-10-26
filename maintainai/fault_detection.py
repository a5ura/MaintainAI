"""Deterministic temporal patterns over log dictionaries; no database or UI access.

Limits are inclusive: repeated failures are at most five minutes apart, and a
warning must strictly precede a failure by at most two minutes. Matches sharing
logs are coalesced per rule, so an episode can span longer than a single window.
Different pattern types can legitimately cite the same log. Blank event codes
match other blank codes, indicating unknown codes rather than a known fault.

Identical detection fields for a repeated log ID count once. Conflicting IDs,
invalid detection fields, and naive/unparseable timestamps raise ValueError.
Only detection fields are inspected; caller dictionaries are never modified.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_FAILURES = frozenset(("ERROR", "CRITICAL"))
_SEVERITIES = frozenset(("INFO", "WARNING", "ERROR", "CRITICAL"))
_REPEAT_WINDOW = timedelta(minutes=5)
_WARNING_WINDOW = timedelta(minutes=2)


@dataclass(frozen=True)
class _Event:
    id: int
    equipment_id: int
    timestamp: datetime
    severity: str
    event_code: str


def _events(logs: list[dict]) -> list[_Event]:
    by_id = {}
    for record in logs:
        if not isinstance(record, dict):
            raise ValueError("Each log must be a dictionary.")
        for field in ("id", "equipment_id"):
            if type(record.get(field)) is not int or record[field] <= 0:
                raise ValueError(f"Log {field} must be a positive integer.")
        severity = record.get("severity")
        if not isinstance(severity, str) or severity not in _SEVERITIES:
            raise ValueError(f"Log {record['id']}: severity must be INFO, WARNING, ERROR, or CRITICAL.")
        code = record.get("event_code", "")
        if not isinstance(code, str):
            raise ValueError(f"Log {record['id']}: event_code must be text.")
        try:
            value = record["timestamp"]
            if not isinstance(value, str):
                raise ValueError("Timestamp must be text")
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("Timestamp requires a timezone")
            parsed = parsed.astimezone(timezone.utc)
        except (KeyError, ValueError, TypeError, OverflowError):
            raise ValueError(f"Log {record['id']}: timestamp must be valid ISO 8601 text with a timezone and a representable UTC date.") from None
        event = _Event(record["id"], record["equipment_id"], parsed, severity, code)
        if event.id in by_id and by_id[event.id] != event:
            raise ValueError(f"Conflicting entries for log ID {event.id}.")
        by_id[event.id] = event
    return sorted(by_id.values(), key=lambda event: (event.timestamp, event.id))


def _finding(pattern_type: str, events: list[_Event]) -> dict:
    events = sorted(events, key=lambda event: (event.timestamp, event.id))
    if pattern_type == "repeated_failures":
        code = repr(events[0].event_code) if events[0].event_code else "not recorded (unknown code)"
        explanation = (
            f"Observed {len(events)} ERROR/CRITICAL events for equipment {events[0].equipment_id} "
            f"with the same recorded event code: {code}. Adjacent failures are at most 5 minutes "
            "apart; overlapping matches sharing logs are reported as one episode. "
            "This is a potential temporal pattern, not a confirmed root cause."
        )
    else:
        warnings = sum(event.severity == "WARNING" for event in events)
        failures = len(events) - warnings
        explanation = (
            f"Observed {warnings} WARNING event(s) and {failures} ERROR/CRITICAL event(s) for "
            f"equipment {events[0].equipment_id}. Every included log participates in a warning "
            "followed by a failure with a positive delay of at most 2 minutes. Matches sharing "
            "logs are reported as one episode. This temporal association is a potential "
            "pattern, not a confirmed root cause."
        )
    return {
        "equipment_id": events[0].equipment_id,
        "pattern_type": pattern_type,
        "severity": "CRITICAL" if any(event.severity == "CRITICAL" for event in events) else "ERROR",
        "start_timestamp": events[0].timestamp.isoformat(timespec="microseconds"),
        "end_timestamp": events[-1].timestamp.isoformat(timespec="microseconds"),
        "log_ids": [event.id for event in events],
        "event_codes": sorted({event.event_code for event in events}),
        "explanation": explanation,
    }


def _repeated_failures(events: list[_Event]) -> list[dict]:
    findings, episode = [], []
    for event in events:
        if episode and event.timestamp - episode[-1].timestamp > _REPEAT_WINDOW:
            if len(episode) >= 2:
                findings.append(_finding("repeated_failures", episode))
            episode = []
        episode.append(event)
    if len(episode) >= 2:
        findings.append(_finding("repeated_failures", episode))
    return findings


def _warning_patterns(events: list[_Event]) -> list[dict]:
    warnings = [event for event in events if event.severity == "WARNING"]
    failures = [event for event in events if event.severity in _FAILURES]
    findings, matched_failures = [], []
    low = high = start = end = 0
    for failure in failures:
        # Eligible warnings form a contiguous slice: inclusive at two minutes,
        # strictly older than the failure. Both pointers only move forwards.
        while low < len(warnings) and failure.timestamp - warnings[low].timestamp > _WARNING_WINDOW:
            low += 1
        while high < len(warnings) and warnings[high].timestamp < failure.timestamp:
            high += 1
        if low >= high:
            continue
        if matched_failures and low >= end:
            findings.append(_finding("warning_to_error", warnings[start:end] + matched_failures))
            matched_failures = []
        if not matched_failures:
            start = low
        end = high
        matched_failures.append(failure)
    if matched_failures:
        findings.append(_finding("warning_to_error", warnings[start:end] + matched_failures))
    return findings


def detect_fault_patterns(logs: list[dict]) -> list[dict]:
    """Return UTC-normalized potential patterns, ordered by start time/equipment/type.

    Output fields: equipment_id, pattern_type (repeated_failures or
    warning_to_error), severity (highest observed), start_timestamp,
    end_timestamp, chronological log_ids, event_codes, and explanation.
    Rolling matches are merged only when they share evidence for the same rule.
    No root causes or code-specific causal relationships are inferred.
    """
    by_equipment, by_code = defaultdict(list), defaultdict(list)
    for event in _events(logs):
        by_equipment[event.equipment_id].append(event)
        if event.severity in _FAILURES:
            by_code[(event.equipment_id, event.event_code)].append(event)
    findings = []
    for events in by_code.values():
        findings.extend(_repeated_failures(events))
    for events in by_equipment.values():
        findings.extend(_warning_patterns(events))
    return sorted(findings, key=lambda finding: (
        finding["start_timestamp"], finding["equipment_id"], finding["pattern_type"],
        finding["end_timestamp"], tuple(finding["log_ids"]),
    ))
