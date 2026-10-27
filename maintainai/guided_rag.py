"""Pure guided evidence preparation for the existing advisory RAG pipeline.

Callers supply registry-resolved equipment, an explicitly analyzed log snapshot,
and explicitly retrieved matches. Registry membership, database/index identity,
and snapshot freshness must be checked by the caller; this module performs no I/O.
Prepare messages once for preview and later pass those same messages to
generate_grounded_analysis(), with historical_findings as records and
guided_validation_logs(context) as logs. This module never invokes generation.
"""

import json

from maintainai.fault_rag import MAX_CONTEXT_LOGS, prepare_historical_records, prepare_log_record
from maintainai.rag import CONTEXT_FIELDS, MAX_FIELD_CHARS, build_messages
from maintainai.service import LOG_SEVERITIES, ValidationError, choice, clean_text, timestamp

MAX_CONTEXT_FINDINGS = 10
MAX_REPORTED_IDS = 20
MAX_SERIALIZED_BYTES = 64 * 1024
_MAX_ID = 2**63 - 1


def _id(value, field: str) -> int:
    if type(value) is not int or not 0 < value <= _MAX_ID:
        raise ValidationError(f"{field} must be a positive SQLite integer ID.")
    return value


def _records(value, field: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValidationError(f"{field} must be a list of records.")
    return value


def _time(value, field: str) -> str:
    return timestamp(clean_text(value, field, required=True), field=field)


def _size(value) -> int:
    # Include JSON escaping, including non-ASCII characters and nested messages.
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8"))


def _messages(context: dict) -> list[dict[str, str]]:
    evidence = {**context["observed_evidence"], "limitations": context["limitations"]}
    return build_messages(context["question"], context["historical_findings"], context["selected_equipment"],
                          guided_context=evidence)


def guided_validation_logs(context: dict) -> list[dict] | None:
    """Select the shared validator's mode; an empty list must never activate log mode."""
    logs = context["observed_evidence"]["logs"]
    expected = "log_aware" if logs else "issue_only"
    if context["citation_mode"] != expected:
        raise ValidationError("Guided citation mode does not match the supplied log evidence.")
    return logs or None


def build_guided_messages(context: dict) -> list[dict[str, str]]:
    """Return exact preview/send messages without retrieval, inference, or mutation."""
    guided_validation_logs(context)
    messages = _messages(context)
    if _size(context) > MAX_SERIALIZED_BYTES or _size(messages) > MAX_SERIALIZED_BYTES:
        raise ValidationError("Prepared guided evidence exceeds the serialized size limit.")
    return messages


def _remove_log(context: dict) -> None:
    evidence, limits = context["observed_evidence"], context["limitations"]
    removed = evidence["logs"].pop()["id"]
    limits["omitted_log_count"] += 1
    if len(limits["omitted_log_ids"]) < MAX_REPORTED_IDS:
        limits["omitted_log_ids"].append(removed)
    retained = []
    for pattern in evidence["detected_patterns"]:
        if removed in pattern["log_ids"]:
            pattern["log_ids"].remove(removed)
            if len(pattern["unavailable_log_ids"]) < MAX_REPORTED_IDS:
                pattern["unavailable_log_ids"].append(removed)
            pattern["unavailable_log_count"] += 1
        if pattern["log_ids"]:
            retained.append(pattern)
        else:
            limits["omitted_finding_count"] += 1
    evidence["detected_patterns"] = retained
    context["citation_mode"] = "log_aware" if evidence["logs"] else "issue_only"


def _fit(context: dict) -> None:
    """Reduce excerpts, then records, until both context and messages fit the cap."""
    evidence, limits = context["observed_evidence"], context["limitations"]
    while _size(context) > MAX_SERIALIZED_BYTES or _size(_messages(context)) > MAX_SERIALIZED_BYTES:
        fields = [(record, field) for record in context["historical_findings"] for field in CONTEXT_FIELDS]
        fields += [(log, "message") for log in evidence["logs"]]
        record, field = max(fields, key=lambda item: len(item[0][item[1]]), default=({}, ""))
        if field and len(record[field]) > 75:
            record[field] = record[field][:max(75, len(record[field]) // 2)]
            if field not in record["truncated_fields"]:
                record["truncated_fields"].append(field)
        elif len(evidence["logs"]) > 1:
            _remove_log(context)
        elif len(context["historical_findings"]) > 1:
            context["historical_findings"].pop()
            limits["omitted_history_count"] += 1
        elif evidence["detected_patterns"]:
            evidence["detected_patterns"].pop()
            limits["omitted_finding_count"] += 1
        elif context["historical_findings"]:
            context["historical_findings"].pop()
            limits["omitted_history_count"] += 1
        else:
            raise ValidationError("The reported problem and required evidence exceed the serialized size limit.")
        limits["total_size_reduced"] = True


def build_guided_context(draft: dict, *, equipment: dict | None = None,
                         analysis: dict | None = None, matches: list[dict] | None = None) -> dict:
    """Allowlist caller-validated evidence, preserving its source and actual IDs.

    No analysis argument means no log evidence. A supplied snapshot must match
    the cleaned draft and selected registry record. Historical matches retain
    retrieval order and scores; they are never manufactured or searched here.
    Omission counts include records excluded by bounds or duplicate elimination.
    """
    if not isinstance(draft, dict):
        raise ValidationError("Supply the submitted guided draft.")
    question = clean_text(draft.get("description"), "Problem description", required=True, limit=4000)
    equipment_id = draft.get("equipment_id")
    selected = None
    if equipment_id is not None:
        _id(equipment_id, "Selected equipment ID")
        if not isinstance(equipment, dict) or _id(equipment.get("id"), "Registered equipment ID") != equipment_id:
            raise ValidationError("Supply matching registry-resolved equipment metadata.")
        selected = {"id": equipment_id,
                    "name": clean_text(equipment.get("name"), "Equipment name", required=True, limit=200),
                    "asset_tag": clean_text(equipment.get("asset_tag"), "Equipment asset tag", required=True, limit=200)}
    elif equipment is not None:
        raise ValidationError("Equipment metadata requires a submitted equipment selection.")

    history = _records(matches, "Historical matches")
    historical = prepare_historical_records(history)
    for record in historical:
        _id(record["id"], "Historical issue ID")
        _id(record["equipment_id"], "Historical equipment ID")
        if equipment_id is not None and record["equipment_id"] != equipment_id:
            raise ValidationError("Historical matches must belong to the submitted equipment scope.")
        record["evidence_type"] = "recorded_historical_issue"
        record["truncated_fields"] = [field for field in CONTEXT_FIELDS if len(record[field]) > MAX_FIELD_CHARS]
        for field in CONTEXT_FIELDS:
            record[field] = record[field][:MAX_FIELD_CHARS]

    raw_logs, raw_findings = [], []
    if analysis is not None:
        if not isinstance(analysis, dict) or selected is None or analysis.get("draft") != {
            "description": question, "equipment_id": equipment_id,
        }:
            raise ValidationError("The explicit log-analysis snapshot must match the submitted problem and equipment.")
        if "equipment_label" in analysis and analysis["equipment_label"] != f"{selected['asset_tag']} — {selected['name']}":
            raise ValidationError("The log-analysis equipment metadata changed.")
        if not isinstance(analysis.get("logs"), list) or not isinstance(analysis.get("findings"), list):
            raise ValidationError("An explicit analysis must contain log and finding lists.")
        raw_logs = _records(analysis.get("logs"), "Snapshot logs")
        raw_findings = _records(analysis.get("findings"), "Snapshot findings")
    logs, originals, excluded = {}, {}, 0
    for log in raw_logs:
        log_id = _id(log.get("id"), "Log ID")
        if _id(log.get("equipment_id"), "Log equipment ID") != equipment_id:
            excluded += 1
            continue
        if log_id in originals and log != originals[log_id]:
            raise ValidationError("Conflicting records identify the same supporting log.")
        originals[log_id] = log
        logs[log_id] = {**prepare_log_record(log, equipment_id), "evidence_type": "recorded_equipment_log"}

    patterns = []
    for finding in raw_findings:
        if _id(finding.get("equipment_id"), "Finding equipment ID") != equipment_id:
            continue
        ids = finding.get("log_ids")
        if not isinstance(ids, list):
            raise ValidationError("Finding log_ids must be a list of supporting IDs.")
        ids = list(dict.fromkeys(_id(value, "Supporting log ID") for value in ids))
        start = _time(finding.get("start_timestamp"), "Finding start timestamp")
        end = _time(finding.get("end_timestamp"), "Finding end timestamp")
        if start > end:
            raise ValidationError("Finding end timestamp must not precede its start.")
        patterns.append({"equipment_id": equipment_id,
                         "pattern_type": choice(finding.get("pattern_type"), "Pattern type", ("repeated_failures", "warning_to_error")),
                         "severity": choice(finding.get("severity"), "Finding severity", LOG_SEVERITIES),
                         "start_timestamp": start, "end_timestamp": end, "log_ids": ids,
                         "evidence_type": "detected_temporal_pattern"})
    selected_patterns = patterns[:MAX_CONTEXT_FINDINGS]
    wanted = list(dict.fromkeys(value for finding in selected_patterns for value in finding["log_ids"])) if patterns else list(logs)
    selected_ids = [value for value in wanted if value in logs][:MAX_CONTEXT_LOGS]
    observed_patterns = []
    for pattern in selected_patterns:
        included = [value for value in pattern["log_ids"] if value in selected_ids]
        if not included:
            continue
        unavailable = [value for value in pattern["log_ids"] if value not in selected_ids]
        observed_patterns.append({**pattern, "log_ids": included, "source_log_count": len(pattern["log_ids"]),
                                  "unavailable_log_ids": unavailable[:MAX_REPORTED_IDS], "unavailable_log_count": len(unavailable)})
    omitted_ids = [value for value in logs if value not in selected_ids]
    context = {
        "question": question, "selected_equipment": selected,
        "observed_evidence": {"logs": [logs[value] for value in selected_ids], "detected_patterns": observed_patterns},
        "historical_findings": historical, "citation_mode": "log_aware" if selected_ids else "issue_only",
        "limitations": {"omitted_log_count": len(omitted_ids), "omitted_log_ids": omitted_ids[:MAX_REPORTED_IDS],
                        "excluded_log_count": excluded, "duplicate_log_count": len(raw_logs) - excluded - len(logs),
                        "omitted_finding_count": len(raw_findings) - len(observed_patterns),
                        "omitted_history_count": len(history) - len(historical), "total_size_reduced": False},
    }
    _fit(context)
    return context
