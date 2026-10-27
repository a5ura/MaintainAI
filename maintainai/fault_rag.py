"""Fault evidence adapter for the existing advisory RAG pipeline.

No storage, retrieval, model loading, or tool execution occurs here. Callers supply
current findings/log snapshots and ranked matches from the existing matching service.
Callers remain responsible for snapshot freshness. Cloud use requires explicit
authorization for THIS context, including its bounded log-message excerpts.
"""

from dataclasses import dataclass
import math

from maintainai.fault_matching import build_fault_query
from maintainai.llm import LLMProvider
from maintainai.rag import (
    ADVISORY_WARNING, CLAIM_FIELDS, MAX_FIELD_CHARS, build_messages, generate_grounded_analysis,
)
from maintainai.service import (
    LOG_SEVERITIES, ValidationError, choice, clean_text, timestamp,
)

MAX_CONTEXT_LOGS = 20
HISTORICAL_WARNING = (
    "Historical root causes and actions describe earlier tickets, not confirmed current causes. "
    "Open or unresolved historical outcomes are unverified; a recorded resolution is not proof "
    "of applicability. Cosine similarity is not confidence. AI output, when present, consists "
    "of unverified hypotheses requiring qualified human review."
)


@dataclass(frozen=True)
class FaultTroubleshootingResult:
    observed_evidence: dict
    historical_findings: list[dict]
    analysis: dict
    ai_generated: bool
    notice: str | None
    advisory_warning: str = ADVISORY_WARNING
    historical_warning: str = HISTORICAL_WARNING


def _positive_id(value, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValidationError(f"{field} must be a positive integer.")
    return value


def _event_time(value, field: str) -> str:
    # Never let the service timestamp helper default a missing observation to now.
    return timestamp(clean_text(value, field, required=True), field=field)


def _historical_records(matches: list[dict] | None) -> list[dict]:
    records, seen = [], set()
    for match in matches if matches is not None else []:
        if not isinstance(match, dict):
            raise ValidationError("Historical matches must be dictionaries.")
        issue_id = _positive_id(match.get("issue_id", match.get("id")), "Historical issue ID")
        if "id" in match and _positive_id(match["id"], "Historical id") != issue_id:
            raise ValidationError("Historical id and issue_id must identify the same issue.")
        if issue_id in seen:
            continue
        score = match.get("similarity")
        if (type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1):
            raise ValidationError("Historical similarity must be a finite cosine score from -1 to 1.")
        status = clean_text(match.get("status"), "Historical status", limit=200)
        record = {
            "id": issue_id, "issue_id": issue_id,
            "equipment_id": _positive_id(match.get("equipment_id"), "Historical equipment_id"),
            "asset_tag": clean_text(match.get("asset_tag"), "Historical asset_tag", limit=200),
            "equipment_name": clean_text(match.get("equipment_name"), "Historical equipment_name", limit=200),
            "status": status, "similarity": score,
            "outcome_verification": "recorded resolution only" if status in ("Resolved", "Closed") else "unverified outcome",
        }
        for field in ("description", "troubleshooting_notes", "root_cause", "corrective_action"):
            record[field] = clean_text(match.get(field), f"Historical {field}")
        records.append(record)
        seen.add(issue_id)
        if len(records) == 5:
            break
    return records


def prepare_historical_records(matches: list[dict] | None) -> list[dict]:
    """Normalize up to five ranked historical records without retrieval or inference."""
    return _historical_records(matches)


def prepare_log_record(log: dict, equipment_id: int) -> dict:
    """Allowlist and bound one supplied event, never inventing an event timestamp."""
    message = clean_text(log.get("message"), "Log message", required=True)
    return {
        "id": log["id"], "equipment_id": equipment_id,
        "timestamp": _event_time(log.get("timestamp"), "Log timestamp"),
        "severity": choice(log.get("severity"), "Log severity", LOG_SEVERITIES),
        "event_code": clean_text(log.get("event_code"), "Log event_code", limit=200),
        "message": message[:MAX_FIELD_CHARS],
        "truncated_fields": ["message"] if len(message) > MAX_FIELD_CHARS else [],
    }


def build_fault_context(finding: dict, logs: list[dict] | None,
                        matches: list[dict] | None) -> dict:
    """Prepare allowlisted, bounded evidence without persisting it or using inference.

Only cited logs belonging to the fault's equipment are included. Log messages use
the existing RAG field limit; at most 20 logs and five distinct ranked issues are
included. Missing and omitted IDs are explicit and cannot be used as AI citations.
Reported finding metadata is separated from the underlying observed log records.
"""
    # Reuse the matching API's evidence-selection validation, including conflicts.
    build_fault_query(finding, logs)
    wanted = list(dict.fromkeys(finding.get("log_ids", [])))
    by_id = {log["id"]: log for log in (logs or [])
             if type(log.get("id")) is int and log["id"] in wanted
             and type(log.get("equipment_id")) is int and log["equipment_id"] == finding["equipment_id"]}
    available = [log_id for log_id in wanted if log_id in by_id]
    selected = available[:MAX_CONTEXT_LOGS]
    observations = []
    for log_id in selected:
        log = by_id[log_id]
        observations.append(prepare_log_record(log, finding["equipment_id"]))
    first = by_id[selected[0]] if selected else {}
    start = _event_time(finding.get("start_timestamp"), "Fault start_timestamp")
    end = _event_time(finding.get("end_timestamp"), "Fault end_timestamp")
    if start > end:
        raise ValidationError("Fault end_timestamp must not precede start_timestamp.")
    observed = {
        "equipment": {
            "id": finding["equipment_id"],
            "name": clean_text(first.get("equipment_name"), "Equipment name", limit=200),
            "asset_tag": clean_text(first.get("asset_tag"), "Equipment asset_tag", limit=200),
        },
        "reported_pattern": {
            "pattern_type": clean_text(finding["pattern_type"], "Fault pattern_type", required=True, limit=200),
            "severity": choice(finding.get("severity"), "Fault severity", LOG_SEVERITIES),
            "start_timestamp": start, "end_timestamp": end,
        },
        "logs": observations,
        "event_codes": list(dict.fromkeys(log["event_code"] for log in observations)),
        "missing_log_ids": [log_id for log_id in wanted if log_id not in by_id],
        "omitted_log_ids": available[MAX_CONTEXT_LOGS:],
    }
    return {"observed_evidence": observed, "historical_findings": _historical_records(matches)}


def build_fault_messages(context: dict) -> list[dict[str, str]]:
    """Build the exact existing RAG messages for inference or a local consent preview."""
    observed, records = context["observed_evidence"], context["historical_findings"]
    question = "Review this observed fault pattern using the supplied logs and historical tickets. Provide cautious inspection-first hypotheses only; do not diagnose a confirmed current cause."
    return build_messages(question, records, observed["equipment"], fault_context=observed)


def troubleshoot_fault(finding: dict, logs: list[dict] | None, matches: list[dict] | None,
                       *, use_cloud: bool = False, provider: LLMProvider | None = None) -> FaultTroubleshootingResult:
    """Explain supplied evidence through the existing RAG pipeline, local-only by default.

Only use_cloud=True authorizes sending this bounded context to the existing
configured provider (or an explicitly injected provider). A future UI must obtain
consent covering log messages AND historical excerpts for each request. Supplying
a provider alone never authorizes a request. There is no inference with no logs or
no history. Invalid input raises ValidationError before provider initialization;
provider/grounding failures retain local evidence with no generated hypotheses.
There are no new providers, credentials, environment settings, or persistence.
"""
    if type(use_cloud) is not bool:
        raise ValidationError("use_cloud must be a boolean explicit authorization for this fault context.")
    context = build_fault_context(finding, logs, matches)
    observed, records = context["observed_evidence"], context["historical_findings"]
    uncertainty = ["Current equipment conditions and causes have not been verified. No AI hypotheses were generated in local-only mode."]
    if observed["missing_log_ids"] or observed["omitted_log_ids"]:
        uncertainty.append("Supporting log evidence is incomplete; missing or omitted events cannot support conclusions.")
    if any(log["truncated_fields"] for log in observed["logs"]):
        uncertainty.append("Log messages were truncated; review the original local evidence before considering any action.")
    if any(record["outcome_verification"] == "unverified outcome" for record in records):
        uncertainty.append("Open or unresolved historical tickets have unverified outcomes, regardless of recorded root causes or actions.")
    fallback = {
        "issue_summary": "Observed temporal pattern: " + observed["reported_pattern"]["pattern_type"] + ". This is not a confirmed root cause.",
        **{field: [] for field in CLAIM_FIELDS},
        "evidence_references": [record["id"] for record in records],
        "log_evidence_references": [log["id"] for log in observed["logs"]],
        "uncertainty_or_missing_information": uncertainty,
    }
    if not observed["logs"] or not records:
        return FaultTroubleshootingResult(observed, records, fallback, False,
            "Insufficient observed logs or historical matches. Showing local evidence only; no provider request was made.")
    if not use_cloud:
        return FaultTroubleshootingResult(observed, records, fallback, False,
            "Local-only mode. No cloud transmission or generated hypotheses.")
    messages = build_fault_messages(context)
    analysis, generated, notice = generate_grounded_analysis(
        messages, records, fallback, use_cloud=use_cloud, provider=provider, logs=observed["logs"])
    if generated:
        # Keep known evidence limitations visible even if a provider omits them.
        analysis["uncertainty_or_missing_information"] = list(dict.fromkeys(
            analysis["uncertainty_or_missing_information"] + uncertainty[1:]))
    return FaultTroubleshootingResult(observed, records, analysis, generated, notice)
