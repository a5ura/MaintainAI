"""Bounded description-aware queries for the existing local semantic search."""

from maintainai.fault_matching import build_fault_query
from maintainai.service import ValidationError, clean_text

MAX_QUERY_CHARS = 8000
_MAX_LOG_CONTEXT_CHARS = 3500


def build_guided_query(description: str, *, equipment_id: int | None = None,
                       analysis: dict | None = None) -> str:
    """Keep the technician report separate from recorded logs and real findings.

    The caller supplies a current guided analysis snapshot, if available. No logs,
    detection, embeddings, or database operations are performed here.
    """
    description = clean_text(description, "Problem description", required=True, limit=4000)
    if equipment_id is not None and (type(equipment_id) is not int or equipment_id <= 0):
        raise ValidationError("Select a valid registered equipment ID.")
    query = "Technician-reported problem:\n" + description
    if analysis is None:
        return query
    if equipment_id is None or analysis["draft"] != {"description": description, "equipment_id": equipment_id}:
        raise ValidationError("The log-analysis snapshot does not match the submitted problem.")
    logs = [log for log in analysis["logs"]
            if type(log.get("equipment_id")) is int and log["equipment_id"] == equipment_id]
    findings = [finding for finding in analysis["findings"]
                if type(finding.get("equipment_id")) is int and finding["equipment_id"] == equipment_id]
    evidence = []
    if findings:
        for finding in findings[:3]:
            context = build_fault_query(finding, logs)
            if context and context not in evidence:
                evidence.append(context)
    else:
        # Logs can inform retrieval even when neither supported temporal rule fires.
        # These are observations, never a fabricated finding or a technician report.
        for log in logs[:6]:
            message = " ".join(clean_text(log.get("message"), "Log message").split())[:240]
            code = clean_text(log.get("event_code"), "Log event code", limit=200)
            if message or code:
                evidence.append(f"Log #{log['id']} | {code} | {message}")
    if evidence:
        context = "\n\n".join(evidence)[:_MAX_LOG_CONTEXT_CHARS]
        query += "\n\nRecorded log evidence (bounded excerpts):\n" + context
    return query[:MAX_QUERY_CHARS]
