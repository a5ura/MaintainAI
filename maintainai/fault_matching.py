"""Local historical retrieval for observed faults; no diagnosis or generation.

Pass a finding from detect_fault_patterns(), logs from MaintenanceStore.list_logs(),
and an existing SemanticSearch instance. Only cited logs for the finding's equipment
contribute to the query. Searches cover all equipment, allowing relevant historical
cases from other assets. The existing search service owns indexing, stale-index
checks, embedding, cosine scoring, and ordering; this module never rebuilds an index.
"""

from dataclasses import dataclass

from maintainai.semantic_search import SemanticSearch
from maintainai.service import ValidationError, clean_text

HISTORICAL_NOTICE = (
    "Matches contain recorded historical findings, not troubleshooting hypotheses. "
    "Similarity does not establish the current root cause. Historical corrective "
    "actions are not recommendations for the current equipment. No hypotheses are generated."
)
_MAX_OBSERVATIONS = 6
_MAX_MESSAGE_CHARS = 240


@dataclass(frozen=True)
class FaultMatchResult:
    query: str
    matches: list[dict]
    evidence_log_ids: list[int]
    missing_log_ids: list[int]
    notice: str


def _supporting_logs(finding: dict, logs: list[dict] | None) -> list[dict]:
    if not isinstance(finding, dict):
        raise ValidationError("Fault finding must be a dictionary.")
    equipment_id = finding.get("equipment_id")
    if type(equipment_id) is not int or equipment_id <= 0:
        raise ValidationError("Fault equipment_id must be a positive integer.")
    clean_text(finding.get("pattern_type"), "Fault pattern_type", required=True, limit=200)
    log_ids = finding.get("log_ids", [])
    if not isinstance(log_ids, list) or any(type(item) is not int or item <= 0 for item in log_ids):
        raise ValidationError("Fault log_ids must be a list of positive integers.")
    wanted = set(log_ids)
    by_id = {}
    for log in logs if logs is not None else []:
        if not isinstance(log, dict):
            raise ValidationError("Each supporting log must be a dictionary.")
        log_id = log.get("id")
        if (type(log_id) is not int or log_id not in wanted
                or type(log.get("equipment_id")) is not int or log["equipment_id"] != equipment_id):
            continue
        if log_id in by_id and by_id[log_id] != log:
            raise ValidationError(f"Conflicting supporting records for log ID {log_id}.")
        by_id[log_id] = log
    # Finding IDs already follow event order. Ignore caller ordering and duplicate IDs.
    return [by_id[log_id] for log_id in dict.fromkeys(log_ids) if log_id in by_id]


def _compact(value, field: str, *, limit: int) -> str:
    return " ".join(clean_text(value, field, limit=limit).split())


def _query(finding: dict, logs: list[dict]) -> str:
    observations = []
    for log in logs:
        code = _compact(log.get("event_code"), "event_code", limit=200)
        message = _compact(log.get("message"), "message", limit=10000)
        if not code and not message:
            continue
        if len(message) > _MAX_MESSAGE_CHARS:
            message = message[:_MAX_MESSAGE_CHARS] + "..."
        observation = ": ".join(part for part in (code, message) if part)
        if observation not in observations:
            observations.append(observation)
    if not observations:
        return ""
    names, tags = [], []
    for log in logs:
        name = _compact(log.get("equipment_name"), "equipment_name", limit=200)
        tag = _compact(log.get("asset_tag"), "asset_tag", limit=200)
        if name and name not in names:
            names.append(name)
        if tag and tag not in tags:
            tags.append(tag)
    equipment = "; ".join([f"ID {finding['equipment_id']}", *names[:1], *tags[:1]])
    pattern = _compact(finding["pattern_type"], "pattern_type", limit=200).replace("_", " ")
    return (f"Pattern: {pattern}\nEquipment: {equipment}\nObserved logs:\n"
            + "\n".join(f"- {item}" for item in observations[:_MAX_OBSERVATIONS]))


def build_fault_query(finding: dict, logs: list[dict] | None) -> str:
    """Build a bounded query from observed evidence, never from causal explanations.

Return empty text if no usable cited logs exist. To keep input concise, include up
to six distinct observations with each message clipped to 240 characters. Equipment
metadata is optional; the equipment ID is always included. Inputs are not modified.
"""
    return _query(finding, _supporting_logs(finding, logs))


def match_fault_to_issues(
    finding: dict, logs: list[dict] | None, search: SemanticSearch,
) -> FaultMatchResult:
    """Retrieve up to five historical tickets using the supplied local search service.

Scores and rank order are passed through unchanged, including ties and negative
cosine scores. Every match retains the search service's original fields (id,
equipment_id, asset_tag, equipment_name, similarity, description, root_cause,
corrective_action, status, etc.), plus issue_id and an explicit evidence_type.
Missing logs are reported; partial evidence uses only available cited records.
Empty evidence skips search. StaleIndexError and EmbeddingError propagate for the
caller to handle using existing conventions. No LLM provider is invoked, and no
ticket, log, or index is written. The existing model's first use may download weights;
log contents are processed locally, never sent to cloud inference APIs.
"""
    selected = _supporting_logs(finding, logs)
    evidence_ids = [log["id"] for log in selected]
    present = set(evidence_ids)
    missing_ids = [log_id for log_id in dict.fromkeys(finding.get("log_ids", [])) if log_id not in present]
    query = _query(finding, selected)
    notice = HISTORICAL_NOTICE
    if missing_ids:
        notice += " Supporting logs are missing; any matches use only the available evidence."
    if not query:
        return FaultMatchResult("", [], evidence_ids, missing_ids,
                                notice + " No usable supporting log text; search was not run.")
    records = search.search(query, limit=5)
    matches = [{**record, "issue_id": record["id"], "evidence_type": "recorded_historical_issue"}
               for record in records]
    if not matches:
        notice += " No historical issues were retrieved."
    return FaultMatchResult(query, matches, evidence_ids, missing_ids, notice)
