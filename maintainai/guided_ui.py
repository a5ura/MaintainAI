"""Guided drafts, explicit stored-log analysis, and local historical retrieval."""

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import streamlit as st

from maintainai.database import default_db_path
from maintainai.embeddings import EmbeddingError, MiniLMEmbedder
from maintainai.fault_detection import detect_fault_patterns
from maintainai.guided_matching import build_guided_query
from maintainai.guided_rag import build_guided_context, build_guided_messages, guided_validation_logs
from maintainai.rag import ADVISORY_WARNING, CLAIM_FIELDS, generate_grounded_analysis
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError, clean_text

_DRAFT_KEY = "guided_troubleshooting_draft"
_EQUIPMENT_KEY = "guided_troubleshooting_equipment"
_DESCRIPTION_KEY = "guided_troubleshooting_description"
_ANALYSIS_KEY = "guided_troubleshooting_log_analysis"
_MATCHES_KEY = "guided_troubleshooting_matches"
_ANALYSIS_REVISION_KEY = "guided_troubleshooting_analysis_revision"
_RETRIEVAL_REVISION_KEY = "guided_troubleshooting_retrieval_revision"
_MATCH_AUDIT_KEY = "guided_troubleshooting_match_audit"
_AI_KEY = "guided_troubleshooting_ai"
_AI_NOTICE_KEY = "guided_troubleshooting_ai_notice"


def guided_troubleshooting_page(store: MaintenanceStore) -> None:
    st.subheader("Troubleshoot a Problem")
    equipment = store.list_equipment()
    equipment_records = {item["id"]: item for item in equipment}
    labels = {item["id"]: f"{item['asset_tag']} — {item['name']}" for item in equipment}
    draft = st.session_state.get(_DRAFT_KEY, {})
    # Restore submitted inputs when Streamlit removes widgets during navigation.
    if _EQUIPMENT_KEY not in st.session_state:
        st.session_state[_EQUIPMENT_KEY] = draft.get("equipment_id")
    if st.session_state[_EQUIPMENT_KEY] not in [None, *labels]:
        st.session_state[_EQUIPMENT_KEY] = None
    if _DESCRIPTION_KEY not in st.session_state:
        st.session_state[_DESCRIPTION_KEY] = draft.get("description", "")

    st.info("Stored-log analysis runs only when you click Analyze Stored Logs after submitting a problem.")
    with st.form("guided_troubleshooting_form"):
        equipment_id = st.selectbox(
            "Equipment (optional)", [None, *labels], key=_EQUIPMENT_KEY,
            format_func=lambda value: "No equipment selected" if value is None else labels[value],
        )
        description = st.text_area("Problem description *", max_chars=4000, key=_DESCRIPTION_KEY)
        submitted = st.form_submit_button("Analyze Problem", type="primary")
    if submitted:
        try:
            description = clean_text(description, "Problem description", required=True, limit=4000)
            if equipment_id is not None and (type(equipment_id) is not int or equipment_id not in labels):
                raise ValidationError("Select a valid registered equipment ID.")
        except ValidationError as exc:
            st.error(str(exc))
        else:
            new_draft = {"description": description, "equipment_id": equipment_id}
            if new_draft != st.session_state.get(_DRAFT_KEY):
                st.session_state.pop(_ANALYSIS_KEY, None)
                st.session_state.pop(_MATCHES_KEY, None)
            st.session_state[_DRAFT_KEY] = new_draft

    draft = st.session_state.get(_DRAFT_KEY)
    if draft is not None:
        st.success("Problem draft saved for this session.")
        st.subheader("Last submitted problem")
        equipment_id = draft["equipment_id"]
        equipment_label = labels.get(equipment_id) if type(equipment_id) is int else None
        st.text("Equipment: " + ("No equipment selected" if equipment_id is None else
                equipment_label or f"Equipment ID {equipment_id} (no longer registered)"))
        st.text(draft["description"])
        _stored_log_analysis(store, draft, labels, equipment_records)
        match_status = _historical_matches(store, draft, labels)
        _ai_troubleshooting_section(store, draft, equipment_records, match_status)


def _stored_log_analysis(store: MaintenanceStore, draft: dict, labels: dict, equipment_records: dict) -> None:
    equipment_id = draft["equipment_id"]
    registered = type(equipment_id) is int and equipment_id in labels
    database = str(default_db_path().resolve())
    result = st.session_state.get(_ANALYSIS_KEY)
    if result is not None and (
        not registered or result["draft"] != draft or result["database"] != database
        or result["equipment_label"] != labels[equipment_id]
        or result.get("equipment_identity") != equipment_records[equipment_id]
        or result.get("revision") != st.session_state.get(_ANALYSIS_REVISION_KEY, 0)
    ):
        st.session_state.pop(_ANALYSIS_KEY, None)
        result = None
    if equipment_id is None:
        st.info("Select registered equipment and submit the problem to analyze stored logs. "
                "Your description remains usable for later description-only troubleshooting.")
    elif not registered:
        st.warning("The submitted equipment is no longer valid or registered. Select registered equipment "
                   "and submit the problem again before analyzing stored logs.")
    requested = st.button("Analyze Stored Logs", key="guided_troubleshooting_analyze_logs", disabled=not registered)
    if requested and registered:
        st.session_state[_ANALYSIS_REVISION_KEY] = st.session_state.get(_ANALYSIS_REVISION_KEY, 0) + 1
        # A failed explicit refresh must not leave earlier results on display.
        st.session_state.pop(_ANALYSIS_KEY, None)
        st.session_state.pop(_MATCHES_KEY, None)
        try:
            logs = store.list_logs(equipment_id=equipment_id)
            findings = detect_fault_patterns(logs)
        except ValueError as exc:
            st.error("Stored-log analysis could not be completed. No results are retained for this attempt.")
            st.text(str(exc))
            return
        result = {"draft": dict(draft), "database": database, "equipment_label": labels[equipment_id],
                  "equipment_identity": dict(equipment_records[equipment_id]),
                  "revision": st.session_state[_ANALYSIS_REVISION_KEY], "logs": logs, "findings": findings}
        st.session_state[_ANALYSIS_KEY] = result
    if result is None:
        return

    st.subheader("Stored-log analysis")
    st.text("Equipment: " + labels[equipment_id])
    left, right = st.columns(2)
    left.metric("Logs examined", len(result["logs"]))
    right.metric("Detected fault patterns", len(result["findings"]))
    st.caption("Results reflect the last explicit analysis of the submitted problem. "
               "Click Analyze Stored Logs again after logs change. Stored-log analysis makes no AI requests.")
    if not result["logs"]:
        st.info("No stored logs were found for the submitted equipment. Import events under Equipment Logs, "
                "then click Analyze Stored Logs again.")
    elif not result["findings"]:
        st.info("No supported fault patterns were detected. This does not establish that the equipment is healthy "
                "or identify a root cause.")
    else:
        st.warning("Detected patterns are observed temporal correlations, not confirmed fault causes.")
        for number, finding in enumerate(result["findings"], start=1):
            st.subheader(f"{number}. {finding['pattern_type']}")
            st.caption("Supporting log IDs: " + ", ".join(map(str, finding["log_ids"])))
            st.text(finding["explanation"])


def _matching_context(draft: dict, labels: dict) -> str:
    equipment_id = draft["equipment_id"]
    context = {"database": str(default_db_path().resolve()), "draft": draft,
               "equipment_label": labels.get(equipment_id) if type(equipment_id) is int else None,
               "analysis": st.session_state.get(_ANALYSIS_KEY)}
    return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _index_context(search: SemanticSearch, status) -> str:
    metadata, rows = search.index.read()
    fingerprint = hashlib.sha256(json.dumps({"status": asdict(status), "metadata": metadata},
                                            sort_keys=True).encode("utf-8"))
    for row in rows:
        fingerprint.update(json.dumps({key: value for key, value in row.items() if key != "vector"},
                                      sort_keys=True).encode("utf-8"))
        vector = row["vector"]
        fingerprint.update(bytes(vector) if isinstance(vector, (bytes, bytearray, memoryview))
                           else json.dumps(vector, sort_keys=True).encode("utf-8"))
    return fingerprint.hexdigest()


def _remember_match_audit(context: str, audit: dict) -> dict:
    st.session_state[_MATCH_AUDIT_KEY] = {"context": context,
        "revision": st.session_state.get(_RETRIEVAL_REVISION_KEY, 0), "audit": dict(audit)}
    return audit


def _historical_matches(store: MaintenanceStore, draft: dict, labels: dict) -> dict:
    audit = {"index": None, "current": False, "error": None}
    st.subheader("Similar maintenance issues")
    equipment_id = draft["equipment_id"]
    valid_equipment = equipment_id is None or (type(equipment_id) is int and equipment_id in labels)
    if equipment_id is None:
        st.caption("Search scope: all registered equipment (no equipment selected).")
    elif valid_equipment:
        st.caption("Search scope: submitted equipment only — " + labels[equipment_id])
    st.caption("Semantic similarity is cosine similarity, not diagnostic confidence or probability. "
               "Historical recorded actions are not verified fixes for the current problem.")
    requested = st.button("Find Similar Maintenance Issues", key="guided_troubleshooting_find_issues",
                          disabled=not valid_equipment)
    context = _matching_context(draft, labels)
    cached = st.session_state.get(_MATCHES_KEY)
    if cached is not None and (not valid_equipment or cached["context"] != context
                              or cached.get("revision") != st.session_state.get(_RETRIEVAL_REVISION_KEY, 0)):
        st.session_state.pop(_MATCHES_KEY, None)
        cached = None
    if not valid_equipment:
        audit["error"] = "Select valid registered equipment and submit the problem again."
        return _remember_match_audit(context, audit)
    if not requested and cached is None:
        st.info("Click Find Similar Maintenance Issues to search using the submitted description "
                "and available analyzed log evidence.")
        previous = st.session_state.get(_MATCH_AUDIT_KEY)
        if (previous is not None and previous["context"] == context
                and previous["revision"] == st.session_state.get(_RETRIEVAL_REVISION_KEY, 0)
                and not previous["audit"]["current"]):
            return dict(previous["audit"])
        audit["error"] = "Find Similar Maintenance Issues first to prepare current historical evidence."
        return _remember_match_audit(context, audit)
    if requested:
        st.session_state[_RETRIEVAL_REVISION_KEY] = st.session_state.get(_RETRIEVAL_REVISION_KEY, 0) + 1
        st.session_state.pop(_MATCHES_KEY, None)
        cached = None
    try:
        # Cached weights only: this action must never initiate a model download.
        search = SemanticSearch(store, MiniLMEmbedder(offline=True))
        status = search.status()
        audit["index"] = _index_context(search, status)
        if not status.current:
            raise StaleIndexError(status.reason)
        index = audit["index"]
        if cached is not None and cached["index"] != index:
            st.session_state.pop(_MATCHES_KEY, None)
            cached = None
            st.warning("The semantic index or maintenance history changed. Earlier matches are hidden. "
                       "Click Find Similar Maintenance Issues to search again.")
            audit["error"] = "The semantic index or maintenance history changed. Find Similar Maintenance Issues again."
        if requested:
            query = build_guided_query(draft["description"], equipment_id=equipment_id,
                                       analysis=st.session_state.get(_ANALYSIS_KEY))
            records = search.search(query, limit=5, equipment_id=equipment_id)
            current = search.status()
            if not current.current or _index_context(search, current) != index:
                raise StaleIndexError("Maintenance history or its index changed during retrieval.")
            if _matching_context(st.session_state[_DRAFT_KEY], labels) != context:
                raise ValidationError("The submitted problem or log-analysis snapshot changed during retrieval. Search again.")
            cached = {"context": context, "index": index, "query": query, "matches": records,
                      "revision": st.session_state[_RETRIEVAL_REVISION_KEY]}
            st.session_state[_MATCHES_KEY] = cached
    except StaleIndexError as exc:
        st.session_state.pop(_MATCHES_KEY, None)
        audit["error"] = (str(exc) + " Open Similar Issues and click Rebuild / refresh search index, "
                          "then return and click Find Similar Maintenance Issues. No automatic rebuild was performed.")
        st.warning(audit["error"])
        return _remember_match_audit(context, audit)
    except (EmbeddingError, ValidationError) as exc:
        st.session_state.pop(_MATCHES_KEY, None)
        st.error("Historical matching could not be completed.")
        st.text(str(exc))
        audit["error"] = "Historical matching could not be completed. " + str(exc)
        return _remember_match_audit(context, audit)
    if cached is None:
        return _remember_match_audit(context, audit)
    audit["current"] = True
    if not cached["matches"]:
        st.info("No historical issues were found within the submitted equipment scope." if equipment_id is not None
                else "No historical issues were found across registered equipment.")
    for rank, issue in enumerate(cached["matches"], start=1):
        with st.expander(f"{rank}. Issue #{issue['id']} | semantic similarity {issue['similarity']:.3f}"):
            st.text(f"Equipment: {issue['equipment_name']} ({issue['asset_tag']}) | Status: {issue['status']}")
            st.caption("Historical resolution recorded; applicability to this problem is unverified."
                       if issue["status"] in ("Resolved", "Closed") else
                       "Open or unresolved historical outcome — unverified; recorded actions do not establish a successful repair.")
            for label, field in (("Description", "description"), ("Recorded troubleshooting notes", "troubleshooting_notes"),
                                 ("Recorded root cause", "root_cause"), ("Recorded corrective action", "corrective_action")):
                st.write(f"**{label}**")
                st.text(issue.get(field) or "Not recorded")
            st.caption(f"Original record: Issues → select Issue #{issue['id']}.")
    return _remember_match_audit(context, audit)


def _authorization_data(draft: dict, equipment: dict | None, analysis: dict | None,
                        matches: dict | None, index: str | None, errors: list[str],
                        context: dict | None, messages: list | None) -> dict:
    labels = {} if equipment is None else {equipment["id"]: f"{equipment['asset_tag']} — {equipment['name']}"}
    return {
        "messages": messages, "prepared_context": context, "draft": draft, "equipment": equipment,
        "database": str(default_db_path().resolve()), "analysis": analysis, "matches": matches,
        "matching_context": _matching_context(draft, labels), "index": index, "validation_errors": errors,
        "analysis_revision": st.session_state.get(_ANALYSIS_REVISION_KEY, 0),
        "retrieval_revision": st.session_state.get(_RETRIEVAL_REVISION_KEY, 0),
    }


def _authorization_signature(authorization: dict) -> str:
    return hashlib.sha256(json.dumps(authorization, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _ensure_generation_current(store: MaintenanceStore, approved: dict, signature: str, messages: list) -> None:
    """Explicit-request checks only; never retrieve logs or run embeddings on render."""
    state = st.session_state.get(_AI_KEY)
    checkbox_key = f"guided_ai_consent_{signature}"
    if (state is None or state.get("signature") != signature or state.get("consent") is not True
            or st.session_state.get(checkbox_key) is not True or state.get("validation_errors")):
        raise ValidationError("Authorization changed. Review the evidence and consent again.")
    if (_authorization_signature(approved) != signature or messages != approved["messages"]
            or state.get("messages") != approved["messages"] or state.get("context") != approved["prepared_context"]
            or state.get("serialized_messages") != json.dumps(messages, ensure_ascii=True)):
        raise ValidationError("The exact previewed provider messages or their evidence changed.")
    database = str(default_db_path().resolve())
    connected = next(row["file"] for row in store.connection.execute("PRAGMA database_list") if row["name"] == "main")
    if database != approved["database"] or Path(connected).resolve() != Path(database):
        raise ValidationError("The database workspace changed. Prepare the evidence again.")
    draft = st.session_state.get(_DRAFT_KEY)
    if not isinstance(draft, dict):
        raise ValidationError("The submitted problem is no longer available.")
    clean_text(draft.get("description"), "Problem description", required=True, limit=4000)
    equipment_id = draft.get("equipment_id")
    if equipment_id is not None and (type(equipment_id) is not int or equipment_id <= 0):
        raise ValidationError("Select valid registered equipment and submit the problem again.")
    equipment = store.get_equipment(equipment_id) if equipment_id is not None else None
    analysis, matches = st.session_state.get(_ANALYSIS_KEY), st.session_state.get(_MATCHES_KEY)
    if matches is None or not matches["matches"]:
        raise ValidationError("Current explicit historical matches are required; find similar issues again.")
    search = SemanticSearch(store, MiniLMEmbedder(offline=True))
    status = search.status()
    if not status.current:
        raise StaleIndexError("Maintenance history is stale. Open Similar Issues and Rebuild / refresh search index.")
    index = _index_context(search, status)
    current = _authorization_data(draft, equipment, analysis, matches, index, [], state["context"], state["messages"])
    if _authorization_signature(current) != signature or matches["context"] != current["matching_context"]:
        raise ValidationError("The submitted problem, equipment, historical matches, index, or evidence revisions changed.")
    if matches.get("revision") != current["retrieval_revision"] or matches["index"] != index:
        raise ValidationError("Historical matching evidence changed. Find similar issues again.")
    if analysis is not None:
        if (equipment_id is None or analysis["draft"] != draft or analysis["database"] != database
                or analysis.get("equipment_identity") != equipment
                or analysis.get("revision") != current["analysis_revision"]):
            raise ValidationError("The log-analysis snapshot changed. Analyze Stored Logs again.")
        if store.list_logs(equipment_id=equipment_id) != analysis["logs"]:
            raise ValidationError("Stored logs changed. Run Analyze Stored Logs and Find Similar Maintenance Issues again.")
    for record in matches["matches"]:
        original = store.get_issue(record["id"])
        if any(original[field] != record.get(field) for field in
               ("equipment_id", "description", "troubleshooting_notes", "root_cause", "corrective_action", "status")):
            raise ValidationError("A historical record changed. Find Similar Maintenance Issues again.")


def _guided_local_fallback(context: dict) -> dict:
    records, logs = context["historical_findings"], guided_validation_logs(context)
    uncertainty = ["Current equipment conditions and causes have not been verified. No AI hypotheses are displayed.",
                   "The technician description is a report, not an independently verified observation.",
                   "Historical actions are recorded outcomes, not verified fixes for the current equipment."]
    if not records:
        uncertainty.append("No historical matches are available; there is insufficient evidence for grounded AI advice.")
    if (any(value for key, value in context["limitations"].items() if key != "omitted_log_ids")
            or any(record["truncated_fields"] for record in records)
            or any(log["truncated_fields"] for log in logs or [])):
        uncertainty.append("Evidence was bounded, omitted, or truncated. Review the original records before drawing conclusions.")
    fallback = {"issue_summary": context["question"], **{field: [] for field in CLAIM_FIELDS},
                "evidence_references": [record["id"] for record in records],
                "uncertainty_or_missing_information": uncertainty}
    if logs:
        fallback["log_evidence_references"] = [log["id"] for log in logs]
    return fallback


def _render_guided_response(response: dict) -> None:
    analysis, generated = response["analysis"], response["ai_generated"]
    st.subheader("Guided troubleshooting assessment")
    st.caption("AI-generated hypotheses — unverified" if generated else "Local fallback — no AI-generated hypotheses")
    st.caption("Saved session result from the last explicit request. Historical actions are not verified current repairs.")
    if response["notice"]:
        st.warning(response["notice"])
    st.text(analysis["issue_summary"])
    for field, label in zip(CLAIM_FIELDS, ("Possible causes", "Suggested diagnostic checks", "Suggested next steps")):
        st.subheader(label)
        if not generated or not analysis[field]:
            st.info("No generated recommendations. Review the recorded evidence with qualified personnel using approved procedures.")
        else:
            for claim in analysis[field]:
                st.text(claim["text"])
                citations = [f"Issue #{issue_id}" for issue_id in claim["issue_ids"]]
                citations += [f"Log #{log_id}" for log_id in claim.get("log_ids", [])]
                st.caption("Supporting evidence: " + ", ".join(citations))
    st.subheader("Evidence references")
    st.text("Historical issues: " + (", ".join(f"Issue #{value}" for value in analysis["evidence_references"]) or "None"))
    if "log_evidence_references" in analysis:
        st.text("Recorded logs: " + (", ".join(f"Log #{value}" for value in analysis["log_evidence_references"]) or "None"))
    st.subheader("Uncertainty or missing information")
    for text in analysis["uncertainty_or_missing_information"]:
        st.text(text)


def _ai_troubleshooting_section(store: MaintenanceStore, draft: dict, equipment_records: dict, match_status: dict) -> None:
    st.subheader("AI Troubleshooting")
    st.warning(ADVISORY_WARNING)
    if _AI_NOTICE_KEY in st.session_state:
        st.warning(st.session_state.pop(_AI_NOTICE_KEY))
    st.info("The exact previewed messages may be sent to your configured external AI provider, including Gemini. "
            "They can contain sensitive equipment and maintenance information. Cloud use may incur provider charges. "
            "Sending requires your consent and an explicit Generate AI Troubleshooting click.")
    equipment_id = draft["equipment_id"]
    equipment = equipment_records.get(equipment_id) if type(equipment_id) is int else None
    analysis = st.session_state.get(_ANALYSIS_KEY)
    matches = st.session_state.get(_MATCHES_KEY)
    errors = [match_status["error"]] if match_status["error"] else []
    context, messages = None, None
    try:
        if equipment_id is not None and equipment is None:
            raise ValidationError("The submitted equipment is invalid or no longer registered. Submit a valid selection.")
        records = matches["matches"] if match_status["current"] and matches is not None else None
        context = build_guided_context(draft, equipment=equipment, analysis=analysis, matches=records)
        messages = build_guided_messages(context)
        if not messages:
            raise ValidationError("No exact provider messages were prepared.")
        if not context["historical_findings"] and not errors:
            errors.append("No current historical matches are available. Find Similar Maintenance Issues; "
                          "consent is unavailable without historical evidence.")
    except (ValueError, KeyError, TypeError) as exc:
        errors.append("Guided evidence could not be prepared. " + str(exc))
        messages = None
    authorization = _authorization_data(draft, equipment, analysis, matches, match_status["index"], errors, context, messages)
    signature = _authorization_signature(authorization)
    state = st.session_state.get(_AI_KEY)
    checkbox_key = f"guided_ai_consent_{signature}"
    if state is None or state["signature"] != signature:
        state = {"signature": signature, "consent": False, "response": None}
        st.session_state[_AI_KEY] = state
        # Only write checkbox state before creating the widget in this run.
        # This also prevents old authorization resurfacing after an A → B → A change.
        st.session_state[checkbox_key] = False
    elif checkbox_key not in st.session_state:
        # Widget keys can be removed while another page renders. Restore only
        # the authorization held for this exact unchanged evidence in this session.
        st.session_state[checkbox_key] = state["consent"] and not errors
    if errors:
        state["consent"], state["response"] = False, None
        st.session_state[checkbox_key] = False
    state["context"], state["messages"], state["validation_errors"] = context, messages, errors
    if messages is not None:
        st.subheader("Exact provider messages")
        state["serialized_messages"] = json.dumps(messages, ensure_ascii=True)
        st.code(state["serialized_messages"], language="json")
    else:
        state["serialized_messages"] = None
    for error in errors:
        st.warning(error)
    consent = st.checkbox("Allow sending these exact previewed messages to the configured external AI provider",
                          value=False, key=checkbox_key, disabled=bool(errors))
    state["consent"] = bool(consent) and not errors
    requested = st.button("Generate AI Troubleshooting", key="guided_ai_generate", disabled=bool(errors) or not state["consent"])
    if requested and bool(consent) and not errors:
        approved = deepcopy(authorization)
        prepared_messages = deepcopy(approved["messages"])
        prepared_context = approved["prepared_context"]
        state["response"] = None
        try:
            _ensure_generation_current(store, approved, signature, prepared_messages)
            with st.spinner("Requesting advisory guided troubleshooting..."):
                result, generated, notice = generate_grounded_analysis(
                    prepared_messages, prepared_context["historical_findings"], _guided_local_fallback(prepared_context),
                    use_cloud=True, logs=guided_validation_logs(prepared_context))
            _ensure_generation_current(store, approved, signature, prepared_messages)
            state["response"] = {"analysis": result, "ai_generated": generated, "notice": notice}
        except (ValueError, KeyError, TypeError, StaleIndexError, EmbeddingError, sqlite3.Error) as exc:
            st.session_state.pop(_AI_KEY, None)
            st.session_state.pop(_ANALYSIS_KEY, None)
            st.session_state.pop(_MATCHES_KEY, None)
            st.session_state[_AI_NOTICE_KEY] = "Troubleshooting response discarded: " + str(exc) + " Review fresh evidence and consent again."
            # The checkbox already exists in this run. Revoke it on the next
            # run, before widget creation, without retrying generation.
            st.rerun()
    if state["response"] is not None:
        _render_guided_response(state["response"])
    elif context is not None and match_status["current"] and not context["historical_findings"]:
        _render_guided_response({"analysis": _guided_local_fallback(context), "ai_generated": False,
                                 "notice": "No historical matches. Showing local evidence only; no provider request was made."})
