"""Consent-bound fault troubleshooting; responses live only in the Streamlit session."""

from dataclasses import asdict
import hashlib
import json

import streamlit as st

from maintainai.database import default_db_path
from maintainai.embeddings import EmbeddingError
from maintainai.fault_rag import (
    HISTORICAL_WARNING, build_fault_context, build_fault_messages, troubleshoot_fault,
)
from maintainai.llm import ProviderError
from maintainai.rag import ADVISORY_WARNING, CLAIM_FIELDS
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError

_FINDING_KEY = "fault_ai_finding"


def _fingerprint(analysis: dict, selected: int) -> str:
    # Include full snapshots, not just clipped preview text. Explicit analysis/match
    # revisions also revoke earlier authorization when identical data is refreshed.
    inputs = {"database": str(default_db_path().resolve()), "selected": selected,
              "analysis_id": analysis.get("analysis_id"), "analyzed_at": analysis["analyzed_at"],
              "equipment_id": analysis["equipment_id"], "findings": analysis["findings"],
              "logs": analysis["logs"], "matches": analysis.get("historical_matches")}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ensure_current(store: MaintenanceStore, analysis: dict) -> None:
    cached = analysis.get("historical_matches")
    if cached is None:
        raise ValidationError("Find Similar Historical Issues first to prepare historical evidence.")
    status = SemanticSearch(store).status()
    if not status.current or cached["source_signature"] != status.source_signature:
        raise StaleIndexError("Historical evidence is stale. Open Similar Issues and use Rebuild / refresh search index, then find historical matches again.")
    if store.list_logs(equipment_id=analysis["equipment_id"]) != analysis["logs"]:
        raise ValidationError("Stored logs changed since this analysis. Run Analyze Logs and Find Similar Historical Issues again before authorizing cloud use.")


def _resolve(analysis: dict, selected: int) -> tuple[dict, list[dict], list[dict]]:
    finding = analysis["findings"][selected]
    cached = analysis.get("historical_matches")
    if cached is None or selected >= len(cached["entries"]):
        raise ValidationError("Find Similar Historical Issues first to prepare historical evidence.")
    entry = cached["entries"][selected]
    if "error" in entry:
        raise ValidationError("Historical matching failed for this finding. Resolve the error and find historical matches again.")
    by_id = {log["id"]: log for log in analysis["logs"]}
    supporting = [by_id[log_id] for log_id in finding["log_ids"] if log_id in by_id]
    return finding, supporting, entry["result"]["matches"]


def _render_response(response: dict) -> None:
    generated = response["ai_generated"]
    st.subheader("Fault troubleshooting assessment")
    st.caption("AI-generated interpretation and hypotheses — unverified" if generated else "Local fallback — no AI-generated hypotheses")
    if response["notice"]:
        st.warning(response["notice"])
    st.text(response["analysis"]["issue_summary"])
    for field, label in zip(CLAIM_FIELDS, ("Possible causes", "Recommended diagnostic checks", "Suggested next steps")):
        st.subheader(label)
        st.caption("Unverified hypotheses/advisory suggestions; qualified human verification is required.")
        claims = response["analysis"][field]
        if not generated or not claims:
            st.info("No generated recommendations in this section. Review the observed evidence and approved procedures with qualified personnel.")
        else:
            for claim in claims:
                st.text(claim["text"])
                st.caption("Evidence: " + ", ".join(f"Log #{log_id}" for log_id in claim["log_ids"])
                           + " | " + ", ".join(f"Issue #{issue_id}" for issue_id in claim["issue_ids"]))
    st.subheader("Fault troubleshooting evidence references")
    st.text("Logs: " + (", ".join(f"Log #{value}" for value in response["analysis"]["log_evidence_references"]) or "None"))
    st.text("Historical issues: " + (", ".join(f"Issue #{value}" for value in response["analysis"]["evidence_references"]) or "None"))
    st.subheader("Uncertainty or missing information")
    for text in response["analysis"]["uncertainty_or_missing_information"]:
        st.text(text)


def fault_troubleshooting_section(store: MaintenanceStore, analysis: dict) -> None:
    st.subheader("AI Troubleshooting")
    st.warning(ADVISORY_WARNING)
    st.caption(HISTORICAL_WARNING)
    st.caption("Results are held only in this session, never written to the database. Cloud use may incur provider charges and sends the previewed evidence to the existing configured endpoint.")
    if "ai_invalidated_notice" in analysis:
        st.warning(analysis.pop("ai_invalidated_notice"))
    options = list(range(len(analysis["findings"])))
    if not options:
        return
    saved = analysis.get("ai_selected_finding", 0)
    if st.session_state.get(_FINDING_KEY, saved) not in options:
        st.session_state.pop(_FINDING_KEY, None)
    if _FINDING_KEY not in st.session_state:
        st.session_state[_FINDING_KEY] = saved if saved in options else 0
    selected = st.selectbox("Fault finding for AI troubleshooting", options, key=_FINDING_KEY,
                            format_func=lambda index: f"{index + 1}. {analysis['findings'][index]['pattern_type']} | equipment ID {analysis['findings'][index]['equipment_id']}")
    analysis["ai_selected_finding"] = selected
    fingerprint = _fingerprint(analysis, selected)
    context, messages, problem = None, None, None
    try:
        _ensure_current(store, analysis)
        finding, supporting, matches = _resolve(analysis, selected)
        context = build_fault_context(finding, supporting, matches)
        messages = build_fault_messages(context)
        observed = context["observed_evidence"]
        if not observed["logs"] or observed["missing_log_ids"] or not context["historical_findings"]:
            problem = "Complete supporting logs and historical matches are required. Run Analyze Logs and Find Similar Historical Issues again."
    except (ValidationError, StaleIndexError, EmbeddingError) as exc:
        problem = str(exc)
    # The checkbox key changes whenever evidence or validity changes. A previous
    # checked widget cannot authorize a different payload, selection, or snapshot.
    signature = hashlib.sha256(json.dumps({"evidence": fingerprint, "problem": problem,
                                           "messages": messages}, sort_keys=True).encode("utf-8")).hexdigest()
    state = analysis.get("ai_troubleshooting")
    if state is None or state["signature"] != signature:
        state = {"signature": signature, "response": None}
        analysis["ai_troubleshooting"] = state
    if context is not None:
        envelope = json.loads(messages[1]["content"])
        st.subheader("Observed evidence")
        st.json(context["observed_evidence"])
        st.subheader("Historical findings")
        st.caption("Untrusted recorded ticket excerpts. Recorded corrective actions are not confirmed current fixes; open outcomes remain unverified.")
        st.json(envelope["untrusted_records"])
        st.subheader("Cloud evidence preview")
        st.caption("This is the exact evidence envelope sent on authorization, with the existing advisory system instructions. No reporter details, credentials, or additional records are added. Truncation and omissions are shown explicitly.")
        st.json(envelope)
    if problem:
        st.warning(problem)
    consent = st.checkbox("Allow sending this previewed fault evidence and historical tickets to the configured provider",
                          value=False, key=f"fault_ai_consent_{signature}", disabled=problem is not None)
    requested = st.button("Generate AI troubleshooting", key="fault_ai_generate",
                          disabled=problem is not None or not consent)
    if requested and consent and problem is None:
        state["response"] = None
        try:
            _ensure_current(store, analysis)
            with st.spinner("Requesting advisory fault troubleshooting..."):
                response = troubleshoot_fault(finding, supporting, matches, use_cloud=True)
            _ensure_current(store, analysis)
            if _fingerprint(analysis, selected) != fingerprint:
                raise ValidationError("Fault evidence changed during the request.")
            state["response"] = asdict(response)
        except (ValidationError, StaleIndexError, EmbeddingError, ProviderError) as exc:
            analysis.pop("ai_troubleshooting", None)
            analysis["ai_invalidated_notice"] = "Troubleshooting response discarded: " + str(exc)
            # Refresh disabled controls and the consent widget before displaying
            # another request opportunity. No automatic retry or regeneration.
            st.rerun()
    if state["response"] is not None:
        _render_response(state["response"])
