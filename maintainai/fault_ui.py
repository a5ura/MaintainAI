"""Explicit fault analysis with session-local snapshots, never automatic detection."""

from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from maintainai.database import default_db_path
from maintainai.embeddings import EmbeddingError
from maintainai.fault_detection import detect_fault_patterns
from maintainai.fault_matching import HISTORICAL_NOTICE, match_fault_to_issues
from maintainai.fault_rag_ui import fault_troubleshooting_section
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError

_RESULT_KEY = "fault_analysis_result"
_SELECTION_KEY = "fault_analysis_equipment_id"
_WIDGET_KEY = "fault_analysis_equipment_widget"
_DATABASE_KEY = "fault_analysis_database"
_SEVERITY_LABELS = {"INFO": "⚪ INFO", "WARNING": "🟡 WARNING", "ERROR": "🔴 ERROR", "CRITICAL": "🟣 CRITICAL"}


def _equipment_changed() -> None:
    selected = st.session_state[_WIDGET_KEY]
    if selected != st.session_state.get(_SELECTION_KEY):
        st.session_state.pop(_RESULT_KEY, None)
    # Keep the selection separately from widget state so it survives navigation.
    st.session_state[_SELECTION_KEY] = selected


def _evidence_table(logs: list[dict]) -> None:
    if not logs:
        st.info("Supporting evidence logs are unavailable in this analysis snapshot. Run Analyze Logs again.")
        return
    columns = ["id", "timestamp", "asset_tag", "severity", "event_code", "message"]
    frame = pd.DataFrame(logs)[columns]
    frame["severity"] = frame["severity"].map(_SEVERITY_LABELS)
    frame.columns = ["Log ID", "Timestamp (UTC)", "Asset Tag", "Severity", "Event Code", "Message"]
    st.dataframe(frame, hide_index=True, use_container_width=True)


def _historical_matches(store: MaintenanceStore, analysis: dict) -> None:
    st.subheader("Similar historical maintenance issues")
    st.caption(HISTORICAL_NOTICE)
    st.caption("Scores are cosine similarities from -1 to 1, not confidence percentages. Search runs locally on CPU across all equipment and issue statuses. Matches reflect the last explicit search against this analysis snapshot.")
    requested = st.button("Find Similar Historical Issues", key="fault_matching_button")
    cached = analysis.get("historical_matches")
    if not requested and cached is None:
        st.info("Click Find Similar Historical Issues to search the existing maintenance history for each finding.")
        return
    if requested:
        # A failed new attempt must not leave the previous matches on screen.
        analysis.pop("historical_matches", None)
        cached = None
    search = SemanticSearch(store)
    try:
        status = search.status()
        if cached is not None and (not status.current or cached["source_signature"] != status.source_signature):
            analysis.pop("historical_matches", None)
            cached = None
            st.warning("Maintenance history or its index changed. Earlier historical matches are hidden. Search again after the index is current.")
        if not status.total_issues:
            st.info("No historical maintenance issues are available. Create an issue or load synthetic demo history, then refresh the index on Similar Issues.")
            return
        if not status.current:
            raise StaleIndexError(status.reason)
        if requested:
            by_id = {log["id"]: log for log in analysis["logs"]}
            entries = []
            with st.spinner("Matching observed faults to historical issues on CPU..."):
                for finding in analysis["findings"]:
                    supporting = [by_id[log_id] for log_id in finding["log_ids"] if log_id in by_id]
                    try:
                        matched = match_fault_to_issues(finding, supporting, search)
                        entries.append({"result": asdict(matched)})
                    except (EmbeddingError, ValidationError) as exc:
                        entries.append({"error": str(exc)})
                # All findings must refer to the same current ticket snapshot.
                current = search.status()
                if not current.current or current.source_signature != status.source_signature:
                    raise StaleIndexError("Maintenance history changed while matching faults.")
            cached = {"source_signature": status.source_signature, "entries": entries, "revision": uuid4().hex}
            analysis["historical_matches"] = cached
    except StaleIndexError as exc:
        analysis.pop("historical_matches", None)
        st.warning(str(exc) + " Open Similar Issues and click Rebuild / refresh search index, then return and press Find Similar Historical Issues. No automatic rebuild was performed.")
        return
    except (EmbeddingError, ValidationError) as exc:
        analysis.pop("historical_matches", None)
        st.error("Historical matching could not be completed.")
        st.text(str(exc))
        return
    if cached is None:
        return
    for number, (finding, entry) in enumerate(zip(analysis["findings"], cached["entries"]), start=1):
        st.subheader(f"Historical matches for finding {number}: {finding['pattern_type']}")
        if "error" in entry:
            st.error("Historical matching unavailable for this finding. Retry with Find Similar Historical Issues after resolving the error.")
            st.text(entry["error"])
            continue
        matched = entry["result"]
        st.caption(matched["notice"])
        if matched["missing_log_ids"]:
            st.warning("Missing supporting log IDs: " + ", ".join(map(str, matched["missing_log_ids"])) + ". Run Analyze Logs again to refresh the evidence.")
        if not matched["matches"]:
            st.info("No historical issues were retrieved for this finding.")
            continue
        for rank, issue in enumerate(matched["matches"], start=1):
            with st.expander(f"{rank}. Issue #{issue['issue_id']} | cosine similarity {issue['similarity']:.3f}"):
                st.text(f"Equipment: {issue['equipment_name']} ({issue['asset_tag']}) | Status: {issue.get('status') or 'Not recorded'}")
                st.caption(f"Original record: Issues → select Issue #{issue['issue_id']} (clear filters if necessary).")
                for label, field in (("Description", "description"), ("Recorded root cause", "root_cause"),
                                     ("Recorded corrective action", "corrective_action")):
                    st.write(f"**{label}**")
                    st.text(issue.get(field) or "Not recorded")


def fault_analysis_page(store: MaintenanceStore) -> None:
    st.warning("Findings are potential correlations in recorded events, not confirmed root causes. Qualified personnel must verify the evidence before making maintenance or operational decisions.")
    st.caption("Severity legend: 🟡 WARNING · 🔴 ERROR · 🟣 CRITICAL · ⚪ INFO. All timestamps are UTC.")
    equipment = store.list_equipment()
    labels = {item["id"]: f"{item['asset_tag']} — {item['name']}" for item in equipment}
    options = [None] + list(labels)
    database = str(default_db_path().resolve())
    if st.session_state.get(_DATABASE_KEY) != database:
        for key in (_RESULT_KEY, _SELECTION_KEY, _WIDGET_KEY):
            st.session_state.pop(key, None)
        st.session_state[_DATABASE_KEY] = database
    saved_selection = st.session_state.get(_SELECTION_KEY)
    if saved_selection not in options or st.session_state.get(_WIDGET_KEY) not in options:
        st.session_state.pop(_RESULT_KEY, None)
        saved_selection = None
        st.session_state[_SELECTION_KEY] = None
        st.session_state.pop(_WIDGET_KEY, None)
    if _WIDGET_KEY not in st.session_state:
        st.session_state[_WIDGET_KEY] = saved_selection
    selected = st.selectbox("Equipment", options,
                            format_func=lambda value: "All Equipment" if value is None else labels[value],
                            key=_WIDGET_KEY, on_change=_equipment_changed)
    result = st.session_state.get(_RESULT_KEY)
    if result is not None and result["equipment_id"] != selected:
        st.session_state.pop(_RESULT_KEY, None)
        result = None
    if not equipment:
        st.info("No equipment registered. Add equipment and import events under Equipment Logs to build a history for analysis.")
    if st.button("Analyze Logs", type="primary", key="fault_analysis_button"):
        # A failed explicit refresh must not leave earlier results on screen.
        st.session_state.pop(_RESULT_KEY, None)
        try:
            logs = store.list_logs(equipment_id=selected)
            findings = detect_fault_patterns(logs)
        except ValueError as exc:
            st.error("Fault analysis could not be completed. No results are retained for this attempt.")
            st.text(str(exc))
            return
        result = {"equipment_id": selected, "logs": logs, "findings": findings,
                  "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "analysis_id": uuid4().hex}
        st.session_state[_RESULT_KEY] = result
    if result is None:
        st.info("Select equipment and click Analyze Logs. Analysis runs only when you press the button.")
        return

    logs, findings = result["logs"], result["findings"]
    left, right = st.columns(2)
    left.metric("Logs examined", len(logs))
    right.metric("Findings", len(findings))
    st.caption(f"Analysis snapshot (UTC): {result['analyzed_at']}. Results reflect the last explicit run. Press Analyze Logs again after importing or changing logs.")
    if not logs:
        st.info("No logs found for this selection. Import events under Equipment Logs, then press Analyze Logs again.")
        return
    if not findings:
        st.info("No supported fault patterns were detected. This does not establish that the equipment is healthy or identify a root cause.")
        st.subheader("Logs examined")
        _evidence_table(logs)
        return
    by_id = {log["id"]: log for log in logs}
    evidence_ids = set()
    for number, finding in enumerate(findings, start=1):
        st.subheader(f"{number}. {finding['pattern_type']}")
        severity = finding["severity"]
        if severity == "WARNING":
            st.warning("WARNING — observed severity", icon="⚠️")
        else:
            st.error(f"{severity} — highest observed severity", icon="🚨" if severity == "CRITICAL" else "🔴")
        supporting = [by_id[log_id] for log_id in finding["log_ids"] if log_id in by_id]
        if supporting:
            first = supporting[0]
            st.text(f"Equipment: {first['equipment_name']} ({first['asset_tag']}); ID {finding['equipment_id']}")
        else:
            st.text(f"Equipment ID {finding['equipment_id']}: supporting logs unavailable in this snapshot.")
        st.text(f"Start (UTC): {finding['start_timestamp']}\nEnd (UTC): {finding['end_timestamp']}")
        st.text(finding["explanation"])
        st.caption("Supporting log IDs: " + ", ".join(str(log_id) for log_id in finding["log_ids"]))
        evidence_ids.update(finding["log_ids"])
    st.subheader("Supporting evidence logs")
    st.caption("Each supporting log is listed once, even when it supports more than one pattern. Overlapping matches within a rule are merged into episodes; different rules can share evidence.")
    _evidence_table([log for log in logs if log["id"] in evidence_ids])
    _historical_matches(store, result)
    fault_troubleshooting_section(store, result)
