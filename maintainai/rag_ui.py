"""Human-controlled troubleshooting screen. Cloud requests require an explicit opt-in."""

import streamlit as st

from maintainai.embeddings import EmbeddingError
from maintainai.rag import ADVISORY_WARNING, CLAIM_FIELDS, TroubleshootingAssistant
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError


def troubleshooting_page(store: MaintenanceStore) -> None:
    st.warning(ADVISORY_WARNING)
    st.caption("Historical records are untrusted source text. Historical findings are not a diagnosis of the current problem.")
    search = SemanticSearch(store)
    status = search.status()
    if not status.total_issues:
        st.info("No maintenance history yet. Load synthetic demo data or create a maintenance issue first.")
        return
    if not status.current:
        st.warning(status.reason + " Open Similar Issues and rebuild the search index before troubleshooting.")
    else:
        st.caption(f"Search index is current: {status.indexed_issues} issues. Responses are not saved or cached.")
    mode = st.radio("Assistant mode", ("Retrieval only (no cloud request)", "Cloud-assisted RAG"))
    cloud = mode == "Cloud-assisted RAG"
    consent = False
    if cloud:
        st.info("Cloud mode sends your question, selected equipment information, and up to five ticket excerpts to your configured inference endpoint. These may contain sensitive data. Provider charges may apply.")
        consent = st.checkbox("Allow sending this question and retrieved excerpts to the configured provider")
        st.caption("Configure the endpoint, model, and API key in environment variables before launching Streamlit. Missing configuration falls back to retrieval-only evidence.")
    equipment = store.list_equipment()
    labels = {item["id"]: f"{item['asset_tag']} — {item['name']}" for item in equipment}
    with st.form("troubleshooting_assistant"):
        equipment_id = st.selectbox("Equipment (optional)", [None] + list(labels),
                                     format_func=lambda value: "All equipment" if value is None else labels[value])
        question = st.text_area("Describe the equipment problem", max_chars=4000,
                                placeholder="The vacuum gripper loses suction while transferring a part")
        submitted = st.form_submit_button("Get troubleshooting guidance", disabled=not status.current or (cloud and not consent))
    if not submitted:
        return
    try:
        with st.spinner("Retrieving maintenance evidence" + (" and requesting advisory analysis..." if cloud else "...")):
            result = TroubleshootingAssistant(store, search).answer(question, equipment_id=equipment_id, use_cloud=cloud)
    except (ValidationError, EmbeddingError, StaleIndexError) as exc:
        st.error(str(exc))
        return
    if result.notice:
        st.warning(result.notice)
    st.subheader("Issue summary")
    st.caption("AI-generated interpretation — unverified" if result.ai_generated else "Your submitted description — no AI generation")
    st.text(result.analysis["issue_summary"])
    st.subheader("Retrieved historical facts — untrusted ticket quotations")
    for record in result.records:
        with st.expander(f"Issue #{record['id']} · {record['equipment_name']} · similarity {record['similarity']:.3f}"):
            st.text(f"Equipment ID #{record['equipment_id']} | {record['asset_tag']} | Status: {record['status']}")
            for label, field in (("Description", "description"), ("Troubleshooting notes", "troubleshooting_notes"),
                                 ("Recorded root cause", "root_cause"), ("Recorded corrective action", "corrective_action")):
                st.write(f"**{label}**")
                st.text(record[field] or "Not recorded")
            st.caption(f"Original record: Issues → select Issue #{record['id']} (clear filters if necessary).")
    for field, label in zip(CLAIM_FIELDS, ("Possible causes", "Recommended diagnostic checks", "Suggested next steps")):
        st.subheader(label)
        if not result.ai_generated or not result.analysis[field]:
            st.info("No generated recommendations in this section. Have qualified personnel review the historical evidence and approved procedures.")
            continue
        st.caption("AI-generated hypotheses/advisory suggestions — not verified facts or authorization to operate equipment")
        for item in result.analysis[field]:
            st.text(item["text"])
            st.caption("Historical evidence: " + ", ".join(f"Issue #{issue_id}" for issue_id in item["issue_ids"]))
    st.subheader("Evidence references")
    ids = result.analysis["evidence_references"]
    st.text(", ".join(f"Issue #{issue_id}" for issue_id in ids) or "No historical evidence available")
    st.subheader("Uncertainty or missing information")
    for item in result.analysis["uncertainty_or_missing_information"]:
        st.text(item)
    st.caption("Citations identify retrieved tickets; they do not prove that a generated hypothesis is correct. No equipment or ticket changes were made.")
