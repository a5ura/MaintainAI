"""Semantic search screen; no model is loaded until a rebuild or search is requested."""

import streamlit as st

from maintainai.embeddings import EmbeddingError
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError


def similar_issues_page(store: MaintenanceStore) -> None:
    st.write("Describe a maintenance problem to find up to five similar historical issues across all equipment and issue statuses.")
    st.caption("Similarity is cosine similarity from -1 to 1, not a confidence or diagnosis. Open Issues and select a result's issue number to review or update the original record.")
    search = SemanticSearch(store)
    status = search.status()
    if not status.total_issues:
        st.info("No maintenance issues yet. Load synthetic demo data from the sidebar or create an issue first.")
        return
    st.caption(f"{status.indexed_issues} indexed / {status.total_issues} current issues. Last rebuilt (UTC): {status.built_at or 'Never'}")
    if status.current:
        st.success("Search index is current.")
    else:
        st.warning(status.reason + " Rebuild to enable search. Stale results are hidden.")
    st.caption("The first rebuild downloads the free model. Embedding inference runs locally on CPU. Subsequent searches use the cached model.")
    if st.button("Rebuild / refresh search index", type="primary"):
        try:
            with st.spinner("Building local embeddings; the first run may take several minutes..."):
                count = search.rebuild()
            st.session_state["flash"] = f"Search index refreshed for {count} issues."
            st.rerun()
        except (EmbeddingError, StaleIndexError) as exc:
            st.error(str(exc))
            return
    with st.form("similar_issues_search"):
        query = st.text_area("Maintenance problem description", max_chars=10000,
                             placeholder="Vacuum gripper loses suction and drops parts during transfer")
        submitted = st.form_submit_button("Find similar issues", disabled=not status.current)
    if submitted:
        try:
            with st.spinner("Searching historical issues on CPU..."):
                results = search.search(query)
        except (EmbeddingError, StaleIndexError, ValidationError) as exc:
            st.error(str(exc))
            return
        st.caption(f"{len(results)} results. Results are recomputed on submission and are not cached between page reruns.")
        for rank, issue in enumerate(results, start=1):
            st.subheader(f"{rank}. Issue #{issue['id']} · similarity {issue['similarity']:.3f}")
            st.text(f"Equipment: {issue['equipment_name']} ({issue['asset_tag']}); equipment ID #{issue['equipment_id']}")
            st.caption(f"Original record: Issues → Issue #{issue['id']} | Status: {issue['status']} | Priority: {issue['priority']}")
            for label, field in (("Description", "description"), ("Troubleshooting notes", "troubleshooting_notes"),
                                 ("Root cause", "root_cause"), ("Corrective action", "corrective_action")):
                st.write(f"**{label}**")
                st.text(issue[field] or "Not recorded")
