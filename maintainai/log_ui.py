"""Equipment event log browsing and explicit CSV imports using the current store."""

import pandas as pd
import streamlit as st

from maintainai.csv_import import CSVImportError
from maintainai.log_import import import_logs_csv
from maintainai.service import LOG_SEVERITIES, MaintenanceStore


def equipment_logs_page(store: MaintenanceStore) -> None:
    equipment = store.list_equipment()
    st.subheader("Import equipment logs")
    st.write("Upload a UTF-8 CSV for equipment already in the registry. Event timestamps must include a timezone, for example 2026-10-25T09:30:00Z.")
    st.caption("Required columns: equipment_asset_tag, timestamp, severity, message. Optional columns: event_code, source.")
    st.code("equipment_asset_tag,timestamp,severity,message,event_code,source", language=None)
    st.caption("Maximum 5 MiB / 10,000 rows. Invalid files make no changes. Imports append logs; importing the same file again adds more records.")
    if not equipment:
        st.info("No equipment registered. Add assets under Equipment or load synthetic demo data before importing logs. Log imports never create equipment.")
    uploaded = st.file_uploader("Choose equipment logs CSV", type=["csv"], key="logs_csv_upload")
    if uploaded is not None:
        st.write(f"Selected: {uploaded.name}")
    if st.button("Import logs", type="primary", disabled=uploaded is None or not equipment, key="logs_import_button"):
        try:
            count = import_logs_csv(store, uploaded.getvalue())
            st.success(f"Imported {count} equipment log record(s).")
        except CSVImportError as exc:
            st.error("Log import rejected. No logs were added.")
            for error in exc.errors:
                st.text(error)

    st.subheader("Stored equipment logs")
    left, right = st.columns(2)
    labels = {item["id"]: f"{item['asset_tag']} — {item['name']}" for item in equipment}
    with left:
        equipment_id = st.selectbox("Equipment filter", [None] + list(labels),
                                    format_func=lambda value: "All equipment" if value is None else labels[value],
                                    key="logs_equipment_filter")
    with right:
        severity = st.selectbox("Severity filter", (None,) + LOG_SEVERITIES,
                                format_func=lambda value: "All severities" if value is None else value,
                                key="logs_severity_filter")
    records = store.list_logs(equipment_id=equipment_id, severity=severity)
    st.caption(f"{len(records)} matching log record(s). All event timestamps are displayed in UTC, newest first.")
    if not records:
        if equipment_id is not None or severity is not None:
            st.info("No logs match the selected filters. Choose All equipment and All severities to view every stored log.")
        else:
            st.info("No equipment logs yet. Import a CSV file to start building equipment event history.")
        return
    columns = ["timestamp", "asset_tag", "equipment_name", "severity", "event_code", "message", "source"]
    frame = pd.DataFrame(records)[columns]
    frame.columns = ["Timestamp (UTC)", "Asset Tag", "Equipment Name", "Severity", "Event Code", "Message", "Source"]
    st.dataframe(frame, hide_index=True, use_container_width=True)
