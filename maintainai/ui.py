"""Streamlit presentation; domain operations live in service.py."""

import pandas as pd
import streamlit as st

from maintainai.csv_import import CSVImportError, import_csv
from maintainai.database import default_db_path
from maintainai.seed import SAMPLE_DIR, seed_demo
from maintainai.semantic_ui import similar_issues_page
from maintainai.rag_ui import troubleshooting_page
from maintainai.service import EQUIPMENT_STATUSES, ISSUE_STATUSES, PRIORITIES, MaintenanceStore, ValidationError

PAGES = ("Dashboard", "Equipment", "Create issue", "Issues", "Equipment history", "CSV import", "Similar Issues", "AI Troubleshooting Assistant")


def table(records: list[dict], columns: list[str]) -> None:
    if not records:
        st.info("No records to display.")
        return
    frame = pd.DataFrame(records)[columns]
    frame.columns = [column.replace("_", " ").title() for column in columns]
    st.dataframe(frame, hide_index=True, use_container_width=True)


def saved(message: str) -> None:
    st.session_state["flash"] = message
    st.rerun()


def equipment_selector(equipment: list[dict], *, label: str = "Equipment", key: str, allow_all: bool = False):
    labels = {item["id"]: f"{item['asset_tag']} — {item['name']}" for item in equipment}
    options = ([None] if allow_all else []) + list(labels)
    return st.selectbox(label, options, format_func=lambda item: "All equipment" if item is None else labels[item], key=key)


def render(store: MaintenanceStore) -> None:
    with st.sidebar:
        st.header("Workspace")
        page = st.radio("Navigate", PAGES)
        st.caption("Database")
        st.code(str(default_db_path()), language=None)
        if st.button("Load synthetic demo data", use_container_width=True):
            result = seed_demo(store.connection)
            saved(f"Synthetic demo data loaded: {result['equipment_added']} equipment and {result['issues_added']} issues added. Existing data was preserved.")
        st.caption("Run this loader again safely; existing demo records are preserved.")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    st.header(page)
    {
        "Dashboard": dashboard,
        "Equipment": equipment_page,
        "Create issue": create_issue_page,
        "Issues": issues_page,
        "Equipment history": history_page,
        "CSV import": import_page,
        "Similar Issues": similar_issues_page,
        "AI Troubleshooting Assistant": troubleshooting_page,
    }[page](store)


def dashboard(store: MaintenanceStore) -> None:
    stats = store.dashboard()
    for column, label, value in zip(st.columns(4), ("Equipment", "All issues", "Open issues", "Closed issues"),
                                    (stats["total_equipment"], stats["total_issues"], stats["open_issues"], stats["closed_issues"])):
        column.metric(label, value)
    st.caption("Open = Open + In Progress. Closed = Resolved + Closed. Equipment condition is managed independently of issue status. All timestamps are UTC.")
    if not stats["total_equipment"]:
        st.info("Start by loading the synthetic demo data from the sidebar, adding equipment, or importing an equipment CSV.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Issue status")
        st.bar_chart(pd.DataFrame.from_dict(stats["issues_by_status"], orient="index", columns=["Issues"]))
    with right:
        st.subheader("Equipment condition")
        st.bar_chart(pd.DataFrame.from_dict(stats["equipment_by_status"], orient="index", columns=["Equipment"]))
    st.subheader("Equipment categories")
    frame = pd.DataFrame(store.list_equipment())
    st.dataframe(frame.groupby("category").size().rename("Equipment").reset_index(), hide_index=True, use_container_width=True)
    st.subheader("Recent issues")
    table(store.list_issues()[:10], ["id", "asset_tag", "description", "priority", "status", "created_at"])


def equipment_page(store: MaintenanceStore) -> None:
    equipment = store.list_equipment()
    table(equipment, ["id", "asset_tag", "name", "category", "manufacturer", "model", "location", "operational_status"])
    with st.expander("Register equipment", expanded=not equipment):
        with st.form("register_equipment", clear_on_submit=True):
            asset_tag = st.text_input("Asset tag *", max_chars=200)
            name = st.text_input("Name *", max_chars=200)
            category = st.text_input("Category *", max_chars=200)
            manufacturer = st.text_input("Manufacturer", max_chars=200)
            model = st.text_input("Model", max_chars=200)
            location = st.text_input("Location", max_chars=200)
            condition = st.selectbox("Condition", EQUIPMENT_STATUSES)
            if st.form_submit_button("Register equipment"):
                try:
                    store.create_equipment(asset_tag=asset_tag, name=name, category=category, manufacturer=manufacturer,
                                           model=model, location=location, operational_status=condition)
                    saved(f"Equipment {asset_tag.strip()} registered.")
                except ValidationError as exc:
                    st.error(str(exc))
    if equipment:
        with st.expander("Update equipment condition"):
            equipment_id = equipment_selector(equipment, key="condition_equipment")
            current = store.get_equipment(equipment_id)
            with st.form(f"condition_form_{equipment_id}"):
                condition = st.selectbox("New condition", EQUIPMENT_STATUSES, index=EQUIPMENT_STATUSES.index(current["operational_status"]))
                if st.form_submit_button("Save condition"):
                    store.update_equipment_status(equipment_id, condition)
                    saved("Equipment condition updated.")


def create_issue_page(store: MaintenanceStore) -> None:
    equipment = store.list_equipment()
    if not equipment:
        st.info("Register equipment or load synthetic demo data before creating an issue.")
        return
    with st.form("create_issue", clear_on_submit=True):
        equipment_id = equipment_selector(equipment, key="new_issue_equipment")
        description = st.text_area("Description *", max_chars=10000)
        left, right = st.columns(2)
        priority = left.selectbox("Priority", PRIORITIES, index=1)
        status = right.selectbox("Status", ISSUE_STATUSES)
        reporter = st.text_input("Reporter *", max_chars=200)
        st.caption("Created and updated timestamps are recorded automatically in UTC.")
        if st.form_submit_button("Create issue"):
            try:
                issue_id = store.create_issue(equipment_id=equipment_id, description=description, priority=priority, status=status, reporter=reporter)
                saved(f"Issue #{issue_id} created.")
            except ValidationError as exc:
                st.error(str(exc))


def issues_page(store: MaintenanceStore) -> None:
    equipment = store.list_equipment()
    if not equipment:
        st.info("No equipment registered yet. Add equipment or load synthetic demo data.")
        return
    left, middle, right = st.columns(3)
    with left:
        equipment_id = equipment_selector(equipment, key="issue_filter_equipment", allow_all=True)
    status = middle.selectbox("Status filter", ("All statuses",) + ISSUE_STATUSES)
    priority = right.selectbox("Priority filter", ("All priorities",) + PRIORITIES)
    search = st.text_input("Search description or reporter")
    records = store.list_issues(equipment_id=equipment_id, status=None if status == "All statuses" else status,
                                priority=None if priority == "All priorities" else priority, search=search)
    st.caption(f"{len(records)} matching issue(s). Timestamps are UTC.")
    table(records, ["id", "asset_tag", "description", "priority", "status", "reporter", "created_at", "updated_at"])
    if not records:
        return
    labels = {row["id"]: f"#{row['id']} · {row['asset_tag']} · {row['description'][:80]}" for row in records}
    issue_id = st.selectbox("Issue to review or update", list(labels), format_func=labels.get)
    issue = store.get_issue(issue_id)
    st.subheader(f"Issue #{issue_id}")
    st.text(issue["description"])
    st.caption(f"Reporter: {issue['reporter']} | Priority: {issue['priority']} | Created: {issue['created_at']}")
    with st.form(f"update_issue_{issue_id}"):
        status = st.selectbox("Issue status", ISSUE_STATUSES, index=ISSUE_STATUSES.index(issue["status"]))
        notes = st.text_area("Troubleshooting notes", value=issue["troubleshooting_notes"], max_chars=10000)
        cause = st.text_area("Root cause", value=issue["root_cause"], max_chars=10000)
        action = st.text_area("Corrective action", value=issue["corrective_action"], max_chars=10000)
        if st.form_submit_button("Save issue updates"):
            try:
                store.update_issue(issue_id, status=status, troubleshooting_notes=notes, root_cause=cause, corrective_action=action)
                saved(f"Issue #{issue_id} updated.")
            except ValidationError as exc:
                st.error(str(exc))


def history_page(store: MaintenanceStore) -> None:
    equipment = store.list_equipment()
    if not equipment:
        st.info("No equipment registered yet.")
        return
    equipment_id = equipment_selector(equipment, key="history_equipment")
    item = store.get_equipment(equipment_id)
    st.write(f"**{item['name']}** · {item['operational_status']}")
    st.caption(f"{item['category']} | {item['manufacturer']} | {item['model']} | {item['location']}")
    history = store.equipment_history(equipment_id)
    table(history, ["id", "description", "priority", "status", "reporter", "created_at", "updated_at"])
    for issue in history:
        with st.expander(f"Issue #{issue['id']} · {issue['status']} · {issue['created_at']}"):
            st.text(issue["description"])
            for label, field in (("Troubleshooting notes", "troubleshooting_notes"), ("Root cause", "root_cause"), ("Corrective action", "corrective_action")):
                st.write(f"**{label}**")
                st.text(issue[field] or "Not recorded")


def import_page(store: MaintenanceStore) -> None:
    st.write("Import equipment first, then issues linked by asset tag. Files must use UTF-8 encoding, exact column names, and the listed status and priority values.")
    st.caption("Maximum 5 MiB / 10,000 rows. An invalid file makes no changes. Imports add records; they do not update existing rows. Reimporting issues creates additional issues.")
    kind = st.radio("Import type", ("equipment", "issues"), horizontal=True)
    sample = (SAMPLE_DIR / f"{kind}.csv").read_bytes()
    st.download_button("Download synthetic example CSV", data=sample, file_name=f"synthetic_{kind}.csv", mime="text/csv")
    st.code(sample.decode("utf-8").splitlines()[0], language=None)
    st.caption("Equipment conditions: " + ", ".join(EQUIPMENT_STATUSES))
    st.caption("Issue priorities: " + ", ".join(PRIORITIES) + ". Issue statuses: " + ", ".join(ISSUE_STATUSES))
    uploaded = st.file_uploader("Choose CSV", type=["csv"], key=f"upload_{kind}")
    if uploaded is not None:
        st.write(f"Selected: {uploaded.name}")
        if st.button("Validate and import", type="primary"):
            try:
                count = import_csv(store, uploaded.getvalue(), kind)
                saved(f"Imported {count} {kind} record(s).")
            except CSVImportError as exc:
                st.error("Import rejected. No records were added.")
                for error in exc.errors:
                    st.text(error)
