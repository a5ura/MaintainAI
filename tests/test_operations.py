from datetime import datetime
import sqlite3

import pytest

from maintainai.database import connect, initialize
from maintainai.seed import seed_demo
from maintainai.service import ISSUE_STATUSES, PRIORITIES, MaintenanceStore, ValidationError


def test_equipment_and_issue_survive_reopen(store, equipment_id, db_path):
    issue_id = store.create_issue(equipment_id=equipment_id, description="Test fault", reporter="Demo operator", priority="High")
    with connect(db_path) as connection:
        initialize(connection)
        reopened = MaintenanceStore(connection)
        assert reopened.get_equipment(equipment_id)["name"] == "Test robot"
        issue = reopened.get_issue(issue_id)
        assert issue["description"] == "Test fault"
        assert issue["status"] == "Open"
        assert datetime.fromisoformat(issue["created_at"]).utcoffset().total_seconds() == 0
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


@pytest.mark.parametrize("field", ["asset_tag", "name", "category"])
def test_equipment_requires_fields(store, field):
    record = dict(asset_tag="TEST", name="Robot", category="Robotics")
    record[field] = "  "
    with pytest.raises(ValidationError, match=field):
        store.create_equipment(**record)
    assert store.list_equipment() == []


def test_duplicate_asset_tags_ignore_case(store, equipment_id):
    with pytest.raises(ValidationError, match="unique"):
        store.create_equipment(asset_tag="test-001", name="Duplicate", category="Test")
    assert len(store.list_equipment()) == 1


def test_equipment_batch_rolls_back_on_database_conflict(store, equipment_id):
    with pytest.raises(ValidationError):
        store.import_equipment_records([
            dict(asset_tag="NEW", name="New", category="Test"),
            dict(asset_tag="TEST-001", name="Duplicate", category="Test"),
        ])
    assert len(store.list_equipment()) == 1


@pytest.mark.parametrize("field,value", [("description", " "), ("reporter", ""), ("priority", "Urgent"), ("status", "Done"), ("equipment_id", 999), ("equipment_id", True)])
def test_invalid_issues_are_rejected(store, equipment_id, field, value):
    record = dict(equipment_id=equipment_id, description="Test", reporter="Operator")
    record[field] = value
    with pytest.raises(ValidationError):
        store.create_issue(**record)
    assert store.list_issues() == []


@pytest.mark.parametrize("priority", PRIORITIES)
def test_priorities_are_supported(store, equipment_id, priority):
    issue_id = store.create_issue(equipment_id=equipment_id, description="Test", reporter="Operator", priority=priority)
    assert store.get_issue(issue_id)["priority"] == priority


def test_update_preserves_issue_identity_and_timestamp(store, equipment_id):
    issue_id = store.create_issue(equipment_id=equipment_id, description="Test", reporter="Operator", created_at="2026-01-01T08:00:00Z")
    before = store.get_issue(issue_id)
    store.update_issue(issue_id, status="Closed", troubleshooting_notes="Checked sensor", root_cause="Loose cable", corrective_action="Secured cable")
    after = store.get_issue(issue_id)
    assert after["status"] == "Closed"
    assert after["root_cause"] == "Loose cable"
    assert after["corrective_action"] == "Secured cable"
    assert after["troubleshooting_notes"] == "Checked sensor"
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] > before["updated_at"]
    assert after["equipment_id"] == equipment_id


def test_status_only_update_preserves_findings_and_persists(store, equipment_id, db_path):
    issue_id = store.create_issue(
        equipment_id=equipment_id, description="Synthetic fault", reporter="Demo tester",
        troubleshooting_notes="Inspected sensor", root_cause="Loose connector",
        corrective_action="Secured connector",
    )
    before = store.get_issue(issue_id)
    store.update_issue(issue_id, status="Closed")
    with connect(db_path) as connection:
        reopened = MaintenanceStore(connection)
        issue = reopened.get_issue(issue_id)
        assert issue["status"] == "Closed"
        for field in ("troubleshooting_notes", "root_cause", "corrective_action", "created_at"):
            assert issue[field] == before[field]
        assert reopened.equipment_history(equipment_id)[0]["id"] == issue_id
        assert reopened.dashboard()["closed_issues"] == 1
    # Explicit blank text still clears a field; reopening retains other findings.
    store.update_issue(issue_id, status="Open", root_cause="")
    issue = store.get_issue(issue_id)
    assert issue["root_cause"] == ""
    assert issue["troubleshooting_notes"] == "Inspected sensor"
    assert issue["corrective_action"] == "Secured connector"


def test_invalid_update_does_not_change_issue(store, equipment_id):
    issue_id = store.create_issue(equipment_id=equipment_id, description="Test", reporter="Operator")
    before = store.get_issue(issue_id)
    with pytest.raises(ValidationError):
        store.update_issue(issue_id, status="Invalid", root_cause="Should not persist")
    assert store.get_issue(issue_id) == before
    with pytest.raises(ValidationError, match="does not exist"):
        store.update_issue(999, status="Closed")


def test_combined_filters_and_history_are_scoped(store, equipment_id):
    second = store.create_equipment(asset_tag="OTHER", name="Conveyor", category="Transport")
    first_issue = store.create_issue(equipment_id=equipment_id, description="Vacuum fault 100%", reporter="Alex", priority="High")
    store.create_issue(equipment_id=equipment_id, description="Routine check", reporter="Sam", status="Closed")
    store.create_issue(equipment_id=second, description="Vacuum fault", reporter="Alex", priority="High")
    results = store.list_issues(equipment_id=equipment_id, priority="High", status="Open", search="alex")
    assert [row["id"] for row in results] == [first_issue]
    assert len(store.equipment_history(equipment_id)) == 2
    assert len(store.equipment_history(second)) == 1
    assert len(store.list_issues(search="100%")) == 1
    assert store.list_issues(search="' OR 1=1 --") == []
    with pytest.raises(ValidationError):
        store.equipment_history(999)


def test_dashboard_counts_all_statuses_and_equipment_conditions(store, equipment_id):
    empty = store.dashboard()
    assert empty["total_issues"] == 0
    for status in ISSUE_STATUSES:
        store.create_issue(equipment_id=equipment_id, description=status, reporter="Operator", status=status)
    store.update_equipment_status(equipment_id, "Maintenance")
    counts = store.dashboard()
    assert counts["total_equipment"] == 1
    assert counts["total_issues"] == 4
    assert counts["open_issues"] == 2
    assert counts["closed_issues"] == 2
    assert counts["equipment_by_status"] == {"Operational": 0, "Maintenance": 1, "Offline": 0}
    with pytest.raises(ValidationError):
        store.update_equipment_status(equipment_id, "Broken")


def test_database_constraints_enforce_relationships(store, equipment_id):
    store.create_issue(equipment_id=equipment_id, description="Test", reporter="Operator")
    with pytest.raises(sqlite3.IntegrityError), store.connection:
        store.connection.execute("DELETE FROM equipment WHERE id=?", (equipment_id,))
    assert store.get_equipment(equipment_id)


def test_repeatable_seed_preserves_edited_and_user_data(store):
    assert seed_demo(store.connection) == {"equipment_added": 5, "issues_added": 5}
    equipment = store.list_equipment()
    assert {item["name"] for item in equipment} == {"ABB robotic arm", "Vacuum gripper", "Zaber gantry", "Conveyor", "Machine vision station"}
    issue_id = store.list_issues()[0]["id"]
    store.update_issue(issue_id, status="Closed", root_cause="User edited this")
    store.create_issue(equipment_id=equipment[0]["id"], description="User issue", reporter="User")
    assert seed_demo(store.connection) == {"equipment_added": 0, "issues_added": 0}
    assert store.get_issue(issue_id)["root_cause"] == "User edited this"
    assert store.dashboard()["total_issues"] == 6


def test_seed_transaction_rolls_back_if_fixture_is_invalid(store, tmp_path):
    (tmp_path / "equipment.csv").write_text("asset_tag,name,category\nNEW,Robot,Robotics\n", encoding="utf-8")
    (tmp_path / "issues.csv").write_text("equipment_asset_tag,description,reporter,priority\nNEW,Test,Demo,Invalid\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        seed_demo(store.connection, tmp_path)
    assert store.list_equipment() == []
