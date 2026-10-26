"""Validated maintenance operations, independent of Streamlit."""

from datetime import datetime, timezone
import sqlite3
from typing import Any

PRIORITIES = ("Low", "Medium", "High", "Critical")
ISSUE_STATUSES = ("Open", "In Progress", "Resolved", "Closed")
EQUIPMENT_STATUSES = ("Operational", "Maintenance", "Offline")
ACTIVE_STATUSES = ("Open", "In Progress")
LOG_SEVERITIES = ('INFO', 'WARNING', 'ERROR', 'CRITICAL')


class ValidationError(ValueError):
    """An actionable input error safe to display in the UI."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def clean_text(value: Any, field: str, *, required: bool = False, limit: int = 10000) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text.")
    value = value.strip()
    if required and not value:
        raise ValidationError(f"{field} is required.")
    if len(value) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters.")
    return value


def choice(value: Any, field: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ValidationError(f"{field} must be one of: {', '.join(options)}.")
    return value


def timestamp(value: str | None = None, *, field: str = "created_at") -> str:
    if value is None or value == "":
        return utc_now()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("missing timezone")
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise ValidationError(f"{field} must be an ISO 8601 timestamp with a timezone and a UTC date within years 1 through 9999, e.g. 2026-01-15T09:00:00Z.") from exc


def validate_equipment(record: dict) -> dict:
    result = {
        field: clean_text(record.get(field), field, required=field in ("asset_tag", "name", "category"), limit=200)
        for field in ("asset_tag", "name", "category", "manufacturer", "model", "location")
    }
    result["operational_status"] = choice(record.get("operational_status", "Operational"), "operational_status", EQUIPMENT_STATUSES)
    result["created_at"] = timestamp(record.get("created_at"))
    return result


def validate_issue(record: dict) -> dict:
    equipment_id = record.get("equipment_id")
    if type(equipment_id) is not int or equipment_id <= 0:
        raise ValidationError("Select a valid equipment ID.")
    result = {"equipment_id": equipment_id}
    for field in ("description", "reporter", "troubleshooting_notes", "root_cause", "corrective_action"):
        result[field] = clean_text(record.get(field), field, required=field in ("description", "reporter"), limit=200 if field == "reporter" else 10000)
    result["priority"] = choice(record.get("priority", "Medium"), "priority", PRIORITIES)
    result["status"] = choice(record.get("status", "Open"), "status", ISSUE_STATUSES)
    result["created_at"] = timestamp(record.get("created_at"))
    result["updated_at"] = result["created_at"]
    return result


def validate_log(record: dict) -> dict:
    """Validate an equipment event without inventing a missing event timestamp."""
    equipment_id = record.get("equipment_id")
    if type(equipment_id) is not int or equipment_id <= 0:
        raise ValidationError("equipment_id must be a positive integer.")
    event_timestamp = clean_text(record.get("timestamp"), "timestamp", required=True)
    result = {
        "equipment_id": equipment_id,
        "timestamp": timestamp(event_timestamp, field="timestamp"),
        "severity": choice(record.get("severity"), "severity", LOG_SEVERITIES),
        "message": clean_text(record.get("message"), "message", required=True, limit=10000),
        "event_code": clean_text(record.get("event_code"), "event_code", limit=200),
        "source": clean_text(record.get("source"), "source", limit=200),
    }
    result["imported_at"] = utc_now()
    return result


EQUIPMENT_INSERT = """INSERT INTO equipment
    (asset_tag, name, category, manufacturer, model, location, operational_status, created_at)
    VALUES (:asset_tag, :name, :category, :manufacturer, :model, :location, :operational_status, :created_at)"""
ISSUE_INSERT = """INSERT INTO issues
    (equipment_id, description, priority, status, reporter, created_at, updated_at,
     troubleshooting_notes, root_cause, corrective_action)
    VALUES (:equipment_id, :description, :priority, :status, :reporter, :created_at, :updated_at,
            :troubleshooting_notes, :root_cause, :corrective_action)"""
LOG_INSERT = """INSERT INTO equipment_logs
    (equipment_id, timestamp, severity, event_code, message, source, imported_at)
    VALUES (:equipment_id, :timestamp, :severity, :event_code, :message, :source, :imported_at)"""


class MaintenanceStore:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def list_equipment(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM equipment ORDER BY asset_tag COLLATE NOCASE")]

    def get_equipment(self, equipment_id: int) -> dict:
        row = self.connection.execute("SELECT * FROM equipment WHERE id = ?", (equipment_id,)).fetchone()
        if row is None:
            raise ValidationError(f"Equipment {equipment_id} does not exist.")
        return dict(row)

    def create_equipment(self, **record: Any) -> int:
        return self.import_equipment_records([record])[0]

    def import_equipment_records(self, records: list[dict]) -> list[int]:
        validated = [validate_equipment(record) for record in records]
        try:
            with self.connection:
                return [self.connection.execute(EQUIPMENT_INSERT, record).lastrowid for record in validated]
        except sqlite3.IntegrityError as exc:
            raise ValidationError("Asset tags must be unique (case-insensitive). No equipment was imported.") from exc

    def create_issue(self, **record: Any) -> int:
        return self.import_issue_records([record])[0]

    def import_issue_records(self, records: list[dict]) -> list[int]:
        validated = [validate_issue(record) for record in records]
        for record in validated:
            self.get_equipment(record["equipment_id"])
        try:
            with self.connection:
                return [self.connection.execute(ISSUE_INSERT, record).lastrowid for record in validated]
        except sqlite3.IntegrityError as exc:
            raise ValidationError("Issue import failed database validation. No issues were imported.") from exc

    def create_log(self, **record: Any) -> int:
        return self.import_log_records([record])[0]

    def import_log_records(self, records: list[dict]) -> list[int]:
        validated = [validate_log(record) for record in records]
        try:
            with self.connection:
                for equipment_id in dict.fromkeys(record["equipment_id"] for record in validated):
                    self.get_equipment(equipment_id)
                return [self.connection.execute(LOG_INSERT, record).lastrowid for record in validated]
        except sqlite3.IntegrityError as exc:
            raise ValidationError(
                "Equipment log import failed database integrity checks. Verify that referenced equipment "
                "still exists and log fields meet the table constraints. No logs were imported."
            ) from exc

    def list_logs(self, equipment_id: int | None = None, severity: str | None = None) -> list[dict]:
        conditions, params = [], []
        if equipment_id is not None:
            if type(equipment_id) is not int or equipment_id <= 0:
                raise ValidationError("equipment_id must be a positive integer.")
            self.get_equipment(equipment_id)
            conditions.append("l.equipment_id = ?")
            params.append(equipment_id)
        if severity is not None:
            choice(severity, "severity", LOG_SEVERITIES)
            conditions.append("l.severity = ?")
            params.append(severity)
        query = """SELECT l.*, e.asset_tag, e.name AS equipment_name FROM equipment_logs l
                   JOIN equipment e ON e.id = l.equipment_id"""
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY l.timestamp DESC, l.id DESC"
        return [dict(row) for row in self.connection.execute(query, params)]

    def list_issues(self, *, equipment_id: int | None = None, status: str | None = None, priority: str | None = None, search: str = "") -> list[dict]:
        conditions, params = [], []
        for column, value in (("equipment_id", equipment_id), ("status", status), ("priority", priority)):
            if value is not None:
                conditions.append(f"i.{column} = ?")
                params.append(value)
        if search.strip():
            # instr treats %, _, quotes and backslashes as literal search text.
            conditions.append("instr(lower(i.description || ' ' || i.reporter), lower(?)) > 0")
            params.append(search.strip())
        query = """SELECT i.*, e.asset_tag, e.name AS equipment_name FROM issues i
                   JOIN equipment e ON e.id = i.equipment_id"""
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY i.created_at DESC, i.id DESC"
        return [dict(row) for row in self.connection.execute(query, params)]

    def get_issue(self, issue_id: int) -> dict:
        row = self.connection.execute("SELECT * FROM issues WHERE id = ?", (issue_id,)).fetchone()
        if row is None:
            raise ValidationError(f"Issue {issue_id} does not exist.")
        return dict(row)

    def update_issue(self, issue_id: int, *, status: str, troubleshooting_notes: str | None = None, root_cause: str | None = None, corrective_action: str | None = None) -> None:
        """Preserve omitted findings; explicitly pass empty text to clear a field."""
        current = self.get_issue(issue_id)
        values = {field: current[field] if value is None else clean_text(value, field) for field, value in {
            "troubleshooting_notes": troubleshooting_notes,
            "root_cause": root_cause, "corrective_action": corrective_action,
        }.items()}
        values.update(status=choice(status, "status", ISSUE_STATUSES), updated_at=utc_now(), id=issue_id)
        with self.connection:
            self.connection.execute("""UPDATE issues SET status=:status,
                troubleshooting_notes=:troubleshooting_notes, root_cause=:root_cause,
                corrective_action=:corrective_action, updated_at=:updated_at WHERE id=:id""", values)

    def update_equipment_status(self, equipment_id: int, status: str) -> None:
        self.get_equipment(equipment_id)
        choice(status, "operational_status", EQUIPMENT_STATUSES)
        with self.connection:
            self.connection.execute("UPDATE equipment SET operational_status=? WHERE id=?", (status, equipment_id))

    def equipment_history(self, equipment_id: int) -> list[dict]:
        self.get_equipment(equipment_id)
        return self.list_issues(equipment_id=equipment_id)

    def dashboard(self) -> dict:
        equipment = self.list_equipment()
        counts = {status: 0 for status in ISSUE_STATUSES}
        for row in self.connection.execute("SELECT status, COUNT(*) AS total FROM issues GROUP BY status"):
            counts[row["status"]] = row["total"]
        return {
            "total_equipment": len(equipment),
            "total_issues": sum(counts.values()),
            "open_issues": sum(counts[status] for status in ACTIVE_STATUSES),
            "closed_issues": counts["Resolved"] + counts["Closed"],
            "issues_by_status": counts,
            "equipment_by_status": {status: sum(e["operational_status"] == status for e in equipment) for status in EQUIPMENT_STATUSES},
        }
