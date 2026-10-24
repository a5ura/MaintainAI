"""SQLite configuration and schema. Connections are short-lived and never shared."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_db_path() -> Path:
    return Path(os.environ.get("MAINTAINAI_DB_PATH", PROJECT_ROOT / "data" / "maintainai.db"))


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    target = Path(path) if path is not None else default_db_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Keep path/permission failures inside the app's database error handling.
        raise sqlite3.OperationalError(f"Cannot create database directory: {target.parent}") from exc
    connection = sqlite3.connect(target, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def initialize(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY,
            asset_tag TEXT NOT NULL UNIQUE COLLATE NOCASE CHECK(length(trim(asset_tag)) > 0),
            name TEXT NOT NULL CHECK(length(trim(name)) > 0),
            category TEXT NOT NULL CHECK(length(trim(category)) > 0),
            manufacturer TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            operational_status TEXT NOT NULL DEFAULT 'Operational'
                CHECK(operational_status IN ('Operational', 'Maintenance', 'Offline')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY,
            equipment_id INTEGER NOT NULL REFERENCES equipment(id),
            description TEXT NOT NULL CHECK(length(trim(description)) > 0),
            priority TEXT NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
            status TEXT NOT NULL CHECK(status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
            reporter TEXT NOT NULL CHECK(length(trim(reporter)) > 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            troubleshooting_notes TEXT NOT NULL DEFAULT '',
            root_cause TEXT NOT NULL DEFAULT '',
            corrective_action TEXT NOT NULL DEFAULT '',
            seed_key TEXT UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_issues_equipment ON issues(equipment_id);
        CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
    """)
