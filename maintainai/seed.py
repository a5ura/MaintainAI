"""Repeatable synthetic fixture loading; existing records are preserved."""

import csv
from pathlib import Path
import sqlite3

from maintainai.database import PROJECT_ROOT
from maintainai.service import EQUIPMENT_INSERT, validate_equipment, validate_issue

SAMPLE_DIR = PROJECT_ROOT / "sample_data"


def seed_demo(connection: sqlite3.Connection, sample_dir: Path = SAMPLE_DIR) -> dict[str, int]:
    equipment_added, issues_added = 0, 0
    with connection:
        with (sample_dir / "equipment.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                record = validate_equipment(row)
                cursor = connection.execute(EQUIPMENT_INSERT + " ON CONFLICT(asset_tag) DO NOTHING", record)
                equipment_added += cursor.rowcount
        with (sample_dir / "issues.csv").open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                tag = row.pop("equipment_asset_tag")
                row["equipment_id"] = connection.execute("SELECT id FROM equipment WHERE asset_tag=?", (tag,)).fetchone()["id"]
                record = validate_issue(row)
                record["seed_key"] = f"synthetic-demo-{index:03}"
                # A stable key prevents duplicate fixtures even after users edit them.
                cursor = connection.execute("""INSERT INTO issues
                    (equipment_id, description, priority, status, reporter, created_at, updated_at,
                     troubleshooting_notes, root_cause, corrective_action, seed_key)
                    VALUES (:equipment_id, :description, :priority, :status, :reporter, :created_at, :updated_at,
                            :troubleshooting_notes, :root_cause, :corrective_action, :seed_key)
                    ON CONFLICT(seed_key) DO NOTHING""", record)
                issues_added += cursor.rowcount
    return {"equipment_added": equipment_added, "issues_added": issues_added}
