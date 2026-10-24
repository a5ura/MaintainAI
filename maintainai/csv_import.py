"""Strict UTF-8 CSV validation and atomic imports for equipment and issues."""

import csv
import io

from maintainai.service import MaintenanceStore, ValidationError, validate_equipment, validate_issue

EQUIPMENT_REQUIRED = {"asset_tag", "name", "category"}
EQUIPMENT_OPTIONAL = {"manufacturer", "model", "location", "operational_status", "created_at"}
ISSUE_REQUIRED = {"equipment_asset_tag", "description", "priority", "status", "reporter"}
ISSUE_OPTIONAL = {"created_at", "troubleshooting_notes", "root_cause", "corrective_action"}
MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10000
# Match SQLite's built-in NOCASE collation, which folds ASCII letters only.
_NOCASE_TRANSLATION = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


class CSVImportError(ValidationError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def import_csv(store: MaintenanceStore, content: bytes | str, kind: str) -> int:
    if kind not in ("equipment", "issues"):
        raise CSVImportError(["Import type must be equipment or issues."])
    try:
        if isinstance(content, bytes):
            if len(content) > MAX_BYTES:
                raise CSVImportError(["CSV exceeds the 5 MiB size limit."])
            content = content.decode("utf-8-sig")
        elif len(content.encode("utf-8")) > MAX_BYTES:
            raise CSVImportError(["CSV exceeds the 5 MiB size limit."])
    except UnicodeDecodeError as exc:
        raise CSVImportError(["Use a UTF-8 encoded CSV file (Excel: CSV UTF-8)."]) from exc
    content = content.lstrip("\ufeff")
    if "\x00" in content:
        raise CSVImportError(["CSV contains null characters. Export it as UTF-8 text."])
    reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
    required = EQUIPMENT_REQUIRED if kind == "equipment" else ISSUE_REQUIRED
    optional = EQUIPMENT_OPTIONAL if kind == "equipment" else ISSUE_OPTIONAL
    errors, records = [], []
    try:
        headers = reader.fieldnames
        if not headers:
            raise CSVImportError(["CSV is empty. Include a header row and at least one data row."])
        if len(set(headers)) != len(headers):
            raise CSVImportError(["CSV contains duplicate column names."])
        if required - set(headers):
            errors.append("Missing required columns: " + ", ".join(sorted(required - set(headers))))
        if set(headers) - required - optional:
            errors.append("Unknown columns: " + ", ".join(sorted(set(headers) - required - optional)))
        if errors:
            raise CSVImportError(errors)
        equipment = {item["asset_tag"].translate(_NOCASE_TRANSLATION): item["id"] for item in store.list_equipment()}
        seen = set(equipment)
        for row_number, row in enumerate(reader, start=2):
            if row_number > MAX_ROWS + 1:
                raise CSVImportError([f"CSV exceeds the {MAX_ROWS:,} row limit."])
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValidationError("Column count does not match the header; quote descriptions containing commas.")
                row = {key: value.strip() for key, value in row.items()}
                # Blank optional cells use the same defaults as omitted columns.
                row = {key: value for key, value in row.items() if value or key in required}
                if kind == "equipment":
                    validated = validate_equipment(row)
                    tag = validated["asset_tag"].translate(_NOCASE_TRANSLATION)
                    if tag in seen:
                        raise ValidationError(f"asset_tag '{validated['asset_tag']}' already exists or is duplicated in this file.")
                    seen.add(tag)
                else:
                    tag = row.pop("equipment_asset_tag")
                    lookup = tag.translate(_NOCASE_TRANSLATION)
                    if lookup not in equipment:
                        raise ValidationError(f"Unknown equipment_asset_tag '{tag}'. Import or create equipment first.")
                    row["equipment_id"] = equipment[lookup]
                    validated = validate_issue(row)
                records.append(validated)
            except ValidationError as exc:
                errors.append(f"Row {row_number}: {exc}")
                if len(errors) >= 25:
                    errors.append("Stopped after 25 errors. Correct these and try again.")
                    break
    except csv.Error as exc:
        raise CSVImportError([f"Malformed CSV near line {reader.line_num}: {exc}"]) from exc
    if errors:
        raise CSVImportError(errors)
    if not records:
        raise CSVImportError(["CSV contains no data rows."])
    try:
        if kind == "equipment":
            store.import_equipment_records(records)
        else:
            store.import_issue_records(records)
    except ValidationError as exc:
        raise CSVImportError([str(exc)]) from exc
    return len(records)
