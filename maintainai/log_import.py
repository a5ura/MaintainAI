"""UTF-8 equipment-log CSV validation and atomic batch ingestion."""

import csv
import io

from maintainai.csv_import import CSVImportError, MAX_BYTES, MAX_ROWS, _NOCASE_TRANSLATION
from maintainai.service import MaintenanceStore, ValidationError, validate_log

REQUIRED_COLUMNS = {"equipment_asset_tag", "timestamp", "severity", "message"}
OPTIONAL_COLUMNS = {"event_code", "source"}


def import_logs_csv(store: MaintenanceStore, content: bytes | str) -> int:
    try:
        if isinstance(content, bytes):
            if len(content) > MAX_BYTES:
                raise CSVImportError(["CSV exceeds the 5 MiB size limit."])
            content = content.decode("utf-8-sig")
        elif isinstance(content, str):
            if len(content.encode("utf-8")) > MAX_BYTES:
                raise CSVImportError(["CSV exceeds the 5 MiB size limit."])
        else:
            raise CSVImportError(["CSV content must be UTF-8 bytes or text."])
    except (UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise CSVImportError(["Use a UTF-8 encoded CSV file (Excel: CSV UTF-8)."]) from exc
    content = content.lstrip("\ufeff")
    if "\x00" in content:
        raise CSVImportError(["CSV contains null characters. Export it as UTF-8 text."])
    reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
    errors, records = [], []
    try:
        headers = reader.fieldnames
        if not headers:
            raise CSVImportError(["CSV is empty. Include a header row and at least one data row."])
        if len(set(headers)) != len(headers):
            raise CSVImportError(["CSV contains duplicate column names."])
        missing = REQUIRED_COLUMNS - set(headers)
        unknown = set(headers) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS
        if missing:
            errors.append("Missing required columns: " + ", ".join(sorted(missing)))
        if unknown:
            errors.append("Unknown columns: " + ", ".join(column or "<empty>" for column in sorted(unknown)))
        if errors:
            raise CSVImportError(errors)
        equipment = {item["asset_tag"].translate(_NOCASE_TRANSLATION): item["id"] for item in store.list_equipment()}
        for row_number, row in enumerate(reader, start=2):
            if row_number > MAX_ROWS + 1:
                raise CSVImportError([f"CSV exceeds the {MAX_ROWS:,} row limit."])
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValidationError("Column count does not match the header; quote messages containing commas.")
                row = {key: value.strip() for key, value in row.items()}
                tag = row.pop("equipment_asset_tag")
                if not tag:
                    raise ValidationError("equipment_asset_tag is required.")
                lookup = tag.translate(_NOCASE_TRANSLATION)
                if lookup not in equipment:
                    raise ValidationError(f"Unknown equipment_asset_tag '{tag}'. Import or create equipment first.")
                row["equipment_id"] = equipment[lookup]
                records.append(validate_log(row))
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
        store.import_log_records(records)
    except ValidationError as exc:
        raise CSVImportError([str(exc)]) from exc
    return len(records)
