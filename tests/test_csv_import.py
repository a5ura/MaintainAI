import pytest

from maintainai.csv_import import CSVImportError, import_csv
from maintainai.seed import SAMPLE_DIR

EQUIPMENT_HEADER = "asset_tag,name,category\n"
ISSUE_HEADER = "equipment_asset_tag,description,priority,status,reporter\n"


def test_sample_files_import_in_order(store):
    assert import_csv(store, (SAMPLE_DIR / "equipment.csv").read_bytes(), "equipment") == 5
    assert import_csv(store, (SAMPLE_DIR / "issues.csv").read_bytes(), "issues") == 5
    assert store.dashboard()["open_issues"] == 3
    assert store.dashboard()["closed_issues"] == 2


def test_bom_quoted_commas_and_blank_optional_defaults(store):
    content = '\ufeffasset_tag,name,category,operational_status\r\nA,"Robot, demo",Robotics,\r\n'
    assert import_csv(store, content.encode("utf-8"), "equipment") == 1
    equipment = store.list_equipment()[0]
    assert equipment["name"] == "Robot, demo"
    assert equipment["operational_status"] == "Operational"


@pytest.mark.parametrize("content,error", [
    ("", "empty"),
    (EQUIPMENT_HEADER, "no data"),
    ("asset_tag,name\nA,Test\n", "Missing required columns: category"),
    ("asset_tag,name,category,typo\nA,Test,Test,x\n", "Unknown columns: typo"),
    ("asset_tag,name,name,category\nA,Test,Test,Test\n", "duplicate column"),
    (EQUIPMENT_HEADER + "A,Test\n", "Row 2: Column count"),
    (EQUIPMENT_HEADER + "A,Test,Test,Extra\n", "Row 2: Column count"),
    (EQUIPMENT_HEADER + 'A,"unclosed,Test\n', "Malformed CSV"),
    (EQUIPMENT_HEADER + "A,,Test\n", "Row 2: name is required"),
    (EQUIPMENT_HEADER + "A,Test,Test\nA,Duplicate,Test\n", "Row 3:"),
    (EQUIPMENT_HEADER + "A,Test,Test\na,Duplicate,Test\n", "Row 3:"),
    (EQUIPMENT_HEADER + "A,\x00,Test\n", "null characters"),
    (b"asset_tag,name,category\nA,\xff,Test", "UTF-8"),
])
def test_invalid_equipment_csv_is_atomic(store, content, error):
    with pytest.raises(CSVImportError, match=error):
        import_csv(store, content, "equipment")
    assert store.list_equipment() == []


def test_existing_tag_error_identifies_row(store, equipment_id):
    with pytest.raises(CSVImportError, match="Row 3:.*already exists"):
        import_csv(store, EQUIPMENT_HEADER + "NEW,New,Test\ntest-001,Duplicate,Test\n", "equipment")
    assert len(store.list_equipment()) == 1


@pytest.mark.parametrize("bad_row,error", [
    ("UNKNOWN,Fault,High,Open,Demo", "Unknown equipment_asset_tag"),
    ("TEST-001,Fault,Urgent,Open,Demo", "priority must be"),
    ("TEST-001,Fault,High,Done,Demo", "status must be"),
    ("TEST-001,,High,Open,Demo", "description is required"),
    ("TEST-001,Fault,High,Open,", "reporter is required"),
])
def test_invalid_issue_file_rolls_back_valid_rows(store, equipment_id, bad_row, error):
    content = ISSUE_HEADER + "TEST-001,Valid fault,Medium,Open,Demo\n" + bad_row + "\n"
    with pytest.raises(CSVImportError, match=f"Row 3:.*{error}"):
        import_csv(store, content, "issues")
    assert store.list_issues() == []


@pytest.mark.parametrize("created_at", ["yesterday", "2026-01-01T10:00:00", "2026-99-99T00:00:00Z"])
def test_import_rejects_invalid_or_ambiguous_timestamps(store, equipment_id, created_at):
    content = ISSUE_HEADER.rstrip() + ",created_at\nTEST-001,Fault,Medium,Open,Demo," + created_at + "\n"
    with pytest.raises(CSVImportError, match="ISO 8601"):
        import_csv(store, content, "issues")


def test_timestamp_normalized_to_utc(store, equipment_id):
    content = ISSUE_HEADER.rstrip() + ",created_at\ntest-001,Fault,High,Open,Demo,2026-01-01T12:00:00+02:00\n"
    assert import_csv(store, content, "issues") == 1
    assert store.list_issues()[0]["created_at"] == "2026-01-01T10:00:00.000000+00:00"


@pytest.mark.parametrize("created_at", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_timestamp_utc_overflow_is_a_validation_error(store, equipment_id, created_at):
    content = ISSUE_HEADER.rstrip() + ",created_at\nTEST-001,Fault,Medium,Open,Demo," + created_at + "\n"
    with pytest.raises(CSVImportError, match="Row 2:.*ISO 8601"):
        import_csv(store, content, "issues")
    assert store.list_issues() == []


def test_issue_asset_matching_uses_sqlite_nocase_rules(store):
    # SQLite NOCASE folds ASCII letters only; these are distinct existing assets.
    first = store.create_equipment(asset_tag="STRASSE", name="First", category="Test")
    second = store.create_equipment(asset_tag="Stra\u00dfe", name="Second", category="Test")
    content = ISSUE_HEADER + "strasse,First asset fault,High,Open,Demo\nStra\u00dfe,Second asset fault,Low,Open,Demo\n"
    assert import_csv(store, content, "issues") == 2
    assert [row["description"] for row in store.equipment_history(first)] == ["First asset fault"]
    assert [row["description"] for row in store.equipment_history(second)] == ["Second asset fault"]


def test_equipment_csv_uniqueness_matches_sqlite_nocase_rules(store):
    content = EQUIPMENT_HEADER + "STRASSE,First,Test\nStra\u00dfe,Second,Test\n"
    assert import_csv(store, content, "equipment") == 2
    assert len(store.list_equipment()) == 2


def test_file_size_limit(store):
    with pytest.raises(CSVImportError, match="5 MiB"):
        import_csv(store, b"x" * (5 * 1024 * 1024 + 1), "equipment")


def test_row_limit(store, monkeypatch):
    monkeypatch.setattr("maintainai.csv_import.MAX_ROWS", 1)
    with pytest.raises(CSVImportError, match="row limit"):
        import_csv(store, EQUIPMENT_HEADER + "A,Test,Test\nB,Test,Test\n", "equipment")
    assert store.list_equipment() == []


def test_errors_are_collected_with_row_numbers(store):
    with pytest.raises(CSVImportError) as caught:
        import_csv(store, EQUIPMENT_HEADER + "A,,Test\nB,Test,\n", "equipment")
    assert len(caught.value.errors) == 2
    assert caught.value.errors[0].startswith("Row 2:")
    assert caught.value.errors[1].startswith("Row 3:")
