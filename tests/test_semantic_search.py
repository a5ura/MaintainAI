import csv
import sqlite3
from unittest.mock import Mock
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from maintainai.database import connect, initialize
from maintainai.embeddings import EmbeddingError, MODEL_DIMENSION, MODEL_ID, MODEL_REVISION, MiniLMEmbedder
from maintainai.csv_import import import_csv
from maintainai.seed import SAMPLE_DIR, seed_demo
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import MaintenanceStore, ValidationError


def add_issue(store, equipment_id, **overrides):
    record = dict(equipment_id=equipment_id, description="Synthetic vacuum fault", reporter="Demo tester")
    record.update(overrides)
    return store.create_issue(**record)


def test_schema_upgrade_is_additive(store, equipment_id):
    issue_id = add_issue(store, equipment_id, root_cause="Existing user finding")
    before = store.get_issue(issue_id)
    columns = list(store.connection.execute("PRAGMA table_info(issues)"))
    # Simulate a v0.1 database in this disposable fixture only.
    store.connection.executescript("DROP TABLE issue_embeddings; DROP TABLE semantic_index_metadata;")
    initialize(store.connection)
    initialize(store.connection)
    assert store.get_issue(issue_id) == before
    assert list(store.connection.execute("PRAGMA table_info(issues)")) == columns
    assert store.get_equipment(equipment_id)["asset_tag"] == "TEST-001"
    assert store.connection.execute("SELECT COUNT(*) FROM issue_embeddings").fetchone()[0] == 0


def test_index_contains_all_fields_and_persists_ids(store, equipment_id, fake_embedder, db_path):
    issue_id = add_issue(store, equipment_id, troubleshooting_notes="Inspected vacuum hose",
                         root_cause="Suction seal leak", corrective_action="Replaced cup")
    before = store.get_issue(issue_id)
    search = SemanticSearch(store, fake_embedder)
    assert search.rebuild() == 1
    assert search.status().current
    texts = fake_embedder.calls[0]
    assert len(texts) == 4
    assert all(any(before[field] in text for text in texts) for field in (
        "description", "troubleshooting_notes", "root_cause", "corrective_action"))
    metadata, rows = search.index.read()
    assert metadata["issue_count"] == 1
    assert metadata["model_key"] == fake_embedder.model_key
    assert rows[0]["issue_id"] == issue_id
    assert rows[0]["equipment_id"] == equipment_id
    assert np.linalg.norm(np.frombuffer(rows[0]["vector"], dtype="<f4")) == pytest.approx(1)
    assert store.get_issue(issue_id) == before
    with connect(db_path) as connection:
        reopened = SemanticSearch(MaintenanceStore(connection), fake_embedder)
        assert reopened.status().current
        assert reopened.search("suction leak")[0]["id"] == issue_id


def test_retrieval_returns_five_ranked_original_records(store, fake_embedder):
    seed_demo(store.connection)
    search = SemanticSearch(store, fake_embedder)
    assert search.rebuild() == 20
    results = search.search("vacuum gripper suction cup seal leak")
    assert len(results) == 5
    assert results[0]["asset_tag"] == "DEMO-VAC-001"
    assert [row["similarity"] for row in results] == sorted((row["similarity"] for row in results), reverse=True)
    assert all(-1 <= row["similarity"] <= 1 for row in results)
    assert all(row["equipment_id"] and row["equipment_name"] for row in results)
    assert all(store.get_issue(row["id"])["root_cause"] == row["root_cause"] for row in results)
    assert len(search.search("camera lens focus", limit=2)) == 2
    assert search.search("camera lens focus")[0]["asset_tag"] == "DEMO-VIS-001"


def test_empty_dataset_never_loads_model(store, fake_embedder):
    search = SemanticSearch(store, fake_embedder)
    assert search.search("vacuum fault") == []
    assert search.rebuild() == 0
    assert search.status().current
    assert search.search("vacuum fault") == []
    assert fake_embedder.calls == []


def test_missing_index_blocks_search_without_model_load(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    assert not search.status().current
    with pytest.raises(StaleIndexError, match="not been built"):
        search.search("vacuum fault")
    assert fake_embedder.calls == []


@pytest.mark.parametrize("field", ["status", "troubleshooting_notes", "root_cause", "corrective_action"])
def test_issue_updates_block_stale_results_until_rebuild(store, equipment_id, fake_embedder, field):
    issue_id = add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    update = {"status": "Closed"}
    if field != "status":
        update[field] = "Synthetic camera lens finding"
    store.update_issue(issue_id, **update)
    assert not search.status().current
    calls = len(fake_embedder.calls)
    with pytest.raises(StaleIndexError):
        search.search("camera lens")
    assert len(fake_embedder.calls) == calls
    search.rebuild()
    result = search.search("camera lens")[0]
    assert result[field] == update[field]
    assert search.status().current


def test_direct_edits_without_timestamp_changes_are_detected(store, equipment_id, fake_embedder):
    issue_id = add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    with store.connection:
        store.connection.execute("UPDATE issues SET description=? WHERE id=?", ("Changed description", issue_id))
    assert not search.status().current
    with pytest.raises(StaleIndexError):
        search.search("changed")


def test_new_csv_tickets_invalidate_index(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    content = "equipment_asset_tag,description,priority,status,reporter\nTEST-001,Synthetic new fault,High,Open,Demo\n"
    assert import_csv(store, content, "issues") == 1
    assert not search.status().current
    search.rebuild()
    assert len(search.search("new fault")) == 2


def test_equipment_name_changes_invalidate_index(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    with store.connection:
        store.connection.execute("UPDATE equipment SET name='Renamed equipment' WHERE id=?", (equipment_id,))
    assert not search.status().current


def test_model_version_change_requires_rebuild(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    fake_embedder.model_key = "another-model-revision"
    with pytest.raises(StaleIndexError, match="model"):
        search.search("vacuum")
    search.rebuild()
    assert search.status().current


def test_rebuild_and_seed_do_not_duplicate_tickets_or_vectors(store, fake_embedder):
    seed_demo(store.connection)
    before = store.list_issues()
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    assert seed_demo(store.connection) == {"equipment_added": 0, "issues_added": 0}
    search.rebuild()
    assert store.list_issues() == before
    assert store.connection.execute("SELECT COUNT(*) FROM issue_embeddings").fetchone()[0] == 20
    assert store.connection.execute("SELECT COUNT(*) FROM semantic_index_metadata").fetchone()[0] == 1
    assert store.connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("invalid", [np.zeros((1, 6)), np.ones((1, 2)), np.full((1, 6), np.nan), np.full((1, 6), np.inf)])
def test_invalid_model_output_leaves_previous_index_intact(store, equipment_id, fake_embedder, monkeypatch, invalid):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    before = search.index.read()
    monkeypatch.setattr(fake_embedder, "encode", lambda texts: invalid)
    with pytest.raises(EmbeddingError):
        search.rebuild()
    assert search.index.read() == before
    assert search.status().current


def test_corrupt_vectors_are_hidden(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    with store.connection:
        store.connection.execute("UPDATE issue_embeddings SET vector=?", (np.full(6, np.nan, dtype="<f4").tobytes(),))
    assert not search.status().current
    with pytest.raises(StaleIndexError, match="invalid"):
        search.search("vacuum")
    search.rebuild()
    assert search.status().current


def test_missing_vectors_require_rebuild(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    with store.connection:
        store.connection.execute("DELETE FROM issue_embeddings")
    with pytest.raises(StaleIndexError, match="missing"):
        search.search("vacuum")


def test_storage_failure_rolls_back_derived_index_replacement(store, equipment_id, fake_embedder, monkeypatch):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    before = search.index.read()
    issues_before = store.list_issues()

    def fail_after_delete(*args, **kwargs):
        store.connection.execute("DELETE FROM issue_embeddings")
        raise sqlite3.OperationalError("Simulated storage failure")

    monkeypatch.setattr(search.index, "replace", fail_after_delete)
    with pytest.raises(sqlite3.OperationalError, match="storage failure"):
        search.rebuild()
    assert search.index.read() == before
    assert store.list_issues() == issues_before
    assert search.status().current


def test_concurrent_edit_during_rebuild_keeps_previous_index(store, equipment_id, fake_embedder):
    add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    before = search.index.read()
    fake_embedder.hook = lambda: add_issue(store, equipment_id, description="Concurrent synthetic issue")
    with pytest.raises(StaleIndexError, match="during indexing"):
        search.rebuild()
    assert search.index.read() == before
    assert not search.status().current
    assert store.dashboard()["total_issues"] == 2


def test_concurrent_edit_during_query_blocks_results(store, equipment_id, fake_embedder):
    issue_id = add_issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    fake_embedder.hook = lambda: store.update_issue(issue_id, status="Closed", root_cause="New finding")
    with pytest.raises(StaleIndexError):
        search.search("vacuum fault")


@pytest.mark.parametrize("query,limit", [(" ", 5), ("fault", 0), ("fault", 6), ("fault", True)])
def test_invalid_queries_are_rejected(store, fake_embedder, query, limit):
    with pytest.raises(ValidationError):
        SemanticSearch(store, fake_embedder).search(query, limit=limit)
    assert fake_embedder.calls == []


def test_extended_seed_preserves_original_five_and_user_data(store, tmp_path):
    (tmp_path / "equipment.csv").write_bytes((SAMPLE_DIR / "equipment.csv").read_bytes())
    with (SAMPLE_DIR / "issues.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        examples = list(reader)
        with (tmp_path / "issues.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(examples[:5])
    assert seed_demo(store.connection, tmp_path)["issues_added"] == 5
    legacy = store.list_issues()
    user_id = add_issue(store, legacy[0]["equipment_id"], description="User issue remains unchanged")
    store.update_issue(legacy[0]["id"], status="Closed", root_cause="Existing user edit")
    before = store.get_issue(legacy[0]["id"])
    assert seed_demo(store.connection) == {"equipment_added": 0, "issues_added": 15}
    assert store.get_issue(legacy[0]["id"]) == before
    assert store.get_issue(user_id)["description"] == "User issue remains unchanged"
    assert seed_demo(store.connection)["issues_added"] == 0
    assert len(store.list_issues()) == 21
    assert all(row["description"].startswith("Synthetic demo:") for row in examples[5:])
    assert all(row["root_cause"] and row["corrective_action"] for row in examples[5:])
    assert len({row["equipment_asset_tag"] for row in examples[5:]}) == 5


def test_real_adapter_uses_cpu_pinned_revision_and_local_cache(tmp_path, monkeypatch, block_real_model_loads):
    model = Mock()
    model.encode.return_value = np.ones((1, MODEL_DIMENSION), dtype=np.float32)
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    original_loader = block_real_model_loads
    original_loader.cache_clear()
    monkeypatch.setattr("maintainai.embeddings._load_model", original_loader)
    monkeypatch.setenv("MAINTAINAI_MODEL_OFFLINE", "1")
    try:
        result = MiniLMEmbedder().encode(["Synthetic fault"])
        assert result.shape == (1, MODEL_DIMENSION)
        assert factory.call_args.args == (MODEL_ID,)
        kwargs = factory.call_args.kwargs
        assert kwargs["device"] == "cpu"
        assert kwargs["revision"] == MODEL_REVISION
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["token"] is False
        assert ".cache" in kwargs["cache_folder"]
        assert model.encode.call_args.kwargs["device"] == "cpu"
        assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    finally:
        original_loader.cache_clear()


def test_model_load_failure_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Simulated download failure")

    monkeypatch.setattr("maintainai.embeddings._load_model", fail)
    with pytest.raises(EmbeddingError, match="first rebuild needs internet"):
        MiniLMEmbedder().encode(["Fault"])


@pytest.mark.parametrize("configured_offline", ["0", "1"])
def test_explicit_offline_embedder_uses_cached_weights_without_changing_global_settings(monkeypatch, configured_offline):
    import os

    monkeypatch.setenv("MAINTAINAI_MODEL_OFFLINE", configured_offline)
    model = Mock()
    model.encode.return_value = np.ones((1, MODEL_DIMENSION), dtype=np.float32)
    loader = Mock(return_value=model)
    monkeypatch.setattr("maintainai.embeddings._load_model", loader)
    MiniLMEmbedder(offline=True).encode(["Synthetic problem"])
    assert loader.call_args.args[1] is True
    assert os.environ["MAINTAINAI_MODEL_OFFLINE"] == configured_offline
