"""Synthetic evidence only; real HTTP and model loading are blocked by conftest."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from maintainai.embeddings import EmbeddingError
from maintainai.fault_detection import detect_fault_patterns
from maintainai.fault_matching import HISTORICAL_NOTICE, build_fault_query, match_fault_to_issues
from maintainai.semantic_search import SemanticSearch, StaleIndexError
from maintainai.service import ValidationError


def logs(equipment_id=1):
    return [
        {"id": 10, "equipment_id": equipment_id, "timestamp": "2026-10-25T09:15:15Z",
         "severity": "ERROR", "event_code": "PICK_FAIL", "message": "Synthetic vacuum gripper drops part",
         "asset_tag": "SYNTHETIC-VAC-001", "equipment_name": "Synthetic vacuum gripper"},
        {"id": 11, "equipment_id": equipment_id, "timestamp": "2026-10-25T09:15:18Z",
         "severity": "ERROR", "event_code": "PICK_FAIL", "message": "Synthetic suction pressure loss",
         "asset_tag": "SYNTHETIC-VAC-001", "equipment_name": "Synthetic vacuum gripper"},
    ]


def fault(equipment_id=1):
    return detect_fault_patterns(logs(equipment_id))[0]


def issue(store, equipment_id, **fields):
    return store.create_issue(equipment_id=equipment_id, reporter="Synthetic tester",
                              **{"description": "Synthetic vacuum gripper suction pressure loss", **fields})


@pytest.mark.parametrize("pattern", ["repeated_failures", "warning_to_error"])
def test_query_contains_observations_pattern_and_equipment(pattern):
    query = build_fault_query({**fault(), "pattern_type": pattern}, logs())
    assert f"Pattern: {pattern.replace('_', ' ')}" in query
    assert "ID 1" in query
    assert "Synthetic vacuum gripper" in query
    assert "SYNTHETIC-VAC-001" in query
    assert "PICK_FAIL" in query
    assert "Synthetic vacuum gripper drops part" in query
    assert "Synthetic suction pressure loss" in query


def test_query_uses_only_cited_logs_for_the_same_equipment():
    finding = fault()
    finding["explanation"] = "Unsupported diagnosis: gearbox caused everything"
    finding["root_cause"] = "Unsupported bearing diagnosis"
    records = logs() + [{**logs()[0], "id": 99, "message": "Unrelated conveyor event"},
                        {**logs()[0], "equipment_id": 2, "message": "Different asset event"}]
    query = build_fault_query(finding, records)
    for excluded in ("gearbox", "bearing", "Unrelated conveyor", "Different asset"):
        assert excluded not in query


def test_query_order_deduplication_and_inputs_are_stable():
    finding, records = fault(), logs()
    original = deepcopy((finding, records))
    expected = build_fault_query(finding, records)
    assert build_fault_query(finding, records[::-1]) == expected
    assert build_fault_query({**finding, "log_ids": [10, 11, 10]}, records + [records[0]]) == expected
    assert (finding, records) == original


def test_query_is_bounded_and_repeated_observations_do_not_crowd_it():
    records = [{**logs()[0], "id": index, "message": f"Synthetic observation {index} " + "x" * 9900}
               for index in range(1, 11)]
    query = build_fault_query({**fault(), "log_ids": list(range(1, 11))}, records)
    assert len(query) < 4000  # Safely below the existing service's 10,000-character input limit.
    assert "Synthetic observation 6" in query
    assert "Synthetic observation 7" not in query
    duplicate = [{**record, "message": "Same synthetic event"} for record in records]
    assert build_fault_query({**fault(), "log_ids": list(range(1, 11))}, duplicate).count("Same synthetic event") == 1


def test_equipment_metadata_is_optional():
    records = [{key: value for key, value in log.items() if key not in ("asset_tag", "equipment_name")}
               for log in logs()]
    query = build_fault_query(fault(), records)
    assert "Equipment: ID 1" in query
    assert "PICK_FAIL" in query


def test_matching_integrates_with_existing_search_and_returns_historical_fields(store, equipment_id, fake_embedder):
    target = issue(store, equipment_id, root_cause="Recorded synthetic vacuum seal leak",
                   corrective_action="Recorded synthetic cup replacement", status="Closed")
    camera = store.create_equipment(asset_tag="SYNTHETIC-CAM", name="Synthetic vision camera", category="Vision")
    issue(store, camera, description="Synthetic camera image focus illumination issue")
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    expected = search.search(build_fault_query(fault(equipment_id), logs(equipment_id)), limit=5)
    spy = Mock(wraps=search.search)
    search.search = spy
    result = match_fault_to_issues(fault(equipment_id), logs(equipment_id), search)
    spy.assert_called_once_with(result.query, limit=5)
    assert result.matches[0]["issue_id"] == target
    for actual, original in zip(result.matches, expected):
        assert {field: actual[field] for field in original} == original
        assert actual["issue_id"] == original["id"]
        for field in ("equipment_id", "asset_tag", "equipment_name", "similarity", "description",
                      "root_cause", "corrective_action", "status"):
            assert field in actual
    assert result.evidence_log_ids == [10, 11]
    assert result.missing_log_ids == []


def test_rank_ties_and_raw_cosine_scores_pass_through_without_reinterpretation():
    records = [{"id": issue_id, "similarity": score}
               for issue_id, score in [(8, 0.8), (2, 0.8), (7, 0.01), (3, -0.2), (1, -0.9)]]
    search = Mock(spec=SemanticSearch)
    search.search.return_value = records
    before = deepcopy(records)
    result = match_fault_to_issues(fault(), logs(), search)
    assert [match["issue_id"] for match in result.matches] == [8, 2, 7, 3, 1]
    assert [match["similarity"] for match in result.matches] == [0.8, 0.8, 0.01, -0.2, -0.9]
    assert all("confidence" not in match for match in result.matches)
    assert records == before
    search.search.assert_called_once_with(result.query, limit=5)


def test_existing_search_limits_results_to_five(store, equipment_id, fake_embedder):
    ids = [issue(store, equipment_id) for _ in range(7)]
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    result = match_fault_to_issues(fault(equipment_id), logs(equipment_id), search)
    assert [match["issue_id"] for match in result.matches] == ids[:5]


def test_store_logs_detection_and_matching_are_read_only(store, equipment_id, fake_embedder):
    target = issue(store, equipment_id)
    for record in logs(equipment_id):
        store.create_log(**{key: record[key] for key in
                            ("equipment_id", "timestamp", "severity", "event_code", "message")})
    evidence = store.list_logs(equipment_id=equipment_id)
    finding = detect_fault_patterns(evidence)[0]
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    records_before = deepcopy((store.list_issues(), evidence, search.index.read()))
    result = match_fault_to_issues(finding, evidence, search)
    assert result.matches[0]["issue_id"] == target
    assert result.evidence_log_ids == finding["log_ids"]
    assert "TEST-001" in result.query
    assert (store.list_issues(), store.list_logs(equipment_id=equipment_id), search.index.read()) == records_before


def test_similar_history_can_come_from_another_asset(store, equipment_id, fake_embedder):
    other = store.create_equipment(asset_tag="SYNTHETIC-OTHER", name="Synthetic vacuum gripper", category="Gripper")
    target = issue(store, other)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    result = match_fault_to_issues(fault(equipment_id), logs(equipment_id), search)
    assert result.matches[0]["issue_id"] == target
    assert result.matches[0]["equipment_id"] == other


@pytest.mark.parametrize("records", [None, [], [{**logs()[0], "equipment_id": 2}],
                                    [{**logs()[0], "id": 99}]])
def test_missing_logs_skip_search_safely(records):
    search = Mock(spec=SemanticSearch)
    result = match_fault_to_issues(fault(), records, search)
    assert result.query == ""
    assert result.matches == []
    assert result.evidence_log_ids == []
    assert result.missing_log_ids == [10, 11]
    assert "No usable supporting log text" in result.notice
    search.search.assert_not_called()


def test_missing_log_id_list_cannot_accidentally_search_unrelated_logs():
    finding = fault()
    del finding["log_ids"]
    search = Mock(spec=SemanticSearch)
    result = match_fault_to_issues(finding, logs(), search)
    assert result.matches == []
    assert result.missing_log_ids == []
    search.search.assert_not_called()


def test_partial_evidence_reports_missing_ids_and_searches_only_available_text():
    search = Mock(spec=SemanticSearch)
    search.search.return_value = []
    result = match_fault_to_issues(fault(), logs()[:1], search)
    assert result.evidence_log_ids == [10]
    assert result.missing_log_ids == [11]
    assert "drops part" in result.query
    assert "pressure loss" not in result.query
    assert "Supporting logs are missing" in result.notice
    search.search.assert_called_once_with(result.query, limit=5)


def test_logs_with_no_usable_text_skip_search():
    records = [{**log, "event_code": "", "message": "  "} for log in logs()]
    search = Mock(spec=SemanticSearch)
    result = match_fault_to_issues(fault(), records, search)
    assert result.matches == []
    assert result.evidence_log_ids == [10, 11]
    search.search.assert_not_called()


def test_event_codes_are_usable_when_messages_are_missing():
    records = [{**log, "message": None} for log in logs()]
    assert "PICK_FAIL" in build_fault_query(fault(), records)


def test_no_historical_issues_returns_empty_without_loading_model(store, equipment_id, fake_embedder):
    result = match_fault_to_issues(fault(equipment_id), logs(equipment_id), SemanticSearch(store, fake_embedder))
    assert result.matches == []
    assert result.query
    assert "No historical issues" in result.notice
    assert fake_embedder.calls == []


def test_recorded_findings_are_explicitly_separate_from_current_hypotheses(store, equipment_id, fake_embedder):
    target = issue(store, equipment_id, root_cause="Historical synthetic seal wear",
                   corrective_action="Historical synthetic seal replacement", status="Resolved")
    before = store.get_issue(target)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    result = match_fault_to_issues(fault(equipment_id), logs(equipment_id), search)
    match = result.matches[0]
    assert match["evidence_type"] == "recorded_historical_issue"
    assert match["root_cause"] == before["root_cause"]
    assert match["corrective_action"] == before["corrective_action"]
    assert result.notice == HISTORICAL_NOTICE
    assert "Similarity does not establish the current root cause" in result.notice
    assert "No hypotheses are generated" in result.notice
    assert store.get_issue(target) == before


@pytest.mark.parametrize("change", ["new", "updated"])
def test_index_freshness_checks_are_preserved(store, equipment_id, fake_embedder, change):
    target = issue(store, equipment_id)
    search = SemanticSearch(store, fake_embedder)
    search.rebuild()
    calls_before = len(fake_embedder.calls)
    if change == "new":
        issue(store, equipment_id)
    else:
        store.update_issue(target, status="Closed", root_cause="New synthetic evidence")
    with pytest.raises(StaleIndexError, match="Rebuild"):
        match_fault_to_issues(fault(equipment_id), logs(equipment_id), search)
    assert len(fake_embedder.calls) == calls_before  # Matching never silently rebuilds.
    search.rebuild()
    assert match_fault_to_issues(fault(equipment_id), logs(equipment_id), search).matches


def test_existing_embedding_errors_propagate_without_fabricated_matches():
    search = Mock(spec=SemanticSearch)
    search.search.side_effect = EmbeddingError("Local model unavailable")
    with pytest.raises(EmbeddingError, match="Local model unavailable"):
        match_fault_to_issues(fault(), logs(), search)


def test_conflicting_duplicate_logs_are_rejected():
    with pytest.raises(ValidationError, match="Conflicting.*10"):
        build_fault_query(fault(), logs() + [{**logs()[0], "message": "Different observation"}])


def test_boolean_equipment_id_is_not_accepted():
    with pytest.raises(ValidationError, match="equipment_id"):
        build_fault_query({**fault(), "equipment_id": True}, logs())
