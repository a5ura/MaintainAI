from copy import deepcopy

import pytest

from maintainai.guided_matching import MAX_QUERY_CHARS, build_guided_query
from maintainai.service import ValidationError


def snapshot(description="Technician's pressure observation", *, findings=True):
    logs = [{"id": number, "equipment_id": 7, "equipment_name": "Pump", "asset_tag": "PUMP-7",
             "event_code": "PRESSURE_LOW", "message": "Observed pressure loss " + "x" * 2000}
            for number in range(1, 25)]
    patterns = [{"equipment_id": 7, "pattern_type": "repeated_failures", "log_ids": list(range(1, 25))}]
    return {"draft": {"description": description, "equipment_id": 7},
            "logs": logs, "findings": patterns if findings else []}


def test_description_only_query_preserves_clean_technician_report():
    assert build_guided_query(" \n Vacuum gripper loses suction. \t") == (
        "Technician-reported problem:\nVacuum gripper loses suction.")


@pytest.mark.parametrize("findings", [True, False])
def test_combined_query_is_bounded_and_separates_description_from_actual_evidence(findings):
    description = "Technician report " + "z" * 3980
    analysis = snapshot(description, findings=findings)
    before = deepcopy(analysis)
    analysis["logs"].append({**analysis["logs"][0], "equipment_id": 99, "message": "OTHER ASSET SECRET"})
    query = build_guided_query(description, equipment_id=7, analysis=analysis)
    assert len(query) <= MAX_QUERY_CHARS
    assert query.startswith("Technician-reported problem:\n" + description)
    assert "Recorded log evidence" in query and "PRESSURE_LOW" in query
    assert "Observed pressure loss" in query
    assert "OTHER ASSET SECRET" not in query
    assert query.count(description) == 1
    assert ("Pattern: repeated failures" in query) is findings
    assert before == {**analysis, "logs": analysis["logs"][:-1]}


def test_only_real_supporting_logs_contribute_to_fault_query():
    analysis = snapshot()
    analysis["findings"][0]["log_ids"] = [1]
    analysis["logs"][1]["message"] = "UNCITED LOG SECRET"
    query = build_guided_query(analysis["draft"]["description"], equipment_id=7, analysis=analysis)
    assert "UNCITED LOG SECRET" not in query


def test_log_context_budget_does_not_grow_with_number_of_findings():
    analysis = snapshot()
    analysis["findings"] *= 1000
    query = build_guided_query(analysis["draft"]["description"], equipment_id=7, analysis=analysis)
    assert len(query) <= MAX_QUERY_CHARS
    assert query.count("Pattern: repeated failures") == 1


@pytest.mark.parametrize("description, equipment_id", [("", None), ("x" * 4001, None), ("Problem", True), ("Problem", -1)])
def test_invalid_query_inputs_are_rejected(description, equipment_id):
    with pytest.raises(ValidationError):
        build_guided_query(description, equipment_id=equipment_id)


@pytest.mark.parametrize("equipment_id", [None, 8])
def test_snapshot_for_a_different_draft_cannot_be_used(equipment_id):
    with pytest.raises(ValidationError, match="snapshot does not match"):
        build_guided_query("Different technician description", equipment_id=equipment_id, analysis=snapshot())
