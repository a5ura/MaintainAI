"""Safe rejection metadata and one active guided schema using the reported ID sets."""

from copy import deepcopy
from hashlib import sha256
import json
from unittest.mock import Mock

import pytest

from maintainai.fault_detection import detect_fault_patterns
from maintainai.guided_rag import build_guided_context, build_guided_messages, guided_validation_logs
from maintainai.rag import (
    CLAIM_FIELDS, CONTEXT_FIELDS, FAULT_PROMPT, GUIDED_PROMPT, LOG_CITATION_PROMPT, SYSTEM_PROMPT,
    GroundingError, VALIDATION_FIELDS, VALIDATION_REJECTION_CODES, build_messages,
    generate_grounded_analysis, validate_response,
)

ISSUE_IDS = [6, 1, 7, 9, 8]
LOG_IDS = [2, 3, 4]


@pytest.fixture
def evidence():
    # Synthetic observations reproduce the reported log/issue ID layout only.
    equipment = {"id": 42, "asset_tag": "DEMO-ABB-001", "name": "Synthetic robot"}
    draft = {"equipment_id": 42, "description": "Synthetic technician report of pressure loss"}
    logs = [{"id": value, "equipment_id": 42, "timestamp": f"2026-10-25T12:00:{second:02d}Z",
             "severity": severity, "event_code": code, "message": "Synthetic recorded observation"}
            for value, second, severity, code in [
                (1, 0, "INFO", "START"), (2, 10, "WARNING", "PRESSURE_LOW"),
                (3, 20, "ERROR", "PICK_FAIL"), (4, 30, "ERROR", "PICK_FAIL"),
                (5, 40, "INFO", "EVENT"), (6, 50, "INFO", "EVENT"), (7, 59, "INFO", "EVENT"),
            ]]
    records = [{"id": value, "equipment_id": 42, "asset_tag": equipment["asset_tag"],
                "equipment_name": equipment["name"], "status": "Open", "similarity": .8,
                **{field: "Synthetic historical observation" for field in CONTEXT_FIELDS}}
               for value in ISSUE_IDS]
    analysis = {"draft": deepcopy(draft), "logs": logs, "findings": detect_fault_patterns(logs)}
    context = build_guided_context(draft, equipment=equipment, analysis=analysis, matches=records)
    return context


def valid_output():
    return {
        "issue_summary": "The reported symptoms require qualified review.",
        "possible_causes": [{"text": "A recorded historical condition may be relevant; applicability is unverified.",
                             "issue_ids": [6, 1], "log_ids": [2, 3]}],
        "recommended_diagnostic_checks": [{"text": "Qualified personnel should inspect using approved procedures.",
                                           "issue_ids": [7, 9], "log_ids": [4]}],
        "suggested_next_steps": [{"text": "Review the supplied evidence before considering corrective work.",
                                  "issue_ids": [8], "log_ids": [3]}],
        "evidence_references": ISSUE_IDS[:], "log_evidence_references": LOG_IDS[:],
        "uncertainty_or_missing_information": ["Current conditions have not been independently verified."],
    }


def validate(data, evidence):
    raw = data if isinstance(data, str) else json.dumps(data)
    return validate_response(raw, evidence["historical_findings"], logs=guided_validation_logs(evidence))


def test_reported_allowed_ids_and_pattern_support_have_a_valid_log_aware_contract(evidence):
    messages = build_guided_messages(evidence)
    envelope = json.loads(messages[1]["content"])
    assert envelope["allowed_issue_ids"] == ISSUE_IDS
    assert envelope["allowed_log_ids"] == LOG_IDS
    patterns = evidence["observed_evidence"]["detected_patterns"]
    assert [(pattern["pattern_type"], pattern["log_ids"]) for pattern in patterns] == [
        ("warning_to_error", [2, 3, 4]), ("repeated_failures", [3, 4]),
    ]
    assert validate(valid_output(), evidence) == valid_output()


@pytest.mark.parametrize("log_aware", [False, True])
def test_guided_prompt_has_one_complete_active_schema_and_all_validator_limits(evidence, log_aware):
    if not log_aware:
        evidence["observed_evidence"] = {"logs": [], "detected_patterns": []}
        evidence["citation_mode"] = "issue_only"
    messages = build_guided_messages(evidence)
    prompt = messages[0]["content"]
    assert prompt.startswith(GUIDED_PROMPT)
    assert SYSTEM_PROMPT not in prompt
    assert LOG_CITATION_PROMPT not in prompt
    assert prompt.count("Active response schema:") == 1
    schema = json.loads(prompt.split("Active response schema:\n", 1)[1].split("\nEnd response schema.", 1)[0])
    expected = {"issue_summary", *CLAIM_FIELDS, "evidence_references", "uncertainty_or_missing_information"}
    if log_aware:
        expected.add("log_evidence_references")
    assert set(schema["properties"]) == set(schema["required"]) == expected
    assert schema["additionalProperties"] is False
    assert schema["properties"]["issue_summary"]["maxLength"] == 1500
    for section in CLAIM_FIELDS:
        array = schema["properties"][section]
        assert array["maxItems"] == 5 and array["minItems"] == 0
        claim = array["items"]
        fields = {"text", "issue_ids"} | ({"log_ids"} if log_aware else set())
        assert set(claim["required"]) == set(claim["properties"]) == fields
        assert claim["additionalProperties"] is False
        assert claim["properties"]["text"]["maxLength"] == 1500
        for field in fields - {"text"}:
            assert claim["properties"][field] == {"type": "array", "minItems": 1, "maxItems": 5,
                                                  "uniqueItems": True, "items": {"type": "integer"}}
    uncertainty = schema["properties"]["uncertainty_or_missing_information"]
    assert uncertainty["minItems"] == 1 and uncertainty["maxItems"] == 10
    assert uncertainty["items"]["maxLength"] == 1000
    assert schema["properties"]["evidence_references"]["maxItems"] == 5
    if log_aware:
        assert schema["properties"]["log_evidence_references"]["maxItems"] == 3
        assert "BOTH issue and log citations" in prompt
    assert "32000 characters" in prompt and "before trimming" in prompt
    assert "exactly the union" in prompt and "retrieved-but-uncited" in prompt
    assert "duplicate JSON keys" in prompt and "never strings or booleans" in prompt
    assert "To abstain" in prompt and "free-text field" in prompt
    assert "Use modal hypotheses" in prompt
    assert "Pneumatic pressure loss may be relevant; current applicability\nis unverified" in prompt
    assert 'Avoid "cause is" and "cause was" constructions throughout generated prose' in prompt
    assert "diagnostic checks, next steps, summaries, and uncertainty fields" in prompt
    assert 'ticket recorded..."' in prompt and 'root cause was..."' in prompt
    assert json.loads(json.dumps(messages)) == messages


def test_legacy_assistant_and_fault_prompt_bytes_are_unchanged():
    assert sha256(SYSTEM_PROMPT.encode()).hexdigest() == "02e85a81b3eace97c3ee2653199990a0fe12ba34ac63614bf3c7491cf902d63f"
    assert sha256(FAULT_PROMPT.encode()).hexdigest() == "8339d16c1d26d561683284e6796fe1d4d4075bb2cc73ed4348224bc31ff2071d"
    assert build_messages("Synthetic question", [])[0]["content"] == SYSTEM_PROMPT
    assert build_messages("Synthetic question", [], fault_context={"logs": []})[0]["content"] == SYSTEM_PROMPT + FAULT_PROMPT


def malformed(case):
    result = valid_output()
    if case == "json":
        return "SENSITIVE malformed provider payload"
    if case == "duplicate_key":
        return '{"SENSITIVE_KEY":1,"SENSITIVE_KEY":2}'
    if case == "root_fields":
        result["SENSITIVE_KEY"] = "SENSITIVE_VALUE"
    elif case == "section":
        result["possible_causes"] = {}
    elif case == "claim_fields":
        result["possible_causes"][0]["SENSITIVE_KEY"] = "SENSITIVE_VALUE"
    elif case == "missing_issue":
        del result["possible_causes"][0]["issue_ids"]
    elif case == "missing_log":
        del result["possible_causes"][0]["log_ids"]
    elif case == "issue_id":
        result["possible_causes"][0]["issue_ids"] = [999999]
    elif case == "log_id":
        result["possible_causes"][0]["log_ids"] = [1]  # Analyzed, but not supplied as supporting evidence.
    elif case == "string_id":
        result["possible_causes"][0]["log_ids"] = ["2"]
    elif case == "boolean_id":
        result["possible_causes"][0]["issue_ids"] = [True]
    elif case == "duplicate_id":
        result["possible_causes"][0]["log_ids"] = [2, 2]
    elif case == "issue_union":
        result["evidence_references"] = [6]
    elif case == "log_union":
        result["log_evidence_references"] = []
    elif case == "empty_text":
        result["issue_summary"] = " "
    elif case == "text_limit":
        result["issue_summary"] = "x" * 1501
    elif case == "uncertainty_limit":
        result["uncertainty_or_missing_information"] = ["x" * 1001]
    elif case == "uncertainty_count":
        result["uncertainty_or_missing_information"] *= 11
    elif case == "inline":
        result["issue_summary"] = "Review Log #2 before drawing conclusions."
    elif case == "unsafe":
        result["suggested_next_steps"][0]["text"] = "Disable the safety interlock"
    elif case == "cause":
        result["issue_summary"] = "The current root cause is a torn seal."
    elif case == "response_limit":
        return "x" * 32001
    return result


@pytest.mark.parametrize("case, code, field", [
    ("json", "json.invalid", "response"), ("duplicate_key", "json.duplicate_key", "response"),
    ("root_fields", "schema.response_fields", "response"), ("section", "schema.claim_section", "possible_causes"),
    ("claim_fields", "schema.claim_fields", "possible_causes"),
    ("missing_issue", "citation.issue_ids", "possible_causes.issue_ids"),
    ("missing_log", "citation.log_ids", "possible_causes.log_ids"),
    ("issue_id", "citation.issue_ids", "possible_causes.issue_ids"),
    ("log_id", "citation.log_ids", "possible_causes.log_ids"),
    ("string_id", "citation.log_ids", "possible_causes.log_ids"),
    ("boolean_id", "citation.issue_ids", "possible_causes.issue_ids"),
    ("duplicate_id", "citation.log_ids", "possible_causes.log_ids"),
    ("issue_union", "citation.reference_union", "evidence_references"),
    ("log_union", "citation.reference_union", "log_evidence_references"),
    ("empty_text", "text.invalid", "issue_summary"), ("text_limit", "text.limit", "issue_summary"),
    ("uncertainty_limit", "text.limit", "uncertainty_or_missing_information"),
    ("uncertainty_count", "schema.uncertainty", "uncertainty_or_missing_information"),
    ("inline", "text.inline_reference", "issue_summary"),
    ("unsafe", "safety.operational_instruction", "suggested_next_steps.text"),
    ("cause", "text.cause_language", "issue_summary"), ("response_limit", "response.limit", "response"),
])
def test_rejection_codes_and_fields_are_fixed_and_safe(evidence, case, code, field):
    with pytest.raises(GroundingError) as caught:
        validate(malformed(case), evidence)
    assert caught.value.code == code and caught.value.field == field
    assert caught.value.safe_diagnostic == f"{code} [{field}]"
    assert code in VALIDATION_REJECTION_CODES and field in VALIDATION_FIELDS
    assert "SENSITIVE" not in caught.value.safe_diagnostic


@pytest.mark.parametrize("field", ["issue_summary", "possible_causes.text", "uncertainty_or_missing_information"])
def test_only_standalone_benign_uncertainty_is_accepted(evidence, field):
    result = valid_output()
    if field == "issue_summary":
        result[field] = "The root cause is unverified."
    elif field == "uncertainty_or_missing_information":
        result[field] = ["The root cause is unverified"]
    else:
        result["possible_causes"][0]["text"] = "The current root cause is unverified."
    assert validate(result, evidence)


@pytest.mark.parametrize("text", [
    "The root cause is unverified but definitely a torn seal.",
    "The root cause is unverified. The current root cause is a torn seal.",
    "The root cause is unverified; the current cause is a torn seal.",
    "The root cause is unverified. Replace the pipe immediately.",
    "The historical root cause was a worn seal.",
    "The root cause is a worn seal.", "Confirmed current root cause: worn seal.",
])
def test_uncertainty_prefixes_do_not_bypass_cause_or_operational_safety(evidence, text):
    result = valid_output()
    result["issue_summary"] = text
    with pytest.raises(GroundingError) as caught:
        validate(result, evidence)
    assert caught.value.code == "text.cause_language"


def test_explicit_recorded_historical_observation_is_supported_without_asserting_current_cause(evidence):
    result = valid_output()
    result["possible_causes"][0]["text"] = "A historical ticket recorded a worn seal; current applicability is unverified."
    assert validate(result, evidence)


@pytest.mark.parametrize("section", CLAIM_FIELDS)
@pytest.mark.parametrize("text", [
    "Pneumatic pressure loss may be relevant; current applicability is unverified and requires qualified verification.",
    "A pneumatic leak could explain the reported pickup failures; qualified personnel must verify applicability.",
    "A historical ticket recorded pressure loss as the confirmed cause of that historical event; current applicability is unverified.",
    "A historical ticket recorded that low pneumatic pressure was the confirmed cause; current applicability is unverified.",
    "A historical ticket recorded that a worn seal had been confirmed as the cause; pressure loss may be relevant here, subject to qualified verification.",
])
def test_realistic_modal_and_recorded_historical_wording_in_every_claim_section(evidence, section, text):
    result = valid_output()
    result[section][0]["text"] = text
    assert validate(result, evidence) == result


@pytest.mark.parametrize("field", ["issue_summary", "uncertainty_or_missing_information",
                                  *(f"{section}.text" for section in CLAIM_FIELDS)])
@pytest.mark.parametrize("text", [
    "Low pneumatic pressure is the confirmed cause.",
    "Low pneumatic pressure is the confirmed current root cause of these pickup failures.",
    "The root cause has been confirmed as a pneumatic leak.",
    "The current cause has been confirmed to be a worn seal.",
    "A possible cause is low pneumatic pressure.",
    "The root cause is possibly a pneumatic leak; this remains unverified.",
    "A historical ticket recorded a worn seal; low pneumatic pressure is the confirmed cause of the current failures.",
    "Current applicability is unverified. The root cause has been confirmed as a pneumatic leak.",
    "Pressure loss may be relevant. Low pneumatic pressure is the confirmed cause.",
    "The root cause is unverified; qualified inspection is required.",
])
def test_asserted_cause_paraphrases_and_existing_restrictions_in_every_text_field(evidence, field, text):
    result = valid_output()
    if field == "issue_summary":
        result[field] = text
    elif field == "uncertainty_or_missing_information":
        result[field] = [text]
    else:
        result[field.removesuffix(".text")][0]["text"] = text
    with pytest.raises(GroundingError) as caught:
        validate(result, evidence)
    assert caught.value.safe_diagnostic == f"text.cause_language [{field}]"


@pytest.mark.parametrize("section", CLAIM_FIELDS)
@pytest.mark.parametrize("text,code", [
    ("Replace the pneumatic pipe immediately.", "safety.operational_instruction"),
    ("Pressure loss may be relevant; disable the safety interlock.", "safety.operational_instruction"),
    ("The root cause is unverified. Replace the pneumatic pipe immediately.", "text.cause_language"),
])
def test_modal_and_uncertainty_wording_cannot_bypass_operational_safety(evidence, section, text, code):
    result = valid_output()
    result[section][0]["text"] = text
    with pytest.raises(GroundingError) as caught:
        validate(result, evidence)
    assert caught.value.safe_diagnostic == f"{code} [{section}.text]"


def test_confirmed_cause_paraphrase_uses_local_fallback_and_one_provider_call(evidence):
    result = valid_output()
    result["possible_causes"][0]["text"] = "Low pneumatic pressure is the confirmed cause."
    provider = Mock()
    provider.generate.return_value = json.dumps(result)
    fallback = {"local": "retrieval-only"}
    analysis, generated, notice = generate_grounded_analysis(
        build_guided_messages(evidence), evidence["historical_findings"], fallback,
        use_cloud=True, provider=provider, logs=guided_validation_logs(evidence))
    assert analysis is fallback and not generated
    assert "text.cause_language [possible_causes.text]" in notice
    assert result["possible_causes"][0]["text"] not in notice
    provider.generate.assert_called_once()


@pytest.mark.parametrize("log_aware", [False, True])
def test_active_mode_abstention_still_requires_uncertainty_and_no_fabricated_references(evidence, log_aware):
    result = valid_output()
    for field in CLAIM_FIELDS:
        result[field] = []
    result["evidence_references"] = []
    result["log_evidence_references"] = []
    logs = guided_validation_logs(evidence)
    if not log_aware:
        del result["log_evidence_references"]
        logs = None
    assert validate_response(json.dumps(result), evidence["historical_findings"], logs=logs)


@pytest.mark.parametrize("tamper", [False, True])
def test_fallback_notices_never_echo_exception_messages_or_non_allowlisted_metadata(evidence, monkeypatch, tamper):
    import maintainai.rag as rag

    error = GroundingError("PRIVATE credential and maintenance payload", code="PRIVATE_CODE", field="PRIVATE_FIELD")
    if tamper:
        error.code, error.field = ["PRIVATE_CODE"], {"PRIVATE_FIELD": "SECRET"}
    monkeypatch.setattr(rag, "validate_response", Mock(side_effect=error))
    provider = Mock()
    provider.generate.return_value = "PRIVATE provider response"
    fallback = {"local": "unchanged"}
    result, generated, notice = generate_grounded_analysis(
        build_guided_messages(evidence), evidence["historical_findings"], fallback,
        use_cloud=True, provider=provider, logs=guided_validation_logs(evidence))
    assert result is fallback and not generated
    assert "validation.failed [response]" in notice
    assert all(secret not in notice for secret in ("PRIVATE", "SECRET", "credential", "payload"))
    provider.generate.assert_called_once()


def test_valid_diagnostic_does_not_include_arbitrary_exception_message(evidence, monkeypatch):
    import maintainai.rag as rag

    monkeypatch.setattr(rag, "validate_response", Mock(side_effect=GroundingError(
        "PRIVATE API key", code="citation.log_ids", field="possible_causes.log_ids")))
    provider = Mock()
    provider.generate.return_value = "PRIVATE raw output"
    _, generated, notice = generate_grounded_analysis(
        build_guided_messages(evidence), evidence["historical_findings"], {},
        use_cloud=True, provider=provider, logs=guided_validation_logs(evidence))
    assert not generated and "citation.log_ids [possible_causes.log_ids]" in notice
    assert "PRIVATE" not in notice and "API key" not in notice
    provider.generate.assert_called_once()
