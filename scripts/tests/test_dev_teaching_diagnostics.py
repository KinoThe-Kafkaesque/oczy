"""Fail closed on unapproved runs and keep new rendering within teaching data."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from oczy.experiments.meta_cortex.contracts import DialogueMessage, TaskGeneratorConfig
from oczy.experiments.meta_cortex.taskgen import build_dev_catalog
from scripts.audit_dev_teaching_coverage import audit_catalog, taught_facts
from scripts.materialize_dev_prompt_repair import amended_catalog
from scripts.probe_dev_corrective_facts import (
    build_cases,
    corrective_facts_messages,
    run,
    validate_approval,
)


@pytest.fixture
def catalog():
    return amended_catalog(build_dev_catalog(TaskGeneratorConfig(20260709, 30, 5)), "v4")


def test_coverage_audit_distinguishes_missing_lookup_from_rule_transfer(catalog):
    audit = audit_catalog(catalog)
    assert audit["model_runs"] == 0
    assert not audit["scoring_or_instrument_changed"]
    for row in audit["rows"]:
        if row["family"] == "rule_transformation" and row["kind"] == "transfer":
            assert not row["directly_taught"]
            assert not row["unconstrained_lookup_not_taught"]
        if row["family"] == "finite_state" and row["kind"] == "same_rule":
            assert row["directly_taught"]
        if row["family"] == "contextual_remap" and row["kind"] == "composition":
            assert not row["second_operand_defined"]


def test_supported_target_contradiction_fails(catalog):
    task = next(t for t in catalog.meta_validation if t.family.value == "finite_state")
    probe = replace(task.probes.same_rule[0], expected_response="contradiction")
    broken = replace(task, probes=replace(task.probes, same_rule=(probe,)))
    with pytest.raises(ValueError, match="contradicts"):
        audit_catalog(replace(catalog, meta_validation=(broken,)))


def test_new_renderer_uses_only_corrections_and_preserves_query(catalog):
    for task in (*catalog.meta_train, *catalog.meta_validation):
        for probe in task.probes.same_rule:
            messages = corrective_facts_messages(task.events, probe.messages)
            assert messages[0] == probe.messages[0]
            assert messages[2:] == probe.messages[1:]
            assert messages[1].content == "Corrected examples:\n" + "\n".join("- " + event.correction for event in task.events)
            poisoned = tuple(replace(event, attempted_behavior="DO_NOT_COPY", observation_messages=(DialogueMessage("user", "DO_NOT_COPY"),)) for event in task.events)
            assert corrective_facts_messages(poisoned, probe.messages) == messages


def test_all_old_prompts_targets_and_untaught_cases_are_preserved(catalog):
    cases = build_cases(catalog)
    assert len(cases) == 24
    evidence = json.loads((Path(__file__).resolve().parents[2] / "experiments_logs/2026-09-11_r20_dev_output_path.json").read_text())
    previous = evidence["runs"]["articulation-v4"]["rows"]
    for old in previous:
        case = next(c for c in cases if all(c[k] == old[k] for k in ("family", "condition", "probe_index")))
        assert case["prompt_sha256"] == old["prompt_sha256"]
        assert case["expected"] == old["expected"]
    new = [c for c in cases if c["condition"] == "corrective_facts_context"]
    assert len(new) == 7
    assert sum(c["directly_taught"] for c in new) == 4


@pytest.mark.parametrize("approval", [{}, {"decision": "approved", "manifest_sha256": "wrong", "scope": "DEV_DIAGNOSTIC_ONLY", "human_reply": "yes"}])
def test_unapproved_execution_fails_before_model_load(tmp_path, approval):
    with pytest.raises(ValueError, match="human approval"):
        run({"manifest_sha256": "expected"}, tmp_path / "no-run", approval)
    assert not (tmp_path / "no-run").exists()


def test_approval_is_bound_to_exact_proposal():
    manifest = {"manifest_sha256": "expected"}
    approval = {"decision": "approved", "manifest_sha256": "expected", "scope": "DEV_DIAGNOSTIC_ONLY", "human_reply": "Approve v5"}
    validate_approval(approval, manifest)
    with pytest.raises(ValueError):
        validate_approval(dict(approval, human_reply=""), manifest)


def test_unknown_teaching_grammar_is_not_silently_classified(catalog):
    task = catalog.meta_validation[0]
    changed = replace(task, events=(replace(task.events[0], correction="new grammar"), *task.events[1:]))
    with pytest.raises(ValueError, match="Unrecognized"):
        taught_facts(changed)
