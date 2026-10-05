"""Focused tests for the R20 task-support repair (taskgen lineage v2-dev).

These tests cover the contract of the repaired DEV task generator and of its
independently implemented checker:

  - the unchanged v1 generator still reproduces the recorded instrument digest,
    so historical tasks and failures keep reproducing byte-for-byte;
  - the v2 lineage is deterministic and passes the unchanged split firewall;
  - every defect class counted by the 2026-09-12 audit is zero on v2 and is
    still *detected* on v1 (the checker is not vacuously green);
  - support certificates verify, and deleting a required teaching fact or
    flipping a target makes admission fail;
  - the certificate report keeps its sign-off counts split (verified vs
    pre-learning baseline vs failed) and its mutation rows tagged by kind
    (delete_fact / flip_target), so nothing gets folded back;
  - the finite-state composition decision depends on the taught goal (paired
    counterfactual), and the contextual composition chain is typed and taught;
  - no observable question carries two different targets.

No model, scorer, sealed file, calibration file or meta-test task is involved.
"""

from __future__ import annotations

import dataclasses

import pytest

from oczy.experiments.meta_cortex.contracts import TaskGeneratorConfig
from oczy.experiments.meta_cortex.taskgen import build_dev_catalog
from oczy.experiments.meta_cortex.taskgen_v2 import (
    TASKGEN_SCHEMA_V2,
    build_dev_catalog_v2,
)
from scripts.r20_task_support_check import (
    EXPECTED_V1_DEFECTS,
    check_catalog,
    verify_certificates,
)

RECORDED_ROOT_SEED = 20260709
RECORDED_CONFIG = dict(
    root_seed=RECORDED_ROOT_SEED,
    train_tasks_per_family=30,
    validation_tasks_per_family=5,
)
# Digest of the materialized v2-instrument public DEV tasks
# (public/tasks/meta_train.jsonl + meta_validation_tuning.jsonl) of the
# r20-int8-dev-calibration-v6 campaign, built from the unchanged v1 generator
# with the recorded config above.
RECORDED_V1_CATALOG_SHA256 = (
    "c0034bcde0d21e05d151b6a08d34f8cad95c7e4317c8eb919ec3d2fa3fdabb79"
)

SMALL_CONFIG = TaskGeneratorConfig(
    root_seed=RECORDED_ROOT_SEED, train_tasks_per_family=6, validation_tasks_per_family=2
)


@pytest.fixture(scope="module")
def v1_catalog():
    return build_dev_catalog(TaskGeneratorConfig(**RECORDED_CONFIG))


@pytest.fixture(scope="module")
def v2_small():
    return build_dev_catalog_v2(SMALL_CONFIG)


def _zero_classes(summary):
    return {cls: counts for cls, counts in summary.items() if counts["total"]}


def test_v1_generator_still_reproduces_the_recorded_instrument(v1_catalog):
    """Guards "historical failures must keep reproducing": the v1 generator and
    its recorded config still yield the exact catalog of the materialized
    instrument."""
    assert v1_catalog.catalog_sha256 == RECORDED_V1_CATALOG_SHA256


def test_v2_lineage_name_is_distinct_from_v1():
    assert TASKGEN_SCHEMA_V2 == "oczy/meta-cortex/taskgen/v2-dev"
    assert TASKGEN_SCHEMA_V2 != "oczy/meta-cortex/taskgen/v1-dev"


def test_v2_catalog_is_byte_deterministic():
    first, first_bundle = build_dev_catalog_v2(SMALL_CONFIG)
    second, second_bundle = build_dev_catalog_v2(SMALL_CONFIG)
    assert first.catalog_sha256 == second.catalog_sha256
    assert first_bundle.as_dict() == second_bundle.as_dict()
    assert [
        (t.family.value, [e.correction for e in t.events]) for t in first.meta_train
    ] == [
        (t.family.value, [e.correction for e in t.events]) for t in second.meta_train
    ]


def test_v2_passes_the_unchanged_split_firewall(v2_small):
    catalog, _bundle = v2_small
    audit = catalog.split_audit
    assert audit.rule_overlap == 0
    assert audit.assignment_overlap == 0
    assert audit.composition_overlap == 0
    assert audit.paraphrase_overlap == 0


def test_v2_tasks_teach_distinct_facts_within_the_event_budget(v2_small):
    catalog, _bundle = v2_small
    for task in (*catalog.meta_train, *catalog.meta_validation):
        corrections = [event.correction for event in task.events]
        assert 2 <= len(corrections) <= 5
        assert len(set(corrections)) == len(corrections), "silently duplicated teaching"


def test_v2_defect_counts_are_all_zero(v2_small):
    catalog, _bundle = v2_small
    summary = check_catalog(catalog)["summary"]
    assert _zero_classes(summary) == {}


def test_checker_still_detects_the_v1_defects():
    """The checker must not be vacuously green: the same defect classes are
    still found on the unchanged v1 generator."""
    small_v1 = build_dev_catalog(
        TaskGeneratorConfig(
            root_seed=RECORDED_ROOT_SEED,
            train_tasks_per_family=6,
            validation_tasks_per_family=2,
        )
    )
    summary = check_catalog(small_v1)["summary"]
    observed = {cls: counts["total"] for cls, counts in summary.items()}
    for cls in (
        "contextual_same_rule_lookup_untaught",
        "contextual_transfer_lookup_untaught",
        "fsm_transfer_edge_untaught",
        "composition_second_operand_undefined",
        "fsm_composition_action_untaught",
        "fsm_action_mapping_untaught",
        "fsm_goal_untaught",
    ):
        assert observed.get(cls, 0) > 0, f"checker lost the v1 defect {cls}"
    assert set(EXPECTED_V1_DEFECTS) <= set(summary), "class inventory drifted"


def test_v1_recorded_defect_counts_are_reproduced(v1_catalog):
    summary = check_catalog(v1_catalog)["summary"]
    for cls, expected in EXPECTED_V1_DEFECTS.items():
        counts = summary[cls]
        for split_name in ("train", "tuning", "total"):
            assert counts[split_name] == expected[split_name], (cls, split_name)
            assert counts["of"][split_name] == expected["of"][split_name], (cls, split_name)


def test_support_certificates_verify_and_mutations_fail_admission(v2_small):
    catalog, bundle = v2_small
    report = verify_certificates(catalog, bundle)
    assert report["failed"] == 0, report["failures"][:5]
    assert report["mutations_missed"] == [], report["mutations_missed"][:5]
    assert report["mutation_checks"] > 0
    assert report["mutations_detected"] == report["mutation_checks"]


def test_certificate_report_keeps_the_split_counts_and_per_kind_tags(v2_small):
    """Sign-off honesty of the report shape: the counts the human signer reads
    must stay split. Fails if they are folded back (pre-learning baseline into
    ``verified``, or ``failed`` folded from totals) and if the mutation rows'
    ``kind`` tags are swapped or renamed: on a clean run every non-baseline
    certificate reaches the mutation stage, so exactly one ``flip_target`` row
    is emitted per non-baseline certificate and a swapped tag breaks that
    identity. The identity is a clean-run property, NOT a claim about every
    config: a certificate that exits early on a failure status
    (``probe_not_found``, ``certificate_target_differs_from_probe``,
    ``required_teaching_fact_absent``, ``independent_derivation_failed``,
    ``derivation_disagrees``) still counts in ``certificates`` but emits no
    ``flip_target`` row, so ``flip_target == certificates -
    baseline_not_required`` holds only when every non-baseline certificate
    reaches the mutation stage (true for this fixture and for the
    SMALL_CONFIG dry run: 758 == 933 - 175). The two families are not
    interchangeable on this fixture."""
    catalog, bundle = v2_small
    report = verify_certificates(catalog, bundle)

    # certificates partition into verified / baseline_not_required / failed
    assert report["verified"] + report["baseline_not_required"] == report["certificates"]
    assert report["failed"] == len(report["failures"]) == 0
    assert (
        report["verified"] + report["baseline_not_required"] + report["failed"]
        == report["certificates"]
    )
    assert (
        report["failed"]
        == report["certificates"] - report["verified"] - report["baseline_not_required"]
    )

    checks = report["mutation_checks_by_kind"]
    detected = report["mutations_detected_by_kind"]
    assert set(checks) == {"delete_fact", "flip_target"}, "mutation kind tags drifted"
    assert set(detected) == {"delete_fact", "flip_target"}, "mutation kind tags drifted"
    assert sum(checks.values()) == report["mutation_checks"]
    assert sum(detected.values()) == report["mutations_detected"]
    assert all(m["kind"] in {"delete_fact", "flip_target"}
               for m in report["mutations_missed"])
    # one flip-target row per non-baseline certificate that reaches the
    # mutation stage — every one of them on this clean run; a certificate that
    # exits early on a failure status emits no flip row (see the docstring).
    # The delete-a-fact family counts per required fact (or per whole required
    # set).
    assert checks["flip_target"] == report["certificates"] - report["baseline_not_required"]
    assert checks["delete_fact"] > 0
    assert checks["delete_fact"] != checks["flip_target"], \
        "fixture is not discriminating for the kind tags"


def _bundle_with_probes(bundle, entry, broken_probes):
    """A copy of ``bundle`` with one entry's probe tuple replaced."""
    return dataclasses.replace(
        bundle,
        tasks=tuple(
            dataclasses.replace(t, probes=broken_probes) if t is entry else t
            for t in bundle.tasks
        ),
    )


def _first_non_baseline_probe(bundle, *, need_facts=False):
    for entry in bundle.tasks:
        for probe in entry.probes:
            if probe.category != "pre_learning_baseline" and (
                not need_facts or probe.required_facts
            ):
                return entry, probe
    raise AssertionError("fixture has no usable non-baseline certificate")


def _corrupt_expected(bundle):
    """Recorded target mismatches the probe -> ``certificate_target_differs_from_probe``."""
    entry, probe = _first_non_baseline_probe(bundle)
    broken = tuple(
        dataclasses.replace(p, expected_response=p.expected_response + "x")
        if p.probe_id == probe.probe_id else p
        for p in entry.probes
    )
    return _bundle_with_probes(bundle, entry, broken), \
        "certificate_target_differs_from_probe"


def _corrupt_question(bundle):
    """The certificate's question matches no rendered probe -> ``probe_not_found``."""
    entry, probe = _first_non_baseline_probe(bundle)
    broken = tuple(
        dataclasses.replace(p, question=p.question + "?")
        if p.probe_id == probe.probe_id else p
        for p in entry.probes
    )
    return _bundle_with_probes(bundle, entry, broken), "probe_not_found"


def _corrupt_required_fact(bundle):
    """A required fact no teaching event carries -> ``required_teaching_fact_absent``."""
    entry, probe = _first_non_baseline_probe(bundle, need_facts=True)
    fact = probe.required_facts[0]
    untaught = dataclasses.replace(fact, correction=fact.correction + " (never taught)")
    broken = tuple(
        dataclasses.replace(
            p, required_facts=(untaught,) + p.required_facts[1:]
        )
        if p.probe_id == probe.probe_id else p
        for p in entry.probes
    )
    return _bundle_with_probes(bundle, entry, broken), "required_teaching_fact_absent"


def _assert_failed_tracks_one_failure_row(catalog, bundle, breaker):
    """``failed`` is recomputed from the failure rows: exactly one corrupted
    certificate becomes exactly one failure row, and ``failed`` follows it
    while the partition identity holds."""
    clean = verify_certificates(catalog, bundle)
    broken_bundle, expected_status = breaker(bundle)
    report = verify_certificates(catalog, broken_bundle)
    assert report["certificates"] == clean["certificates"]
    assert report["failed"] == 1
    assert report["verified"] == clean["verified"] - 1
    (row,) = report["failures"]
    assert row["status"] == expected_status
    assert row["status"] not in {"verified", "baseline_not_required"}
    assert (
        report["verified"] + report["baseline_not_required"] + report["failed"]
        == report["certificates"]
    )
    assert (
        report["failed"]
        == report["certificates"] - report["verified"] - report["baseline_not_required"]
    )


def test_failed_is_counted_from_failure_status_rows_only(v2_small):
    """``failed`` must be recomputed from the failure rows, not folded out of
    the certificate totals: corrupting one certificate's recorded target moves
    exactly one row into a failure status and ``failed`` follows it."""
    catalog, bundle = v2_small
    _assert_failed_tracks_one_failure_row(catalog, bundle, _corrupt_expected)


def test_failed_counts_every_failure_status_not_only_the_target_mismatch(v2_small):
    """One failure status is not enough of a proof: a ``failed``/``failures``
    recomputation keyed only to ``certificate_target_differs_from_probe`` (the
    status the target corruption above produces) would survive both that case
    and the clean run. Two further corruptions force DIFFERENT failure
    statuses — an unmatched certificate question (``probe_not_found``) and a
    required fact no teaching event carries (``required_teaching_fact_absent``)
    — and ``failed`` must follow each of them through the same failed == 1 /
    one failure row / partition-identity assertions."""
    catalog, bundle = v2_small
    _assert_failed_tracks_one_failure_row(catalog, bundle, _corrupt_question)
    _assert_failed_tracks_one_failure_row(catalog, bundle, _corrupt_required_fact)


def test_fsm_composition_decision_depends_on_the_taught_goal(v2_small):
    catalog, bundle = v2_small
    fsm_entries = [t for t in bundle.tasks if t.family == "finite_state"]
    assert fsm_entries
    for entry in fsm_entries:
        composition = [p for p in entry.probes if p.kind == "composition"][0]
        assert composition.paired_goal_counterfactual, "no registered goal pair"
        for goal, counterfactual_expected in composition.paired_goal_counterfactual:
            assert counterfactual_expected != composition.expected_response
            assert goal
        fact_kinds = {fact.kind for fact in composition.required_facts}
        assert {"fsm_edge", "fsm_action", "fsm_goal"} <= fact_kinds
        goals = [fact for fact in entry.facts if fact.kind == "fsm_goal"]
        assert len(goals) == 1
        actions = [fact for fact in composition.required_facts if fact.kind == "fsm_action"]
        assert actions and actions[0].target in composition.expected_response


def test_contextual_composition_chain_is_typed_and_taught(v2_small):
    catalog, bundle = v2_small
    contextual = [t for t in bundle.tasks if t.family == "contextual_remap"]
    assert contextual
    for entry in contextual:
        composition = [p for p in entry.probes if p.kind == "composition"][0]
        assert len(composition.required_facts) == 2
        (first, second) = composition.required_facts
        # The second lookup key is the first lookup's taught output.
        assert second.key[1] == first.target
        assert composition.expected_response == f"{first.target} then {second.target}"
        taught_keys = {fact.key for fact in entry.facts}
        assert first.key in taught_keys and second.key in taught_keys


def test_specificity_probes_state_their_scope_and_never_contradict_teaching(v2_small):
    catalog, _bundle = v2_small
    report = check_catalog(catalog)
    assert report["summary"]["specificity_scope_unstated"]["total"] == 0
    assert report["summary"]["question_target_contradiction"]["total"] == 0
    assert report["summary"]["specificity_contradiction"]["total"] == 0
