"""High-signal contract tests for Research/20 task generation and split firewall.

These tests verify the behavioral contracts of the DEV-only task generator:

  - Byte-deterministic catalogs across repeated calls
  - Different root seeds change every family while preserving schema/counts
  - Both splits contain exactly the three families; every task has 2–5 events
    and all six probe categories
  - Train/validation rule, assignment, composition, and paraphrase fingerprint
    intersections are empty
  - Training operands/events do not reappear in transfer probes where prohibited
  - Invalid/test split cannot be constructed or requested
  - Collision regeneration is deterministic; firewall failure is fail-closed
  - Task fingerprints do not appear in rendered observation/probe messages

No network or model download is required.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from oczy.experiments.meta_cortex.contracts import (
    ContractError,
    DevSplit,
    DialogueMessage,
    LearningEvent,
    MetaTask,
    OutcomeCode,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    SplitFirewallAudit,
    TaskFamily,
    TaskGeneratorConfig,
)
from oczy.experiments.meta_cortex.taskgen import (
    MAX_COLLISION_NONCE,
    SplitFirewallError,
    assert_split_firewall,
    audit_split_firewall,
    build_dev_catalog,
    generate_contextual_remapping_task,
    generate_finite_state_task,
    generate_rule_transformation_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SMOKE_CONFIG = TaskGeneratorConfig(
    root_seed=20260709,
    train_tasks_per_family=2,
    validation_tasks_per_family=2,
    min_events=2,
    max_events=2,
)

_ALL_FAMILIES = (
    TaskFamily.CONTEXTUAL_REMAP,
    TaskFamily.RULE_TRANSFORMATION,
    TaskFamily.FINITE_STATE,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_config_byte_identical_catalog(self) -> None:
        cat1 = build_dev_catalog(_SMOKE_CONFIG)
        cat2 = build_dev_catalog(_SMOKE_CONFIG)
        assert cat1.catalog_sha256 == cat2.catalog_sha256
        assert cat1.meta_train == cat2.meta_train
        assert cat1.meta_validation == cat2.meta_validation
        assert cat1.split_audit == cat2.split_audit

    def test_same_config_byte_identical_fingerprints(self) -> None:
        cat1 = build_dev_catalog(_SMOKE_CONFIG)
        cat2 = build_dev_catalog(_SMOKE_CONFIG)
        for t1, t2 in zip(cat1.meta_train, cat2.meta_train, strict=True):
            assert t1.rule_fingerprint == t2.rule_fingerprint
            assert t1.assignment_fingerprint == t2.assignment_fingerprint
            assert t1.composition_fingerprint == t2.composition_fingerprint
            assert t1.paraphrase_group_fingerprint == t2.paraphrase_group_fingerprint

    def test_same_config_identical_ordering(self) -> None:
        cat1 = build_dev_catalog(_SMOKE_CONFIG)
        cat2 = build_dev_catalog(_SMOKE_CONFIG)
        for t1, t2 in zip(cat1.meta_train, cat2.meta_train, strict=True):
            assert t1.family == t2.family
            assert t1.split == t2.split
            # Event content must be byte-identical.
            for e1, e2 in zip(t1.events, t2.events, strict=True):
                assert e1 == e2
            # Probe content must be byte-identical.
            for kind in ProbeKind:
                assert t1.probes.by_kind(kind) == t2.probes.by_kind(kind)

    def test_per_family_generator_deterministic(self) -> None:
        for gen in (
            generate_contextual_remapping_task,
            generate_rule_transformation_task,
            generate_finite_state_task,
        ):
            t1 = gen(0, _SMOKE_CONFIG)
            t2 = gen(0, _SMOKE_CONFIG)
            assert t1 == t2

    def test_different_key_different_task(self) -> None:
        for gen in (
            generate_contextual_remapping_task,
            generate_rule_transformation_task,
            generate_finite_state_task,
        ):
            t0 = gen(0, _SMOKE_CONFIG)
            t1 = gen(1, _SMOKE_CONFIG)
            assert t0.rule_fingerprint != t1.rule_fingerprint


class TestDifferentSeed:
    def test_different_seed_changes_every_family(self) -> None:
        cfg_a = TaskGeneratorConfig(
            root_seed=20260709,
            train_tasks_per_family=2,
            validation_tasks_per_family=2,
            min_events=2,
            max_events=2,
        )
        cfg_b = TaskGeneratorConfig(
            root_seed=99999,
            train_tasks_per_family=2,
            validation_tasks_per_family=2,
            min_events=2,
            max_events=2,
        )
        cat_a = build_dev_catalog(cfg_a)
        cat_b = build_dev_catalog(cfg_b)
        assert cat_a.catalog_sha256 != cat_b.catalog_sha256
        # Every train task should have a different rule fingerprint.
        for ta, tb in zip(cat_a.meta_train, cat_b.meta_train, strict=True):
            assert ta.rule_fingerprint != tb.rule_fingerprint

    def test_different_seed_preserves_counts(self) -> None:
        cfg_a = TaskGeneratorConfig(
            root_seed=20260709,
            train_tasks_per_family=2,
            validation_tasks_per_family=2,
        )
        cfg_b = TaskGeneratorConfig(
            root_seed=11111,
            train_tasks_per_family=2,
            validation_tasks_per_family=2,
        )
        cat_a = build_dev_catalog(cfg_a)
        cat_b = build_dev_catalog(cfg_b)
        assert len(cat_a.meta_train) == len(cat_b.meta_train)
        assert len(cat_a.meta_validation) == len(cat_b.meta_validation)


# ---------------------------------------------------------------------------
# Structure: families, events, probes
# ---------------------------------------------------------------------------


class TestTaskStructure:
    def test_both_splits_contain_all_three_families(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for split_tasks in (cat.meta_train, cat.meta_validation):
            families = {t.family for t in split_tasks}
            assert families == set(_ALL_FAMILIES)

    def test_every_task_has_2_to_5_events(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            assert 2 <= len(task.events) <= 5

    def test_every_task_has_all_six_probe_categories(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            for kind in ProbeKind:
                probes = task.probes.by_kind(kind)
                assert len(probes) >= 1, (
                    f"Task {task.family.value} split={task.split.value} "
                    f"has empty {kind.value} probe category"
                )

    def test_probe_kinds_match_category(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            for kind in ProbeKind:
                for probe in task.probes.by_kind(kind):
                    assert probe.kind == kind

    def test_events_are_learning_events(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            for event in task.events:
                assert isinstance(event, LearningEvent)
                assert len(event.observation_messages) >= 1
                assert event.attempted_behavior
                assert event.correction
                assert isinstance(event.outcome, OutcomeCode)

    def test_probes_are_probe_cases(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            for kind in ProbeKind:
                for probe in task.probes.by_kind(kind):
                    assert isinstance(probe, ProbeCase)
                    assert len(probe.messages) >= 1
                    assert probe.expected_response
                    for msg in probe.messages:
                        assert isinstance(msg, DialogueMessage)

    def test_fingerprints_are_64_char_hex(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            for fp in (
                task.rule_fingerprint,
                task.assignment_fingerprint,
                task.composition_fingerprint,
                task.paraphrase_group_fingerprint,
            ):
                assert len(fp) == 64
                int(fp, 16)  # must be valid hex

    def test_split_assignment_before_rendering(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train:
            assert task.split == DevSplit.META_TRAIN
        for task in cat.meta_validation:
            assert task.split == DevSplit.META_VALIDATION

    def test_catalog_digest_is_64_char_hex(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        assert len(cat.catalog_sha256) == 64
        int(cat.catalog_sha256, 16)


# ---------------------------------------------------------------------------
# Split firewall
# ---------------------------------------------------------------------------


class TestSplitFirewall:
    def test_no_fingerprint_overlap(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        audit = cat.split_audit
        assert audit.rule_overlap == 0
        assert audit.assignment_overlap == 0
        assert audit.composition_overlap == 0
        assert audit.paraphrase_overlap == 0

    def test_audit_passed_property(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        assert cat.split_audit.passed is True

    def test_assert_split_firewall_passes(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        assert_split_firewall(cat.split_audit)  # should not raise

    def test_assert_split_firewall_raises_on_overlap(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        # Manually create an overlapping audit.
        bad_audit = SplitFirewallAudit(
            train_rule_digests=cat.split_audit.train_rule_digests,
            validation_rule_digests=cat.split_audit.train_rule_digests,  # overlap!
            train_assignment_digests=cat.split_audit.train_assignment_digests,
            validation_assignment_digests=cat.split_audit.validation_assignment_digests,
            train_composition_digests=cat.split_audit.train_composition_digests,
            validation_composition_digests=cat.split_audit.validation_composition_digests,
            train_paraphrase_digests=cat.split_audit.train_paraphrase_digests,
            validation_paraphrase_digests=cat.split_audit.validation_paraphrase_digests,
            rule_overlap=1,
            assignment_overlap=0,
            composition_overlap=0,
            paraphrase_overlap=0,
            train_task_count=cat.split_audit.train_task_count,
            validation_task_count=cat.split_audit.validation_task_count,
        )
        with pytest.raises(SplitFirewallError, match="rule overlap"):
            assert_split_firewall(bad_audit)

    def test_audit_split_firewall_directly(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        audit = audit_split_firewall(cat.meta_train, cat.meta_validation)
        assert audit.passed is True
        assert audit.train_task_count == len(cat.meta_train)
        assert audit.validation_task_count == len(cat.meta_validation)

    def test_firewall_detects_overlap_directly(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        # Mix train tasks into validation to create overlap.
        mixed_validation = cat.meta_train[:1] + cat.meta_validation
        audit = audit_split_firewall(cat.meta_train, mixed_validation)
        assert audit.passed is False
        assert audit.rule_overlap > 0


# ---------------------------------------------------------------------------
# Invalid/test split rejection
# ---------------------------------------------------------------------------


class TestSplitRejection:
    def test_no_test_split_member(self) -> None:
        values = {m.value for m in DevSplit}
        assert "meta_test" not in values
        assert values == {"meta_train", "meta_validation"}

    def test_invalid_split_string_fails(self) -> None:
        with pytest.raises(ValueError):
            DevSplit("meta_test")

    def test_tasks_for_rejects_string(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        with pytest.raises(ContractError):
            cat.tasks_for("meta_train")  # type: ignore[arg-type]

    def test_tasks_for_accepts_only_dev_split(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        train = cat.tasks_for(DevSplit.META_TRAIN)
        val = cat.tasks_for(DevSplit.META_VALIDATION)
        assert train == cat.meta_train
        assert val == cat.meta_validation


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_min_events_must_be_at_least_1(self) -> None:
        with pytest.raises(ContractError):
            TaskGeneratorConfig(
                root_seed=0,
                train_tasks_per_family=1,
                validation_tasks_per_family=1,
                min_events=0,
                max_events=5,
            )

    def test_min_must_be_le_max(self) -> None:
        with pytest.raises(ContractError):
            TaskGeneratorConfig(
                root_seed=0,
                train_tasks_per_family=1,
                validation_tasks_per_family=1,
                min_events=3,
                max_events=2,
            )

    def test_max_events_must_be_le_5(self) -> None:
        with pytest.raises(ContractError):
            TaskGeneratorConfig(
                root_seed=0,
                train_tasks_per_family=1,
                validation_tasks_per_family=1,
                min_events=1,
                max_events=6,
            )

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ContractError):
            TaskGeneratorConfig(
                root_seed=-1,
                train_tasks_per_family=1,
                validation_tasks_per_family=1,
            )

    def test_zero_tasks_per_family_rejected(self) -> None:
        with pytest.raises(ContractError):
            TaskGeneratorConfig(
                root_seed=0,
                train_tasks_per_family=0,
                validation_tasks_per_family=1,
            )


# ---------------------------------------------------------------------------
# Fingerprint not in rendered text
# ---------------------------------------------------------------------------


class TestFingerprintNotInText:
    def test_fingerprints_not_in_messages(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            fps = {
                task.rule_fingerprint,
                task.assignment_fingerprint,
                task.composition_fingerprint,
                task.paraphrase_group_fingerprint,
            }
            # Check event messages.
            for event in task.events:
                for msg in event.observation_messages:
                    for fp in fps:
                        assert fp not in msg.content
                assert fp not in event.attempted_behavior
                assert fp not in event.correction
            # Check probe messages.
            for kind in ProbeKind:
                for probe in task.probes.by_kind(kind):
                    for msg in probe.messages:
                        for fp in fps:
                            assert fp not in msg.content
                    for fp in fps:
                        assert fp not in probe.expected_response


# ---------------------------------------------------------------------------
# Training operands not reused in transfer probes
# ---------------------------------------------------------------------------


class TestTransferDisjoint:
    def test_transfer_probes_do_not_reuse_teaching_sentences(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        for task in cat.meta_train + cat.meta_validation:
            teaching_texts: set[str] = set()
            for event in task.events:
                for msg in event.observation_messages:
                    teaching_texts.add(msg.content)
                teaching_texts.add(event.attempted_behavior)
                teaching_texts.add(event.correction)

            for probe in task.probes.transfer:
                for msg in probe.messages:
                    # No teaching sentence should appear verbatim in transfer.
                    for teaching in teaching_texts:
                        if len(teaching) > 10:
                            assert teaching not in msg.content, (
                                f"Transfer probe reuses teaching text: "
                                f"'{teaching[:40]}...' in '{msg.content[:40]}...'"
                            )


# ---------------------------------------------------------------------------
# Collision regeneration
# ---------------------------------------------------------------------------


class TestCollisionRegeneration:
    def test_collision_nonce_exists(self) -> None:
        assert MAX_COLLISION_NONCE > 0

    def test_catalog_builds_without_collision_error(self) -> None:
        # With normal config, no collision should occur.
        cfg = TaskGeneratorConfig(
            root_seed=42,
            train_tasks_per_family=3,
            validation_tasks_per_family=3,
        )
        cat = build_dev_catalog(cfg)
        assert cat.split_audit.passed is True

    def test_large_catalog_no_collision(self) -> None:
        cfg = TaskGeneratorConfig(
            root_seed=12345,
            train_tasks_per_family=5,
            validation_tasks_per_family=5,
        )
        cat = build_dev_catalog(cfg)
        assert cat.split_audit.passed is True
        assert len(cat.meta_train) == 15
        assert len(cat.meta_validation) == 15


# ---------------------------------------------------------------------------
# MetaTask immutability and validation
# ---------------------------------------------------------------------------


class TestMetaTaskValidation:
    def test_meta_task_is_frozen(self) -> None:
        task = generate_contextual_remapping_task(0, _SMOKE_CONFIG)
        with pytest.raises(FrozenInstanceError):
            task.family = TaskFamily.RULE_TRANSFORMATION  # type: ignore[misc]

    def test_meta_task_rejects_wrong_event_count(self) -> None:
        with pytest.raises(ContractError):
            MetaTask(
                family=TaskFamily.CONTEXTUAL_REMAP,
                split=DevSplit.META_TRAIN,
                events=(),
                probes=generate_contextual_remapping_task(0, _SMOKE_CONFIG).probes,
                rule_fingerprint="a" * 64,
                assignment_fingerprint="b" * 64,
                composition_fingerprint="c" * 64,
                paraphrase_group_fingerprint="d" * 64,
            )

    def test_meta_task_rejects_short_fingerprint(self) -> None:
        task = generate_contextual_remapping_task(0, _SMOKE_CONFIG)
        with pytest.raises(ContractError):
            MetaTask(
                family=task.family,
                split=task.split,
                events=task.events,
                probes=task.probes,
                rule_fingerprint="short",
                assignment_fingerprint="b" * 64,
                composition_fingerprint="c" * 64,
                paraphrase_group_fingerprint="d" * 64,
            )


# ---------------------------------------------------------------------------
# ProbeBattery validation
# ---------------------------------------------------------------------------


class TestProbeBatteryValidation:
    def test_empty_category_rejected(self) -> None:
        task = generate_contextual_remapping_task(0, _SMOKE_CONFIG)
        with pytest.raises(ContractError):
            ProbeBattery(
                pre=(),
                same_rule=task.probes.same_rule,
                transfer=task.probes.transfer,
                composition=task.probes.composition,
                specificity=task.probes.specificity,
                oracle_context=task.probes.oracle_context,
            )

    def test_by_kind_returns_correct_category(self) -> None:
        task = generate_contextual_remapping_task(0, _SMOKE_CONFIG)
        for kind in ProbeKind:
            probes = task.probes.by_kind(kind)
            assert all(p.kind == kind for p in probes)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_split_audit_to_json_is_canonical(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        j1 = cat.split_audit.to_json()
        j2 = cat.split_audit.to_json()
        assert j1 == j2
        parsed = json.loads(j1)
        assert parsed["passed"] is True
        assert parsed["rule_overlap"] == 0

    def test_split_audit_to_json_no_nan(self) -> None:
        cat = build_dev_catalog(_SMOKE_CONFIG)
        j = cat.split_audit.to_json()
        assert "NaN" not in j
        assert "Infinity" not in j
