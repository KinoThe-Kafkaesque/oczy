from __future__ import annotations

import hashlib
import itertools
import json

import pytest

from oczy.experiments.r24_tiny_decoder.toy_catalog_v3 import (
    BASE64_ALPHABET,
    BIT_WIDTH,
    CATALOG_SCHEMA_VERSION,
    HELDOUT_INPUTS,
    HELDOUT_SYMBOLS,
    INDEX_MASK,
    REGISTERED_CANDIDATE_COUNT,
    ROTATIONS,
    TEACHING_INPUTS,
    TEACHING_SYMBOLS,
    ToyFamily,
    ToyRule,
    ToySplit,
    brute_force_identifiability_audit,
    build_toy_catalog_v3,
    identify_rule,
    index_for_symbol,
    matching_candidates,
    registered_candidates,
    rotate_left_6,
    symbol_for_index,
)


def _rule_fingerprints(tasks: object) -> set[str]:
    return {task.rule_fingerprint for task in tasks}  # type: ignore[union-attr]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_base64_six_bit_rule_and_exact_byte_protocol() -> None:
    assert BASE64_ALPHABET == b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    assert len(BASE64_ALPHABET) == 1 << BIT_WIDTH == 64
    assert INDEX_MASK == 63
    assert ROTATIONS == (0, 1, 2)
    assert TEACHING_INPUTS == (0, 1, 2)
    assert TEACHING_SYMBOLS == b"ABC"
    assert HELDOUT_INPUTS == tuple(range(3, 64))
    assert HELDOUT_SYMBOLS == BASE64_ALPHABET[3:]
    assert set(TEACHING_INPUTS).isdisjoint(HELDOUT_INPUTS)

    for index, symbol in enumerate(BASE64_ALPHABET):
        assert symbol_for_index(index) == bytes((symbol,))
        assert index_for_symbol(bytes((symbol,))) == index
        assert index_for_symbol(chr(symbol)) == index

    assert rotate_left_6(0b100001, 0) == 0b100001
    assert rotate_left_6(0b100001, 1) == 0b000011
    assert rotate_left_6(0b100001, 2) == 0b000110
    rule = ToyRule(rotation=2, mask=0b101010)
    assert rule.apply(0b100001) == 0b000110 ^ 0b101010

    catalog = build_toy_catalog_v3()
    for task in catalog.all_tasks:
        assert task.rule.family in set(ToyFamily)
        assert task.rule.family is tuple(ToyFamily)[task.rule.rotation]
        assert len(task.teaching) == 3
        assert len(task.heldout) == 61
        for interaction in task.teaching + task.heldout:
            expected_symbol = bytes((BASE64_ALPHABET[interaction.input_value],))
            assert interaction.query_bytes == b"x=" + expected_symbol
            assert interaction.answer_bytes in [bytes((value,)) for value in BASE64_ALPHABET]
            assert len(interaction.answer_bytes) == 1
            assert interaction.output_value == task.rule.apply(interaction.input_value)


@pytest.mark.parametrize("bad", [-1, 64, True])
def test_six_bit_validation_rejects_out_of_range_values(bad: int) -> None:
    with pytest.raises(ValueError):
        ToyRule(rotation=0, mask=bad)
    with pytest.raises(ValueError):
        rotate_left_6(bad, 0)


def test_all_192_candidates_require_exactly_three_rows_and_are_functionally_unique() -> None:
    candidates = registered_candidates()
    assert REGISTERED_CANDIDATE_COUNT == 192
    assert len(candidates) == len(ROTATIONS) * 64 == REGISTERED_CANDIDATE_COUNT
    assert {(rule.rotation, rule.mask) for rule in candidates} == set(
        itertools.product(ROTATIONS, range(64))
    )
    assert len({rule.fingerprint for rule in candidates}) == 192
    assert len({tuple(rule.apply(value) for value in range(64)) for rule in candidates}) == 192
    assert (
        len({tuple(rule.apply(value) for value in HELDOUT_INPUTS) for rule in candidates})
        == 192
    )

    audit = brute_force_identifiability_audit(candidates)
    assert audit.exactly_three is True
    assert audit.all_identified is True
    assert audit.candidate_count == 192
    assert audit.unique_after_one == 0
    assert audit.minimum_matches_after_one == audit.maximum_matches_after_one == 3
    assert audit.unique_after_two == 0
    assert audit.minimum_matches_after_two == audit.maximum_matches_after_two == 3
    assert audit.unique_after_three == 192
    assert audit.ambiguous_after_three == 0
    assert audit.minimum_matches_after_three == audit.maximum_matches_after_three == 1

    catalog = build_toy_catalog_v3()
    for task in catalog.all_tasks:
        assert tuple(item.input_value for item in task.teaching) == TEACHING_INPUTS
        assert tuple(item.input_value for item in task.heldout) == HELDOUT_INPUTS
        assert task.teaching[0].answer_bytes == task.teaching[1].answer_bytes
        assert len(matching_candidates(task.teaching[:1], candidates=candidates)) == 3
        assert len(matching_candidates(task.teaching[:2], candidates=candidates)) == 3
        assert matching_candidates(task.teaching, candidates=candidates) == (task.rule,)
        assert identify_rule(task.teaching, candidates=candidates) == task.rule

def test_fixed_five_way_split_is_disjoint_complete_and_test_firewalled() -> None:
    catalog = build_toy_catalog_v3(root_seed=24_003)
    expected_counts = {
        ToySplit.CORTEX_META_TRAIN: 50,
        ToySplit.CORTEX_META_DEV: 25,
        ToySplit.SEALED_TEST: 50,
        ToySplit.ORGAN_DEV: 25,
        ToySplit.ORGAN_TRAIN: 42,
    }
    assert sum(expected_counts.values()) == 192
    assert {split: len(catalog.tasks_for_split(split)) for split in ToySplit} == expected_counts

    split_sets = {split: _rule_fingerprints(catalog.tasks_for_split(split)) for split in ToySplit}
    for left, right in itertools.combinations(ToySplit, 2):
        assert split_sets[left].isdisjoint(split_sets[right])
    assert set().union(*split_sets.values()) == {rule.fingerprint for rule in catalog.candidates}

    sealed = split_sets[ToySplit.SEALED_TEST]
    assert sealed.isdisjoint(_rule_fingerprints(catalog.non_test_tasks))
    assert catalog.phase_a_organ_corpus() == catalog.organ_train

    unioned_phase_a = catalog.phase_a_organ_corpus(include_all_non_test_development=True)
    assert _rule_fingerprints(unioned_phase_a) == set().union(
        split_sets[ToySplit.CORTEX_META_TRAIN],
        split_sets[ToySplit.CORTEX_META_DEV],
        split_sets[ToySplit.ORGAN_DEV],
        split_sets[ToySplit.ORGAN_TRAIN],
    )
    assert len(unioned_phase_a) == 142
    assert sealed.isdisjoint(_rule_fingerprints(unioned_phase_a))
    # Deliberate corpus union does not alter the catalog's disjoint partitions.
    assert len(catalog.organ_train) == 42


def test_every_task_has_deterministic_same_family_counterfactual_donor() -> None:
    catalog = build_toy_catalog_v3()
    for split in ToySplit:
        pairs = catalog.donor_pairs(split)
        assert len(pairs) == len(catalog.tasks_for_split(split))
        for task, donor in pairs:
            assert donor.split is task.split
            assert donor.rule.family is task.rule.family
            assert donor.rule.rotation == task.rule.rotation
            assert donor.rule_fingerprint != task.rule_fingerprint
            assert donor.donor_rule_fingerprint != ""
            assert [item.query_bytes for item in donor.teaching] == [
                item.query_bytes for item in task.teaching
            ]
            for target_row, donor_row in zip(task.teaching, donor.teaching, strict=True):
                assert donor_row.answer_bytes != target_row.answer_bytes
                assert len(donor_row.answer_bytes) == len(target_row.answer_bytes) == 1
            assert len(donor.teaching_feedback) == len(task.teaching_feedback)


def test_model_facing_records_have_no_task_ids_or_catalog_metadata() -> None:
    catalog = build_toy_catalog_v3()
    for task in catalog.all_tasks:
        payload = task.model_facing_dict()
        assert set(payload) == {"teaching", "heldout"}
        assert payload["teaching"] == list(task.teaching_records)
        assert payload["heldout"] == list(task.heldout_records)
        for record in task.teaching_records + task.heldout_records:
            assert set(record) == {"query", "answer"}
            assert record["query"].startswith("x=")
            assert len(record["query"].encode("ascii")) == 3
            assert len(record["answer"].encode("ascii")) == 1
        serialized = json.dumps(payload, sort_keys=True)
        assert "fingerprint" not in serialized
        assert "task_id" not in serialized
        assert "split" not in serialized


def test_catalog_hashes_and_allocations_are_stable_and_seeded() -> None:
    first = build_toy_catalog_v3(root_seed=777)
    repeated = build_toy_catalog_v3(root_seed=777)
    other_seed = build_toy_catalog_v3(root_seed=778)

    assert first.registry_sha256 == repeated.registry_sha256 == other_seed.registry_sha256
    assert first.identifiability == repeated.identifiability == other_seed.identifiability
    assert first.split_sha256 == repeated.split_sha256
    assert first.manifest_sha256 == repeated.manifest_sha256
    assert [task.canonical_dict() for task in first.all_tasks] == [
        task.canonical_dict() for task in repeated.all_tasks
    ]

    assert first.manifest_sha256 != other_seed.manifest_sha256
    assert (
        first.split_sha256[ToySplit.SEALED_TEST.value]
        != other_seed.split_sha256[ToySplit.SEALED_TEST.value]
    )
    assert _canonical_sha256(first.manifest_dict()) == first.manifest_sha256
    assert first.manifest_dict()["schema_version"] == CATALOG_SCHEMA_VERSION
    assert len(first.registry_sha256) == len(first.manifest_sha256) == 64
    assert all(len(value) == 64 for value in first.split_sha256.values())


def test_split_sizes_are_protocol_fixed() -> None:
    with pytest.raises(ValueError, match="split sizes are fixed"):
        build_toy_catalog_v3(cortex_meta_dev_size=24)
