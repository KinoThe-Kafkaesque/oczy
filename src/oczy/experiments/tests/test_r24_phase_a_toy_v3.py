from __future__ import annotations

from dataclasses import replace

import torch

from oczy.experiments.r24_tiny_decoder.phase_a_toy_v3 import (
    BASE_CONFIG,
    phase_a_tasks,
    suite_sha256,
    train_phase_a_toy_v3,
)
from oczy.experiments.r24_tiny_decoder.toy_catalog_v3 import (
    ToySplit,
    build_toy_catalog_v3,
)
from oczy.experiments.r24_tiny_decoder.toy_features_v3 import (
    batch_query_features,
    batch_teaching_features,
    feature_definition_sha256,
    structured_rule_state,
    teaching_features,
)


def test_fixed_features_are_id_free_deterministic_and_c4_changes_only_feedback() -> None:
    catalog = build_toy_catalog_v3()
    task, donor = catalog.donor_pairs(ToySplit.CORTEX_META_DEV)[0]
    correct = teaching_features(task)
    repeated = teaching_features(task)
    wrong = teaching_features(task, donor=donor)
    assert torch.equal(correct, repeated)
    assert correct.shape == wrong.shape == (3, 64)
    assert torch.equal(correct[:, 0:12], wrong[:, 0:12])
    assert torch.equal(correct[:, 18:19], wrong[:, 18:19])
    assert not torch.equal(correct[:, 12:18], wrong[:, 12:18])
    assert batch_teaching_features([task, donor]).shape == (2, 3, 64)
    assert batch_query_features([task.heldout[0], donor.heldout[0]]).shape == (2, 64)
    assert len(feature_definition_sha256()) == 64


def test_structured_oracle_encodes_only_registered_rule_parameters() -> None:
    catalog = build_toy_catalog_v3()
    first = catalog.candidates[0]
    state = structured_rule_state(first)
    assert state.shape == (64,)
    assert state.dtype == torch.float32
    assert torch.count_nonzero(state[:3]).item() == 1
    assert set(state[3:9].tolist()) <= {-1.0, 1.0}
    assert torch.equal(state, structured_rule_state(first))
    assert not torch.equal(state, structured_rule_state(catalog.candidates[-1]))


def test_phase_a_split_excludes_sealed_test_and_is_frozen() -> None:
    catalog = build_toy_catalog_v3()
    train, validation = phase_a_tasks(catalog)
    assert len(train) == 92
    assert len(validation) == 50
    train_ids = {task.rule_fingerprint for task in train}
    validation_ids = {task.rule_fingerprint for task in validation}
    test_ids = {task.rule_fingerprint for task in catalog.sealed_test}
    assert train_ids.isdisjoint(validation_ids)
    assert test_ids.isdisjoint(train_ids | validation_ids)
    assert len(suite_sha256()) == 64


def test_one_step_phase_a_writes_hash_bound_reloadable_artifact(tmp_path) -> None:
    config = replace(
        BASE_CONFIG,
        steps=1,
        batch_size=8,
        dropout=0.0,
    )
    artifact = train_phase_a_toy_v3(config, output_dir=tmp_path)
    assert artifact["sealed_test_accessed"] is False
    assert artifact["train_task_count"] == 92
    assert artifact["validation_task_count"] == 50
    assert artifact["validation_example_count"] == 3_200
    assert artifact["weight_hash"] == artifact["frozen_hash"]
    assert artifact["reload_verification"]["output_bit_equal"] is True
    assert artifact["reload_verification"]["weight_hash"] == artifact["weight_hash"]
    assert len(artifact["files"]["decoder.pt"]["sha256"]) == 64
    assert (tmp_path / "decoder.pt").is_file()
    assert (tmp_path / "artifact.json").is_file()
