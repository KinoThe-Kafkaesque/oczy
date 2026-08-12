from __future__ import annotations

import pytest
import torch

from oczy.experiments.r24_tiny_decoder.confirmation_v2 import (
    SEED_TUPLES,
    evaluate_gate,
)
from oczy.experiments.r24_tiny_decoder.confirmation_v2 import (
    materialize_case as materialize_confirmation_case,
)
from oczy.experiments.r24_tiny_decoder.confirmation_v2 import (
    suite_sha256 as confirmation_suite_sha256,
)
from oczy.experiments.r24_tiny_decoder.corpus_v2 import (
    build_phase_a_v2_corpus,
    input_conflicts,
)
from oczy.experiments.r24_tiny_decoder.decoder import (
    Conditioning,
    TinyDecoderConfig,
    TinySharedDecoder,
)
from oczy.experiments.r24_tiny_decoder.diagnostics_v2 import build_overfit_cases
from oczy.experiments.r24_tiny_decoder.factorial_v2 import (
    CASE_OVERRIDES as FACTORIAL_CASE_OVERRIDES,
)
from oczy.experiments.r24_tiny_decoder.factorial_v2 import (
    materialize_case as materialize_factorial_case,
)
from oczy.experiments.r24_tiny_decoder.factorial_v2 import (
    select_winner,
    selection_record,
)
from oczy.experiments.r24_tiny_decoder.factorial_v2 import (
    suite_sha256 as factorial_suite_sha256,
)
from oczy.experiments.r24_tiny_decoder.oracle_v2 import OraclePooling, PoolingOracleEncoder
from oczy.experiments.r24_tiny_decoder.phase_a_v2 import (
    PhaseAV2Config,
    _pad_examples,
    seed_everything,
    train_phase_a_v2,
)
from oczy.experiments.r24_tiny_decoder.pretrain import PretrainExample
from oczy.experiments.r24_tiny_decoder.suite_v2 import (
    CASE_OVERRIDES,
    materialize_case,
    suite_sha256,
)
from oczy.experiments.r24_tiny_decoder.vocab import PAD_ID, encode_bytes, encode_with_eos


def _example(query: str, answer: str = "up", rule: str = "r1") -> PretrainExample:
    return PretrainExample(
        rule_fp=rule,
        oracle_text="Complete mapping:\n  jade / wix -> up",
        query_text=query,
        answer_text=answer,
        family="contextual_remap",
        kind="same_rule",
    )


@pytest.mark.parametrize("pooling", ["mean", "cls", "attention", "line_attention"])
def test_v2_oracle_pooling_preserves_shared_r64_contract(pooling: OraclePooling) -> None:
    seed_everything(7)
    encoder = PoolingOracleEncoder(d_model=64, n_layers=2, n_heads=2, pooling=pooling, dropout=0.0)
    texts = [b"a -> up\nb -> down", b"a -> left"]
    width = max(map(len, texts))
    tokens = torch.full((2, width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((2, width), dtype=torch.bool)
    for index, text in enumerate(texts):
        tokens[index, : len(text)] = torch.tensor(list(text))
        mask[index, : len(text)] = True
    state = encoder(tokens, mask)
    assert state.shape == (2, 64)
    assert torch.isfinite(state).all()
    assert not torch.equal(state[0], state[1])


def test_v2_rejects_mixed_query_lengths_instead_of_silently_padding() -> None:
    with pytest.raises(ValueError, match="one query length"):
        _pad_examples([_example("short"), _example("a longer query")], torch.device("cpu"))


def test_teacher_forced_batch_loss_matches_individual_same_length() -> None:
    seed_everything(11)
    decoder = TinySharedDecoder(
        TinyDecoderConfig(d_model=64, n_layers=2, deep_film=True, dropout=0.0)
    )
    examples = [_example("query-a", "up", "r1"), _example("query-b", "forward", "r2")]
    query, answer = _pad_examples(examples, torch.device("cpu"))
    state = torch.randn(2, 64)
    batched = decoder.teacher_forced_loss_per_example(query, state, answer)
    individual = []
    for index, example in enumerate(examples):
        q = torch.tensor([encode_bytes(example.query_text)])
        a = torch.tensor([encode_with_eos(example.answer_text)])
        individual.append(decoder.teacher_forced_loss(q, state[index : index + 1], a))
    assert torch.allclose(batched, torch.stack(individual), atol=1e-6)


def test_prefix_conditioning_keeps_token_logit_shape_and_uses_state() -> None:
    seed_everything(13)
    decoder = TinySharedDecoder(
        TinyDecoderConfig(
            d_model=64,
            n_layers=2,
            conditioning="prefix",
            deep_film=False,
            n_prefix_tokens=4,
            dropout=0.0,
            version="r24-tiny-decoder/v2",
        )
    ).eval()
    query = torch.tensor([encode_bytes("query")])
    zero = torch.zeros(1, 64)
    one = torch.ones(1, 64)
    with torch.inference_mode():
        zero_logits = decoder(query, zero)
        one_logits = decoder(query, one)
    assert zero_logits.shape == one_logits.shape == (1, len("query"), 260)
    assert not torch.equal(zero_logits, one_logits)


def test_seed_before_construction_reproduces_initial_hash() -> None:
    seed_everything(123)
    first = TinySharedDecoder(
        TinyDecoderConfig(d_model=64, n_layers=2, deep_film=True)
    ).parameter_hash()
    seed_everything(123)
    second = TinySharedDecoder(
        TinyDecoderConfig(d_model=64, n_layers=2, deep_film=True)
    ).parameter_hash()
    assert first == second


def test_v2_one_step_training_is_reproducible() -> None:
    examples = [_example("query-a", "up", "r1")]
    config = PhaseAV2Config(
        root_seed=23,
        d_model=64,
        n_layers=2,
        deep_film=True,
        oracle_mode="hash",
        steps=1,
        lr=1e-3,
        batch_size=1,
        dropout=0.0,
        max_train_eval_examples=1,
    )
    first = train_phase_a_v2(config, train_examples=examples, val_examples=examples)
    second = train_phase_a_v2(config, train_examples=examples, val_examples=examples)
    assert first["weight_hash"] == second["weight_hash"]
    assert first["oracle_dev_accuracy"] == second["oracle_dev_accuracy"]


def test_overfit_ladder_holds_rules_constant_only_in_middle_cases() -> None:
    cases = build_overfit_cases(root_seed=123)
    one_rule_train, one_rule_val, _ = cases["one_rule_learned"]
    assert {example.rule_fp for example in one_rule_train} == {
        example.rule_fp for example in one_rule_val
    }
    conflict_train, conflict_val, _ = cases["conflicting_query_learned"]
    assert len({example.query_text for example in conflict_train}) == 1
    assert len({example.answer_text for example in conflict_train}) == 2
    assert conflict_train == conflict_val
    held_train, held_val, _ = cases["held_query_learned"]
    assert {example.rule_fp for example in held_train} == {example.rule_fp for example in held_val}
    assert {(example.rule_fp, example.query_text) for example in held_train}.isdisjoint(
        {(example.rule_fp, example.query_text) for example in held_val}
    )
    assert {example.kind for example in held_val} <= {example.kind for example in held_train}
    unseen_train, unseen_val, _ = cases["unseen_rule_text"]
    assert {example.rule_fp for example in unseen_train}.isdisjoint(
        {example.rule_fp for example in unseen_val}
    )


@pytest.mark.parametrize("pooling", ["mean", "cls", "attention", "line_attention"])
def test_oracle_state_is_invariant_to_padded_batch_companion(pooling: OraclePooling) -> None:
    seed_everything(31)
    encoder = PoolingOracleEncoder(
        d_model=64, n_layers=2, n_heads=2, pooling=pooling, dropout=0.0
    ).eval()
    short = b"a -> up"
    long = b"a -> up\nb -> down\nc -> left"
    alone_tokens = torch.tensor([list(short)], dtype=torch.long)
    alone_mask = torch.ones_like(alone_tokens, dtype=torch.bool)
    width = len(long)
    batch_tokens = torch.full((2, width), PAD_ID, dtype=torch.long)
    batch_mask = torch.zeros((2, width), dtype=torch.bool)
    batch_tokens[0, : len(short)] = torch.tensor(list(short))
    batch_tokens[1] = torch.tensor(list(long))
    batch_mask[0, : len(short)] = True
    batch_mask[1] = True
    with torch.inference_mode():
        alone = encoder(alone_tokens, alone_mask)[0]
        batched = encoder(batch_tokens, batch_mask)[0]
    assert torch.allclose(alone, batched, atol=1e-6)


def test_different_init_seed_changes_initial_hash() -> None:
    seed_everything(101)
    first = TinySharedDecoder(
        TinyDecoderConfig(d_model=64, n_layers=2, deep_film=True)
    ).parameter_hash()
    seed_everything(102)
    second = TinySharedDecoder(
        TinyDecoderConfig(d_model=64, n_layers=2, deep_film=True)
    ).parameter_hash()
    assert first != second


def test_v2_writes_reloadable_hash_bound_artifacts(tmp_path) -> None:
    examples = [_example("query-a", "up", "r1")]
    config = PhaseAV2Config(
        root_seed=41,
        d_model=64,
        n_layers=2,
        deep_film=True,
        oracle_mode="hash",
        steps=1,
        lr=1e-3,
        batch_size=1,
        dropout=0.0,
        max_train_eval_examples=1,
    )
    artifact = train_phase_a_v2(
        config, train_examples=examples, val_examples=examples, output_dir=tmp_path
    )
    assert artifact["reload_verification"]["output_bit_equal"] is True
    assert artifact["reload_verification"]["hash"] == artifact["weight_hash"]
    assert artifact["files"]["decoder.pt"]["bytes"] > 0
    assert len(artifact["files"]["decoder.pt"]["sha256"]) == 64
    assert (tmp_path / "artifact.json").is_file()
    assert (tmp_path / "decoder.pt").is_file()


def test_catalog_counts_and_rule_firewall_are_recorded() -> None:
    cases = build_overfit_cases(root_seed=123)
    train, validation, _ = cases["unseen_rule_text"]
    assert len(train) == 448
    assert len(validation) == 224
    assert all(example.kind != "specificity" for example in train + validation)
    assert not any(
        example.family == "contextual_remap" and example.kind == "composition"
        for example in train + validation
    )
    assert {example.rule_fp for example in train}.isdisjoint(
        {example.rule_fp for example in validation}
    )


def test_common_backbone_is_paired_across_conditioners() -> None:
    models = {}
    variants: list[tuple[Conditioning, bool]] = [
        ("none", False),
        ("additive", False),
        ("additive", True),
        ("film", False),
        ("film", True),
        ("prefix", False),
    ]
    for conditioning, deep in variants:
        seed_everything(211)
        models[(conditioning, deep)] = TinySharedDecoder(
            TinyDecoderConfig(
                d_model=64,
                n_layers=2,
                conditioning=conditioning,
                deep_film=deep,
                dropout=0.0,
                version="r24-tiny-decoder/v2",
            )
        ).state_dict()
    reference = models[("none", False)]
    shared_keys = {
        key
        for key in reference
        if not any(part in key for part in ("films", "additive_projs", "prefix_proj"))
    }
    for state in models.values():
        for key in shared_keys:
            assert torch.equal(reference[key], state[key]), key


def test_v2_corpus_has_one_target_per_actual_model_input() -> None:
    train, validation, audit = build_phase_a_v2_corpus(
        root_seed=123, train_per_family=20, val_per_family=10
    )
    assert audit.excluded == {
        "contextual_composition_undefined_mapping": 30,
        "specificity_rendered_input_conflict": 90,
    }
    for examples in (train, validation):
        assert input_conflicts(examples, oracle_mode="text") == {}
        assert input_conflicts(examples, oracle_mode="hash") == {}


def test_screen_suite_has_fixed_validation_and_nested_data_prefixes() -> None:
    _, base_train, base_validation, base_metadata = materialize_case("base")
    _, small_train, small_validation, small_metadata = materialize_case("data_n5")
    _, large_train, large_validation, large_metadata = materialize_case("data_n40")
    assert len(CASE_OVERRIDES) == 22
    assert len(suite_sha256()) == 64
    assert (
        base_metadata["fixed_validation_sha256"]
        == small_metadata["fixed_validation_sha256"]
        == large_metadata["fixed_validation_sha256"]
    )
    assert (
        {example.rule_fp for example in small_train}
        < {example.rule_fp for example in base_train}
        < {example.rule_fp for example in large_train}
    )
    assert (
        [(example.rule_fp, example.query_text, example.answer_text) for example in base_validation]
        == [
            (example.rule_fp, example.query_text, example.answer_text)
            for example in small_validation
        ]
        == [
            (example.rule_fp, example.query_text, example.answer_text)
            for example in large_validation
        ]
    )



def test_v2_artifact_records_clustered_per_rule_controls() -> None:
    examples = [
        _example("query-a", "up", "r1"),
        _example("query-b", "down", "r1"),
        _example("query-c", "left", "r2"),
    ]
    config = PhaseAV2Config(
        root_seed=53,
        oracle_mode="hash",
        steps=1,
        batch_size=1,
        dropout=0.0,
        max_train_eval_examples=3,
    )
    artifact = train_phase_a_v2(config, train_examples=examples, val_examples=examples)
    per_rule = artifact["validation"]["per_rule_controls"]
    assert set(per_rule) == {"r1", "r2"}
    assert per_rule["r1"]["total"] == 2
    assert per_rule["r2"]["total"] == 1
    for row in per_rule.values():
        assert set(row["controls"]) == {"oracle", "zero", "random", "swapped"}
        assert sum(row["paired_swapped"].values()) == row["total"]


def test_factorial_suite_completes_only_the_four_missing_cells() -> None:
    assert set(FACTORIAL_CASE_OVERRIDES) == {
        "factor_ac",
        "factor_al",
        "factor_cl",
        "factor_acl",
    }
    assert len(factorial_suite_sha256()) == 64
    _, ac_train, ac_validation, ac_metadata = materialize_factorial_case("factor_ac")
    _, acl_train, acl_validation, acl_metadata = materialize_factorial_case("factor_acl")
    assert ac_metadata["fixed_validation_sha256"] == acl_metadata["fixed_validation_sha256"]
    assert ac_metadata["materialized_train_sha256"] == acl_metadata["materialized_train_sha256"]
    assert ac_train == acl_train
    assert ac_validation == acl_validation


def _selection_artifact(
    oracle_correct: int,
    swapped_correct: int,
    oracle_only: int,
    control_only: int,
    macro_rows: list[tuple[int, int, int]],
) -> dict[str, object]:
    return {
        "validation": {
            "controls": {
                "oracle": {"correct": oracle_correct, "total": 219},
                "swapped": {"correct": swapped_correct, "total": 219},
            },
            "paired_controls": {
                "swapped": {
                    "oracle_only": oracle_only,
                    "control_only": control_only,
                    "both": oracle_correct - oracle_only,
                    "neither": 219 - oracle_correct - control_only,
                }
            },
            "per_rule_controls": {
                f"r{index}": {
                    "controls": {
                        "oracle": {"accuracy": oracle / total},
                        "swapped": {"accuracy": swapped / total},
                    }
                }
                for index, (oracle, swapped, total) in enumerate(macro_rows)
            },
        }
    }


def test_factorial_selection_is_causal_first_then_exact() -> None:
    positive_rows = [(1, 0, 1)] + [(0, 0, 1)] * 29
    records = [
        selection_record("base", _selection_artifact(24, 17, 9, 2, positive_rows)),
        selection_record(
            "conditioning_additive_deep",
            _selection_artifact(28, 15, 18, 5, positive_rows),
        ),
        selection_record("factor_acl", _selection_artifact(30, 29, 3, 2, positive_rows)),
    ]
    assert records[-1]["eligible"] is False
    assert select_winner(records) == "conditioning_additive_deep"



def test_confirmation_is_fresh_and_seed_paired_across_two_frozen_arms() -> None:
    validation_hashes = set()
    for seed_index, expected_seeds in enumerate(SEED_TUPLES):
        base_config, base_train, base_validation, base_metadata = materialize_confirmation_case(
            "base", seed_index
        )
        finalist_config, finalist_train, finalist_validation, finalist_metadata = (
            materialize_confirmation_case("finalist", seed_index)
        )
        assert base_train == finalist_train
        assert base_validation == finalist_validation
        assert base_config.resolved_seeds() == finalist_config.resolved_seeds()
        assert base_config.resolved_seeds() == {
            "catalog": 25001,
            "init": expected_seeds["init_seed"],
            "batch": expected_seeds["batch_seed"],
            "dropout": expected_seeds["dropout_seed"],
            "control": expected_seeds["control_seed"],
        }
        assert base_config.conditioning == "film"
        assert finalist_config.conditioning == "additive"
        assert base_metadata["fixed_validation_sha256"] == finalist_metadata[
            "fixed_validation_sha256"
        ]
        validation_hashes.add(base_metadata["fixed_validation_sha256"])
    assert len(validation_hashes) == 1
    assert validation_hashes != {
        "6626ad31c8577c4b827c47a440df1e5d8538d24433a4f7dba26d58b806fdaabd"
    }
    assert len(confirmation_suite_sha256()) == 64


def _confirmation_artifact(
    arm: str, seed_index: int, oracle_correct: int, swapped_correct: int
) -> dict[str, object]:
    oracle_accuracy = oracle_correct / 100
    swapped_accuracy = swapped_correct / 100
    return {
        "confirmation": {"arm": arm, "seed_index": seed_index},
        "validation": {
            "controls": {
                "oracle": {
                    "correct": oracle_correct,
                    "total": 100,
                    "exact_accuracy": oracle_accuracy,
                },
                "swapped": {
                    "correct": swapped_correct,
                    "total": 100,
                    "exact_accuracy": swapped_accuracy,
                },
            },
            "per_rule_controls": {
                "rule": {
                    "controls": {
                        "oracle": {"accuracy": oracle_accuracy},
                        "swapped": {"accuracy": swapped_accuracy},
                    }
                }
            },
        },
    }


def test_confirmation_gate_requires_all_five_positive_paired_seeds() -> None:
    artifacts = []
    for seed_index in range(5):
        artifacts.append(_confirmation_artifact("base", seed_index, 20, 17))
        artifacts.append(_confirmation_artifact("finalist", seed_index, 21, 16))
    decision = evaluate_gate(artifacts)
    assert decision["passed"] is True
    assert decision["positive_finalist_delta_seeds"] == 5

    artifacts[-1] = _confirmation_artifact("finalist", 4, 21, 21)
    rejected = evaluate_gate(artifacts)
    assert rejected["passed"] is False
    assert rejected["positive_finalist_delta_seeds"] == 4
