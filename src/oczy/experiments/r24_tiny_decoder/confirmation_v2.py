"""Fresh-catalog five-seed confirmation for the frozen R24 Phase-A finalist.

This is not another search.  The closed tuning factorial selected deep additive
conditioning by its registered exact-first causal rule.  Confirmation compares
only that frozen finalist against the original base at five paired RNG tuples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .corpus_v2 import build_phase_a_v2_corpus, hash_examples
from .factorial_v2 import suite_sha256 as factorial_suite_sha256
from .phase_a_v2 import PhaseAV2Config, train_phase_a_v2
from .pretrain import PretrainExample
from .suite_v2 import (
    BASE_CONFIG,
    MAX_TRAIN_TASKS_PER_FAMILY,
    VALIDATION_TASKS_PER_FAMILY,
    _nested_train_prefix,
)

CONFIRM_CATALOG_SEED = 25001
FINALIST_CASE = "conditioning_additive_deep"
ARMS: dict[str, dict[str, Any]] = {
    "base": {},
    "finalist": {"conditioning": "additive", "deep_film": True},
}
SEED_TUPLES: tuple[dict[str, int], ...] = (
    {"init_seed": 11, "batch_seed": 1011, "dropout_seed": 2011, "control_seed": 3011},
    {"init_seed": 29, "batch_seed": 1029, "dropout_seed": 2029, "control_seed": 3029},
    {"init_seed": 47, "batch_seed": 1047, "dropout_seed": 2047, "control_seed": 3047},
    {"init_seed": 71, "batch_seed": 1071, "dropout_seed": 2071, "control_seed": 3071},
    {"init_seed": 97, "batch_seed": 1097, "dropout_seed": 2097, "control_seed": 3097},
)
CONFIRMATION_GATE: dict[str, Any] = {
    "finalist_positive_swapped_delta_seeds": 5,
    "finalist_mean_swapped_delta_min": 0.02,
    "finalist_mean_oracle_minus_base_min": 0.0,
    "finalist_mean_swapped_delta_minus_base_min": 0.0,
    "unit": "paired seed; equal-rule macro and family results are mandatory secondary summaries",
    "note": "Zero/random controls are diagnostic and never enter promotion.",
}


def suite_definition() -> dict[str, Any]:
    base = replace(
        BASE_CONFIG,
        root_seed=CONFIRM_CATALOG_SEED,
        catalog_seed=CONFIRM_CATALOG_SEED,
    )
    return {
        "schema_version": "oczy/r24-phase-a-confirmation/v2",
        "upstream_factorial_suite_sha256": factorial_suite_sha256(),
        "fresh_catalog_seed": CONFIRM_CATALOG_SEED,
        "base_config_before_paired_seeds": asdict(base),
        "selected_case": FINALIST_CASE,
        "arms": ARMS,
        "seed_tuples": list(SEED_TUPLES),
        "confirmation_gate": CONFIRMATION_GATE,
    }


def suite_sha256() -> str:
    return hashlib.sha256(
        json.dumps(suite_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def materialize_case(
    arm: str, seed_index: int
) -> tuple[PhaseAV2Config, list[PretrainExample], list[PretrainExample], dict[str, Any]]:
    if arm not in ARMS:
        raise KeyError(f"unknown confirmation arm {arm!r}")
    if seed_index < 0 or seed_index >= len(SEED_TUPLES):
        raise IndexError(f"seed_index must be in [0,{len(SEED_TUPLES)})")
    max_train, validation, max_audit = build_phase_a_v2_corpus(
        root_seed=CONFIRM_CATALOG_SEED,
        train_per_family=MAX_TRAIN_TASKS_PER_FAMILY,
        val_per_family=VALIDATION_TASKS_PER_FAMILY,
    )
    paired_seeds = SEED_TUPLES[seed_index]
    config = replace(
        BASE_CONFIG,
        root_seed=CONFIRM_CATALOG_SEED,
        catalog_seed=CONFIRM_CATALOG_SEED,
        **paired_seeds,
        **ARMS[arm],
    )
    train = _nested_train_prefix(max_train, config.train_per_family)
    metadata = {
        "schema_version": "oczy/r24-phase-a-confirmation-case/v2",
        "arm": arm,
        "seed_index": seed_index,
        "suite_sha256": suite_sha256(),
        "selected_case": FINALIST_CASE,
        "arm_overrides": ARMS[arm],
        "paired_seeds": paired_seeds,
        "max_catalog_audit": asdict(max_audit),
        "materialized_train_sha256": hash_examples(train),
        "fixed_validation_sha256": hash_examples(validation),
        "materialized_train_examples": len(train),
        "fixed_validation_examples": len(validation),
        "materialized_train_rules": len({example.rule_fp for example in train}),
        "fixed_validation_rules": len({example.rule_fp for example in validation}),
    }
    return config, train, validation, metadata


def run_case(arm: str, seed_index: int, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, train, validation, metadata = materialize_case(arm, seed_index)
    print(
        f"R24 v2 confirmation {arm} seed={seed_index}: "
        f"{len(train)} train / {len(validation)} validation "
        f"suite={metadata['suite_sha256'][:12]}",
        flush=True,
    )
    artifact = train_phase_a_v2(
        config, train_examples=train, val_examples=validation, output_dir=output
    )
    artifact["confirmation"] = metadata
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    (output / "case_manifest.json").write_text(
        json.dumps({"confirmation": metadata, "config": asdict(config)}, indent=2, sort_keys=True)
    )
    return artifact


def _record(artifact: Mapping[str, Any]) -> dict[str, Any]:
    metadata = artifact["confirmation"]
    controls = artifact["validation"]["controls"]
    oracle = controls["oracle"]
    swapped = controls["swapped"]
    per_rule = artifact["validation"]["per_rule_controls"]
    macro_delta = sum(
        float(row["controls"]["oracle"]["accuracy"])
        - float(row["controls"]["swapped"]["accuracy"])
        for row in per_rule.values()
    ) / len(per_rule)
    return {
        "arm": str(metadata["arm"]),
        "seed_index": int(metadata["seed_index"]),
        "oracle_correct": int(oracle["correct"]),
        "swapped_correct": int(swapped["correct"]),
        "total": int(oracle["total"]),
        "swapped_delta": float(oracle["exact_accuracy"])
        - float(swapped["exact_accuracy"]),
        "equal_rule_macro_delta": macro_delta,
    }


def evaluate_gate(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [_record(artifact) for artifact in artifacts]
    indexed = {(record["arm"], record["seed_index"]): record for record in records}
    expected = {(arm, index) for arm in ARMS for index in range(len(SEED_TUPLES))}
    if set(indexed) != expected:
        raise ValueError("confirmation requires exactly both arms at all five seed indexes")
    pairs = []
    for index in range(len(SEED_TUPLES)):
        base = indexed[("base", index)]
        finalist = indexed[("finalist", index)]
        pairs.append(
            {
                "seed_index": index,
                "base": base,
                "finalist": finalist,
                "oracle_difference": (
                    finalist["oracle_correct"] - base["oracle_correct"]
                ) / finalist["total"],
                "swapped_delta_difference": (
                    finalist["swapped_delta"] - base["swapped_delta"]
                ),
            }
        )
    mean_finalist_delta = sum(
        float(pair["finalist"]["swapped_delta"]) for pair in pairs
    ) / len(pairs)
    mean_oracle_difference = sum(float(pair["oracle_difference"]) for pair in pairs) / len(
        pairs
    )
    mean_delta_difference = sum(
        float(pair["swapped_delta_difference"]) for pair in pairs
    ) / len(pairs)
    positive_seeds = sum(float(pair["finalist"]["swapped_delta"]) > 0 for pair in pairs)
    passed = bool(
        positive_seeds == 5
        and mean_finalist_delta >= 0.02
        and mean_oracle_difference >= 0.0
        and mean_delta_difference >= 0.0
    )
    return {
        "schema_version": "oczy/r24-phase-a-confirmation-decision/v2",
        "suite_sha256": suite_sha256(),
        "pairs": pairs,
        "positive_finalist_delta_seeds": positive_seeds,
        "mean_finalist_swapped_delta": mean_finalist_delta,
        "mean_finalist_oracle_minus_base": mean_oracle_difference,
        "mean_finalist_delta_minus_base": mean_delta_difference,
        "passed": passed,
        "decision": "promote_finalist_to_phase_c_organ" if passed else "do_not_promote",
    }


def _emit(artifact: Mapping[str, Any]) -> None:
    for name in (
        "oracle_dev_accuracy",
        "query_only_dev_accuracy",
        "swapped_dev_accuracy",
        "swapped_delta",
        "random_delta",
        "zero_state_delta",
    ):
        print(f"METRIC {name}={float(artifact[name]):.6f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed-index", required=True, type=int, choices=range(len(SEED_TUPLES)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-definition", action="store_true")
    args = parser.parse_args(argv)
    if args.print_definition:
        print(json.dumps(suite_definition(), indent=2, sort_keys=True))
        return 0
    _emit(run_case(args.arm, args.seed_index, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
