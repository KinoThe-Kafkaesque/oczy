"""Pre-registered R24 Phase-A v2 screening suite.

Every screening case changes one named variable relative to ``base``.  All use
the same maximum generated catalog and nested training-rule prefixes, so data
scale never changes the validation rows.  The viewed seed-123 ladder remains
diagnostic only; this suite uses a fresh tuning catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .corpus_v2 import build_phase_a_v2_corpus, hash_examples
from .phase_a_v2 import PhaseAV2Config, train_phase_a_v2
from .pretrain import PretrainExample

TUNE_CATALOG_SEED = 24001
MAX_TRAIN_TASKS_PER_FAMILY = 100
VALIDATION_TASKS_PER_FAMILY = 10
BASE_INIT_SEED = 0
BASE_BATCH_SEED = 101
BASE_DROPOUT_SEED = 202
BASE_CONTROL_SEED = 303

BASE_CONFIG = PhaseAV2Config(
    root_seed=TUNE_CATALOG_SEED,
    catalog_seed=TUNE_CATALOG_SEED,
    init_seed=BASE_INIT_SEED,
    batch_seed=BASE_BATCH_SEED,
    dropout_seed=BASE_DROPOUT_SEED,
    control_seed=BASE_CONTROL_SEED,
    train_per_family=20,
    val_per_family=VALIDATION_TASKS_PER_FAMILY,
    d_model=64,
    n_layers=2,
    conditioning="film",
    deep_film=True,
    encoder_pooling="mean",
    oracle_mode="text",
    steps=800,
    lr=1e-3,
    encoder_lr_multiplier=1.0,
    weight_decay=0.01,
    batch_size=32,
    scheduler="constant",
    warmup_steps=0,
    counterfactual_weight=0.0,
    dropout=0.1,
    max_train_eval_examples=256,
)

# Overrides are deliberately one-variable relative to BASE_CONFIG.
CASE_OVERRIDES: dict[str, dict[str, Any]] = {
    "base": {},
    # Rule encoder pooling.
    "encoder_cls": {"encoder_pooling": "cls"},
    "encoder_attention": {"encoder_pooling": "attention"},
    "encoder_line_attention": {"encoder_pooling": "line_attention"},
    # Nested data scale; materialized separately below.
    "data_n1": {"train_per_family": 1},
    "data_n5": {"train_per_family": 5},
    "data_n40": {"train_per_family": 40},
    # Conditioning geometry, with paired shared backbone initialization.
    "conditioning_none": {"conditioning": "none", "deep_film": False},
    "conditioning_additive_shallow": {"conditioning": "additive", "deep_film": False},
    "conditioning_additive_deep": {"conditioning": "additive", "deep_film": True},
    "conditioning_film_shallow": {"conditioning": "film", "deep_film": False},
    "conditioning_prefix4": {"conditioning": "prefix", "deep_film": False},
    # Capacity control (not an expected primary improvement).
    "capacity_d128_l4": {"d_model": 128, "n_layers": 4},
    # Optimization; each case changes one registered field.
    "optimizer_lr3e4": {"lr": 3e-4},
    "optimizer_lr3e3": {"lr": 3e-3},
    "optimizer_wd0": {"weight_decay": 0.0},
    "optimizer_cosine": {"scheduler": "cosine"},
    "optimizer_warmup40": {"warmup_steps": 40},
    "optimizer_encoder_lr03": {"encoder_lr_multiplier": 0.3},
    "optimizer_encoder_lr3": {"encoder_lr_multiplier": 3.0},
    "optimizer_counterfactual01": {"counterfactual_weight": 0.1},
    "optimizer_steps1500": {"steps": 1500},
}


def suite_definition() -> dict[str, Any]:
    return {
        "schema_version": "oczy/r24-phase-a-screen/v2",
        "tune_catalog_seed": TUNE_CATALOG_SEED,
        "max_train_tasks_per_family": MAX_TRAIN_TASKS_PER_FAMILY,
        "validation_tasks_per_family": VALIDATION_TASKS_PER_FAMILY,
        "base_config": asdict(BASE_CONFIG),
        "case_overrides": CASE_OVERRIDES,
        "data_scale_note": (
            "The frozen generator exposes only 45 unique rule-transformation fingerprints "
            "at max scale, so the balanced nested curve stops at 40 rules/family."
        ),
        "selection_note": (
            "Exploratory tuning only. Compare exact correct/total and correct-vs-swapped "
            "paired controls; do not select on zero-state delta. Confirmation requires a "
            "new frozen catalog and five paired init seeds."
        ),
    }


def suite_sha256() -> str:
    return hashlib.sha256(
        json.dumps(suite_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _nested_train_prefix(
    examples: list[PretrainExample], tasks_per_family: int
) -> list[PretrainExample]:
    rules_by_family: dict[str, set[str]] = {}
    for example in examples:
        rules_by_family.setdefault(example.family, set()).add(example.rule_fp)
    allowed: set[str] = set()
    for family in sorted(rules_by_family):
        ordered = sorted(rules_by_family[family])
        if len(ordered) < tasks_per_family:
            raise ValueError(f"family {family} has {len(ordered)} rules, needs {tasks_per_family}")
        allowed.update(ordered[:tasks_per_family])
    return [example for example in examples if example.rule_fp in allowed]


def materialize_case(
    case_id: str,
) -> tuple[PhaseAV2Config, list[PretrainExample], list[PretrainExample], dict[str, Any]]:
    if case_id not in CASE_OVERRIDES:
        raise KeyError(f"unknown suite case {case_id!r}")
    max_train, validation, max_audit = build_phase_a_v2_corpus(
        root_seed=TUNE_CATALOG_SEED,
        train_per_family=MAX_TRAIN_TASKS_PER_FAMILY,
        val_per_family=VALIDATION_TASKS_PER_FAMILY,
    )
    overrides = CASE_OVERRIDES[case_id]
    config = replace(BASE_CONFIG, **overrides)
    train = _nested_train_prefix(max_train, config.train_per_family)
    metadata = {
        "schema_version": "oczy/r24-phase-a-screen-case/v2",
        "case_id": case_id,
        "suite_sha256": suite_sha256(),
        "reference_case": "base",
        "overrides": overrides,
        "max_catalog_audit": asdict(max_audit),
        "materialized_train_sha256": hash_examples(train),
        "fixed_validation_sha256": hash_examples(validation),
        "materialized_train_examples": len(train),
        "fixed_validation_examples": len(validation),
        "materialized_train_rules": len({example.rule_fp for example in train}),
        "fixed_validation_rules": len({example.rule_fp for example in validation}),
    }
    return config, train, validation, metadata


def run_case(case_id: str, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, train, validation, metadata = materialize_case(case_id)
    print(
        f"R24 v2 screen {case_id}: {len(train)} train / {len(validation)} validation "
        f"suite={metadata['suite_sha256'][:12]}",
        flush=True,
    )
    artifact = train_phase_a_v2(
        config,
        train_examples=train,
        val_examples=validation,
        output_dir=output,
    )
    artifact["suite"] = metadata
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    (output / "case_manifest.json").write_text(
        json.dumps({"suite": metadata, "config": asdict(config)}, indent=2, sort_keys=True)
    )
    return artifact


def _emit(artifact: dict[str, Any]) -> None:
    for name in (
        "oracle_dev_accuracy",
        "query_only_dev_accuracy",
        "swapped_dev_accuracy",
        "swapped_delta",
        "random_delta",
        "zero_state_delta",
    ):
        print(f"METRIC {name}={float(artifact[name]):.6f}", flush=True)
    validation = artifact["validation"]
    assert isinstance(validation, dict)
    print(
        "METRIC teacher_forced_token_accuracy="
        f"{float(validation['teacher_forced_token_accuracy']):.6f}",
        flush=True,
    )
    print(
        f"METRIC weight_hash_prefix={int(str(artifact['weight_hash'])[:8], 16) % 1_000_000}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--case", required=True, choices=sorted(CASE_OVERRIDES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-definition", action="store_true")
    args = parser.parse_args(argv)
    if args.print_definition:
        print(json.dumps(suite_definition(), indent=2, sort_keys=True))
        return 0
    artifact = run_case(args.case, args.output)
    _emit(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
