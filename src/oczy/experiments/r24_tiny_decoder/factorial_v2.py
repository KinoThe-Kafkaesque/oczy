"""Closed 2^3 R24 Phase-A v2 factorial on the frozen tuning catalog.

The one-factor screen identified three causally eligible ingredients.  This
module completes their four missing interaction cells without changing the
catalog, validation rows, budget, or any other base field.  It remains tuning
only; the selected arm must be confirmed on a fresh catalog and five paired
seed tuples.
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
from .phase_a_v2 import PhaseAV2Config, train_phase_a_v2
from .pretrain import PretrainExample
from .suite_v2 import (
    BASE_CONFIG,
    MAX_TRAIN_TASKS_PER_FAMILY,
    VALIDATION_TASKS_PER_FAMILY,
    _nested_train_prefix,
)
from .suite_v2 import (
    suite_sha256 as screen_suite_sha256,
)

CASE_OVERRIDES: dict[str, dict[str, Any]] = {
    "factor_ac": {
        "conditioning": "additive",
        "deep_film": True,
        "encoder_pooling": "cls",
    },
    "factor_al": {
        "conditioning": "additive",
        "deep_film": True,
        "encoder_lr_multiplier": 0.3,
    },
    "factor_cl": {
        "encoder_pooling": "cls",
        "encoder_lr_multiplier": 0.3,
    },
    "factor_acl": {
        "conditioning": "additive",
        "deep_film": True,
        "encoder_pooling": "cls",
        "encoder_lr_multiplier": 0.3,
    },
}

SELECTION_RULE: dict[str, Any] = {
    "eligible_when": {
        "oracle_correct_min": 24,
        "oracle_minus_swapped_correct_min": 5,
        "paired_oracle_only_gt_control_only": True,
        "equal_rule_macro_oracle_minus_swapped_min": 0.02,
    },
    "rank": [
        "oracle_correct_desc",
        "paired_net_desc",
        "paired_control_only_asc",
        "changed_factor_count_asc",
        "case_id_lexicographic",
    ],
    "candidate_cells": [
        "base",
        "conditioning_additive_deep",
        "encoder_cls",
        "optimizer_encoder_lr03",
        "factor_ac",
        "factor_al",
        "factor_cl",
        "factor_acl",
    ],
    "note": "Selection uses only oracle and in-distribution swapped-state results; zero/random are diagnostic.",
}


def suite_definition() -> dict[str, Any]:
    return {
        "schema_version": "oczy/r24-phase-a-factorial/v2",
        "upstream_screen_suite_sha256": screen_suite_sha256(),
        "base_config": asdict(BASE_CONFIG),
        "case_overrides": CASE_OVERRIDES,
        "selection_rule": SELECTION_RULE,
    }


def suite_sha256() -> str:
    return hashlib.sha256(
        json.dumps(suite_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def materialize_case(
    case_id: str,
) -> tuple[PhaseAV2Config, list[PretrainExample], list[PretrainExample], dict[str, Any]]:
    if case_id not in CASE_OVERRIDES:
        raise KeyError(f"unknown factorial case {case_id!r}")
    catalog_seed = BASE_CONFIG.catalog_seed
    if catalog_seed is None:
        raise ValueError("factorial base config requires an explicit catalog seed")
    max_train, validation, max_audit = build_phase_a_v2_corpus(
        root_seed=catalog_seed,
        train_per_family=MAX_TRAIN_TASKS_PER_FAMILY,
        val_per_family=VALIDATION_TASKS_PER_FAMILY,
    )
    overrides = CASE_OVERRIDES[case_id]
    config = replace(BASE_CONFIG, **overrides)
    train = _nested_train_prefix(max_train, config.train_per_family)
    metadata = {
        "schema_version": "oczy/r24-phase-a-factorial-case/v2",
        "case_id": case_id,
        "suite_sha256": suite_sha256(),
        "upstream_screen_suite_sha256": screen_suite_sha256(),
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


def _rule_macro_delta(artifact: Mapping[str, Any]) -> float:
    per_rule = artifact["validation"]["per_rule_controls"]
    deltas = [
        float(row["controls"]["oracle"]["accuracy"])
        - float(row["controls"]["swapped"]["accuracy"])
        for row in per_rule.values()
    ]
    return sum(deltas) / len(deltas)


def selection_record(case_id: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    oracle = artifact["validation"]["controls"]["oracle"]
    swapped = artifact["validation"]["controls"]["swapped"]
    paired = artifact["validation"]["paired_controls"]["swapped"]
    changed_factors = {
        "base": 0,
        "conditioning_additive_deep": 1,
        "encoder_cls": 1,
        "optimizer_encoder_lr03": 1,
        "factor_ac": 2,
        "factor_al": 2,
        "factor_cl": 2,
        "factor_acl": 3,
    }[case_id]
    record = {
        "case_id": case_id,
        "oracle_correct": int(oracle["correct"]),
        "swapped_correct": int(swapped["correct"]),
        "total": int(oracle["total"]),
        "paired_oracle_only": int(paired["oracle_only"]),
        "paired_control_only": int(paired["control_only"]),
        "equal_rule_macro_delta": _rule_macro_delta(artifact),
        "changed_factor_count": changed_factors,
    }
    record["paired_net"] = record["paired_oracle_only"] - record["paired_control_only"]
    record["eligible"] = bool(
        record["oracle_correct"] >= 24
        and record["oracle_correct"] - record["swapped_correct"] >= 5
        and record["paired_oracle_only"] > record["paired_control_only"]
        and record["equal_rule_macro_delta"] >= 0.02
    )
    return record


def select_winner(records: Sequence[Mapping[str, Any]]) -> str:
    eligible = [record for record in records if bool(record["eligible"])]
    if not eligible:
        raise ValueError("no causally eligible factorial candidate")
    ordered = sorted(
        eligible,
        key=lambda record: (
            -int(record["oracle_correct"]),
            -int(record["paired_net"]),
            int(record["paired_control_only"]),
            int(record["changed_factor_count"]),
            str(record["case_id"]),
        ),
    )
    return str(ordered[0]["case_id"])


def run_case(case_id: str, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, train, validation, metadata = materialize_case(case_id)
    print(
        f"R24 v2 factorial {case_id}: {len(train)} train / {len(validation)} validation "
        f"suite={metadata['suite_sha256'][:12]}",
        flush=True,
    )
    artifact = train_phase_a_v2(
        config, train_examples=train, val_examples=validation, output_dir=output
    )
    artifact["factorial"] = metadata
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    (output / "case_manifest.json").write_text(
        json.dumps({"factorial": metadata, "config": asdict(config)}, indent=2, sort_keys=True)
    )
    return artifact


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
    validation = artifact["validation"]
    print(
        "METRIC teacher_forced_token_accuracy="
        f"{float(validation['teacher_forced_token_accuracy']):.6f}",
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
    _emit(run_case(args.case, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
