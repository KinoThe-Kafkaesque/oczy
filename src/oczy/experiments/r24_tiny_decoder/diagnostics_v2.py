"""Discriminating learnability ladder for R24 Phase-A v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_v2 import build_phase_a_v2_corpus
from .phase_a_v2 import PhaseAV2Config, train_phase_a_v2
from .pretrain import PretrainExample


def _examples_by_rule(examples: list[PretrainExample]) -> dict[str, list[PretrainExample]]:
    grouped: dict[str, list[PretrainExample]] = {}
    for example in examples:
        grouped.setdefault(example.rule_fp, []).append(example)
    return grouped


def build_overfit_cases(
    *, root_seed: int = 123, train_per_family: int = 20, val_per_family: int = 10
) -> dict[str, tuple[list[PretrainExample], list[PretrainExample], str]]:
    train, val, _ = build_phase_a_v2_corpus(
        root_seed=root_seed,
        train_per_family=train_per_family,
        val_per_family=val_per_family,
    )
    by_rule = _examples_by_rule(train)
    ordered_rules = [by_rule[key] for key in sorted(by_rule)]
    first = ordered_rules[0]
    by_query: dict[str, list[PretrainExample]] = {}
    for example in train:
        by_query.setdefault(example.query_text, []).append(example)
    conflicting: list[PretrainExample] | None = None
    for query in sorted(by_query):
        candidates = by_query[query]
        for left in candidates:
            for right in candidates:
                if left.rule_fp != right.rule_fp and left.answer_text != right.answer_text:
                    conflicting = [left, right]
                    break
            if conflicting is not None:
                break
        if conflicting is not None:
            break
    if conflicting is None:
        raise RuntimeError("catalog lacks a same-query/different-answer pair")

    held_train: list[PretrainExample] = []
    held_val: list[PretrainExample] = []
    for rule_examples in ordered_rules[:12]:
        by_kind: dict[str, list[PretrainExample]] = {}
        for example in rule_examples:
            by_kind.setdefault(example.kind, []).append(example)
        for kind in sorted(by_kind):
            examples = sorted(
                by_kind[kind], key=lambda example: (example.query_text, example.answer_text)
            )
            if len(examples) >= 2:
                held_train.extend(examples[:-1])
                held_val.append(examples[-1])
            else:
                held_train.extend(examples)
    train_inputs = {(example.rule_fp, example.query_text) for example in held_train}
    held_val = [
        example for example in held_val if (example.rule_fp, example.query_text) not in train_inputs
    ]
    if not held_val:
        raise RuntimeError("kind-stratified held-query split is empty")

    return {
        "one_example_text": ([first[0]], [first[0]], "text"),
        "one_rule_learned": (first, first, "learned"),
        "conflicting_query_learned": (conflicting, conflicting, "learned"),
        "held_query_learned": (held_train, held_val, "learned"),
        "unseen_rule_text": (train, val, "text"),
    }


def run_overfit_ladder(
    *,
    output_dir: str | Path,
    root_seed: int = 123,
    quick: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Locate failure: sequence learning → shared code → unseen rule parsing."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = build_overfit_cases(root_seed=root_seed)
    steps = {
        "one_example_text": 80 if quick else 250,
        "one_rule_learned": 120 if quick else 400,
        "conflicting_query_learned": 200 if quick else 800,
        "held_query_learned": 200 if quick else 800,
        "unseen_rule_text": 200 if quick else 800,
    }
    artifacts: dict[str, Any] = {}
    for name, (train, val, oracle_mode) in cases.items():
        config = PhaseAV2Config(
            root_seed=root_seed,
            d_model=64,
            n_layers=2,
            conditioning="film",
            deep_film=True,
            encoder_pooling="mean",
            oracle_mode=oracle_mode,  # type: ignore[arg-type]
            steps=steps[name],
            lr=1e-3,
            batch_size=min(32, max(1, len(train))),
            scheduler="cosine",
            warmup_steps=min(20, max(0, steps[name] // 10)),
            weight_decay=0.0,
            counterfactual_weight=0.1 if name == "conflicting_query_learned" else 0.0,
            dropout=0.0
            if name in {"one_example_text", "one_rule_learned", "conflicting_query_learned"}
            else 0.1,
            device=device,
            max_train_eval_examples=256,
        )
        print(f"\n=== {name}: {len(train)} train / {len(val)} val ===", flush=True)
        artifact = train_phase_a_v2(
            config,
            train_examples=train,
            val_examples=val,
            output_dir=output / name,
        )
        artifacts[name] = artifact
    summary = {
        "schema_version": "oczy/r24-overfit-ladder/v2",
        "root_seed": root_seed,
        "quick": quick,
        "cases": {
            name: {
                "config": artifact["config"],
                "oracle_dev_accuracy": artifact["oracle_dev_accuracy"],
                "query_only_dev_accuracy": artifact["query_only_dev_accuracy"],
                "zero_state_delta": artifact["zero_state_delta"],
                "swapped_delta": artifact["swapped_delta"],
                "train_oracle_accuracy": artifact["train"]["controls"]["oracle"]["exact_accuracy"],
                "validation_teacher_forced_token_accuracy": artifact["validation"][
                    "teacher_forced_token_accuracy"
                ],
            }
            for name, artifact in artifacts.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
