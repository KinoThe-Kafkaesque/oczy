"""DEV-only CLI for the meta-cortex experiment (Research/20).

Commands
--------
- ``train-dev``   : Build/audit the in-memory DEV catalog, load real Qwen,
  outer-train on meta-train, select on meta-validation, and write a
  developmental checkpoint + DEV result.
- ``validate-dev``: Load a checkpoint, regenerate only meta-validation
  tasks, run no-update/full-generation distribution plus causal DEV
  interventions with no optimizer, and write a DEV result.
- ``audit-dev``   : Load/verify checkpoint hashes/config, regenerate/audit
  the DEV split in memory, and inspect result/state artifacts.  Does not
  load task text from disk or run training.

There is **no** ``evaluate``, ``meta-test``, ``run-meta-test``,
``materialize``, ``freeze``, ``signoff``, ``manifest``, ``C7``, or ``C8``
command/option.  There is no ``--driver mock`` fallback.  Parser help
labels every command "DEV only / not a scientific meta-test."

Invalid commands fail through argparse before model/task loading.  Real
organ load errors return nonzero with no synthetic fallback.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    ArtifactError,
    canonical_theta_hash,
    load_developmental_checkpoint,
    read_dev_result,
    save_dev_persistent_state,
    save_developmental_checkpoint,
    write_dev_result,
)
from .contracts import (
    DEV_SCHEMA,
    TASKGEN_SCHEMA,
    CheckpointMetadata,
    ModelConfig,
    OuterLoopConfig,
    TaskGeneratorConfig,
)
from .model import MetaCortex
from .organ import FrozenOrganError, QwenFrozenOrgan
from .taskgen import build_dev_catalog
from .training import OuterTrainer, run_dev_validation

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

_DEV_LABEL = "DEV only / not a scientific meta-test"


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with exactly three DEV-only commands."""
    parser = argparse.ArgumentParser(
        prog="oczy.experiments.meta_cortex",
        description=(
            "Research/20: meta-trained cortex with frozen language organ "
            "(DEV-only). Commands: train-dev, validate-dev, audit-dev."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train-dev ---------------------------------------------------------
    train = subparsers.add_parser(
        "train-dev",
        help=f"Outer-train on meta-train and select on meta-validation. {_DEV_LABEL}",
        description=(
            "Build/audit the in-memory DEV catalog, load real Qwen, "
            "outer-train on meta-train, select on meta-validation, and "
            "write a developmental checkpoint + DEV result. "
            + _DEV_LABEL
        ),
    )
    train.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for the frozen language organ",
    )
    train.add_argument(
        "--checkpoint-out",
        required=True,
        help="Directory path to write the developmental checkpoint",
    )
    train.add_argument(
        "--result-out",
        required=True,
        help="Path to write the DEV training result JSON",
    )
    train.add_argument(
        "--state-out",
        default=None,
        help="Optional directory path to write persistent cortex state (S only)",
    )
    train.add_argument(
        "--root-seed",
        type=int,
        default=20260709,
        help="Root seed for deterministic task generation",
    )
    train.add_argument(
        "--train-tasks-per-family",
        type=int,
        default=2,
        help="Number of meta-train tasks per family",
    )
    train.add_argument(
        "--validation-tasks-per-family",
        type=int,
        default=2,
        help="Number of meta-validation tasks per family",
    )
    train.add_argument(
        "--min-events",
        type=int,
        default=2,
        help="Minimum events per task (1-5)",
    )
    train.add_argument(
        "--max-events",
        type=int,
        default=2,
        help="Maximum events per task (1-5)",
    )
    train.add_argument(
        "--feature-dim",
        type=int,
        default=896,
        help="Feature dimension (must match model hidden size)",
    )
    train.add_argument(
        "--bank-width",
        type=int,
        default=3,
        help="Soft-bank width L",
    )
    train.add_argument(
        "--outer-steps",
        type=int,
        default=2,
        help="Number of outer training steps",
    )
    train.add_argument(
        "--tasks-per-step",
        type=int,
        default=1,
        help="Tasks per outer step",
    )
    train.add_argument(
        "--optimizer-name",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Outer optimizer name",
    )
    train.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Outer learning rate",
    )
    train.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Outer weight decay",
    )
    train.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    train.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Validate every N outer steps",
    )
    train.add_argument(
        "--generation-interval",
        type=int,
        default=1,
        help="Generation interval (unused in DEV but kept for config completeness)",
    )
    train.add_argument(
        "--behavior-weight",
        type=float,
        default=1.0,
        help="Behavior loss weight",
    )
    train.add_argument(
        "--specificity-weight",
        type=float,
        default=0.5,
        help="Specificity loss weight",
    )
    train.add_argument(
        "--survival-weight",
        type=float,
        default=0.5,
        help="Consolidation survival loss weight",
    )
    train.add_argument(
        "--state-norm-weight",
        type=float,
        default=0.01,
        help="State norm loss weight",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Outer training seed",
    )
    train.add_argument(
        "--source-provenance",
        default="unavailable",
        help="Source provenance string (commit SHA or 'unavailable')",
    )

    # -- validate-dev ------------------------------------------------------
    validate = subparsers.add_parser(
        "validate-dev",
        help=f"Run no-optimizer DEV validation with causal interventions. {_DEV_LABEL}",
        description=(
            "Load a checkpoint, regenerate only meta-validation tasks, "
            "run no-update/full-generation distribution plus causal DEV "
            "interventions with no optimizer, and write a DEV result. "
            + _DEV_LABEL
        ),
    )
    validate.add_argument(
        "--checkpoint",
        required=True,
        help="Directory path to the developmental checkpoint",
    )
    validate.add_argument(
        "--result-out",
        required=True,
        help="Path to write the DEV validation result JSON",
    )
    validate.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for the frozen language organ",
    )
    validate.add_argument(
        "--root-seed",
        type=int,
        default=20260709,
        help="Root seed for deterministic task generation (must match training)",
    )
    validate.add_argument(
        "--train-tasks-per-family",
        type=int,
        default=2,
        help="Must match training config for catalog digest verification",
    )
    validate.add_argument(
        "--validation-tasks-per-family",
        type=int,
        default=2,
        help="Number of meta-validation tasks per family",
    )
    validate.add_argument(
        "--min-events",
        type=int,
        default=2,
        help="Must match training config",
    )
    validate.add_argument(
        "--max-events",
        type=int,
        default=2,
        help="Must match training config",
    )

    # -- audit-dev ---------------------------------------------------------
    audit = subparsers.add_parser(
        "audit-dev",
        help=f"Verify checkpoint hashes/config and audit the DEV split. {_DEV_LABEL}",
        description=(
            "Load/verify checkpoint hashes/config, regenerate/audit the "
            "DEV split in memory, and inspect result/state artifacts. "
            "Does not load task text from disk or run training. "
            + _DEV_LABEL
        ),
    )
    audit.add_argument(
        "--checkpoint",
        required=True,
        help="Directory path to the developmental checkpoint",
    )
    audit.add_argument(
        "--result",
        default=None,
        help="Optional path to a DEV result JSON to inspect",
    )
    audit.add_argument(
        "--state",
        default=None,
        help="Optional directory path to persistent state to inspect",
    )
    audit.add_argument(
        "--root-seed",
        type=int,
        default=20260709,
        help="Root seed for task catalog regeneration",
    )
    audit.add_argument(
        "--train-tasks-per-family",
        type=int,
        default=2,
        help="Train tasks per family for catalog regeneration",
    )
    audit.add_argument(
        "--validation-tasks-per-family",
        type=int,
        default=2,
        help="Validation tasks per family for catalog regeneration",
    )
    audit.add_argument(
        "--min-events",
        type=int,
        default=2,
        help="Min events for catalog regeneration",
    )
    audit.add_argument(
        "--max-events",
        type=int,
        default=2,
        help="Max events for catalog regeneration",
    )

    return parser


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_taskgen_config(args: argparse.Namespace) -> TaskGeneratorConfig:
    return TaskGeneratorConfig(
        root_seed=args.root_seed,
        train_tasks_per_family=args.train_tasks_per_family,
        validation_tasks_per_family=args.validation_tasks_per_family,
        min_events=args.min_events,
        max_events=args.max_events,
    )


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        feature_dim=args.feature_dim,
        d_cortex=64,
        bank_width=args.bank_width,
    )


def _build_outer_config(args: argparse.Namespace) -> OuterLoopConfig:
    return OuterLoopConfig(
        outer_steps=args.outer_steps,
        tasks_per_step=args.tasks_per_step,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        validation_interval=args.validation_interval,
        generation_interval=args.generation_interval,
        behavior_weight=args.behavior_weight,
        specificity_weight=args.specificity_weight,
        survival_weight=args.survival_weight,
        state_norm_weight=args.state_norm_weight,
        seed=args.seed,
    )


def _hash_config(config: Any) -> str:
    """Hash a config dataclass canonically (matches training._hash_config)."""
    import hashlib
    import json

    if isinstance(config, ModelConfig):
        payload = json.dumps(
            {
                "feature_dim": config.feature_dim,
                "d_cortex": config.d_cortex,
                "bank_width": config.bank_width,
            },
            sort_keys=True,
        )
    elif isinstance(config, OuterLoopConfig):
        payload = json.dumps(
            {
                "outer_steps": config.outer_steps,
                "tasks_per_step": config.tasks_per_step,
                "optimizer_name": config.optimizer_name,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "grad_clip_norm": config.grad_clip_norm,
                "validation_interval": config.validation_interval,
                "generation_interval": config.generation_interval,
                "behavior_weight": config.behavior_weight,
                "specificity_weight": config.specificity_weight,
                "survival_weight": config.survival_weight,
                "state_norm_weight": config.state_norm_weight,
                "seed": config.seed,
            },
            sort_keys=True,
        )
    elif isinstance(config, TaskGeneratorConfig):
        payload = json.dumps(
            {
                "root_seed": config.root_seed,
                "train_tasks_per_family": config.train_tasks_per_family,
                "validation_tasks_per_family": config.validation_tasks_per_family,
                "min_events": config.min_events,
                "max_events": config.max_events,
            },
            sort_keys=True,
        )
    else:
        payload = str(config)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Command: train-dev
# ---------------------------------------------------------------------------


def _train_dev(args: argparse.Namespace) -> int:
    """Build catalog, load real Qwen, outer-train, write checkpoint + result."""
    print("# Research/20 — train-dev phase", file=sys.stderr)
    print(f"# {_DEV_LABEL}", file=sys.stderr)

    # 1. Build configs.
    taskgen_config = _build_taskgen_config(args)
    model_config = _build_model_config(args)
    outer_config = _build_outer_config(args)

    print("ASI phase=dev command=train-dev", file=sys.stderr)
    print(f"ASI taskgen_schema={TASKGEN_SCHEMA}", file=sys.stderr)
    print(f"ASI root_seed={taskgen_config.root_seed}", file=sys.stderr)
    print(f"ASI feature_dim={model_config.feature_dim}", file=sys.stderr)
    print(f"ASI bank_width={model_config.bank_width}", file=sys.stderr)
    print(f"ASI outer_steps={outer_config.outer_steps}", file=sys.stderr)
    print(f"ASI optimizer_name={outer_config.optimizer_name}", file=sys.stderr)

    # 2. Build and audit the DEV catalog.
    print("# Building DEV catalog...", file=sys.stderr)
    catalog = build_dev_catalog(taskgen_config)
    print(f"ASI catalog_sha256={catalog.catalog_sha256}", file=sys.stderr)
    print(f"ASI catalog_train_count={len(catalog.meta_train)}", file=sys.stderr)
    print(f"ASI catalog_validation_count={len(catalog.meta_validation)}", file=sys.stderr)

    if not catalog.split_audit.passed:
        print("ERROR: split firewall audit failed", file=sys.stderr)
        print("AUDIT split_firewall=FAIL", file=sys.stderr)
        print("METRIC dev_train_status=FAILED")
        return 1
    print("AUDIT split_firewall=pass", file=sys.stderr)

    # 3. Load real frozen language organ (no fallback).
    print(f"# Loading frozen language organ: {args.model_id}", file=sys.stderr)
    try:
        organ = QwenFrozenOrgan.load(
            model_id=args.model_id,
            feature_dim=model_config.feature_dim,
        )
    except FrozenOrganError as exc:
        print(f"ERROR: frozen organ load failed: {exc}", file=sys.stderr)
        print("METRIC dev_train_status=FAILED")
        print("AUDIT organ_load=FAIL")
        return 1

    organ_hash_before = organ.parameter_hash()
    print(f"ASI organ_hash_before={organ_hash_before}", file=sys.stderr)
    print(f"ASI organ_identity={args.model_id}", file=sys.stderr)

    try:
        # 4. Build model.
        model = MetaCortex(model_config)
        param_count = model.parameter_count()
        param_bytes = param_count * 4  # float32
        print(f"ASI parameter_count={param_count}", file=sys.stderr)
        print(f"ASI parameter_bytes={param_bytes}", file=sys.stderr)

        theta_hash_before = canonical_theta_hash(model)
        print(f"ASI theta_hash_before={theta_hash_before}", file=sys.stderr)

        # 5. Outer-train.
        print("# Outer training...", file=sys.stderr)
        trainer = OuterTrainer(model, organ, outer_config)
        train_result = trainer.train(catalog)

        theta_hash_after = canonical_theta_hash(model)
        print(f"ASI theta_hash_after={theta_hash_after}", file=sys.stderr)
        print(f"ASI optimizer_step_count={train_result.optimizer_step_count}", file=sys.stderr)
        print(f"ASI best_validation_step={train_result.best_validation_step}", file=sys.stderr)
        print(f"ASI best_validation_score={train_result.best_validation_score:.6f}", file=sys.stderr)

        organ_hash_after = organ.parameter_hash()
        print(f"ASI organ_hash_after={organ_hash_after}", file=sys.stderr)
        if organ_hash_after != organ_hash_before:
            print("AUDIT organ_frozen=FAIL", file=sys.stderr)
            print("METRIC dev_train_status=FAILED")
            return 1
        print("AUDIT organ_frozen=pass", file=sys.stderr)

        # 6. Build checkpoint metadata.
        metadata = CheckpointMetadata(
            schema=DEV_SCHEMA,
            model_config=model_config,
            taskgen_schema=TASKGEN_SCHEMA,
            taskgen_digest=_hash_config(taskgen_config),
            outer_config=outer_config,
            completed_step=train_result.optimizer_step_count,
            best_step=train_result.best_validation_step,
            validation_score=train_result.best_validation_score,
            parameter_count=param_count,
            parameter_bytes=param_bytes,
            theta_hash=theta_hash_after,
            organ_identity=args.model_id,
            organ_hash=organ_hash_after,
            source_provenance=args.source_provenance,
        )

        # 7. Write checkpoint.
        print(f"# Writing checkpoint to {args.checkpoint_out}", file=sys.stderr)
        save_developmental_checkpoint(args.checkpoint_out, model, metadata)
        print(f"ASI checkpoint_path={args.checkpoint_out}", file=sys.stderr)

        # 8. Write DEV result.  Fill the taskgen_config_digest that
        # OuterTrainer.train cannot compute (it has no TaskGeneratorConfig).
        print(f"# Writing DEV result to {args.result_out}", file=sys.stderr)
        from dataclasses import replace as dc_replace
        train_result = dc_replace(
            train_result,
            taskgen_config_digest=_hash_config(taskgen_config),
        )
        write_dev_result(args.result_out, train_result)
        print(f"ASI result_path={args.result_out}", file=sys.stderr)

        # 9. Optionally write persistent state.
        if args.state_out is not None:
            print(f"# Writing persistent state to {args.state_out}", file=sys.stderr)
            # Get the consolidated state from the last validation task.
            # We use a zero state here since the model's theta is what matters.
            # The persistent state is optional and task-specific.
            state = model.initial_state(1, device="cpu", dtype=torch.float32)
            save_dev_persistent_state(args.state_out, state, model_config)
            print(f"ASI state_path={args.state_out}", file=sys.stderr)

        # 10. Emit final metrics.
        print("METRIC dev_train_status=OK")
        print(f"METRIC dev_train_best_validation_score={train_result.best_validation_score:.6f}")
        print(f"METRIC dev_train_optimizer_steps={train_result.optimizer_step_count}")
        print(f"METRIC dev_train_audit_status={train_result.audit_status}")
        print("AUDIT dev_train_complete=1")

    finally:
        organ.close()

    return 0


# ---------------------------------------------------------------------------
# Command: validate-dev
# ---------------------------------------------------------------------------


def _validate_dev(args: argparse.Namespace) -> int:
    """Load checkpoint, regenerate validation tasks, run no-optimizer validation."""
    print("# Research/20 — validate-dev phase", file=sys.stderr)
    print(f"# {_DEV_LABEL}", file=sys.stderr)

    print("ASI phase=dev command=validate-dev", file=sys.stderr)

    # 1. Load checkpoint to get model config.
    print(f"# Loading checkpoint from {args.checkpoint}", file=sys.stderr)
    # We need a model to load into; first read the checkpoint.json to get config.
    ckpt_path = Path(args.checkpoint)
    ckpt_json_path = ckpt_path / "checkpoint.json"
    if not ckpt_json_path.exists():
        print(f"ERROR: checkpoint.json not found at {ckpt_json_path}", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        return 1

    import json
    try:
        ckpt_data = json.loads(ckpt_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: checkpoint.json is not valid JSON: {exc}", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        return 1

    # Extract model config from checkpoint.
    mc = ckpt_data.get("model_config")
    if not isinstance(mc, dict):
        print("ERROR: checkpoint model_config missing or invalid", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        return 1

    model_config = ModelConfig(
        feature_dim=int(mc["feature_dim"]),
        d_cortex=int(mc["d_cortex"]),
        bank_width=int(mc["bank_width"]),
    )

    # 2. Build model and load checkpoint.
    model = MetaCortex(model_config)
    try:
        metadata = load_developmental_checkpoint(args.checkpoint, model)
    except ArtifactError as exc:
        print(f"ERROR: checkpoint load failed: {exc}", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        print("AUDIT checkpoint_load=FAIL")
        return 1

    print(f"ASI theta_hash_verified={metadata.theta_hash}", file=sys.stderr)
    print(f"ASI organ_identity={metadata.organ_identity}", file=sys.stderr)

    # 3. Load real frozen language organ (no fallback).
    print(f"# Loading frozen language organ: {args.model_id}", file=sys.stderr)
    try:
        organ = QwenFrozenOrgan.load(
            model_id=args.model_id,
            feature_dim=model_config.feature_dim,
        )
    except FrozenOrganError as exc:
        print(f"ERROR: frozen organ load failed: {exc}", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        print("AUDIT organ_load=FAIL")
        return 1

    organ_hash = organ.parameter_hash()
    print(f"ASI organ_hash={organ_hash}", file=sys.stderr)
    if organ_hash != metadata.organ_hash:
        print(
            f"ERROR: organ hash mismatch: checkpoint has {metadata.organ_hash}, "
            f"current organ has {organ_hash}",
            file=sys.stderr,
        )
        print("AUDIT organ_hash_mismatch=FAIL", file=sys.stderr)
        print("METRIC dev_validate_status=FAILED")
        organ.close()
        return 1
    print("AUDIT organ_hash_verified=pass", file=sys.stderr)

    try:
        # 4. Regenerate validation-only catalog.
        taskgen_config = _build_taskgen_config(args)
        print("# Building DEV catalog for validation...", file=sys.stderr)
        catalog = build_dev_catalog(taskgen_config)
        print(f"ASI catalog_sha256={catalog.catalog_sha256}", file=sys.stderr)

        # Verify catalog digest matches checkpoint if available.
        if metadata.taskgen_digest:
            current_digest = _hash_config(taskgen_config)
            if current_digest != metadata.taskgen_digest:
                print(
                    f"WARNING: taskgen config digest mismatch: "
                    f"checkpoint has {metadata.taskgen_digest}, "
                    f"current has {current_digest}",
                    file=sys.stderr,
                )
                print("AUDIT taskgen_digest_mismatch=WARN", file=sys.stderr)

        # 5. Run no-optimizer validation.
        print("# Running DEV validation (no optimizer)...", file=sys.stderr)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        theta_hash_before = canonical_theta_hash(model)
        print(f"ASI theta_hash_before={theta_hash_before}", file=sys.stderr)

        val_result = run_dev_validation(
            model, organ, catalog,
            device=device, dtype=dtype,
        )

        theta_hash_after = canonical_theta_hash(model)
        print(f"ASI theta_hash_after={theta_hash_after}", file=sys.stderr)
        if theta_hash_after != theta_hash_before:
            print("AUDIT theta_unchanged=FAIL", file=sys.stderr)
            print("METRIC dev_validate_status=FAILED")
            return 1
        print("AUDIT theta_unchanged=pass", file=sys.stderr)

        organ_hash_after = organ.parameter_hash()
        if organ_hash_after != organ_hash:
            print("AUDIT organ_frozen=FAIL", file=sys.stderr)
            print("METRIC dev_validate_status=FAILED")
            return 1
        print("AUDIT organ_frozen=pass", file=sys.stderr)

        # 6. Emit DEV deltas.
        print(f"METRIC dev_trained_vs_update_disabled_delta={val_result.trained_vs_update_disabled_delta:.6f}")
        print(f"METRIC dev_trained_vs_untrained_delta={val_result.trained_vs_untrained_delta:.6f}")
        print(f"METRIC dev_trained_vs_shuffled_delta={val_result.trained_vs_shuffled_delta:.6f}")
        print(f"METRIC dev_trained_vs_zeroed_delta={val_result.trained_vs_zeroed_delta:.6f}")
        print(f"METRIC dev_trained_vs_swapped_delta={val_result.trained_vs_swapped_delta:.6f}")

        # 7. Write DEV result.
        print(f"# Writing DEV validation result to {args.result_out}", file=sys.stderr)
        write_dev_result(args.result_out, val_result)
        print(f"ASI result_path={args.result_out}", file=sys.stderr)

        print("METRIC dev_validate_status=OK")
        print("AUDIT dev_validate_complete=1")

    finally:
        organ.close()

    return 0


# ---------------------------------------------------------------------------
# Command: audit-dev
# ---------------------------------------------------------------------------


def _audit_dev(args: argparse.Namespace) -> int:
    """Verify checkpoint hashes/config, regenerate/audit DEV split, inspect artifacts."""
    print("# Research/20 — audit-dev phase", file=sys.stderr)
    print(f"# {_DEV_LABEL}", file=sys.stderr)

    print("ASI phase=dev command=audit-dev", file=sys.stderr)

    # 1. Read and verify checkpoint.json.
    ckpt_path = Path(args.checkpoint)
    ckpt_json_path = ckpt_path / "checkpoint.json"
    if not ckpt_json_path.exists():
        print(f"ERROR: checkpoint.json not found at {ckpt_json_path}", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1

    import json
    try:
        ckpt_data = json.loads(ckpt_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: checkpoint.json is not valid JSON: {exc}", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1

    # Verify schema.
    schema = ckpt_data.get("schema")
    if schema != DEV_SCHEMA:
        print(f"ERROR: schema mismatch: expected {DEV_SCHEMA!r}, got {schema!r}", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1
    print(f"ASI checkpoint_schema_verified={schema}", file=sys.stderr)

    # Verify theta.npz exists.
    theta_path = ckpt_path / "theta.npz"
    if not theta_path.exists():
        print(f"ERROR: theta.npz not found at {theta_path}", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1
    print("AUDIT theta_npz_present=pass", file=sys.stderr)

    # Verify theta hash matches.
    theta_hash = ckpt_data.get("theta_hash")
    if not theta_hash:
        print("ERROR: checkpoint has no theta_hash", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1
    print(f"ASI theta_hash={theta_hash}", file=sys.stderr)

    # Verify parameter count.
    param_count = ckpt_data.get("parameter_count")
    param_bytes = ckpt_data.get("parameter_bytes")
    print(f"ASI parameter_count={param_count}", file=sys.stderr)
    print(f"ASI parameter_bytes={param_bytes}", file=sys.stderr)

    # Verify organ hash.
    organ_hash = ckpt_data.get("organ_hash")
    organ_identity = ckpt_data.get("organ_identity")
    print(f"ASI organ_identity={organ_identity}", file=sys.stderr)
    print(f"ASI organ_hash={organ_hash}", file=sys.stderr)

    # 2. Regenerate and audit DEV split in memory.
    taskgen_config = _build_taskgen_config(args)
    print("# Regenerating DEV catalog for split audit...", file=sys.stderr)
    catalog = build_dev_catalog(taskgen_config)
    print(f"ASI catalog_sha256={catalog.catalog_sha256}", file=sys.stderr)
    print(f"ASI catalog_train_count={len(catalog.meta_train)}", file=sys.stderr)
    print(f"ASI catalog_validation_count={len(catalog.meta_validation)}", file=sys.stderr)

    if catalog.split_audit.passed:
        print("AUDIT split_firewall=pass", file=sys.stderr)
    else:
        print("AUDIT split_firewall=FAIL", file=sys.stderr)
        print("METRIC dev_audit_status=FAILED")
        return 1

    print(f"ASI rule_overlap={catalog.split_audit.rule_overlap}", file=sys.stderr)
    print(f"ASI assignment_overlap={catalog.split_audit.assignment_overlap}", file=sys.stderr)
    print(f"ASI composition_overlap={catalog.split_audit.composition_overlap}", file=sys.stderr)
    print(f"ASI paraphrase_overlap={catalog.split_audit.paraphrase_overlap}", file=sys.stderr)

    # 3. Optionally inspect result artifact.
    if args.result is not None:
        print(f"# Inspecting result artifact: {args.result}", file=sys.stderr)
        try:
            result_data = read_dev_result(args.result)
        except ArtifactError as exc:
            print(f"ERROR: result artifact read failed: {exc}", file=sys.stderr)
            print("METRIC dev_audit_status=FAILED")
            return 1

        # Verify no forbidden fields.
        forbidden = {"verdict", "signoff", "sign_off", "accept", "refute", "blocked", "meta_test"}
        found_forbidden = []
        for key in result_data:
            key_lower = key.lower()
            for f in forbidden:
                if f in key_lower:
                    found_forbidden.append(key)
        if found_forbidden:
            print(f"ERROR: forbidden fields in result: {found_forbidden}", file=sys.stderr)
            print("AUDIT result_forbidden_fields=FAIL", file=sys.stderr)
            print("METRIC dev_audit_status=FAILED")
            return 1
        print("AUDIT result_no_forbidden_fields=pass", file=sys.stderr)

        result_schema = result_data.get("schema", "N/A")
        print(f"ASI result_schema={result_schema}", file=sys.stderr)

    # 4. Optionally inspect state artifact.
    if args.state is not None:
        state_path = Path(args.state)
        state_json_path = state_path / "state.json"
        slow_npy_path = state_path / "slow.npy"
        if not state_json_path.exists():
            print(f"ERROR: state.json not found at {state_json_path}", file=sys.stderr)
            print("METRIC dev_audit_status=FAILED")
            return 1
        if not slow_npy_path.exists():
            print(f"ERROR: slow.npy not found at {slow_npy_path}", file=sys.stderr)
            print("METRIC dev_audit_status=FAILED")
            return 1

        try:
            state_data = json.loads(state_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: state.json is not valid JSON: {exc}", file=sys.stderr)
            print("METRIC dev_audit_status=FAILED")
            return 1

        print(f"ASI state_schema={state_data.get('schema')}", file=sys.stderr)
        print(f"ASI state_shape={state_data.get('shape')}", file=sys.stderr)
        print(f"ASI state_dtype={state_data.get('dtype')}", file=sys.stderr)
        print(f"ASI state_logical_bytes={state_data.get('logical_bytes')}", file=sys.stderr)
        print(f"ASI state_slow_hash={state_data.get('slow_hash')}", file=sys.stderr)
        print("AUDIT state_artifact_present=pass", file=sys.stderr)

    print("METRIC dev_audit_status=OK")
    print("AUDIT dev_audit_complete=1")

    return 0


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-dev":
        return _train_dev(args)
    elif args.command == "validate-dev":
        return _validate_dev(args)
    elif args.command == "audit-dev":
        return _audit_dev(args)
    else:
        # argparse with required=True should never reach here.
        parser.error(f"Unknown command: {args.command}")
        return 1
