"""Safe, versioned serialization and canonical hashing for Research/20 DEV.

This module owns three separate artifact boundaries:

1. **Developmental checkpoint** — ``checkpoint.json`` + ``theta.npz``.
   Contains *only* detached CPU float32 model parameters (theta).  No F/S,
   optimizer state, tasks, events, traces, features, prompts, labels, or
   targets are ever serialized.

2. **Persistent cortex state** (optional, DEV-only) — ``state.json`` +
   ``slow.npy``.  Requires F to be exactly zero; serializes S only.
   Logical experience-dependent persistent storage is exactly
   ``64 × 64 × 4 = 16 384`` bytes per task.

3. **DEV results** — canonical JSON (``allow_nan=False``, ``sort_keys=True``).
   Contains only hashes, counts, shapes, scalar losses/accuracies, and
   operational pass/fail.  No raw trace text, sign-off, or scientific
   verdict fields.

All canonical hashes iterate sorted tensor names and hash name, dtype,
shape, and C-contiguous raw bytes — never pickle bytes or ZIP timestamps.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import (
    DEV_SCHEMA,
    CheckpointMetadata,
    DevTrainingResult,
    DevValidationResult,
    ModelConfig,
    OuterLoopConfig,
)
from .model import CORTEX_DIM, CortexState, MetaCortex

__all__ = [
    "save_developmental_checkpoint",
    "load_developmental_checkpoint",
    "save_dev_persistent_state",
    "load_dev_persistent_state",
    "write_dev_result",
    "read_dev_result",
    "canonical_theta_hash",
    "canonical_state_hash",
    "ArtifactError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArtifactError(ValueError):
    """Raised on artifact corruption, schema mismatch, or validation failure."""


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def canonical_theta_hash(model: MetaCortex) -> str:
    """Return SHA-256 over sorted named-parameter raw bytes.

    Iterates ``sorted(model.state_dict().keys())`` and hashes name, dtype,
    shape, and C-contiguous raw bytes.  Never hashes pickle bytes or
    depends on ZIP timestamps.
    """
    state = model.state_dict()
    parts: list[str] = []
    for name in sorted(state.keys()):
        tensor = state[name]
        arr = tensor.detach().cpu().contiguous().numpy()
        parts.append(
            f"{name}:{str(arr.dtype)}:{tuple(arr.shape)}:"
            f"{hashlib.sha256(arr.tobytes()).hexdigest()}"
        )
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_state_hash(state: CortexState) -> str:
    """Return SHA-256 over S raw bytes (F must be zero).

    Hashes only the slow state S, since F must be exactly zero before
    any persistent-state save.  The hash covers name, dtype, shape, and
    C-contiguous raw bytes.
    """
    s_arr = state.slow.detach().cpu().contiguous().numpy()
    payload = (
        f"slow:{str(s_arr.dtype)}:{tuple(s_arr.shape)}:"
        f"{hashlib.sha256(s_arr.tobytes()).hexdigest()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_norm(state: CortexState) -> float:
    """L2 norm of S (the persistent component)."""
    return float(state.slow.detach().pow(2).sum().sqrt().item())


# ---------------------------------------------------------------------------
# Developmental checkpoint
# ---------------------------------------------------------------------------

_CHECKPOINT_SCHEMA = DEV_SCHEMA
_THETA_FILE = "theta.npz"
_CHECKPOINT_FILE = "checkpoint.json"


def _model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    return {
        "feature_dim": config.feature_dim,
        "d_cortex": config.d_cortex,
        "bank_width": config.bank_width,
    }


def _outer_config_to_dict(config: OuterLoopConfig) -> dict[str, Any]:
    return {
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
    }


def _metadata_to_dict(metadata: CheckpointMetadata) -> dict[str, Any]:
    """Serialize CheckpointMetadata to a JSON-safe dict."""
    return {
        "schema": metadata.schema,
        "model_config": _model_config_to_dict(metadata.model_config),
        "taskgen_schema": metadata.taskgen_schema,
        "taskgen_digest": metadata.taskgen_digest,
        "outer_config": _outer_config_to_dict(metadata.outer_config),
        "completed_step": metadata.completed_step,
        "best_step": metadata.best_step,
        "validation_score": metadata.validation_score,
        "parameter_count": metadata.parameter_count,
        "parameter_bytes": metadata.parameter_bytes,
        "theta_hash": metadata.theta_hash,
        "organ_identity": metadata.organ_identity,
        "organ_hash": metadata.organ_hash,
        "source_provenance": metadata.source_provenance,
    }


def _metadata_from_dict(data: dict[str, Any]) -> CheckpointMetadata:
    """Reconstruct CheckpointMetadata from a JSON-safe dict, fail-closed."""
    if not isinstance(data, dict):
        raise ArtifactError("checkpoint metadata must be a JSON object")

    # Reject explicit nulls.
    for key, value in data.items():
        if value is None:
            raise ArtifactError(
                f"Explicit null for field '{key}' is not allowed"
            )

    # Parse model_config.
    mc = data.get("model_config")
    if not isinstance(mc, dict):
        raise ArtifactError("model_config must be a JSON object")
    model_config = ModelConfig(
        feature_dim=int(mc["feature_dim"]),
        d_cortex=int(mc["d_cortex"]),
        bank_width=int(mc["bank_width"]),
    )

    # Parse outer_config.
    oc = data.get("outer_config")
    if not isinstance(oc, dict):
        raise ArtifactError("outer_config must be a JSON object")
    outer_config = OuterLoopConfig(
        outer_steps=int(oc["outer_steps"]),
        tasks_per_step=int(oc["tasks_per_step"]),
        optimizer_name=str(oc["optimizer_name"]),
        learning_rate=float(oc["learning_rate"]),
        weight_decay=float(oc["weight_decay"]),
        grad_clip_norm=float(oc["grad_clip_norm"]),
        validation_interval=int(oc["validation_interval"]),
        generation_interval=int(oc["generation_interval"]),
        behavior_weight=float(oc["behavior_weight"]),
        specificity_weight=float(oc["specificity_weight"]),
        survival_weight=float(oc["survival_weight"]),
        state_norm_weight=float(oc["state_norm_weight"]),
        seed=int(oc["seed"]),
    )

    return CheckpointMetadata(
        schema=str(data["schema"]),
        model_config=model_config,
        taskgen_schema=str(data["taskgen_schema"]),
        taskgen_digest=str(data["taskgen_digest"]),
        outer_config=outer_config,
        completed_step=int(data["completed_step"]),
        best_step=int(data["best_step"]),
        validation_score=float(data["validation_score"]),
        parameter_count=int(data["parameter_count"]),
        parameter_bytes=int(data["parameter_bytes"]),
        theta_hash=str(data["theta_hash"]),
        organ_identity=str(data["organ_identity"]),
        organ_hash=str(data["organ_hash"]),
        source_provenance=str(data.get("source_provenance", "unavailable")),
    )


def _expected_param_shapes(model: MetaCortex) -> dict[str, dict[str, Any]]:
    """Return expected parameter names -> {shape, dtype} from model."""
    result: dict[str, dict[str, Any]] = {}
    for name, tensor in model.state_dict().items():
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return result


def save_developmental_checkpoint(
    path: str | Path,
    model: MetaCortex,
    metadata: CheckpointMetadata,
) -> None:
    """Write a developmental checkpoint atomically.

    Creates a directory at *path* containing:

    - ``checkpoint.json``: schema, config, digests, metrics, expected
      parameter names/shapes/dtypes, theta canonical hash, organ
      identity/hash, and honest provenance.
    - ``theta.npz``: **only** detached CPU float32 ``model.state_dict()``
      arrays, loaded with ``allow_pickle=False``.

    Explicitly excludes F, S, latent banks, optimizer/moment state, tasks,
    events, traces, features, prompts, labels, targets, result rows, RNG
    objects, and LM parameters.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Extract theta: detached CPU float32 state_dict arrays only.
    state = model.state_dict()
    theta_arrays: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        arr = tensor.detach().cpu().contiguous().numpy()
        if arr.dtype != np.float32:
            raise ArtifactError(
                f"Parameter {name!r} has dtype {arr.dtype}, expected float32"
            )
        theta_arrays[name] = arr

    # Compute canonical theta hash.
    theta_hash = canonical_theta_hash(model)

    # Verify metadata theta hash matches.
    if metadata.theta_hash != theta_hash:
        raise ArtifactError(
            f"Metadata theta_hash {metadata.theta_hash!r} does not match "
            f"computed hash {theta_hash!r}"
        )

    # Build expected parameter shapes for load-time validation.
    expected_shapes = _expected_param_shapes(model)

    # Build checkpoint.json content.
    checkpoint_dict = _metadata_to_dict(metadata)
    checkpoint_dict["expected_param_shapes"] = expected_shapes
    checkpoint_dict["theta_file"] = _THETA_FILE

    # Write theta.npz atomically.  np.savez appends .npz, so write to
    # a temp dir entry without the extension and then rename.
    tmp_theta = path / "theta.tmp.npz"
    np.savez(tmp_theta, **theta_arrays)  # type: ignore[bad-argument-type]
    os.replace(tmp_theta, path / _THETA_FILE)

    # Write checkpoint.json atomically.
    tmp_json = path / (_CHECKPOINT_FILE + ".tmp")
    tmp_json.write_text(
        json.dumps(checkpoint_dict, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_json, path / _CHECKPOINT_FILE)


def load_developmental_checkpoint(
    path: str | Path,
    model: MetaCortex,
) -> CheckpointMetadata:
    """Load and validate a developmental checkpoint into *model*.

    Validates schema, exact key set, each shape/dtype, canonical raw-byte
    theta hash, parameter count, and model config before strict load.

    Raises :class:`ArtifactError` on any corruption or mismatch.
    """
    path = Path(path)

    # Read checkpoint.json.
    ckpt_path = path / _CHECKPOINT_FILE
    if not ckpt_path.exists():
        raise ArtifactError(f"checkpoint.json not found at {ckpt_path}")
    try:
        ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"checkpoint.json is not valid JSON: {exc}") from exc

    if not isinstance(ckpt_data, dict):
        raise ArtifactError("checkpoint.json must be a JSON object")

    # Validate schema.
    schema = ckpt_data.get("schema")
    if schema != _CHECKPOINT_SCHEMA:
        raise ArtifactError(
            f"Schema mismatch: expected {_CHECKPOINT_SCHEMA!r}, got {schema!r}"
        )

    # Parse and validate metadata (fail-closed on nulls/wrong types).
    metadata = _metadata_from_dict(ckpt_data)

    # Validate model config matches.
    if metadata.model_config != model.config:
        raise ArtifactError(
            f"Model config mismatch: checkpoint has {metadata.model_config}, "
            f"model has {model.config}"
        )

    # Read expected parameter shapes.
    expected_shapes = ckpt_data.get("expected_param_shapes")
    if not isinstance(expected_shapes, dict):
        raise ArtifactError("expected_param_shapes must be a JSON object")

    # Load theta.npz with allow_pickle=False.
    theta_path = path / _THETA_FILE
    if not theta_path.exists():
        raise ArtifactError(f"theta.npz not found at {theta_path}")
    try:
        npz = np.load(theta_path, allow_pickle=False)
    except Exception as exc:
        raise ArtifactError(f"Failed to load theta.npz: {exc}") from exc

    # Validate exact key set.
    npz_keys = set(npz.files)
    model_keys = set(model.state_dict().keys())
    if npz_keys != model_keys:
        missing = model_keys - npz_keys
        extra = npz_keys - model_keys
        raise ArtifactError(
            f"theta.npz key set mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # Validate each shape/dtype and build load dict.
    load_dict: dict[str, torch.Tensor] = {}
    for name in sorted(model_keys):
        arr = npz[name]
        if arr.dtype != np.float32:
            raise ArtifactError(
                f"Parameter {name!r} has dtype {arr.dtype}, expected float32"
            )
        expected = expected_shapes.get(name)
        if not isinstance(expected, dict):
            raise ArtifactError(
                f"Expected shape entry for parameter {name!r} "
                f"must be a JSON object"
            )
        exp_shape_raw = expected.get("shape")
        if not isinstance(exp_shape_raw, list):
            raise ArtifactError(
                f"Expected shape for parameter {name!r} must be a list"
            )
        exp_shape = tuple(exp_shape_raw)
        if tuple(arr.shape) != exp_shape:
            raise ArtifactError(
                f"Parameter {name!r} shape mismatch: "
                f"expected {exp_shape}, got {tuple(arr.shape)}"
            )
        # Validate against model's expected shape.
        model_shape = tuple(model.state_dict()[name].shape)
        if tuple(arr.shape) != model_shape:
            raise ArtifactError(
                f"Parameter {name!r} shape mismatch with model: "
                f"model expects {model_shape}, got {tuple(arr.shape)}"
            )
        load_dict[name] = torch.from_numpy(np.array(arr))

    # Validate parameter count.
    param_count = sum(arr.numel() for arr in load_dict.values())
    if param_count != metadata.parameter_count:
        raise ArtifactError(
            f"Parameter count mismatch: metadata says {metadata.parameter_count}, "
            f"got {param_count}"
        )

    # Validate canonical theta hash.
    # Build a temporary state dict for hash computation.
    theta_hash_parts: list[str] = []
    for name in sorted(load_dict.keys()):
        arr = load_dict[name].detach().cpu().contiguous().numpy()
        theta_hash_parts.append(
            f"{name}:{str(arr.dtype)}:{tuple(arr.shape)}:"
            f"{hashlib.sha256(arr.tobytes()).hexdigest()}"
        )
    computed_hash = hashlib.sha256(
        "\n".join(theta_hash_parts).encode("utf-8")
    ).hexdigest()
    if computed_hash != metadata.theta_hash:
        raise ArtifactError(
            f"Theta hash mismatch: metadata says {metadata.theta_hash!r}, "
            f"computed {computed_hash!r}"
        )

    # Strict load into model.
    model.load_state_dict(load_dict, strict=True)

    return metadata


# ---------------------------------------------------------------------------
# Persistent cortex state (optional, DEV-only)
# ---------------------------------------------------------------------------

_STATE_FILE = "state.json"
_SLOW_FILE = "slow.npy"


def save_dev_persistent_state(
    path: str | Path,
    state: CortexState,
    model_config: ModelConfig,
) -> None:
    """Save persistent cortex state (S only, F must be zero).

    Creates a directory at *path* containing:

    - ``state.json``: schema, shape, dtype, slow-state hash, logical bytes.
    - ``slow.npy``: S only, loaded with ``allow_pickle=False`` on load.

    Raises :class:`ArtifactError` if F is not exactly zero.
    """
    # Require F to be exactly zero.
    if torch.count_nonzero(state.fast).item() != 0:
        raise ArtifactError(
            "Cannot save persistent state: F (fast state) is not zero. "
            "Consolidation must clear F before persistence."
        )

    # Validate S shape: [B, 64, 64] with B == 1 (per-task persistent state).
    s_arr = state.slow.detach().cpu().contiguous().numpy()
    if s_arr.ndim != 3 or s_arr.shape[1] != CORTEX_DIM or s_arr.shape[2] != CORTEX_DIM:
        raise ArtifactError(
            f"S shape must be [B, {CORTEX_DIM}, {CORTEX_DIM}], "
            f"got {tuple(s_arr.shape)}"
        )
    if s_arr.shape[0] != 1:
        raise ArtifactError(
            f"Persistent state batch must be 1, got {s_arr.shape[0]}"
        )
    if s_arr.dtype != np.float32:
        raise ArtifactError(
            f"S dtype must be float32, got {s_arr.dtype}"
        )
    # Squeeze batch dimension for storage: [64, 64].
    s_arr = s_arr[0]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    slow_hash = hashlib.sha256(
        f"slow:{str(s_arr.dtype)}:{tuple(s_arr.shape)}:"
        f"{hashlib.sha256(s_arr.tobytes()).hexdigest()}".encode()
    ).hexdigest()
    logical_bytes = int(s_arr.nbytes)  # 64 * 64 * 4 = 16384

    state_dict = {
        "schema": DEV_SCHEMA,
        "shape": [CORTEX_DIM, CORTEX_DIM],
        "dtype": "float32",
        "slow_hash": slow_hash,
        "logical_bytes": logical_bytes,
        "model_config": _model_config_to_dict(model_config),
        "slow_file": _SLOW_FILE,
    }

    # Write slow.npy atomically.  np.save appends .npy, so write to
    # a temp entry with a .npy suffix and then rename.
    tmp_slow = path / "slow.tmp.npy"
    np.save(tmp_slow, s_arr)
    os.replace(tmp_slow, path / _SLOW_FILE)

    # Write state.json atomically.
    tmp_json = path / (_STATE_FILE + ".tmp")
    tmp_json.write_text(
        json.dumps(state_dict, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_json, path / _STATE_FILE)


def load_dev_persistent_state(
    path: str | Path,
    model_config: ModelConfig,
) -> CortexState:
    """Load persistent cortex state, reconstructing F as zeros.

    Validates schema, shape, dtype, and slow-state hash before returning.
    F is always reconstructed as zeros.
    """
    path = Path(path)

    # Read state.json.
    state_path = path / _STATE_FILE
    if not state_path.exists():
        raise ArtifactError(f"state.json not found at {state_path}")
    try:
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"state.json is not valid JSON: {exc}") from exc

    if not isinstance(state_data, dict):
        raise ArtifactError("state.json must be a JSON object")

    # Validate schema.
    schema = state_data.get("schema")
    if schema != DEV_SCHEMA:
        raise ArtifactError(
            f"Schema mismatch: expected {DEV_SCHEMA!r}, got {schema!r}"
        )

    # Reject nulls.
    for key, value in state_data.items():
        if value is None:
            raise ArtifactError(
                f"Explicit null for field '{key}' is not allowed"
            )

    # Validate shape.
    expected_shape = (CORTEX_DIM, CORTEX_DIM)
    actual_shape = tuple(state_data.get("shape", []))
    if actual_shape != expected_shape:
        raise ArtifactError(
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    # Validate dtype.
    if state_data.get("dtype") != "float32":
        raise ArtifactError(
            f"dtype must be float32, got {state_data.get('dtype')!r}"
        )

    # Validate model_config against saved state config.
    saved_mc = state_data.get("model_config")
    if not isinstance(saved_mc, dict):
        raise ArtifactError("state.json must contain a model_config object")
    for key in ("feature_dim", "d_cortex", "bank_width"):
        if key not in saved_mc:
            raise ArtifactError(f"state.json model_config missing key {key!r}")
    saved_model_config = ModelConfig(
        feature_dim=int(saved_mc["feature_dim"]),
        d_cortex=int(saved_mc["d_cortex"]),
        bank_width=int(saved_mc["bank_width"]),
    )
    if saved_model_config != model_config:
        raise ArtifactError(
            f"Model config mismatch: state has {saved_model_config}, "
            f"caller passed {model_config}"
        )

    # Load slow.npy with allow_pickle=False.
    slow_path = path / _SLOW_FILE
    if not slow_path.exists():
        raise ArtifactError(f"slow.npy not found at {slow_path}")
    try:
        s_arr = np.load(slow_path, allow_pickle=False)
    except Exception as exc:
        raise ArtifactError(f"Failed to load slow.npy: {exc}") from exc

    if s_arr.dtype != np.float32:
        raise ArtifactError(
            f"slow.npy dtype must be float32, got {s_arr.dtype}"
        )
    if tuple(s_arr.shape) != expected_shape:
        raise ArtifactError(
            f"slow.npy shape must be {expected_shape}, got {tuple(s_arr.shape)}"
        )

    # Validate slow hash.
    expected_hash = state_data.get("slow_hash")
    # Recompute hash from loaded array.
    recomputed = hashlib.sha256(
        f"slow:{str(s_arr.dtype)}:{tuple(s_arr.shape)}:"
        f"{hashlib.sha256(s_arr.tobytes()).hexdigest()}".encode()
    ).hexdigest()
    if expected_hash != recomputed:
        raise ArtifactError(
            f"Slow-state hash mismatch: expected {expected_hash!r}, "
            f"computed {recomputed!r}"
        )

    # Reconstruct F as zeros, S from loaded array.
    fast = torch.zeros(
        (1, CORTEX_DIM, CORTEX_DIM), dtype=torch.float32
    )
    slow = torch.from_numpy(np.array(s_arr)).unsqueeze(0)  # [1, 64, 64]

    return CortexState(fast=fast, slow=slow)


# ---------------------------------------------------------------------------
# DEV results
# ---------------------------------------------------------------------------


def write_dev_result(
    path: str | Path,
    result: DevTrainingResult | DevValidationResult,
) -> None:
    """Write a DEV result as canonical JSON, separate from checkpoints/state.

    The JSON is canonical (``allow_nan=False``, ``sort_keys=True``) and
    contains only hashes, counts, shapes, scalar losses/accuracies, and
    operational pass/fail.  No raw trace text, sign-off, or scientific
    verdict fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, DevTrainingResult):
        json_str = result.to_json()
    elif isinstance(result, DevValidationResult):
        json_str = result.to_json()
    else:
        raise ArtifactError(
            f"write_dev_result expects DevTrainingResult or DevValidationResult, "
            f"got {type(result).__name__}"
        )

    # Write atomically.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_str, encoding="utf-8")
    os.replace(tmp, path)


def read_dev_result(
    path: str | Path,
) -> dict[str, Any]:
    """Read a DEV result JSON file and return the parsed dict.

    This is a lightweight reader for audit-dev.  It does not reconstruct
    the full dataclass; it returns the raw JSON dict for inspection.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"Result file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"Result file is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ArtifactError("Result file must be a JSON object")

    # Verify no NaN/Infinity leaked through.
    json.dumps(data, allow_nan=False)

    return data
