"""R20 INT8 DEV campaign orchestrator — two-stage: training then calibration.

Strict CLI ``run --config <json> --state <json>`` for an
``oczy/r20-int8-dev-orchestrator/v2`` config.

Stage 1 runs the already-prepared training scheduler, requires at least one
verified success for every canonical checkpoint seed while retaining failed
attempts, then recursively discovers checkpoint outputs, binds each to the
expected seed index, verifies identity/hash/provenance, and packages them as
``d0/checkpoint.json`` + ``d0/theta.npz`` etc. in a deterministic gzip
(mtime=0), publishes the private Kaggle dataset once, and verifies it is ready.

Stage 2 loads CALIBRATION_VIEW, builds 5 * ceil(N/5) calibration jobs
with real ``collect-calibration-shard`` args, generates owner-qualified
kernels via prepare_kernel, writes a v3 batch, obtains a dispatch plan
with the configured pool_inventory_limit, runs the scheduler, and exits
success only when all shards are verified.

Persists atomic stage transitions/hashes so restart resumes without
duplicate publication/submission.  Never merges/calibrates/finalizes or
accesses meta-test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

CONFIG_SCHEMA = "oczy/r20-int8-dev-orchestrator/v2"
STATE_SCHEMA = "oczy/r20-int8-dev-orchestrator-state/v1"
CHECKPOINT_SCHEMA = "oczy/meta-cortex/dev/v1"
DEFAULT_CONCURRENCY = 5

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_TRAINING_PLACEHOLDERS = {"{seed}", "{checkpoint_out}", "{result_out}"}
OPAQUE_TARGZ_BASENAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*\.tar\.gz\.bin$"
)
_LOCAL_CHECKPOINT_ARCHIVE_FILENAME = "checkpoints.tar.gz"
_PUBLISHED_CHECKPOINT_ARCHIVE_FILENAME = f"{_LOCAL_CHECKPOINT_ARCHIVE_FILENAME}.bin"

# Stage state values
STAGE_TRAINING = "training"
STAGE_ARCHIVE = "archive"
STAGE_CALIBRATION = "calibration"

_STAGE_ORDER = [STAGE_TRAINING, STAGE_ARCHIVE, STAGE_CALIBRATION]

# Calibration module and subcommand
_CAL_MODULE = "oczy.experiments.meta_cortex"
_CAL_SUBCOMMAND = "collect-calibration-shard"

# Scheduler schema versions for validation
_BATCH_V3 = "oczy/remote-parallel-batch/v3"
_STATE_V4 = "oczy/remote-parallel-state/v4"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OrchestratorError(ValueError):
    """Config, state, or orchestration contract violation."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = json.loads(f.read().decode("utf-8"))
    if not isinstance(raw, dict):
        raise OrchestratorError(f"expected JSON object at {path}")
    return raw


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _runtime_manifest_self_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OrchestratorError("runtime manifest is not canonical JSON") from exc
    return _sha256_bytes(canonical)


def _validate_training_contract(contract: Any) -> dict[str, Any]:
    required_fields = {
        "source_commit",
        "module",
        "runtime_manifest_sha256",
        "argument_template",
    }
    if not isinstance(contract, dict):
        raise OrchestratorError("config.training_contract must be an object")
    if set(contract) != required_fields:
        raise OrchestratorError(
            "config.training_contract must contain exactly "
            "source_commit, module, runtime_manifest_sha256, argument_template"
        )

    source_commit = contract.get("source_commit")
    if not isinstance(source_commit, str) or not HEX40_RE.fullmatch(source_commit):
        raise OrchestratorError(
            "config.training_contract.source_commit must be a 40-char hex Git SHA"
        )
    module = contract.get("module")
    if not isinstance(module, str) or not module:
        raise OrchestratorError("config.training_contract.module must be a non-empty string")
    runtime_hash = contract.get("runtime_manifest_sha256")
    if not isinstance(runtime_hash, str) or not SHA256_RE.fullmatch(runtime_hash):
        raise OrchestratorError(
            "config.training_contract.runtime_manifest_sha256 "
            "must be a 64-char hex SHA-256"
        )

    template = contract.get("argument_template")
    if (
        not isinstance(template, list)
        or not template
        or not all(isinstance(token, str) for token in template)
    ):
        raise OrchestratorError(
            "config.training_contract.argument_template must be a non-empty list of strings"
        )
    if template.count("train-dev") != 1:
        raise OrchestratorError(
            "config.training_contract.argument_template must contain train-dev exactly once"
        )
    for placeholder in sorted(_TRAINING_PLACEHOLDERS):
        if template.count(placeholder) != 1:
            raise OrchestratorError(
                "config.training_contract.argument_template must contain "
                f"{placeholder} exactly once"
            )
    for token in template:
        if ("{" in token or "}" in token) and token not in _TRAINING_PLACEHOLDERS:
            raise OrchestratorError(
                "config.training_contract.argument_template contains an unknown "
                f"or malformed placeholder: {token!r}"
            )
    return contract


def _validate_training_output_path(value: str, field: str, job_name: str) -> None:
    prefix = "/kaggle/working/"
    if (
        not value.startswith(prefix)
        or not value[len(prefix):]
        or any(part in ("", ".", "..") for part in value[len(prefix):].split("/"))
    ):
        raise OrchestratorError(
            f"training job {job_name!r} {field} must be a safe non-empty "
            "absolute /kaggle/working/... path"
        )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_config(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """Validate orchestrator config, resolve relative paths, return enriched dict."""
    if not isinstance(config, dict):
        raise OrchestratorError("config must be a JSON object")

    schema = config.get("schema")
    if schema != CONFIG_SCHEMA:
        raise OrchestratorError(
            f"expected config schema {CONFIG_SCHEMA!r}, got {schema!r}"
        )

    # --- Required string fields ---
    for field in ("campaign_id", "owner", "training_batch_path", "training_state_path"):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            raise OrchestratorError(f"config.{field} must be a non-empty string")


    # --- Pool config ---
    pool_config_path = config.get("pool_config_path")
    dispatch_plan_path = config.get("dispatch_plan_path")
    if dispatch_plan_path is not None and pool_config_path is None:
        raise OrchestratorError(
            "config.pool_config_path is required when config.dispatch_plan_path is set"
        )
    if pool_config_path is not None and not isinstance(pool_config_path, str):
        raise OrchestratorError("config.pool_config_path must be a string if present")
    if dispatch_plan_path is not None and not isinstance(dispatch_plan_path, str):
        raise OrchestratorError("config.dispatch_plan_path must be a string if present")

    pool_inventory_limit = config.get("pool_inventory_limit")
    if pool_config_path is not None:
        if not isinstance(pool_inventory_limit, int) or pool_inventory_limit < 1:
            raise OrchestratorError(
                "config.pool_inventory_limit must be an int >= 1 when pool_config_path is set"
            )
    elif pool_inventory_limit is not None:
        raise OrchestratorError(
            "config.pool_inventory_limit requires pool_config_path to be set"
        )

    # --- Exact training contract ---
    _validate_training_contract(config.get("training_contract"))

    # --- Checkpoint contract ---
    ckpt_contract = config.get("checkpoint_contract")
    if not isinstance(ckpt_contract, dict):
        raise OrchestratorError("config.checkpoint_contract must be an object")
    if not isinstance(ckpt_contract.get("organ_identity"), str) or not ckpt_contract["organ_identity"]:
        raise OrchestratorError("config.checkpoint_contract.organ_identity required")
    if (
        not isinstance(ckpt_contract.get("organ_hash"), str)
        or not SHA256_RE.fullmatch(ckpt_contract["organ_hash"])
    ):
        raise OrchestratorError(
            "config.checkpoint_contract.organ_hash must be a 64-char hex SHA-256"
        )
    seeds = ckpt_contract.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or not all(type(seed) is int for seed in seeds)
        or len(set(seeds)) != 5
    ):
        raise OrchestratorError(
            "config.checkpoint_contract.seeds must be a list of exactly 5 unique ints"
        )

    # --- Checkpoint archive dataset ---
    archive_dataset = config.get("checkpoint_archive_dataset")
    if not isinstance(archive_dataset, str) or "/" not in archive_dataset:
        raise OrchestratorError("config.checkpoint_archive_dataset must be an owner/dataset-slug")

    # --- Model identity and source ---
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise OrchestratorError("config.model_id must be e.g. Qwen/Qwen2.5-0.5B-Instruct")
    model_source = config.get("model_source")
    if not isinstance(model_source, str) or not model_source:
        raise OrchestratorError("config.model_source must be e.g. qwen-lm/qwen2.5/transformers/0.5b-instruct/1")

    # --- Directory layout: jobs, results, archive ---
    for dir_field in ("jobs_dir", "results_dir", "archive_dir"):
        v = config.get(dir_field)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.{dir_field} must be a non-empty string")

    # --- Publication command ---
    pub = config.get("publication")
    if not isinstance(pub, dict):
        raise OrchestratorError("config.publication must be an object")
    for fld in ("dataset_slug", "title", "message"):
        v = pub.get(fld)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.publication.{fld} must be a non-empty string")
    pub_timeout = pub.get("timeout_seconds")
    if pub_timeout is not None and (not isinstance(pub_timeout, int) or pub_timeout < 1):
        raise OrchestratorError("config.publication.timeout_seconds must be a positive int if set")

    # --- Stage 2: calibration config ---
    cal_config = config.get("calibration")
    if not isinstance(cal_config, dict):
        raise OrchestratorError("config.calibration must be an object")

    for fld in ("batch_path", "state_path"):
        v = cal_config.get(fld)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.calibration.{fld} must be a non-empty string")

    # Calibration source
    cal_source = cal_config.get("source")
    if not isinstance(cal_source, dict):
        raise OrchestratorError("config.calibration.source must be an object")
    for fld in ("dataset", "commit", "archive_sha256"):
        v = cal_source.get(fld)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.calibration.source.{fld} must be a non-empty string")
    if not HEX40_RE.fullmatch(cal_source["commit"]):
        raise OrchestratorError("config.calibration.source.commit must be a 40-char hex Git SHA")
    if not SHA256_RE.fullmatch(cal_source["archive_sha256"]):
        raise OrchestratorError("config.calibration.source.archive_sha256 must be a 64-char hex SHA-256")

    # Runtime manifest
    cal_runtime = cal_config.get("runtime_manifest")
    if not isinstance(cal_runtime, dict):
        raise OrchestratorError("config.calibration.runtime_manifest must be an object")

    # Pinned wheel
    pinned_wheel = cal_config.get("pinned_wheel")
    if not isinstance(pinned_wheel, dict):
        raise OrchestratorError("config.calibration.pinned_wheel must be an object")
    for fld in ("dataset", "filename", "sha256"):
        v = pinned_wheel.get(fld)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.calibration.pinned_wheel.{fld} required")

    # Instrument archive — validate all offline-archive fields
    instr_archive = cal_config.get("instrument_archive")
    if not isinstance(instr_archive, dict):
        raise OrchestratorError("config.calibration.instrument_archive must be an object")
    for fld in ("dataset", "filename", "sha256"):
        v = instr_archive.get(fld)
        if not isinstance(v, str) or not v:
            raise OrchestratorError(f"config.calibration.instrument_archive.{fld} required")
    instr_filename = instr_archive["filename"]
    if not OPAQUE_TARGZ_BASENAME_PATTERN.fullmatch(instr_filename):
        raise OrchestratorError(
            "config.calibration.instrument_archive.filename must be a "
            f"basename-only .tar.gz.bin file; got {instr_filename!r}"
        )
    if instr_archive.get("format") != "tar.gz":
        raise OrchestratorError("config.calibration.instrument_archive.format must be 'tar.gz'")
    if instr_archive.get("destination") != "instrument":
        raise OrchestratorError("config.calibration.instrument_archive.destination must be 'instrument'")

    # Kernel slug prefix for calibration job IDs
    kernel_slug_prefix = cal_config.get("kernel_slug_prefix")
    if not isinstance(kernel_slug_prefix, str) or not kernel_slug_prefix:
        raise OrchestratorError("config.calibration.kernel_slug_prefix must be a non-empty string")

    # Concurrency
    concurrency = config.get("concurrency")
    if concurrency is not None:
        if not isinstance(concurrency, int) or concurrency < 1 or concurrency > 5:
            raise OrchestratorError("config.concurrency must be 1..5 if provided")

    # Scheduler tuning: submit pacing and timeouts
    kaggle_submit_interval = config.get("kaggle_submit_interval")
    if kaggle_submit_interval is not None:
        if not isinstance(kaggle_submit_interval, (int, float)) or kaggle_submit_interval < 0:
            raise OrchestratorError("config.kaggle_submit_interval must be a number >= 0")
    push_timeout = config.get("push_timeout_seconds")
    if push_timeout is not None:
        if not isinstance(push_timeout, (int, float)) or push_timeout <= 0:
            raise OrchestratorError("config.push_timeout_seconds must be a number > 0")
    job_timeout = config.get("job_timeout_seconds")
    if job_timeout is not None:
        if not isinstance(job_timeout, (int, float)) or job_timeout <= 0:
            raise OrchestratorError("config.job_timeout_seconds must be a number > 0")
    # Resolve paths
    result = dict(config)
    result["_resolved"] = {
        "training_batch_path": _resolve(base, config["training_batch_path"]),
        "training_state_path": _resolve(base, config["training_state_path"]),
        "jobs_dir": _resolve(base, config["jobs_dir"]),
        "results_dir": _resolve(base, config["results_dir"]),
        "archive_dir": _resolve(base, config["archive_dir"]),
        "cal_batch_path": _resolve(base, cal_config["batch_path"]),
        "cal_state_path": _resolve(base, cal_config["state_path"]),
        "calibration_view_root": _resolve(base, cal_config["calibration_view_root"]),
        "pool_inventory_limit": pool_inventory_limit,
    }
    if pool_config_path:
        result["_resolved"]["pool_config_path"] = _resolve(base, pool_config_path)
    if dispatch_plan_path:
        result["_resolved"]["dispatch_plan_path"] = _resolve(base, dispatch_plan_path)

    return result


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def load_or_init_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {
            "schema": STATE_SCHEMA,
            "campaign_id": "",
            "stages": {},
            "artifacts": {},
            "updated_at": "",
        }
    raw = _read_json(state_path)
    if raw.get("schema") != STATE_SCHEMA:
        raise OrchestratorError(f"state schema mismatch: expected {STATE_SCHEMA!r}")
    return raw


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    _atomic_write_json(state_path, state)


def _stage_complete(state: dict[str, Any], stage: str) -> bool:
    stage_data = state.get("stages", {}).get(stage)
    return isinstance(stage_data, dict) and stage_data.get("status") == "complete"


# ---------------------------------------------------------------------------
# Checkpoint helpers — recursive discovery with seed binding
# ---------------------------------------------------------------------------


def _discover_checkpoints(results_dir: Path) -> list[Path]:
    """Recursively find all checkpoint directories under *results_dir*.

    A checkpoint directory contains both ``checkpoint.json`` and ``theta.npz``.
    """
    found: list[Path] = []
    for root, dirs, files in os.walk(str(results_dir)):
        root_path = Path(root)
        relative_parts = root_path.relative_to(results_dir).parts
        if "provisional" in relative_parts or "provisional.json" in files:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name != "provisional"]
        has_ckpt = "checkpoint.json" in files
        has_theta = "theta.npz" in files
        if has_ckpt and has_theta:
            found.append(root_path)
    if len(found) != 5:
        raise OrchestratorError(
            f"expected exactly 5 checkpoint directories in {results_dir}, found {len(found)}"
        )
    return sorted(found)


def _read_checkpoint_seed(ckpt_dir: Path) -> int:
    data = _read_json(ckpt_dir / "checkpoint.json")
    if data.get("schema") != CHECKPOINT_SCHEMA:
        raise OrchestratorError(f"checkpoint schema mismatch in {ckpt_dir}")
    oc = data.get("outer_config")
    if not isinstance(oc, dict):
        raise OrchestratorError(f"missing outer_config in {ckpt_dir}")
    seed = oc.get("seed")
    if not isinstance(seed, int):
        raise OrchestratorError(f"missing outer_config.seed in {ckpt_dir}")
    return seed


def _validate_checkpoints(
    checkpoint_dirs: list[Path],
    expected_identity: str,
    expected_organ_hash: str,
    expected_seeds: list[int],
    expected_source_provenance: str,
) -> dict[str, Any]:
    """Verify checkpoint identity/hash/seeds and bind each to its seed index."""
    seed_to_ckpt: dict[int, Path] = {}
    verified: list[dict[str, Any]] = []

    for ckpt_dir in checkpoint_dirs:
        data = _read_json(ckpt_dir / "checkpoint.json")
        identity = data.get("organ_identity")
        if not isinstance(identity, str) or not identity:
            raise OrchestratorError(f"missing organ_identity in {ckpt_dir}")
        if identity != expected_identity:
            raise OrchestratorError(
                f"organ_identity mismatch in {ckpt_dir.name}: {identity!r} != {expected_identity!r}"
            )
        organ_hash = data.get("organ_hash")
        if not isinstance(organ_hash, str) or not SHA256_RE.fullmatch(organ_hash):
            raise OrchestratorError(f"missing/invalid organ_hash in {ckpt_dir}")
        if organ_hash != expected_organ_hash:
            raise OrchestratorError(
                f"organ_hash mismatch in {ckpt_dir.name}: {organ_hash!r} != {expected_organ_hash!r}"
            )
        source_prov = data.get("source_provenance")
        if not isinstance(source_prov, str) or not source_prov:
            raise OrchestratorError(f"missing source_provenance in {ckpt_dir}")

        if source_prov != expected_source_provenance:
            raise OrchestratorError(
                f"source_provenance mismatch in {ckpt_dir.name}: "
                f"{source_prov!r} != {expected_source_provenance!r}"
            )
        oc = data.get("outer_config")
        if not isinstance(oc, dict):
            raise OrchestratorError(f"missing outer_config in {ckpt_dir}")
        seed = oc.get("seed")
        if not isinstance(seed, int):
            raise OrchestratorError(f"missing outer_config.seed in {ckpt_dir}")
        if seed not in expected_seeds:
            raise OrchestratorError(
                f"unexpected seed {seed} in {ckpt_dir.name}, expected one of {expected_seeds}"
            )

        if seed in seed_to_ckpt:
            raise OrchestratorError(f"duplicate seed {seed} across checkpoints")

        seed_to_ckpt[seed] = ckpt_dir
        ckpt_fhash = _sha256_file(ckpt_dir / "checkpoint.json")
        theta_fhash = _sha256_file(ckpt_dir / "theta.npz")

        verified.append({
            "dir": str(ckpt_dir),
            "seed": seed,
            "checkpoint_json_sha256": ckpt_fhash,
            "theta_npz_sha256": theta_fhash,
        })

    missing = set(expected_seeds) - set(seed_to_ckpt)
    if missing:
        raise OrchestratorError(f"missing seeds in checkpoints: {sorted(missing)}")

    extras = set(seed_to_ckpt) - set(expected_seeds)
    if extras:
        raise OrchestratorError(f"unexpected extra seeds: {sorted(extras)}")

    # Sort by seed for deterministic ordering
    verified.sort(key=lambda e: e["seed"])
    return {"verified_checkpoints": verified, "checkpoint_count": 5, "seed_map": {s: str(seed_to_ckpt[s]) for s in expected_seeds}}


# ---------------------------------------------------------------------------
# Deterministic archive — gzip mtime=0, entries as d0/, d1/, etc.
# ---------------------------------------------------------------------------


def _build_checkpoint_archive(
    verified: list[dict[str, Any]],
    expected_seeds: list[int],
    output_dir: Path,
) -> tuple[Path, str, int]:
    """Build a deterministic ``checkpoints.tar.gz`` with ``d{idx}/`` entries.

    Uses gzip mtime=0 and PAX_FORMAT for byte-reproducible output.
    Entry paths are ``d{seed_idx}/checkpoint.json`` and ``d{seed_idx}/theta.npz``
    where seed_idx is the exact registered position (0-4) in expected_seeds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_by_seed = {entry["seed"]: entry for entry in verified}

    # Build deterministic gzip: create GzipFile with mtime=0, then tar inside it
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for idx, seed in enumerate(expected_seeds):
                entry = entries_by_seed[seed]
                ckpt_dir = Path(entry["dir"])
                prefix = f"d{idx}"

                for filename in ("checkpoint.json", "theta.npz"):
                    file_path = ckpt_dir / filename
                    info = tf.gettarinfo(str(file_path), arcname=f"{prefix}/{filename}")
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with file_path.open("rb") as f:
                        tf.addfile(info, f)

    raw = buf.getvalue()
    archive_hash = _sha256_bytes(raw)
    archive_path = output_dir / _LOCAL_CHECKPOINT_ARCHIVE_FILENAME
    _atomic_write_bytes(archive_path, raw)

    return archive_path, archive_hash, len(raw)


def _build_archive_manifest(
    verified: list[dict[str, Any]],
    archive_path: Path,
    archive_hash: str,
    archive_size: int,
) -> dict[str, Any]:
    return {
        "schema": "oczy/r20-int8-dev-checkpoint-archive/v1",
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive_size,
        "checkpoint_count": len(verified),
        "checkpoints": sorted(
            [{"seed": e["seed"], "checkpoint_json_sha256": e["checkpoint_json_sha256"],
              "theta_npz_sha256": e["theta_npz_sha256"]} for e in verified],
            key=lambda x: x["seed"],
        ),
    }


# ---------------------------------------------------------------------------
# Publication — exactly once, verify ready
# ---------------------------------------------------------------------------


def _publish_checkpoint_dataset(
    pub_config: dict[str, Any],
    archive_path: Path,
    archive_manifest: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    _run_subprocess: Any = None,
) -> None:
    """Publish checkpoint archive as Kaggle dataset, create-if-absent.

    Persists publication state immediately so resume never re-publishes.
    Polls until ready within configured timeout.
    """
    pub_key = "_checkpoint_dataset_published"
    if state.get(pub_key):
        return

    slug = pub_config["dataset_slug"]
    title = pub_config["title"]
    message = pub_config["message"]
    timeout = pub_config.get("timeout_seconds") or 600

    ds_dir = archive_path.parent / "dataset_dir"
    ds_dir.mkdir(parents=True, exist_ok=True)
    target = ds_dir / _PUBLISHED_CHECKPOINT_ARCHIVE_FILENAME
    legacy_target = ds_dir / _LOCAL_CHECKPOINT_ARCHIVE_FILENAME
    if legacy_target != target:
        legacy_target.unlink(missing_ok=True)
    _atomic_write_bytes(target, archive_path.read_bytes())

    # Write checkpoint_manifest.json alongside the archive
    _atomic_write_json(ds_dir / "checkpoint_manifest.json", archive_manifest)

    # dataset-metadata.json with license "other"
    _atomic_write_json(ds_dir / "dataset-metadata.json", {
        "title": title,
        "id": slug,
        "licenses": [{"name": "other"}],
    })

    run = _run_subprocess or subprocess.run

    # Check if dataset already exists; create if absent, version if present
    check_cmd = ["kaggle", "datasets", "status", slug]
    check_result = run(check_cmd, capture_output=True, text=True, timeout=120)
    dataset_exists = check_result.returncode == 0

    if dataset_exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(ds_dir), "-m", message]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(ds_dir), "--dir-mode", "tar"]

    result = run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise OrchestratorError(
            f"dataset publication failed (exist={dataset_exists}): {result.stderr.strip()}"
        )

    # Poll until ready
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_result = run(
            ["kaggle", "datasets", "status", slug],
            capture_output=True, text=True, timeout=120,
        )
        if "ready" in (status_result.stdout or "").lower():
            break
        time.sleep(10)
    else:
        raise OrchestratorError(f"dataset {slug} not ready within {timeout}s")

    state[pub_key] = True
    save_state(state_path, state)


# ---------------------------------------------------------------------------
# Stage 2: Calibration
# ---------------------------------------------------------------------------


def _load_calibration_view_task_count(calibration_view_root: Path) -> int:
    cal_view_path = calibration_view_root / "CALIBRATION_VIEW.json"
    if not cal_view_path.is_file():
        raise OrchestratorError(f"CALIBRATION_VIEW.json not found at {calibration_view_root}")
    raw = _read_json(cal_view_path)
    task_files = raw.get("task_files")
    if not isinstance(task_files, list) or not task_files:
        raise OrchestratorError("CALIBRATION_VIEW.json has no task_files")
    count = 0
    task_dir = calibration_view_root.parent
    for tf in task_files:
        tf_path = task_dir / tf
        if tf_path.is_file():
            for line in tf_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    count += 1
    if count == 0:
        raise OrchestratorError("CALIBRATION_VIEW has zero calibration tasks")
    return count


def _build_calibration_jobs(
    task_count: int,
    owner: str,
    kernel_slug_prefix: str,
    cal_config: dict[str, Any],
    checkpoint_archive_dataset: str,
    checkpoint_archive_filename: str,
    checkpoint_archive_sha256: str,
    model_id: str,
    model_source: str,
    organ_hash: str,
    archived_checkpoint_seeds: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build 5 × ceil(N/5) calibration job definitions with real CLI args."""
    group_size = 5
    num_groups = (task_count + group_size - 1) // group_size
    num_seeds = len(archived_checkpoint_seeds)
    jobs = []

    source = cal_config["source"]
    runtime_manifest = cal_config["runtime_manifest"]
    pinned_wheel = cal_config["pinned_wheel"]
    instr_archive = cal_config["instrument_archive"]
    if not OPAQUE_TARGZ_BASENAME_PATTERN.fullmatch(checkpoint_archive_filename):
        raise OrchestratorError(
            "checkpoint archive transport filename must be a basename-only "
            f".tar.gz.bin file; got {checkpoint_archive_filename!r}"
        )

    cal_view_path = "/tmp/oczy-offline-inputs/instrument/instrument/public/CALIBRATION_VIEW.json"
    ckpt_base = "/tmp/oczy-offline-inputs/checkpoints"

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, task_count)

        for seed_idx, seed in enumerate(archived_checkpoint_seeds):
            job_idx = group_idx * num_seeds + seed_idx
            dev_seed_index = seed_idx
            kernel_id = f"{owner}/{kernel_slug_prefix}-s{job_idx:02d}"
            title = f"{kernel_slug_prefix.replace('-', ' ').title()} S{job_idx:02d}"

            output_file = f"shard-d{dev_seed_index}-t{start:02d}-{end:02d}.json"
            output_path = f"/kaggle/working/{output_file}"

            arguments = [
                _CAL_SUBCOMMAND,
                "--calibration-view", cal_view_path,
                "--checkpoint", f"{ckpt_base}/d{dev_seed_index}",
                "--model-id", model_id,
                "--organ-hash", organ_hash,
                "--dev-seed-index", str(dev_seed_index),
                "--eval-seed-indices", "0,1,2,3,4",
                "--task-start", str(start),
                "--task-end", str(end),
                "--output", output_path,
            ]

            jobs.append({
                "name": f"cal-s{job_idx:02d}",
                "provider": "kaggle",
                "kernel_id": kernel_id,
                "title": title,
                "phase": "development",
                "profile": "cpu",
                "module": _CAL_MODULE,
                "arguments": arguments,
                "task_range_start": start,
                "task_range_end": end,
                "checkpoint_seed": seed,
                "group_index": group_idx,
                "model_id": model_id,
                "model_source": model_source,
                "source": dict(source),
                "runtime_manifest": dict(runtime_manifest),
                "pinned_wheel": dict(pinned_wheel),
                "instrument_archive": dict(instr_archive),
                "checkpoint_archive": {
                    "dataset": checkpoint_archive_dataset,
                    "filename": checkpoint_archive_filename,
                    "sha256": checkpoint_archive_sha256,
                    "format": "tar.gz",
                    "destination": "checkpoints",
                },
            })

    return jobs


def _generate_calibration_batch(
    jobs: list[dict[str, Any]],
    batch_path: Path,
    jobs_dir: Path,
) -> None:
    """Generate a v3 batch manifest and kernel directories."""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from prepare_research_kernel import prepare_kernel  # type: ignore[import-not-found]

    batch_dir = batch_path.parent
    batch_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    batch_jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for job in jobs:
        kid = job["kernel_id"]
        if kid in seen_ids:
            raise OrchestratorError(f"duplicate kernel_id: {kid}")
        seen_ids.add(kid)

        kernel_dir = jobs_dir / kid.replace("/", "_")
        kernel_dir.mkdir(parents=True, exist_ok=True)

        prepare_kernel(
            output=kernel_dir,
            kernel_id=kid,
            title=job["title"],
            phase=job["phase"],
            profile=job["profile"],
            source_dataset=job["source"]["dataset"],
            source_commit=job["source"]["commit"],
            source_archive_sha256=job["source"]["archive_sha256"],
            module=job["module"],
            arguments=job["arguments"],
            model_source=job.get("model_source"),  # passed through
            instrument_manifest_sha256=None,
            human_signoff_id=None,
            runtime_manifest=job["runtime_manifest"],
            force=True,
            offline_wheels=[job["pinned_wheel"]] if job.get("pinned_wheel") else None,
            offline_archives=[
                job["instrument_archive"],
                job["checkpoint_archive"],
            ],
        )

        batch_jobs.append({
            "name": job["name"],
            "provider": "kaggle",
            "kernel_dir": str(kernel_dir.relative_to(batch_dir)),
            "output_dir": str((jobs_dir / f"output_{job['name']}").relative_to(batch_dir)),
            "runtime_manifest": job["runtime_manifest"],
        })

    _atomic_write_json(batch_path, {
        "schema_version": _BATCH_V3,
        "jobs": batch_jobs,
    })


# ---------------------------------------------------------------------------
# Training batch validation
# ---------------------------------------------------------------------------


def _validate_training_batch(
    batch_path: Path,
    expected_seeds: list[int],
    training_contract: dict[str, Any],
) -> dict[str, int]:
    """Validate unique Kaggle jobs and return each job name's canonical seed."""
    batch = _read_json(batch_path)
    if batch.get("schema_version") != _BATCH_V3:
        raise OrchestratorError(f"training batch must be {_BATCH_V3!r}")
    jobs = batch.get("jobs", [])
    if not isinstance(jobs, list) or len(jobs) < 5:
        count = len(jobs) if isinstance(jobs, list) else 0
        raise OrchestratorError(
            f"training batch must have at least 5 jobs, got {count}"
        )

    seen_names: set[str] = set()
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise OrchestratorError(f"training job #{i} must be an object")
        name = job.get("name")
        if not isinstance(name, str) or not name:
            raise OrchestratorError(f"training job #{i} must have a non-empty name")
        if name in seen_names:
            raise OrchestratorError(f"duplicate training job name {name!r}")
        seen_names.add(name)
        if job.get("provider") != "kaggle":
            raise OrchestratorError(f"training job #{i} must be kaggle")

    expected_seed_set = set(expected_seeds)
    job_seeds: dict[str, int] = {}
    batch_dir = batch_path.resolve().parent
    for job in jobs:
        name = job["name"]
        kernel_dir_value = job.get("kernel_dir")
        if not isinstance(kernel_dir_value, str) or not kernel_dir_value:
            raise OrchestratorError(
                f"training job {name!r} has no parseable job_spec seed: "
                "kernel_dir is missing"
            )
        kernel_dir = Path(kernel_dir_value)
        if not kernel_dir.is_absolute():
            kernel_dir = batch_dir / kernel_dir
        spec_path = kernel_dir / "job_spec.json"
        try:
            spec = _read_json(spec_path)
        except (OSError, UnicodeError, json.JSONDecodeError, OrchestratorError) as exc:
            raise OrchestratorError(
                f"training job {name!r} has no parseable job_spec seed: {exc}"
            ) from exc

        if spec.get("source_commit") != training_contract["source_commit"]:
            raise OrchestratorError(
                f"training job {name!r} source_commit does not match training contract"
            )
        if spec.get("module") != training_contract["module"]:
            raise OrchestratorError(
                f"training job {name!r} module does not match training contract"
            )
        manifest = spec.get("runtime_manifest")
        if not isinstance(manifest, dict):
            raise OrchestratorError(
                f"training job {name!r} runtime_manifest must be an object"
            )
        declared_manifest_hash = manifest.get("manifest_sha256")
        computed_manifest_hash = _runtime_manifest_self_hash(manifest)
        if declared_manifest_hash != computed_manifest_hash:
            raise OrchestratorError(
                f"training job {name!r} runtime manifest self-hash mismatch"
            )
        if declared_manifest_hash != training_contract["runtime_manifest_sha256"]:
            raise OrchestratorError(
                f"training job {name!r} runtime manifest does not match training contract"
            )

        arguments = spec.get("arguments")
        template = training_contract["argument_template"]
        if (
            not isinstance(arguments, list)
            or len(arguments) != len(template)
            or not all(isinstance(argument, str) for argument in arguments)
        ):
            raise OrchestratorError(
                f"training job {name!r} arguments do not match argument_template exactly"
            )
        values: dict[str, str] = {}
        for expected, actual in zip(template, arguments, strict=True):
            if expected in _TRAINING_PLACEHOLDERS:
                values[expected] = actual
            elif actual != expected:
                raise OrchestratorError(
                    f"training job {name!r} arguments do not match "
                    "argument_template exactly"
                )

        raw_seed = values["{seed}"]
        try:
            seed = int(raw_seed, 10)
        except ValueError as exc:
            raise OrchestratorError(
                f"training job {name!r} has no parseable job_spec seed: {raw_seed!r}"
            ) from exc
        if str(seed) != raw_seed:
            raise OrchestratorError(
                f"training job {name!r} seed is not a canonical int: {raw_seed!r}"
            )
        checkpoint_out = values["{checkpoint_out}"]
        result_out = values["{result_out}"]
        _validate_training_output_path(checkpoint_out, "checkpoint_out", name)
        _validate_training_output_path(result_out, "result_out", name)
        if checkpoint_out == result_out:
            raise OrchestratorError(
                f"training job {name!r} checkpoint_out and result_out must be distinct"
            )
        job_seeds[name] = seed

    batch_seed_set = set(job_seeds.values())
    unexpected = batch_seed_set - expected_seed_set
    if unexpected:
        raise OrchestratorError(
            f"unexpected seeds in training batch: {sorted(unexpected)}"
        )
    missing = expected_seed_set - batch_seed_set
    if missing:
        raise OrchestratorError(
            f"missing canonical seeds in training batch: {sorted(missing)}"
        )
    return job_seeds


def _validate_training_state(
    training_state: Path,
    training_batch: Path,
    expected_seeds: list[int],
    training_contract: dict[str, Any],
) -> None:
    """Require a verified success for every seed while permitting failed attempts."""
    job_seeds = _validate_training_batch(
        training_batch, expected_seeds, training_contract
    )
    sched_state = _read_json(training_state)
    if sched_state.get("schema_version") != _STATE_V4:
        raise OrchestratorError(f"training state must be {_STATE_V4!r}")
    jobs = sched_state.get("jobs", {})
    if not isinstance(jobs, dict):
        raise OrchestratorError("training state jobs must be an object")

    batch_names = set(job_seeds)
    state_names = set(jobs)
    missing_names = batch_names - state_names
    if missing_names:
        raise OrchestratorError(
            f"training state missing batch jobs: {sorted(missing_names)}"
        )
    unexpected_names = state_names - batch_names
    if unexpected_names:
        raise OrchestratorError(
            f"training state has unexpected jobs: {sorted(unexpected_names)}"
        )

    succeeded_seeds: set[int] = set()
    for name, job_data in jobs.items():
        if not isinstance(job_data, dict):
            raise OrchestratorError(f"training job {name!r} state must be an object")
        job_state = job_data.get("state")
        if job_state not in _TERMINAL_STATES:
            raise OrchestratorError(
                f"training job {name!r} is not terminal (state={job_state})"
            )
        if job_state == "succeeded":
            if not job_data.get("runtime_manifest_verified"):
                raise OrchestratorError(
                    f"successful training job {name!r} not runtime_manifest_verified"
                )
            expected_runtime_hash = training_contract["runtime_manifest_sha256"]
            if (
                job_data.get("expected_runtime_manifest_sha256")
                != expected_runtime_hash
                or job_data.get("observed_runtime_manifest_sha256")
                != expected_runtime_hash
            ):
                raise OrchestratorError(
                    f"successful training job {name!r} runtime manifest hashes "
                    "do not match training contract"
                )
            succeeded_seeds.add(job_seeds[name])

    missing_successes = set(expected_seeds) - succeeded_seeds
    if missing_successes:
        raise OrchestratorError(
            "no succeeded runtime_manifest_verified training attempt for seeds: "
            f"{sorted(missing_successes)}"
        )



_TERMINAL_STATES: set[str] = {"succeeded", "failed", "cancelled"}

def _clean_stale_lock(state_path: Path) -> None:
    """Remove a lock file whose owner PID is no longer alive."""
    lock_path = Path(str(state_path) + ".owner.lock")
    if not lock_path.is_file():
        return
    raw = lock_path.read_text(encoding="utf-8").strip()
    if not raw.startswith("pid="):
        lock_path.unlink(missing_ok=True)
        return
    try:
        pid = int(raw.split("=", 1)[1])
    except ValueError:
        lock_path.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 0)
    except OSError:
        lock_path.unlink(missing_ok=True)


def _raise_if_jobs_not_terminal(training_state: Path, exit_code: int) -> None:
    """Re-raise only when jobs are still running; let validation handle terminal failures."""
    sched_state = _read_json(training_state)
    jobs = sched_state.get("jobs", {})
    running = [
        name for name, j in jobs.items()
        if j.get("state") not in _TERMINAL_STATES
    ]
    if running:
        raise OrchestratorError(
            f"training scheduler exited with code {exit_code}; "
            f"{len(running)} job(s) still not terminal: {running}"
        )
# ---------------------------------------------------------------------------
# Meta-test guard and runtime-equality check
# ---------------------------------------------------------------------------


def _assert_no_meta_test(config: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(key, str) and ("meta-test" in key.lower() or "meta_test" in key.lower()):
            raise OrchestratorError(f"meta-test prohibited in config key {key!r}")
        if isinstance(value, str) and ("meta-test" in value.lower() or "meta_test" in value.lower()):
            raise OrchestratorError(f"meta-test prohibited in config key {key!r}")
        if isinstance(value, dict):
            for sk, sv in value.items():
                if isinstance(sv, str) and ("meta-test" in sk.lower() or "meta_test" in sk.lower()
                                             or "meta-test" in sv.lower() or "meta_test" in sv.lower()):
                    raise OrchestratorError(f"meta-test prohibited in config.{key}.{sk}")
    for job in jobs:
        for k, v in job.items():
            if isinstance(v, str) and ("meta-test" in v.lower() or "meta_test" in v.lower()):
                raise OrchestratorError(f"meta-test prohibited in job {job.get('name')} {k!r}")
        if isinstance(job.get("arguments"), list):
            for a in job["arguments"]:
                if isinstance(a, str) and ("meta-test" in a.lower() or "meta_test" in a.lower()):
                    raise OrchestratorError(f"meta-test prohibited in job {job.get('name')} args")


def _check_calibration_runtime_equality(
    cal_scheduler_state: Path, batch_jobs: list[dict[str, Any]]
) -> None:
    raw = _read_json(cal_scheduler_state) if cal_scheduler_state.is_file() else None
    if raw is None:
        return
    jobs_state = raw.get("jobs", {})
    for batch_job in batch_jobs:
        name = batch_job["name"]
        jstate = jobs_state.get(name)
        if not isinstance(jstate, dict) or jstate.get("state") != "succeeded":
            continue
        expected = jstate.get("expected_runtime_manifest_sha256")
        observed = jstate.get("observed_runtime_manifest_sha256")
        if expected and observed and expected != observed:
            raise OrchestratorError(
                f"calibration shard {name!r}: runtime manifest mismatch "
                f"(expected={expected}, observed={observed})"
            )


# ---------------------------------------------------------------------------
# Dry-run rendering
# ---------------------------------------------------------------------------


def _dry_run_render(config: dict[str, Any]) -> str:
    resolved = config["_resolved"]
    ckpt_contract = config["checkpoint_contract"]
    cal_config = config["calibration"]
    concurrency = config.get("concurrency", DEFAULT_CONCURRENCY)

    lines = [
        "=== R20 INT8 DEV Orchestrator Dry-Run ===",
        f"Campaign: {config['campaign_id']}",
        f"Owner: {config['owner']}",
        f"Concurrency: {concurrency}",
        f"Model source: {config['model_source']}",
        "",
        "--- Stage 1: Training ---",
        f"  Training batch: {resolved['training_batch_path']}",
        f"  Training state: {resolved['training_state_path']}",
        f"  Jobs dir: {resolved['jobs_dir']}",
        f"  Results dir: {resolved['results_dir']}",
        f"  Archive dir: {resolved['archive_dir']}",
        f"  Pool config: {config.get('pool_config_path', '(none)')}",
        f"  Dispatch plan: {config.get('dispatch_plan_path', '(none)')}",
        f"  Pool inventory limit: {resolved.get('pool_inventory_limit', '(none)')}",
        f"  Checkpoint contract: organ={ckpt_contract['organ_identity']}, "
        f"hash={ckpt_contract['organ_hash'][:12]}...",
        f"  Expected seeds: {ckpt_contract['seeds']}",
        f"  Archive dataset: {config['checkpoint_archive_dataset']}",
        f"  Scheduler command: parallel_scheduler run "
        f"{resolved['training_batch_path']} --state {resolved['training_state_path']}",
        f"  Archive output: {resolved['archive_dir']}/checkpoints.tar.gz",
        "",
        "--- Stage 2: Calibration ---",
        f"  Calibration batch: {resolved['cal_batch_path']}",
        f"  Calibration state: {resolved['cal_state_path']}",
        f"  Calibration view root: {resolved['calibration_view_root']}",
        f"  Source: {cal_config['source']['dataset']} @ {cal_config['source']['commit']}",
        f"  Pinned wheel: {cal_config['pinned_wheel']['filename']}",
        f"  Instrument archive: {cal_config['instrument_archive']['filename']}",
        f"  Module: {_CAL_MODULE} {_CAL_SUBCOMMAND}",
        "  Task range size: 5",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_orchestrator(
    config_path: Path,
    state_path: Path,
    *,
    dry_run: bool = False,
    _run_scheduler: Any = None,
    _run_subprocess: Any = None,
) -> dict[str, Any]:
    raw_config = _read_json(config_path)
    config = validate_config(raw_config, config_path.parent)
    resolved = config["_resolved"]

    if dry_run:
        return {"dry_run": True, "rendered": _dry_run_render(config)}

    state = load_or_init_state(state_path)
    existing_state = bool(
        state.get("campaign_id") or state.get("stages") or state.get("artifacts")
    )

    # Bind campaign
    if state.get("campaign_id") and state["campaign_id"] != config["campaign_id"]:
        raise OrchestratorError(
            f"state bound to campaign {state['campaign_id']!r}, "
            f"cannot run {config['campaign_id']!r}"
        )
    state["campaign_id"] = config["campaign_id"]


    training_contract = config["training_contract"]
    state_training_contract = state.get("training_contract")
    if state_training_contract is None:
        if existing_state:
            raise OrchestratorError(
                "existing orchestrator state is missing required training_contract"
            )
        state["training_contract"] = {
            **training_contract,
            "argument_template": list(training_contract["argument_template"]),
        }
    elif state_training_contract != training_contract:
        raise OrchestratorError(
            "state training_contract does not match config.training_contract"
        )
    concurrency = config.get("concurrency", DEFAULT_CONCURRENCY)
    owner = config["owner"]

    # ------------------------------------------------------------------
    # Stage 1: Training
    # ------------------------------------------------------------------
    if not _stage_complete(state, STAGE_TRAINING):
        training_batch = resolved["training_batch_path"]
        training_state = resolved["training_state_path"]
        pool_cfg = resolved.get("pool_config_path")
        dispatch_plan = resolved.get("dispatch_plan_path")

        # Validate unique Kaggle jobs cover every canonical checkpoint seed.
        expected_training_seeds = config["checkpoint_contract"]["seeds"]
        _validate_training_batch(
            training_batch, expected_training_seeds, training_contract
        )

        scheduler_argv = [
            "run",
            str(training_batch),
            "--state", str(training_state),
            "--kaggle-max", str(concurrency),
            "--poll-interval", "30",
            "--kaggle-submit-interval", str(config.get("kaggle_submit_interval", 60)),
        ]
        if config.get("push_timeout_seconds"):
            scheduler_argv.extend(["--push-timeout", str(config["push_timeout_seconds"])])
        if config.get("job_timeout_seconds"):
            scheduler_argv.extend(["--job-timeout", str(config["job_timeout_seconds"])])
        if pool_cfg and dispatch_plan:
            scheduler_argv.extend([
                "--pool-config", str(pool_cfg),
                "--dispatch-plan", str(dispatch_plan),
            ])

        if _run_scheduler is not None:
            result = _run_scheduler(scheduler_argv, training_state)
            exit_code = result.get("exit_code", 0)
        else:
            script_dir = str(Path(__file__).resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from parallel_scheduler import main as scheduler_main  # type: ignore[import-not-found]
            exit_code = scheduler_main(scheduler_argv)
            result = {"exit_code": exit_code}

        if exit_code != 0:
            # Don't crash-loop when all jobs are terminal.  Let
            # _validate_training_state produce a single clear error.
            _raise_if_jobs_not_terminal(training_state, exit_code)

        # Failed historical attempts are valid when every seed has a verified retry.
        _validate_training_state(
            training_state,
            training_batch,
            expected_training_seeds,
            training_contract,
        )

        state["stages"][STAGE_TRAINING] = {
            "status": "complete",
            "scheduler_state_hash": _sha256_file(training_state),
            "completed_at": _utc_now(),
        }
        save_state(state_path, state)

    # ------------------------------------------------------------------
    # Stage Archive: Discover checkpoints, build archive, publish
    # ------------------------------------------------------------------
    if not _stage_complete(state, STAGE_ARCHIVE):
        ckpt_contract = config["checkpoint_contract"]
        results_dir = resolved["results_dir"]
        archive_dir = resolved["archive_dir"]
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Recursively discover checkpoints
        checkpoint_dirs = _discover_checkpoints(results_dir)

        # Validate identity/hash/seeds
        ckpt_result = _validate_checkpoints(
            checkpoint_dirs,
            expected_identity=ckpt_contract["organ_identity"],
            expected_organ_hash=ckpt_contract["organ_hash"],
            expected_seeds=ckpt_contract["seeds"],
            expected_source_provenance=training_contract["source_commit"],
        )
        verified = ckpt_result["verified_checkpoints"]

        # Build deterministic archive with d0/, d1/, etc.
        archive_path, archive_hash, archive_size = _build_checkpoint_archive(
            verified, ckpt_contract["seeds"], archive_dir
        )
        manifest = _build_archive_manifest(verified, archive_path, archive_hash, archive_size)

        # Publish dataset (exactly once)
        _publish_checkpoint_dataset(
            config["publication"], archive_path, manifest, state, state_path,
            _run_subprocess=_run_subprocess,
        )

        state["stages"][STAGE_ARCHIVE] = {
            "status": "complete",
            "archive_sha256": archive_hash,
            "archive_size_bytes": archive_size,
            "archive_path": str(archive_path),
            "checkpoint_count": ckpt_result["checkpoint_count"],
            "checkpoints": verified,
            "completed_at": _utc_now(),
        }
        state["artifacts"]["checkpoint_archive"] = {
            "path": str(archive_path),
            "sha256": archive_hash,
            "size_bytes": archive_size,
            "manifest": manifest,
        }
        save_state(state_path, state)

    # ------------------------------------------------------------------
    # Stage 2: Calibration
    # ------------------------------------------------------------------
    if not _stage_complete(state, STAGE_CALIBRATION):
        cal_config = config["calibration"]
        cal_batch_path = resolved["cal_batch_path"]
        cal_state_path = resolved["cal_state_path"]
        cal_view_root = resolved["calibration_view_root"]
        jobs_dir = resolved["jobs_dir"]
        model_id = config["model_id"]
        model_source = config["model_source"]
        organ_hash = config["checkpoint_contract"]["organ_hash"]

        # Load task count
        task_count = _load_calibration_view_task_count(cal_view_root)

        # The state records the deterministic local artifact path.  Calibration
        # always consumes its opaque published transport name, including resume.
        archive_manifest = state["artifacts"]["checkpoint_archive"]
        archive_hash = archive_manifest["sha256"]

        # Build calibration jobs
        cal_jobs = _build_calibration_jobs(
            task_count=task_count,
            owner=owner,
            kernel_slug_prefix=cal_config["kernel_slug_prefix"],
            cal_config=cal_config,
            checkpoint_archive_dataset=config["checkpoint_archive_dataset"],
            checkpoint_archive_filename=_PUBLISHED_CHECKPOINT_ARCHIVE_FILENAME,
            checkpoint_archive_sha256=archive_hash,
            model_id=model_id,
            model_source=model_source,
            organ_hash=organ_hash,
            archived_checkpoint_seeds=config["checkpoint_contract"]["seeds"],
            output_dir=cal_batch_path.parent,
        )

        # Guard: no meta-test
        _assert_no_meta_test(config, cal_jobs)

        # Generate batch only if not already generated
        if not cal_batch_path.is_file():
            _generate_calibration_batch(cal_jobs, cal_batch_path, jobs_dir)

        # Dispatch plan
        pool_cfg = resolved.get("pool_config_path")
        cal_dispatch_plan_path = cal_batch_path.parent / "cal_dispatch_plan.json"
        if pool_cfg and not cal_dispatch_plan_path.is_file():
            script_dir = str(Path(__file__).resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from runner_pool import (  # type: ignore[import-not-found]
                create_dispatch_plan,
                inspect_pool,
                write_dispatch_plan,
            )
            pool_config_obj = pool_config_loader(pool_cfg)
            pool_limit = resolved.get("pool_inventory_limit", 20)
            snapshot = inspect_pool(pool_config_obj, limit=pool_limit)
            plan = create_dispatch_plan(pool_config_obj, snapshot, cal_batch_path)
            write_dispatch_plan(cal_dispatch_plan_path, plan)

        _clean_stale_lock(cal_state_path)
        # Run scheduler
        cal_scheduler_argv = [
            "run",
            str(cal_batch_path),
            "--state", str(cal_state_path),
            "--kaggle-max", str(concurrency),
            "--poll-interval", "30",
            "--kaggle-submit-interval", str(config.get("kaggle_submit_interval", 60)),
        ]
        if config.get("push_timeout_seconds"):
            cal_scheduler_argv.extend(["--push-timeout", str(config["push_timeout_seconds"])])
        if config.get("job_timeout_seconds"):
            cal_scheduler_argv.extend(["--job-timeout", str(config["job_timeout_seconds"])])
        if pool_cfg and cal_dispatch_plan_path.is_file():
            cal_scheduler_argv.extend([
                "--pool-config", str(pool_cfg),
                "--dispatch-plan", str(cal_dispatch_plan_path),
            ])

        if _run_scheduler is not None:
            result = _run_scheduler(cal_scheduler_argv, cal_state_path)
            exit_code = result.get("exit_code", 0)
        else:
            from parallel_scheduler import main as scheduler_main  # type: ignore[import-not-found]
            exit_code = scheduler_main(cal_scheduler_argv)
            result = {"exit_code": exit_code}

        if exit_code != 0:
            _raise_if_jobs_not_terminal(cal_state_path, exit_code)

        # Verify all calibration jobs succeeded
        cal_sched_state = _read_json(cal_state_path)
        cal_jobs_state = cal_sched_state.get("jobs", {})
        for job in cal_jobs:
            name = job["name"]
            jstate = cal_jobs_state.get(name)
            if not isinstance(jstate, dict) or jstate.get("state") != "succeeded":
                raise OrchestratorError(
                    f"calibration job {name!r} not succeeded "
                    f"(state={jstate.get('state') if isinstance(jstate, dict) else 'missing'})"
                )

        _check_calibration_runtime_equality(cal_state_path, cal_jobs)

        state["stages"][STAGE_CALIBRATION] = {
            "status": "complete",
            "task_count": task_count,
            "job_count": len(cal_jobs),
            "scheduler_state_hash": _sha256_file(cal_state_path),
            "completed_at": _utc_now(),
        }
        save_state(state_path, state)

    return {
        "campaign_id": config["campaign_id"],
        "stages": {k: v.get("status") for k, v in state.get("stages", {}).items()},
        "all_complete": all(_stage_complete(state, s) for s in _STAGE_ORDER),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def pool_config_loader(path: Path) -> Any:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from runner_pool import load_pool_config  # type: ignore[import-not-found]
    return load_pool_config(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r20_int8_dev_orchestrator",
        description="R20 INT8 DEV campaign orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run the orchestrator stages.")
    run_p.add_argument("--config", required=True, help="Path to orchestrator config JSON.")
    run_p.add_argument("--state", required=True, help="Path to durable state file.")
    run_p.add_argument("--dry-run", action="store_true", default=False,
                       help="Validate inputs and render future commands/artifacts without provider mutation.")
    return parser


def main(argv: list[str] | None = None, **inject: Any) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"unknown command: {args.command}")

    config_path = Path(args.config).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()

    if not config_path.is_file():
        print(f"config file not found: {config_path}", file=sys.stderr)
        return 2

    if args.dry_run:
        result = run_orchestrator(config_path, state_path, dry_run=True, **inject)
        print(result.get("rendered", json.dumps(result, indent=2, sort_keys=True)))
        return 0

    try:
        result = run_orchestrator(config_path, state_path, dry_run=False, **inject)
    except (OrchestratorError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("all_complete", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
