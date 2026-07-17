"""Focused contract tests for r20_int8_dev_orchestrator.

Uses fake subprocess/provider hooks and defends:
- Retry-aware training batches require canonical job-spec seeds
- Training state requires a verified success for every canonical seed
- Checkpoint identity/hash/seed rejection
- Deterministic archive bytes (gzip mtime=0)
- Restart after each stage resumes without duplication
- 90 jobs for 90-task view (5x multiplier)
- Five-wide task ranges
- Calibration uses real collect-calibration-shard CLI
- Calibration batch runtime-manifest equality rejection
- No meta-test command/string
- No duplicate publication/submission on resume
- Provider failure remains failed
- kaggle_submit_interval and timeout config fields validate and pass through
- End-to-end exercises generated calibration run.py module/args
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "r20_int8_dev_orchestrator.py"


def _load_module() -> dict[str, Any]:
    import runpy
    return runpy.run_path(str(ORCH_PATH))


_mod = _load_module()

validate_config = _mod["validate_config"]
load_or_init_state = _mod["load_or_init_state"]
save_state = _mod["save_state"]
_stage_complete = _mod["_stage_complete"]
_discover_checkpoints = _mod["_discover_checkpoints"]
_validate_checkpoints = _mod["_validate_checkpoints"]
_build_checkpoint_archive = _mod["_build_checkpoint_archive"]
_build_archive_manifest = _mod["_build_archive_manifest"]
_build_calibration_jobs = _mod["_build_calibration_jobs"]
_generate_calibration_batch = _mod["_generate_calibration_batch"]
_assert_no_meta_test = _mod["_assert_no_meta_test"]
_check_calibration_runtime_equality = _mod["_check_calibration_runtime_equality"]
_validate_training_batch = _mod["_validate_training_batch"]
_validate_training_state = _mod["_validate_training_state"]
_load_calibration_view_task_count = _mod["_load_calibration_view_task_count"]
_publish_checkpoint_dataset = _mod["_publish_checkpoint_dataset"]
run_orchestrator = _mod["run_orchestrator"]
CONFIG_SCHEMA = _mod["CONFIG_SCHEMA"]
STATE_SCHEMA = _mod["STATE_SCHEMA"]
CHECKPOINT_SCHEMA = _mod["CHECKPOINT_SCHEMA"]
STAGE_TRAINING = _mod["STAGE_TRAINING"]
STAGE_ARCHIVE = _mod["STAGE_ARCHIVE"]
STAGE_CALIBRATION = _mod["STAGE_CALIBRATION"]
OrchestratorError = _mod["OrchestratorError"]
_sha256_bytes = _mod["_sha256_bytes"]
_sha256_file = _mod["_sha256_file"]
_CAL_MODULE = _mod["_CAL_MODULE"]
_CAL_SUBCOMMAND = _mod["_CAL_SUBCOMMAND"]

EXPECTED_SEEDS = [100, 200, 300, 400, 500]
TRAINING_SOURCE_COMMIT = "1" * 40
TRAINING_MODULE = "oczy.experiments.meta_cortex"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seal_runtime_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(
        sealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    sealed["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return sealed


def _fake_runtime_manifest() -> dict[str, Any]:
    return _seal_runtime_manifest({
        "schema_version": "oczy/runtime-manifest/v2",
        "model": {
            "convention": "transformers-pretrained-directory",
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": None,
            "architecture": "Qwen2ForCausalLM",
            "parameters": 494032768,
            "artifact_files": [],
        },
        "packages": {"python": "3.10.0", "torch": "2.5.0"},
        "greedy_generation": {"max_tokens": 128},
        "quantization": {
            "recipe": "torchao_w8a32",
            "torchao_version": "0.17.0",
            "config_dict": {"version": 2, "granularity": "per_row"},
        },
    })


def _fake_cal_source() -> dict[str, Any]:
    return {
        "dataset": "kino/oczy-source-aaaaaaaaaaaa",
        "commit": "a" * 40,
        "archive_sha256": "b" * 64,
    }


def _fake_cal_config() -> dict[str, Any]:
    return {
        "batch_path": "cal_batch.json",
        "state_path": "cal_state.json",
        "source": _fake_cal_source(),
        "runtime_manifest": _fake_runtime_manifest(),
        "pinned_wheel": {"dataset": "kino/wheels", "filename": "oczy-0.1.0-py3-none-any.whl", "sha256": "c" * 64},
        "instrument_archive": {"dataset": "kino/instruments", "filename": "instrument.tar.gz",
                               "sha256": "d" * 64, "format": "tar.gz", "destination": "instrument"},
        "calibration_view_root": "cal_view",
        "kernel_slug_prefix": "oczy-r20-int8-cal",
    }


def _fake_training_contract() -> dict[str, Any]:
    return {
        "source_commit": TRAINING_SOURCE_COMMIT,
        "module": TRAINING_MODULE,
        "runtime_manifest_sha256": _fake_runtime_manifest()["manifest_sha256"],
        "argument_template": [
            "train-dev",
            "--train-tasks-per-family",
            "30",
            "--outer-steps",
            "100",
            "--seed",
            "{seed}",
            "--checkpoint-out",
            "{checkpoint_out}",
            "--result-out",
            "{result_out}",
        ],
    }


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "campaign_id": "r20-int8-dev-test",
        "owner": "kino",
        "training_batch_path": "training_batch.json",
        "training_state_path": "training_state.json",
        "jobs_dir": "jobs",
        "results_dir": "results",
        "archive_dir": "archive",
        "training_contract": _fake_training_contract(),
        "checkpoint_contract": {
            "organ_identity": "Qwen2.5-0.5B-Instruct-meta-cortex-v2",
            "organ_hash": "e" * 64,
            "seeds": [100, 200, 300, 400, 500],
        },
        "checkpoint_archive_dataset": "kino/oczy-r20-checkpoints",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_source": "qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        "publication": {
            "dataset_slug": "kino/oczy-r20-checkpoints",
            "title": "R20 Checkpoints",
            "message": "DEV checkpoints v1",
        },
        "calibration": _fake_cal_config(),
    }
    cfg.update(overrides)
    return cfg


def _make_checkpoint_dir(
    root: Path, name: str, *, identity: str = "Qwen2.5-0.5B-Instruct-meta-cortex-v2",
    organ_hash: str = "e" * 64, seed: int = 100, theta_content: bytes = b"",
    source_provenance: str = TRAINING_SOURCE_COMMIT,
) -> Path:
    cdir = root / name
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": {"feature_dim": 896, "d_cortex": 64, "bank_width": 3},
        "taskgen_schema": "oczy/meta-cortex/taskgen/v1-dev",
        "taskgen_digest": "a" * 64,
        "outer_config": {
            "outer_steps": 100, "tasks_per_step": 10, "optimizer_name": "adam",
            "learning_rate": 0.001, "weight_decay": 0.0, "grad_clip_norm": 1.0,
            "validation_interval": 5, "generation_interval": 10,
            "behavior_weight": 1.0, "specificity_weight": 0.1,
            "survival_weight": 0.0, "state_norm_weight": 0.0,
            "seed": seed,
        },
        "completed_step": 100, "best_step": 95, "validation_score": 0.85,
        "parameter_count": 100000, "parameter_bytes": 400000,
        "theta_hash": "f" * 64, "organ_identity": identity,
        "organ_hash": organ_hash, "source_provenance": source_provenance,
        "expected_param_shapes": {}, "theta_file": "theta.npz",
    }
    (cdir / "checkpoint.json").write_text(json.dumps(ckpt, sort_keys=True))
    (cdir / "theta.npz").write_bytes(theta_content or b"fake-theta-bytes")
    return cdir


def _make_calibration_view(root: Path, task_count: int = 90) -> Path:
    cal_view_dir = root / "cal_view"
    cal_view_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = cal_view_dir.parent / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / "cal_tasks.jsonl"
    task_file.write_text("\n".join("{}" for _ in range(task_count)), encoding="utf-8")
    (cal_view_dir / "CALIBRATION_VIEW.json").write_text(json.dumps({
        "schema": "oczy/meta-cortex/calibration-view/v1",
        "task_files": ["tasks/cal_tasks.jsonl"],
    }), encoding="utf-8")
    return cal_view_dir


def _make_training_batch(
    path: Path,
    job_count: int = 5,
    *,
    seeds: list[int] | None = None,
    names: list[str] | None = None,
    training_contract: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_seeds = seeds or EXPECTED_SEEDS[:job_count]
    selected_names = names or [f"job{i}" for i in range(len(selected_seeds))]
    contract = training_contract or _fake_training_contract()
    assert len(selected_names) == len(selected_seeds)
    jobs = []
    for i, (name, seed) in enumerate(zip(selected_names, selected_seeds, strict=True)):
        kernel_dir = path.parent / f"kernel_{i}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        arguments = [
            {
                "{seed}": str(seed),
                "{checkpoint_out}": f"/kaggle/working/checkpoint-{i}",
                "{result_out}": f"/kaggle/working/result-{i}.json",
            }.get(token, token)
            for token in contract["argument_template"]
        ]
        (kernel_dir / "job_spec.json").write_text(
            json.dumps({
                "source_commit": contract["source_commit"],
                "module": contract["module"],
                "runtime_manifest": _fake_runtime_manifest(),
                "arguments": arguments,
            }),
            encoding="utf-8",
        )
        jobs.append({
            "name": name,
            "provider": "kaggle",
            "kernel_dir": kernel_dir.name,
            "output_dir": f"output_{i}",
            "runtime_manifest": _fake_runtime_manifest(),
        })
    path.write_text(json.dumps({
        "schema_version": "oczy/remote-parallel-batch/v3",
        "jobs": jobs,
    }), encoding="utf-8")


def _read_training_job_spec(
    batch_path: Path, index: int = 0
) -> tuple[Path, dict[str, Any]]:
    spec_path = batch_path.parent / f"kernel_{index}" / "job_spec.json"
    return spec_path, json.loads(spec_path.read_text(encoding="utf-8"))


def _write_training_job_spec(path: Path, spec: dict[str, Any]) -> None:
    path.write_text(json.dumps(spec), encoding="utf-8")


def _make_training_state(
    path: Path,
    job_state: str = "succeeded",
    verified: bool = True,
    *,
    attempts: dict[str, tuple[str, bool]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_attempts = attempts or {
        f"job{i}": (job_state, verified) for i in range(5)
    }
    runtime_hash = _fake_training_contract()["runtime_manifest_sha256"]
    path.write_text(json.dumps({
        "schema_version": "oczy/remote-parallel-state/v4",
        "jobs": {
            name: {
                "state": state,
                "expected_runtime_manifest_sha256": runtime_hash,
                "observed_runtime_manifest_sha256": runtime_hash,
                "runtime_manifest_verified": manifest_verified,
            }
            for name, (state, manifest_verified) in selected_attempts.items()
        },
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def test_config_rejects_wrong_schema(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorError, match="expected config schema"):
        validate_config({"schema": "wrong/v1"}, tmp_path)


def test_config_schema_is_v2_and_rejects_v1(tmp_path: Path) -> None:
    assert CONFIG_SCHEMA == "oczy/r20-int8-dev-orchestrator/v2"
    cfg = _minimal_config()
    cfg["schema"] = "oczy/r20-int8-dev-orchestrator/v1"
    with pytest.raises(OrchestratorError, match="expected config schema"):
        validate_config(cfg, tmp_path)


def test_config_rejects_missing_owner(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["owner"]
    with pytest.raises(OrchestratorError, match="owner"):
        validate_config(cfg, tmp_path)


def test_config_rejects_missing_model_source(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["model_source"]
    with pytest.raises(OrchestratorError, match="model_source"):
        validate_config(cfg, tmp_path)


def test_config_rejects_missing_model_id(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["model_id"]
    with pytest.raises(OrchestratorError, match="model_id"):
        validate_config(cfg, tmp_path)


def test_config_rejects_missing_publication(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["publication"]
    with pytest.raises(OrchestratorError, match="publication"):
        validate_config(cfg, tmp_path)


def test_config_rejects_missing_dir_fields(tmp_path: Path) -> None:
    for field in ("jobs_dir", "results_dir", "archive_dir"):
        cfg = _minimal_config()
        del cfg[field]
        with pytest.raises(OrchestratorError, match=field):
            validate_config(cfg, tmp_path)


def test_config_requires_pool_config_with_dispatch_plan(tmp_path: Path) -> None:
    cfg = _minimal_config(dispatch_plan_path="plan.json")
    with pytest.raises(OrchestratorError, match="pool_config_path is required"):
        validate_config(cfg, tmp_path)


def test_config_requires_pool_inventory_limit(tmp_path: Path) -> None:
    cfg = _minimal_config(pool_config_path="pool.json")
    with pytest.raises(OrchestratorError, match="pool_inventory_limit"):
        validate_config(cfg, tmp_path)


def test_config_accepts_valid_pool_inventory_limit(tmp_path: Path) -> None:
    cfg = _minimal_config(pool_config_path="pool.json", pool_inventory_limit=2)
    result = validate_config(cfg, tmp_path)
    assert result["_resolved"]["pool_inventory_limit"] == 2


def test_config_rejects_pool_inventory_limit_zero(tmp_path: Path) -> None:
    cfg = _minimal_config(pool_config_path="pool.json", pool_inventory_limit=0)
    with pytest.raises(OrchestratorError, match="pool_inventory_limit must be an int >= 1"):
        validate_config(cfg, tmp_path)


def test_config_rejects_bad_organ_hash(tmp_path: Path) -> None:
    cfg = _minimal_config()
    cfg["checkpoint_contract"]["organ_hash"] = "not-hex"
    with pytest.raises(OrchestratorError, match="organ_hash must be"):
        validate_config(cfg, tmp_path)


def test_config_rejects_wrong_seed_count(tmp_path: Path) -> None:
    cfg = _minimal_config()
    cfg["checkpoint_contract"]["seeds"] = [1, 2, 3]
    with pytest.raises(OrchestratorError, match="exactly 5"):
        validate_config(cfg, tmp_path)


def test_config_rejects_duplicate_checkpoint_seeds(tmp_path: Path) -> None:
    cfg = _minimal_config()
    cfg["checkpoint_contract"]["seeds"] = [500, 100, 400, 200, 500]
    with pytest.raises(OrchestratorError, match="exactly 5 unique ints"):
        validate_config(cfg, tmp_path)


def test_config_rejects_bad_source_commit(tmp_path: Path) -> None:
    cfg = _minimal_config()
    cfg["calibration"]["source"]["commit"] = "bad"
    with pytest.raises(OrchestratorError, match="40-char hex"):
        validate_config(cfg, tmp_path)


def test_config_requires_training_contract(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["training_contract"]
    with pytest.raises(OrchestratorError, match="training_contract"):
        validate_config(cfg, tmp_path)


@pytest.mark.parametrize(
    "template",
    [
        [
            "train-dev", "--seed", "{seed}", "--checkpoint-out",
            "{checkpoint_out}", "--result-out",
        ],
        [
            "train-dev", "--seed", "{seed}", "{seed}", "--checkpoint-out",
            "{checkpoint_out}", "--result-out", "{result_out}",
        ],
        [
            "train-dev", "--seed", "{seed}", "--checkpoint-out",
            "{checkpoint_out}", "--result-out", "{result_out}", "{unknown}",
        ],
        [
            "train-dev", "--seed", "{seed", "--checkpoint-out",
            "{checkpoint_out}", "--result-out", "{result_out}",
        ],
        [
            "other-command", "--seed", "{seed}", "--checkpoint-out",
            "{checkpoint_out}", "--result-out", "{result_out}",
        ],
    ],
)
def test_config_rejects_missing_duplicate_or_malformed_placeholders(
    tmp_path: Path, template: list[str]
) -> None:
    cfg = _minimal_config()
    cfg["training_contract"]["argument_template"] = template
    with pytest.raises(OrchestratorError, match="placeholder|exactly once"):
        validate_config(cfg, tmp_path)


def test_config_rejects_malformed_training_identity_fields(tmp_path: Path) -> None:
    for field, value, match in (
        ("source_commit", "abc", "source_commit"),
        ("module", "", "module"),
        ("runtime_manifest_sha256", "abc", "runtime_manifest_sha256"),
    ):
        cfg = _minimal_config()
        cfg["training_contract"][field] = value
        with pytest.raises(OrchestratorError, match=match):
            validate_config(cfg, tmp_path)


def test_valid_config_passes(tmp_path: Path) -> None:
    cfg = _minimal_config()
    result = validate_config(cfg, tmp_path)
    assert result["_resolved"]["training_batch_path"] == tmp_path / "training_batch.json"


def test_config_kaggle_submit_interval_defaults(tmp_path: Path) -> None:
    cfg = _minimal_config()
    result = validate_config(cfg, tmp_path)
    assert result.get("kaggle_submit_interval") is None  # defaults not stored


def test_config_kaggle_submit_interval_valid(tmp_path: Path) -> None:
    cfg = _minimal_config(kaggle_submit_interval=60)
    result = validate_config(cfg, tmp_path)
    assert result["kaggle_submit_interval"] == 60


def test_config_rejects_negative_submit_interval(tmp_path: Path) -> None:
    cfg = _minimal_config(kaggle_submit_interval=-1)
    with pytest.raises(OrchestratorError, match="kaggle_submit_interval must be a number >= 0"):
        validate_config(cfg, tmp_path)


def test_config_rejects_zero_timeout(tmp_path: Path) -> None:
    cfg = _minimal_config(push_timeout_seconds=0)
    with pytest.raises(OrchestratorError, match="push_timeout_seconds must be a number > 0"):
        validate_config(cfg, tmp_path)

    cfg2 = _minimal_config(job_timeout_seconds=0)
    with pytest.raises(OrchestratorError, match="job_timeout_seconds must be a number > 0"):
        validate_config(cfg2, tmp_path)


def test_config_accepts_timeouts(tmp_path: Path) -> None:
    cfg = _minimal_config(push_timeout_seconds=43200, job_timeout_seconds=43200, kaggle_submit_interval=60)
    result = validate_config(cfg, tmp_path)
    assert result["push_timeout_seconds"] == 43200
    assert result["job_timeout_seconds"] == 43200


# ---------------------------------------------------------------------------
# Training batch validation
# ---------------------------------------------------------------------------

def test_training_batch_requires_at_least_5_jobs(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp, 3)
    with pytest.raises(OrchestratorError, match="at least 5 jobs"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_requires_all_kaggle(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    batch = json.loads(bp.read_text(encoding="utf-8"))
    batch["jobs"][-1]["provider"] = "colab"
    bp.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="must be kaggle"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_accepts_existing_five_job_case(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    assert _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract()) == {
        f"job{i}": seed for i, seed in enumerate(EXPECTED_SEEDS)
    }


def test_training_batch_rejects_duplicate_names(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(
        bp,
        names=["job0", "job1", "job2", "job3", "job3"],
    )
    with pytest.raises(OrchestratorError, match="duplicate training job name"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_retry_with_unparseable_seed(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(
        bp,
        seeds=EXPECTED_SEEDS + [100],
        names=[f"job{i}" for i in range(5)] + ["job0-retry"],
    )
    retry_spec, spec = _read_training_job_spec(bp, 5)
    seed_index = spec["arguments"].index("--seed") + 1
    spec["arguments"][seed_index] = "not-an-int"
    _write_training_job_spec(retry_spec, spec)
    with pytest.raises(OrchestratorError, match="no parseable job_spec seed"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_missing_canonical_seed(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp, seeds=[100, 200, 300, 400, 400])
    with pytest.raises(OrchestratorError, match="missing canonical seeds.*500"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_unexpected_seed(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp, seeds=[100, 200, 300, 400, 999])
    with pytest.raises(OrchestratorError, match="unexpected seeds.*999"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


@pytest.mark.parametrize("replacement", ["15", "8"])
def test_training_batch_rejects_train_tasks_per_family_mismatch(
    tmp_path: Path, replacement: str
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    position = spec["arguments"].index("--train-tasks-per-family") + 1
    spec["arguments"][position] = replacement
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="argument_template exactly"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_any_other_fixed_argument_mismatch(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    position = spec["arguments"].index("--outer-steps") + 1
    spec["arguments"][position] = "99"
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="argument_template exactly"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_commit", "2" * 40, "source_commit"),
        ("module", "oczy.experiments.other", "module"),
    ],
)
def test_training_batch_rejects_source_or_module_mismatch(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    spec[field] = value
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match=match):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_runtime_manifest_contract_mismatch(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    manifest = spec["runtime_manifest"]
    manifest["packages"]["python"] = "3.11.0"
    spec["runtime_manifest"] = _seal_runtime_manifest(manifest)
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="runtime manifest does not match"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_runtime_manifest_bad_self_hash(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    spec["runtime_manifest"]["packages"]["python"] = "3.11.0"
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="self-hash mismatch"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "checkpoint",
        "/kaggle/input/checkpoint",
        "/kaggle/working/",
        "/kaggle/working/../input/checkpoint",
        "/kaggle/working//checkpoint",
    ],
)
def test_training_batch_rejects_unsafe_output_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    checkpoint_index = spec["arguments"].index("--checkpoint-out") + 1
    spec["arguments"][checkpoint_index] = unsafe_path
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="safe non-empty absolute"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_identical_output_paths(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    checkpoint_index = spec["arguments"].index("--checkpoint-out") + 1
    result_index = spec["arguments"].index("--result-out") + 1
    spec["arguments"][result_index] = spec["arguments"][checkpoint_index]
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="must be distinct"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_batch_rejects_noncanonical_seed(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    _make_training_batch(bp)
    spec_path, spec = _read_training_job_spec(bp)
    seed_index = spec["arguments"].index("--seed") + 1
    spec["arguments"][seed_index] = "0100"
    _write_training_job_spec(spec_path, spec)
    with pytest.raises(OrchestratorError, match="not a canonical int"):
        _validate_training_batch(bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_state_accepts_failed_attempt_and_succeeded_retry(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    sp = tmp_path / "state.json"
    names = [f"job{i}" for i in range(5)] + ["job0-retry"]
    _make_training_batch(
        bp,
        seeds=EXPECTED_SEEDS + [100],
        names=names,
    )
    _make_training_state(
        sp,
        attempts={
            **{f"job{i}": ("succeeded", True) for i in range(1, 5)},
            "job0": ("failed", False),
            "job0-retry": ("succeeded", True),
        },
    )
    _validate_training_state(sp, bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_state_rejects_unverified_sole_success(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    sp = tmp_path / "state.json"
    _make_training_batch(bp)
    _make_training_state(
        sp,
        attempts={
            **{f"job{i}": ("succeeded", True) for i in range(1, 5)},
            "job0": ("succeeded", False),
        },
    )
    with pytest.raises(OrchestratorError, match="not runtime_manifest_verified"):
        _validate_training_state(sp, bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_state_rejects_verified_runtime_hash_outside_contract(
    tmp_path: Path,
) -> None:
    bp = tmp_path / "batch.json"
    sp = tmp_path / "state.json"
    _make_training_batch(bp)
    _make_training_state(sp)
    state = json.loads(sp.read_text(encoding="utf-8"))
    state["jobs"]["job0"]["expected_runtime_manifest_sha256"] = "f" * 64
    state["jobs"]["job0"]["observed_runtime_manifest_sha256"] = "f" * 64
    sp.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="hashes do not match"):
        _validate_training_state(sp, bp, EXPECTED_SEEDS, _fake_training_contract())


def test_training_state_requires_a_success_for_every_seed(tmp_path: Path) -> None:
    bp = tmp_path / "batch.json"
    sp = tmp_path / "state.json"
    _make_training_batch(bp)
    _make_training_state(sp, job_state="failed")
    with pytest.raises(OrchestratorError, match="no succeeded.*100"):
        _validate_training_state(sp, bp, EXPECTED_SEEDS, _fake_training_contract())


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_state_init_empty(tmp_path: Path) -> None:
    s = load_or_init_state(tmp_path / "state.json")
    assert s["schema"] == STATE_SCHEMA
    assert s["stages"] == {}


def test_state_save_reload(tmp_path: Path) -> None:
    sp = tmp_path / "state.json"
    s = load_or_init_state(sp)
    s["campaign_id"] = "test"
    save_state(sp, s)
    assert load_or_init_state(sp)["campaign_id"] == "test"


def test_state_rejects_wrong_schema(tmp_path: Path) -> None:
    sp = tmp_path / "state.json"
    sp.write_text('{"schema": "wrong/v1"}')
    with pytest.raises(OrchestratorError, match="state schema mismatch"):
        load_or_init_state(sp)


def test_stage_complete_detection(tmp_path: Path) -> None:
    sp = tmp_path / "state.json"
    s = load_or_init_state(sp)
    assert not _stage_complete(s, STAGE_TRAINING)
    s["stages"][STAGE_TRAINING] = {"status": "complete"}
    assert _stage_complete(s, STAGE_TRAINING)


# ---------------------------------------------------------------------------
# Checkpoint discovery + validation
# ---------------------------------------------------------------------------

def test_discover_checkpoints_exactly_five(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    for i in range(3):
        _make_checkpoint_dir(rd / f"sub_{i}", "ckpt", seed=100 + i * 100)
    with pytest.raises(OrchestratorError, match="expected exactly 5"):
        _discover_checkpoints(rd)


def test_discover_checkpoints_recursive(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    # Nest them in various subdirs
    for i in range(5):
        _make_checkpoint_dir(rd / f"run_{i}" / "out" / "ckpt", "x", seed=100 + i * 100)
    found = _discover_checkpoints(rd)
    assert len(found) == 5


def test_discover_checkpoints_ignores_provisional_snapshots(
    tmp_path: Path,
) -> None:
    rd = tmp_path / "results"
    final_dirs = [
        _make_checkpoint_dir(rd / f"run_{i}", "final", seed=100 + i * 100)
        for i in range(5)
    ]

    marked = _make_checkpoint_dir(rd / "run_marked", "step_20", seed=100)
    (marked / "provisional.json").write_text("{}", encoding="utf-8")
    _make_checkpoint_dir(
        rd / "run_ancestor" / "provisional" / "step_40",
        "snapshot",
        seed=200,
    )

    assert _discover_checkpoints(rd) == sorted(final_dirs)


def test_discover_checkpoints_still_rejects_extra_non_provisional_checkpoint(
    tmp_path: Path,
) -> None:
    rd = tmp_path / "results"
    for i in range(6):
        _make_checkpoint_dir(rd / f"run_{i}", "final", seed=100 + i * 100)

    with pytest.raises(OrchestratorError, match="expected exactly 5.*found 6"):
        _discover_checkpoints(rd)


def test_validate_checkpoints_rejects_wrong_identity(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = []
    for i in range(5):
        ident = "Qwen2.5-0.5B-Instruct-meta-cortex-v2" if i != 2 else "wrong"
        ckpt_dirs.append(_make_checkpoint_dir(rd, f"ckpt_{i}", identity=ident, seed=100 + i * 100))
    with pytest.raises(OrchestratorError, match="organ_identity mismatch"):
        _validate_checkpoints(ckpt_dirs, "Qwen2.5-0.5B-Instruct-meta-cortex-v2", "e" * 64, [100, 200, 300, 400, 500], TRAINING_SOURCE_COMMIT)


def test_validate_checkpoints_succeeds(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = [_make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100, theta_content=b"t" + bytes([i]))
                 for i in range(5)]
    result = _validate_checkpoints(ckpt_dirs, "Qwen2.5-0.5B-Instruct-meta-cortex-v2", "e" * 64, [100, 200, 300, 400, 500], TRAINING_SOURCE_COMMIT)
    assert result["checkpoint_count"] == 5
    assert len(result["verified_checkpoints"]) == 5


def test_validate_checkpoints_rejects_source_provenance_mismatch(
    tmp_path: Path,
) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = [
        _make_checkpoint_dir(
            rd,
            f"ckpt_{i}",
            seed=100 + i * 100,
            source_provenance="2" * 40 if i == 2 else TRAINING_SOURCE_COMMIT,
        )
        for i in range(5)
    ]
    with pytest.raises(OrchestratorError, match="source_provenance mismatch"):
        _validate_checkpoints(
            ckpt_dirs,
            "Qwen2.5-0.5B-Instruct-meta-cortex-v2",
            "e" * 64,
            EXPECTED_SEEDS,
            TRAINING_SOURCE_COMMIT,
        )


# ---------------------------------------------------------------------------
# Deterministic archive (gzip mtime=0, entries d0/...)
# ---------------------------------------------------------------------------

def test_archive_is_deterministic(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = [_make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100, theta_content=b"t" + bytes([i]))
                 for i in range(5)]
    v = _validate_checkpoints(ckpt_dirs, "Qwen2.5-0.5B-Instruct-meta-cortex-v2", "e" * 64, [100, 200, 300, 400, 500], TRAINING_SOURCE_COMMIT)["verified_checkpoints"]

    a1, h1, s1 = _build_checkpoint_archive(v, [100, 200, 300, 400, 500], tmp_path / "out1")
    a2, h2, s2 = _build_checkpoint_archive(v, [100, 200, 300, 400, 500], tmp_path / "out2")
    assert h1 == h2
    assert s1 == s2
    assert a1.read_bytes() == a2.read_bytes()


def test_archive_entries_are_d0_format(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = [_make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100, theta_content=b"t" + bytes([i]))
                 for i in range(5)]
    v = _validate_checkpoints(ckpt_dirs, "Qwen2.5-0.5B-Instruct-meta-cortex-v2", "e" * 64, [100, 200, 300, 400, 500], TRAINING_SOURCE_COMMIT)["verified_checkpoints"]

    a, _h, _s = _build_checkpoint_archive(v, [100, 200, 300, 400, 500], tmp_path / "out")

    with gzip.GzipFile(fileobj=io.BytesIO(a.read_bytes()), mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tf:
            names = sorted(tf.getnames())
    expected = sorted(f"d{i}/{f}" for i in range(5) for f in ("checkpoint.json", "theta.npz"))
    assert names == expected


def test_archive_manifest_structure(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    ckpt_dirs = [_make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100) for i in range(5)]
    v = _validate_checkpoints(ckpt_dirs, "Qwen2.5-0.5B-Instruct-meta-cortex-v2", "e" * 64, [100, 200, 300, 400, 500], TRAINING_SOURCE_COMMIT)["verified_checkpoints"]
    a, h, s = _build_checkpoint_archive(v, [100, 200, 300, 400, 500], tmp_path / "out")
    m = _build_archive_manifest(v, a, h, s)
    assert m["schema"] == "oczy/r20-int8-dev-checkpoint-archive/v1"
    assert m["checkpoint_count"] == 5
    assert m["archive_sha256"] == h
    assert m["archive_size_bytes"] == s


def test_unsorted_registered_seed_order_matches_archive_and_calibration(
    tmp_path: Path,
) -> None:
    registered_seeds = [500, 100, 400, 200, 300]
    rd = tmp_path / "results"
    checkpoint_dirs = [
        _make_checkpoint_dir(rd, f"ckpt_{seed}", seed=seed)
        for seed in registered_seeds
    ]
    verified = _validate_checkpoints(
        checkpoint_dirs,
        "Qwen2.5-0.5B-Instruct-meta-cortex-v2",
        "e" * 64,
        registered_seeds,
        TRAINING_SOURCE_COMMIT,
    )["verified_checkpoints"]
    archive, _archive_hash, _archive_size = _build_checkpoint_archive(
        verified, registered_seeds, tmp_path / "out"
    )

    archived_seeds = []
    with gzip.GzipFile(fileobj=io.BytesIO(archive.read_bytes()), mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tf:
            for index in range(5):
                checkpoint_file = tf.extractfile(f"d{index}/checkpoint.json")
                assert checkpoint_file is not None
                checkpoint = json.loads(checkpoint_file.read().decode("utf-8"))
                archived_seeds.append(checkpoint["outer_config"]["seed"])

    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=1,
        owner="kino",
        kernel_slug_prefix=cal_config["kernel_slug_prefix"],
        cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=registered_seeds,
        output_dir=tmp_path,
    )

    assert archived_seeds == registered_seeds
    for index, seed in enumerate(registered_seeds):
        checkpoint_arg = jobs[index]["arguments"].index("--checkpoint") + 1
        assert jobs[index]["checkpoint_seed"] == seed
        assert jobs[index]["arguments"][checkpoint_arg].endswith(f"/d{index}")


# ---------------------------------------------------------------------------
# Calibration jobs — real module/subcommand, owner-qualified ids
# ---------------------------------------------------------------------------

def test_calibration_jobs_use_real_module(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=10, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    # Every job uses the real module and collect-calibration-shard subcommand
    for job in jobs:
        assert job["module"] == _CAL_MODULE
        assert job["arguments"][0] == _CAL_SUBCOMMAND


def test_calibration_jobs_owner_qualified_ids(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=10, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    for job in jobs:
        assert "/" in job["kernel_id"], f"kernel_id {job['kernel_id']} missing owner"
        assert job["kernel_id"].startswith("kino/")


def test_calibration_jobs_contain_real_args(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=10, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    j0 = jobs[0]
    args = j0["arguments"]
    assert "--calibration-view" in args
    assert "--checkpoint" in args
    assert "--model-id" in args
    assert "--organ-hash" in args
    assert "--dev-seed-index" in args
    assert "--eval-seed-indices" in args
    assert "--task-start" in args
    assert "--task-end" in args
    assert "--output" in args
    assert "/tmp/oczy-offline-inputs/checkpoints/d" in args[args.index("--checkpoint") + 1]
    assert "/kaggle/working/shard-d" in args[args.index("--output") + 1]


def test_90_jobs_for_90_tasks(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=90, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    assert len(jobs) == 90


def test_five_wide_task_ranges(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=90, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    for job in jobs:
        diff = job["task_range_end"] - job["task_range_start"]
        assert diff == 5


def test_calibration_count_formula(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    for n in (1, 5, 6, 10, 42, 85, 90, 100):
        jobs = _build_calibration_jobs(
            task_count=n, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"], cal_config=cal_config,
            checkpoint_archive_dataset="kino/ckpts",
            checkpoint_archive_filename="checkpoints.tar.gz",
            checkpoint_archive_sha256="a" * 64,
            model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
            organ_hash="e" * 64,
            archived_checkpoint_seeds=[100, 200, 300, 400, 500],
            output_dir=tmp_path,
        )
        assert len(jobs) == 5 * ((n + 4) // 5)


# ---------------------------------------------------------------------------
# Runtime manifest equality
# ---------------------------------------------------------------------------

def test_runtime_equality_rejects_mismatch(tmp_path: Path) -> None:
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema_version": "oczy/remote-parallel-state/v4",
        "jobs": {"cal-s00": {"state": "succeeded",
                              "expected_runtime_manifest_sha256": "aaa",
                              "observed_runtime_manifest_sha256": "bbb"}},
    }))
    with pytest.raises(OrchestratorError, match="runtime manifest mismatch"):
        _check_calibration_runtime_equality(sp, [{"name": "cal-s00"}])


# ---------------------------------------------------------------------------
# Meta-test guard
# ---------------------------------------------------------------------------

def test_no_meta_test_in_config_key(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorError, match="meta-test prohibited"):
        _assert_no_meta_test({"meta_test": True}, [])


def test_no_meta_test_in_job_arg(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorError, match="meta-test prohibited"):
        _assert_no_meta_test({}, [{"name": "j", "arguments": ["--meta-test"]}])




def test_instrument_archive_requires_format_tar_gz(tmp_path: Path) -> None:
    cfg = _minimal_config()
    cfg["calibration"]["instrument_archive"]["format"] = "zip"
    with pytest.raises(OrchestratorError, match="format must be 'tar.gz'"):
        validate_config(cfg, tmp_path)


def test_instrument_archive_requires_destination_instrument(tmp_path: Path) -> None:
    cfg = _minimal_config()
    del cfg["calibration"]["instrument_archive"]["destination"]
    with pytest.raises(OrchestratorError, match="destination must be 'instrument'"):
        validate_config(cfg, tmp_path)


def test_calibration_jobs_store_model_source(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=10, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"],
        cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    for job in jobs:
        assert job.get("model_id") == "Qwen/Qwen2.5-0.5B-Instruct", (
            f"job {job['name']} missing model_id"
        )
        assert job.get("model_source") == "qwen-lm/qwen2.5/transformers/0.5b-instruct/1", (
            f"job {job['name']} has wrong model_source: {job.get('model_source')}"
        )


def test_kernel_slug_prefix_in_ids(tmp_path: Path) -> None:
    cal_config = _fake_cal_config()
    jobs = _build_calibration_jobs(
        task_count=10, owner="kino", kernel_slug_prefix=cal_config["kernel_slug_prefix"],
        cal_config=cal_config,
        checkpoint_archive_dataset="kino/ckpts",
        checkpoint_archive_filename="checkpoints.tar.gz",
        checkpoint_archive_sha256="a" * 64,
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_source="qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        organ_hash="e" * 64,
        archived_checkpoint_seeds=[100, 200, 300, 400, 500],
        output_dir=tmp_path,
    )
    for job in jobs:
        assert "oczy-r20-int8-cal" in job["kernel_id"], (
            f"kernel_id {job['kernel_id']} missing slug prefix"
        )
        assert job["kernel_id"].startswith("kino/")
# ---------------------------------------------------------------------------
# Publication guard
# ---------------------------------------------------------------------------
def test_publish_only_once(tmp_path: Path) -> None:
    pub = {"dataset_slug": "x/y", "title": "t", "message": "m", "timeout_seconds": 10}
    state: dict[str, Any] = {}
    state_path = tmp_path / "pub_state.json"
    manifest: dict[str, Any] = {"checkpoints": []}
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"fake-archive")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ready", stderr="")

    _publish_checkpoint_dataset(pub, archive, manifest, state, state_path, _run_subprocess=fake_run)
    # create-if-absent: status check + create + poll
    assert len(calls) >= 2  # at least status + create + poll status
    assert state.get("_checkpoint_dataset_published")

    # Second call: skipped entirely
    calls.clear()
    _publish_checkpoint_dataset(pub, archive, manifest, state, state_path, _run_subprocess=fake_run)
    assert len(calls) == 0


# ---------------------------------------------------------------------------
# Scheduler argv includes submit/timeout args
# ---------------------------------------------------------------------------

def test_training_scheduler_argv_has_submit_interval(tmp_path: Path) -> None:
    config = _minimal_config(kaggle_submit_interval=60, push_timeout_seconds=43200, job_timeout_seconds=43200)
    bp = tmp_path / "training_batch.json"
    sp = tmp_path / "training_state.json"
    _make_training_batch(bp)
    _make_training_state(sp)
    _make_calibration_view(tmp_path)

    # Make results dir with 5 checkpoints
    rd = tmp_path / "results"
    for i in range(5):
        _make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100, theta_content=b"t" + bytes([i]))

    config_path = tmp_path / "config.json"
    full_cfg = {**config,
                "training_batch_path": str(bp), "training_state_path": str(sp),
                "jobs_dir": str(tmp_path / "jobs"), "results_dir": str(rd),
                "archive_dir": str(tmp_path / "archive")}
    config_path.write_text(json.dumps(full_cfg), encoding="utf-8")

    scheduler_calls = []

    def fake_scheduler(argv, state_file):
        scheduler_calls.append(argv)
        sf = Path(state_file)
        is_training = "training" in sf.name or "train" in sf.name
        if is_training:
            _make_training_state(sf)
        else:
            sf.write_text(json.dumps({
                "schema_version": "oczy/remote-parallel-state/v4",
                "jobs": {f"cal-s{i:02d}": {"state": "succeeded",
                         "expected_runtime_manifest_sha256": "a" * 64,
                         "observed_runtime_manifest_sha256": "a" * 64,
                         "runtime_manifest_verified": True}
                         for i in range(90)},
            }), encoding="utf-8")
        return {"exit_code": 0}

    def fake_pub_subprocess(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="ready", stderr="")

    run_orchestrator(config_path, tmp_path / "orchestrator_state.json",
                     _run_scheduler=fake_scheduler, _run_subprocess=fake_pub_subprocess)

    # First scheduler call is training
    train_argv = scheduler_calls[0]
    assert "--kaggle-submit-interval" in train_argv
    assert "60" in train_argv
    if "--push-timeout" in train_argv:
        assert "43200" in train_argv
    if "--job-timeout" in train_argv:
        assert "43200" in train_argv


# ---------------------------------------------------------------------------
# End-to-end: generated calibration run.py exercises real module/args
# ---------------------------------------------------------------------------

def test_e2e_exercises_real_module_args(tmp_path: Path) -> None:
    """End-to-end: training succeeds, archive built, calibration batch generated
    with real collect-calibration-shard args, scheduler runs, all verified."""
    bp = tmp_path / "training_batch.json"
    sp = tmp_path / "training_state.json"
    _make_training_batch(bp)
    _make_training_state(sp)

    rd = tmp_path / "results"
    for i in range(5):
        _make_checkpoint_dir(rd, f"ckpt_{i}", seed=100 + i * 100, theta_content=b"t" + bytes([i]))

    _make_calibration_view(tmp_path)

    config_path = tmp_path / "config.json"
    cfg = _minimal_config(
        training_batch_path=str(bp), training_state_path=str(sp),
        jobs_dir=str(tmp_path / "jobs"), results_dir=str(rd),
        archive_dir=str(tmp_path / "archive"),
        calibration={**_fake_cal_config(),
                     "calibration_view_root": str(tmp_path / "cal_view"),
                     "batch_path": str(tmp_path / "cal_batch.json"),
                     "state_path": str(tmp_path / "cal_state.json")},
        kaggle_submit_interval=60, push_timeout_seconds=43200,
    )
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    state_path = tmp_path / "state.json"
    scheduler_calls = []

    def fake_scheduler(argv, state_file):
        scheduler_calls.append(dict(argv=argv, state_file=state_file))
        # Determine if training or calibration
        if any("training" in a for a in argv if isinstance(a, str)):
            job_names = [f"job{i}" for i in range(5)]
            runtime_hash = cfg["training_contract"]["runtime_manifest_sha256"]
        else:
            job_names = [f"cal-s{i:02d}" for i in range(90)]
            runtime_hash = "a" * 64
        Path(state_file).write_text(json.dumps({
            "schema_version": "oczy/remote-parallel-state/v4",
            "jobs": {n: {"state": "succeeded",
                          "expected_runtime_manifest_sha256": runtime_hash,
                          "observed_runtime_manifest_sha256": runtime_hash,
                          "runtime_manifest_verified": True}
                     for n in job_names},
        }), encoding="utf-8")
        return {"exit_code": 0}

    def fake_pub(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="ready", stderr="")

    result = run_orchestrator(config_path, state_path,
                              _run_scheduler=fake_scheduler,
                              _run_subprocess=fake_pub)

    assert result["all_complete"]
    assert len(scheduler_calls) >= 2

    # Verify calibration batch was generated
    cal_batch_path = tmp_path / "cal_batch.json"
    assert cal_batch_path.is_file()

    # Verify a generated kernel directory contains the real module in job_spec
    kernel_dir = tmp_path / "jobs" / "kino_oczy-r20-cal-s00"
    if not kernel_dir.is_dir():
        # Try alternate naming
        for d in (tmp_path / "jobs").iterdir():
            if d.is_dir():
                kernel_dir = d
                break
    if kernel_dir.is_dir():
        job_spec = json.loads((kernel_dir / "job_spec.json").read_text())
        assert job_spec["module"] == _CAL_MODULE
        args = job_spec["arguments"]
        assert _CAL_SUBCOMMAND in args
        assert "--calibration-view" in args
        assert "--checkpoint" in args
        assert "--model-id" in args
        assert args[args.index("--model-id") + 1] == "Qwen/Qwen2.5-0.5B-Instruct"

    # Archive exists
    archive = tmp_path / "archive" / "checkpoints.tar.gz"
    assert archive.is_file()
    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["training_contract"] == cfg["training_contract"]

    # Restart is idempotent
    scheduler_calls.clear()
    result2 = run_orchestrator(config_path, state_path,
                               _run_scheduler=fake_scheduler,
                               _run_subprocess=fake_pub)
    assert len(scheduler_calls) == 0
    assert result2["all_complete"]


# ---------------------------------------------------------------------------
# Restart / duplicate prevention
# ---------------------------------------------------------------------------

def test_restart_all_stages_complete_no_work(tmp_path: Path) -> None:
    config = _minimal_config()
    sp = tmp_path / "state.json"
    s = load_or_init_state(sp)
    s["campaign_id"] = config["campaign_id"]
    s["training_contract"] = config["training_contract"]
    for stage in (STAGE_TRAINING, STAGE_ARCHIVE, STAGE_CALIBRATION):
        s["stages"][stage] = {"status": "complete"}
    s["artifacts"] = {"checkpoint_archive": {"path": "x.tar.gz", "sha256": "a" * 64, "size_bytes": 1, "manifest": {}}}
    save_state(sp, s)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    scheduler_calls = []
    def fake_scheduler(argv, sf):
        scheduler_calls.append(argv)
        return {"exit_code": 0}

    result = run_orchestrator(config_path, sp, _run_scheduler=fake_scheduler)
    assert len(scheduler_calls) == 0
    assert result["all_complete"]


@pytest.mark.parametrize("state_contract", [None, {"module": "mixed"}])
def test_restart_fails_closed_without_exact_state_training_contract(
    tmp_path: Path, state_contract: dict[str, Any] | None
) -> None:
    config = _minimal_config()
    state_path = tmp_path / "state.json"
    state = load_or_init_state(state_path)
    state["campaign_id"] = config["campaign_id"]
    if state_contract is not None:
        state["training_contract"] = state_contract
    for stage in (STAGE_TRAINING, STAGE_ARCHIVE, STAGE_CALIBRATION):
        state["stages"][stage] = {"status": "complete"}
    state["artifacts"] = {
        "checkpoint_archive": {
            "path": "x.tar.gz",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "manifest": {},
        }
    }
    save_state(state_path, state)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(OrchestratorError, match="training_contract"):
        run_orchestrator(config_path, state_path)


def test_provider_failure_remains_failed(tmp_path: Path) -> None:
    config = _minimal_config()
    bp = tmp_path / "training_batch.json"
    sp = tmp_path / "training_state.json"
    _make_training_batch(bp)
    _make_training_state(sp, job_state="failed")

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({**config,
        "training_batch_path": str(bp), "training_state_path": str(sp),
        "jobs_dir": str(tmp_path / "j"), "results_dir": str(tmp_path / "r"),
        "archive_dir": str(tmp_path / "a")}), encoding="utf-8")

    def fake_scheduler(argv, sf):
        _make_training_state(Path(sf), job_state="failed")
        return {"exit_code": 0}

    with pytest.raises(OrchestratorError, match="no succeeded"):
        run_orchestrator(config_path, tmp_path / "ostate.json", _run_scheduler=fake_scheduler)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def test_dry_run_no_side_effects(tmp_path: Path) -> None:
    config = _minimal_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sp = tmp_path / "state.json"

    result = run_orchestrator(config_path, sp, dry_run=True)
    assert result["dry_run"]
    assert "Stage 1: Training" in result["rendered"]
    assert "Stage 2: Calibration" in result["rendered"]
    assert not sp.exists()
