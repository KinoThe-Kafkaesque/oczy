"""Focused contracts for reviewed pool planning and account-aware dispatch."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "runner_pool.py"
SCHEDULER_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "parallel_scheduler.py"

_pool_spec = importlib.util.spec_from_file_location("oczy_pool_aware_tests", POOL_PATH)
assert _pool_spec is not None and _pool_spec.loader is not None
pool = importlib.util.module_from_spec(_pool_spec)
sys.modules[_pool_spec.name] = pool
_pool_spec.loader.exec_module(pool)

scheduler = runpy.run_path(str(SCHEDULER_PATH))
ParallelScheduler = scheduler["ParallelScheduler"]
BatchValidationError = scheduler["BatchValidationError"]
SchedulerAlreadyRunningError = scheduler["SchedulerAlreadyRunningError"]
SchedulerStateLock = scheduler["SchedulerStateLock"]
STATE_SCHEMA_VERSION = scheduler["STATE_SCHEMA_VERSION"]
BATCH_SCHEMA_VERSION = scheduler["BATCH_SCHEMA_VERSION"]
compute_manifest_sha256 = scheduler["compute_manifest_sha256"]
apply_dispatch_plan = scheduler["apply_dispatch_plan"]
build_account_clients = scheduler["build_account_clients"]


def _valid_runtime_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "oczy/runtime-manifest/v2",
        "python_version": "3.12.0",
        "packages": {
            "torchao": "0.17.0",
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        },
        "model": {
            "logical_model_id": None,
            "quantization": None,
            "resolved_model_convention": "none",
            "artifact_files": [],
            "model_weights_sha256": None,
            "model_config_sha256": None,
            "tokenizer_sha256": None,
            "chat_template_sha256": None,
        },
        "greedy_generation": None,
    }
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    return manifest


def _execution_report() -> dict[str, Any]:
    manifest = _valid_runtime_manifest()
    return {
        "schema_version": "oczy/execution-report/v2",
        "status": "complete",
        "exit_code": 0,
        "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
        "observed_runtime_manifest": manifest,
    }


class FakeKaggleClient:
    def __init__(self) -> None:
        self.pushed: list[str] = []
        self.polled: list[str] = []
        self.outputs: list[tuple[str, str]] = []

    def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
        self.pushed.append(kernel_dir)
        metadata = json.loads(
            (Path(kernel_dir) / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        return str(metadata["id"])

    def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
        self.polled.append(kernel_id)
        return "complete"

    def output(
        self, kernel_id: str, output_dir: str, *, timeout: float | None = None
    ) -> None:
        self.outputs.append((kernel_id, output_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "execution_report.json").write_text(
            json.dumps(_execution_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_pool(tmp_path: Path) -> Any:
    config_path = tmp_path / "pool.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": pool.POOL_CONFIG_SCHEMA,
                "accounts": [
                    {
                        "id": "kaggle-a",
                        "provider": "kaggle",
                        "capacity": 1,
                        "config_dir": "accounts/a",
                    },
                    {
                        "id": "kaggle-b",
                        "provider": "kaggle",
                        "capacity": 1,
                        "config_dir": "accounts/b",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return pool.load_pool_config(config_path)


def _write_batch(tmp_path: Path) -> Path:
    jobs: list[dict[str, Any]] = []
    for index in range(2):
        kernel_dir = tmp_path / f"kernel-{index}"
        kernel_dir.mkdir()
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"owner/pool-job-{index}",
                    "title": f"Pool Job {index}",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_tpu": False,
                    "enable_internet": False,
                    "machine_shape": "",
                }
            ),
            encoding="utf-8",
        )
        runtime_manifest = _valid_runtime_manifest()
        (kernel_dir / "job_spec.json").write_text(
            json.dumps({"profile": "cpu", "runtime_manifest": runtime_manifest}), encoding="utf-8"
        )
        jobs.append(
            {
                "name": f"job-{index}",
                "provider": "kaggle",
                "kernel_dir": str(kernel_dir),
                "output_dir": str(tmp_path / f"output-{index}"),
                "runtime_manifest": runtime_manifest,
            }
        )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {"schema_version": BATCH_SCHEMA_VERSION, "jobs": jobs}
        ),
        encoding="utf-8",
    )
    return batch


def _plan(config: Any, batch: Path) -> dict[str, Any]:
    snapshot = pool.PoolSnapshot(
        accounts=[
            pool.AccountSnapshot("kaggle-a", "kaggle", pool.ACCOUNT_OK, 1),
            pool.AccountSnapshot("kaggle-b", "kaggle", pool.ACCOUNT_OK, 1),
        ],
        jobs=[],
    )
    return pool.create_dispatch_plan(config, snapshot, batch)


def test_pool_dispatch_routes_each_job_and_persists_account(tmp_path: Path) -> None:
    config = _write_pool(tmp_path)
    batch = _write_batch(tmp_path)
    plan = _plan(config, batch)
    state = tmp_path / "state.json"
    leases = pool.SlotLeaseStore(tmp_path / "leases.json", ttl=3600)
    account_a = FakeKaggleClient()
    account_b = FakeKaggleClient()
    fallback = FakeKaggleClient()
    runtime = ParallelScheduler(
        fallback,
        account_clients={"kaggle-a": account_a, "kaggle-b": account_b},
        account_capacities={"kaggle-a": 1, "kaggle-b": 1},
        lease_store=leases,
        lease_owner_id="queue-owner",
        sleeper=lambda _seconds: None,
    )

    summary = runtime.run(
        batch,
        state,
        poll_interval=0,
        pool_config=config,
        dispatch_plan=plan,
    )

    assert summary["all_succeeded"] is True
    assert len(account_a.pushed) == 1
    assert len(account_b.pushed) == 1
    assert fallback.pushed == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == STATE_SCHEMA_VERSION
    assert persisted["jobs"]["job-0"]["account_id"] == "kaggle-a"
    assert persisted["jobs"]["job-1"]["account_id"] == "kaggle-b"
    assert leases.snapshot()["leases"] == []


def test_dispatch_plan_rejects_changed_batch_hash(tmp_path: Path) -> None:
    config = _write_pool(tmp_path)
    batch = _write_batch(tmp_path)
    plan = _plan(config, batch)
    payload = json.loads(batch.read_text(encoding="utf-8"))
    payload["jobs"][0]["output_dir"] = str(tmp_path / "changed-output")
    batch.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BatchValidationError, match="batch SHA-256"):
        apply_dispatch_plan(scheduler["load_batch"](batch), batch, config, plan)


def test_state_lock_rejects_second_scheduler_owner(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    with SchedulerStateLock(state):
        with pytest.raises(SchedulerAlreadyRunningError, match="another scheduler"):
            with SchedulerStateLock(state):
                pass


def test_real_account_clients_are_credential_scoped(tmp_path: Path) -> None:
    config_path = tmp_path / "pool.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": pool.POOL_CONFIG_SCHEMA,
                "accounts": [
                    {
                        "id": "kaggle-a",
                        "provider": "kaggle",
                        "capacity": 1,
                        "config_dir": "accounts/kaggle-a",
                    },
                    {
                        "id": "colab-a",
                        "provider": "colab",
                        "capacity": 1,
                        "home_dir": "accounts/colab-a/home",
                        "session_config": "accounts/colab-a/sessions.json",
                        "client_oauth_config": "accounts/colab-a/oauth.json",
                        "auth": "adc",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = pool.load_pool_config(config_path)

    clients, capacities = build_account_clients(config)

    assert capacities == {"kaggle-a": 1, "colab-a": 1}
    assert clients["kaggle-a"].env_overrides == {
        "KAGGLE_CONFIG_DIR": str((tmp_path / "accounts/kaggle-a").resolve())
    }
    assert clients["colab-a"]._argv_prefix == [
        "colab",
        "--client-oauth-config",
        str((tmp_path / "accounts/colab-a/oauth.json").resolve()),
        "--config",
        str((tmp_path / "accounts/colab-a/sessions.json").resolve()),
        "--auth",
        "adc",
    ]
