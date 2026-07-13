"""Tests for the unified remote runner account/job inventory."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = REPO_ROOT / "infrastructure" / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))
from runtime_manifest import compute_manifest_sha256  # type: ignore[import-not-found]

POOL_PATH = KAGGLE_DIR / "runner_pool.py"
_spec = importlib.util.spec_from_file_location("oczy_runner_pool_tests", POOL_PATH)
assert _spec is not None and _spec.loader is not None
_pool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pool
_spec.loader.exec_module(_pool)

ACCOUNT_DISABLED = _pool.ACCOUNT_DISABLED
ACCOUNT_ERROR = _pool.ACCOUNT_ERROR
ACCOUNT_OK = _pool.ACCOUNT_OK
POOL_CONFIG_SCHEMA = _pool.POOL_CONFIG_SCHEMA
DISPATCH_PLAN_SCHEMA = _pool.DISPATCH_PLAN_SCHEMA
PoolConfigError = _pool.PoolConfigError
SlotLeaseStore = _pool.SlotLeaseStore
create_dispatch_plan = _pool.create_dispatch_plan
inspect_pool = _pool.inspect_pool
load_dispatch_plan = _pool.load_dispatch_plan
load_pool_config = _pool.load_pool_config
render_table = _pool.render_table
write_dispatch_plan = _pool.write_dispatch_plan


def _write_config(
    tmp_path: Path,
    *,
    accounts: list[dict] | None = None,
    state_files: list[str] | None = None,
) -> Path:
    if accounts is None:
        accounts = [
            {
                "id": "kaggle-a",
                "provider": "kaggle",
                "capacity": 10,
                "config_dir": "accounts/kaggle-a",
            },
            {
                "id": "colab-a",
                "provider": "colab",
                "capacity": 2,
                "home_dir": "accounts/colab-a/home",
                "session_config": "accounts/colab-a/sessions.json",
                "client_oauth_config": "accounts/colab-a/oauth.json",
                "auth": "oauth2",
            },
        ]
    path = tmp_path / "pool.json"
    for account in accounts:
        if account.get("provider") == "colab" and account.get("auth", "oauth2") == "oauth2":
            home = Path(account.get("home_dir", "~"))
            if not home.is_absolute():
                home = tmp_path / home
            token = home / ".config/colab-cli/token.json"
            token.parent.mkdir(parents=True, exist_ok=True)
            token.write_text("{}", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": POOL_CONFIG_SCHEMA,
                "accounts": accounts,
                "state_files": state_files or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _completed(
    argv: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


class FakePoolRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str], float | None]] = []

    def __call__(
        self, argv: list[str], env: dict[str, str], timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(env), timeout))
        if argv[:3] == ["kaggle", "kernels", "list"]:
            payload = [
                {
                    "ref": "owner/job-a",
                    "title": "Job A",
                    "author": "Owner",
                    "lastRunTime": "2026-07-11T20:00:00",
                },
                {
                    "ref": "owner/job-b",
                    "title": "Job B",
                    "author": "Owner",
                    "lastRunTime": "2026-07-11T19:00:00",
                },
            ]
            return _completed(argv, stdout=json.dumps(payload))
        if argv[:3] == ["kaggle", "kernels", "status"]:
            state = "RUNNING" if argv[-1] == "owner/job-a" else "COMPLETE"
            return _completed(
                argv,
                stdout=f'{argv[-1]} has status "KernelWorkerStatus.{state}"\n',
            )
        if argv[0] == "colab" and argv[-1] == "sessions":
            return _completed(
                argv, stdout="[session-a] https://us-west1 | running\n"
            )
        raise AssertionError(f"unexpected command: {argv!r}")


def test_config_resolves_account_scoped_paths(tmp_path: Path) -> None:
    config = load_pool_config(_write_config(tmp_path))
    kaggle, colab = config.accounts
    assert kaggle.config_dir == str((tmp_path / "accounts/kaggle-a").resolve())
    assert colab.home_dir == str((tmp_path / "accounts/colab-a/home").resolve())
    assert colab.session_config == str(
        (tmp_path / "accounts/colab-a/sessions.json").resolve()
    )
    assert colab.client_oauth_config == str(
        (tmp_path / "accounts/colab-a/oauth.json").resolve()
    )


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda payload: payload.update(schema_version="wrong"), "schema_version"),
        (lambda payload: payload.update(accounts=[]), "non-empty list"),
        (
            lambda payload: payload["accounts"].append(dict(payload["accounts"][0])),
            "duplicate account id",
        ),
        (
            lambda payload: payload["accounts"][0].update(token="secret"),
            "unknown fields",
        ),
    ],
)
def test_config_rejects_invalid_or_secret_fields(
    tmp_path: Path, mutator, expected: str
) -> None:
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PoolConfigError, match=expected):
        load_pool_config(path)


def test_public_config_never_reads_credential_contents(tmp_path: Path) -> None:
    config_dir = tmp_path / "account"
    config_dir.mkdir()
    marker = "SUPER-SECRET-CREDENTIAL-MARKER"
    (config_dir / "credentials.json").write_text(marker, encoding="utf-8")
    path = _write_config(
        tmp_path,
        accounts=[
            {
                "id": "kg",
                "provider": "kaggle",
                "config_dir": str(config_dir),
            }
        ],
    )
    public = json.dumps(load_pool_config(path).public_dict())
    assert str(config_dir) in public
    assert marker not in public


def test_pool_aggregates_accounts_and_merges_scheduler_state(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": "oczy/remote-parallel-state/v4",
                "updated_at": 1783800000.0,
                "jobs": {
                    "local-job-a": {
                        "provider": "kaggle",
                        "remote_id": "owner/job-a",
                        "state": "running",
                        "error": None,
                    },
                    "local-failed": {
                        "provider": "kaggle",
                        "remote_id": "owner/failed-job",
                        "state": "failed",
                        "error": "push failed",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_pool_config(_write_config(tmp_path, state_files=[str(state)]))
    runner = FakePoolRunner()
    snapshot = inspect_pool(config, runner=runner, limit=20, timeout=7.0)

    assert [account.status for account in snapshot.accounts] == [ACCOUNT_OK, ACCOUNT_OK]
    assert snapshot.accounts[0].to_dict()["active_jobs"] == 1
    assert snapshot.accounts[0].to_dict()["available_capacity"] == 9
    assert snapshot.accounts[1].to_dict()["active_jobs"] == 1
    assert snapshot.to_dict()["summary"]["known_capacity"] == 12
    assert snapshot.to_dict()["summary"]["active_slots"] == 2
    assert snapshot.to_dict()["summary"]["available_capacity"] == 10
    assert len(snapshot.jobs) == 4
    by_id = {(job.provider, job.job_id): job for job in snapshot.jobs}

    job_a = by_id[("kaggle", "owner/job-a")]
    assert job_a.account_id == "kaggle-a"
    assert job_a.name == "local-job-a"
    assert job_a.state == "running"
    assert job_a.remote_state == "running"
    assert job_a.scheduler_state == "running"
    assert job_a.sources == ("remote", "scheduler")
    assert job_a.state_paths == (str(state.resolve()),)

    assert by_id[("kaggle", "owner/job-b")].state == "succeeded"
    assert by_id[("colab", "session-a")].state == "running"
    failed = by_id[("kaggle", "owner/failed-job")]
    assert failed.account_id == "unassigned"
    assert failed.state == "failed"
    assert failed.error == "push failed"

    kaggle_calls = [call for call in runner.calls if call[0][0] == "kaggle"]
    assert kaggle_calls
    assert all(
        call[1]["KAGGLE_CONFIG_DIR"]
        == str((tmp_path / "accounts/kaggle-a").resolve())
        for call in kaggle_calls
    )
    colab_call = next(call for call in runner.calls if call[0][0] == "colab")
    assert colab_call[1]["HOME"] == str(
        (tmp_path / "accounts/colab-a/home").resolve()
    )
    assert colab_call[0][-1] == "sessions"
    assert "--config" in colab_call[0]
    assert "--client-oauth-config" in colab_call[0]


def test_active_only_filters_terminal_remote_and_scheduler_jobs(tmp_path: Path) -> None:
    config = load_pool_config(_write_config(tmp_path))
    snapshot = inspect_pool(
        config,
        runner=FakePoolRunner(),
        active_only=True,
    )
    assert {(job.provider, job.job_id) for job in snapshot.jobs} == {
        ("kaggle", "owner/job-a"),
        ("colab", "session-a"),
    }


def test_disabled_account_is_visible_but_not_called(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        accounts=[
            {
                "id": "kg-disabled",
                "provider": "kaggle",
                "enabled": False,
                "config_dir": "account",
            }
        ],
    )
    runner = FakePoolRunner()
    snapshot = inspect_pool(load_pool_config(path), runner=runner)
    assert snapshot.accounts[0].status == ACCOUNT_DISABLED
    assert runner.calls == []


def test_one_account_failure_does_not_hide_other_accounts(tmp_path: Path) -> None:
    config = load_pool_config(_write_config(tmp_path))

    def runner(argv, env, timeout):
        if argv[0] == "kaggle":
            return _completed(argv, stderr="authentication failed", returncode=1)
        return _completed(argv, stdout="[still-live] endpoint | running\n")

    snapshot = inspect_pool(config, runner=runner)
    assert snapshot.accounts[0].status == ACCOUNT_ERROR
    assert snapshot.accounts[0].error == "kaggle command failed (exit 1): authentication failed"
    assert snapshot.accounts[1].status == ACCOUNT_OK
    assert [(job.provider, job.job_id) for job in snapshot.jobs] == [
        ("colab", "still-live")
    ]


def test_missing_colab_token_is_noninteractive_account_error(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        accounts=[
            {
                "id": "colab-missing-token",
                "provider": "colab",
                "home_dir": "missing-home",
                "session_config": "missing-home/sessions.json",
                "client_oauth_config": "missing-home/oauth.json",
                "auth": "adc",
            }
        ],
    )
    # Switch to oauth2 after the helper has deliberately skipped token setup.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accounts"][0]["auth"] = "oauth2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    runner = FakePoolRunner()
    snapshot = inspect_pool(load_pool_config(path), runner=runner)
    assert snapshot.accounts[0].status == ACCOUNT_ERROR
    assert "OAuth token not found" in (snapshot.accounts[0].error or "")
    assert runner.calls == []


def test_missing_state_is_reported_without_hiding_remote_jobs(tmp_path: Path) -> None:
    missing = tmp_path / "missing-state.json"
    config = load_pool_config(
        _write_config(tmp_path, state_files=[str(missing)])
    )
    snapshot = inspect_pool(config, runner=FakePoolRunner())
    assert len(snapshot.jobs) == 3
    assert snapshot.state_errors == [
        {"path": str(missing.resolve()), "error": f"[Errno 2] No such file or directory: '{missing.resolve()}'"}
    ]


def test_table_contains_account_state_and_sources(tmp_path: Path) -> None:
    snapshot = inspect_pool(
        load_pool_config(_write_config(tmp_path)), runner=FakePoolRunner()
    )
    rendered = render_table(snapshot)
    assert "account kaggle-a (kaggle) status=ok jobs=2 active=1" in rendered
    assert "ACCOUNT" in rendered
    assert "owner/job-a" in rendered
    assert "running" in rendered


def test_limit_validation_happens_before_provider_calls(tmp_path: Path) -> None:
    runner = FakePoolRunner()
    with pytest.raises(ValueError, match="limit must be 1..200"):
        inspect_pool(load_pool_config(_write_config(tmp_path)), runner=runner, limit=0)
    assert runner.calls == []


def _valid_runtime_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "oczy/runtime-manifest/v1",
        "python_version": "3.12.0",
        "packages": {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        },
        "model": {
            "logical_model_id": None,
            "resolved_model_convention": "none",
            "artifact_files": [],
            "model_weights_sha256": None,
            "model_config_sha256": None,
            "tokenizer_sha256": None,
            "chat_template_sha256": None,
        },
        "greedy_generation": None,
    }
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)  # type: ignore[arg-type]
    return manifest


def _write_kaggle_batch(tmp_path: Path, names: list[str]) -> Path:
    jobs: list[dict[str, object]] = []
    for index, name in enumerate(names):
        kernel_id = f"owner/planned-job-{index}"
        kernel_dir = tmp_path / f"kernel-{index}"
        kernel_dir.mkdir()
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_id,
                    "title": f"Planned Job {index}",
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
                "name": name,
                "provider": "kaggle",
                "kernel_dir": str(kernel_dir),
                "output_dir": str(tmp_path / f"output-{index}"),
                "runtime_manifest": runtime_manifest,
            }
        )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {"schema_version": "oczy/remote-parallel-batch/v3", "jobs": jobs}
        ),
        encoding="utf-8",
    )
    return batch


def test_dispatch_plan_balances_jobs_and_is_hash_bound(tmp_path: Path) -> None:
    config = load_pool_config(
        _write_config(
            tmp_path,
            accounts=[
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "capacity": 3,
                    "config_dir": "a",
                },
                {
                    "id": "kaggle-b",
                    "provider": "kaggle",
                    "capacity": 3,
                    "config_dir": "b",
                },
            ],
        )
    )
    snapshot = inspect_pool(config, runner=FakePoolRunner())
    batch = _write_kaggle_batch(tmp_path, ["fresh-a", "fresh-b"])

    plan = create_dispatch_plan(config, snapshot, batch)

    assert plan["schema_version"] == DISPATCH_PLAN_SCHEMA
    assert plan["all_assigned"] is True
    assert plan["ready_for_dispatch"] is True
    assert [item["account_id"] for item in plan["assignments"]] == [
        "kaggle-a",
        "kaggle-b",
    ]
    assert len(plan["batch_sha256"]) == 64
    assert len(plan["pool_config_sha256"]) == 64

    output = tmp_path / "plan.json"
    write_dispatch_plan(output, plan)
    assert load_dispatch_plan(output) == plan


def test_dispatch_plan_hash_changes_when_only_runtime_manifest_changes(
    tmp_path: Path,
) -> None:
    """Reviewed pool plans bind the full batch, including runtime identity."""
    config = load_pool_config(_write_config(tmp_path))
    snapshot = inspect_pool(config, runner=FakePoolRunner())
    batch = _write_kaggle_batch(tmp_path, ["fresh-a"])
    original_plan = create_dispatch_plan(config, snapshot, batch)

    raw = json.loads(batch.read_text(encoding="utf-8"))
    changed_manifest = _valid_runtime_manifest()
    changed_manifest["python_version"] = "3.13.0"
    changed_manifest["manifest_sha256"] = compute_manifest_sha256(changed_manifest)  # type: ignore[arg-type]
    raw["jobs"][0]["runtime_manifest"] = changed_manifest
    batch.write_text(json.dumps(raw), encoding="utf-8")
    kernel_spec = tmp_path / "kernel-0" / "job_spec.json"
    spec = json.loads(kernel_spec.read_text(encoding="utf-8"))
    spec["runtime_manifest"] = changed_manifest
    kernel_spec.write_text(json.dumps(spec), encoding="utf-8")

    changed_plan = create_dispatch_plan(config, snapshot, batch)

    assert changed_plan["batch_sha256"] != original_plan["batch_sha256"]


def test_dispatch_plan_refuses_unhealthy_or_capacity_unknown_accounts(
    tmp_path: Path,
) -> None:
    config = load_pool_config(
        _write_config(
            tmp_path,
            accounts=[
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "config_dir": "a",
                }
            ],
        )
    )
    snapshot = inspect_pool(config, runner=FakePoolRunner())
    plan = create_dispatch_plan(
        config, snapshot, _write_kaggle_batch(tmp_path, ["fresh"])
    )
    assert plan["all_assigned"] is False
    assert "configured capacity" in plan["errors"][0]["error"]


def test_slot_leases_enforce_capacity_ownership_and_expiry(tmp_path: Path) -> None:
    now = [100.0]
    store = SlotLeaseStore(
        tmp_path / "leases.json", ttl=10.0, clock=lambda: now[0]
    )

    assert store.acquire(
        account_id="kaggle-a", job_name="job-a", owner_id="queue-a", capacity=1
    )
    assert not store.acquire(
        account_id="kaggle-a", job_name="job-b", owner_id="queue-b", capacity=1
    )
    assert not store.acquire(
        account_id="kaggle-a", job_name="job-a", owner_id="queue-b", capacity=1
    )
    assert store.renew(
        account_id="kaggle-a", job_name="job-a", owner_id="queue-a"
    )

    now[0] = 111.0
    assert store.acquire(
        account_id="kaggle-a", job_name="job-b", owner_id="queue-b", capacity=1
    )
    assert store.release(
        account_id="kaggle-a", job_name="job-b", owner_id="queue-b"
    )
    assert store.snapshot()["leases"] == []
