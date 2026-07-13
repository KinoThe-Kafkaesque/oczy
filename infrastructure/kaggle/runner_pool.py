#!/usr/bin/env python3
"""Unified read-only inventory for remote runner accounts and scheduler jobs.

The experiment scheduler historically talks to whichever Kaggle and Colab
accounts are active in the current process environment.  This module adds a
versioned account registry and a normalized inventory without changing job
submission or experiment authority.

It intentionally reads credential *locations*, never credential contents.
Kaggle accounts are isolated with ``KAGGLE_CONFIG_DIR``.  Colab CLI 0.6.0
stores its OAuth token below ``HOME`` and its session registry at ``--config``,
so each Colab account receives an explicit HOME and session-state path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POOL_CONFIG_SCHEMA = "oczy/remote-runner-pool/v1"
POOL_SNAPSHOT_SCHEMA = "oczy/remote-runner-pool-snapshot/v1"
DISPATCH_PLAN_SCHEMA = "oczy/remote-runner-dispatch-plan/v1"
LEASE_STATE_SCHEMA = "oczy/remote-runner-leases/v1"

PROVIDER_KAGGLE = "kaggle"
PROVIDER_COLAB = "colab"
VALID_PROVIDERS = frozenset({PROVIDER_KAGGLE, PROVIDER_COLAB})

ACCOUNT_OK = "ok"
ACCOUNT_DEGRADED = "degraded"
ACCOUNT_ERROR = "error"
ACCOUNT_DISABLED = "disabled"

STATE_PENDING = "pending"
STATE_SUBMITTING = "submitting"
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_COLLECTING = "collecting"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_UNKNOWN = "unknown"

ACTIVE_STATES = frozenset(
    {STATE_PENDING, STATE_SUBMITTING, STATE_QUEUED, STATE_RUNNING, STATE_COLLECTING}
)
TERMINAL_STATES = frozenset({STATE_SUCCEEDED, STATE_FAILED})

DEFAULT_CONFIG_PATH = Path("~/.config/oczy/runner-pool.json").expanduser()
DEFAULT_LIMIT = 20
MAX_KAGGLE_LIMIT = 200
DEFAULT_TIMEOUT = 30.0
DEFAULT_LEASE_TTL = 8 * 60 * 60.0

_COMMON_ACCOUNT_KEYS = frozenset(
    {"id", "provider", "enabled", "capacity", "description"}
)
_KAGGLE_ACCOUNT_KEYS = _COMMON_ACCOUNT_KEYS | {"config_dir"}
_COLAB_ACCOUNT_KEYS = _COMMON_ACCOUNT_KEYS | {
    "home_dir",
    "session_config",
    "client_oauth_config",
    "auth",
}


class PoolConfigError(ValueError):
    """Raised when a runner-pool configuration is invalid."""


class ProviderCommandError(RuntimeError):
    """Raised when an account-scoped provider CLI command fails."""


class LeaseStoreError(RuntimeError):
    """Raised when durable account-slot lease state is invalid."""


CommandRunner = Callable[
    [list[str], dict[str, str], float | None], subprocess.CompletedProcess[str]
]


def _default_command_runner(
    argv: list[str], env: dict[str, str], timeout: float | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def _short_error(text: str, *, limit: int = 1000) -> str:
    """Return a bounded single-line error without inspecting credential files."""
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _run_checked(
    runner: CommandRunner,
    argv: list[str],
    env: dict[str, str],
    timeout: float | None,
) -> str:
    try:
        result = runner(argv, env, timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProviderCommandError(
            f"{argv[0]} {argv[-1]} timed out after {timeout}s"
        ) from exc
    except OSError as exc:
        raise ProviderCommandError(
            f"could not execute {argv[0]!r}: {_short_error(str(exc))}"
        ) from exc

    if result.returncode != 0:
        detail = _short_error(result.stderr or result.stdout or "no command output")
        raise ProviderCommandError(
            f"{argv[0]} command failed (exit {result.returncode}): {detail}"
        )
    return result.stdout


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _required_string(entry: dict[str, Any], key: str, context: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PoolConfigError(f"{context}: {key!r} must be a non-empty string")
    return value.strip()


def _optional_capacity(entry: dict[str, Any], context: str) -> int | None:
    value = entry.get("capacity")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PoolConfigError(f"{context}: 'capacity' must be an integer >= 1")
    return value


@dataclass(frozen=True)
class RunnerAccount:
    """One provider account with credential locations but no credential data."""

    id: str
    provider: str
    enabled: bool = True
    capacity: int | None = None
    description: str = ""
    config_dir: str = ""
    home_dir: str = ""
    session_config: str = ""
    client_oauth_config: str = ""
    auth: str = "oauth2"

    def public_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "provider": self.provider,
            "enabled": self.enabled,
            "capacity": self.capacity,
            "description": self.description,
        }
        if self.provider == PROVIDER_KAGGLE:
            data["config_dir"] = self.config_dir
        else:
            data.update(
                {
                    "home_dir": self.home_dir,
                    "session_config": self.session_config,
                    "client_oauth_config": self.client_oauth_config,
                    "auth": self.auth,
                }
            )
        return data


@dataclass(frozen=True)
class RunnerPoolConfig:
    accounts: tuple[RunnerAccount, ...]
    state_files: tuple[str, ...] = ()
    source_path: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POOL_CONFIG_SCHEMA,
            "source_path": self.source_path,
            "accounts": [account.public_dict() for account in self.accounts],
            "state_files": list(self.state_files),
        }


@dataclass
class RunnerJob:
    account_id: str
    provider: str
    job_id: str
    name: str
    state: str
    sources: tuple[str, ...] = ("remote",)
    title: str = ""
    author: str = ""
    updated_at: str = ""
    remote_state: str | None = None
    scheduler_state: str | None = None
    state_paths: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "provider": self.provider,
            "job_id": self.job_id,
            "name": self.name,
            "title": self.title,
            "author": self.author,
            "state": self.state,
            "remote_state": self.remote_state,
            "scheduler_state": self.scheduler_state,
            "sources": list(self.sources),
            "updated_at": self.updated_at,
            "state_paths": list(self.state_paths),
            "error": self.error,
        }


@dataclass
class AccountSnapshot:
    account_id: str
    provider: str
    status: str
    capacity: int | None
    jobs: list[RunnerJob] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(job.state for job in self.jobs)
        active_jobs = sum(job.state in ACTIVE_STATES for job in self.jobs)
        if self.capacity is None:
            available_capacity = None
        elif self.status in (ACCOUNT_ERROR, ACCOUNT_DISABLED):
            available_capacity = 0
        else:
            available_capacity = max(self.capacity - active_jobs, 0)
        return {
            "account_id": self.account_id,
            "provider": self.provider,
            "status": self.status,
            "capacity": self.capacity,
            "active_jobs": active_jobs,
            "available_capacity": available_capacity,
            "job_count": len(self.jobs),
            "state_counts": dict(sorted(counts.items())),
            "error": self.error,
        }


@dataclass
class PoolSnapshot:
    accounts: list[AccountSnapshot]
    jobs: list[RunnerJob]
    state_errors: list[dict[str, str]] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        account_counts = Counter(account.status for account in self.accounts)
        state_counts = Counter(job.state for job in self.jobs)
        enabled_accounts = [
            account for account in self.accounts if account.status != ACCOUNT_DISABLED
        ]
        observable_accounts = [
            account
            for account in enabled_accounts
            if account.status in (ACCOUNT_OK, ACCOUNT_DEGRADED)
        ]
        known_capacity = sum(
            account.capacity or 0
            for account in enabled_accounts
            if account.capacity is not None
        )
        active_slots = sum(
            job.state in ACTIVE_STATES
            for account in observable_accounts
            for job in account.jobs
        )
        observable_capacity = sum(
            account.capacity or 0
            for account in observable_accounts
            if account.capacity is not None
        )
        return {
            "schema_version": POOL_SNAPSHOT_SCHEMA,
            "generated_at": self.generated_at or _utc_now(),
            "summary": {
                "accounts": len(self.accounts),
                "account_status_counts": dict(sorted(account_counts.items())),
                "jobs": len(self.jobs),
                "state_counts": dict(sorted(state_counts.items())),
                "state_errors": len(self.state_errors),
                "known_capacity": known_capacity,
                "observable_capacity": observable_capacity,
                "active_slots": active_slots,
                "available_capacity": max(observable_capacity - active_slots, 0),
            },
            "accounts": [account.to_dict() for account in self.accounts],
            "jobs": [job.to_dict() for job in self.jobs],
            "state_errors": list(self.state_errors),
        }


class SlotLeaseStore:
    """Cross-process account-capacity leases backed by an atomic JSON file.

    The companion ``.lock`` file is held only while mutating lease state.
    Leases are keyed by account and immutable scheduler job name.  Reacquiring
    the same key by the same durable owner renews it, which lets a restarted
    scheduler reclaim its remote jobs without waiting for expiry.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ttl: float = DEFAULT_LEASE_TTL,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl <= 0:
            raise ValueError(f"lease ttl must be > 0, got {ttl}")
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.ttl = float(ttl)
        self.clock = clock

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": LEASE_STATE_SCHEMA, "leases": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LeaseStoreError(f"invalid lease state {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != LEASE_STATE_SCHEMA:
            raise LeaseStoreError(
                f"unsupported lease state schema in {self.path}; "
                f"expected {LEASE_STATE_SCHEMA!r}"
            )
        leases = raw.get("leases")
        if not isinstance(leases, list) or not all(isinstance(item, dict) for item in leases):
            raise LeaseStoreError(f"lease state {self.path} has invalid 'leases'")
        for index, item in enumerate(leases):
            if not all(
                isinstance(item.get(key), str) and bool(item[key])
                for key in ("account_id", "job_name", "owner_id")
            ) or not all(
                isinstance(item.get(key), (int, float))
                and not isinstance(item.get(key), bool)
                for key in ("acquired_at", "renewed_at", "expires_at")
            ):
                raise LeaseStoreError(
                    f"lease state {self.path} has invalid lease #{index}"
                )
        return raw

    def _write_unlocked(self, leases: list[dict[str, Any]], now: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LEASE_STATE_SCHEMA,
            "updated_at": now,
            "leases": sorted(
                leases,
                key=lambda item: (str(item["account_id"]), str(item["job_name"])),
            ),
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _mutate(
        self, callback: Callable[[list[dict[str, Any]], float], tuple[Any, bool]]
    ) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            now = float(self.clock())
            raw = self._read_unlocked()
            leases = [
                dict(item)
                for item in raw["leases"]
                if isinstance(item.get("expires_at"), (int, float))
                and float(item["expires_at"]) > now
            ]
            pruned = len(leases) != len(raw["leases"])
            result, changed = callback(leases, now)
            if pruned or changed:
                self._write_unlocked(leases, now)
            return result

    def acquire(
        self,
        *,
        account_id: str,
        job_name: str,
        owner_id: str,
        capacity: int,
    ) -> bool:
        """Acquire or renew one slot, returning ``False`` when full/owned."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        def mutate(leases: list[dict[str, Any]], now: float) -> tuple[bool, bool]:
            existing = next(
                (
                    item
                    for item in leases
                    if item.get("account_id") == account_id
                    and item.get("job_name") == job_name
                ),
                None,
            )
            if existing is not None:
                if existing.get("owner_id") != owner_id:
                    return False, False
                existing["expires_at"] = now + self.ttl
                existing["renewed_at"] = now
                return True, True
            used = sum(item.get("account_id") == account_id for item in leases)
            if used >= capacity:
                return False, False
            leases.append(
                {
                    "account_id": account_id,
                    "job_name": job_name,
                    "owner_id": owner_id,
                    "acquired_at": now,
                    "renewed_at": now,
                    "expires_at": now + self.ttl,
                }
            )
            return True, True

        return bool(self._mutate(mutate))

    def renew(self, *, account_id: str, job_name: str, owner_id: str) -> bool:
        """Renew an existing lease without allocating a new slot."""

        def mutate(leases: list[dict[str, Any]], now: float) -> tuple[bool, bool]:
            for item in leases:
                if (
                    item.get("account_id") == account_id
                    and item.get("job_name") == job_name
                    and item.get("owner_id") == owner_id
                ):
                    item["expires_at"] = now + self.ttl
                    item["renewed_at"] = now
                    return True, True
            return False, False

        return bool(self._mutate(mutate))

    def release(self, *, account_id: str, job_name: str, owner_id: str) -> bool:
        """Release one lease if it belongs to *owner_id*."""

        def mutate(leases: list[dict[str, Any]], _now: float) -> tuple[bool, bool]:
            for index, item in enumerate(leases):
                if (
                    item.get("account_id") == account_id
                    and item.get("job_name") == job_name
                    and item.get("owner_id") == owner_id
                ):
                    leases.pop(index)
                    return True, True
            return False, False

        return bool(self._mutate(mutate))

    def snapshot(self) -> dict[str, Any]:
        """Return live (non-expired) leases and prune stale records."""

        def mutate(leases: list[dict[str, Any]], _now: float) -> tuple[dict[str, Any], bool]:
            return {
                "schema_version": LEASE_STATE_SCHEMA,
                "leases": [dict(item) for item in leases],
            }, False

        return dict(self._mutate(mutate))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_to_iso(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def load_pool_config(path: str | Path) -> RunnerPoolConfig:
    """Load and strictly validate a v1 runner-pool configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise PoolConfigError(f"runner pool config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PoolConfigError(f"invalid runner pool JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PoolConfigError("runner pool config must be a JSON object")
    if raw.get("schema_version") != POOL_CONFIG_SCHEMA:
        raise PoolConfigError(
            f"unsupported schema_version {raw.get('schema_version')!r}; "
            f"expected {POOL_CONFIG_SCHEMA!r}"
        )

    raw_accounts = raw.get("accounts")
    if not isinstance(raw_accounts, list) or not raw_accounts:
        raise PoolConfigError("runner pool 'accounts' must be a non-empty list")

    base = config_path.parent
    accounts: list[RunnerAccount] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_accounts):
        context = f"account #{index}"
        if not isinstance(entry, dict):
            raise PoolConfigError(f"{context} must be an object")
        account_id = _required_string(entry, "id", context)
        context = f"account {account_id!r}"
        if account_id in seen:
            raise PoolConfigError(f"duplicate account id: {account_id!r}")
        seen.add(account_id)
        provider = _required_string(entry, "provider", context).lower()
        if provider not in VALID_PROVIDERS:
            raise PoolConfigError(
                f"{context}: provider must be one of {sorted(VALID_PROVIDERS)!r}"
            )
        allowed = (
            _KAGGLE_ACCOUNT_KEYS
            if provider == PROVIDER_KAGGLE
            else _COLAB_ACCOUNT_KEYS
        )
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise PoolConfigError(f"{context}: unknown fields: {unknown!r}")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise PoolConfigError(f"{context}: 'enabled' must be a boolean")
        description = entry.get("description", "")
        if not isinstance(description, str):
            raise PoolConfigError(f"{context}: 'description' must be a string")
        capacity = _optional_capacity(entry, context)

        if provider == PROVIDER_KAGGLE:
            config_dir_raw = entry.get("config_dir", "~/.kaggle")
            if not isinstance(config_dir_raw, str) or not config_dir_raw.strip():
                raise PoolConfigError(
                    f"{context}: 'config_dir' must be a non-empty string"
                )
            accounts.append(
                RunnerAccount(
                    id=account_id,
                    provider=provider,
                    enabled=enabled,
                    capacity=capacity,
                    description=description,
                    config_dir=str(_resolve_path(config_dir_raw, base)),
                )
            )
            continue

        home_raw = entry.get("home_dir", "~")
        if not isinstance(home_raw, str) or not home_raw.strip():
            raise PoolConfigError(f"{context}: 'home_dir' must be a non-empty string")
        home = _resolve_path(home_raw, base)
        session_raw = entry.get(
            "session_config", str(home / ".config/colab-cli/sessions.json")
        )
        oauth_raw = entry.get(
            "client_oauth_config", str(home / ".colab-cli-oauth-config.json")
        )
        if not isinstance(session_raw, str) or not session_raw.strip():
            raise PoolConfigError(
                f"{context}: 'session_config' must be a non-empty string"
            )
        if not isinstance(oauth_raw, str) or not oauth_raw.strip():
            raise PoolConfigError(
                f"{context}: 'client_oauth_config' must be a non-empty string"
            )
        auth = entry.get("auth", "oauth2")
        if auth not in ("oauth2", "adc"):
            raise PoolConfigError(f"{context}: 'auth' must be 'oauth2' or 'adc'")
        accounts.append(
            RunnerAccount(
                id=account_id,
                provider=provider,
                enabled=enabled,
                capacity=capacity,
                description=description,
                home_dir=str(home),
                session_config=str(_resolve_path(session_raw, base)),
                client_oauth_config=str(_resolve_path(oauth_raw, base)),
                auth=auth,
            )
        )

    raw_states = raw.get("state_files", [])
    if not isinstance(raw_states, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_states
    ):
        raise PoolConfigError("'state_files' must be a list of non-empty strings")
    state_files = tuple(str(_resolve_path(item, base)) for item in raw_states)
    return RunnerPoolConfig(
        accounts=tuple(accounts),
        state_files=state_files,
        source_path=str(config_path),
    )


def _normalize_kaggle_status(raw: str) -> str:
    text = raw.strip().lower()
    if "complete" in text and "error" not in text:
        return STATE_SUCCEEDED
    if any(word in text for word in ("error", "failed", "cancel")):
        return STATE_FAILED
    if "running" in text:
        return STATE_RUNNING
    if any(word in text for word in ("queued", "pending", "launching")):
        return STATE_QUEUED
    return STATE_UNKNOWN


def _normalize_scheduler_state(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "complete": STATE_SUCCEEDED,
        "completed": STATE_SUCCEEDED,
        "success": STATE_SUCCEEDED,
        "succeeded": STATE_SUCCEEDED,
        "error": STATE_FAILED,
        "cancelled": STATE_FAILED,
        "canceled": STATE_FAILED,
    }
    value = aliases.get(value, value)
    if value in ACTIVE_STATES or value in TERMINAL_STATES:
        return value
    return STATE_UNKNOWN


def _parse_colab_sessions(output: str) -> list[dict[str, str]]:
    """Use the scheduler's Colab parser so inventory and admission agree."""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from colab_provider import parse_sessions  # type: ignore[import-not-found]

    return parse_sessions(output)


def _inspect_kaggle_account(
    account: RunnerAccount,
    *,
    runner: CommandRunner,
    limit: int,
    timeout: float,
) -> AccountSnapshot:
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = account.config_dir
    try:
        stdout = _run_checked(
            runner,
            [
                "kaggle",
                "kernels",
                "list",
                "--mine",
                "--page-size",
                str(limit),
                "--sort-by",
                "dateRun",
                "--format",
                "json",
            ],
            env,
            timeout,
        )
        payload = json.loads(stdout)
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("items", []))
        if not isinstance(payload, list):
            raise ValueError("Kaggle list output is not a JSON list")
    except (ProviderCommandError, json.JSONDecodeError, ValueError) as exc:
        return AccountSnapshot(
            account_id=account.id,
            provider=account.provider,
            status=ACCOUNT_ERROR,
            capacity=account.capacity,
            error=_short_error(str(exc)),
        )

    jobs: list[RunnerJob] = []
    degraded = False
    for item in payload[:limit]:
        if not isinstance(item, dict):
            degraded = True
            continue
        ref = item.get("ref") or item.get("id")
        if not isinstance(ref, str) or not ref.strip():
            degraded = True
            continue
        ref = ref.strip()
        status_error: str | None = None
        try:
            status_raw = _run_checked(
                runner,
                ["kaggle", "kernels", "status", ref],
                env,
                timeout,
            )
            remote_state = _normalize_kaggle_status(status_raw)
            if remote_state == STATE_UNKNOWN:
                degraded = True
                status_error = "unrecognized Kaggle status output"
        except ProviderCommandError as exc:
            remote_state = STATE_UNKNOWN
            degraded = True
            status_error = _short_error(str(exc))
        jobs.append(
            RunnerJob(
                account_id=account.id,
                provider=account.provider,
                job_id=ref,
                name=ref.rsplit("/", 1)[-1],
                title=str(item.get("title") or ""),
                author=str(item.get("author") or ""),
                state=remote_state,
                remote_state=remote_state,
                updated_at=str(item.get("lastRunTime") or item.get("updatedAt") or ""),
                error=status_error,
            )
        )
    return AccountSnapshot(
        account_id=account.id,
        provider=account.provider,
        status=ACCOUNT_DEGRADED if degraded else ACCOUNT_OK,
        capacity=account.capacity,
        jobs=jobs,
    )


def _inspect_colab_account(
    account: RunnerAccount,
    *,
    runner: CommandRunner,
    timeout: float,
) -> AccountSnapshot:
    # Colab's OAuth2 path starts an interactive browser/code flow when the
    # account token is missing.  Pool inventory must remain non-interactive, so
    # fail this account explicitly and leave the rest of the pool observable.
    if account.auth == "oauth2":
        token_path = Path(account.home_dir) / ".config/colab-cli/token.json"
        if not token_path.is_file():
            return AccountSnapshot(
                account_id=account.id,
                provider=account.provider,
                status=ACCOUNT_ERROR,
                capacity=account.capacity,
                error=(
                    f"Colab OAuth token not found at {token_path}; "
                    "authenticate this account before inventory polling"
                ),
            )
    env = dict(os.environ)
    env["HOME"] = account.home_dir
    argv = [
        "colab",
        "--client-oauth-config",
        account.client_oauth_config,
        "--config",
        account.session_config,
        "--auth",
        account.auth,
        "sessions",
    ]
    try:
        stdout = _run_checked(runner, argv, env, timeout)
        sessions = _parse_colab_sessions(stdout)
    except (ProviderCommandError, ValueError) as exc:
        return AccountSnapshot(
            account_id=account.id,
            provider=account.provider,
            status=ACCOUNT_ERROR,
            capacity=account.capacity,
            error=_short_error(str(exc)),
        )
    jobs = [
        RunnerJob(
            account_id=account.id,
            provider=account.provider,
            job_id=session["name"],
            name=session["name"],
            state=_normalize_scheduler_state(session.get("state", STATE_RUNNING)),
            remote_state=_normalize_scheduler_state(
                session.get("state", STATE_RUNNING)
            ),
        )
        for session in sessions
    ]
    return AccountSnapshot(
        account_id=account.id,
        provider=account.provider,
        status=ACCOUNT_OK,
        capacity=account.capacity,
        jobs=jobs,
    )


def _scheduler_jobs(
    state_paths: Iterable[str],
) -> tuple[list[RunnerJob], list[dict[str, str]]]:
    jobs: list[RunnerJob] = []
    errors: list[dict[str, str]] = []
    for raw_path in state_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_jobs = payload.get("jobs")
            if not isinstance(raw_jobs, dict):
                raise ValueError("state file 'jobs' must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"path": str(path), "error": _short_error(str(exc))})
            continue
        updated_at = _timestamp_to_iso(payload.get("updated_at"))
        default_account = payload.get("account_id")
        for name, raw_job in raw_jobs.items():
            if not isinstance(raw_job, dict):
                errors.append(
                    {
                        "path": str(path),
                        "error": f"job {name!r} is not an object",
                    }
                )
                continue
            provider = str(raw_job.get("provider") or PROVIDER_KAGGLE).lower()
            if provider not in VALID_PROVIDERS:
                provider = str(raw_job.get("provider") or STATE_UNKNOWN)
            job_id = str(
                raw_job.get("remote_id")
                or raw_job.get("kernel_id")
                or name
            )
            state = _normalize_scheduler_state(raw_job.get("state"))
            account_id = str(raw_job.get("account_id") or default_account or "unassigned")
            jobs.append(
                RunnerJob(
                    account_id=account_id,
                    provider=provider,
                    job_id=job_id,
                    name=str(name),
                    state=state,
                    sources=("scheduler",),
                    scheduler_state=state,
                    updated_at=updated_at,
                    state_paths=(str(path),),
                    error=(
                        str(raw_job["error"])
                        if raw_job.get("error") is not None
                        else None
                    ),
                )
            )
    return jobs, errors


def _merge_scheduler_jobs(
    remote_jobs: list[RunnerJob], scheduler_jobs: list[RunnerJob]
) -> list[RunnerJob]:
    merged = list(remote_jobs)
    remote_index: dict[tuple[str, str], RunnerJob] = {
        (job.provider, job.job_id): job for job in remote_jobs
    }
    unmatched_index: dict[tuple[str, str], RunnerJob] = {}
    for local in scheduler_jobs:
        key = (local.provider, local.job_id)
        target = remote_index.get(key)
        if target is None:
            target = unmatched_index.get(key)
        if target is None:
            merged.append(local)
            unmatched_index[key] = local
            continue

        target.sources = tuple(dict.fromkeys((*target.sources, *local.sources)))
        target.scheduler_state = local.scheduler_state
        target.state_paths = tuple(
            dict.fromkeys((*target.state_paths, *local.state_paths))
        )
        if target.account_id == "unassigned" and local.account_id != "unassigned":
            target.account_id = local.account_id
        if local.name:
            target.name = local.name
        if not target.updated_at:
            target.updated_at = local.updated_at
        if local.scheduler_state in TERMINAL_STATES:
            target.state = local.scheduler_state
        elif target.remote_state and target.remote_state != STATE_UNKNOWN:
            target.state = target.remote_state
        else:
            target.state = local.scheduler_state or STATE_UNKNOWN
        if local.error:
            target.error = local.error
    return merged


def inspect_pool(
    config: RunnerPoolConfig,
    *,
    runner: CommandRunner = _default_command_runner,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    extra_state_files: Sequence[str] = (),
    account_ids: set[str] | None = None,
    providers: set[str] | None = None,
    active_only: bool = False,
) -> PoolSnapshot:
    """Inspect every selected account and merge durable scheduler states."""
    if limit < 1 or limit > MAX_KAGGLE_LIMIT:
        raise ValueError(f"limit must be 1..{MAX_KAGGLE_LIMIT}, got {limit}")
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")

    account_snapshots: list[AccountSnapshot] = []
    remote_jobs: list[RunnerJob] = []
    for account in config.accounts:
        if account_ids is not None and account.id not in account_ids:
            continue
        if providers is not None and account.provider not in providers:
            continue
        if not account.enabled:
            account_snapshots.append(
                AccountSnapshot(
                    account_id=account.id,
                    provider=account.provider,
                    status=ACCOUNT_DISABLED,
                    capacity=account.capacity,
                )
            )
            continue
        if account.provider == PROVIDER_KAGGLE:
            snapshot = _inspect_kaggle_account(
                account, runner=runner, limit=limit, timeout=timeout
            )
        else:
            snapshot = _inspect_colab_account(
                account, runner=runner, timeout=timeout
            )
        account_snapshots.append(snapshot)
        remote_jobs.extend(snapshot.jobs)

    state_files = tuple(
        dict.fromkeys(
            str(Path(path).expanduser().resolve())
            for path in (*config.state_files, *extra_state_files)
        )
    )
    local_jobs, state_errors = _scheduler_jobs(state_files)
    jobs = _merge_scheduler_jobs(remote_jobs, local_jobs)
    if active_only:
        jobs = [job for job in jobs if job.state in ACTIVE_STATES]

    # Account summaries describe the same merged/filtered view rendered below,
    # not the raw provider response before scheduler-state correlation.
    for account_snapshot in account_snapshots:
        account_snapshot.jobs = [
            job for job in jobs if job.account_id == account_snapshot.account_id
        ]

    state_order = {
        STATE_RUNNING: 0,
        STATE_QUEUED: 1,
        STATE_SUBMITTING: 2,
        STATE_COLLECTING: 3,
        STATE_PENDING: 4,
        STATE_FAILED: 5,
        STATE_UNKNOWN: 6,
        STATE_SUCCEEDED: 7,
    }
    jobs.sort(
        key=lambda job: (
            state_order.get(job.state, 99),
            job.account_id,
            job.provider,
            job.name,
        )
    )
    return PoolSnapshot(
        accounts=account_snapshots,
        jobs=jobs,
        state_errors=state_errors,
        generated_at=_utc_now(),
    )


def _load_validated_batch(batch_path: str | Path) -> list[dict[str, Any]]:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from parallel_scheduler import load_batch  # type: ignore[import-not-found]

    return load_batch(batch_path)


def create_dispatch_plan(
    config: RunnerPoolConfig,
    snapshot: PoolSnapshot,
    batch_path: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, non-submitting account assignment plan.

    Only healthy enabled accounts with configured capacities receive new
    assignments.  Existing scheduler-to-account correlations are preserved.
    Jobs may be assigned with ``waiting_for_capacity`` when every healthy
    account for their provider is currently full; the lease gate will keep
    them pending until a slot becomes available.
    """
    resolved_batch = Path(batch_path).expanduser().resolve()
    batch_jobs = _load_validated_batch(resolved_batch)
    account_order = {account.id: index for index, account in enumerate(config.accounts)}
    account_config = {account.id: account for account in config.accounts}
    account_snapshots = {account.account_id: account for account in snapshot.accounts}
    active = {
        account.account_id: int(account.to_dict()["active_jobs"])
        for account in snapshot.accounts
    }
    planned = {account.id: 0 for account in config.accounts}
    existing_by_name: dict[tuple[str, str], list[RunnerJob]] = {}
    for remote_job in snapshot.jobs:
        if remote_job.account_id == "unassigned":
            continue
        existing_by_name.setdefault(
            (remote_job.provider, remote_job.name), []
        ).append(remote_job)

    assignments: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for batch_job in batch_jobs:
        name = str(batch_job["name"])
        provider = str(batch_job.get("provider", PROVIDER_KAGGLE))
        existing = existing_by_name.get((provider, name), [])
        existing_accounts = sorted({job.account_id for job in existing})
        if len(existing_accounts) > 1:
            errors.append(
                {
                    "job_name": name,
                    "provider": provider,
                    "error": f"ambiguous existing account assignments: {existing_accounts!r}",
                }
            )
            continue
        if existing_accounts:
            account_id = existing_accounts[0]
            account = account_config.get(account_id)
            observed = account_snapshots.get(account_id)
            if (
                account is None
                or account.provider != provider
                or not account.enabled
                or account.capacity is None
                or observed is None
                or observed.status != ACCOUNT_OK
            ):
                errors.append(
                    {
                        "job_name": name,
                        "provider": provider,
                        "error": (
                            f"existing assignment references unavailable account "
                            f"{account_id!r}"
                        ),
                    }
                )
                continue
            assignments.append(
                {
                    "job_name": name,
                    "provider": provider,
                    "account_id": account_id,
                    "status": "existing",
                    "reason": "preserved scheduler/provider correlation",
                }
            )
            continue

        candidates: list[RunnerAccount] = []
        for account in config.accounts:
            observed = account_snapshots.get(account.id)
            if (
                account.enabled
                and account.provider == provider
                and account.capacity is not None
                and observed is not None
                and observed.status == ACCOUNT_OK
            ):
                candidates.append(account)
        if not candidates:
            errors.append(
                {
                    "job_name": name,
                    "provider": provider,
                    "error": "no healthy enabled account with configured capacity",
                }
            )
            continue

        def pressure(account: RunnerAccount) -> tuple[float, int, int]:
            assert account.capacity is not None
            projected = active.get(account.id, 0) + planned[account.id]
            return (
                projected / account.capacity,
                projected,
                account_order[account.id],
            )

        selected = min(candidates, key=pressure)
        assert selected.capacity is not None
        projected_before = active.get(selected.id, 0) + planned[selected.id]
        available_before = max(selected.capacity - projected_before, 0)
        assignments.append(
            {
                "job_name": name,
                "provider": provider,
                "account_id": selected.id,
                "status": "ready" if available_before > 0 else "waiting_for_capacity",
                "reason": (
                    f"deterministic lowest load ratio; "
                    f"projected={projected_before}/{selected.capacity}"
                ),
            }
        )
        planned[selected.id] += 1

    batch_providers = {
        str(job.get("provider", PROVIDER_KAGGLE)) for job in batch_jobs
    }
    inventory_errors = [dict(item) for item in snapshot.state_errors]
    for observed in snapshot.accounts:
        if (
            observed.provider in batch_providers
            and observed.status in (ACCOUNT_DEGRADED, ACCOUNT_ERROR)
        ):
            inventory_errors.append(
                {
                    "account_id": observed.account_id,
                    "provider": observed.provider,
                    "error": observed.error or f"account status is {observed.status}",
                }
            )
    all_assigned = not errors and len(assignments) == len(batch_jobs)
    return {
        "schema_version": DISPATCH_PLAN_SCHEMA,
        "generated_at": _utc_now(),
        "batch_path": str(resolved_batch),
        "batch_sha256": _sha256_file(resolved_batch),
        "pool_config_path": config.source_path,
        "pool_config_sha256": _sha256_file(config.source_path),
        "all_assigned": all_assigned,
        "ready_for_dispatch": all_assigned and not inventory_errors,
        "assignments": assignments,
        "errors": errors,
        "inventory_errors": inventory_errors,
    }


def load_dispatch_plan(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a dispatch plan artifact."""
    plan_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolConfigError(f"invalid dispatch plan {plan_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != DISPATCH_PLAN_SCHEMA:
        raise PoolConfigError(
            f"unsupported dispatch plan schema in {plan_path}; "
            f"expected {DISPATCH_PLAN_SCHEMA!r}"
        )
    if raw.get("all_assigned") is not True:
        raise PoolConfigError(f"dispatch plan {plan_path} is incomplete")
    if raw.get("ready_for_dispatch") is not True:
        raise PoolConfigError(
            f"dispatch plan {plan_path} was produced from a degraded inventory"
        )
    assignments = raw.get("assignments")
    if not isinstance(assignments, list) or not all(
        isinstance(item, dict) for item in assignments
    ):
        raise PoolConfigError(f"dispatch plan {plan_path} has invalid assignments")
    return raw


def write_dispatch_plan(path: str | Path, plan: dict[str, Any]) -> None:
    """Atomically write a dispatch plan without modifying its source batch."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _clip(value: Any, width: int) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def render_table(snapshot: PoolSnapshot) -> str:
    """Render a compact human-readable pool snapshot."""
    lines: list[str] = []
    for account in snapshot.accounts:
        capacity = "?" if account.capacity is None else str(account.capacity)
        active = sum(job.state in ACTIVE_STATES for job in account.jobs)
        if account.capacity is None:
            available = "?"
        elif account.status in (ACCOUNT_ERROR, ACCOUNT_DISABLED):
            available = "0"
        else:
            available = str(max(account.capacity - active, 0))
        line = (
            f"account {account.account_id} ({account.provider}) "
            f"status={account.status} jobs={len(account.jobs)} active={active} "
            f"capacity={capacity} available={available}"
        )
        if account.error:
            line += f" error={account.error}"
        lines.append(line)
    for error in snapshot.state_errors:
        lines.append(f"state-error {error['path']}: {error['error']}")

    lines.append("")
    headers = ("ACCOUNT", "PROVIDER", "STATE", "NAME", "REMOTE ID", "SOURCES")
    rows = [
        (
            _clip(job.account_id, 18),
            _clip(job.provider, 8),
            _clip(job.state, 10),
            _clip(job.name, 36),
            _clip(job.job_id, 48),
            _clip(",".join(job.sources), 18),
        )
        for job in snapshot.jobs
    ]
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]
    lines.append("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * width for width in widths))
    if rows:
        lines.extend(
            "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))
            for row in rows
        )
    else:
        lines.append("(no jobs)")

    counts = Counter(job.state for job in snapshot.jobs)
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    lines.append("")
    lines.append(f"jobs={len(snapshot.jobs)}" + (f" ({count_text})" if count_text else ""))
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runner_pool",
        description="Unified read-only inventory across remote runner accounts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate and print public pool config.")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    plan = sub.add_parser(
        "plan", help="Dry-run deterministic account assignment for a batch."
    )
    plan.add_argument("batch", type=Path, help="Validated scheduler batch JSON.")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    plan.add_argument(
        "--state",
        type=Path,
        action="append",
        default=[],
        help="Scheduler state to correlate before planning; repeatable.",
    )
    plan.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Recent Kaggle kernels per account (default {DEFAULT_LIMIT}).",
    )
    plan.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-provider-command timeout in seconds (default {DEFAULT_TIMEOUT}).",
    )
    plan.add_argument(
        "--output",
        type=Path,
        help="Atomically write the reviewed plan artifact to this path.",
    )

    status = sub.add_parser("status", help="List jobs across the configured pool.")
    status.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    status.add_argument(
        "--state",
        type=Path,
        action="append",
        default=[],
        help="Additional scheduler state file to merge; repeatable.",
    )
    status.add_argument(
        "--account",
        action="append",
        default=[],
        help="Only inspect this account id; repeatable.",
    )
    status.add_argument(
        "--provider",
        action="append",
        choices=sorted(VALID_PROVIDERS),
        default=[],
        help="Only inspect this provider; repeatable.",
    )
    status.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Recent Kaggle kernels per account (default {DEFAULT_LIMIT}, max {MAX_KAGGLE_LIMIT}).",
    )
    status.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-provider-command timeout in seconds (default {DEFAULT_TIMEOUT}).",
    )
    status.add_argument(
        "--active-only",
        action="store_true",
        help="Show pending/submitting/queued/running/collecting jobs only.",
    )
    status.add_argument("--json", action="store_true", help="Print stable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_pool_config(args.config)
    except PoolConfigError as exc:
        parser.error(str(exc))

    if args.command == "validate":
        print(json.dumps(config.public_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "plan":
        try:
            snapshot = inspect_pool(
                config,
                limit=args.limit,
                timeout=args.timeout,
                extra_state_files=[str(path) for path in args.state],
            )
            plan_payload = create_dispatch_plan(config, snapshot, args.batch)
        except (PoolConfigError, ValueError, OSError) as exc:
            parser.error(str(exc))
        if args.output is not None:
            write_dispatch_plan(args.output, plan_payload)
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        return 0 if plan_payload["ready_for_dispatch"] else 1

    selected_accounts = set(args.account) if args.account else None
    if selected_accounts:
        known = {account.id for account in config.accounts}
        unknown = sorted(selected_accounts - known)
        if unknown:
            parser.error(f"unknown account ids: {unknown!r}")
    try:
        snapshot = inspect_pool(
            config,
            limit=args.limit,
            timeout=args.timeout,
            extra_state_files=[str(path) for path in args.state],
            account_ids=selected_accounts,
            providers=set(args.provider) if args.provider else None,
            active_only=args.active_only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_table(snapshot))
    unhealthy = any(
        account.status in (ACCOUNT_DEGRADED, ACCOUNT_ERROR)
        for account in snapshot.accounts
    ) or bool(snapshot.state_errors)
    return 1 if unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
