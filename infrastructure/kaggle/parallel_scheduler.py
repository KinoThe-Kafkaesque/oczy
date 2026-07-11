"""Durable CPU-only parallel experiment runtime and scheduler.

Runs many private, CPU-only Kaggle kernels and/or Google Colab CLI jobs in
bounded parallelism with crash-safe state, resume semantics, output
collection, and timeout handling.

Batch manifest schema v1 (``oczy/kaggle-parallel-batch/v1``) — implicit
Kaggle only::

    {
      "schema_version": "oczy/kaggle-parallel-batch/v1",
      "jobs": [
        {"name": "...", "kernel_dir": "...", "output_dir": "..."}
      ]
    }

Batch manifest schema v2 (``oczy/remote-parallel-batch/v2``) — mixed
providers, each job carries a ``provider`` field::

    {
      "schema_version": "oczy/remote-parallel-batch/v2",
      "jobs": [
        {"name": "...", "provider": "kaggle", "kernel_dir": "...", "output_dir": "..."},
        {"name": "...", "provider": "colab", "script": "...", "output_dir": "...",
         "arguments": ["--flag"], "timeout": 3600}
      ]
    }

Durable state schema v1 (``oczy/kaggle-parallel-state/v1``) — Kaggle only.
Durable state schema v2 (``oczy/remote-parallel-state/v2``) — provider-neutral
with ``provider``/``remote_id`` per job and ``colab_learned_limit`` at top
level.  v1 state files migrate automatically: every job is treated as
Kaggle with ``remote_id = kernel_id``.

Lifecycle: ``pending -> submitting -> running -> collecting -> succeeded``
with terminal ``failed``.  Resume converts interrupted ``submitting`` to
``pending`` and ``collecting`` to ``running`` (Kaggle) safely, and never
resubmits a kernel already recorded as ``running``.  For Colab, a running
job on restart has no local process handle — it is failed explicitly as
interrupted and the named session is best-effort stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys as _sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Optional Colab provider import (same directory; direct-script importable).
# Falls back gracefully so v1-only operation never depends on colab_provider.
# ---------------------------------------------------------------------------
# Ensure the script's own directory is importable (runpy.run_path does not
# always add it to sys.path).
_script_dir = str(Path(__file__).resolve().parent) if "__file__" in dir() else "."
if _script_dir not in _sys.path:
    _sys.path.insert(0, _script_dir)

try:
    from colab_provider import (  # type: ignore[import-not-found]
        COLAB_CAPACITY_REJECTED,
        COLAB_COMPLETE,
        COLAB_ERROR,
        ColabCliClient,
        classify_colab_output,
        detect_orphaned_sessions,
        _read_proc_output,
    )
    _COLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - colab_provider is co-located
    _COLAB_AVAILABLE = False
    ColabCliClient = None  # type: ignore[assignment, misc]
    COLAB_COMPLETE = "complete"
    COLAB_CAPACITY_REJECTED = "capacity_rejected"
    COLAB_ERROR = "error"

    def classify_colab_output(_proc: Any) -> str:  # type: ignore[no-redef]
        raise RuntimeError("colab_provider not available")

    def detect_orphaned_sessions(  # type: ignore[no-redef]
        _client: Any, _known_names: set[str]
    ) -> list[str]:
        raise RuntimeError("colab_provider not available")

    def _read_proc_output(_proc: Any, _stream: str) -> str:  # type: ignore[no-redef]
        raise RuntimeError("colab_provider not available")


# ---------------------------------------------------------------------------
# Schema versions — v1 constants preserved for backward compatibility.
# ---------------------------------------------------------------------------
BATCH_SCHEMA_VERSION = "oczy/kaggle-parallel-batch/v1"
STATE_SCHEMA_VERSION = "oczy/kaggle-parallel-state/v1"

BATCH_SCHEMA_V2 = "oczy/remote-parallel-batch/v2"
STATE_SCHEMA_V2 = "oczy/remote-parallel-state/v2"

_VALID_BATCH_SCHEMAS = frozenset({BATCH_SCHEMA_VERSION, BATCH_SCHEMA_V2})
_VALID_STATE_SCHEMAS = frozenset({STATE_SCHEMA_VERSION, STATE_SCHEMA_V2})

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
PROVIDER_KAGGLE = "kaggle"
PROVIDER_COLAB = "colab"
_VALID_PROVIDERS = frozenset({PROVIDER_KAGGLE, PROVIDER_COLAB})

# ---------------------------------------------------------------------------
# Concurrency defaults and limits.
# ---------------------------------------------------------------------------
DEFAULT_MAX_PARALLEL = 8
HARD_MAX_PARALLEL = 10
MIN_PARALLEL = 1

DEFAULT_KAGGLE_MAX = 8
HARD_KAGGLE_MAX = 10
MIN_KAGGLE_MAX = 1

DEFAULT_COLAB_MAX = 10
MIN_COLAB_MAX = 1

DEFAULT_COLAB_COOLDOWN = 60.0
COLAB_AIMD_START = 1
COLAB_AIMD_MIN = 1
COLAB_MAX_CAPACITY_REJECTIONS = 10

DEFAULT_POLL_INTERVAL = 30
DEFAULT_PUSH_TIMEOUT = 21600
DEFAULT_JOB_TIMEOUT = 21600
DEFAULT_STATUS_TIMEOUT = 120
DEFAULT_OUTPUT_TIMEOUT = 1800
DEFAULT_PUSH_SUBPROCESS_TIMEOUT = 600

# ---------------------------------------------------------------------------
# Job lifecycle states.
# ---------------------------------------------------------------------------
PENDING = "pending"
SUBMITTING = "submitting"
RUNNING = "running"
COLLECTING = "collecting"
SUCCEEDED = "succeeded"
FAILED = "failed"

ACTIVE_STATES = frozenset({SUBMITTING, RUNNING, COLLECTING})
TERMINAL_STATES = frozenset({SUCCEEDED, FAILED})

# Kaggle CLI status classifications.
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
STATUS_RUNNING = "running"
STATUS_QUEUED = "queued"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Internal job representation with durable lifecycle state.

    Provider-neutral: ``provider`` distinguishes Kaggle vs Colab.  For
    Kaggle, ``kernel_id`` and ``remote_id`` are the same value.  For Colab,
    ``remote_id`` is the stable session name (equals the job name) and
    ``kernel_id`` is empty.  The ``proc`` field holds a live
    ``subprocess.Popen`` for Colab during a single run — it is **never**
    serialized and is always ``None`` after state reload.
    """

    name: str
    kernel_dir: str = ""
    output_dir: str = ""
    kernel_id: str = ""
    state: str = PENDING
    error: str | None = None
    submitted_at: float | None = None
    completed_at: float | None = None
    collected_at: float | None = None
    attempts: int = 0
    provider: str = PROVIDER_KAGGLE
    remote_id: str = ""
    script: str = ""
    arguments: list[str] = field(default_factory=list)
    timeout: float | None = None
    # Transient — never serialized.
    proc: Any = None
    capacity_rejections: int = 0
    # Transient — never serialized.  True when loaded from a state file
    # (non-fresh: active/interrupted/terminal); False when created from a
    # batch manifest (fresh: never submitted).  Used by orphan recovery to
    # avoid claiming/stopping preexisting external sessions for fresh jobs.
    _from_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "kernel_dir": self.kernel_dir,
            "output_dir": self.output_dir,
            "kernel_id": self.kernel_id,
            "state": self.state,
            "error": self.error,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "collected_at": self.collected_at,
            "attempts": self.attempts,
            "provider": self.provider,
            "remote_id": self.remote_id or self.kernel_id,
        }
        if self.provider == PROVIDER_COLAB:
            d["script"] = self.script
            d["arguments"] = list(self.arguments)
            d["timeout"] = self.timeout
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Reconstruct a Job from a state-file dict.

        v1 state (no ``provider``/``remote_id``) migrates automatically:
        ``provider`` defaults to ``"kaggle"`` and ``remote_id`` defaults
        to ``kernel_id``.
        """
        provider = data.get("provider", PROVIDER_KAGGLE)
        kernel_id = data.get("kernel_id", "")
        remote_id = data.get("remote_id", "")
        if not remote_id:
            remote_id = kernel_id
        job = cls(
            name=data["name"],
            kernel_dir=data.get("kernel_dir", ""),
            output_dir=data.get("output_dir", ""),
            kernel_id=kernel_id,
            state=data.get("state", PENDING),
            error=data.get("error"),
            submitted_at=data.get("submitted_at"),
            completed_at=data.get("completed_at"),
            collected_at=data.get("collected_at"),
            attempts=data.get("attempts", 0),
            provider=provider,
            remote_id=remote_id,
            script=data.get("script", ""),
            arguments=list(data.get("arguments", [])),
            timeout=data.get("timeout"),
        )
        job._from_state = True
        return job


# ---------------------------------------------------------------------------
# Kaggle client protocol and real CLI implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class KaggleClient(Protocol):
    """Dependency-injectable Kaggle CLI adapter.

    Any object with matching ``push``, ``status``, and ``output`` methods
    satisfies this protocol — tests can pass a fake without subclassing.
    """

    def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
        """Push kernel directory, return the canonical kernel id/slug."""
        ...

    def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
        """Return one of: complete, error, running, queued."""
        ...

    def output(self, kernel_id: str, output_dir: str, *, timeout: float | None = None) -> None:
        """Pull kernel output to *output_dir*."""
        ...


def _run_kaggle(argv: list[str], *, timeout: float | None) -> str:
    """Run a kaggle CLI command, returning stdout.  No shell=True."""
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle {' '.join(argv[:2])} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


class KaggleCliClient:
    """Real Kaggle CLI adapter using subprocess (no shell=True)."""

    def __init__(
        self,
        *,
        push_timeout: float = DEFAULT_PUSH_TIMEOUT,
        status_timeout: float = DEFAULT_STATUS_TIMEOUT,
        output_timeout: float = DEFAULT_OUTPUT_TIMEOUT,
    ) -> None:
        self.push_timeout = push_timeout
        self.status_timeout = status_timeout
        self.output_timeout = output_timeout

    def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
        """Submit a kernel, sending *timeout* as the Kaggle run-time limit."""
        to = timeout if timeout is not None else self.push_timeout
        _run_kaggle(
            [
                "kaggle", "kernels", "push", "-p", kernel_dir,
                "--timeout", str(int(to)),
            ],
            timeout=DEFAULT_PUSH_SUBPROCESS_TIMEOUT,
        )
        meta_path = Path(kernel_dir) / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta["id"]

    def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
        to = timeout if timeout is not None else self.status_timeout
        stdout = _run_kaggle(
            ["kaggle", "kernels", "status", kernel_id], timeout=to
        )
        return classify_status(stdout)

    def output(self, kernel_id: str, output_dir: str, *, timeout: float | None = None) -> None:
        """Download kernel output, forcing overwrite for resume idempotency."""
        to = timeout if timeout is not None else self.output_timeout
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _run_kaggle(
            ["kaggle", "kernels", "output", kernel_id, "-p", output_dir, "--force"],
            timeout=to,
        )


def classify_status(raw: str) -> str:
    """Classify Kaggle CLI status output into complete/error/running/queued.

    The Kaggle CLI prints lines like ``status: running`` or
    ``status: complete``.  We match case-insensitively and fall back to
    ``error`` for unknown output so the scheduler can make progress.
    """
    text = raw.strip().lower()
    if "complete" in text and "error" not in text:
        return STATUS_COMPLETE
    if "error" in text or "failed" in text or "cancel" in text:
        return STATUS_ERROR
    if "running" in text:
        return STATUS_RUNNING
    if "queued" in text or "pending" in text or "launching" in text:
        return STATUS_QUEUED
    return STATUS_ERROR


# ---------------------------------------------------------------------------
# Batch manifest loading and validation
# ---------------------------------------------------------------------------


class BatchValidationError(ValueError):
    """Raised when a batch manifest or kernel directory fails validation."""


def _title_slug(title: str) -> str:
    """Kaggle clean-URL title slug.

    Lowercase, collapse each run of non-ASCII-alphanumeric characters to a
    single hyphen, and strip leading/trailing hyphens.  Mirrors how Kaggle
    derives the final path component of a kernel URL from its title.

    Duplicated locally (rather than imported from prepare_research_kernel)
    so this module remains directly executable without a package context.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _resolve(base: Path, rel: str) -> Path:
    """Resolve *rel* relative to *base* (the batch manifest directory)."""
    p = Path(rel)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _load_kernel_metadata(kernel_dir: Path) -> dict[str, Any]:
    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        raise BatchValidationError(
            f"missing kernel-metadata.json in {kernel_dir}"
        )
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchValidationError(
            f"invalid kernel-metadata.json in {kernel_dir}: {exc}"
        ) from exc


def _load_job_spec(kernel_dir: Path) -> dict[str, Any]:
    spec_path = kernel_dir / "job_spec.json"
    if not spec_path.is_file():
        raise BatchValidationError(
            f"missing job_spec.json in {kernel_dir}"
        )
    try:
        return json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchValidationError(
            f"invalid job_spec.json in {kernel_dir}: {exc}"
        ) from exc


def _validate_kernel(kernel_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Validate a kernel directory is private + CPU-only.

    Returns (kernel_id, metadata, job_spec).
    """
    meta = _load_kernel_metadata(kernel_dir)

    required_false = ("enable_gpu", "enable_tpu", "enable_internet")
    for key in required_false:
        if meta.get(key) is not False:
            raise BatchValidationError(
                f"{kernel_dir}: {key} must be false (CPU-only, no internet)"
            )
    if meta.get("is_private") is not True:
        raise BatchValidationError(
            f"{kernel_dir}: is_private must be true"
        )
    if meta.get("machine_shape", "") != "":
        raise BatchValidationError(
            f"{kernel_dir}: machine_shape must be empty for CPU-only"
        )

    kernel_id = meta.get("id", "")
    if not kernel_id or not isinstance(kernel_id, str):
        raise BatchValidationError(
            f"{kernel_dir}: kernel-metadata.json missing string 'id'"
        )

    title = meta.get("title", "")
    if not title or not isinstance(title, str):
        raise BatchValidationError(
            f"{kernel_dir}: kernel-metadata.json missing string 'title'"
        )
    title_slug = _title_slug(title)
    id_slug = kernel_id.rsplit("/", 1)[-1]
    if title_slug != id_slug:
        raise BatchValidationError(
            f"{kernel_dir}: title {title!r} resolves to slug {title_slug!r} "
            f"but kernel_id {kernel_id!r} ends with {id_slug!r}; Kaggle would "
            f"create the kernel under a different slug and polling the "
            f"requested id would fail"
        )

    spec = _load_job_spec(kernel_dir)
    if spec.get("profile") != "cpu":
        raise BatchValidationError(
            f"{kernel_dir}: job_spec.json profile must be 'cpu', got "
            f"{spec.get('profile')!r}"
        )

    return kernel_id, meta, spec


def _validate_colab_job(
    base: Path, entry: dict[str, Any], name: str
) -> dict[str, Any]:
    """Validate a v2 Colab job entry.  Returns a job dict."""
    script_rel = entry.get("script")
    if not script_rel or not isinstance(script_rel, str):
        raise BatchValidationError(f"job {name!r} missing string 'script'")
    output_dir_rel = entry.get("output_dir")
    if not output_dir_rel or not isinstance(output_dir_rel, str):
        raise BatchValidationError(f"job {name!r} missing string 'output_dir'")

    script_path = _resolve(base, script_rel)
    output_dir = _resolve(base, output_dir_rel)
    if not script_path.is_file():
        raise BatchValidationError(
            f"job {name!r}: script does not exist: {script_path}"
        )

    arguments = entry.get("arguments", [])
    if arguments is None:
        arguments = []
    if not isinstance(arguments, list) or not all(
        isinstance(a, str) for a in arguments
    ):
        raise BatchValidationError(
            f"job {name!r}: arguments must be a list of strings"
        )

    timeout = entry.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise BatchValidationError(
            f"job {name!r}: timeout must be a positive number"
        )

    return {
        "name": name,
        "provider": PROVIDER_COLAB,
        "schema_version": BATCH_SCHEMA_V2,
        "script": str(script_path),
        "arguments": list(arguments),
        "output_dir": output_dir,
        "timeout": float(timeout) if timeout is not None else None,
    }


def _validate_kaggle_job(
    base: Path, entry: dict[str, Any], name: str, schema: str
) -> dict[str, Any]:
    """Validate a Kaggle job entry (v1 or v2).  Returns a job dict."""
    kernel_dir_rel = entry.get("kernel_dir")
    if not kernel_dir_rel or not isinstance(kernel_dir_rel, str):
        raise BatchValidationError(f"job {name!r} missing string 'kernel_dir'")
    output_dir_rel = entry.get("output_dir")
    if not output_dir_rel or not isinstance(output_dir_rel, str):
        raise BatchValidationError(f"job {name!r} missing string 'output_dir'")

    kernel_dir = _resolve(base, kernel_dir_rel)
    output_dir = _resolve(base, output_dir_rel)

    if not kernel_dir.is_dir():
        raise BatchValidationError(
            f"job {name!r}: kernel_dir does not exist: {kernel_dir}"
        )

    kernel_id, metadata, job_spec = _validate_kernel(kernel_dir)

    return {
        "name": name,
        "provider": PROVIDER_KAGGLE,
        "schema_version": schema,
        "kernel_dir": kernel_dir,
        "output_dir": output_dir,
        "kernel_id": kernel_id,
        "metadata": metadata,
        "job_spec": job_spec,
    }


def load_batch(batch_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a batch manifest.

    Accepts both v1 (``oczy/kaggle-parallel-batch/v1``) and v2
    (``oczy/remote-parallel-batch/v2``) schemas.  v1 jobs are implicitly
    Kaggle.  v2 jobs dispatch by their ``provider`` field.

    Returns a list of job dicts.  Kaggle job dicts contain:
        name, provider, schema_version, kernel_dir (Path),
        output_dir (Path), kernel_id (str), metadata (dict), job_spec (dict).
    Colab job dicts contain:
        name, provider, schema_version, script (str), arguments (list[str]),
        output_dir (Path), timeout (float|None).

    Paths in the manifest resolve relative to the manifest file's directory.
    Raises BatchValidationError on any validation failure.
    """
    batch_path = Path(batch_path).resolve()
    if not batch_path.is_file():
        raise BatchValidationError(f"batch manifest not found: {batch_path}")
    try:
        manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchValidationError(f"invalid batch manifest: {exc}") from exc

    schema = manifest.get("schema_version")
    if schema not in _VALID_BATCH_SCHEMAS:
        raise BatchValidationError(
            f"unsupported batch schema_version: "
            f"{schema!r}, expected one of {sorted(_VALID_BATCH_SCHEMAS)!r}"
        )

    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise BatchValidationError("batch manifest 'jobs' must be a non-empty list")

    base = batch_path.parent
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    jobs: list[dict[str, Any]] = []

    for i, entry in enumerate(raw_jobs):
        if not isinstance(entry, dict):
            raise BatchValidationError(f"job #{i} is not an object")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise BatchValidationError(f"job #{i} missing string 'name'")
        if name in seen_names:
            raise BatchValidationError(f"duplicate job name: {name}")
        seen_names.add(name)

        # Determine provider: v2 requires it, v1 implies kaggle.
        if schema == BATCH_SCHEMA_V2:
            provider = entry.get("provider")
            if provider not in _VALID_PROVIDERS:
                raise BatchValidationError(
                    f"job {name!r} has invalid or missing 'provider' "
                    f"(must be one of {sorted(_VALID_PROVIDERS)!r})"
                )
        else:
            provider = PROVIDER_KAGGLE

        if provider == PROVIDER_COLAB:
            jobs.append(_validate_colab_job(base, entry, name))
        else:
            job = _validate_kaggle_job(base, entry, name, schema)
            kid = job["kernel_id"]
            if kid in seen_ids:
                raise BatchValidationError(
                    f"job {name!r}: duplicate kernel id {kid!r}"
                )
            seen_ids.add(kid)
            jobs.append(job)

    return jobs


# ---------------------------------------------------------------------------
# Colab AIMD controller
# ---------------------------------------------------------------------------


class ColabAimdController:
    """AIMD controller for Colab admission capacity.

    Starts at ``COLAB_AIMD_START`` (1).  Increases by one after each
    successful admission.  On capacity rejection (HTTP 412), reduces the
    learned limit to the currently active Colab job count (minimum
    ``COLAB_AIMD_MIN`` = 1), meaning "don't admit more until some finish."
    The learned limit never exceeds ``ceiling`` (``colab_max``).
    """

    def __init__(self, *, ceiling: int = DEFAULT_COLAB_MAX) -> None:
        self.ceiling = ceiling
        self.learned_limit = max(COLAB_AIMD_MIN, min(COLAB_AIMD_START, ceiling))

    def on_success(self) -> None:
        if self.learned_limit < self.ceiling:
            self.learned_limit += 1

    def on_capacity_rejected(self, active_count: int) -> None:
        self.learned_limit = max(COLAB_AIMD_MIN, min(active_count, self.ceiling))

    def effective_limit(self) -> int:
        return min(self.learned_limit, self.ceiling)


# ---------------------------------------------------------------------------
# Durable state
# ---------------------------------------------------------------------------


def _state_to_dict(
    batch_path: str,
    jobs: dict[str, Job],
    now: float,
    *,
    schema_version: str = STATE_SCHEMA_VERSION,
    colab_learned_limit: int | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "schema_version": schema_version,
        "batch_path": str(batch_path),
        "jobs": {name: job.to_dict() for name, job in jobs.items()},
        "updated_at": now,
    }
    if colab_learned_limit is not None:
        d["colab_learned_limit"] = colab_learned_limit
    return d


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON to *path* via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.name + "."
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_state(state_path: str | Path) -> dict[str, Any]:
    """Load durable state file.  Returns empty structure if not found."""
    state_path = Path(state_path)
    if not state_path.is_file():
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "batch_path": "",
            "jobs": {},
            "updated_at": 0.0,
        }
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "batch_path": "",
            "jobs": {},
            "updated_at": 0.0,
        }
    return raw


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ParallelScheduler:
    """Bounded-parallel mixed-provider scheduler with durable state.

    Concurrency model: at most ``max_parallel`` remote jobs are active at
    once globally.  Within that, Kaggle jobs are bounded by ``kaggle_max``
    (fixed) and Colab jobs by an AIMD controller (``colab_max`` ceiling).
    The scheduler is single-threaded and synchronous — remote submissions
    are asynchronous work, so parallelism means N remote slots, not N local
    threads.

    Constructor injects the Kaggle client, optional Colab client, clock,
    and sleeper for deterministic testing.  The default clock is wall-clock
    time (``time.time``) so persisted timestamps survive process/host
    restart; tests may inject ``time.monotonic`` for deterministic values.
    Per-run parameters (max_parallel, timeouts, per-provider limits) are
    passed to :meth:`run`.
    """

    def __init__(
        self,
        client: KaggleClient,
        *,
        colab_client: Any = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.colab_client = colab_client
        self.clock = clock
        self.sleeper = sleeper
        # Set during run(); persists across the run for _save_state.
        self._state_schema: str = STATE_SCHEMA_VERSION
        self._colab_controller: ColabAimdController | None = None
        self._colab_cooldown_until: float = 0.0

    # -- state persistence -------------------------------------------------

    def _save_state(
        self, state_path: Path, batch_path: str, jobs: dict[str, Job]
    ) -> None:
        colab_limit = (
            self._colab_controller.learned_limit
            if self._colab_controller is not None
            else None
        )
        _atomic_write_json(
            state_path,
            _state_to_dict(
                batch_path,
                jobs,
                self.clock(),
                schema_version=self._state_schema,
                colab_learned_limit=colab_limit,
            ),
        )

    def _load_or_init_state(
        self, state_path: Path, batch_path: str, batch_jobs: list[dict[str, Any]]
    ) -> dict[str, Job]:
        """Load durable state and merge with batch jobs.

        New batch jobs not in state are added as pending.  Existing state
        jobs are kept (preserving lifecycle progress).  Interrupted states
        are resolved for resume.  A nonempty persisted ``batch_path`` that
        differs from the requested batch raises ``BatchValidationError``.

        For Colab, a running/collecting job on restart has no local process
        handle — it is failed as interrupted.  The caller (``run``) is
        responsible for best-effort ``stop()`` of orphaned sessions.
        """
        raw = load_state(state_path)
        state_schema = raw.get("schema_version")
        if state_schema not in _VALID_STATE_SCHEMAS:
            # Unknown schema — start fresh from batch.
            return self._fresh_jobs_from_batch(batch_jobs)

        stored_batch_path = raw.get("batch_path", "")
        if stored_batch_path and stored_batch_path != batch_path:
            raise BatchValidationError(
                f"state file is bound to batch {stored_batch_path!r}, "
                f"cannot resume batch {batch_path!r}"
            )

        jobs: dict[str, Job] = {}
        existing = raw.get("jobs", {})

        for batch_job in batch_jobs:
            name = batch_job["name"]
            provider = batch_job.get("provider", PROVIDER_KAGGLE)
            if name in existing:
                stored = existing[name]
                job = Job.from_dict(stored)
                # Update paths from batch (source of truth).
                job.output_dir = str(batch_job["output_dir"])
                if provider == PROVIDER_KAGGLE:
                    job.kernel_dir = str(batch_job["kernel_dir"])
                    job.kernel_id = batch_job["kernel_id"]
                    job.remote_id = job.kernel_id
                else:  # colab
                    job.script = batch_job.get("script", "")
                    job.arguments = list(batch_job.get("arguments", []))
                    job.timeout = batch_job.get("timeout")
                    job.remote_id = job.remote_id or name
                # Resume: interrupted submitting -> pending.
                if job.state == SUBMITTING:
                    job.state = PENDING
                elif job.state == COLLECTING:
                    if provider == PROVIDER_COLAB:
                        # No local proc to collect from — fail.
                        job.state = FAILED
                        job.error = "interrupted: collecting on restart with no process handle"
                    else:
                        job.state = RUNNING
                elif job.state == RUNNING:
                    if provider == PROVIDER_COLAB:
                        # No local proc handle — fail as interrupted.
                        job.state = FAILED
                        job.error = "interrupted: no local process handle on restart"
                    # Kaggle RUNNING: keep as RUNNING, poll status (v1 behavior).
                jobs[name] = job
            else:
                jobs[name] = self._new_job_from_batch(batch_job)

        return jobs

    @staticmethod
    def _fresh_jobs_from_batch(
        batch_jobs: list[dict[str, Any]]
    ) -> dict[str, Job]:
        """Create fresh pending Jobs from batch job dicts."""
        jobs: dict[str, Job] = {}
        for batch_job in batch_jobs:
            jobs[batch_job["name"]] = ParallelScheduler._new_job_from_batch(batch_job)
        return jobs

    @staticmethod
    def _new_job_from_batch(batch_job: dict[str, Any]) -> Job:
        provider = batch_job.get("provider", PROVIDER_KAGGLE)
        if provider == PROVIDER_COLAB:
            return Job(
                name=batch_job["name"],
                output_dir=str(batch_job["output_dir"]),
                script=batch_job.get("script", ""),
                arguments=list(batch_job.get("arguments", [])),
                timeout=batch_job.get("timeout"),
                provider=PROVIDER_COLAB,
                remote_id=batch_job["name"],
            )
        return Job(
            name=batch_job["name"],
            kernel_dir=str(batch_job["kernel_dir"]),
            output_dir=str(batch_job["output_dir"]),
            kernel_id=batch_job["kernel_id"],
            remote_id=batch_job["kernel_id"],
            provider=PROVIDER_KAGGLE,
        )

    # -- scheduling primitives ---------------------------------------------

    def _active_count(self, jobs: dict[str, Job]) -> int:
        return sum(1 for j in jobs.values() if j.state in ACTIVE_STATES)

    def _active_provider_count(
        self, jobs: dict[str, Job], provider: str
    ) -> int:
        return sum(
            1
            for j in jobs.values()
            if j.state in ACTIVE_STATES and j.provider == provider
        )

    # -- Kaggle submit/poll/collect (unchanged v1 behavior) ----------------

    def _submit(self, job: Job, push_timeout: float) -> None:
        """Push a kernel to Kaggle, transitioning pending -> submitting -> running."""
        job.state = SUBMITTING
        job.attempts += 1
        try:
            kernel_id = self.client.push(job.kernel_dir, timeout=push_timeout)
            if kernel_id:
                job.kernel_id = kernel_id
                job.remote_id = kernel_id
            job.submitted_at = self.clock()
            job.state = RUNNING
            job.error = None
        except Exception as exc:
            job.state = FAILED
            job.error = f"push failed: {exc}"

    def _poll(self, job: Job, job_timeout: float) -> None:
        """Poll a running Kaggle job, transitioning to collecting or failed."""
        now = self.clock()
        if job.submitted_at is not None:
            elapsed = now - job.submitted_at
            if elapsed > job_timeout:
                job.state = FAILED
                job.error = (
                    f"job timed out after {elapsed:.0f}s "
                    f"(limit {job_timeout:.0f}s)"
                )
                return
        try:
            status = self.client.status(job.kernel_id)
        except Exception as exc:
            job.error = f"status poll failed: {exc}"
            return
        job.error = None
        if status == STATUS_COMPLETE:
            job.completed_at = now
            job.state = COLLECTING
        elif status == STATUS_ERROR:
            job.completed_at = now
            job.state = FAILED
            job.error = "kernel reported error status"
        elif status in (STATUS_RUNNING, STATUS_QUEUED):
            pass
        else:
            job.state = FAILED
            job.error = f"unrecognized status: {status!r}"

    def _collect(self, job: Job) -> None:
        """Pull output for a completed Kaggle job, transitioning to succeeded/failed."""
        try:
            self.client.output(job.kernel_id, job.output_dir)
            job.collected_at = self.clock()
            job.state = SUCCEEDED
            job.error = None
        except Exception as exc:
            job.state = FAILED
            job.error = f"output collection failed: {exc}"

    # -- Colab submit/poll/collect -----------------------------------------

    def _submit_colab(
        self,
        job: Job,
        job_timeout: float,
        save_state: Callable[[], None] | None = None,
    ) -> None:
        """Launch a Colab session, transitioning pending -> submitting -> running.

        The optional *save_state* callback is invoked after the job is marked
        SUBMITTING (with ``remote_id`` already set) but **before** the Popen
        is launched.  This closes the crash race where a session is created
        on the backend but the state file has not yet recorded it — on
        restart, orphan recovery detects and stops the leaked session.
        """
        job.state = SUBMITTING
        job.attempts += 1
        job.admission_confirmed = False
        # Persist the SUBMITTING intent before launching the Popen so a
        # crash in the launch window is recoverable via orphan recovery.
        if save_state is not None:
            save_state()
        try:
            session_name = job.remote_id
            to = job.timeout if job.timeout is not None else job_timeout
            proc = self.colab_client.run(
                session_name,
                job.script,
                arguments=job.arguments or None,
                timeout=to,
            )
            job.proc = proc
            # Let the client track the proc for collect().
            self.colab_client.remember(session_name, proc)
            job.submitted_at = self.clock()
            job.state = RUNNING
            job.error = None
        except Exception as exc:
            err = f"colab run failed: {exc}"
            self._persist_colab_diagnostics(job, error=err)
            job.state = FAILED
            job.error = err

    def _poll_colab(self, job: Job, job_timeout: float, jobs: dict[str, Job]) -> None:
        """Poll a running Colab job's Popen.

        On capacity rejection: return job to pending, reduce AIMD limit to
        the account-wide active Colab count (min 1), set cooldown.
        On completion: transition to collecting.
        On error: transition to failed.
        On missing proc (restart): fail as interrupted.
        On first confirmed running poll: increment AIMD (delayed on_success).
        """
        now = self.clock()

        # Timeout check.
        if job.submitted_at is not None:
            elapsed = now - job.submitted_at
            effective_timeout = job.timeout if job.timeout is not None else job_timeout
            if elapsed > effective_timeout:
                err = (
                    f"job timed out after {elapsed:.0f}s "
                    f"(limit {effective_timeout:.0f}s)"
                )
                self._kill_proc(job)
                self._persist_colab_diagnostics(job, error=err)
                self._best_effort_stop(job)
                job.proc = None
                job.state = FAILED
                job.error = err
                return

        if job.proc is None:
            err = "interrupted: no local process handle on restart"
            self._persist_colab_diagnostics(job, error=err)
            job.state = FAILED
            job.error = err
            self._best_effort_stop(job)
            return

        rc = job.proc.poll()
        if rc is None:
            # Still running — session was created on the backend.  This is
            # the first confirmation of admission; increment AIMD once.
            if not job.admission_confirmed:
                job.admission_confirmed = True
                job.capacity_rejections = 0
                if self._colab_controller is not None:
                    self._colab_controller.on_success()
            return

        status = classify_colab_output(job.proc)

        if status == COLAB_CAPACITY_REJECTED:
            # Return to pending for retry after cooldown.  Stop the session
            # to free the VM slot (the --keep daemon may persist even on 412).
            # Capture the account-wide session count BEFORE stopping the
            # rejected session — stopping may drain sessions on the backend
            # (e.g. external sessions that were co-occupying capacity), and
            # AIMD reduction needs the count at rejection time (F2).
            try:
                active_colab = len(self.colab_client.sessions())
            except Exception:
                active_colab = self._active_provider_count(jobs, PROVIDER_COLAB)
            # If this rejection will exhaust the retry limit, persist
            # diagnostics BEFORE stop() cleans up temp files.
            new_rejection_count = job.capacity_rejections + 1
            will_fail = new_rejection_count >= COLAB_MAX_CAPACITY_REJECTIONS
            fail_err: str | None = None
            if will_fail:
                fail_err = (
                    f"colab job failed after {new_rejection_count} "
                    f"consecutive capacity rejections"
                )
                self._persist_colab_diagnostics(job, error=fail_err)
            self._best_effort_stop(job)
            job.proc = None
            job.admission_confirmed = False
            job.capacity_rejections = new_rejection_count
            # Reduce AIMD limit to the account-wide active Colab count
            # (including external sessions), not just scheduler-local jobs.
            # This prevents permanent under-admission when external sessions
            # occupy capacity at rejection time but later drain away.
            if self._colab_controller is not None:
                self._colab_controller.on_capacity_rejected(active_colab)
            self._colab_cooldown_until = now + self._colab_cooldown
            # Fail after too many consecutive capacity rejections to avoid
            # an infinite retry loop when external sessions permanently
            # occupy all account capacity.
            if will_fail:
                job.state = FAILED
                job.error = fail_err
            else:
                job.state = PENDING
                job.error = None
            return

        job.completed_at = now
        if status == COLAB_COMPLETE:
            job.state = COLLECTING
        else:
            exit_code = job.proc.returncode if job.proc is not None else None
            err = f"colab job failed: status={status}, exit_code={exit_code}"
            # Persist diagnostics BEFORE stop() cleans up temp files.
            self._persist_colab_diagnostics(job, error=err)
            # Stop the session on error to free the VM.
            self._best_effort_stop(job)
            job.state = FAILED
            job.error = err

    def _collect_colab(self, job: Job) -> None:
        """Collect Colab output and stop the session."""
        try:
            result = self.colab_client.collect(job.remote_id, job.output_dir)
            if result.get("ok"):
                job.collected_at = self.clock()
                job.state = SUCCEEDED
                job.error = None
            else:
                job.state = FAILED
                job.error = f"colab collection failed: {result.get('error')}"
        except Exception as exc:
            err = f"colab collection failed: {exc}"
            # collect() may have partially written files or cleaned up
            # temp files in its finally block.  Persist what we can —
            # cached proc output survives _cleanup_proc_tempfiles.
            self._persist_colab_diagnostics(job, error=err)
            job.state = FAILED
            job.error = err
        finally:
            self._best_effort_stop(job)
            job.proc = None

    def _kill_proc(self, job: Job) -> None:
        """Terminate and reap the local Colab Popen, swallowing errors.

        ``stop()`` unassigns the backend VM, but the local ``colab run`` CLI
        process may still be alive (blocked on pipe write, teardown, etc.).
        This ensures the OS reaps it promptly rather than leaking a process.
        """
        proc = job.proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    def _best_effort_stop(self, job: Job) -> None:
        """Best-effort stop a Colab session, swallowing errors."""
        if self.colab_client is None:
            return
        try:
            self.colab_client.stop(job.remote_id)
        except Exception:
            pass

    def _persist_colab_diagnostics(
        self, job: Job, *, error: str | None = None
    ) -> None:
        """Persist stdout/stderr/result.json to *job.output_dir* before terminal state.

        Writes diagnostics regardless of success or failure.  Does NOT stop
        the session, kill the proc, or clean up temp files — the caller
        retains responsibility for those lifecycle steps.  Safe to call
        with no proc (writes empty stdout/stderr and a result.json carrying
        the *error* message).

        When a proc is available, stdout/stderr are read from the cached
        output set by :func:`classify_colab_output` (if already classified)
        or directly from the proc's temp files / pipes via
        :func:`_read_proc_output`.  This ensures output is captured even
        for a still-running proc that was killed before classification.
        """
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stdout_text = ""
        stderr_text = ""
        exit_code: int | None = None
        status = COLAB_ERROR

        proc = job.proc
        if proc is not None:
            cached_stdout = getattr(proc, "_colab_stdout", None)
            cached_stderr = getattr(proc, "_colab_stderr", None)
            cached_status = getattr(proc, "_colab_classified_output", None)
            if cached_stdout is not None:
                stdout_text = cached_stdout
            else:
                stdout_text = _read_proc_output(proc, "stdout")
            if cached_stderr is not None:
                stderr_text = cached_stderr
            else:
                stderr_text = _read_proc_output(proc, "stderr")
            exit_code = proc.returncode
            if cached_status is not None:
                status = cached_status  # type: ignore[assignment]

        (out / "stdout.log").write_text(stdout_text, encoding="utf-8")
        (out / "stderr.log").write_text(stderr_text, encoding="utf-8")

        result_meta: dict[str, Any] = {
            "ok": status == COLAB_COMPLETE and error is None,
            "error": error if error is not None else (
                None if status == COLAB_COMPLETE else status
            ),
            "exit_code": exit_code,
            "status": status,
            "session": job.remote_id,
        }
        (out / "result.json").write_text(
            json.dumps(result_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # -- main loop ---------------------------------------------------------

    def run(
        self,
        batch_path: str | Path,
        state_path: str | Path,
        *,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        push_timeout: float = DEFAULT_PUSH_TIMEOUT,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
        kaggle_max: int = DEFAULT_KAGGLE_MAX,
        colab_max: int = DEFAULT_COLAB_MAX,
        colab_cooldown: float = DEFAULT_COLAB_COOLDOWN,
    ) -> dict[str, Any]:
        """Run the full scheduling loop until all jobs reach a terminal state.

        Returns a summary dict with per-job status, per-provider breakdown,
        and overall success flag.
        """
        if max_parallel < MIN_PARALLEL or max_parallel > HARD_MAX_PARALLEL:
            raise ValueError(
                f"max_parallel must be {MIN_PARALLEL}..{HARD_MAX_PARALLEL}, "
                f"got {max_parallel}"
            )
        if kaggle_max < MIN_KAGGLE_MAX or kaggle_max > HARD_KAGGLE_MAX:
            raise ValueError(
                f"kaggle_max must be {MIN_KAGGLE_MAX}..{HARD_KAGGLE_MAX}, "
                f"got {kaggle_max}"
            )
        if colab_max < MIN_COLAB_MAX:
            raise ValueError(
                f"colab_max must be >= {MIN_COLAB_MAX}, got {colab_max}"
            )

        batch_path_resolved = str(Path(batch_path).resolve())
        state_path_resolved = Path(state_path).resolve()

        batch_jobs = load_batch(batch_path_resolved)

        # Determine state schema from batch schema.
        batch_schema = (
            batch_jobs[0].get("schema_version", BATCH_SCHEMA_VERSION)
            if batch_jobs
            else BATCH_SCHEMA_VERSION
        )
        self._state_schema = (
            STATE_SCHEMA_V2 if batch_schema == BATCH_SCHEMA_V2 else STATE_SCHEMA_VERSION
        )

        jobs = self._load_or_init_state(
            state_path_resolved, batch_path_resolved, batch_jobs
        )

        # Set up Colab AIMD controller and restore learned limit from state.
        has_colab = any(j.provider == PROVIDER_COLAB for j in jobs.values())
        if has_colab:
            if self.colab_client is None:
                # Fail all pending Colab jobs — no client configured.
                for job in jobs.values():
                    if job.provider == PROVIDER_COLAB and job.state == PENDING:
                        self._persist_colab_diagnostics(
                            job, error="no colab client configured"
                        )
                        job.state = FAILED
                        job.error = "no colab client configured"
            else:
                raw = load_state(state_path_resolved)
                restored_limit = raw.get("colab_learned_limit")
                if restored_limit is not None and isinstance(restored_limit, int):
                    self._colab_controller = ColabAimdController(ceiling=colab_max)
                    self._colab_controller.learned_limit = max(
                        COLAB_AIMD_MIN, min(restored_limit, colab_max)
                    )
                else:
                    self._colab_controller = ColabAimdController(ceiling=colab_max)

        self._colab_cooldown = colab_cooldown
        self._colab_cooldown_until = 0.0

        # Best-effort stop for Colab jobs failed as interrupted on restart,
        # and recover orphaned sessions from SUBMITTING crash windows.
        if self.colab_client is not None:
            for job in jobs.values():
                if (
                    job.provider == PROVIDER_COLAB
                    and job.state == FAILED
                    and job.error
                    and "interrupted" in job.error
                ):
                    self._persist_colab_diagnostics(job, error=job.error)
                    self._best_effort_stop(job)
            # Detect and stop orphaned sessions from SUBMITTING crash
            # windows.  A job that was SUBMITTING when the scheduler crashed
            # is now PENDING (converted above), but its session may still
            # exist on the backend.  Include any non-fresh Colab job (one
            # loaded from the state file — active, interrupted, or terminal)
            # so detect_orphaned_sessions can flag sessions that exist on
            # the backend but should not.  Fresh pending jobs (never
            # submitted, created from the batch manifest) are excluded so
            # they do not claim or stop preexisting external sessions (F5).
            # The sessions() probe always runs once at restart so external
            # sessions can drain without false failure.  detect_orphaned_sessions
            # intersects known_names with the backend session list, so fresh
            # jobs (excluded from known_names) never cause external sessions
            # to be stopped (F5).  When known_names is empty (all fresh), the
            # probe returns no orphans but still counts as one call (F8).
            known_names = {
                job.remote_id
                for job in jobs.values()
                if job.provider == PROVIDER_COLAB
                and job._from_state
                and job.remote_id
            }
            try:
                orphans = detect_orphaned_sessions(
                    self.colab_client, known_names
                )
                for orphan_name in orphans:
                    try:
                        self.colab_client.stop(orphan_name)
                    except Exception:
                        pass
            except Exception:
                pass

        self._save_state(state_path_resolved, batch_path_resolved, jobs)

        while True:
            # Phase 1: submit pending jobs into available slots.
            active = self._active_count(jobs)
            # Cache the account-wide Colab session count once per loop
            # iteration — sessions() spawns a subprocess with a 30s timeout,
            # so calling it per pending Colab job is wasteful (F8).
            cached_external_active: int | None = None
            for job in jobs.values():
                if active >= max_parallel:
                    break
                if job.state != PENDING:
                    continue
                if job.provider == PROVIDER_KAGGLE:
                    if self._active_provider_count(jobs, PROVIDER_KAGGLE) >= kaggle_max:
                        continue
                    self._submit(job, push_timeout)
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)
                    if job.state in ACTIVE_STATES:
                        active += 1
                elif job.provider == PROVIDER_COLAB:
                    if self.colab_client is None:
                        continue  # already failed above
                    # Cooldown check.
                    if self.clock() < self._colab_cooldown_until:
                        continue
                    # External session capacity accounting (cached).
                    if cached_external_active is None:
                        try:
                            cached_external_active = len(self.colab_client.sessions())
                        except Exception:
                            cached_external_active = self._active_provider_count(
                                jobs, PROVIDER_COLAB
                            )
                    effective_limit = (
                        self._colab_controller.effective_limit()
                        if self._colab_controller
                        else colab_max
                    )
                    if cached_external_active >= effective_limit:
                        # Admission gate blocked — account is at capacity.
                        # This is normal queueing, not a rejection: keep the
                        # job pending and wait for a slot to free.  Only
                        # actual 412 TooManyAssignments rejections (handled
                        # in _poll_colab) count toward capacity_rejections.
                        continue
                    # Pass a save callback so _submit_colab can persist the
                    # SUBMITTING intent BEFORE launching the Popen (F7).
                    # on_success is deferred to the first confirmed running
                    # poll (F9), not called here.
                    self._submit_colab(
                        job,
                        job_timeout,
                        save_state=lambda: self._save_state(
                            state_path_resolved, batch_path_resolved, jobs
                        ),
                    )
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)
                    if job.state in ACTIVE_STATES:
                        active += 1
                        # Refresh the cached count after a successful submit
                        # so the next pending Colab job sees the updated
                        # account-wide count.
                        cached_external_active = None

            # Phase 2: poll running jobs.
            for job in list(jobs.values()):
                if job.state == RUNNING:
                    if job.provider == PROVIDER_KAGGLE:
                        self._poll(job, job_timeout)
                    elif job.provider == PROVIDER_COLAB:
                        self._poll_colab(job, job_timeout, jobs)
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)

            # Phase 3: collect completed jobs.
            for job in list(jobs.values()):
                if job.state == COLLECTING:
                    if job.provider == PROVIDER_KAGGLE:
                        self._collect(job)
                    elif job.provider == PROVIDER_COLAB:
                        self._collect_colab(job)
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)

            # Check if we're done.
            active = self._active_count(jobs)
            pending = sum(1 for j in jobs.values() if j.state == PENDING)
            if active == 0 and pending == 0:
                break

            if active > 0:
                self.sleeper(poll_interval)
            elif pending > 0:
                self.sleeper(poll_interval)

        self._save_state(state_path_resolved, batch_path_resolved, jobs)
        return self._summary(jobs)

    def _summary(
        self,
        jobs: dict[str, Job],
        *,
        colab_learned_limit: int | None = None,
    ) -> dict[str, Any]:
        succeeded = sum(1 for j in jobs.values() if j.state == SUCCEEDED)
        failed = sum(1 for j in jobs.values() if j.state == FAILED)
        total = len(jobs)
        kaggle_jobs = {n: j for n, j in jobs.items() if j.provider == PROVIDER_KAGGLE}
        colab_jobs = {n: j for n, j in jobs.items() if j.provider == PROVIDER_COLAB}
        kaggle_succeeded = sum(1 for j in kaggle_jobs.values() if j.state == SUCCEEDED)
        kaggle_failed = sum(1 for j in kaggle_jobs.values() if j.state == FAILED)
        colab_succeeded = sum(1 for j in colab_jobs.values() if j.state == SUCCEEDED)
        colab_failed = sum(1 for j in colab_jobs.values() if j.state == FAILED)
        effective_colab_limit = (
            colab_learned_limit
            if colab_learned_limit is not None
            else (
                self._colab_controller.learned_limit
                if self._colab_controller is not None
                else None
            )
        )
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "all_succeeded": failed == 0 and succeeded == total,
            "providers": {
                "kaggle": {
                    "total": len(kaggle_jobs),
                    "succeeded": kaggle_succeeded,
                    "failed": kaggle_failed,
                },
                "colab": {
                    "total": len(colab_jobs),
                    "succeeded": colab_succeeded,
                    "failed": colab_failed,
                    "learned_limit": effective_colab_limit,
                },
            },
            "jobs": {
                name: {
                    "state": job.state,
                    "error": job.error,
                    "kernel_id": job.kernel_id,
                    "attempts": job.attempts,
                    "provider": job.provider,
                    "remote_id": job.remote_id,
                }
                for name, job in jobs.items()
            },
        }

    def get_status(
        self, batch_path: str | Path, state_path: str | Path
    ) -> dict[str, Any]:
        """Return a read-only status summary without running the scheduler."""
        batch_path_resolved = str(Path(batch_path).resolve())
        state_path_resolved = Path(state_path).resolve()
        batch_jobs = load_batch(batch_path_resolved)
        jobs = self._load_or_init_state(state_path_resolved, batch_path_resolved, batch_jobs)
        raw = load_state(state_path_resolved)
        colab_limit = raw.get("colab_learned_limit")
        return self._summary(jobs, colab_learned_limit=colab_limit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallel_scheduler",
        description="Durable CPU-only Kaggle/Colab parallel experiment scheduler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a batch of remote jobs.")
    run_p.add_argument("batch", help="Path to batch manifest JSON.")
    run_p.add_argument(
        "--state",
        required=True,
        help="Path to durable state file (created/updated during run).",
    )
    run_p.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Max concurrent remote jobs globally (default {DEFAULT_MAX_PARALLEL}, "
        f"max {HARD_MAX_PARALLEL}).",
    )
    run_p.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status polls (default {DEFAULT_POLL_INTERVAL}).",
    )
    run_p.add_argument(
        "--push-timeout",
        type=float,
        default=DEFAULT_PUSH_TIMEOUT,
        help=f"Kaggle kernel run-time limit in seconds (default {DEFAULT_PUSH_TIMEOUT}).",
    )
    run_p.add_argument(
        "--job-timeout",
        type=float,
        default=DEFAULT_JOB_TIMEOUT,
        help=f"Timeout for a running job in seconds (default {DEFAULT_JOB_TIMEOUT}).",
    )
    run_p.add_argument(
        "--kaggle-max",
        type=int,
        default=DEFAULT_KAGGLE_MAX,
        help=f"Max concurrent Kaggle jobs (default {DEFAULT_KAGGLE_MAX}, "
        f"max {HARD_KAGGLE_MAX}).",
    )
    run_p.add_argument(
        "--colab-max",
        type=int,
        default=DEFAULT_COLAB_MAX,
        help=f"Colab AIMD ceiling (default {DEFAULT_COLAB_MAX}).",
    )
    run_p.add_argument(
        "--colab-cooldown",
        type=float,
        default=DEFAULT_COLAB_COOLDOWN,
        help=f"Colab capacity-rejection cooldown in seconds "
        f"(default {DEFAULT_COLAB_COOLDOWN}).",
    )

    status_p = sub.add_parser("status", help="Print status summary for a batch.")
    status_p.add_argument("batch", help="Path to batch manifest JSON.")
    status_p.add_argument(
        "--state",
        required=True,
        help="Path to durable state file.",
    )

    return parser


def main(
    argv: list[str] | None = None,
    *,
    client: KaggleClient | None = None,
    colab_client: Any = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """CLI entry point.

    *client* lets tests inject a fake Kaggle adapter.  *colab_client* lets
    tests inject a fake Colab adapter.  When either is ``None``, a real
    client is constructed from CLI arguments / defaults.

    Prints a JSON summary and returns 0 if all jobs succeeded, 1 if any
    failed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.max_parallel < MIN_PARALLEL or args.max_parallel > HARD_MAX_PARALLEL:
            parser.error(
                f"--max-parallel must be {MIN_PARALLEL}..{HARD_MAX_PARALLEL}"
            )
        if args.kaggle_max < MIN_KAGGLE_MAX or args.kaggle_max > HARD_KAGGLE_MAX:
            parser.error(
                f"--kaggle-max must be {MIN_KAGGLE_MAX}..{HARD_KAGGLE_MAX}"
            )
        if args.colab_max < MIN_COLAB_MAX:
            parser.error(f"--colab-max must be >= {MIN_COLAB_MAX}")

        cli_client = client or KaggleCliClient(
            push_timeout=args.push_timeout,
            status_timeout=DEFAULT_STATUS_TIMEOUT,
            output_timeout=DEFAULT_OUTPUT_TIMEOUT,
        )

        # Construct Colab client if needed and not injected.
        cli_colab_client = colab_client
        if cli_colab_client is None and _COLAB_AVAILABLE:
            # Peek at the batch to see if Colab jobs exist.
            try:
                batch_jobs = load_batch(args.batch)
                has_colab = any(
                    j.get("provider") == PROVIDER_COLAB for j in batch_jobs
                )
            except Exception:
                has_colab = False
            if has_colab:
                cli_colab_client = ColabCliClient()

        scheduler = ParallelScheduler(
            cli_client,
            colab_client=cli_colab_client,
            clock=clock,
            sleeper=sleeper,
        )
        summary = scheduler.run(
            args.batch,
            args.state,
            max_parallel=args.max_parallel,
            poll_interval=args.poll_interval,
            push_timeout=args.push_timeout,
            job_timeout=args.job_timeout,
            kaggle_max=args.kaggle_max,
            colab_max=args.colab_max,
            colab_cooldown=args.colab_cooldown,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["all_succeeded"] else 1

    if args.command == "status":
        cli_client = client or KaggleCliClient()
        scheduler = ParallelScheduler(cli_client, clock=clock, sleeper=sleeper)
        summary = scheduler.get_status(args.batch, args.state)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["all_succeeded"] else 1

    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
