"""Durable CPU-only Kaggle parallel experiment runtime and scheduler.

Runs many private, CPU-only Kaggle kernels in bounded parallelism with
crash-safe state, resume semantics, output collection, and timeout handling.

Batch manifest schema (``oczy/kaggle-parallel-batch/v1``)::

    {
      "schema_version": "oczy/kaggle-parallel-batch/v1",
      "jobs": [
        {"name": "...", "kernel_dir": "...", "output_dir": "..."}
      ]
    }

Durable state schema (``oczy/kaggle-parallel-state/v1``)::

    {
      "schema_version": "oczy/kaggle-parallel-state/v1",
      "batch_path": "...",
      "jobs": {
        "job-name": {
          "name": "...", "kernel_id": "...", "state": "...",
          "submitted_at": null, "completed_at": null,
          "collected_at": null, "error": null, "attempts": 0,
          "kernel_dir": "...", "output_dir": "..."
        }
      },
      "updated_at": 0.0
    }

Lifecycle: ``pending -> submitting -> running -> collecting -> succeeded``
with terminal ``failed``.  Resume converts interrupted ``submitting`` to
``pending`` and ``collecting`` to ``running`` safely, and never resubmits a
kernel already recorded as ``running``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

BATCH_SCHEMA_VERSION = "oczy/kaggle-parallel-batch/v1"
STATE_SCHEMA_VERSION = "oczy/kaggle-parallel-state/v1"

DEFAULT_MAX_PARALLEL = 8
HARD_MAX_PARALLEL = 10
MIN_PARALLEL = 1
DEFAULT_POLL_INTERVAL = 30
DEFAULT_PUSH_TIMEOUT = 21600
DEFAULT_JOB_TIMEOUT = 21600
DEFAULT_STATUS_TIMEOUT = 120
DEFAULT_OUTPUT_TIMEOUT = 1800
DEFAULT_PUSH_SUBPROCESS_TIMEOUT = 600

# Job lifecycle states.
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
    """Internal job representation with durable lifecycle state."""

    name: str
    kernel_dir: str
    output_dir: str
    kernel_id: str = ""
    state: str = PENDING
    error: str | None = None
    submitted_at: float | None = None
    completed_at: float | None = None
    collected_at: float | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            name=data["name"],
            kernel_dir=data.get("kernel_dir", ""),
            output_dir=data.get("output_dir", ""),
            kernel_id=data.get("kernel_id", ""),
            state=data.get("state", PENDING),
            error=data.get("error"),
            submitted_at=data.get("submitted_at"),
            completed_at=data.get("completed_at"),
            collected_at=data.get("collected_at"),
            attempts=data.get("attempts", 0),
        )


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


def load_batch(batch_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a batch manifest.

    Returns a list of job dicts with keys:
        name, kernel_dir (Path), output_dir (Path),
        kernel_id (str), metadata (dict), job_spec (dict).

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

    if manifest.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise BatchValidationError(
            f"unsupported batch schema_version: "
            f"{manifest.get('schema_version')!r}, expected {BATCH_SCHEMA_VERSION!r}"
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
        if kernel_id in seen_ids:
            raise BatchValidationError(
                f"job {name!r}: duplicate kernel id {kernel_id!r}"
            )
        seen_ids.add(kernel_id)

        jobs.append(
            {
                "name": name,
                "kernel_dir": kernel_dir,
                "output_dir": output_dir,
                "kernel_id": kernel_id,
                "metadata": metadata,
                "job_spec": job_spec,
            }
        )

    return jobs


# ---------------------------------------------------------------------------
# Durable state
# ---------------------------------------------------------------------------


def _state_to_dict(batch_path: str, jobs: dict[str, Job], now: float) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "batch_path": str(batch_path),
        "jobs": {name: job.to_dict() for name, job in jobs.items()},
        "updated_at": now,
    }


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
    return json.loads(state_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ParallelScheduler:
    """Bounded-parallel Kaggle kernel scheduler with durable state.

    Concurrency model: at most ``max_parallel`` remote jobs are active at
    once.  The scheduler is single-threaded and synchronous — Kaggle
    submissions are asynchronous remote work, so parallelism means N remote
    slots, not N local threads.

    Constructor injects the Kaggle client, clock, and sleeper for
    deterministic testing.  The default clock is wall-clock time
    (``time.time``) so persisted timestamps survive process/host restart;
    tests may inject ``time.monotonic`` for deterministic values.  Per-run
    parameters (max_parallel, timeouts) are passed to :meth:`run`.
    """

    def __init__(
        self,
        client: KaggleClient,
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.clock = clock
        self.sleeper = sleeper

    # -- state persistence -------------------------------------------------

    def _save_state(
        self, state_path: Path, batch_path: str, jobs: dict[str, Job]
    ) -> None:
        _atomic_write_json(
            state_path,
            _state_to_dict(batch_path, jobs, self.clock()),
        )

    def _load_or_init_state(
        self, state_path: Path, batch_path: str, batch_jobs: list[dict[str, Any]]
    ) -> dict[str, Job]:
        """Load durable state and merge with batch jobs.

        New batch jobs not in state are added as pending.  Existing state
        jobs are kept (preserving lifecycle progress).  Interrupted states
        are resolved for resume.  A nonempty persisted ``batch_path`` that
        differs from the requested batch raises ``BatchValidationError``.
        """
        raw = load_state(state_path)
        if raw.get("schema_version") != STATE_SCHEMA_VERSION:
            jobs = {
                j["name"]: Job(
                    name=j["name"],
                    kernel_dir=str(j["kernel_dir"]),
                    output_dir=str(j["output_dir"]),
                    kernel_id=j["kernel_id"],
                )
                for j in batch_jobs
            }
            return jobs

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
            if name in existing:
                stored = existing[name]
                job = Job.from_dict(stored)
                # Update paths from batch in case they moved.
                job.kernel_dir = str(batch_job["kernel_dir"])
                job.output_dir = str(batch_job["output_dir"])
                job.kernel_id = batch_job["kernel_id"]
                # Resume: interrupted submitting -> pending, collecting -> running.
                if job.state == SUBMITTING:
                    job.state = PENDING
                elif job.state == COLLECTING:
                    job.state = RUNNING
                jobs[name] = job
            else:
                jobs[name] = Job(
                    name=name,
                    kernel_dir=str(batch_job["kernel_dir"]),
                    output_dir=str(batch_job["output_dir"]),
                    kernel_id=batch_job["kernel_id"],
                )

        return jobs

    # -- scheduling primitives ---------------------------------------------

    def _active_count(self, jobs: dict[str, Job]) -> int:
        return sum(1 for j in jobs.values() if j.state in ACTIVE_STATES)

    def _submit(self, job: Job, push_timeout: float) -> None:
        """Push a kernel to Kaggle, transitioning pending -> submitting -> running."""
        job.state = SUBMITTING
        job.attempts += 1
        try:
            kernel_id = self.client.push(job.kernel_dir, timeout=push_timeout)
            if kernel_id:
                job.kernel_id = kernel_id
            job.submitted_at = self.clock()
            job.state = RUNNING
            job.error = None
        except Exception as exc:
            job.state = FAILED
            job.error = f"push failed: {exc}"

    def _poll(self, job: Job, job_timeout: float) -> None:
        """Poll a running job, transitioning to collecting or failed."""
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
        """Pull output for a completed job, transitioning to succeeded/failed."""
        try:
            self.client.output(job.kernel_id, job.output_dir)
            job.collected_at = self.clock()
            job.state = SUCCEEDED
            job.error = None
        except Exception as exc:
            job.state = FAILED
            job.error = f"output collection failed: {exc}"

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
    ) -> dict[str, Any]:
        """Run the full scheduling loop until all jobs reach a terminal state.

        Returns a summary dict with per-job status and overall success flag.
        """
        if max_parallel < MIN_PARALLEL or max_parallel > HARD_MAX_PARALLEL:
            raise ValueError(
                f"max_parallel must be {MIN_PARALLEL}..{HARD_MAX_PARALLEL}, "
                f"got {max_parallel}"
            )

        batch_path_resolved = str(Path(batch_path).resolve())
        state_path_resolved = Path(state_path).resolve()

        batch_jobs = load_batch(batch_path_resolved)
        jobs = self._load_or_init_state(state_path_resolved, batch_path_resolved, batch_jobs)
        self._save_state(state_path_resolved, batch_path_resolved, jobs)

        while True:
            # Phase 1: submit pending jobs into available slots.
            active = self._active_count(jobs)
            for job in jobs.values():
                if active >= max_parallel:
                    break
                if job.state == PENDING:
                    self._submit(job, push_timeout)
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)
                    if job.state in ACTIVE_STATES:
                        active += 1

            # Phase 2: poll running jobs.
            for job in list(jobs.values()):
                if job.state == RUNNING:
                    self._poll(job, job_timeout)
                    self._save_state(state_path_resolved, batch_path_resolved, jobs)

            # Phase 3: collect completed jobs.
            for job in list(jobs.values()):
                if job.state == COLLECTING:
                    self._collect(job)
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

    def _summary(self, jobs: dict[str, Job]) -> dict[str, Any]:
        succeeded = sum(1 for j in jobs.values() if j.state == SUCCEEDED)
        failed = sum(1 for j in jobs.values() if j.state == FAILED)
        total = len(jobs)
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "all_succeeded": failed == 0 and succeeded == total,
            "jobs": {
                name: {
                    "state": job.state,
                    "error": job.error,
                    "kernel_id": job.kernel_id,
                    "attempts": job.attempts,
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
        return self._summary(jobs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallel_scheduler",
        description="Durable CPU-only Kaggle parallel experiment scheduler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a batch of Kaggle kernels.")
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
        help=f"Max concurrent remote jobs (default {DEFAULT_MAX_PARALLEL}, "
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
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """CLI entry point.

    *client* lets tests inject a fake Kaggle adapter.  When ``None``, a real
    :class:`KaggleCliClient` is constructed from CLI arguments.

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
        cli_client = client or KaggleCliClient(
            push_timeout=args.push_timeout,
            status_timeout=DEFAULT_STATUS_TIMEOUT,
            output_timeout=DEFAULT_OUTPUT_TIMEOUT,
        )
        scheduler = ParallelScheduler(cli_client, clock=clock, sleeper=sleeper)
        summary = scheduler.run(
            args.batch,
            args.state,
            max_parallel=args.max_parallel,
            poll_interval=args.poll_interval,
            push_timeout=args.push_timeout,
            job_timeout=args.job_timeout,
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
