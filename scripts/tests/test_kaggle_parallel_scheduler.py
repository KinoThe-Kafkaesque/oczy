"""High-signal contract tests for the durable CPU-only Kaggle parallel scheduler.

These tests exercise the *behavior* specified in the shared contract:

* Manifest loading and validation (schema, relative paths, CPU-only rejection,
  duplicate name/id detection, profile/cpu checks).
* Concurrency bounds: default 8, hard cap 10, minimum 1, bounded active jobs.
* Durable state transitions: pending -> submitting -> running -> collecting ->
  succeeded, with terminal failed; resume converts interrupted
  submitting/collecting to pending/running and never resubmits a recorded
  running kernel.
* Mixed success/failure runs with correct CLI exit code and JSON summary.
* Status string classification (complete/error/running/queued).
* Timeout handling for push and job execution.
* Output collection on completion.
* Atomic, resumable state file written after each transition.
* CLI ``run`` and ``status`` subcommands with JSON output and exit codes.

All tests use injected fake Kaggle clients and deterministic clock/sleep
functions — never the network or real Kaggle CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module import — the implementation lives at infrastructure/kaggle/parallel_scheduler.py
# and is loaded via runpy so tests work without an installed package.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "parallel_scheduler.py"


def _load_scheduler_module() -> dict[str, Any]:
    """Import the scheduler module, skipping the entire file if it doesn't
    exist yet (implementation may land concurrently)."""
    import runpy

    if not SCHEDULER_PATH.exists():
        pytest.skip(
            f"implementation not found at {SCHEDULER_PATH}",
            allow_module_level=True,
        )
    return runpy.run_path(str(SCHEDULER_PATH))


_mod = _load_scheduler_module()

BATCH_SCHEMA_VERSION: str = _mod["BATCH_SCHEMA_VERSION"]
STATE_SCHEMA_VERSION: str = _mod["STATE_SCHEMA_VERSION"]
BATCH_SCHEMA_V2: str = _mod["BATCH_SCHEMA_V2"]
STATE_SCHEMA_V2: str = _mod["STATE_SCHEMA_V2"]
classify_status = _mod["classify_status"]
load_batch = _mod["load_batch"]
BatchValidationError = _mod["BatchValidationError"]
ParallelScheduler = _mod["ParallelScheduler"]
main = _mod["main"]
KaggleCliClient = _mod["KaggleCliClient"]

PROVIDER_KAGGLE: str = _mod["PROVIDER_KAGGLE"]
PROVIDER_COLAB: str = _mod["PROVIDER_COLAB"]
DEFAULT_KAGGLE_MAX: int = _mod["DEFAULT_KAGGLE_MAX"]
HARD_KAGGLE_MAX: int = _mod["HARD_KAGGLE_MAX"]
DEFAULT_COLAB_MAX: int = _mod["DEFAULT_COLAB_MAX"]
ColabAimdController = _mod["ColabAimdController"]

PENDING = _mod.get("PENDING", "pending")
SUBMITTING = _mod.get("SUBMITTING", "submitting")
RUNNING = _mod.get("RUNNING", "running")
COLLECTING = _mod.get("COLLECTING", "collecting")
SUCCEEDED = _mod.get("SUCCEEDED", "succeeded")
FAILED = _mod.get("FAILED", "failed")

EXPECTED_BATCH_SCHEMA = "oczy/kaggle-parallel-batch/v1"
EXPECTED_STATE_SCHEMA = "oczy/kaggle-parallel-state/v1"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _title_from_kernel_id(kernel_id: str) -> str:
    """Derive a title whose Kaggle clean-URL slug equals the id's final component.

    Kaggle lowercases the title and collapses each run of non-ASCII-
    alphanumeric characters to a single hyphen, stripping edge hyphens.
    Splitting the id slug on hyphens and title-casing each word round-trips
    back to the same slug.
    """
    slug = kernel_id.rsplit("/", 1)[-1]
    return " ".join(word.capitalize() for word in slug.split("-"))


def _valid_metadata(
    kernel_id: str = "owner/test-kernel-a",
    *,
    is_private: bool = True,
    enable_gpu: bool = False,
    enable_tpu: bool = False,
    enable_internet: bool = False,
    machine_shape: str = "",
) -> dict[str, Any]:
    return {
        "id": kernel_id,
        "title": _title_from_kernel_id(kernel_id),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": is_private,
        "enable_gpu": enable_gpu,
        "enable_tpu": enable_tpu,
        "enable_internet": enable_internet,
        "machine_shape": machine_shape,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _valid_job_spec(profile: str = "cpu") -> dict[str, Any]:
    return {
        "schema_version": "oczy/kaggle-research-job/v1",
        "phase": "development",
        "profile": profile,
        "source_dataset": "owner/test-source",
        "source_commit": "a" * 40,
        "source_archive_sha256": "b" * 64,
        "model_source": None,
        "module": "run_cortex_smoke",
        "arguments": [],
        "instrument_manifest_sha256": None,
        "human_signoff_id": None,
    }


def _make_kernel_dir(
    base: Path,
    name: str,
    kernel_id: str = "owner/test-kernel-a",
    *,
    metadata: dict[str, Any] | None = None,
    job_spec: dict[str, Any] | None = None,
) -> Path:
    """Create a kernel directory with metadata and job_spec."""
    kd = base / name
    kd.mkdir(parents=True, exist_ok=True)
    meta = metadata if metadata is not None else _valid_metadata(kernel_id)
    spec = job_spec if job_spec is not None else _valid_job_spec()
    (kd / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (kd / "job_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return kd


def _make_batch(
    base: Path,
    jobs: list[dict[str, Any]],
    *,
    schema_version: str | None = EXPECTED_BATCH_SCHEMA,
) -> Path:
    """Write a batch manifest. Each job dict should have name, kernel_dir,
    output_dir (paths relative to base). Returns the manifest path."""
    manifest = {
        "schema_version": schema_version or EXPECTED_BATCH_SCHEMA,
        "jobs": jobs,
    }
    p = base / "batch.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def _make_batch_with_kernels(
    base: Path,
    count: int,
    *,
    prefix: str = "job",
    kernel_prefix: str = "owner/k",
) -> tuple[Path, list[dict[str, Any]]]:
    """Create ``count`` kernel dirs and a batch manifest referencing them.
    Returns (manifest_path, job_manifest_entries)."""
    entries: list[dict[str, Any]] = []
    for i in range(count):
        name = f"{prefix}{i}"
        kid = f"{kernel_prefix}{i}"
        _make_kernel_dir(base, name, kid)
        entries.append(
            {"name": name, "kernel_dir": name, "output_dir": f"out/{name}"}
        )
    manifest = _make_batch(base, entries)
    return manifest, entries


class FakeKaggleClient:
    """Deterministic fake Kaggle client for testing.

    Each kernel_id is configured with a sequence of status responses followed
    by a terminal status.  ``output`` writes a marker file into the output dir.
    """

    def __init__(
        self,
        *,
        status_sequences: dict[str, list[str]] | None = None,
        push_results: dict[str, str] | None = None,
        output_files: dict[str, dict[str, str]] | None = None,
        push_raises: dict[str, Exception] | None = None,
        output_raises: dict[str, Exception] | None = None,
    ) -> None:
        self._status_sequences = status_sequences or {}
        self._push_results = push_results or {}
        self._output_files = output_files or {}
        self._push_raises = push_raises or {}
        self._output_raises = output_raises or {}
        self.push_calls: list[tuple[str, float | None]] = []
        self.status_calls: list[tuple[str, float | None]] = []
        self.output_calls: list[tuple[str, str, float | None]] = []

    def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
        self.push_calls.append((kernel_dir, timeout))
        # Determine kernel_id from the last component or metadata
        kd = Path(kernel_dir)
        meta_path = kd / "kernel-metadata.json"
        kid = "unknown"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            kid = meta["id"]
        if kid in self._push_raises:
            raise self._push_raises[kid]
        return self._push_results.get(kid, kid)

    def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
        self.status_calls.append((kernel_id, timeout))
        seq = self._status_sequences.get(kernel_id)
        if seq:
            if len(seq) > 1:
                return seq.pop(0)
            return seq[0]
        return "complete"

    def output(self, kernel_id: str, output_dir: str, *, timeout: float | None = None) -> None:
        self.output_calls.append((kernel_id, output_dir, timeout))
        if kernel_id in self._output_raises:
            raise self._output_raises[kernel_id]
        files = self._output_files.get(kernel_id, {"result.json": '{"ok": true}'})
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (p / fname).write_text(content, encoding="utf-8")


class CountingSleeper:
    """Sleeper that records calls and advances a fake clock."""

    def __init__(self, clock: list[float], interval: float = 30.0) -> None:
        self._clock = clock
        self._interval = interval
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock[0] += seconds


def _make_clock(start: float = 1000.0) -> tuple[list[float], Any]:
    """Return (mutable clock holder, clock function)."""
    holder = [start]

    def _now() -> float:
        return holder[0]

    return holder, _now


# ---------------------------------------------------------------------------
# 1. Schema version constants
# ---------------------------------------------------------------------------


def test_batch_schema_version_matches_contract() -> None:
    assert BATCH_SCHEMA_VERSION == EXPECTED_BATCH_SCHEMA


def test_state_schema_version_matches_contract() -> None:
    assert STATE_SCHEMA_VERSION == EXPECTED_STATE_SCHEMA


# ---------------------------------------------------------------------------
# 2. Manifest validation
# ---------------------------------------------------------------------------


def test_load_batch_valid_manifest(tmp_path: Path) -> None:
    """A well-formed batch with valid CPU kernels loads successfully."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    jobs = load_batch(manifest)
    assert len(jobs) == 2
    for job in jobs:
        assert "name" in job
        assert "kernel_dir" in job
        assert "output_dir" in job
        assert "kernel_id" in job
        assert "metadata" in job
        assert "job_spec" in job
        assert isinstance(job["kernel_dir"], Path)
        assert isinstance(job["output_dir"], Path)
        assert job["metadata"]["is_private"] is True
        assert job["job_spec"]["profile"] == "cpu"


def test_load_batch_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A manifest with the wrong schema_version must be rejected."""
    _make_batch_with_kernels(tmp_path, 1)
    manifest = tmp_path / "batch.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = "wrong/schema/v2"
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_missing_schema_version(tmp_path: Path) -> None:
    """A manifest without schema_version must be rejected."""
    _make_batch_with_kernels(tmp_path, 1)
    manifest = tmp_path / "batch.json"
    data = json.loads(manifest.read_text())
    del data["schema_version"]
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_empty_jobs(tmp_path: Path) -> None:
    """An empty jobs list must be rejected."""
    manifest = _make_batch(tmp_path, [])
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_job_missing_name(tmp_path: Path) -> None:
    """Each job entry must have a name."""
    _make_kernel_dir(tmp_path, "k1", "owner/k1")
    manifest = _make_batch(
        tmp_path,
        [{"kernel_dir": "k1", "output_dir": "out/k1"}],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_job_missing_kernel_dir(tmp_path: Path) -> None:
    """Each job entry must have a kernel_dir."""
    _make_kernel_dir(tmp_path, "k1", "owner/k1")
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "output_dir": "out/j1"}],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_job_missing_output_dir(tmp_path: Path) -> None:
    """Each job entry must have an output_dir."""
    _make_kernel_dir(tmp_path, "k1", "owner/k1")
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1"}],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


# ---------------------------------------------------------------------------
# 3. Relative path resolution
# ---------------------------------------------------------------------------


def test_load_batch_resolves_paths_relative_to_manifest(tmp_path: Path) -> None:
    """kernel_dir and output_dir must resolve relative to the manifest's
    directory, not the current working directory."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    _make_kernel_dir(sub, "k1", "owner/k1")
    manifest = sub / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_BATCH_SCHEMA,
                "jobs": [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
            }
        )
    )
    jobs = load_batch(manifest)
    assert jobs[0]["kernel_dir"] == (sub / "k1").resolve()
    assert jobs[0]["output_dir"] == (sub / "out" / "j1").resolve()
    assert jobs[0]["kernel_dir"].exists()
    assert (sub / "k1" / "kernel-metadata.json").exists()


def test_load_batch_resolves_absolute_paths(tmp_path: Path) -> None:
    """Absolute paths in the manifest should also work."""
    kd = _make_kernel_dir(tmp_path, "k1", "owner/k1")
    out = tmp_path / "abs_out"
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": str(kd), "output_dir": str(out)}],
    )
    jobs = load_batch(manifest)
    assert jobs[0]["kernel_dir"] == kd.resolve()
    assert jobs[0]["output_dir"] == out.resolve()


# ---------------------------------------------------------------------------
# 4. CPU-only rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_gpu", True),
        ("enable_tpu", True),
        ("enable_internet", True),
        ("is_private", False),
        ("machine_shape", "gpu"),
    ],
)
def test_load_batch_rejects_non_cpu_metadata(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Each metadata field that violates the CPU-only/private/offline contract
    must be rejected."""
    meta = _valid_metadata()
    meta[field] = value
    _make_kernel_dir(tmp_path, "k1", "owner/k1", metadata=meta)
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_non_cpu_profile(tmp_path: Path) -> None:
    """job_spec with profile != 'cpu' must be rejected."""
    _make_kernel_dir(
        tmp_path, "k1", "owner/k1", job_spec=_valid_job_spec(profile="t4")
    )
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_missing_metadata_file(tmp_path: Path) -> None:
    """A kernel dir without kernel-metadata.json must be rejected."""
    kd = tmp_path / "k1"
    kd.mkdir()
    (kd / "job_spec.json").write_text(json.dumps(_valid_job_spec()))
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        load_batch(manifest)


def test_load_batch_rejects_missing_job_spec_file(tmp_path: Path) -> None:
    """A kernel dir without job_spec.json must be rejected."""
    kd = tmp_path / "k1"
    kd.mkdir()
    (kd / "kernel-metadata.json").write_text(
        json.dumps(_valid_metadata("owner/k1"))
    )
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        load_batch(manifest)


# ---------------------------------------------------------------------------
# 5. Duplicate names and kernel IDs
# ---------------------------------------------------------------------------


def test_load_batch_rejects_duplicate_names(tmp_path: Path) -> None:
    """Two jobs with the same name must be rejected."""
    _make_kernel_dir(tmp_path, "k1", "owner/k1")
    _make_kernel_dir(tmp_path, "k2", "owner/k2")
    manifest = _make_batch(
        tmp_path,
        [
            {"name": "same", "kernel_dir": "k1", "output_dir": "out/a"},
            {"name": "same", "kernel_dir": "k2", "output_dir": "out/b"},
        ],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


def test_load_batch_rejects_duplicate_kernel_ids(tmp_path: Path) -> None:
    """Two jobs whose kernels share the same kernel id must be rejected."""
    _make_kernel_dir(tmp_path, "k1", "owner/dup")
    _make_kernel_dir(tmp_path, "k2", "owner/dup")
    manifest = _make_batch(
        tmp_path,
        [
            {"name": "j1", "kernel_dir": "k1", "output_dir": "out/a"},
            {"name": "j2", "kernel_dir": "k2", "output_dir": "out/b"},
        ],
    )
    with pytest.raises((ValueError, RuntimeError)):
        load_batch(manifest)


# ---------------------------------------------------------------------------
# 5b. Title/id slug consistency — regression for Kaggle clean-URL mismatch
# ---------------------------------------------------------------------------


def test_load_batch_rejects_title_id_slug_mismatch(tmp_path: Path) -> None:
    """Regression: Kaggle derives the kernel URL slug from the *title*, not
    the metadata ``id``.  When the title-derived slug differs from the
    ``id``'s final component, Kaggle creates the kernel under an unexpected
    slug and subsequent polling of the requested ``id`` fails.

    Real failure: id ``abdellahkadem/oczy-scheduler-smoke-1`` with title
    ``Oczy Scheduler CPU Smoke 1`` was created as
    ``abdellahkadem/oczy-scheduler-cpu-smoke-1``, so polling the requested
    id never found the kernel.

    Batch loading must reject this *before* any submission.
    """
    meta = _valid_metadata("abdellahkadem/oczy-scheduler-smoke-1")
    meta["title"] = "Oczy Scheduler CPU Smoke 1"
    _make_kernel_dir(tmp_path, "k1", metadata=meta)
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    with pytest.raises(BatchValidationError, match="resolves to slug"):
        load_batch(manifest)


def test_load_batch_accepts_matching_title_id_slug(tmp_path: Path) -> None:
    """When the title-derived slug equals the ``id``'s final component,
    batch loading succeeds and the job is ready for submission."""
    _make_kernel_dir(tmp_path, "k1", "abdellahkadem/oczy-scheduler-smoke-1")
    manifest = _make_batch(
        tmp_path,
        [{"name": "j1", "kernel_dir": "k1", "output_dir": "out/j1"}],
    )
    jobs = load_batch(manifest)
    assert len(jobs) == 1
    assert jobs[0]["kernel_id"] == "abdellahkadem/oczy-scheduler-smoke-1"


# ---------------------------------------------------------------------------
# 6. Concurrency bounds: explicit cap, additive default, minimum 1
# ---------------------------------------------------------------------------


def test_run_respects_default_max_parallel(tmp_path: Path) -> None:
    """With default max_parallel=8, at most 8 jobs should be actively
    submitted/running at any time."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 12)
    state = tmp_path / "state.json"

    class ConcurrencyClient(FakeKaggleClient):
        """Tracks the maximum number of concurrently active jobs by
        adding on push and removing on terminal status."""

        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/k{i}": ["running", "complete"] for i in range(12)
                },
            )
            self._active: set[str] = set()
            self.max_concurrent = 0

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            self._active.add(kid)
            self.max_concurrent = max(self.max_concurrent, len(self._active))
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                self._active.discard(kernel_id)
            return result

    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = ConcurrencyClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=8, poll_interval=1)
    assert client.max_concurrent <= 8, (
        f"exceeded default max_parallel: {client.max_concurrent}"
    )


def test_run_respects_custom_max_parallel(tmp_path: Path) -> None:
    """A custom max_parallel=3 should bound concurrency to 3."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 10)
    state = tmp_path / "state.json"

    class ConcurrencyClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/k{i}": ["running", "complete"] for i in range(10)
                },
            )
            self._active: set[str] = set()
            self.max_concurrent = 0

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            self._active.add(kid)
            self.max_concurrent = max(self.max_concurrent, len(self._active))
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                self._active.discard(kernel_id)
            return result

    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = ConcurrencyClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=3, poll_interval=1)
    assert client.max_concurrent <= 3


def test_run_accepts_max_parallel_above_old_hard_cap(tmp_path: Path) -> None:
    """max_parallel=11 (above the former hard cap of 10) must now be accepted.

    The global cap is optional and has no upper bound; only max_parallel=0
    (or negative) is rejected.  This replaces the old test that asserted
    max_parallel > 10 raises.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 11)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(11)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=11, poll_interval=1)
    assert summary is not None
    assert summary["total"] == 11


def test_run_rejects_max_parallel_below_minimum(tmp_path: Path) -> None:
    """max_parallel=0 must be rejected (minimum is 1 when provided)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    with pytest.raises((ValueError, RuntimeError)):
        sched.run(manifest, state, max_parallel=0)


def test_run_accepts_max_parallel_none(tmp_path: Path) -> None:
    """max_parallel=None (the default) must be accepted — no global cap."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 3)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(3)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=None, poll_interval=1)
    assert summary is not None
    assert summary["all_succeeded"]


def test_run_accepts_max_parallel_at_10(tmp_path: Path) -> None:
    """max_parallel=10 must be accepted (no longer a hard cap, just a value)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 10)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(10)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=10, poll_interval=1)
    assert summary is not None

def test_run_accepts_max_parallel_one(tmp_path: Path) -> None:
    """max_parallel=1 (the minimum) must be accepted and serialize jobs."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 3)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(3)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=1, poll_interval=1)
    assert summary is not None


def test_run_rejects_kaggle_max_above_hard_cap(tmp_path: Path) -> None:
    """kaggle_max=11 must be rejected — HARD_KAGGLE_MAX=10 is still enforced."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    with pytest.raises((ValueError, RuntimeError)):
        sched.run(manifest, state, kaggle_max=11)


def test_run_default_kaggle_max_is_8(tmp_path: Path) -> None:
    """The default kaggle_max is 8 (DEFAULT_KAGGLE_MAX), unchanged."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 12)
    state = tmp_path / "state.json"

    class ConcurrencyClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/k{i}": ["running", "complete"] for i in range(12)
                },
            )
            self._active: set[str] = set()
            self.max_concurrent = 0

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            self._active.add(kid)
            self.max_concurrent = max(self.max_concurrent, len(self._active))
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                self._active.discard(kernel_id)
            return result

    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = ConcurrencyClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    # max_parallel=None (no global cap), default kaggle_max=8
    sched.run(manifest, state, max_parallel=None, poll_interval=1)
    assert client.max_concurrent <= DEFAULT_KAGGLE_MAX, (
        f"exceeded default kaggle_max: {client.max_concurrent}"
    )


# ---------------------------------------------------------------------------
# 7. State transitions: pending -> submitting -> running -> collecting -> succeeded
# ---------------------------------------------------------------------------


def test_full_lifecycle_success(tmp_path: Path) -> None:
    """A single job that completes successfully transitions through all
    states and ends at succeeded."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["running", "complete"]},
        output_files={"owner/k0": {"result.json": '{"score": 0.9}'}},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # Summary should report success
    assert summary is not None
    # State file should show succeeded
    state_data = json.loads(state.read_text())
    assert state_data["schema_version"] == EXPECTED_STATE_SCHEMA
    job_state = state_data["jobs"]["job0"]
    assert job_state["state"] == SUCCEEDED
    assert job_state["kernel_id"] == "owner/k0"
    # Output should have been collected
    assert job_state.get("collected_at") is not None or job_state.get("completed_at") is not None
    # Output file should exist
    out_dir = (tmp_path / "out" / "job0").resolve()
    assert (out_dir / "result.json").exists()


def test_state_file_written_atomically(tmp_path: Path) -> None:
    """The state file must exist and be valid JSON after the run completes."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(2)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # State file must be valid JSON (atomic write = no partial content)
    data = json.loads(state.read_text())
    assert data["schema_version"] == EXPECTED_STATE_SCHEMA
    assert "jobs" in data
    assert len(data["jobs"]) == 2
    for job in data["jobs"].values():
        assert job["state"] in (SUCCEEDED, FAILED)


def test_state_file_has_required_keys(tmp_path: Path) -> None:
    """Each job in the state file must have the required keys."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    data = json.loads(state.read_text())
    job = data["jobs"]["job0"]
    for key in ("name", "kernel_id", "state"):
        assert key in job, f"missing key {key} in state job"


# ---------------------------------------------------------------------------
# 8. Resume: interrupted submitting/collecting -> pending/running
# ---------------------------------------------------------------------------


def test_resume_converts_submitting_to_pending(tmp_path: Path) -> None:
    """If the state file has a job in 'submitting' state, resume should
    convert it to 'pending' so it gets resubmitted."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    # Write a state file with a job stuck in submitting
    state.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_STATE_SCHEMA,
                "batch_path": str(manifest),
                "jobs": {
                    "job0": {
                        "name": "job0",
                        "kernel_dir": str((tmp_path / "job0").resolve()),
                        "output_dir": str((tmp_path / "out" / "job0").resolve()),
                        "kernel_id": "owner/k0",
                        "state": SUBMITTING,
                        "error": None,
                        "submitted_at": None,
                        "completed_at": None,
                        "collected_at": None,
                        "attempts": 0,
                    }
                },
                "updated_at": 1000.0,
            }
        )
    )
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    push_count = 0

    class CountingPushClient(FakeKaggleClient):
        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            nonlocal push_count
            push_count += 1
            return super().push(kernel_dir, timeout=timeout)

    client = CountingPushClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # The job should have been resubmitted (push called)
    assert push_count == 1, "stuck submitting job was not resubmitted"
    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED


def test_resume_converts_collecting_to_running(tmp_path: Path) -> None:
    """If the state file has a job in 'collecting' state, resume should
    convert it to 'running' and then collect output."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_STATE_SCHEMA,
                "batch_path": str(manifest),
                "jobs": {
                    "job0": {
                        "name": "job0",
                        "kernel_dir": str((tmp_path / "job0").resolve()),
                        "output_dir": str((tmp_path / "out" / "job0").resolve()),
                        "kernel_id": "owner/k0",
                        "state": COLLECTING,
                        "error": None,
                        "submitted_at": 1000.0,
                        "completed_at": None,
                        "collected_at": None,
                        "attempts": 1,
                    }
                },
                "updated_at": 1001.0,
            }
        )
    )
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    push_count = 0

    class CountingPushClient(FakeKaggleClient):
        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            nonlocal push_count
            push_count += 1
            return super().push(kernel_dir, timeout=timeout)

    client = CountingPushClient(
        status_sequences={"owner/k0": ["complete"]},
        output_files={"owner/k0": {"result.json": '{"ok": true}'}},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # Should NOT resubmit a job that was in collecting (already running)
    assert push_count == 0, "collecting job was resubmitted instead of resumed"
    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED
    # Output should have been collected
    out_dir = (tmp_path / "out" / "job0").resolve()
    assert (out_dir / "result.json").exists()


def test_resume_does_not_resubmit_running_kernel(tmp_path: Path) -> None:
    """If the state file has a job in 'running' state with a recorded
    kernel_id, resume should NOT resubmit it — only poll its status."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_STATE_SCHEMA,
                "batch_path": str(manifest),
                "jobs": {
                    "job0": {
                        "name": "job0",
                        "kernel_dir": str((tmp_path / "job0").resolve()),
                        "output_dir": str((tmp_path / "out" / "job0").resolve()),
                        "kernel_id": "owner/k0",
                        "state": RUNNING,
                        "error": None,
                        "submitted_at": 1000.0,
                        "completed_at": None,
                        "collected_at": None,
                        "attempts": 1,
                    }
                },
                "updated_at": 1001.0,
            }
        )
    )
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    push_count = 0

    class CountingPushClient(FakeKaggleClient):
        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            nonlocal push_count
            push_count += 1
            return super().push(kernel_dir, timeout=timeout)

    client = CountingPushClient(
        status_sequences={"owner/k0": ["complete"]},
        output_files={"owner/k0": {"result.json": '{"ok": true}'}},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    assert push_count == 0, "running job was resubmitted"
    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED


def test_resume_skips_already_succeeded(tmp_path: Path) -> None:
    """A job already in 'succeeded' state should not be re-run."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_STATE_SCHEMA,
                "batch_path": str(manifest),
                "jobs": {
                    "job0": {
                        "name": "job0",
                        "kernel_dir": str((tmp_path / "job0").resolve()),
                        "output_dir": str((tmp_path / "out" / "job0").resolve()),
                        "kernel_id": "owner/k0",
                        "state": SUCCEEDED,
                        "error": None,
                        "submitted_at": 1000.0,
                        "completed_at": 1005.0,
                        "collected_at": 1006.0,
                        "attempts": 1,
                    }
                },
                "updated_at": 1006.0,
            }
        )
    )
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    push_count = 0

    class CountingPushClient(FakeKaggleClient):
        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            nonlocal push_count
            push_count += 1
            return super().push(kernel_dir, timeout=timeout)

    client = CountingPushClient(
        status_sequences={"owner/k1": ["complete"]},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # Only job1 should have been pushed
    assert push_count == 1
    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED
    assert data["jobs"]["job1"]["state"] == SUCCEEDED


# ---------------------------------------------------------------------------
# 9. Mixed success and failure
# ---------------------------------------------------------------------------


def test_mixed_success_and_failure(tmp_path: Path) -> None:
    """A batch where some jobs succeed and some fail should record each
    correctly in the state file."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 3)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={
            "owner/k0": ["complete"],
            "owner/k1": ["error"],
            "owner/k2": ["complete"],
        },
        output_files={
            "owner/k0": {"r.txt": "ok"},
            "owner/k2": {"r.txt": "ok"},
        },
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED
    assert data["jobs"]["job1"]["state"] == FAILED
    assert data["jobs"]["job2"]["state"] == SUCCEEDED
    assert data["jobs"]["job1"].get("error") is not None


def test_push_failure_marks_job_failed(tmp_path: Path) -> None:
    """If push raises an exception, the job should be marked failed."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        push_raises={"owner/k0": RuntimeError("push failed")},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == FAILED
    assert "push failed" in (data["jobs"]["job0"].get("error") or "")


def test_output_failure_marks_job_failed(tmp_path: Path) -> None:
    """If output collection raises an exception, the job should be marked
    failed (not left in collecting)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
        output_raises={"owner/k0": RuntimeError("output download failed")},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == FAILED
    assert "output download failed" in (data["jobs"]["job0"].get("error") or "")


# ---------------------------------------------------------------------------
# 10. Status classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("complete", "complete"),
        ("COMPLETED", "complete"),
        ("complete ", "complete"),
        ("error", "error"),
        ("errored", "error"),
        ("failed", "error"),
        ("cancelled", "error"),
        ("running", "running"),
        ("running ", "running"),
        ("queued", "queued"),
        ("pending", "queued"),
        ("launching", "queued"),
    ],
)
def test_classify_status_known_values(raw: str, expected: str) -> None:
    """Known status strings must classify to the correct bucket."""
    assert classify_status(raw) == expected


def test_classify_status_unknown_falls_back_to_error() -> None:
    """An unrecognized status string should fall back to error classification,
    not silently pass as running/queued."""
    result = classify_status("some_unknown_status_xyz")
    assert result == "error"


# ---------------------------------------------------------------------------
# 11. Timeout handling
# ---------------------------------------------------------------------------


def test_job_timeout_marks_failed(tmp_path: Path) -> None:
    """A job that stays running beyond job_timeout should be marked failed."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["running"]},  # never completes
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(
        manifest,
        state,
        max_parallel=4,
        poll_interval=1,
        job_timeout=5,
    )

    data = json.loads(state.read_text())
    assert "timed out" in (data["jobs"]["job0"].get("error") or "").lower()


def test_push_timeout_passed_to_client(tmp_path: Path) -> None:
    """The push_timeout parameter should be forwarded to the client's push
    method as the timeout argument."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(
        manifest,
        state,
        max_parallel=4,
        poll_interval=1,
        push_timeout=300,
    )

    # Check that push was called with the timeout
    assert len(client.push_calls) == 1
    _, timeout = client.push_calls[0]
    assert timeout == 300


# ---------------------------------------------------------------------------
# 12. Output collection
# ---------------------------------------------------------------------------


def test_output_collected_on_success(tmp_path: Path) -> None:
    """When a job completes, its output should be downloaded to output_dir."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
        output_files={"owner/k0": {"output.json": '{"result": 42}', "log.txt": "done\n"}},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    out_dir = (tmp_path / "out" / "job0").resolve()
    assert (out_dir / "output.json").exists()
    assert (out_dir / "log.txt").exists()
    assert json.loads((out_dir / "output.json").read_text())["result"] == 42


def test_output_not_collected_on_failure(tmp_path: Path) -> None:
    """When a job errors, output should not be collected."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["error"]},
        output_files={"owner/k0": {"should_not_exist.txt": "nope"}},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    out_dir = (tmp_path / "out" / "job0").resolve()
    assert not (out_dir / "should_not_exist.txt").exists()
    assert len(client.output_calls) == 0


# ---------------------------------------------------------------------------
# 13. CLI: run and status subcommands
# ---------------------------------------------------------------------------


def test_cli_run_all_succeed_exits_zero(tmp_path: Path) -> None:
    """CLI run with all jobs succeeding should exit 0 and print JSON."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(2)},
    )
    rc = main(
        ["run", str(manifest), "--state", str(state), "--max-parallel", "4", "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc == 0


def test_cli_run_with_failure_exits_nonzero(tmp_path: Path) -> None:
    """CLI run with any job failing should exit nonzero."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={
            "owner/k0": ["complete"],
            "owner/k1": ["error"],
        },
    )
    rc = main(
        ["run", str(manifest), "--state", str(state), "--max-parallel", "4", "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc != 0


def test_cli_run_prints_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI run should print a JSON summary to stdout."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    main(
        ["run", str(manifest), "--state", str(state), "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "jobs" in data or "succeeded" in data or "total" in data


def test_cli_status_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI status should print a JSON status summary."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    # First run to populate state
    main(
        ["run", str(manifest), "--state", str(state), "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    capsys.readouterr()  # clear
    rc = main(
        ["status", str(manifest), "--state", str(state)],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert rc == 0


def test_cli_run_default_is_additive(tmp_path: Path) -> None:
    """The CLI --max-parallel should default to None (additive provider capacity).

    With 12 Kaggle jobs and no --max-parallel, concurrency is bounded only by
    the default kaggle_max=8, not by a global cap.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 12)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)

    class ConcurrencyClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/k{i}": ["running", "complete"] for i in range(12)
                },
            )
            self._active: set[str] = set()
            self.max_concurrent = 0

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            self._active.add(kid)
            self.max_concurrent = max(self.max_concurrent, len(self._active))
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                self._active.discard(kernel_id)
            return result

    client = ConcurrencyClient()
    main(
        ["run", str(manifest), "--state", str(state), "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    # No global cap — bounded only by default kaggle_max=8
    assert client.max_concurrent <= DEFAULT_KAGGLE_MAX


def test_cli_run_accepts_max_parallel_above_10(tmp_path: Path) -> None:
    """CLI --max-parallel 11 must now be accepted (no hard upper bound)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 11)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(11)},
    )
    rc = main(
        ["run", str(manifest), "--state", str(state),
         "--max-parallel", "11", "--poll-interval", "1"],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc == 0


def test_cli_run_rejects_max_parallel_zero(tmp_path: Path) -> None:
    """CLI --max-parallel 0 should still fail via SystemExit (parser.error)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    with pytest.raises(SystemExit):
        main(
            ["run", str(manifest), "--state", str(state), "--max-parallel", "0"],
            client=client,
            clock=clock,
            sleeper=sleeper,
        )


def test_cli_run_rejects_kaggle_max_above_10(tmp_path: Path) -> None:
    """CLI --kaggle-max 11 should fail via SystemExit (parser.error)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    with pytest.raises(SystemExit):
        main(
            ["run", str(manifest), "--state", str(state), "--kaggle-max", "11"],
            client=client,
            clock=clock,
            sleeper=sleeper,
        )


def test_cli_run_accepts_push_timeout_and_job_timeout(tmp_path: Path) -> None:
    """CLI should accept --push-timeout and --job-timeout flags."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    rc = main(
        [
            "run", str(manifest), "--state", str(state),
            "--push-timeout", "600",
            "--job-timeout", "900",
            "--poll-interval", "1",
        ],
        client=client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc == 0
    assert client.push_calls[0][1] == 600


# ---------------------------------------------------------------------------
# 14. get_status / status_summary
# ---------------------------------------------------------------------------


def test_get_status_returns_current_state(tmp_path: Path) -> None:
    """get_status should return the current state from the state file."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["complete"] for i in range(2)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # Try both possible method names
    get_status = getattr(sched, "get_status", None)
    if get_status is None:
        get_status = getattr(sched, "status_summary", None)
    assert get_status is not None, "scheduler must have get_status or status_summary"
    result = get_status(manifest, state)
    assert isinstance(result, dict)
    assert "jobs" in result or "summary" in result


def test_get_status_without_state_file(tmp_path: Path) -> None:
    """get_status on a batch with no state file should report all pending
    or empty, not crash."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "nonexistent_state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    get_status = getattr(sched, "get_status", None)
    if get_status is None:
        get_status = getattr(sched, "status_summary", None)
    assert get_status is not None
    result = get_status(manifest, state)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 15. Run summary structure
# ---------------------------------------------------------------------------


def test_run_summary_has_job_counts(tmp_path: Path) -> None:
    """The run summary should include counts of succeeded/failed jobs."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 3)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={
            "owner/k0": ["complete"],
            "owner/k1": ["error"],
            "owner/k2": ["complete"],
        },
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(manifest, state, max_parallel=4, poll_interval=1)

    # Summary should have some way to distinguish succeeded vs failed
    assert isinstance(summary, dict)
    # Check for either explicit counts or per-job status
    has_counts = any(
        k in summary
        for k in ("succeeded", "failed", "total", "success_count", "failure_count")
    )
    has_jobs = "jobs" in summary
    assert has_counts or has_jobs, (
        f"summary must include job counts or per-job status: {summary}"
    )


# ---------------------------------------------------------------------------
# 16. No real subprocess / network calls
# ---------------------------------------------------------------------------


def test_fake_client_does_not_invoke_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure that using a fake client never triggers a real subprocess."""
    import subprocess as sp

    called: list[str] = []

    orig_popen = sp.Popen

    def _tracking_popen(*args: Any, **kwargs: Any) -> Any:
        called.append(str(args))
        return orig_popen(*args, **kwargs)

    monkeypatch.setattr(sp, "Popen", _tracking_popen)
    monkeypatch.setattr(sp, "run", lambda *a, **k: called.append(str(a)))

    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)
    assert called == [], f"subprocess was invoked: {called}"


# ---------------------------------------------------------------------------
# 17. Multiple jobs complete in parallel
# ---------------------------------------------------------------------------


def test_all_jobs_complete_in_parallel(tmp_path: Path) -> None:
    """A batch of 4 jobs with max_parallel=4 should all be submitted without
    waiting, and all should complete successfully."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 4)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient(
        status_sequences={f"owner/k{i}": ["running", "complete"] for i in range(4)},
        output_files={f"owner/k{i}": {"r.txt": "ok"} for i in range(4)},
    )
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    sched.run(manifest, state, max_parallel=4, poll_interval=1)

    data = json.loads(state.read_text())
    for i in range(4):
        assert data["jobs"][f"job{i}"]["state"] == SUCCEEDED
    # All 4 should have been pushed
    assert len(client.push_calls) == 4
    # All 4 should have had output collected
    assert len(client.output_calls) == 4


# ---------------------------------------------------------------------------
# 18. State file batch binding: a state file is bound to its batch manifest
# ---------------------------------------------------------------------------


def test_state_file_rejects_foreign_batch(tmp_path: Path) -> None:
    """A nonempty durable state file created for batch A must be rejected
    when reused with a different batch B — the scheduler must raise
    BatchValidationError rather than silently merging unrelated jobs."""
    # Batch A: run a single job to completion so the state file is nonempty
    # and bound to batch A's resolved manifest path.
    manifest_a, _ = _make_batch_with_kernels(tmp_path, 1, prefix="a")
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client_a = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched_a = ParallelScheduler(client_a, clock=clock, sleeper=sleeper)
    sched_a.run(manifest_a, state, max_parallel=4, poll_interval=1)

    # The state file must exist and be nonempty (bound to batch A).
    assert state.is_file(), "state file must exist after batch A run"
    raw = json.loads(state.read_text())
    assert raw["jobs"], "state file must be nonempty after a completed run"
    assert raw["batch_path"] == str(manifest_a.resolve())

    # Batch B: a different manifest path in a separate directory.
    sub = tmp_path / "batch_b"
    sub.mkdir()
    manifest_b, _ = _make_batch_with_kernels(sub, 1, prefix="b")
    assert manifest_b.resolve() != manifest_a.resolve()

    # Reusing A's state file with batch B must raise BatchValidationError.
    client_b = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    sched_b = ParallelScheduler(client_b, clock=clock, sleeper=sleeper)
    with pytest.raises(BatchValidationError):
        sched_b.run(manifest_b, state, max_parallel=4, poll_interval=1)


# ---------------------------------------------------------------------------
# 19. Real KaggleCliClient argv contracts (monkeypatched subprocess)
# ---------------------------------------------------------------------------


def test_cli_client_push_argv_contains_kaggle_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real KaggleCliClient must send the push timeout as a Kaggle CLI
    --timeout/-t argument (the remote kernel run-time limit), not merely as
    the local subprocess timeout."""
    import subprocess as sp
    from types import SimpleNamespace

    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **_kw: Any) -> Any:
        captured.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sp, "run", _fake_run)

    kd = _make_kernel_dir(tmp_path, "k0", "owner/k0")
    client = KaggleCliClient(push_timeout=300)
    client.push(str(kd), timeout=300)

    assert len(captured) == 1
    argv = captured[0]
    assert argv[:3] == ["kaggle", "kernels", "push"]

    # The argv must contain --timeout or -t with the run-limit value.
    flag_idx = None
    for flag in ("--timeout", "-t"):
        if flag in argv:
            flag_idx = argv.index(flag)
            break
    assert flag_idx is not None, "push argv missing --timeout/-t flag"
    assert flag_idx + 1 < len(argv), "push argv has --timeout/-t without a value"
    assert float(argv[flag_idx + 1]) == 300.0


def test_cli_client_push_argv_timeout_defaults_to_client_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When push is called without an explicit timeout, the KaggleCliClient
    must still send its configured push_timeout as the Kaggle run-time limit."""
    import subprocess as sp
    from types import SimpleNamespace

    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **_kw: Any) -> Any:
        captured.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sp, "run", _fake_run)

    kd = _make_kernel_dir(tmp_path, "k0", "owner/k0")
    client = KaggleCliClient(push_timeout=600)
    client.push(str(kd))  # no explicit timeout — must use client default

    assert len(captured) == 1
    argv = captured[0]
    flag_idx = None
    for flag in ("--timeout", "-t"):
        if flag in argv:
            flag_idx = argv.index(flag)
            break
    assert flag_idx is not None, "push argv missing --timeout/-t flag"
    assert flag_idx + 1 < len(argv), "push argv has --timeout/-t without a value"
    assert float(argv[flag_idx + 1]) == 600.0


def test_cli_client_output_argv_contains_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real KaggleCliClient must pass --force to kaggle kernels output
    so that resume is idempotent (re-downloading without prompting)."""
    import subprocess as sp
    from types import SimpleNamespace

    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **_kw: Any) -> Any:
        captured.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sp, "run", _fake_run)

    out_dir = tmp_path / "out"
    client = KaggleCliClient()
    client.output("owner/k0", str(out_dir))

    assert len(captured) == 1
    argv = captured[0]
    assert argv[:3] == ["kaggle", "kernels", "output"]
    assert "--force" in argv


def test_default_clock_produces_wall_clock_timestamps(tmp_path: Path) -> None:
    """The scheduler's default clock must produce wall-clock (Unix epoch)
    timestamps in durable state, not monotonic time that resets across
    reboots.  Verified at the behavior level: submitted_at recorded with
    the default clock must be close to time.time()."""
    import time as _time

    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    client = FakeKaggleClient(status_sequences={"owner/k0": ["complete"]})
    # Use the DEFAULT clock (do not inject clock=) but a no-op sleeper so
    # the test never really sleeps.
    sched = ParallelScheduler(client, sleeper=lambda _: None)

    wall_before = _time.time()
    sched.run(manifest, state, max_parallel=4, poll_interval=1)
    wall_after = _time.time()

    data = json.loads(state.read_text())
    job = data["jobs"]["job0"]
    assert job["submitted_at"] is not None
    # Wall-clock time is ~1.75e9 in 2026; monotonic is seconds-since-boot
    # (orders of magnitude smaller).  A 120-second tolerance is generous
    # yet still distinguishes the two clocks decisively.
    assert wall_before - 120 <= job["submitted_at"] <= wall_after + 120
