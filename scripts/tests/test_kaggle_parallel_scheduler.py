"""High-signal contract tests for the durable CPU-only Kaggle parallel scheduler.

These tests exercise the *behavior* specified in the shared contract:

* Manifest loading and validation (schema, relative paths, CPU-only rejection,
  duplicate name/id detection, profile/cpu checks).
* Concurrency bounds: default 10, hard cap 10, minimum 1, bounded active jobs.
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
classify_status = _mod["classify_status"]
load_batch = _mod["load_batch"]
BatchValidationError = _mod["BatchValidationError"]
ParallelScheduler = _mod["ParallelScheduler"]
main = _mod["main"]
KaggleCliClient = _mod["KaggleCliClient"]
compute_manifest_sha256 = _mod["compute_manifest_sha256"]

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

RUNTIME_MANIFEST_SCHEMA_VERSION = "oczy/runtime-manifest/v2"
EXECUTION_REPORT_SCHEMA_VERSION = "oczy/execution-report/v2"
EXPECTED_BATCH_SCHEMA = "oczy/remote-parallel-batch/v3"
EXPECTED_STATE_SCHEMA = "oczy/remote-parallel-state/v4"


def _valid_runtime_manifest(**overrides: Any) -> dict[str, Any]:
    """Valid required no-model runtime manifest for scheduler fixtures."""
    manifest: dict[str, Any] = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
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
    manifest.update(overrides)
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    return manifest


def _execution_report(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = manifest if manifest is not None else _valid_runtime_manifest()
    return {
        "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "exit_code": 0,
        "expected_runtime_manifest_sha256": compute_manifest_sha256(observed),
        "observed_runtime_manifest": observed,
    }


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


def _valid_job_spec(
    profile: str = "cpu", runtime_manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = runtime_manifest if runtime_manifest is not None else _valid_runtime_manifest()
    return {
        "schema_version": "oczy/kaggle-research-job/v2",
        "job_name": "test-job",
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
        "runtime_manifest": manifest,
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
    normalized_jobs: list[dict[str, Any]] = []
    for job in jobs:
        entry = dict(job)
        entry.setdefault("provider", PROVIDER_KAGGLE)
        entry.setdefault("runtime_manifest", _valid_runtime_manifest())
        normalized_jobs.append(entry)
    manifest = {
        "schema_version": schema_version or EXPECTED_BATCH_SCHEMA,
        "jobs": normalized_jobs,
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
        if kernel_id in self._output_files:
            files = dict(self._output_files[kernel_id])
            omit_report = files.pop("__omit_execution_report__", None) is not None
            if not omit_report and "execution_report.json" not in files:
                files["execution_report.json"] = json.dumps(_execution_report())
        else:
            files = {
                "result.json": '{"ok": true}',
                "execution_report.json": json.dumps(_execution_report()),
            }
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
                "jobs": [{
                    "name": "j1",
                    "provider": PROVIDER_KAGGLE,
                    "kernel_dir": "k1",
                    "output_dir": "out/j1",
                    "runtime_manifest": _valid_runtime_manifest(),
                }],
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
    """With explicit max_parallel=8, at most 8 jobs should be actively
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
    """kaggle_max=6 must be rejected — HARD_KAGGLE_MAX=5 is still enforced."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    with pytest.raises((ValueError, RuntimeError)):
        sched.run(manifest, state, kaggle_max=6)


def test_run_default_kaggle_max_is_5(tmp_path: Path) -> None:
    """The default kaggle_max is 5 (DEFAULT_KAGGLE_MAX), matching the hard cap."""
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
    # max_parallel=None (no global cap), default kaggle_max=5
    sched.run(manifest, state, max_parallel=None, poll_interval=1)
    assert client.max_concurrent <= DEFAULT_KAGGLE_MAX, (
        f"exceeded default kaggle_max: {client.max_concurrent}"
    )
    assert client.max_concurrent == DEFAULT_KAGGLE_MAX, (
        f"expected exactly {DEFAULT_KAGGLE_MAX} concurrent Kaggle jobs, "
        f"got {client.max_concurrent}"
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
# 9b. Runtime manifest gate
# ---------------------------------------------------------------------------


def test_runtime_gate_rejects_provider_ok_without_execution_report(tmp_path: Path) -> None:
    """Provider output success is not scientific success without report v2."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
        output_files={
            "owner/k0": {
                "__omit_execution_report__": "1",
                "result.json": '{"ok": true}',
            }
        },
    )

    summary = ParallelScheduler(client, clock=clock, sleeper=CountingSleeper(holder)).run(
        manifest, state, max_parallel=4, poll_interval=1
    )

    job = summary["jobs"]["job0"]
    assert job["state"] == FAILED
    assert job["runtime_manifest_verified"] is False
    assert "no execution report" in (job["error"] or "")


def test_runtime_gate_records_hashes_on_matching_report(tmp_path: Path) -> None:
    """A matching report is the condition that upgrades provider success."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()

    summary = ParallelScheduler(
        FakeKaggleClient(status_sequences={"owner/k0": ["complete"]}),
        clock=clock,
        sleeper=CountingSleeper(holder),
    ).run(manifest, state, max_parallel=4, poll_interval=1)

    expected_hash = _valid_runtime_manifest()["manifest_sha256"]
    job = summary["jobs"]["job0"]
    assert job["state"] == SUCCEEDED
    assert job["runtime_manifest_verified"] is True
    assert job["expected_runtime_manifest_sha256"] == expected_hash
    assert job["observed_runtime_manifest_sha256"] == expected_hash


def test_runtime_gate_fails_manifest_mismatch(tmp_path: Path) -> None:
    """A report with a different observed manifest fails before success."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    expected = _valid_runtime_manifest()
    observed = _valid_runtime_manifest(python_version="3.13.0")
    report = _execution_report(observed)
    report["expected_runtime_manifest_sha256"] = compute_manifest_sha256(expected)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
        output_files={"owner/k0": {"execution_report.json": json.dumps(report)}},
    )

    summary = ParallelScheduler(client, clock=clock, sleeper=CountingSleeper(holder)).run(
        manifest, state, max_parallel=4, poll_interval=1
    )

    job = summary["jobs"]["job0"]
    assert job["state"] == FAILED
    assert job["runtime_manifest_verified"] is False
    assert "runtime manifest mismatch" in (job["error"] or "")


def test_resume_rejects_changed_runtime_manifest_hash(tmp_path: Path) -> None:
    """An existing state cannot be resumed under changed runtime identity."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    original = _valid_runtime_manifest()
    changed = _valid_runtime_manifest(python_version="3.13.0")
    raw_batch = json.loads(manifest.read_text(encoding="utf-8"))
    raw_batch["jobs"][0]["runtime_manifest"] = changed
    manifest.write_text(json.dumps(raw_batch), encoding="utf-8")
    (tmp_path / "job0" / "job_spec.json").write_text(
        json.dumps(_valid_job_spec(runtime_manifest=changed)),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_STATE_SCHEMA,
                "batch_path": str(manifest.resolve()),
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
                        "provider": PROVIDER_KAGGLE,
                        "remote_id": "owner/k0",
                        "expected_runtime_manifest_sha256": original["manifest_sha256"],
                        "observed_runtime_manifest_sha256": "",
                        "runtime_manifest_verified": False,
                    }
                },
                "updated_at": 1001.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BatchValidationError, match="batch identity changed"):
        ParallelScheduler(FakeKaggleClient()).run(manifest, state, poll_interval=1)
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
    the default kaggle_max=5, not by a global cap.
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
    # No global cap — bounded only by default kaggle_max=5
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


def test_cli_run_rejects_kaggle_max_above_5(tmp_path: Path) -> None:
    """CLI --kaggle-max 6 should fail via SystemExit (parser.error)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    holder, clock = _make_clock()
    sleeper = CountingSleeper(holder)
    client = FakeKaggleClient()
    with pytest.raises(SystemExit):
        main(
            ["run", str(manifest), "--state", str(state), "--kaggle-max", "6"],
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
# ---------------------------------------------------------------------------
# 20. Live-watch contract: watch_batch mode regression tests
# ---------------------------------------------------------------------------
#
# These tests defend the durable live-watch queue contract:
#   - New jobs appended during an active run are merged and executed.
#   - Existing job definition/state cannot be mutated by reload.
#   - Malformed reload is retried; later valid content succeeds.
#   - Watch mode stays alive across an empty interval and can be
#     terminated deterministically (via KeyboardInterrupt).
#   - Non-watch mode still terminates (backward compatible).
#   - Additive 5 Kaggle + learned Colab X admission in watch mode.
#
# All tests avoid real network and real sleeps by injecting fake clients,
# a deterministic clock, and a bounded or KeyboardInterrupt-raising sleeper.
# ---------------------------------------------------------------------------

DEFAULT_WATCH_INTERVAL: float | None = _mod.get("DEFAULT_WATCH_INTERVAL")


def _append_job_to_batch(
    base: Path,
    batch_path: Path,
    name: str,
    kernel_id: str,
) -> None:
    """Append a new job to an existing v1 batch manifest, creating its
    kernel directory.  Raises if the name already exists in the batch."""
    manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    existing_names = {j["name"] for j in manifest["jobs"]}
    assert name not in existing_names, f"job {name} already in batch"
    _make_kernel_dir(base, name, kernel_id)
    manifest["jobs"].append(
        {
            "name": name,
            "provider": PROVIDER_KAGGLE,
            "kernel_dir": name,
            "output_dir": f"out/{name}",
            "runtime_manifest": _valid_runtime_manifest(),
        }
    )
    batch_path.write_text(json.dumps(manifest), encoding="utf-8")


class _KeyboardInterruptSleeper:
    """Sleeper that raises KeyboardInterrupt after *max_calls* calls,
    simulating Ctrl-C to terminate watch mode deterministically."""

    def __init__(self, clock: list[float], max_calls: int) -> None:
        self._clock = clock
        self._max = max_calls
        self._n = 0
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock[0] += seconds
        self._n += 1
        if self._n >= self._max:
            raise KeyboardInterrupt()


class _CountingSleeperNoRaise:
    """Sleeper that records calls and advances a fake clock without raising."""

    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock[0] += seconds


def test_watch_rejects_nonpositive_watch_interval(tmp_path: Path) -> None:
    """watch_batch=True with watch_interval <= 0 must raise ValueError."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    client = FakeKaggleClient(status_sequences={"owner/k0": ["complete"]})
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    with pytest.raises(ValueError, match="watch_interval"):
        sched.run(
            manifest, state, max_parallel=4, poll_interval=1,
            watch_batch=True, watch_interval=0,
        )


def test_watch_merges_new_jobs_appended_during_active_run(
    tmp_path: Path,
) -> None:
    """In watch mode, new jobs appended to the batch file while the
    scheduler is running are merged as PENDING and eventually executed.

    The batch starts with 1 job.  While that job is RUNNING, a 2nd job is
    appended to the batch file.  The scheduler's _maybe_reload_batch detects
    the file change, merges the new job as PENDING, submits it, and both
    reach SUCCEEDED.  The sleeper raises KeyboardInterrupt after enough
    cycles to complete both jobs plus one idle watch interval.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    # The first job stays running for 1 poll, then completes.
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["running", "complete"],
                          "owner/k1": ["complete"]},
    )

    # Append job1 to the batch AFTER the scheduler has submitted job0.
    # We use a wrapper on _submit to time the file mutation.
    original_submit = ParallelScheduler._submit

    appended = {"done": False}

    def _wrapped_submit(self, job, push_timeout):
        result = original_submit(self, job, push_timeout)
        if not appended["done"] and job.name == "job0":
            _append_job_to_batch(tmp_path, manifest, "job1", "owner/k1")
            appended["done"] = True
        return result

    # Use a sleeper that raises KeyboardInterrupt after enough cycles.
    # job0: submit (cycle 1) -> poll running (cycle 2) -> poll complete ->
    # collect -> idle.  job1: submit -> poll complete -> collect -> idle.
    # 8 cycles is generous.
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=8)

    import types
    try:
        ParallelScheduler._submit = _wrapped_submit
        sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
        summary = sched.run(
            manifest, state, max_parallel=4, poll_interval=1,
            watch_batch=True, watch_interval=1,
        )
    finally:
        ParallelScheduler._submit = original_submit

    # Both jobs should be in the summary.
    assert "job0" in summary["jobs"]
    assert "job1" in summary["jobs"]
    # job0 completed successfully.
    assert summary["jobs"]["job0"]["state"] == SUCCEEDED
    # job1 was merged and also completed.
    assert summary["jobs"]["job1"]["state"] == SUCCEEDED
    assert summary["all_succeeded"]


def test_watch_does_not_mutate_existing_job_state_on_reload(
    tmp_path: Path,
) -> None:
    """Reloading the batch file must not mutate existing job definitions
    or lifecycle state.  An existing job that has SUCCEEDED must remain
    SUCCEEDED with its original attempts/attempts count, even if the batch
    file is rewritten with a different job order or additional jobs.

    This test starts with 1 job, lets it succeed, then appends a 2nd job.
    The 1st job's state/attempts in the final summary must be unchanged
    by the reload.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"],
                          "owner/k1": ["complete"]},
    )

    appended = {"done": False}
    original_submit = ParallelScheduler._submit

    def _wrapped_submit(self, job, push_timeout):
        result = original_submit(self, job, push_timeout)
        if not appended["done"] and job.name == "job0":
            _append_job_to_batch(tmp_path, manifest, "job1", "owner/k1")
            appended["done"] = True
        return result

    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=8)

    try:
        ParallelScheduler._submit = _wrapped_submit
        sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
        summary = sched.run(
            manifest, state, max_parallel=4, poll_interval=1,
            watch_batch=True, watch_interval=1,
        )
    finally:
        ParallelScheduler._submit = original_submit

    # job0 succeeded with exactly 1 attempt (not re-submitted by reload).
    assert summary["jobs"]["job0"]["state"] == SUCCEEDED
    assert summary["jobs"]["job0"]["attempts"] == 1
    # job1 was added as a fresh pending job.
    assert summary["jobs"]["job1"]["state"] == SUCCEEDED
    assert summary["jobs"]["job1"]["attempts"] == 1


def test_watch_malformed_reload_retried_then_valid_content_succeeds(
    tmp_path: Path,
) -> None:
    """When the batch file is malformed (invalid JSON) during a reload,
    the scheduler logs the error, does NOT update watch_state, and retries
    on the next cycle.  When the file is later written with valid content
    (a new job), the merge succeeds and the new job is executed.

    Sequence:
    1. Start with 1 job (job0).  Let it complete.
    2. While idle, write invalid JSON to the batch file.
    3. The scheduler detects a change but load_batch fails — it logs and
       retries (watch_state unchanged).
    4. Write valid JSON with job0 + job1.
    5. The scheduler retries, merges job1, executes it.
    6. KeyboardInterrupt terminates after both jobs complete.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"],
                          "owner/k1": ["complete"]},
    )

    phase = {"step": 0}

    class _StagedSleeper:
        """Sleeper that mutates the batch file at strategic points and
        eventually raises KeyboardInterrupt."""

        def __init__(self) -> None:
            self.calls: list[float] = []

        def __call__(self, seconds: float) -> None:
            self.calls.append(seconds)
            clock_h[0] += seconds
            phase["step"] += 1
            # After job0 completes and the scheduler goes idle, corrupt
            # the batch file on the first idle sleep.  The scheduler will
            # call _maybe_reload_batch before sleeping, so we need to
            # corrupt the file BEFORE that check.  We do it on the 2nd
            # sleep call (after job0 has been submitted and polled).
            if phase["step"] == 2:
                # Write invalid JSON to the batch file.
                manifest.write_text("{ invalid json !!!", encoding="utf-8")
            # On the 3rd sleep, write valid JSON with job0 + job1.
            if phase["step"] == 3:
                # Write valid JSON from scratch with job0 + job1.
                # (Cannot use _append_job_to_batch because the file is
                # currently corrupted with invalid JSON.)
                _make_kernel_dir(tmp_path, "job1", "owner/k1")
                manifest.write_text(
                    json.dumps({
                        "schema_version": EXPECTED_BATCH_SCHEMA,
                        "jobs": [
                            {
                                "name": "job0",
                                "provider": PROVIDER_KAGGLE,
                                "kernel_dir": "job0",
                                "output_dir": "out/job0",
                                "runtime_manifest": _valid_runtime_manifest(),
                            },
                            {
                                "name": "job1",
                                "provider": PROVIDER_KAGGLE,
                                "kernel_dir": "job1",
                                "output_dir": "out/job1",
                                "runtime_manifest": _valid_runtime_manifest(),
                            },
                        ],
                    }),
                    encoding="utf-8",
                )
            # After enough cycles, terminate.
            if phase["step"] >= 10:
                raise KeyboardInterrupt()

    sleeper = _StagedSleeper()
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1,
        watch_batch=True, watch_interval=1,
    )

    # job0 succeeded.
    assert summary["jobs"]["job0"]["state"] == SUCCEEDED
    # job1 was merged after the malformed retry and succeeded.
    assert "job1" in summary["jobs"]
    assert summary["jobs"]["job1"]["state"] == SUCCEEDED


def test_watch_stays_alive_across_empty_interval_then_terminated(
    tmp_path: Path,
) -> None:
    """Watch mode must NOT terminate when all jobs are done.  It stays
    alive across idle intervals (sleeping watch_interval) until
    KeyboardInterrupt.  This test verifies the scheduler survives at
    least one full idle cycle before the sleeper raises KeyboardInterrupt.

    A pre-change scheduler that breaks on idle would exit before the
    sleeper raises, producing a summary with no KeyboardInterrupt-induced
    state.  We verify the sleeper was called at least twice (once for
    poll, once for idle watch_interval) before termination.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    # 5 calls: 1 for poll, then at least 2 idle watch intervals, then
    # KeyboardInterrupt on the 5th.
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=5)
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1,
        watch_batch=True, watch_interval=1,
    )

    # job0 succeeded.
    assert summary["jobs"]["job0"]["state"] == SUCCEEDED
    # The sleeper was called at least 3 times (poll + at least 2 idle).
    # A pre-change scheduler that breaks on idle would have called the
    # sleeper fewer times (only during active jobs).
    assert len(sleeper.calls) >= 3, (
        f"expected >=3 sleeper calls (poll + idle watch), "
        f"got {len(sleeper.calls)}: {sleeper.calls}"
    )
    # State file is valid after Ctrl-C termination.
    data = json.loads(state.read_text())
    assert data["jobs"]["job0"]["state"] == SUCCEEDED


def test_non_watch_mode_still_terminates(tmp_path: Path) -> None:
    """Without watch_batch, the scheduler must terminate when all jobs
    reach terminal state — identical to historical behavior.  The sleeper
    must NOT be called enough to trigger KeyboardInterrupt, and the run
    returns normally.

    This is the backward-compatibility regression: watch_batch=False
    (default) must not enter the infinite watch loop.
    """
    manifest, _ = _make_batch_with_kernels(tmp_path, 2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    client = FakeKaggleClient(
        status_sequences={
            "owner/k0": ["complete"],
            "owner/k1": ["complete"],
        },
    )
    # A sleeper that would raise KeyboardInterrupt after 100 calls.
    # If non-watch mode terminates correctly, this never fires.
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=100)
    sched = ParallelScheduler(client, clock=clock, sleeper=sleeper)
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1,
        # watch_batch defaults to False
    )

    assert summary["all_succeeded"]
    assert summary["total"] == 2
    # The sleeper was called fewer than 100 times (no KeyboardInterrupt).
    assert len(sleeper.calls) < 100
    # No KeyboardInterrupt was raised — run returned normally.
    data = json.loads(state.read_text())
    assert all(
        data["jobs"][j]["state"] == SUCCEEDED for j in ("job0", "job1")
    )


def test_watch_cli_rejects_nonpositive_watch_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI --watch-batch with --watch-interval 0 must fail via SystemExit
    (parser.error)."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    client = FakeKaggleClient(status_sequences={"owner/k0": ["complete"]})
    with pytest.raises(SystemExit):
        main(
            ["run", str(manifest), "--state", str(state),
             "--watch-batch", "--watch-interval", "0"],
            client=client,
        )


def test_watch_cli_accepts_watch_batch_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI --watch-batch with a valid --watch-interval must be accepted
    by the parser (no SystemExit from parser.error).  The run itself is
    terminated via KeyboardInterrupt from the sleeper."""
    manifest, _ = _make_batch_with_kernels(tmp_path, 1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=5)
    client = FakeKaggleClient(
        status_sequences={"owner/k0": ["complete"]},
    )
    rc = main(
        ["run", str(manifest), "--state", str(state),
         "--watch-batch", "--watch-interval", "1", "--poll-interval", "1"],
        client=client, clock=clock, sleeper=sleeper,
    )
    # run() catches KeyboardInterrupt and returns a summary; main() prints
    # it and returns 0 if all_succeeded.
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["all_succeeded"]


def test_watch_default_watch_interval_constant() -> None:
    """The DEFAULT_WATCH_INTERVAL constant must exist and be 30.0 seconds."""
    assert DEFAULT_WATCH_INTERVAL is not None, (
        "DEFAULT_WATCH_INTERVAL not defined in scheduler module"
    )
    assert DEFAULT_WATCH_INTERVAL == 30.0
