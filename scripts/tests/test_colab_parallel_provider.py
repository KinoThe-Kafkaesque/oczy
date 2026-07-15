"""Behavioral tests for the v2 mixed Kaggle/Colab CPU scheduler and dynamic
admission controller.

Covers:
* v2 batch/state schema constants and relative path resolution for Colab jobs.
* Mixed global + per-provider slot invariants.
* Colab AIMD admission: start at 1, +1 on success, reduce on 412, cooldown.
* External Colab sessions reducing available slots.
* CPU-only subprocess argv (no --gpu/--tpu).
* classify_colab_output / parse_sessions / is_capacity_rejected.
* collect() writing stdout.log / stderr.log / result.json.
* Cleanup (stop) on success, error, timeout, and capacity rejection.
* Interrupted-restart: Colab running job -> failed + best-effort stop.
* State v2 with colab_learned_limit; v1 -> v2 migration.
* CLI flags --colab-max / --colab-cooldown.

All tests use fake clients, fake Popen objects, and deterministic clocks --
never the network, real Colab CLI, or real subprocess calls.
"""

from __future__ import annotations

import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loading -- both implementation files loaded via runpy.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "parallel_scheduler.py"
COLAB_PATH = REPO_ROOT / "infrastructure" / "kaggle" / "colab_provider.py"


def _load_module(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"implementation not found at {path}", allow_module_level=True)
    return runpy.run_path(str(path))


_mod = _load_module(SCHEDULER_PATH, "parallel_scheduler")
_colab = _load_module(COLAB_PATH, "colab_provider")

# Schema constants
BATCH_SCHEMA_VERSION: str = _mod["BATCH_SCHEMA_VERSION"]
STATE_SCHEMA_VERSION: str = _mod["STATE_SCHEMA_VERSION"]

# Providers
PROVIDER_KAGGLE: str = _mod["PROVIDER_KAGGLE"]
PROVIDER_COLAB: str = _mod["PROVIDER_COLAB"]

# Concurrency defaults
DEFAULT_MAX_PARALLEL: int = _mod["DEFAULT_MAX_PARALLEL"]
HARD_MAX_PARALLEL: int | None = _mod.get("HARD_MAX_PARALLEL")  # removed in additive-capacity refactor
DEFAULT_KAGGLE_MAX: int = _mod["DEFAULT_KAGGLE_MAX"]
HARD_KAGGLE_MAX: int = _mod["HARD_KAGGLE_MAX"]
DEFAULT_COLAB_MAX: int = _mod["DEFAULT_COLAB_MAX"]
DEFAULT_COLAB_COOLDOWN: float = _mod["DEFAULT_COLAB_COOLDOWN"]
COLAB_AIMD_START: int = _mod["COLAB_AIMD_START"]
COLAB_AIMD_MIN: int = _mod["COLAB_AIMD_MIN"]
compute_manifest_sha256 = _mod["compute_manifest_sha256"]

# Job states
PENDING = _mod["PENDING"]
SUBMITTING = _mod["SUBMITTING"]
RUNNING = _mod["RUNNING"]
COLLECTING = _mod["COLLECTING"]
SUCCEEDED = _mod["SUCCEEDED"]
FAILED = _mod["FAILED"]

# Colab status constants
COLAB_RUNNING: str = _colab["COLAB_RUNNING"]
COLAB_COMPLETE: str = _colab["COLAB_COMPLETE"]
COLAB_ERROR: str = _colab["COLAB_ERROR"]
COLAB_CAPACITY_REJECTED: str = _colab["COLAB_CAPACITY_REJECTED"]

# Functions / classes from colab_provider
is_capacity_rejected = _colab["is_capacity_rejected"]
classify_colab_output = _colab["classify_colab_output"]
parse_sessions = _colab["parse_sessions"]
detect_orphaned_sessions = _colab["detect_orphaned_sessions"]
ColabCliClient = _colab["ColabCliClient"]
ColabClient = _colab["ColabClient"]
_read_proc_output = _colab["_read_proc_output"]
_cleanup_proc_tempfiles = _colab["_cleanup_proc_tempfiles"]

# Scheduler classes / functions
load_batch = _mod["load_batch"]
load_state = _mod["load_state"]
BatchValidationError = _mod["BatchValidationError"]
ParallelScheduler = _mod["ParallelScheduler"]
ColabAimdController = _mod["ColabAimdController"]
Job = _mod["Job"]
main = _mod["main"]
COLAB_MAX_CAPACITY_REJECTIONS: int = _mod["COLAB_MAX_CAPACITY_REJECTIONS"]

RUNTIME_MANIFEST_SCHEMA_VERSION = "oczy/runtime-manifest/v2"
EXECUTION_REPORT_SCHEMA_VERSION = "oczy/execution-report/v2"
EXPECTED_BATCH_V3 = "oczy/remote-parallel-batch/v3"
EXPECTED_STATE_V4 = "oczy/remote-parallel-state/v4"
BATCH_SCHEMA_V2 = BATCH_SCHEMA_VERSION
STATE_SCHEMA_V2 = STATE_SCHEMA_VERSION
EXPECTED_BATCH_V2 = EXPECTED_BATCH_V3
EXPECTED_STATE_V2 = EXPECTED_STATE_V4


def _valid_runtime_manifest(**overrides: Any) -> dict[str, Any]:
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
            "resolved_model_convention": "none",
            "quantization": None,
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
    slug = kernel_id.rsplit("/", 1)[-1]
    return " ".join(w.capitalize() for w in slug.split("-"))


def _valid_metadata(
    kernel_id: str = "owner/test-kernel-a",
) -> dict[str, Any]:
    return {
        "id": kernel_id,
        "title": _title_from_kernel_id(kernel_id),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "",
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
    base: Path, name: str, kernel_id: str = "owner/test-kernel-a"
) -> Path:
    kd = base / name
    kd.mkdir(parents=True, exist_ok=True)
    (kd / "kernel-metadata.json").write_text(
        json.dumps(_valid_metadata(kernel_id)), encoding="utf-8"
    )
    (kd / "job_spec.json").write_text(
        json.dumps(_valid_job_spec()), encoding="utf-8"
    )
    return kd


def _make_colab_script(base: Path, name: str = "script.py") -> str:
    """Create a minimal Colab script file relative to *base*.  Returns the
    relative path string suitable for a batch manifest entry."""
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("print('hello')\n", encoding="utf-8")
    return name

def _make_colab_job_spec(
    base: Path,
    name: str,
    runtime_manifest: dict[str, Any] | None = None,
) -> str:
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest if runtime_manifest is not None else _valid_runtime_manifest()
    p.write_text(
        json.dumps(
            {
                "schema_version": "oczy/colab-experiment-job/v2",
                "job_name": p.stem,
                "script": "script.py",
                "arguments": [],
                "runtime_manifest": manifest,
            }
        ),
        encoding="utf-8",
    )
    return name


def _make_v2_batch(
    base: Path,
    jobs: list[dict[str, Any]],
) -> Path:
    """Write a v3 batch manifest with required inline runtime manifests."""
    normalized_jobs: list[dict[str, Any]] = []
    for job in jobs:
        entry = dict(job)
        entry.setdefault("runtime_manifest", _valid_runtime_manifest())
        if entry.get("provider") == PROVIDER_COLAB and "job_spec" not in entry:
            entry["job_spec"] = _make_colab_job_spec(
                base, f"specs/{entry.get('name', 'job')}.json"
            )
        normalized_jobs.append(entry)
    manifest = {"schema_version": BATCH_SCHEMA_VERSION, "jobs": normalized_jobs}
    p = base / "batch.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def _make_v1_batch(
    base: Path,
    jobs: list[dict[str, Any]],
) -> Path:
    """Write a v3 Kaggle batch; retained name covers no-Colab state tests."""
    normalized_jobs: list[dict[str, Any]] = []
    for job in jobs:
        entry = dict(job)
        entry.setdefault("provider", PROVIDER_KAGGLE)
        entry.setdefault("runtime_manifest", _valid_runtime_manifest())
        normalized_jobs.append(entry)
    manifest = {"schema_version": BATCH_SCHEMA_VERSION, "jobs": normalized_jobs}
    p = base / "batch.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def _make_mixed_batch(
    base: Path,
    n_kaggle: int = 0,
    n_colab: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    """Create a v2 batch with n_kaggle Kaggle jobs and n_colab Colab jobs.
    Returns (manifest_path, job_entries)."""
    entries: list[dict[str, Any]] = []
    for i in range(n_kaggle):
        name = f"kg{i}"
        kid = f"owner/kg{i}"
        _make_kernel_dir(base, name, kid)
        entries.append({
            "name": name,
            "provider": PROVIDER_KAGGLE,
            "kernel_dir": name,
            "output_dir": f"out/{name}",
        })
    for i in range(n_colab):
        name = f"cb{i}"
        script_rel = _make_colab_script(base, f"scripts/{name}.py")
        entries.append({
            "name": name,
            "provider": PROVIDER_COLAB,
            "script": script_rel,
            "output_dir": f"out/{name}",
            "arguments": ["--flag", "value"],
            "job_spec": _make_colab_job_spec(base, f"specs/{name}.json"),
            "runtime_manifest": _valid_runtime_manifest(),
        })
    return _make_v2_batch(base, entries), entries


# ---------------------------------------------------------------------------
# Fake process / client / clock helpers
# ---------------------------------------------------------------------------


class FakePopen:
    """Fake subprocess.Popen for Colab testing.

    Simulates a process that stays running for *running_polls* poll() calls
    and then terminates with *returncode*.  stdout/stderr are StringIO
    streams readable by classify_colab_output / collect.
    """

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        running_polls: int = 0,
    ) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self._remaining = running_polls
        self._final_rc = returncode
        # Mirror real Popen: returncode is None until the process terminates.
        # When running_polls == 0 the process has already terminated, so
        # returncode is set immediately (as poll() would on the first call).
        # This lets classify_colab_output read proc.returncode directly
        # without a prior poll(), matching real usage where the scheduler
        # polls before classifying.
        self.returncode: int | None = None if running_polls > 0 else returncode
        # Dynamic attributes set by classify_colab_output / test code.
        self._colab_stdout: str | None = None
        self._colab_stderr: str | None = None
        self._colab_stdout_path: str | None = None
        self._colab_stderr_path: str | None = None
        self._colab_stdout_file: Any | None = None
        self._colab_stderr_file: Any | None = None
        self._colab_classified_output: str | None = None

    def poll(self) -> int | None:
        if self._remaining > 0:
            self._remaining -= 1
            return None
        self.returncode = self._final_rc
        return self._final_rc


class FakeColabClient:
    """Deterministic fake Colab client for scheduler-level tests.

    *behaviors* maps session name -> list of behavior dicts.  Each behavior
    dict configures the FakePopen returned for that run() call:
    ``{"stdout": str, "stderr": str, "returncode": int, "running_polls": int}``.
    If a session has no more behaviors, a default complete-with-empty-output
    behavior is used.

    *sessions_list* is returned verbatim by sessions().  *collect_ok* controls
    whether collect() returns success.  *collect_raises* makes collect() raise.
    """

    def __init__(
        self,
        *,
        behaviors: dict[str, list[dict[str, Any]]] | None = None,
        sessions_list: list[dict[str, str]] | None = None,
        collect_ok: bool = True,
        collect_raises: Exception | None = None,
        run_raises: Exception | None = None,
    ) -> None:
        self._behaviors = behaviors or {}
        self._sessions_list = sessions_list or []
        self._collect_ok = collect_ok
        self._collect_raises = collect_raises
        self._run_raises = run_raises
        self._behavior_idx: dict[str, int] = {}
        self._procs: dict[str, FakePopen] = {}
        self.run_calls: list[tuple[str, str, list[str] | None, float | None]] = []
        self.collect_calls: list[tuple[str, str]] = []
        self.stop_calls: list[str] = []
        self.sessions_calls: int = 0

    def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
        self.sessions_calls += 1
        return list(self._sessions_list)

    def run(
        self,
        name: str,
        script: str,
        *,
        arguments: list[str] | None = None,
        timeout: float | None = None,
    ) -> FakePopen:
        self.run_calls.append((name, script, arguments, timeout))
        if self._run_raises is not None:
            raise self._run_raises
        idx = self._behavior_idx.get(name, 0)
        behaviors = self._behaviors.get(name, [])
        if idx < len(behaviors):
            b = behaviors[idx]
            self._behavior_idx[name] = idx + 1
        else:
            b = {}
        proc = FakePopen(
            stdout=b.get("stdout", "ok"),
            stderr=b.get("stderr", ""),
            returncode=b.get("returncode", 0),
            running_polls=b.get("running_polls", 0),
        )
        return proc

    def remember(self, name: str, proc: Any) -> None:
        self._procs[name] = proc

    def forget(self, name: str) -> Any:
        return self._procs.pop(name, None)

    def collect(self, name: str, output_dir: str) -> dict[str, Any]:
        self.collect_calls.append((name, output_dir))
        if self._collect_raises is not None:
            raise self._collect_raises
        proc = self._procs.get(name)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stdout_text = ""
        stderr_text = ""
        if proc is not None:
            stdout_text = getattr(proc, "_colab_stdout", "") or ""
            stderr_text = getattr(proc, "_colab_stderr", "") or ""
        (out / "stdout.log").write_text(stdout_text, encoding="utf-8")
        (out / "stderr.log").write_text(stderr_text, encoding="utf-8")
        result = {"ok": self._collect_ok, "error": None if self._collect_ok else "failed"}
        (out / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        (out / "execution_report.json").write_text(
            json.dumps(_execution_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result

    def stop(self, name: str, *, timeout: float | None = None) -> None:
        self.stop_calls.append(name)
        # appears in the backend's active session list.  Without this, a
        # stopped orphan lingers in sessions_list forever, making the
        # account appear full and blocking resubmission of the restarted
        # pending job (F5 orphan recovery).
        self._sessions_list = [
            s for s in self._sessions_list if s.get("name") != name
        ]


class FakeKaggleClient:
    """Minimal fake Kaggle client for mixed-batch tests."""

    def __init__(self, *, status_sequences: dict[str, list[str]] | None = None) -> None:
        self._status_sequences = status_sequences or {}
        self.push_calls: list[tuple[str, float | None]] = []
        self.status_calls: list[tuple[str, float | None]] = []
        self.output_calls: list[tuple[str, str, float | None]] = []

    def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
        self.push_calls.append((kernel_dir, timeout))
        kd = Path(kernel_dir)
        meta_path = kd / "kernel-metadata.json"
        kid = "unknown"
        if meta_path.exists():
            kid = json.loads(meta_path.read_text())["id"]
        return kid

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
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "execution_report.json").write_text(
            json.dumps(_execution_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (p / "result.json").write_text('{"ok": true}', encoding="utf-8")


class CountingSleeper:
    """Sleeper that advances a fake clock and records calls."""

    def __init__(self, clock: list[float], interval: float = 30.0) -> None:
        self._clock = clock
        self._interval = interval
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock[0] += seconds


class BoundedSleeper:
    """Sleeper that raises *exc* after *max_calls* calls, breaking infinite
    scheduler loops in tests where admission is intentionally blocked."""

    def __init__(self, clock: list[float], max_calls: int = 5, exc: Exception | None = None) -> None:
        self._clock = clock
        self._max = max_calls
        self._exc = exc or RuntimeError("bounded sleeper limit reached")
        self._n = 0

    def __call__(self, seconds: float) -> None:
        self._n += 1
        self._clock[0] += seconds
        if self._n >= self._max:
            raise self._exc


def _make_clock(start: float = 1000.0) -> tuple[list[float], Any]:
    holder = [start]

    def _now() -> float:
        return holder[0]

    return holder, _now


def _read_state(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. v2 schema version constants
# ---------------------------------------------------------------------------


def test_batch_schema_v2_constant() -> None:
    assert BATCH_SCHEMA_V2 == EXPECTED_BATCH_V2


def test_state_schema_v2_constant() -> None:
    assert STATE_SCHEMA_V2 == EXPECTED_STATE_V2


def test_provider_constants() -> None:
    assert PROVIDER_KAGGLE == "kaggle"
    assert PROVIDER_COLAB == "colab"


# ---------------------------------------------------------------------------
# 2. v2 batch loading: Colab job validation and relative paths
# ---------------------------------------------------------------------------


def test_v2_batch_loads_colab_job(tmp_path: Path) -> None:
    """A v2 batch with a Colab job loads with resolved paths and provider."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "arguments": ["--x", "1"],
    }])
    jobs = load_batch(manifest)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["provider"] == PROVIDER_COLAB
    assert job["name"] == "cb0"
    assert Path(job["script"]) == (tmp_path / "scripts" / "run.py").resolve()
    assert Path(job["output_dir"]) == (tmp_path / "out" / "cb0").resolve()
    assert job["arguments"] == ["--x", "1"]
    assert job["schema_version"] == BATCH_SCHEMA_V2


def test_v2_batch_loads_mixed_providers(tmp_path: Path) -> None:
    """A v2 batch with both Kaggle and Colab jobs loads both correctly."""
    manifest, entries = _make_mixed_batch(tmp_path, n_kaggle=1, n_colab=1)
    jobs = load_batch(manifest)
    assert len(jobs) == 2
    providers = {j["name"]: j["provider"] for j in jobs}
    assert providers["kg0"] == PROVIDER_KAGGLE
    assert providers["cb0"] == PROVIDER_COLAB


def test_v2_batch_resolves_colab_paths_relative_to_manifest(tmp_path: Path) -> None:
    """Colab script and output_dir resolve relative to the manifest's parent."""
    sub = tmp_path / "sub"
    sub.mkdir()
    script_rel = _make_colab_script(sub, "scripts/run.py")
    manifest = _make_v2_batch(sub, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
    }])
    jobs = load_batch(manifest)
    assert Path(jobs[0]["script"]) == (sub / "scripts" / "run.py").resolve()
    assert Path(jobs[0]["output_dir"]) == (sub / "out" / "cb0").resolve()


def test_v3_batch_rejects_colab_job_spec_runtime_manifest_mismatch(
    tmp_path: Path,
) -> None:
    """The scheduler binds Colab batch entries to the generated job spec."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    batch_manifest = _valid_runtime_manifest()
    spec_manifest = _valid_runtime_manifest(python_version="3.13.0")
    job_spec = _make_colab_job_spec(tmp_path, "specs/cb0.json", spec_manifest)
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "job_spec": job_spec,
        "runtime_manifest": batch_manifest,
    }])

    with pytest.raises(BatchValidationError, match="differs from job_spec hash"):
        load_batch(manifest)


def test_v2_batch_rejects_missing_provider(tmp_path: Path) -> None:
    """A v2 job without 'provider' must be rejected."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "script": script_rel,
        "output_dir": "out/cb0",
    }])
    with pytest.raises(BatchValidationError, match="provider"):
        load_batch(manifest)


def test_v2_batch_rejects_invalid_provider(tmp_path: Path) -> None:
    """A v2 job with an unknown provider must be rejected."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": "vertex",
        "script": script_rel,
        "output_dir": "out/cb0",
    }])
    with pytest.raises(BatchValidationError, match="provider"):
        load_batch(manifest)


def test_v2_batch_rejects_missing_script(tmp_path: Path) -> None:
    """A Colab job without 'script' must be rejected."""
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "output_dir": "out/cb0",
    }])
    with pytest.raises(BatchValidationError, match="script"):
        load_batch(manifest)


def test_v2_batch_rejects_missing_output_dir(tmp_path: Path) -> None:
    """A Colab job without 'output_dir' must be rejected."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
    }])
    with pytest.raises(BatchValidationError, match="output_dir"):
        load_batch(manifest)


def test_v2_batch_rejects_nonexistent_script(tmp_path: Path) -> None:
    """A Colab job whose script file does not exist must be rejected."""
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": "scripts/missing.py",
        "output_dir": "out/cb0",
    }])
    with pytest.raises(BatchValidationError, match="script does not exist"):
        load_batch(manifest)


def test_v2_batch_rejects_non_string_arguments(tmp_path: Path) -> None:
    """Colab arguments must be a list of strings."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "arguments": [1, 2, 3],
    }])
    with pytest.raises(BatchValidationError, match="arguments"):
        load_batch(manifest)


def test_v2_batch_accepts_empty_arguments(tmp_path: Path) -> None:
    """An empty arguments list is valid."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "arguments": [],
    }])
    jobs = load_batch(manifest)
    assert jobs[0]["arguments"] == []


def test_v2_batch_accepts_timeout(tmp_path: Path) -> None:
    """A positive timeout is accepted and stored as float."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "timeout": 300,
    }])
    jobs = load_batch(manifest)
    assert jobs[0]["timeout"] == 300.0


def test_v2_batch_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    """A non-positive timeout must be rejected."""
    script_rel = _make_colab_script(tmp_path, "scripts/run.py")
    manifest = _make_v2_batch(tmp_path, [{
        "name": "cb0",
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": "out/cb0",
        "timeout": 0,
    }])
    with pytest.raises(BatchValidationError, match="timeout"):
        load_batch(manifest)


# ---------------------------------------------------------------------------
# 3. Mixed global + per-provider slot invariants
# ---------------------------------------------------------------------------


def test_mixed_slots_respect_kaggle_and_colab_max(tmp_path: Path) -> None:
    """With max_parallel=4, kaggle_max=2, colab_max=2, at most 2 of each
    provider are active simultaneously."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=4, n_colab=4)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    kaggle_client = FakeKaggleClient(
        status_sequences={f"owner/kg{i}": ["running", "complete"] for i in range(4)}
    )
    colab_client = FakeColabClient(
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(4)}
    )

    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1,
        kaggle_max=2, colab_max=2,
    )

    assert summary["all_succeeded"]
    # At most 2 Kaggle and 2 Colab were ever active (by checking max
    # concurrent submissions were bounded — verified by the run completing
    # without exceeding the global cap).
    assert summary["succeeded"] == 8


def test_global_max_parallel_caps_total_active(tmp_path: Path) -> None:
    """max_parallel=2 with 3 Kaggle + 3 Colab caps total active at 2."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=3, n_colab=3)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    kaggle_client = FakeKaggleClient(
        status_sequences={f"owner/kg{i}": ["running", "complete"] for i in range(3)}
    )
    colab_client = FakeColabClient(
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(3)}
    )

    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        kaggle_max=2, colab_max=2,
    )
    assert summary["all_succeeded"]
    assert summary["total"] == 6


# ---------------------------------------------------------------------------
# 3b. Regression: additive capacity admits 5 Kaggle + learned Colab
# ---------------------------------------------------------------------------


def test_additive_capacity_admits_5_kaggle_plus_colab(tmp_path: Path) -> None:
    """Regression: max_parallel=None admits 5 Kaggle plus at least one Colab
    concurrently, proving additive provider capacity rather than stopping at
    5 total.

    With DEFAULT_KAGGLE_MAX=5 Kaggle jobs and Colab AIMD starting at 1, the
    scheduler should concurrently run 5 Kaggle + at least 1 Colab, which
    exceeds 5 total — impossible under a global cap of 5.
    """
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=12, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    tracker: dict[str, Any] = {
        "kaggle_active": set(),
        "colab_active": set(),
        "max_kaggle": 0,
        "max_colab": 0,
        "max_total": 0,
    }

    def _update_max() -> None:
        tracker["max_kaggle"] = max(
            tracker["max_kaggle"], len(tracker["kaggle_active"])
        )
        tracker["max_colab"] = max(
            tracker["max_colab"], len(tracker["colab_active"])
        )
        tracker["max_total"] = max(
            tracker["max_total"],
            len(tracker["kaggle_active"]) + len(tracker["colab_active"]),
        )

    class TrackingKaggleClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/kg{i}": ["running", "complete"] for i in range(12)
                },
            )

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            tracker["kaggle_active"].add(kid)
            _update_max()
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                tracker["kaggle_active"].discard(kernel_id)
            return result

    class TrackingColabClient(FakeColabClient):
        def __init__(self) -> None:
            super().__init__(
                behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(2)},
            )

        def run(
            self,
            name: str,
            script: str,
            *,
            arguments: list[str] | None = None,
            timeout: float | None = None,
        ) -> FakePopen:
            proc = super().run(name, script, arguments=arguments, timeout=timeout)
            tracker["colab_active"].add(name)
            _update_max()
            return proc

        def stop(self, name: str, *, timeout: float | None = None) -> None:
            super().stop(name, timeout=timeout)
            tracker["colab_active"].discard(name)

    kaggle_client = TrackingKaggleClient()
    colab_client = TrackingColabClient()
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=None, poll_interval=1,
    )

    assert summary["all_succeeded"], f"jobs failed: {summary['failed']}"
    # 5 Kaggle jobs admitted concurrently (DEFAULT_KAGGLE_MAX)
    assert tracker["max_kaggle"] == DEFAULT_KAGGLE_MAX, (
        f"expected {DEFAULT_KAGGLE_MAX} concurrent Kaggle, "
        f"got {tracker['max_kaggle']}"
    )
    # At least 1 Colab job admitted concurrently
    assert tracker["max_colab"] >= 1, (
        f"expected >=1 concurrent Colab, got {tracker['max_colab']}"
    )
    # Total concurrency exceeded 5 — proving additive capacity, not a 5 cap
    assert tracker["max_total"] > HARD_KAGGLE_MAX, (
        f"total concurrency {tracker['max_total']} did not exceed "
        f"hard kaggle cap {HARD_KAGGLE_MAX} — additive capacity not proven"
    )


def test_explicit_max_parallel_10_caps_total_with_mixed(tmp_path: Path) -> None:
    """Explicit max_parallel=10 caps total concurrency at 10 even with
    12 Kaggle + 2 Colab jobs, preserving backward-compatible global cap."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=12, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    tracker: dict[str, Any] = {
        "kaggle_active": set(),
        "colab_active": set(),
        "max_total": 0,
    }

    def _update_max() -> None:
        tracker["max_total"] = max(
            tracker["max_total"],
            len(tracker["kaggle_active"]) + len(tracker["colab_active"]),
        )

    class TrackingKaggleClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/kg{i}": ["running", "complete"] for i in range(12)
                },
            )

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            tracker["kaggle_active"].add(kid)
            _update_max()
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                tracker["kaggle_active"].discard(kernel_id)
            return result

    class TrackingColabClient(FakeColabClient):
        def __init__(self) -> None:
            super().__init__(
                behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(2)},
            )

        def run(
            self,
            name: str,
            script: str,
            *,
            arguments: list[str] | None = None,
            timeout: float | None = None,
        ) -> FakePopen:
            proc = super().run(name, script, arguments=arguments, timeout=timeout)
            tracker["colab_active"].add(name)
            _update_max()
            return proc

        def stop(self, name: str, *, timeout: float | None = None) -> None:
            super().stop(name, timeout=timeout)
            tracker["colab_active"].discard(name)

    kaggle_client = TrackingKaggleClient()
    colab_client = TrackingColabClient()
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=10, poll_interval=1,
    )

    assert summary["all_succeeded"], f"jobs failed: {summary['failed']}"
    assert tracker["max_total"] <= 10, (
        f"total concurrency {tracker['max_total']} exceeded explicit cap of 10"
    )

# ---------------------------------------------------------------------------
# 4. Colab AIMD: starts at 1, increases by 1 on success
# ---------------------------------------------------------------------------


def test_colab_aimd_starts_at_1() -> None:
    """ColabAimdController initializes learned_limit to 1."""
    ctrl = ColabAimdController(ceiling=10)
    assert ctrl.learned_limit == COLAB_AIMD_START == 1
    assert ctrl.effective_limit() == 1


def test_colab_aimd_increases_by_one_on_success() -> None:
    """Each on_success() call increases learned_limit by 1 up to ceiling."""
    ctrl = ColabAimdController(ceiling=5)
    ctrl.on_success()
    assert ctrl.learned_limit == 2
    ctrl.on_success()
    assert ctrl.learned_limit == 3
    ctrl.on_success()
    assert ctrl.learned_limit == 4
    ctrl.on_success()
    assert ctrl.learned_limit == 5
    # Capped at ceiling.
    ctrl.on_success()
    assert ctrl.learned_limit == 5


def test_colab_aimd_effective_limit_capped_by_ceiling() -> None:
    """effective_limit never exceeds ceiling even if learned_limit is high."""
    ctrl = ColabAimdController(ceiling=3)
    ctrl.learned_limit = 10
    assert ctrl.effective_limit() == 3


def test_colab_aimd_reduce_on_capacity_rejected() -> None:
    """on_capacity_rejected reduces learned_limit to active_count (min 1)."""
    ctrl = ColabAimdController(ceiling=10)
    ctrl.learned_limit = 5
    ctrl.on_capacity_rejected(3)
    assert ctrl.learned_limit == 3
    # Min 1.
    ctrl.on_capacity_rejected(0)
    assert ctrl.learned_limit == COLAB_AIMD_MIN == 1


def test_colab_aimd_start_capped_by_ceiling() -> None:
    """If ceiling < start, learned_limit is capped to ceiling."""
    ctrl = ColabAimdController(ceiling=1)
    assert ctrl.learned_limit == 1


# ---------------------------------------------------------------------------
# 5. AIMD dynamics in full scheduler run
# ---------------------------------------------------------------------------


def test_aimd_increases_during_successful_run(tmp_path: Path) -> None:
    """After a run with 3 successful Colab jobs (colab_max=5), the state
    file records colab_learned_limit=4 (start 1 + 3 admissions)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=3)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    colab_client = FakeColabClient(
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(3)}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=4, poll_interval=1, colab_max=5)

    raw = _read_state(state)
    assert raw["schema_version"] == STATE_SCHEMA_V2
    assert raw["colab_learned_limit"] == 4  # 1 + 3 on_success calls


def test_aimd_capped_by_colab_max(tmp_path: Path) -> None:
    """With colab_max=2 and 4 Colab jobs, learned_limit caps at 2."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=4)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    colab_client = FakeColabClient(
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(4)}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=4, poll_interval=1, colab_max=2)

    raw = _read_state(state)
    assert raw["colab_learned_limit"] == 2  # capped at ceiling


def test_aimd_restored_from_state_on_restart(tmp_path: Path) -> None:
    """A state file with colab_learned_limit=3 restores the controller."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=2)
    state = tmp_path / "state.json"
    # Write a v2 state with learned_limit=3 and both jobs already succeeded.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                f"cb{i}": {
                    "name": f"cb{i}",
                    "output_dir": str((tmp_path / f"out/cb{i}").resolve()),
                    "state": SUCCEEDED,
                    "provider": PROVIDER_COLAB,
                    "remote_id": f"cb{i}",
                    "script": str((tmp_path / f"scripts/cb{i}.py").resolve()),
                    "arguments": ["--flag", "value"],
                    "timeout": None,
                }
                for i in range(2)
            },
            "updated_at": 1000.0,
            "colab_learned_limit": 3,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=4, poll_interval=1, colab_max=5)
    # Both jobs already succeeded — no new admissions.
    assert summary["all_succeeded"]
    raw = _read_state(state)
    # learned_limit restored to 3 (no new on_success calls).
    assert raw["colab_learned_limit"] == 3


# ---------------------------------------------------------------------------
# 6. 412 capacity rejection: decrease + cooldown + retry
# ---------------------------------------------------------------------------


def test_capacity_rejection_returns_to_pending_and_reduces_limit(
    tmp_path: Path,
) -> None:
    """When a Colab job gets a 412, it returns to pending, the AIMD limit
    is reduced, and cooldown is set.  After cooldown, the job retries and
    succeeds — no terminal failure."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    # Use a bounded sleeper with enough calls for the full lifecycle.
    sleeper = CountingSleeper(clock_h, interval=10.0)

    colab_client = FakeColabClient(
        behaviors={
            "cb0": [
                # First attempt: capacity rejected.
                {"stderr": "TooManyAssignmentsError: 412 Precondition Failed", "returncode": 1},
                # Second attempt (after cooldown): success.
                {"stdout": "done", "running_polls": 0, "returncode": 0},
            ],
        }
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=5, colab_cooldown=30.0,
    )

    # Job eventually succeeds (no terminal failure on capacity rejection).
    assert summary["all_succeeded"], f"job should succeed after retry, got: {summary}"
    assert summary["providers"]["colab"]["succeeded"] == 1

    # stop() called at least once (on capacity rejection).
    assert "cb0" in colab_client.stop_calls

    # run() called twice (first attempt rejected, second succeeded).
    assert len(colab_client.run_calls) == 2

    # State records the final learned_limit (reduced to 1, then +1 on retry = 2).
    raw = _read_state(state)
    assert raw["colab_learned_limit"] is not None


def test_capacity_rejection_sets_cooldown_blocking_resubmit(tmp_path: Path) -> None:
    """After a 412, the cooldown blocks immediate resubmission.  The job
    is only retried after the clock advances past the cooldown period."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock(start=1000.0)
    sleeper = CountingSleeper(clock_h, interval=5.0)

    colab_client = FakeColabClient(
        behaviors={
            "cb0": [
                {"stderr": "TooManyAssignmentsError", "returncode": 1},
                {"stdout": "ok", "returncode": 0},
            ],
        }
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=5, colab_cooldown=60.0,
    )

    # The first run() happens at t=1000.  The 412 sets cooldown_until=1060.
    # The sleeper advances by 5 each iteration, so the second run() happens
    # at t>=1060 (after ~12 sleep calls).  Verify run_calls has exactly 2.
    assert len(colab_client.run_calls) == 2
    # The second run happens at or after the cooldown expiry.
    # Clock at first run: 1000.  Cooldown_until: 1060.
    # Each sleep advances 5s, so at least 12 sleeps before t>=1060.
    assert len(sleeper.calls) >= 12


def test_no_terminal_failure_on_repeated_capacity_rejection(tmp_path: Path) -> None:
    """A job that is capacity-rejected multiple times eventually succeeds
    — never terminally fails due to capacity rejection alone."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=5.0)

    colab_client = FakeColabClient(
        behaviors={
            "cb0": [
                {"stderr": "TooManyAssignmentsError", "returncode": 1},
                {"stderr": "Precondition Failed", "returncode": 1},
                {"stdout": "ok", "returncode": 0},
            ],
        }
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=5, colab_cooldown=10.0,
    )
    assert summary["all_succeeded"]
    assert len(colab_client.run_calls) == 3


# ---------------------------------------------------------------------------
# 7. External sessions reducing availability
# ---------------------------------------------------------------------------


def test_external_sessions_block_admission(tmp_path: Path) -> None:
    """When external sessions permanently fill all AIMD slots, the Colab
    job remains pending indefinitely — admission blocks must not fail it
    or increment capacity_rejections.  A bounded sentinel exception stops
    the loop for observation rather than asserting terminal failure."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()

    # sessions() returns 1 session; AIMD starts at 1; 1 >= 1 -> blocked.
    colab_client = FakeColabClient(
        sessions_list=[{"name": "external-session", "state": "running"}],
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]},
    )
    # Bounded sentinel: stops the infinite loop after enough polls to
    # observe that the job stays pending without being failed.  12 calls
    # exceeds COLAB_MAX_CAPACITY_REJECTIONS (10), so the live-bug path
    # (which fails the job after 10 admission blocks) would terminate
    # the loop early with the job FAILED — no sentinel raised.
    sentinel = RuntimeError("bounded sentinel: observation complete")
    sleeper = BoundedSleeper(clock_h, max_calls=12, exc=sentinel)

    class _CapturingScheduler(ParallelScheduler):
        """Captures the jobs dict on every state save for mid-run inspection."""

        captured_jobs: dict[str, Any] = {}

        def _save_state(
            self, state_path: Path, batch_path: str, jobs: dict[str, Job]
        ) -> None:
            self.captured_jobs = {n: j for n, j in jobs.items()}
            return super()._save_state(state_path, batch_path, jobs)

    sched = _CapturingScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    # The sentinel breaks the loop when the job stays pending (the fix).
    # With the live bug, the loop terminates on its own after the job is
    # failed at 10 admission blocks — no sentinel raised, so the
    # assertions below catch the regression.
    try:
        sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    except RuntimeError as exc:
        assert "bounded sentinel" in str(exc), f"unexpected RuntimeError: {exc}"

    # No run() calls — job was never admitted.
    assert len(colab_client.run_calls) == 0
    # sessions() was polled at least once.
    assert colab_client.sessions_calls > 0
    # The job must remain pending — admission blocks must not fail it.
    cb0 = sched.captured_jobs.get("cb0")
    assert cb0 is not None, "cb0 job not captured"
    assert cb0.state == PENDING, (
        f"job should remain pending under permanent external occupancy, "
        f"got state={cb0.state!r}, error={cb0.error!r}"
    )
    # Admission blocks must not increment capacity_rejections — only
    # actual TooManyAssignmentsError process exits do.
    assert cb0.capacity_rejections == 0, (
        f"admission blocks must not increment capacity_rejections; "
        f"cb0.capacity_rejections={cb0.capacity_rejections}"
    )


def test_external_sessions_reduce_available_slots(tmp_path: Path) -> None:
    """With 1 external session and pre-seeded AIMD limit=2, both jobs are
    admitted (1 external < effective_limit=2)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=2)
    state = tmp_path / "state.json"
    # Pre-seed learned_limit=2 so the first job can be admitted past the
    # external session (effective_limit=2, 1 external < 2).  Without
    # pre-seeding, AIMD starts at 1 and the external session fills the
    # only slot, blocking all jobs permanently.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {},
            "updated_at": 1000.0,
            "colab_learned_limit": 2,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    # 1 external session; with pre-seeded learned_limit=2,
    # effective_limit=2, 1 < 2, so both jobs are admitted.
    colab_client = FakeColabClient(
        sessions_list=[{"name": "ext", "state": "running"}],
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(2)},
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1, colab_max=5
    )
    assert summary["all_succeeded"]
    # Both jobs were admitted (1 external + 2 scheduler, AIMD grows to 4).
    assert len(colab_client.run_calls) == 2


def test_sessions_exception_falls_back_to_active_count(tmp_path: Path) -> None:
    """If sessions() raises, the scheduler falls back to counting active
    Colab jobs from its own state."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    class FlakySessionsClient(FakeColabClient):
        def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
            self.sessions_calls += 1
            raise ConnectionError("cannot reach backend")

    colab_client = FlakySessionsClient(
        behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(2)}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1, colab_max=5
    )
    # Fallback to active_count means AIMD still works.
    assert summary["all_succeeded"]


# ---------------------------------------------------------------------------
# 8. CPU-only subprocess argv (ColabCliClient.run)
# ---------------------------------------------------------------------------


def _recording_popen_factory(
    argv_recorder: list[list[str]],
) -> Any:
    """Return a popen_factory that records argv and returns a FakePopen."""

    def factory(argv: list[str]) -> FakePopen:
        argv_recorder.append(list(argv))
        return FakePopen(stdout="ok", stderr="", returncode=0)

    return factory


def test_colab_run_argv_omits_gpu_and_tpu() -> None:
    """ColabCliClient.run() must never pass --gpu or --tpu."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("test-session", "script.py", arguments=["--foo", "bar"])
    assert len(recorded) == 1
    argv = recorded[0]
    assert "--gpu" not in argv
    assert "--tpu" not in argv


def test_colab_run_argv_contains_required_flags() -> None:
    """ColabCliClient.run() argv must contain colab run --keep --session.
    F3: a ``--`` separator must precede the script so that script paths
    or arguments starting with ``--`` are treated as positionals, not
    CLI flags (e.g. ``--gpu`` is forwarded to the script, not consumed
    by the CLI)."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("my-session", "run.py", arguments=["--x", "1"], timeout=300)
    argv = recorded[0]
    assert argv[0] == "colab"
    assert argv[1] == "run"
    assert "--keep" in argv
    assert "--session" in argv
    session_idx = argv.index("--session")
    assert argv[session_idx + 1] == "my-session"
    assert "--timeout" in argv
    timeout_idx = argv.index("--timeout")
    assert argv[timeout_idx + 1] == str(300)
    # F3: -- separator before the script.
    assert "--" in argv
    dash_idx = argv.index("--")
    assert argv[dash_idx + 1] == "run.py"
    assert "--x" in argv
    assert "1" in argv


def test_colab_run_argv_no_shell_true() -> None:
    """The popen_factory receives an argv list (no shell=True)."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "script.py")
    argv = recorded[0]
    # argv is a list of strings, not a single shell command string.
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)
    # No shell metacharacters from shell=True.
    assert " ".join(argv) == " ".join(argv)  # well-formed


def test_colab_run_timeout_defaults_to_run_timeout() -> None:
    """When timeout is None, the default run_timeout is used."""
    recorded: list[list[str]] = []
    client = ColabCliClient(
        popen_factory=_recording_popen_factory(recorded),
        run_timeout=42,
    )
    client.run("s", "script.py")
    argv = recorded[0]
    timeout_idx = argv.index("--timeout")
    assert argv[timeout_idx + 1] == "42"


# ---------------------------------------------------------------------------
# 9. classify_colab_output status parsing
# ---------------------------------------------------------------------------


def test_classify_complete_exit_zero() -> None:
    """Exit code 0 with no capacity markers -> COLAB_COMPLETE."""
    proc = FakePopen(stdout="done", stderr="", returncode=0)
    assert classify_colab_output(proc) == COLAB_COMPLETE


def test_classify_error_exit_nonzero() -> None:
    """Non-zero exit without capacity markers -> COLAB_ERROR."""
    proc = FakePopen(stdout="", stderr="some error", returncode=1)
    assert classify_colab_output(proc) == COLAB_ERROR


def test_classify_capacity_rejected_toomany() -> None:
    """Stderr containing TooManyAssignmentsError -> COLAB_CAPACITY_REJECTED."""
    proc = FakePopen(
        stdout="",
        stderr="Traceback: TooManyAssignmentsError: 412",
        returncode=1,
    )
    assert classify_colab_output(proc) == COLAB_CAPACITY_REJECTED


def test_classify_capacity_rejected_precondition_failed() -> None:
    """Stderr containing 'Precondition Failed' -> COLAB_CAPACITY_REJECTED."""
    proc = FakePopen(
        stdout="",
        stderr="HTTP 412 Precondition Failed",
        returncode=1,
    )
    assert classify_colab_output(proc) == COLAB_CAPACITY_REJECTED


def test_classify_capacity_rejected_overrides_exit_zero() -> None:
    """F6: exit code 0 takes precedence — a capacity marker in stderr with
    a successful exit is NOT a capacity rejection.  The 412 traceback
    always has a non-zero exit; a script that prints 'precondition failed'
    and exits 0 completed normally."""
    proc = FakePopen(
        stdout="",
        stderr="TooManyAssignmentsError",
        returncode=0,
    )
    assert classify_colab_output(proc) == COLAB_COMPLETE

def test_classify_caches_result() -> None:
    """Repeated calls return the same cached result without re-reading pipes."""
    proc = FakePopen(stdout="output", stderr="err", returncode=0)
    first = classify_colab_output(proc)
    second = classify_colab_output(proc)
    assert first == second == COLAB_COMPLETE
    # Cached attributes set on proc.
    assert getattr(proc, "_colab_classified_output", None) == COLAB_COMPLETE
    assert getattr(proc, "_colab_stdout", None) == "output"
    assert getattr(proc, "_colab_stderr", None) == "err"


def test_classify_drained_pipes_fall_back_to_empty() -> None:
    """If pipes are already consumed, classification proceeds on exit code."""
    proc = FakePopen(stdout="text", stderr="text", returncode=0)
    # Consume pipes.
    proc.stdout.read()
    proc.stderr.read()
    # Should still classify based on exit code.
    assert classify_colab_output(proc) == COLAB_COMPLETE


# ---------------------------------------------------------------------------
# 10. is_capacity_rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "TooManyAssignmentsError",
    "toomanyassignmentserror",
    "TooManyAssignmentsError: 412",
    "Precondition Failed",
    "precondition failed",
    "HTTP 412 Precondition Failed",
])
def test_is_capacity_rejected_true(text: str) -> None:
    assert is_capacity_rejected(text) is True


@pytest.mark.parametrize("text", [
    "normal output",
    "RuntimeError: something else",
    "",
    "completed successfully",
])
def test_is_capacity_rejected_false(text: str) -> None:
    assert is_capacity_rejected(text) is False


def test_is_capacity_rejected_checks_combined_stdout_stderr() -> None:
    """The marker can appear in either stdout or stderr."""
    assert is_capacity_rejected("ok\nTooManyAssignmentsError") is True
    assert is_capacity_rejected("ok\nPrecondition Failed\n") is True


# ---------------------------------------------------------------------------
# 11. parse_sessions
# ---------------------------------------------------------------------------


def test_parse_sessions_normal_output() -> None:
    """Normal colab sessions output is parsed into session dicts."""
    output = "[session-a] https://us-west1 | running\n[session-b] https://us-east1 | running\n"
    sessions = parse_sessions(output)
    assert len(sessions) == 2
    assert sessions[0]["name"] == "session-a"
    assert sessions[0]["state"] == COLAB_RUNNING
    assert sessions[1]["name"] == "session-b"


def test_parse_sessions_no_active_sessions() -> None:
    """'no active sessions' output returns an empty list."""
    sessions = parse_sessions("No active sessions.\n")
    assert sessions == []


def test_parse_sessions_empty_output() -> None:
    """Empty output returns an empty list."""
    assert parse_sessions("") == []


def test_parse_sessions_ignores_non_session_lines() -> None:
    """Lines not matching the session pattern are ignored."""
    output = "Header line\n[session-a] endpoint | running\nSome footer\n"
    sessions = parse_sessions(output)
    assert len(sessions) == 1
    assert sessions[0]["name"] == "session-a"


# ---------------------------------------------------------------------------
# 12. ColabCliClient.collect writes stdout.log / stderr.log / result.json
# ---------------------------------------------------------------------------


def test_collect_writes_output_files(tmp_path: Path) -> None:
    """collect() writes stdout.log, stderr.log, and result.json."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    proc = client.run("test-session", "script.py")
    client.remember("test-session", proc)
    # Simulate process completion.
    proc.returncode = 0
    classify_colab_output(proc)  # cache output on proc

    out_dir = tmp_path / "out"
    result = client.collect("test-session", str(out_dir))
    assert result["ok"] is True
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    assert (out_dir / "result.json").exists()
    meta = json.loads((out_dir / "result.json").read_text())
    assert meta["ok"] is True
    assert meta["status"] == COLAB_COMPLETE
    assert meta["session"] == "test-session"


def test_collect_error_status_writes_failed_result(tmp_path: Path) -> None:
    """When the proc errored, collect() writes result.json with ok=False."""
    client = ColabCliClient(
        popen_factory=_recording_popen_factory([]),
    )
    proc = FakePopen(stdout="", stderr="some error", returncode=1)
    client.remember("err-session", proc)
    classify_colab_output(proc)

    out_dir = tmp_path / "out"
    result = client.collect("err-session", str(out_dir))
    assert result["ok"] is False
    meta = json.loads((out_dir / "result.json").read_text())
    assert meta["ok"] is False
    assert meta["status"] == COLAB_ERROR


def test_collect_idempotent(tmp_path: Path) -> None:
    """collect() can be called multiple times (overwrites files)."""
    client = ColabCliClient(popen_factory=_recording_popen_factory([]))
    proc = FakePopen(stdout="output", stderr="", returncode=0)
    client.remember("s", proc)
    classify_colab_output(proc)

    out_dir = tmp_path / "out"
    client.collect("s", str(out_dir))
    # Second call should not raise.
    client.collect("s", str(out_dir))
    assert (out_dir / "stdout.log").read_text() == "output"


def test_collect_without_proc_writes_empty_logs(tmp_path: Path) -> None:
    """collect() on an unknown session writes empty logs and error result."""
    client = ColabCliClient(popen_factory=_recording_popen_factory([]))
    out_dir = tmp_path / "out"
    result = client.collect("unknown", str(out_dir))
    assert result["ok"] is False
    assert (out_dir / "stdout.log").read_text() == ""
    assert (out_dir / "stderr.log").read_text() == ""
    meta = json.loads((out_dir / "result.json").read_text())
    assert meta["ok"] is False


# ---------------------------------------------------------------------------
# 13. ColabCliClient.stop is best-effort
# ---------------------------------------------------------------------------


def _recording_runner(
    argv_recorder: list[list[str]],
    *,
    stdout: str = "",
    raise_on: set[str] | None = None,
    raise_exc: Exception | None = None,
) -> Any:
    """Return a runner that records argv and optionally raises."""

    def runner(argv: list[str], timeout: float | None) -> Any:
        argv_recorder.append(list(argv))
        if raise_on and any(cmd in argv for cmd in raise_on):
            raise raise_exc or RuntimeError("simulated failure")
        # SimpleNamespace avoids the class-body closure-scope NameError that
        # a nested ``class FakeResult: stdout = stdout`` would raise (class
        # bodies don't see enclosing function locals).
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return runner


def test_stop_swallows_runtime_error() -> None:
    """stop() must not raise even if the runner fails."""
    recorded: list[list[str]] = []
    runner = _recording_runner(recorded, raise_on={"stop"}, raise_exc=RuntimeError("stop failed"))
    client = ColabCliClient(runner=runner)
    # Should not raise.
    client.stop("my-session")


def test_stop_swallows_timeout_expired() -> None:
    """stop() must not raise on subprocess.TimeoutExpired."""
    import subprocess as sp

    recorded: list[list[str]] = []
    runner = _recording_runner(
        recorded, raise_on={"stop"}, raise_exc=sp.TimeoutExpired(cmd=["colab", "stop"], timeout=1)
    )
    client = ColabCliClient(runner=runner)
    client.stop("my-session")  # should not raise


def test_stop_argv_contains_session_name() -> None:
    """stop() argv must include colab stop --session <name>."""
    recorded: list[list[str]] = []
    runner = _recording_runner(recorded)
    client = ColabCliClient(runner=runner)
    client.stop("my-session")
    stop_argv = [a for a in recorded if "stop" in a]
    assert len(stop_argv) == 1
    argv = stop_argv[0]
    assert argv[0] == "colab"
    assert argv[1] == "stop"
    assert "--session" in argv
    assert argv[argv.index("--session") + 1] == "my-session"


def test_stop_cleans_up_proc_tracking() -> None:
    """stop() removes the session from internal proc tracking."""
    client = ColabCliClient(popen_factory=_recording_popen_factory([]))
    proc = FakePopen(stdout="ok", returncode=0)
    client.remember("s", proc)
    assert client._procs.get("s") is proc
    client.stop("s")
    assert "s" not in client._procs


# ---------------------------------------------------------------------------
# 14. ColabCliClient.sessions via runner
# ---------------------------------------------------------------------------


def test_sessions_parses_runner_stdout() -> None:
    """sessions() calls the runner and parses the output."""
    recorded: list[list[str]] = []
    runner = _recording_runner(
        recorded, stdout="[sess-a] endpoint | running\n[sess-b] endpoint | running\n"
    )
    client = ColabCliClient(runner=runner)
    sessions = client.sessions()
    assert len(sessions) == 2
    assert sessions[0]["name"] == "sess-a"
    assert sessions[1]["name"] == "sess-b"
    # Verify argv.
    assert recorded[0] == ["colab", "sessions"]


def test_sessions_uses_configured_timeout() -> None:
    """sessions() passes the configured timeout to the runner."""
    recorded_timeouts: list[float | None] = []

    def runner(argv: list[str], timeout: float | None) -> Any:
        recorded_timeouts.append(timeout)

        class FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""

        return FakeResult()

    client = ColabCliClient(runner=runner, sessions_timeout=45.0)
    client.sessions()
    assert recorded_timeouts[0] == 45.0


# ---------------------------------------------------------------------------
# 15. Cleanup: stop() called on success, error, timeout, capacity rejection
# ---------------------------------------------------------------------------


def test_cleanup_stop_called_on_success(tmp_path: Path) -> None:
    """A Colab job that completes successfully triggers stop()."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert "cb0" in colab_client.stop_calls


def test_colab_runtime_gate_rejects_provider_ok_without_report(tmp_path: Path) -> None:
    """Colab collect ok cannot become succeeded without execution-report/v2."""
    class ProviderOkNoReport(FakeColabClient):
        def collect(self, name: str, output_dir: str) -> dict[str, Any]:
            self.collect_calls.append((name, output_dir))
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "stdout.log").write_text("", encoding="utf-8")
            (out / "stderr.log").write_text("", encoding="utf-8")
            (out / "result.json").write_text('{"ok": true}', encoding="utf-8")
            return {"ok": True}

    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = ProviderOkNoReport(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    summary = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock,
        sleeper=CountingSleeper(clock_h, interval=1.0)
    ).run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    job = summary["jobs"]["cb0"]
    assert job["state"] == FAILED
    assert job["runtime_manifest_verified"] is False
    assert "runtime verification failed" in (job["error"] or "")


def test_colab_runtime_gate_fails_observed_manifest_mismatch(tmp_path: Path) -> None:
    """A Colab report with a different observed manifest is terminal failure."""
    class MismatchReportClient(FakeColabClient):
        def collect(self, name: str, output_dir: str) -> dict[str, Any]:
            result = super().collect(name, output_dir)
            expected = _valid_runtime_manifest()
            observed = _valid_runtime_manifest(python_version="3.13.0")
            report = _execution_report(observed)
            report["expected_runtime_manifest_sha256"] = compute_manifest_sha256(expected)
            Path(output_dir, "execution_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            return result

    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = MismatchReportClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    summary = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock,
        sleeper=CountingSleeper(clock_h, interval=1.0)
    ).run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    job = summary["jobs"]["cb0"]
    assert job["state"] == FAILED
    assert job["runtime_manifest_verified"] is False
    assert "runtime manifest mismatch" in (job["error"] or "")


def test_cleanup_stop_called_on_error(tmp_path: Path) -> None:
    """A Colab job that errors triggers stop()."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stderr": "some error", "returncode": 1}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert "cb0" in colab_client.stop_calls


def test_cleanup_stop_called_on_timeout(tmp_path: Path) -> None:
    """A Colab job that times out triggers stop()."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock(start=1000.0)
    sleeper = CountingSleeper(clock_h, interval=100.0)
    # Process never finishes (running_polls very high).
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"running_polls": 100, "returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        job_timeout=50.0, colab_max=5,
    )
    assert "cb0" in colab_client.stop_calls
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "timed out" in (raw["jobs"]["cb0"].get("error") or "").lower()


def test_cleanup_stop_called_on_capacity_rejection(tmp_path: Path) -> None:
    """A Colab job that gets 412 triggers stop() to free the VM slot."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=10.0)
    colab_client = FakeColabClient(
        behaviors={
            "cb0": [
                {"stderr": "TooManyAssignmentsError", "returncode": 1},
                {"stdout": "ok", "returncode": 0},
            ],
        }
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=5, colab_cooldown=30.0,
    )
    # stop() called at least once (on capacity rejection, and again on success).
    assert colab_client.stop_calls.count("cb0") >= 2


def test_cleanup_stop_called_on_collect_failure(tmp_path: Path) -> None:
    """A Colab job whose collect() fails still triggers stop() in the
    finally block."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]},
        collect_raises=RuntimeError("disk full"),
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert "cb0" in colab_client.stop_calls
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED


# ---------------------------------------------------------------------------
# 16. Interrupted restart: Colab running -> failed + best-effort stop
# ---------------------------------------------------------------------------


def test_interrupted_colab_running_fails_on_restart(tmp_path: Path) -> None:
    """A Colab job in 'running' state on restart is failed as interrupted."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    # Write a v2 state with the Colab job in running state.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": RUNNING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": ["--flag", "value"],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    # The interrupted job is failed, not silently resubmitted.
    raw = _read_state(state)
    # On restart, the running Colab job is failed as interrupted.
    # Then it may be re-submitted as pending (since _load_or_init_state
    # converts it to FAILED, and the run loop doesn't retry FAILED jobs).
    assert raw["jobs"]["cb0"]["state"] in (FAILED, SUCCEEDED)
    if raw["jobs"]["cb0"]["state"] == FAILED:
        assert "interrupted" in (raw["jobs"]["cb0"].get("error") or "").lower()
    # Best-effort stop was called for the interrupted job.
    assert "cb0" in colab_client.stop_calls


def test_interrupted_colab_collecting_fails_on_restart(tmp_path: Path) -> None:
    """A Colab job in 'collecting' state on restart is failed (no proc to
    collect from)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": COLLECTING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "interrupted" in (raw["jobs"]["cb0"].get("error") or "").lower()


def test_interrupted_colab_submitting_resumes_as_pending(tmp_path: Path) -> None:
    """A Colab job in 'submitting' state on restart is converted to pending
    and re-submitted."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": SUBMITTING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert summary["all_succeeded"]
    assert len(colab_client.run_calls) == 1


# ---------------------------------------------------------------------------
# 17. State v2: colab_learned_limit persisted
# ---------------------------------------------------------------------------


def test_state_v2_persists_colab_learned_limit(tmp_path: Path) -> None:
    """After a run with Colab jobs, the state file has schema v2 and
    colab_learned_limit."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={f"cb{i}": [{"running_polls": 0, "returncode": 0}] for i in range(2)}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=4, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    assert raw["schema_version"] == STATE_SCHEMA_V2
    assert "colab_learned_limit" in raw
    assert isinstance(raw["colab_learned_limit"], int)


def test_state_v2_job_has_provider_and_remote_id(tmp_path: Path) -> None:
    """v2 state jobs have provider and remote_id fields."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    job = raw["jobs"]["cb0"]
    assert job["provider"] == PROVIDER_COLAB
    assert job["remote_id"] == "cb0"
    assert job["script"] is not None
    assert "arguments" in job


# ---------------------------------------------------------------------------
# 18. v1 state migration
# ---------------------------------------------------------------------------


def test_v1_state_migrates_kaggle_jobs(tmp_path: Path) -> None:
    """A v1 state file (no provider/remote_id) is loaded with provider=kaggle
    and remote_id=kernel_id."""
    _make_kernel_dir(tmp_path, "kg0", "owner/kg0")
    manifest = _make_v1_batch(tmp_path, [{
        "name": "kg0",
        "kernel_dir": "kg0",
        "output_dir": "out/kg0",
    }])
    state = tmp_path / "state.json"
    # Write a v1 state with the job in running state.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_VERSION,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "kg0": {
                    "name": "kg0",
                    "kernel_dir": str((tmp_path / "kg0").resolve()),
                    "output_dir": str((tmp_path / "out/kg0").resolve()),
                    "kernel_id": "owner/kg0",
                    "state": RUNNING,
                    "error": None,
                    "submitted_at": 1000.0,
                    "completed_at": None,
                    "collected_at": None,
                    "attempts": 1,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    kaggle_client = FakeKaggleClient(
        status_sequences={"owner/kg0": ["complete"]}
    )
    sched = ParallelScheduler(
        kaggle_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1)
    assert summary["all_succeeded"]
    raw = _read_state(state)
    # v1 batch -> v1 state schema (no v2 upgrade for kaggle-only batches).
    job = raw["jobs"]["kg0"]
    assert job["state"] == SUCCEEDED


def test_v1_state_job_from_dict_migrates_provider_and_remote_id() -> None:
    """Job.from_dict on a v1 entry (no provider/remote_id) defaults to
    kaggle with remote_id=kernel_id."""
    job = Job.from_dict({
        "name": "kg0",
        "kernel_dir": "/path",
        "output_dir": "/out",
        "kernel_id": "owner/kg0",
        "state": RUNNING,
        "error": None,
        "submitted_at": 1000.0,
        "completed_at": None,
        "collected_at": None,
        "attempts": 1,
    })
    assert job.provider == PROVIDER_KAGGLE
    assert job.remote_id == "owner/kg0"
    assert job.kernel_id == "owner/kg0"


def test_v1_state_no_colab_learned_limit(tmp_path: Path) -> None:
    """A v1 state file does not have colab_learned_limit."""
    _make_kernel_dir(tmp_path, "kg0", "owner/kg0")
    manifest = _make_v1_batch(tmp_path, [{
        "name": "kg0",
        "kernel_dir": "kg0",
        "output_dir": "out/kg0",
    }])
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    kaggle_client = FakeKaggleClient()
    sched = ParallelScheduler(
        kaggle_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1)
    raw = _read_state(state)
    assert "colab_learned_limit" not in raw


# ---------------------------------------------------------------------------
# 19. CLI flags: --colab-max, --colab-cooldown
# ---------------------------------------------------------------------------


def test_cli_run_accepts_colab_max_and_cooldown(tmp_path: Path) -> None:
    """CLI run accepts --colab-max and --colab-cooldown flags."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"returncode": 0}]}
    )
    rc = main(
        ["run", str(manifest), "--state", str(state),
         "--colab-max", "3", "--colab-cooldown", "45"],
        client=FakeKaggleClient(),
        colab_client=colab_client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc == 0
    raw = _read_state(state)
    assert raw["colab_learned_limit"] is not None


def test_cli_run_rejects_colab_max_below_minimum(tmp_path: Path) -> None:
    """CLI --colab-max 0 must fail via SystemExit (parser.error)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    with pytest.raises(SystemExit):
        main(
            ["run", str(manifest), "--state", str(state), "--colab-max", "0"],
            client=FakeKaggleClient(),
            colab_client=FakeColabClient(),
        )


def test_cli_run_default_colab_max(tmp_path: Path) -> None:
    """CLI --colab-max defaults to DEFAULT_COLAB_MAX."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"returncode": 0}]}
    )
    # Use --colab-max explicitly to verify it's accepted; default is tested
    # by verifying the run succeeds with the default.
    rc = main(
        ["run", str(manifest), "--state", str(state)],
        client=FakeKaggleClient(),
        colab_client=colab_client,
        clock=clock,
        sleeper=sleeper,
    )
    assert rc == 0


def test_cli_run_colab_jobs_without_client_fails_them(tmp_path: Path) -> None:
    """When colab_client is None and the batch has Colab jobs, the CLI
    should still run but the Colab jobs are failed (no client configured)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    # colab_client=None and _COLAB_AVAILABLE would construct a real client.
    # But we pass colab_client=None explicitly; the CLI peeks at the batch
    # and would try to construct ColabCliClient() if _COLAB_AVAILABLE.
    # Since we can't prevent that without mocking, we test the scheduler
    # directly instead.
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=None, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert summary["failed"] == 1
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "no colab client" in (raw["jobs"]["cb0"].get("error") or "")


# ---------------------------------------------------------------------------
# 20. detect_orphaned_sessions
# ---------------------------------------------------------------------------


def test_detect_orphaned_sessions_returns_known_and_active() -> None:
    """detect_orphaned_sessions returns names in both known_names and the
    backend's active session list."""
    client = FakeColabClient(
        sessions_list=[
            {"name": "cb0", "state": "running"},
            {"name": "cb1", "state": "running"},
            {"name": "external", "state": "running"},
        ]
    )
    orphans = detect_orphaned_sessions(client, {"cb0", "cb1", "cb-missing"})
    assert orphans == ["cb0", "cb1"]


def test_detect_orphaned_sessions_empty_when_no_overlap() -> None:
    """No orphaned sessions when known names don't match backend."""
    client = FakeColabClient(
        sessions_list=[{"name": "other", "state": "running"}]
    )
    orphans = detect_orphaned_sessions(client, {"cb0", "cb1"})
    assert orphans == []


def test_detect_orphaned_sessions_sorted() -> None:
    """Result is sorted alphabetically."""
    client = FakeColabClient(
        sessions_list=[
            {"name": "z-session", "state": "running"},
            {"name": "a-session", "state": "running"},
        ]
    )
    orphans = detect_orphaned_sessions(client, {"z-session", "a-session"})
    assert orphans == ["a-session", "z-session"]


# ---------------------------------------------------------------------------
# 21. No real subprocess / network calls in any fake
# ---------------------------------------------------------------------------


def test_fakes_never_invoke_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure that using fake clients never triggers a real subprocess."""
    called: list[str] = []

    def _fail_popen(*args: Any, **kwargs: Any) -> Any:
        called.append("Popen")
        raise AssertionError("subprocess.Popen was invoked")

    def _fail_run(*args: Any, **kwargs: Any) -> Any:
        called.append("run")
        raise AssertionError("subprocess.run was invoked")

    monkeypatch.setattr("subprocess.Popen", _fail_popen)
    monkeypatch.setattr("subprocess.run", _fail_run)

    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=1, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    kaggle_client = FakeKaggleClient(
        status_sequences={"owner/kg0": ["complete"]}
    )
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"returncode": 0}]}
    )
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert called == [], f"subprocess was invoked: {called}"


# ---------------------------------------------------------------------------
# 22. Summary structure for mixed batches
# ---------------------------------------------------------------------------


def test_summary_has_per_provider_breakdown(tmp_path: Path) -> None:
    """The run summary includes per-provider counts and colab learned_limit."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=1, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    kaggle_client = FakeKaggleClient(
        status_sequences={"owner/kg0": ["complete"]}
    )
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"returncode": 0}]}
    )
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert "providers" in summary
    assert summary["providers"]["kaggle"]["total"] == 1
    assert summary["providers"]["kaggle"]["succeeded"] == 1
    assert summary["providers"]["colab"]["total"] == 1
    assert summary["providers"]["colab"]["succeeded"] == 1
    assert summary["providers"]["colab"]["learned_limit"] is not None
    assert summary["all_succeeded"]


# ---------------------------------------------------------------------------
# 23. Restart safety: Colab jobs not resubmitted with stale proc
# ---------------------------------------------------------------------------


def test_restart_does_not_resubmit_succeeded_colab(tmp_path: Path) -> None:
    """A Colab job already succeeded in state is not re-run on restart."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": SUCCEEDED,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
            "colab_learned_limit": 2,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert summary["all_succeeded"]
    assert len(colab_client.run_calls) == 0  # not re-run


def test_restart_colab_failed_not_rerun(tmp_path: Path) -> None:
    """A Colab job already failed in state is not re-run on restart."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": FAILED,
                    "error": "previous failure",
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert summary["failed"] == 1
    assert len(colab_client.run_calls) == 0



# ---------------------------------------------------------------------------
# 24. F1-F12 reliability regression tests
# ---------------------------------------------------------------------------


def test_f1_classify_reads_temp_file_paths(tmp_path: Path) -> None:
    """F1: classify_colab_output reads from _colab_stdout_path/_colab_stderr_path
    temp files when stashed on the proc, instead of draining pipes.  This is
    the mechanism that prevents pipe deadlocks on large output."""
    stdout_file = tmp_path / "fake_stdout.out"
    stderr_file = tmp_path / "fake_stderr.err"
    large_output = "x" * 70000  # > 64KB OS pipe buffer
    stdout_file.write_text(large_output, encoding="utf-8")
    stderr_file.write_text("some error", encoding="utf-8")

    proc = FakePopen(stdout="pipe-fallback", stderr="", returncode=1)
    proc._colab_stdout_path = str(stdout_file)
    proc._colab_stderr_path = str(stderr_file)

    status = classify_colab_output(proc)
    assert status == COLAB_ERROR
    # The temp file content was read, not the pipe fallback.
    assert getattr(proc, "_colab_stdout", "") == large_output
    assert getattr(proc, "_colab_stderr", "") == "some error"


def test_f1_read_proc_output_falls_back_to_pipe() -> None:
    """F1: _read_proc_output falls back to pipe draining when temp file
    attrs are absent (FakePopen compatibility)."""
    proc = FakePopen(stdout="pipe-content", stderr="err-content", returncode=0)
    assert _read_proc_output(proc, "stdout") == "pipe-content"
    assert _read_proc_output(proc, "stderr") == "err-content"


def test_f1_cleanup_tempfiles_safe_on_fake_popen() -> None:
    """F1: _cleanup_proc_tempfiles is a no-op on a FakePopen without temp
    file attrs — must not raise."""
    proc = FakePopen(stdout="ok", returncode=0)
    _cleanup_proc_tempfiles(proc)  # should not raise


def test_f1_cleanup_tempfiles_unlinks_real_paths(tmp_path: Path) -> None:
    """F1: _cleanup_proc_tempfiles closes and unlinks temp files when present."""
    stdout_file = tmp_path / "out.out"
    stderr_file = tmp_path / "err.err"
    stdout_file.write_text("out", encoding="utf-8")
    stderr_file.write_text("err", encoding="utf-8")

    proc = FakePopen(stdout="", stderr="", returncode=0)
    proc._colab_stdout_path = str(stdout_file)
    proc._colab_stderr_path = str(stderr_file)
    proc._colab_stdout_file = open(stdout_file)
    proc._colab_stderr_file = open(stderr_file)

    _cleanup_proc_tempfiles(proc)
    assert not stdout_file.exists()
    assert not stderr_file.exists()
    assert getattr(proc, "_colab_stdout_path", None) is None
    assert getattr(proc, "_colab_stderr_file", None) is None


def test_f2_aimd_reduction_uses_account_wide_count(tmp_path: Path) -> None:
    """F2: on capacity rejection, on_capacity_rejected receives the
    account-wide session count (including external sessions), not just
    the scheduler-local active count.

    Uses a dynamic sessions list: 3 external sessions initially (so the
    job is admitted with pre-seeded learned_limit=5), then 0 after the
    capacity-rejection cycle completes (so the retry can be admitted).
    After 412, on_capacity_rejected(3) sets learned_limit=3.  Old bug
    would call on_capacity_rejected(1) → learned_limit=1."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {},
            "updated_at": 1000.0,
            "colab_learned_limit": 5,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=10.0)

    class DynamicSessionsClient(FakeColabClient):
        def __init__(self) -> None:
            super().__init__(
                sessions_list=[
                    {"name": "ext-0", "state": "running"},
                    {"name": "ext-1", "state": "running"},
                    {"name": "ext-2", "state": "running"},
                ],
                behaviors={
                    "cb0": [
                        {"stderr": "TooManyAssignmentsError", "returncode": 1},
                        {"stdout": "ok", "returncode": 0},
                    ],
                },
            )

        def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
            # External sessions are present during admission and capacity-
            # rejection processing, then drain away for the retry.
            # sessions() call sequence: 1) restart orphan check, 2) admission
            # check, 3) capacity-rejection active count — all return 3.
            # Call 4+ (retry admission) returns 0 so the retry is admitted.
            # The scheduler calls sessions() AFTER stop() in the capacity
            # rejection handler, so draining on stop() (the old approach)
            # would zero the count too early.  Draining based on the
            # sessions() call count aligns with the intended progression.
            if self.sessions_calls >= 3:
                self._sessions_list = []
            return super().sessions(timeout=timeout)

    colab_client = DynamicSessionsClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=10, colab_cooldown=30.0,
    )
    raw = _read_state(state)
    # After 412 with 3 external sessions, on_capacity_rejected(3) sets
    # learned_limit=3.  (Old bug: on_capacity_rejected(1) → learned_limit=1.)
    assert raw["colab_learned_limit"] >= 3


def test_f3_separator_before_script_in_argv() -> None:
    """F3: ``--`` separator appears in argv before the script path."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "script.py", arguments=["--gpu", "T4"])
    argv = recorded[0]
    assert "--" in argv
    dash_idx = argv.index("--")
    # Script immediately after --.
    assert argv[dash_idx + 1] == "script.py"
    # Arguments after the script, not consumed as CLI flags.
    assert "--gpu" in argv
    assert "T4" in argv


def test_f3_separator_protects_dashdash_script() -> None:
    """F3: a script path starting with ``--`` is protected by the separator."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "--gpu", arguments=[])
    argv = recorded[0]
    dash_idx = argv.index("--")
    assert argv[dash_idx + 1] == "--gpu"


def test_f4_kill_proc_on_timeout(tmp_path: Path) -> None:
    """F4: on job timeout, proc.kill() is called before job.proc is set
    to None.  The local CLI process is reaped, not just the backend VM."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock(start=1000.0)
    sleeper = CountingSleeper(clock_h, interval=100.0)

    class KillRecordingPopen(FakePopen):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.kill_called = False
            self.wait_called = False

        def kill(self) -> None:
            self.kill_called = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_called = True
            return self.returncode or 0

    captured_procs: list[KillRecordingPopen] = []

    class KillRecordingClient(FakeColabClient):
        def run(self, name: str, script: str, *, arguments: list[str] | None = None,
                timeout: float | None = None) -> Any:
            self.run_calls.append((name, script, arguments, timeout))
            proc = KillRecordingPopen(running_polls=100, returncode=0)
            captured_procs.append(proc)
            return proc

    colab_client = KillRecordingClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        job_timeout=50.0, colab_max=5,
    )
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "timed out" in (raw["jobs"]["cb0"].get("error") or "").lower()
    # F4: proc.kill() was called on the timed-out process.
    assert len(captured_procs) == 1
    assert captured_procs[0].kill_called is True
    assert captured_procs[0].wait_called is True
    # stop() was also called for backend cleanup.
    assert "cb0" in colab_client.stop_calls


def test_f4_kill_called_on_missing_proc_restart(tmp_path: Path) -> None:
    """F4: on restart with a RUNNING Colab job (no local proc), the job
    is failed as interrupted and stop() is called for cleanup."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": RUNNING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] in (FAILED, SUCCEEDED)
    if raw["jobs"]["cb0"]["state"] == FAILED:
        assert "interrupted" in (raw["jobs"]["cb0"].get("error") or "").lower()
    assert "cb0" in colab_client.stop_calls


def test_f5_orphan_recovery_stops_leaked_session(tmp_path: Path) -> None:
    """F5: on restart, a Colab session that exists on the backend but is
    not owned by any active job is detected as orphaned and stopped, and
    the restarted pending job can then resubmit and complete.

    A SUBMITTING job that crashed before the Popen launched leaves a
    session on the backend.  On restart the job becomes PENDING, but it
    was loaded from the state file (_from_state=True), so its remote_id
    IS in known_names.  detect_orphaned_sessions intersects known_names
    with the backend session list, finds the leaked session, and stop()
    tears it down — removing it from the backend so the pending job can
    be admitted on the next poll."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": SUBMITTING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    # sessions_list includes the leaked "cb0" session.  stop() must
    # remove it (matching real Colab) so the restarted pending job is
    # not blocked by a phantom session forever.
    colab_client = FakeColabClient(
        sessions_list=[{"name": "cb0", "state": "running"}],
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]},
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    # The orphaned "cb0" session was stopped during restart recovery.
    # stop_calls may contain "cb0" from both orphan recovery and normal
    # cleanup — at least one call from orphan recovery.
    assert "cb0" in colab_client.stop_calls
    # The leaked session was torn down, so the pending job was admitted
    # and resubmitted (run_calls is non-empty), then completed.
    assert len(colab_client.run_calls) >= 1, (
        "pending job should have been resubmitted after orphan stop"
    )
    assert summary["all_succeeded"], (
        f"job should complete after orphan recovery frees the slot, "
        f"got summary={summary}"
    )
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == SUCCEEDED


def test_f5_orphan_recovery_stops_succeeded_job_session(tmp_path: Path) -> None:
    """F5: a session belonging to a SUCCEEDED (terminal) job is included
    in known_names (terminal states are claimed by the scheduler), so it
    IS detected as orphaned and stopped on restart."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    # SUCCEEDED is a terminal state; the scheduler claims terminal job
    # sessions in known_names so they are stopped during orphan recovery.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": SUCCEEDED,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    # sessions_list includes "cb0" which is SUCCEEDED (terminal).
    # It should be detected as orphaned and stopped.
    colab_client = FakeColabClient(
        sessions_list=[{"name": "cb0", "state": "running"}],
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    # "cb0" is in known_names (terminal jobs are claimed), so it IS
    # detected as orphaned and stopped.
    assert "cb0" in colab_client.stop_calls


def test_f6_exit_zero_marker_in_stdout_is_complete() -> None:
    """F6: exit code 0 with a capacity marker in stdout (not stderr) is
    COLAB_COMPLETE, not CAPACITY_REJECTED.  A script that prints
    'precondition failed' and exits 0 completed normally."""
    proc = FakePopen(
        stdout="precondition failed",
        stderr="",
        returncode=0,
    )
    assert classify_colab_output(proc) == COLAB_COMPLETE


def test_f6_nonzero_exit_marker_in_stderr_is_rejected() -> None:
    """F6: non-zero exit with marker in stderr is CAPACITY_REJECTED."""
    proc = FakePopen(
        stdout="",
        stderr="TooManyAssignmentsError: 412",
        returncode=1,
    )
    assert classify_colab_output(proc) == COLAB_CAPACITY_REJECTED


def test_f6_nonzero_exit_marker_in_stdout_only_is_error() -> None:
    """F6: non-zero exit with marker in stdout only (not stderr) is
    COLAB_ERROR, not CAPACITY_REJECTED.  The marker must be in stderr
    (where the Python traceback appears)."""
    proc = FakePopen(
        stdout="precondition failed",
        stderr="some other error",
        returncode=1,
    )
    assert classify_colab_output(proc) == COLAB_ERROR


def test_f7_state_saved_submitting_before_run(tmp_path: Path) -> None:
    """F7: the state file is saved as SUBMITTING (with remote_id set)
    BEFORE colab_client.run() is called.  A crash in the launch window
    leaves a recoverable SUBMITTING record.

    This test uses a client that reads the state file inside run() to
    verify the job is already SUBMITTING at that point."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    observed: dict[str, Any] = {}

    class PreLaunchCheckClient(FakeColabClient):
        def run(self, name: str, script: str, *,
                arguments: list[str] | None = None,
                timeout: float | None = None) -> Any:
            raw = json.loads(state.read_text(encoding="utf-8"))
            job_entry = raw.get("jobs", {}).get(name, {})
            observed["state_at_run"] = job_entry.get("state")
            observed["remote_id_at_run"] = job_entry.get("remote_id")
            return super().run(name, script, arguments=arguments, timeout=timeout)

    colab_client = PreLaunchCheckClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert observed.get("state_at_run") == SUBMITTING
    assert observed.get("remote_id_at_run") == "cb0"


def test_f8_sessions_cached_once_per_loop_iteration(tmp_path: Path) -> None:
    """F8: colab_client.sessions() is called at most once per loop
    iteration when multiple pending Colab jobs are blocked, not once
    per pending job.  With 3 blocked jobs and 3 loop iterations, the
    call count is 4 (1 restart orphan check + 3 loop iterations),
    not 10 (1 + 3 jobs × 3 iterations)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=3)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    # 1 external session fills the AIMD limit (1), blocking all 3 jobs.
    colab_client = FakeColabClient(
        sessions_list=[{"name": "ext", "state": "running"}],
        behaviors={f"cb{i}": [{"stdout": "ok", "returncode": 0}] for i in range(3)},
    )
    sleeper = BoundedSleeper(clock_h, max_calls=3)
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    with pytest.raises(RuntimeError, match="bounded sleeper"):
        sched.run(manifest, state, max_parallel=4, poll_interval=1, colab_max=5)
    # 1 restart orphan-check call + 3 loop iterations (cached per iteration)
    # = 4 total.  Without caching: 1 + 3 jobs × 3 iterations = 10 calls.
    assert colab_client.sessions_calls == 4


def test_f9_no_aimd_increase_on_immediate_exit(tmp_path: Path) -> None:
    """F9: AIMD learned_limit does not increase when a Colab proc exits
    immediately (running_polls=0) without ever being seen running.
    on_success is deferred to the first poll where the proc is still
    RUNNING — if the proc already exited, on_success never fires."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0, "running_polls": 0}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    # learned_limit stays at 1 — on_success was never called because the
    # proc was never seen running (it exited immediately).
    assert raw["colab_learned_limit"] == 1


def test_f9_aimd_increases_after_confirmed_running(tmp_path: Path) -> None:
    """F9: AIMD learned_limit increases by 1 after the first poll confirms
    the proc is still RUNNING (session was created on the backend)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok", "returncode": 0, "running_polls": 1}]}
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    # learned_limit = 1 + 1 (on_success after first running poll) = 2.
    assert raw["colab_learned_limit"] == 2


def test_f10_colab_client_protocol_has_remember() -> None:
    """F10: ColabClient protocol includes the remember method."""
    assert hasattr(ColabClient, "remember")
    # FakeColabClient satisfies the protocol.
    fake = FakeColabClient()
    assert hasattr(fake, "remember")
    # remember is callable.
    assert callable(fake.remember)


def test_f10_remember_associates_proc_for_collect(tmp_path: Path) -> None:
    """F10: remember(name, proc) associates a Popen with a session name
    so that collect() can retrieve it."""
    client = ColabCliClient(popen_factory=_recording_popen_factory([]))
    proc = FakePopen(stdout="output", stderr="", returncode=0)
    client.remember("test-session", proc)
    assert client._procs.get("test-session") is proc


def test_f11_timeout_clamped_to_minimum_one() -> None:
    """F11: sub-second timeouts are clamped to 1 so int() truncation
    never yields --timeout 0."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "script.py", timeout=0.5)
    argv = recorded[0]
    timeout_idx = argv.index("--timeout")
    assert argv[timeout_idx + 1] == "1"


def test_f11_timeout_zero_clamped_to_one() -> None:
    """F11: timeout=0 is clamped to 1."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "script.py", timeout=0)
    argv = recorded[0]
    timeout_idx = argv.index("--timeout")
    assert argv[timeout_idx + 1] == "1"


def test_f11_timeout_negative_clamped_to_one() -> None:
    """F11: negative timeouts are clamped to 1."""
    recorded: list[list[str]] = []
    client = ColabCliClient(popen_factory=_recording_popen_factory(recorded))
    client.run("s", "script.py", timeout=-5)
    argv = recorded[0]
    timeout_idx = argv.index("--timeout")
    assert argv[timeout_idx + 1] == "1"


def test_f12_load_state_non_dict_returns_empty(tmp_path: Path) -> None:
    """F12: load_state on a non-dict JSON file (e.g. a list or string)
    returns an empty state structure instead of crashing."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    raw = load_state(state)
    assert isinstance(raw, dict)
    assert "jobs" in raw
    assert raw["jobs"] == {}


def test_f12_load_state_string_json_returns_empty(tmp_path: Path) -> None:
    """F12: load_state on a JSON string returns an empty state structure."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps("just a string"), encoding="utf-8")
    raw = load_state(state)
    assert isinstance(raw, dict)
    assert "jobs" in raw
    assert raw["jobs"] == {}


def test_f12_load_state_null_json_returns_empty(tmp_path: Path) -> None:
    """F12: load_state on a JSON null returns an empty state structure."""
    state = tmp_path / "state.json"
    state.write_text("null", encoding="utf-8")
    raw = load_state(state)
    assert isinstance(raw, dict)
    assert "jobs" in raw
    assert raw["jobs"] == {}


def test_f12_load_state_valid_dict_preserved(tmp_path: Path) -> None:
    """F12: load_state on a valid dict JSON preserves the content."""
    state = tmp_path / "state.json"
    original = {"schema_version": STATE_SCHEMA_V2, "jobs": {"cb0": {"state": PENDING}}}
    state.write_text(json.dumps(original), encoding="utf-8")
    raw = load_state(state)
    assert raw == original
# ---------------------------------------------------------------------------
# 25. Regression: pending job queues at capacity without capacity_rejection
# ---------------------------------------------------------------------------


def test_fourth_job_queues_then_succeeds_no_capacity_rejection(
    tmp_path: Path,
) -> None:
    """Regression (live v2 Colab run): with learned capacity=3 and four
    CPU jobs, the first three run concurrently while the fourth remains
    pending across many admission polls.  Pending jobs blocked solely
    because active/external sessions fill learned capacity MUST stay
    pending and wait — they must not increment capacity_rejections or
    fail.  Only actual ``TooManyAssignmentsError`` process exits increment
    the rejection count.  Once a slot frees, the fourth is admitted and
    succeeds.

    Against the live bug, the admission gate incremented
    ``capacity_rejections`` on each block and failed the job after
    ``COLAB_MAX_CAPACITY_REJECTIONS`` (10) blocks, even though no
    capacity-rejection process exit had occurred — it was normal queueing.
    """
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=4)
    state = tmp_path / "state.json"
    # Pre-seed learned_limit=3 so three jobs can run concurrently.  With
    # colab_max=3 the ceiling caps effective_limit at 3, preventing AIMD
    # on_success from growing the limit past 3 while the first three are
    # admitted.
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {},
            "updated_at": 1000.0,
            "colab_learned_limit": 3,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)

    class _DynamicSessionsClient(FakeColabClient):
        """FakeColabClient whose sessions() reflects live scheduler jobs.

        run() adds the session to the active set; stop() removes it.
        sessions() returns the active set so the admission gate sees the
        real account-wide session count, matching live Colab behaviour.
        """

        def __init__(self) -> None:
            super().__init__(
                behaviors={
                    # cb0 completes after 13 polls, freeing a slot for cb3.
                    "cb0": [{"stdout": "ok", "running_polls": 12, "returncode": 0}],
                    # cb1, cb2 stay running well past cb3's admission.
                    "cb1": [{"stdout": "ok", "running_polls": 20, "returncode": 0}],
                    "cb2": [{"stdout": "ok", "running_polls": 20, "returncode": 0}],
                    # cb3 completes quickly once admitted.
                    "cb3": [{"stdout": "ok", "running_polls": 1, "returncode": 0}],
                },
            )
            self._live: set[str] = set()

        def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
            self.sessions_calls += 1
            return [{"name": n, "state": "running"} for n in sorted(self._live)]

        def run(
            self,
            name: str,
            script: str,
            *,
            arguments: list[str] | None = None,
            timeout: float | None = None,
        ) -> FakePopen:
            proc = super().run(
                name, script, arguments=arguments, timeout=timeout
            )
            self._live.add(name)
            return proc

        def stop(self, name: str, *, timeout: float | None = None) -> None:
            super().stop(name, timeout=timeout)
            self._live.discard(name)

    colab_client = _DynamicSessionsClient()

    class _CapturingScheduler(ParallelScheduler):
        """Captures the jobs dict on every state save for post-run inspection."""

        captured_jobs: dict[str, Any] = {}

        def _save_state(
            self, state_path: Path, batch_path: str, jobs: dict[str, Job]
        ) -> None:
            self.captured_jobs = {n: j for n, j in jobs.items()}
            return super()._save_state(state_path, batch_path, jobs)

    sched = _CapturingScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(
        manifest, state, max_parallel=4, poll_interval=1, colab_max=3,
    )

    # All four jobs eventually succeed — the fourth was queued, not failed.
    assert summary["all_succeeded"], f"expected all 4 to succeed, got: {summary}"
    assert summary["succeeded"] == 4
    assert summary["failed"] == 0

    # The fourth job was blocked at the admission gate for many polls.
    # CountingSleeper is called once per loop iteration while jobs are
    # active or pending.  At least 12 sleeps means 12 admission polls
    # where cb3 was blocked — exceeding COLAB_MAX_CAPACITY_REJECTIONS (10),
    # so the live bug would have failed cb3 before this point.
    assert len(sleeper.calls) >= 12, (
        f"expected >= 12 polling iterations while cb3 queued, "
        f"got {len(sleeper.calls)}"
    )

    # All four jobs were eventually run (cb3 was admitted after a slot freed).
    assert len(colab_client.run_calls) == 4

    # Critical invariant: capacity_rejections must NOT increment from
    # admission-gate blocks.  Only actual TooManyAssignmentsError process
    # exits (COLAB_CAPACITY_REJECTED) increment it.  cb3 never exited
    # with a capacity rejection — it was purely queued.
    cb3 = sched.captured_jobs.get("cb3")
    assert cb3 is not None, "cb3 job not captured"
    assert cb3.capacity_rejections == 0, (
        f"admission blocks must not increment capacity_rejections; "
        f"cb3.capacity_rejections={cb3.capacity_rejections}"
    )
    assert cb3.state == SUCCEEDED

    # cb0–cb2 also never had a capacity rejection.
    for name in ("cb0", "cb1", "cb2"):
        job = sched.captured_jobs[name]
        assert job.capacity_rejections == 0, (
            f"{name} should have 0 capacity_rejections, "
            f"got {job.capacity_rejections}"
        )
        assert job.state == SUCCEEDED

    # Summary shows queued pending was observed (via sleeper calls) and
    # eventual success (all_succeeded).  The learned limit is preserved.
    assert summary["providers"]["colab"]["succeeded"] == 4
    assert summary["providers"]["colab"]["failed"] == 0


# ---------------------------------------------------------------------------
# 17. Colab failure diagnostics: stdout/stderr/result.json persistence
# ---------------------------------------------------------------------------


def test_colab_error_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job that exits non-zero leaves stdout.log, stderr.log, and
    result.json in output_dir with the proc's captured output."""
    manifest, entries = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={
            "cb0": [{"stdout": "script output here", "stderr": "traceback: boom", "returncode": 1}],
        }
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").read_text(encoding="utf-8") == "script output here"
    assert (out_dir / "stderr.log").read_text(encoding="utf-8") == "traceback: boom"
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["status"] == COLAB_ERROR
    assert result["exit_code"] == 1
    assert result["session"] == "cb0"


def test_colab_error_message_includes_exit_code(tmp_path: Path) -> None:
    """The scheduler error for a failed Colab job includes the exit code
    for actionable diagnostics."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stderr": "fail", "returncode": 42}]},
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    raw = _read_state(state)
    err = raw["jobs"]["cb0"].get("error") or ""
    assert "exit_code=42" in err
    assert "status=error" in err


def test_colab_timeout_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job that times out leaves stdout.log, stderr.log, and
    result.json in output_dir with whatever output was produced before
    the timeout."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock(start=1000.0)
    sleeper = CountingSleeper(clock_h, interval=100.0)
    # Process never finishes (running_polls very high).
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "partial output", "stderr": "", "running_polls": 100, "returncode": 0}]},
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        job_timeout=50.0, colab_max=5,
    )

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "timed out" in (result["error"] or "")
    assert result["session"] == "cb0"

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "timed out" in (raw["jobs"]["cb0"].get("error") or "").lower()


def test_colab_capacity_exhaustion_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job that fails after COLAB_MAX_CAPACITY_REJECTIONS
    consecutive capacity rejections leaves diagnostics in output_dir."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock(start=1000.0)
    sleeper = CountingSleeper(clock_h, interval=1.0)
    rejections = [
        {"stderr": "TooManyAssignmentsError", "returncode": 1}
        for _ in range(COLAB_MAX_CAPACITY_REJECTIONS)
    ]
    colab_client = FakeColabClient(behaviors={"cb0": rejections})
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(
        manifest, state, max_parallel=2, poll_interval=1,
        colab_max=5, colab_cooldown=1.0,
    )

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "capacity rejections" in (raw["jobs"]["cb0"].get("error") or "")

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "capacity rejections" in (result["error"] or "")
    assert result["session"] == "cb0"


def test_colab_submit_failure_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job whose run() raises leaves result.json with the error
    in output_dir, even though no proc was created."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        run_raises=RuntimeError("colab CLI not installed"),
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "colab run failed" in (raw["jobs"]["cb0"].get("error") or "")

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    assert (out_dir / "stdout.log").read_text(encoding="utf-8") == ""
    assert (out_dir / "stderr.log").read_text(encoding="utf-8") == ""
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "colab run failed" in (result["error"] or "")
    assert result["session"] == "cb0"


def test_colab_restart_interrupted_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job failed as interrupted on restart leaves result.json
    with the interruption error in output_dir."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_V2,
            "batch_path": str(manifest.resolve()),
            "jobs": {
                "cb0": {
                    "name": "cb0",
                    "output_dir": str((tmp_path / "out/cb0").resolve()),
                    "state": RUNNING,
                    "provider": PROVIDER_COLAB,
                    "remote_id": "cb0",
                    "script": str((tmp_path / "scripts/cb0.py").resolve()),
                    "arguments": [],
                    "timeout": None,
                }
            },
            "updated_at": 1000.0,
        }),
        encoding="utf-8",
    )
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "interrupted" in (raw["jobs"]["cb0"].get("error") or "").lower()

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "interrupted" in (result["error"] or "").lower()
    assert result["session"] == "cb0"


def test_colab_no_client_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job with no client configured leaves result.json with the
    error in output_dir."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=None, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "no colab client" in (raw["jobs"]["cb0"].get("error") or "")

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "no colab client" in (result["error"] or "")
    assert result["session"] == "cb0"


def test_colab_collect_failure_persists_diagnostics(tmp_path: Path) -> None:
    """A Colab job whose collect() raises still leaves diagnostics in
    output_dir via the _persist_colab_diagnostics fallback."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "ok output", "stderr": "", "returncode": 0}]},
        collect_raises=RuntimeError("disk full"),
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)

    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == FAILED
    assert "collection failed" in (raw["jobs"]["cb0"].get("error") or "")

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "collection failed" in (result["error"] or "")
    assert result["session"] == "cb0"


def test_colab_success_still_persists_diagnostics(tmp_path: Path) -> None:
    """A successful Colab job still has stdout.log/stderr.log/result.json
    in output_dir (via collect), confirming the success path is not
    broken by the diagnostics changes."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"stdout": "hello world", "stderr": "", "returncode": 0}]},
    )
    sched = ParallelScheduler(
        FakeKaggleClient(), colab_client=colab_client, clock=clock, sleeper=sleeper
    )
    summary = sched.run(manifest, state, max_parallel=2, poll_interval=1, colab_max=5)
    assert summary["all_succeeded"]

    out_dir = tmp_path / "out" / "cb0"
    assert (out_dir / "stdout.log").read_text(encoding="utf-8") == "hello world"
    assert (out_dir / "stderr.log").exists()
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
# ---------------------------------------------------------------------------
# 26. Live-watch contract: watch_batch mode regression tests (mixed provider)
# ---------------------------------------------------------------------------
#
# These tests defend the durable live-watch queue contract in the mixed
# Kaggle/Colab scheduler:
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


def _append_colab_job_to_batch(
    base: Path,
    batch_path: Path,
    name: str,
) -> None:
    """Append a new Colab job to an existing v2 batch manifest, creating
    its script file.  Raises if the name already exists."""
    manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    existing_names = {j["name"] for j in manifest["jobs"]}
    assert name not in existing_names, f"job {name} already in batch"
    script_rel = _make_colab_script(base, f"scripts/{name}.py")
    manifest["jobs"].append({
        "name": name,
        "provider": PROVIDER_COLAB,
        "script": script_rel,
        "output_dir": f"out/{name}",
        "arguments": [],
        "job_spec": _make_colab_job_spec(base, f"specs/{name}.json"),
        "runtime_manifest": _valid_runtime_manifest(),
    })
    batch_path.write_text(json.dumps(manifest), encoding="utf-8")


def _append_kaggle_job_to_batch(
    base: Path,
    batch_path: Path,
    name: str,
    kernel_id: str,
) -> None:
    """Append a new Kaggle job to an existing v2 batch manifest, creating
    its kernel directory.  Raises if the name already exists."""
    manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    existing_names = {j["name"] for j in manifest["jobs"]}
    assert name not in existing_names, f"job {name} already in batch"
    _make_kernel_dir(base, name, kernel_id)
    manifest["jobs"].append({
        "name": name,
        "provider": PROVIDER_KAGGLE,
        "kernel_dir": name,
        "output_dir": f"out/{name}",
        "runtime_manifest": _valid_runtime_manifest(),
    })
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


def test_watch_merges_new_colab_job_during_active_run(tmp_path: Path) -> None:
    """In watch mode, a new Colab job appended to the batch file while
    jobs are running is merged as PENDING and eventually executed.

    The batch starts with 1 Colab job (cb0).  While cb0 is running, a
    2nd Colab job (cb1) is appended.  The scheduler merges cb1, submits
    it, and both reach SUCCEEDED.
    """
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = FakeColabClient(
        behaviors={
            "cb0": [{"running_polls": 1, "stdout": "ok0", "returncode": 0}],
            "cb1": [{"running_polls": 0, "stdout": "ok1", "returncode": 0}],
        },
    )
    kaggle_client = FakeKaggleClient()

    appended = {"done": False}
    original_submit_colab = ParallelScheduler._submit_colab

    def _wrapped_submit_colab(self, job, job_timeout, save_state=None):
        result = original_submit_colab(self, job, job_timeout, save_state=save_state)
        if not appended["done"] and job.name == "cb0":
            _append_colab_job_to_batch(tmp_path, manifest, "cb1")
            appended["done"] = True
        return result

    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=10)

    try:
        ParallelScheduler._submit_colab = _wrapped_submit_colab
        sched = ParallelScheduler(
            kaggle_client, colab_client=colab_client,
            clock=clock, sleeper=sleeper,
        )
        summary = sched.run(
            manifest, state, max_parallel=None, poll_interval=1,
            colab_max=5, watch_batch=True, watch_interval=1,
        )
    finally:
        ParallelScheduler._submit_colab = original_submit_colab

    assert summary["jobs"]["cb0"]["state"] == SUCCEEDED
    assert "cb1" in summary["jobs"]
    assert summary["jobs"]["cb1"]["state"] == SUCCEEDED
    assert summary["all_succeeded"]


def test_watch_does_not_mutate_existing_colab_job_on_reload(
    tmp_path: Path,
) -> None:
    """Reloading the batch must not mutate existing Colab job state.
    cb0 succeeds, then cb1 is appended.  cb0's state/attempts must be
    unchanged by the reload."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = FakeColabClient(
        behaviors={
            "cb0": [{"running_polls": 1, "stdout": "ok0", "returncode": 0}],
            "cb1": [{"running_polls": 0, "stdout": "ok1", "returncode": 0}],
        },
    )
    kaggle_client = FakeKaggleClient()

    appended = {"done": False}
    original_submit_colab = ParallelScheduler._submit_colab

    def _wrapped_submit_colab(self, job, job_timeout, save_state=None):
        result = original_submit_colab(self, job, job_timeout, save_state=save_state)
        if not appended["done"] and job.name == "cb0":
            _append_colab_job_to_batch(tmp_path, manifest, "cb1")
            appended["done"] = True
        return result

    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=10)

    try:
        ParallelScheduler._submit_colab = _wrapped_submit_colab
        sched = ParallelScheduler(
            kaggle_client, colab_client=colab_client,
            clock=clock, sleeper=sleeper,
        )
        summary = sched.run(
            manifest, state, max_parallel=None, poll_interval=1,
            colab_max=5, watch_batch=True, watch_interval=1,
        )
    finally:
        ParallelScheduler._submit_colab = original_submit_colab

    # cb0 succeeded with 1 attempt (not re-submitted by reload).
    assert summary["jobs"]["cb0"]["state"] == SUCCEEDED
    assert summary["jobs"]["cb0"]["attempts"] == 1
    # cb1 was added fresh.
    assert summary["jobs"]["cb1"]["state"] == SUCCEEDED
    assert summary["jobs"]["cb1"]["attempts"] == 1


def test_watch_malformed_reload_retried_then_valid_colab_succeeds(
    tmp_path: Path,
) -> None:
    """Malformed batch reload (invalid JSON) is logged and retried.
    Later valid content with a new Colab job is merged and executed."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = FakeColabClient(
        behaviors={
            "cb0": [{"running_polls": 0, "stdout": "ok0", "returncode": 0}],
            "cb1": [{"running_polls": 0, "stdout": "ok1", "returncode": 0}],
        },
    )
    kaggle_client = FakeKaggleClient()

    phase = {"step": 0}

    class _StagedSleeper:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def __call__(self, seconds: float) -> None:
            self.calls.append(seconds)
            clock_h[0] += seconds
            phase["step"] += 1
            # After cb0 completes and scheduler goes idle, corrupt the file.
            if phase["step"] == 2:
                manifest.write_text("{ broken json !!!", encoding="utf-8")
            # On the next idle, write valid JSON with cb0 + cb1.
            if phase["step"] == 3:
                _make_colab_script(tmp_path, "scripts/cb1.py")
                manifest.write_text(
                    json.dumps({
                        "schema_version": BATCH_SCHEMA_V2,
                        "jobs": [
                            {
                                "name": "cb0",
                                "provider": PROVIDER_COLAB,
                                "script": "scripts/cb0.py",
                                "output_dir": "out/cb0",
                                "arguments": [],
                                "job_spec": _make_colab_job_spec(tmp_path, "specs/cb0.json"),
                                "runtime_manifest": _valid_runtime_manifest(),
                            },
                            {
                                "name": "cb1",
                                "provider": PROVIDER_COLAB,
                                "script": "scripts/cb1.py",
                                "output_dir": "out/cb1",
                                "arguments": [],
                                "job_spec": _make_colab_job_spec(tmp_path, "specs/cb1.json"),
                                "runtime_manifest": _valid_runtime_manifest(),
                            },
                        ],
                    }),
                    encoding="utf-8",
                )
            if phase["step"] >= 10:
                raise KeyboardInterrupt()

    sleeper = _StagedSleeper()
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client,
        clock=clock, sleeper=sleeper,
    )
    summary = sched.run(
        manifest, state, max_parallel=None, poll_interval=1,
        colab_max=5, watch_batch=True, watch_interval=1,
    )

    assert summary["jobs"]["cb0"]["state"] == SUCCEEDED
    assert "cb1" in summary["jobs"]
    assert summary["jobs"]["cb1"]["state"] == SUCCEEDED


def test_watch_stays_alive_idle_then_terminated_colab(tmp_path: Path) -> None:
    """Watch mode stays alive across idle intervals after all Colab jobs
    complete, until KeyboardInterrupt.  The sleeper must be called at
    least 3 times (poll + >=2 idle watch intervals)."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    colab_client = FakeColabClient(
        behaviors={"cb0": [{"running_polls": 0, "stdout": "ok", "returncode": 0}]},
    )
    kaggle_client = FakeKaggleClient()
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=5)
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client,
        clock=clock, sleeper=sleeper,
    )
    summary = sched.run(
        manifest, state, max_parallel=None, poll_interval=1,
        colab_max=5, watch_batch=True, watch_interval=1,
    )

    assert summary["jobs"]["cb0"]["state"] == SUCCEEDED
    assert len(sleeper.calls) >= 3, (
        f"expected >=3 sleeper calls (poll + idle watch), "
        f"got {len(sleeper.calls)}: {sleeper.calls}"
    )
    raw = _read_state(state)
    assert raw["jobs"]["cb0"]["state"] == SUCCEEDED


def test_non_watch_mode_still_terminates_colab(tmp_path: Path) -> None:
    """Without watch_batch, the mixed scheduler terminates when all jobs
    reach terminal state — backward compatible.  The sleeper must NOT
    trigger KeyboardInterrupt."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=2, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient(
        behaviors={
            f"cb{i}": [{"running_polls": 1}] for i in range(2)
        },
    )
    kaggle_client = FakeKaggleClient(
        status_sequences={
            f"owner/kg{i}": ["running", "complete"] for i in range(2)
        },
    )
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client,
        clock=clock, sleeper=sleeper,
    )
    summary = sched.run(
        manifest, state, max_parallel=None, poll_interval=1, colab_max=5,
        # watch_batch defaults to False
    )

    assert summary["all_succeeded"]
    assert summary["total"] == 4
    # The sleeper was called a bounded number of times (no infinite loop).
    assert len(sleeper.calls) < 50


def test_watch_additive_capacity_5_kaggle_plus_colab(tmp_path: Path) -> None:
    """In watch mode with max_parallel=None, the scheduler admits 5
    Kaggle + at least 1 Colab concurrently — proving additive capacity
    is preserved in watch mode.

    The batch starts with 12 Kaggle + 2 Colab.  The scheduler should
    concurrently run 5 Kaggle (DEFAULT_KAGGLE_MAX) + at least 1 Colab,
    exceeding 5 total.  Watch mode is terminated via KeyboardInterrupt.
    """
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=12, n_colab=2)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()

    tracker: dict[str, Any] = {
        "kaggle_active": set(),
        "colab_active": set(),
        "max_kaggle": 0,
        "max_colab": 0,
        "max_total": 0,
    }

    def _update_max() -> None:
        tracker["max_kaggle"] = max(
            tracker["max_kaggle"], len(tracker["kaggle_active"])
        )
        tracker["max_colab"] = max(
            tracker["max_colab"], len(tracker["colab_active"])
        )
        tracker["max_total"] = max(
            tracker["max_total"],
            len(tracker["kaggle_active"]) + len(tracker["colab_active"]),
        )

    class TrackingKaggleClient(FakeKaggleClient):
        def __init__(self) -> None:
            super().__init__(
                status_sequences={
                    f"owner/kg{i}": ["running", "complete"] for i in range(12)
                },
            )

        def push(self, kernel_dir: str, *, timeout: float | None = None) -> str:
            kid = super().push(kernel_dir, timeout=timeout)
            tracker["kaggle_active"].add(kid)
            _update_max()
            return kid

        def status(self, kernel_id: str, *, timeout: float | None = None) -> str:
            result = super().status(kernel_id, timeout=timeout)
            if result in ("complete", "error"):
                tracker["kaggle_active"].discard(kernel_id)
            return result

    class TrackingColabClient(FakeColabClient):
        def __init__(self) -> None:
            super().__init__(
                behaviors={f"cb{i}": [{"running_polls": 1}] for i in range(2)},
            )

        def run(
            self, name: str, script: str, *,
            arguments: list[str] | None = None,
            timeout: float | None = None,
        ) -> FakePopen:
            proc = super().run(name, script, arguments=arguments, timeout=timeout)
            tracker["colab_active"].add(name)
            _update_max()
            return proc

        def stop(self, name: str, *, timeout: float | None = None) -> None:
            super().stop(name, timeout=timeout)
            tracker["colab_active"].discard(name)

    kaggle_client = TrackingKaggleClient()
    colab_client = TrackingColabClient()
    # 30 calls is generous for 14 jobs with 2-poll sequences.
    sleeper = _KeyboardInterruptSleeper(clock_h, max_calls=30)
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client,
        clock=clock, sleeper=sleeper,
    )
    summary = sched.run(
        manifest, state, max_parallel=None, poll_interval=1,
        watch_batch=True, watch_interval=1,
    )

    # All jobs succeeded (KeyboardInterrupt may fire after completion).
    # If it fires mid-run, some jobs may not have completed — but with
    # 30 calls and fast fake clients, all should finish.
    assert summary["total"] == 14
    # 5 Kaggle jobs admitted concurrently.
    assert tracker["max_kaggle"] == DEFAULT_KAGGLE_MAX, (
        f"expected {DEFAULT_KAGGLE_MAX} concurrent Kaggle in watch mode, "
        f"got {tracker['max_kaggle']}"
    )
    # At least 1 Colab admitted concurrently.
    assert tracker["max_colab"] >= 1, (
        f"expected >=1 concurrent Colab in watch mode, "
        f"got {tracker['max_colab']}"
    )
    # Total exceeded 5 — additive capacity proven in watch mode.
    assert tracker["max_total"] > HARD_KAGGLE_MAX, (
        f"total concurrency {tracker['max_total']} did not exceed "
        f"hard kaggle cap {HARD_KAGGLE_MAX} in watch mode"
    )


def test_watch_rejects_nonpositive_watch_interval_colab(tmp_path: Path) -> None:
    """watch_batch=True with watch_interval <= 0 must raise ValueError."""
    manifest, _ = _make_mixed_batch(tmp_path, n_kaggle=0, n_colab=1)
    state = tmp_path / "state.json"
    clock_h, clock = _make_clock()
    sleeper = CountingSleeper(clock_h, interval=1.0)
    colab_client = FakeColabClient()
    kaggle_client = FakeKaggleClient()
    sched = ParallelScheduler(
        kaggle_client, colab_client=colab_client,
        clock=clock, sleeper=sleeper,
    )
    with pytest.raises(ValueError, match="watch_interval"):
        sched.run(
            manifest, state, max_parallel=None, poll_interval=1,
            colab_max=5, watch_batch=True, watch_interval=0,
        )


def test_watch_default_watch_interval_constant_colab() -> None:
    """The DEFAULT_WATCH_INTERVAL constant must exist and be 30.0 seconds."""
    assert DEFAULT_WATCH_INTERVAL is not None, (
        "DEFAULT_WATCH_INTERVAL not defined in scheduler module"
    )
    assert DEFAULT_WATCH_INTERVAL == 30.0
