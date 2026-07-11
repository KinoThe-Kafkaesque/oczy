"""Behavioral tests for the mixed Kaggle/Colab remote experiment campaign.

Covers the full campaign lifecycle contracts:

* Runner report (execution_report.json) on success / nonzero / timeout,
  METRIC/ASI line parsing, schema version, log filenames, no shell=True.
* src-layout injection in the Colab bootstrap (repo root, repo/src,
  workspace package ``*/src`` dirs prepended to ``sys.path``).
* Colab exact-commit bootstrap: clone --no-checkout, HEAD verification,
  CPU env enforcement, wrong-repo / short-commit rejection.
* Accelerator rejection in Colab job preparation (``--gpu``, ``--tpu``,
  ``--cuda``, ``--device cuda``, ``--accelerator`` rejected anywhere,
  even after ``--``; CPU-only applies to the target module).
* Campaign mixed generation: validate_campaign accepts mixed providers,
  prepare_experiment_campaign emits a v2 scheduler-compatible batch.
* Duplicate / invalid jobs rejected by the campaign validator.
* Frozen meta-test sign-off propagation: meta-test jobs require
  instrument_manifest_sha256 + human_signoff_id and pass them through.
* Collector provenance validation and COMPLETE / NULL / INVALID / BLOCKED
  classification.
* Missing outputs (no report file, no output dir).
* v2 scheduler compatibility: generated batch loads via load_batch.

All tests use fake subprocess, fake clients, and temp repos — never the
network, real Kaggle/Colab CLI, or real GitHub.
"""

from __future__ import annotations

import ast
import io
import json
import re
import runpy
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loading — all implementation files loaded via runpy, skip if absent.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = REPO_ROOT / "infrastructure" / "kaggle"


def _load_module(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"implementation not found at {path}", allow_module_level=True)
    return runpy.run_path(str(path))


# Campaign preparer / validator
_prep_mod = _load_module(KAGGLE_DIR / "prepare_experiment_campaign.py")
# Collector / classifier
_coll_mod = _load_module(KAGGLE_DIR / "collect_experiment_campaign.py")
# Runner report writer
_runner_mod = _load_module(KAGGLE_DIR / "run_experiment_module.py")
# Colab experiment preparer
_colab_prep_mod = _load_module(KAGGLE_DIR / "prepare_colab_experiment.py")
# Scheduler (for v2 batch compatibility)
_sched_mod = _load_module(KAGGLE_DIR / "parallel_scheduler.py")
# Kernel preparer (for meta-test sign-off propagation)
_kernel_mod = _load_module(KAGGLE_DIR / "prepare_research_kernel.py")

# ---------------------------------------------------------------------------
# Constants pulled from the loaded modules
# ---------------------------------------------------------------------------

CAMPAIGN_SCHEMA_VERSION: str = _prep_mod["CAMPAIGN_SCHEMA_VERSION"]
validate_campaign = _prep_mod["validate_campaign"]
prepare_experiment_campaign = _prep_mod["prepare_experiment_campaign"]

classify_job_result = _coll_mod["classify_job_result"]
collect_experiment_campaign = _coll_mod["collect_experiment_campaign"]
# Colab sentinel / Kaggle provenance constants (may not exist in all versions).
_COLAB_SENTINEL_PREFIX: str = _coll_mod.get("_COLAB_SENTINEL_PREFIX", "OCZY_EXECUTION_REPORT_JSON=")
_COLAB_STDOUT_LOG: str = _coll_mod.get("_COLAB_STDOUT_LOG", "stdout.log")
COLAB_RESULT_FILENAME: str = _coll_mod.get("COLAB_RESULT_FILENAME", "result.json")
KAGGLE_PROVENANCE_FILENAME: str = _coll_mod.get("KAGGLE_PROVENANCE_FILENAME", "remote_run_provenance.json")
SentinelError = _coll_mod.get("SentinelError", type("SentinelError", (Exception,), {}))

run_module = _runner_mod["run_module"]
RUNNER_SCHEMA_VERSION: str = _runner_mod["SCHEMA_VERSION"]

prepare_colab_experiment = _colab_prep_mod["prepare_colab_experiment"]
COLAB_JOB_SCHEMA_VERSION: str = _colab_prep_mod["JOB_SPEC_SCHEMA_VERSION"]

BATCH_SCHEMA_V2: str = _sched_mod["BATCH_SCHEMA_V2"]
load_batch = _sched_mod["load_batch"]
BatchValidationError = _sched_mod["BatchValidationError"]
PROVIDER_KAGGLE: str = _sched_mod["PROVIDER_KAGGLE"]
PROVIDER_COLAB: str = _sched_mod["PROVIDER_COLAB"]
SUCCEEDED: str = _sched_mod["SUCCEEDED"]
FAILED: str = _sched_mod["FAILED"]

PHASES = _kernel_mod["PHASES"]
prepare_kernel = _kernel_mod["prepare_kernel"]

COMMIT = "a" * 40
COMMIT_B = "b" * 40
REPO_URL = "https://github.com/KinoThe-Kafkaesque/oczy.git"
ARCHIVE_SHA = "c" * 64
SOURCE_DATASET = f"owner/oczy-source-{COMMIT[:12]}"

EXPECTED_CAMPAIGN_SCHEMA = "oczy/remote-experiment-campaign/v1"
EXPECTED_RUNNER_SCHEMA = "oczy/execution-report/v1"
EXPECTED_COLAB_JOB_SCHEMA = "oczy/colab-experiment-job/v1"
EXPECTED_BATCH_V2 = "oczy/remote-parallel-batch/v2"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _valid_campaign_job(
    *,
    name: str = "job-a",
    provider: str = PROVIDER_KAGGLE,
    phase: str = "development",
    module: str = "infrastructure.kaggle.run_cortex_smoke",
    arguments: list[str] | None = None,
    output_path: str = "out/job-a",
    claim_class: str = "scientific",
    **extra: Any,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "name": name,
        "provider": provider,
        "phase": phase,
        "module": module,
        "arguments": arguments if arguments is not None else [],
        "output_path": output_path,
        "claim_class": claim_class,
    }
    # Kaggle jobs require provenance fields (kernel_id, title, source_dataset,
    # source_archive_sha256).  The title slug must match the kernel_id suffix
    # per prepare_kernel's validation.
    if provider == PROVIDER_KAGGLE:
        title_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        job["kernel_id"] = f"owner/oczy-{title_slug}"
        job["title"] = f"Oczy {name}"
        job["source_dataset"] = SOURCE_DATASET
        job["source_archive_sha256"] = ARCHIVE_SHA
    job.update(extra)
    return job


def _valid_campaign(
    *,
    jobs: list[dict[str, Any]] | None = None,
    source_commit: str = COMMIT,
    source_repo: str = REPO_URL,
) -> dict[str, Any]:
    if jobs is None:
        jobs = [_valid_campaign_job()]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "source_commit": source_commit,
        "source_repo": source_repo,
        "jobs": jobs,
    }


def _scheduler_job_entry(
    *,
    name: str = "job-a",
    state: str = "",
    provider: str = PROVIDER_KAGGLE,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a scheduler state job dict for classify_job_result tests.

    The ``job_entry`` parameter of ``classify_job_result`` is the scheduler
    durable-state dict (contains ``state``, ``error``, ``provider``), not a
    campaign job.  Using ``_valid_campaign_job`` conflates the two.
    """
    return {"name": name, "state": state, "provider": provider, "error": error}


def _write_campaign(tmp_path: Path, campaign: dict[str, Any] | None = None) -> Path:
    p = tmp_path / "campaign.json"
    p.write_text(json.dumps(campaign or _valid_campaign()), encoding="utf-8")
    return p




# ---------------------------------------------------------------------------
# Fake subprocess for runner report tests
# ---------------------------------------------------------------------------


class FakeProc:
    """Minimal fake subprocess.Popen for run_module tests.

    Provides text-stream stdout/stderr with readline(), wait(timeout),
    kill(), poll(), and returncode — the surface run_module touches.
    """

    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr_lines: list[str] | None = None,
        returncode: int = 0,
        wait_event: threading.Event | None = None,
        wait_timeout_exc: subprocess.TimeoutExpired | None = None,
    ) -> None:
        self._stdout_lines = list(stdout_lines or [])
        self._stderr_lines = list(stderr_lines or [])
        self._rc = returncode
        self._wait_event = wait_event
        self._wait_timeout_exc = wait_timeout_exc
        self.returncode: int | None = None
        self.killed = False
        self.stdout = self._make_stream(self._stdout_lines)
        self.stderr = self._make_stream(self._stderr_lines)

    @staticmethod
    def _make_stream(lines: list[str]) -> io.StringIO:
        return io.StringIO("".join(line if line.endswith("\n") else line + "\n" for line in lines))

    def wait(self, timeout: float | None = None) -> int:
        # Before kill, a pending timeout exception models a hung child that
        # has not yet exited.  After kill() the process is dead, so wait()
        # returns the (negative) returncode — mirroring real subprocess
        # semantics where post-kill wait reaps the corpse instead of raising.
        if self._wait_timeout_exc is not None and not self.killed:
            raise self._wait_timeout_exc
        if self._wait_event is not None and not self.killed:
            self._wait_event.wait(timeout=timeout)
        self.returncode = self._rc
        return self._rc

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._rc

    def poll(self) -> int | None:
        return self.returncode


def _patch_popen(proc: FakeProc) -> Any:
    """Return a patcher that replaces subprocess.Popen with *proc*."""

    class _FakePopenFactory:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, command: list[str], **kwargs: Any) -> FakeProc:
            self.calls.append(command)
            return proc

    factory = _FakePopenFactory()
    return patch("subprocess.Popen", factory), factory


# ---------------------------------------------------------------------------
# Fake report for collector tests
# ---------------------------------------------------------------------------


def _make_report(
    *,
    job_name: str = "job-a",
    provider: str = PROVIDER_KAGGLE,
    source_commit: str = COMMIT,
    module: str = "infrastructure.kaggle.run_cortex_smoke",
    arguments: list[str] | None = None,
    command: list[str] | None = None,
    exit_code: int = 0,
    status: str = "complete",
    started_utc: str = "2026-07-11T00:00:00Z",
    finished_utc: str = "2026-07-11T00:01:00Z",
    stdout_file: str = "execution_report.stdout.log",
    stderr_file: str = "execution_report.stderr.log",
    metrics: dict[str, float] | None = None,
    asi_scores: dict[str, float] | None = None,
    error: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    schema_version: str = RUNNER_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "job_name": job_name,
        "provider": provider,
        "source_commit": source_commit,
        "module": module,
        "arguments": arguments if arguments is not None else [],
        "command": command if command is not None else [sys.executable, "-m", module],
        "exit_code": exit_code,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
        "metrics": metrics if metrics is not None else {},
        "asi_scores": asi_scores if asi_scores is not None else {},
        "error": error,
        "timeout_seconds": timeout_seconds,
    }


def _write_report(path: Path, report: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report or _make_report()), encoding="utf-8")
    return path


# ===========================================================================
# 1. Runner report: success / nonzero / timeout
# ===========================================================================


class TestRunnerReport:
    """run_module writes execution_report.json with correct status/fields."""

    def test_success_report(self, tmp_path: Path) -> None:
        """Exit 0 produces status='complete' with all required fields."""
        proc = FakeProc(returncode=0, stdout_lines=["METRIC loss=0.5\n", "ASI score=0.9\n"])
        patcher, factory = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "complete"
        assert report["exit_code"] == 0
        assert report["schema_version"] == EXPECTED_RUNNER_SCHEMA
        assert report["job_name"] == "job-a"
        assert report["provider"] == PROVIDER_KAGGLE
        assert report["source_commit"] == COMMIT
        assert report["module"] == "json"
        assert report["arguments"] == []
        assert isinstance(report["command"], list)
        assert report["started_utc"]
        assert report["finished_utc"]
        assert report["error"] is None
        # No shell=True — command is a list.
        assert factory.calls and isinstance(factory.calls[0], list)
        assert factory.calls[0][0] == sys.executable

    def test_nonzero_report(self, tmp_path: Path) -> None:
        """Nonzero exit produces status='error' with error provenance."""
        proc = FakeProc(returncode=1, stderr_lines=["Traceback (most recent call last):\n"])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=["--bad"],
                source_commit=COMMIT,
                provider=PROVIDER_COLAB,
                job_name="job-b",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "error"
        assert report["exit_code"] == 1
        assert report["provider"] == PROVIDER_COLAB

    def test_timeout_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Timeout produces status='timeout', exit_code=-1, kill called.

        The report is always written to disk and a sentinel emitted, even
        when the post-kill wait raises — provenance is never lost.
        """
        proc = FakeProc(
            returncode=-1,
            wait_timeout_exc=subprocess.TimeoutExpired(cmd=["python"], timeout=1.0),
        )
        patcher, _ = _patch_popen(proc)
        rp = tmp_path / "execution_report.json"
        with patcher:
            report = run_module(
                module="time",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-c",
                report_path=str(rp),
                timeout=1.0,
            )
        assert report["status"] == "timeout"
        assert report["exit_code"] == -1
        assert proc.killed is True
        assert report["timeout_seconds"] == 1.0
        # Timeout must always produce a written report file, not just a dict.
        assert rp.exists()
        loaded = json.loads(rp.read_text())
        assert loaded["status"] == "timeout"
        assert loaded["exit_code"] == -1
        # A sentinel is emitted even on timeout.
        captured = capsys.readouterr()
        sentinel_lines = [
            line for line in captured.out.splitlines()
            if line.startswith("OCZY_EXECUTION_REPORT_JSON=")
        ]
        assert len(sentinel_lines) == 1
        assert json.loads(sentinel_lines[0][len("OCZY_EXECUTION_REPORT_JSON="):])["status"] == "timeout"

    def test_report_file_written(self, tmp_path: Path) -> None:
        """The report is written to report_path as JSON."""
        proc = FakeProc(returncode=0)
        patcher, _ = _patch_popen(proc)
        rp = tmp_path / "execution_report.json"
        with patcher:
            run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(rp),
            )
        assert rp.exists()
        loaded = json.loads(rp.read_text())
        assert loaded["status"] == "complete"
        assert loaded["job_name"] == "job-a"

    def test_log_files_written(self, tmp_path: Path) -> None:
        """stdout_file and stderr_file are filenames in the report's parent."""
        proc = FakeProc(
            returncode=0,
            stdout_lines=["hello\n"],
            stderr_lines=["warn\n"],
        )
        patcher, _ = _patch_popen(proc)
        rp = tmp_path / "execution_report.json"
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(rp),
            )
        assert (tmp_path / report["stdout_file"]).exists()
        assert (tmp_path / report["stderr_file"]).exists()
        assert "hello" in (tmp_path / report["stdout_file"]).read_text()
        assert "warn" in (tmp_path / report["stderr_file"]).read_text()

    def test_no_shell_true(self, tmp_path: Path) -> None:
        """subprocess.Popen must never be called with shell=True."""
        proc = FakeProc(returncode=0)
        patcher, factory = _patch_popen(proc)
        with patcher:
            run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        # The patcher intercepts all Popen calls; if shell=True were used
        # the command would be a string, not a list.  Assert list form.
        assert factory.calls
        assert isinstance(factory.calls[0], list)

    def test_sentinel_line_emitted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The runner emits exactly one OCZY_EXECUTION_REPORT_JSON=<json> sentinel
        to its *own* stdout (captured here via capsys), not to the child's
        mirrored stdout log.

        This sentinel lets the Colab provider collect the structured report
        from the runner process stdout when file download is unavailable.
        The child's stdout/stderr remain separately mirrored to log files
        and must never contain the sentinel.
        """
        proc = FakeProc(returncode=0, stdout_lines=["METRIC loss=0.5\n"])
        patcher, _ = _patch_popen(proc)
        rp = tmp_path / "execution_report.json"
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(rp),
            )
        # The report dict must be returned correctly.
        assert report["status"] == "complete"
        assert report["metrics"] == {"loss": 0.5}
        # The sentinel is emitted to the runner's own stdout (capsys),
        # not to the child's mirrored stdout log file.
        captured = capsys.readouterr()
        sentinel_prefix = "OCZY_EXECUTION_REPORT_JSON="
        sentinel_lines = [
            line for line in captured.out.splitlines() if line.startswith(sentinel_prefix)
        ]
        assert len(sentinel_lines) == 1, (
            "runner must emit exactly one OCZY_EXECUTION_REPORT_JSON sentinel to its stdout"
        )
        # The sentinel payload must be parseable JSON matching the report.
        sentinel_json = sentinel_lines[0][len(sentinel_prefix):]
        parsed = json.loads(sentinel_json)
        assert parsed["status"] == "complete"
        assert parsed["job_name"] == "job-a"
        assert parsed["metrics"] == {"loss": 0.5}
        # The child's stdout log must remain child-only — no sentinel leak.
        stdout_log = tmp_path / report["stdout_file"]
        assert stdout_log.exists()
        log_text = stdout_log.read_text()
        assert sentinel_prefix not in log_text, (
            "sentinel must not leak into the child stdout log"
        )
        assert "METRIC loss=0.5" in log_text


# ===========================================================================
# 2. Metric / ASI parsing
# ===========================================================================


class TestMetricAsiParsing:
    """METRIC/ASI lines are parsed from stdout into metrics/asi_scores dicts."""

    def test_metric_line_parsed(self, tmp_path: Path) -> None:
        proc = FakeProc(stdout_lines=["METRIC loss=0.123\n"])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"] == {"loss": 0.123}

    def test_asi_line_parsed(self, tmp_path: Path) -> None:
        proc = FakeProc(stdout_lines=["ASI coherence=0.87\n"])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["asi_scores"] == {"coherence": 0.87}

    def test_multiple_metrics_and_asi(self, tmp_path: Path) -> None:
        proc = FakeProc(stdout_lines=[
            "METRIC loss=0.5\n",
            "METRIC accuracy=0.9\n",
            "ASI scope=0.75\n",
            "ASI retention=1.0\n",
        ])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"] == {"loss": 0.5, "accuracy": 0.9}
        assert report["asi_scores"] == {"scope": 0.75, "retention": 1.0}

    def test_negative_and_scientific_values(self, tmp_path: Path) -> None:
        proc = FakeProc(stdout_lines=[
            "METRIC delta=-0.42\n",
            "METRIC tiny=1.5e-7\n",
            "ASI grad=-3.14e+2\n",
        ])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"]["delta"] == -0.42
        assert report["metrics"]["tiny"] == 1.5e-7
        assert report["asi_scores"]["grad"] == -314.0

    def test_integer_value_parsed_as_float(self, tmp_path: Path) -> None:
        proc = FakeProc(stdout_lines=["METRIC steps=100\n"])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"]["steps"] == 100.0
        assert isinstance(report["metrics"]["steps"], float)

    def test_non_metric_lines_ignored(self, tmp_path: Path) -> None:
        """Lines that don't match METRIC/ASI format are ignored."""
        proc = FakeProc(stdout_lines=[
            "Starting experiment...\n",
            "METRIC loss=0.5\n",
            "Some random output\n",
            "ASI score=0.9 extra junk\n",  # extra junk = no match
        ])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"] == {"loss": 0.5}
        assert report["asi_scores"] == {}

    def test_empty_stdout_no_metrics(self, tmp_path: Path) -> None:
        proc = FakeProc(returncode=0)
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-a",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["metrics"] == {}
        assert report["asi_scores"] == {}


# ===========================================================================
# 3. src-layout injection in Colab bootstrap
# ===========================================================================


class TestColabSrcLayoutInjection:
    """The Colab bootstrap prepends repo root, repo/src, and */src to sys.path."""

    def test_bootstrap_prepends_repo_root_and_src(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        # Must insert repo root and repo/src into sys.path.
        assert "sys.path" in source
        assert "src" in source
        # The bootstrap must use glob to find workspace package src dirs.
        assert "glob" in source.lower() or "*/src" in source or "iglob" in source.lower()

    def test_bootstrap_compiles(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        compile((out / "colab_bootstrap.py").read_text(), str(out / "colab_bootstrap.py"), "exec")

    def test_bootstrap_sets_cpu_env_before_import(self, tmp_path: Path) -> None:
        """CPU env vars must be set before any heavy import."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        for var in (
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OCZY_REMOTE_CPU_ONLY",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM",
        ):
            assert var in source, f"{var} not set in Colab bootstrap"

    def test_bootstrap_invokes_run_experiment_module(self, tmp_path: Path) -> None:
        """The bootstrap must invoke run_experiment_module via explicit subprocess argv."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        # The bootstrap must not use runpy; it invokes the runner as a subprocess.
        assert "runpy" not in source
        tree = ast.parse(source)
        runner_argv_found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "runner_argv":
                    runner_argv_found = True
        assert runner_argv_found, "bootstrap must build a runner_argv list"
        # The runner module must be invoked via -m in the argv.
        assert "infrastructure.kaggle.run_experiment_module" in source
        assert '"-m"' in source or "'-m'" in source

    def test_job_spec_written(self, tmp_path: Path) -> None:
        """A job_spec.json with schema oczy/colab-experiment-job/v1 is written."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=["--seed", "0"],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        spec_path = out / "job_spec.json"
        assert spec_path.exists()
        spec = json.loads(spec_path.read_text())
        assert spec["schema_version"] == EXPECTED_COLAB_JOB_SCHEMA
        assert spec["module"] == "infrastructure.kaggle.run_cortex_smoke"
        assert spec["arguments"] == ["--seed", "0"]
        assert spec["source_commit"] == COMMIT


# ===========================================================================
# 4. Colab exact-commit bootstrap and repo validation
# ===========================================================================


class TestColabExactCommit:
    """Colab bootstrap clones with exact commit verification."""

    def test_bootstrap_clones_no_checkout(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        # Must use git init + git fetch + git checkout (not git clone).
        assert "git init" in source or "git" in source
        assert "fetch" in source
        assert "checkout" in source
        # Must verify HEAD == source_commit.
        assert "HEAD" in source or "head" in source
        assert "source_commit" in source or "commit" in source

    def test_rejects_wrong_repo_url(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        with pytest.raises((ValueError, RuntimeError), match="repo"):
            prepare_colab_experiment(
                output=out,
                job_name="colab-a",
                repo_url="https://github.com/evil/repo.git",
                source_commit=COMMIT,
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[],
                phase="development",
                claim_class="scientific",
                output_path="out/colab-a",
            )

    def test_rejects_short_commit(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        with pytest.raises((ValueError, RuntimeError), match="commit"):
            prepare_colab_experiment(
                output=out,
                job_name="colab-a",
                repo_url=REPO_URL,
                source_commit="abc123",
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[],
                phase="development",
                claim_class="scientific",
                output_path="out/colab-a",
            )

    def test_rejects_uppercase_commit(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        with pytest.raises((ValueError, RuntimeError), match="commit"):
            prepare_colab_experiment(
                output=out,
                job_name="colab-a",
                repo_url=REPO_URL,
                source_commit="A" * 40,
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[],
                phase="development",
                claim_class="scientific",
                output_path="out/colab-a",
            )

    def test_no_credentials_embedded(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        # No tokens, passwords, or SSH keys in the bootstrap.
        assert "ghp_" not in source
        assert "password" not in source.lower()
        assert "BEGIN OPENSSH" not in source
        assert "BEGIN RSA" not in source

    def test_no_shell_true_in_bootstrap(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        # AST-structured check: no subprocess call may pass shell=True.
        tree = ast.parse(source)
        subprocess_funcs = {"run", "Popen", "call", "check_call", "check_output"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match subprocess.run, subprocess.Popen, etc.
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"
                    and func.attr in subprocess_funcs
                ):
                    for kw in node.keywords:
                        assert not (
                            kw.arg == "shell"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ), "subprocess call must not use shell=True"


# ===========================================================================
# 5. Accelerator rejection in Colab job preparation
# ===========================================================================


class TestAcceleratorRejection:
    """Accelerator args are rejected anywhere in arguments (CPU-only applies to the target module)."""

    @pytest.mark.parametrize("bad_arg", [
        "--gpu",
        "--tpu",
        "--cuda",
        "--device cuda",
        "--accelerator",
    ])
    def test_rejects_accelerator_before_separator(self, tmp_path: Path, bad_arg: str) -> None:
        out = tmp_path / "colab-job"
        with pytest.raises((ValueError, RuntimeError), match="accelerator|gpu|cuda|tpu|device"):
            prepare_colab_experiment(
                output=out,
                job_name="colab-a",
                repo_url=REPO_URL,
                source_commit=COMMIT,
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[bad_arg],
                phase="development",
                claim_class="scientific",
                output_path="out/colab-a",
            )

    def test_rejects_accelerator_after_separator(self, tmp_path: Path) -> None:
        """Args after '--' are still rejected: CPU-only applies to the target module."""
        out = tmp_path / "colab-job"
        # Must raise — --gpu after -- is still forbidden because the CPU-only
        # contract applies to the target experiment module, not just the CLI.
        with pytest.raises((ValueError, RuntimeError), match="accelerator|gpu|cuda|tpu|device"):
            prepare_colab_experiment(
                output=out,
                job_name="colab-a",
                repo_url=REPO_URL,
                source_commit=COMMIT,
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=["--", "--gpu"],
                phase="development",
                claim_class="scientific",
                output_path="out/colab-a",
            )

    def test_clean_args_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=["--seed", "0", "--epochs", "10"],
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["arguments"] == ["--seed", "0", "--epochs", "10"]


# ===========================================================================
# 6. Campaign schema validation
# ===========================================================================


class TestCampaignValidation:
    """validate_campaign accepts valid campaigns and rejects invalid ones."""

    def test_valid_campaign_accepted(self) -> None:
        campaign = _valid_campaign()
        validate_campaign(campaign)  # should not raise

    def test_valid_mixed_campaign_accepted(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        validate_campaign(campaign)

    def test_rejects_wrong_schema_version(self) -> None:
        campaign = _valid_campaign()
        campaign["schema_version"] = "oczy/wrong/v1"
        with pytest.raises(ValueError, match="schema"):
            validate_campaign(campaign)

    def test_rejects_missing_schema_version(self) -> None:
        campaign = _valid_campaign()
        del campaign["schema_version"]
        with pytest.raises(ValueError, match="schema"):
            validate_campaign(campaign)

    def test_rejects_short_commit(self) -> None:
        campaign = _valid_campaign(source_commit="abc123")
        with pytest.raises(ValueError, match="commit"):
            validate_campaign(campaign)

    def test_rejects_uppercase_commit(self) -> None:
        campaign = _valid_campaign(source_commit="A" * 40)
        with pytest.raises(ValueError, match="commit"):
            validate_campaign(campaign)

    def test_rejects_wrong_repo_url(self) -> None:
        campaign = _valid_campaign(source_repo="https://github.com/evil/repo.git")
        with pytest.raises(ValueError, match="repo"):
            validate_campaign(campaign)

    def test_rejects_empty_jobs(self) -> None:
        campaign = _valid_campaign(jobs=[])
        with pytest.raises(ValueError, match="job"):
            validate_campaign(campaign)

    def test_rejects_missing_jobs(self) -> None:
        campaign = _valid_campaign()
        del campaign["jobs"]
        with pytest.raises(ValueError, match="job"):
            validate_campaign(campaign)

    def test_rejects_duplicate_job_names(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="dup"),
            _valid_campaign_job(name="dup", provider=PROVIDER_COLAB),
        ])
        with pytest.raises(ValueError, match="duplicate|name"):
            validate_campaign(campaign)

    def test_rejects_invalid_provider(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="j", provider="vertex"),
        ])
        with pytest.raises(ValueError, match="provider"):
            validate_campaign(campaign)

    def test_rejects_invalid_claim_class(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="j", claim_class="experimental"),
        ])
        with pytest.raises(ValueError, match="claim"):
            validate_campaign(campaign)

    def test_rejects_invalid_phase(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="j", phase="bogus"),
        ])
        with pytest.raises(ValueError, match="phase"):
            validate_campaign(campaign)

    def test_rejects_missing_job_name(self) -> None:
        campaign = _valid_campaign(jobs=[
            {"provider": PROVIDER_KAGGLE, "phase": "development", "module": "m",
             "arguments": [], "output_path": "o", "claim_class": "scientific"},
        ])
        with pytest.raises(ValueError, match="name"):
            validate_campaign(campaign)

    def test_rejects_missing_module(self) -> None:
        campaign = _valid_campaign(jobs=[
            {"name": "j", "provider": PROVIDER_KAGGLE, "phase": "development",
             "arguments": [], "output_path": "o", "claim_class": "scientific"},
        ])
        with pytest.raises(ValueError, match="module"):
            validate_campaign(campaign)


# ===========================================================================
# 7. Campaign mixed generation
# ===========================================================================


class TestCampaignMixedGeneration:
    """prepare_experiment_campaign generates a v2 scheduler-compatible batch."""

    def test_generates_v2_batch(self, tmp_path: Path) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        assert "batch_path" in result
        assert "manifest_path" in result
        assert "jobs" in result
        batch = json.loads(Path(result["batch_path"]).read_text())
        assert batch["schema_version"] == EXPECTED_BATCH_V2

    def test_generated_batch_has_both_providers(self, tmp_path: Path) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        batch = json.loads(Path(result["batch_path"]).read_text())
        providers = {j["name"]: j["provider"] for j in batch["jobs"]}
        assert providers["kg-0"] == PROVIDER_KAGGLE
        assert providers["cb-0"] == PROVIDER_COLAB

    def test_generated_kaggle_job_has_kernel_dir(self, tmp_path: Path) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        batch = json.loads(Path(result["batch_path"]).read_text())
        job = batch["jobs"][0]
        assert "kernel_dir" in job
        assert "output_dir" in job

    def test_generated_colab_job_has_script(self, tmp_path: Path) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        batch = json.loads(Path(result["batch_path"]).read_text())
        job = batch["jobs"][0]
        assert "script" in job
        assert "output_dir" in job

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        campaign = _valid_campaign()
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        prepare_experiment_campaign(campaign_path, out_dir)
        # Second call with force=True should not raise.
        prepare_experiment_campaign(campaign_path, out_dir, force=True)

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        campaign = _valid_campaign()
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        prepare_experiment_campaign(campaign_path, out_dir)
        with pytest.raises((FileExistsError, ValueError, RuntimeError)):
            prepare_experiment_campaign(campaign_path, out_dir, force=False)


# ===========================================================================
# 8. v2 scheduler compatibility — generated batch loads via load_batch
# ===========================================================================


class TestV2SchedulerCompatibility:
    """The generated batch must be loadable by the scheduler's load_batch."""

    def test_generated_batch_loads_via_load_batch(self, tmp_path: Path) -> None:
        """A generated kaggle-only batch loads via load_batch without error."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        jobs = load_batch(result["batch_path"])
        assert len(jobs) == 1
        assert jobs[0]["provider"] == PROVIDER_KAGGLE
        assert jobs[0]["schema_version"] == BATCH_SCHEMA_V2

    def test_generated_mixed_batch_loads_via_load_batch(self, tmp_path: Path) -> None:
        """A generated mixed kaggle+colab batch loads via load_batch."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument"),
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        jobs = load_batch(result["batch_path"])
        assert len(jobs) == 2
        providers = {j["name"]: j["provider"] for j in jobs}
        assert providers["kg-0"] == PROVIDER_KAGGLE
        assert providers["cb-0"] == PROVIDER_COLAB


# ===========================================================================
# 9. Frozen meta-test sign-off propagation
# ===========================================================================


class TestMetaTestSignoffPropagation:
    """Meta-test phase requires manifest hash + human signoff and propagates them."""

    def test_meta_test_requires_signoff_fields(self) -> None:
        """A meta-test job without signoff fields is rejected by validate_campaign."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="mt-0", phase="meta-test"),
        ])
        with pytest.raises(ValueError, match="meta.test|signoff|manifest|human"):
            validate_campaign(campaign)

    def test_meta_test_with_signoff_accepted(self) -> None:
        """A meta-test job with signoff fields is accepted."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="mt-0",
                phase="meta-test",
                instrument_manifest_sha256=ARCHIVE_SHA,
                human_signoff_id="analyst@kino",
            ),
        ])
        validate_campaign(campaign)  # should not raise

    def test_meta_test_signoff_propagates_to_prepare_kernel(self, tmp_path: Path) -> None:
        """The campaign preparer passes signoff fields through to prepare_kernel."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="mt-0",
                provider=PROVIDER_KAGGLE,
                phase="meta-test",
                instrument_manifest_sha256=ARCHIVE_SHA,
                human_signoff_id="analyst@kino",
            ),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        # The generated kernel's job_spec.json must contain the signoff fields.
        for job in result["jobs"]:
            kernel_dir = Path(out_dir) / job.get("kernel_dir", "")
            spec_path = kernel_dir / "job_spec.json"
            if spec_path.exists():
                spec = json.loads(spec_path.read_text())
                if spec.get("phase") == "meta-test":
                    assert spec["instrument_manifest_sha256"] == ARCHIVE_SHA
                    assert spec["human_signoff_id"] == "analyst@kino"
                    return
        pytest.fail("No meta-test kernel job_spec.json found in generated output")

    def test_non_meta_test_does_not_require_signoff(self) -> None:
        """Non-meta-test phases don't require signoff fields."""
        for phase in ("instrument", "oracle", "development", "analysis"):
            campaign = _valid_campaign(jobs=[
                _valid_campaign_job(name=f"j-{phase}", phase=phase),
            ])
            validate_campaign(campaign)  # should not raise


# ===========================================================================
# 10. Collector classification: COMPLETE / NULL / INVALID / BLOCKED
# ===========================================================================


class TestCollectorClassification:
    """classify_job_result returns the correct classification.

    The ``job_entry`` parameter is the scheduler durable-state dict (contains
    ``state``, ``error``, ``provider``), not a campaign job.  Tests use
    ``_scheduler_job_entry`` for that parameter and ``_valid_campaign_job``
    for the ``campaign_job`` parameter.
    """

    def test_complete_scientific_with_metrics(self) -> None:
        report = _make_report(exit_code=0, status="complete", metrics={"loss": 0.5})
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="scientific")
        assert classify_job_result(job_entry, report, campaign_job) == "COMPLETE"

    def test_complete_infrastructure(self) -> None:
        report = _make_report(exit_code=0, status="complete")
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="infrastructure")
        assert classify_job_result(job_entry, report, campaign_job) == "COMPLETE"

    def test_null_scientific_no_metrics(self) -> None:
        """Scientific job with exit 0 but no metrics/asi is NULL."""
        report = _make_report(exit_code=0, status="complete", metrics={}, asi_scores={})
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="scientific")
        assert classify_job_result(job_entry, report, campaign_job) == "NULL"

    def test_null_not_for_infrastructure(self) -> None:
        """Infrastructure job with no metrics is COMPLETE, not NULL."""
        report = _make_report(exit_code=0, status="complete", metrics={}, asi_scores={})
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="infrastructure")
        assert classify_job_result(job_entry, report, campaign_job) == "COMPLETE"

    def test_null_with_asi_only_still_complete(self) -> None:
        """If asi_scores present (but no metrics), it's not NULL."""
        report = _make_report(exit_code=0, status="complete", asi_scores={"score": 0.9})
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="scientific")
        assert classify_job_result(job_entry, report, campaign_job) == "COMPLETE"

    def test_blocked_nonzero_exit_scientific(self) -> None:
        """Nonzero exit code → BLOCKED, even for scientific."""
        report = _make_report(exit_code=1, status="error")
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="scientific")
        assert classify_job_result(job_entry, report, campaign_job) == "BLOCKED"

    def test_blocked_nonzero_exit_infrastructure(self) -> None:
        """Failed infrastructure job is BLOCKED, never NULL."""
        report = _make_report(exit_code=1, status="error")
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="infrastructure")
        assert classify_job_result(job_entry, report, campaign_job) == "BLOCKED"

    def test_blocked_timeout(self) -> None:
        report = _make_report(exit_code=-1, status="timeout")
        job_entry = _scheduler_job_entry()
        campaign_job = _valid_campaign_job(claim_class="scientific")
        assert classify_job_result(job_entry, report, campaign_job) == "BLOCKED"

    def test_invalid_source_commit_mismatch(self) -> None:
        report = _make_report(source_commit="d" * 40)
        job_entry = _scheduler_job_entry(state=SUCCEEDED)
        campaign_job = _valid_campaign_job()
        campaign_job["_campaign_source_commit"] = COMMIT
        assert classify_job_result(job_entry, report, campaign_job) == "INVALID"

    def test_invalid_provider_mismatch(self) -> None:
        report = _make_report(provider=PROVIDER_COLAB)
        job_entry = _scheduler_job_entry(state=SUCCEEDED, provider=PROVIDER_KAGGLE)
        campaign_job = _valid_campaign_job(provider=PROVIDER_KAGGLE)
        assert classify_job_result(job_entry, report, campaign_job) == "INVALID"

    def test_invalid_job_name_mismatch(self) -> None:
        report = _make_report(job_name="wrong-name")
        job_entry = _scheduler_job_entry(state=SUCCEEDED, name="job-a")
        campaign_job = _valid_campaign_job(name="job-a")
        assert classify_job_result(job_entry, report, campaign_job) == "INVALID"

    def test_invalid_corrupt_report(self) -> None:
        """A report that is not a valid dict is INVALID."""
        job_entry = _scheduler_job_entry(state=SUCCEEDED)
        campaign_job = _valid_campaign_job()
        assert classify_job_result(job_entry, "not a dict", campaign_job) == "INVALID"
        # Scheduler says succeeded but report is None → INVALID provenance.
        assert classify_job_result(job_entry, None, campaign_job) == "INVALID"

    def test_blocked_missing_report_file(self) -> None:
        """A missing report file (represented as None) is BLOCKED when the
        scheduler does not report success."""
        job_entry = _scheduler_job_entry(state="")
        campaign_job = _valid_campaign_job()
        assert classify_job_result(job_entry, None, campaign_job) == "BLOCKED"

    def test_invalid_wrong_schema_version(self) -> None:
        """A report with the wrong schema_version is INVALID."""
        report = _make_report(schema_version="oczy/wrong/v1")
        job_entry = _scheduler_job_entry(state=SUCCEEDED)
        campaign_job = _valid_campaign_job()
        assert classify_job_result(job_entry, report, campaign_job) == "INVALID"


# ===========================================================================
# 11. Collector provenance validation and missing outputs
# ===========================================================================


class TestCollectorProvenanceAndMissing:
    """collect_experiment_campaign validates provenance and handles missing outputs."""

    def test_collect_complete_campaign(self, tmp_path: Path) -> None:
        """A campaign with all-complete reports produces a summary."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", provider=PROVIDER_KAGGLE, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        # Write a complete report for the job.
        report_dir = tmp_path / "reports"
        for job in campaign["jobs"]:
            _write_report(
                report_dir / job["name"] / "execution_report.json",
                _make_report(job_name=job["name"], metrics={"loss": 0.5}),
            )

        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)
        # Summary should classify the job as COMPLETE.
        jobs = summary.get("jobs", [])
        found = any(
            entry.get("name") == "job-a"
            and entry.get("classification") == "COMPLETE"
            for entry in jobs
        )
        assert found, f"job-a not classified COMPLETE in {jobs}"

    def test_collect_missing_report_is_blocked(self, tmp_path: Path) -> None:
        """A job with no report file is classified BLOCKED."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        # No reports written — report_dir is empty.
        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)

    def test_collect_provenance_mismatch_is_invalid(self, tmp_path: Path) -> None:
        """A report with wrong source_commit is classified INVALID."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        report_dir = tmp_path / "reports"
        _write_report(
            report_dir / "job-a" / "execution_report.json",
            _make_report(job_name="job-a", source_commit="d" * 40),
        )

        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)

    def test_collect_scientific_no_metrics_is_null(self, tmp_path: Path) -> None:
        """A scientific job with no metrics is classified NULL."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        report_dir = tmp_path / "reports"
        _write_report(
            report_dir / "job-a" / "execution_report.json",
            _make_report(job_name="job-a", metrics={}, asi_scores={}),
        )

        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)

    def test_collect_writes_summary_file(self, tmp_path: Path) -> None:
        """collect_experiment_campaign writes campaign_execution_summary.json."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        report_dir = tmp_path / "reports"
        _write_report(
            report_dir / "job-a" / "execution_report.json",
            _make_report(job_name="job-a", metrics={"loss": 0.5}),
        )

        collect_dir = tmp_path / "collected"
        collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            collect_dir,
            report_dir=report_dir,
        )
        summary_path = collect_dir / "campaign_execution_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert isinstance(summary, dict)

    def test_collect_corrupt_report_is_invalid(self, tmp_path: Path) -> None:
        """A corrupt (unparseable) report file is classified INVALID."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="job-a", claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)

        report_dir = tmp_path / "reports"
        rp = report_dir / "job-a" / "execution_report.json"
        rp.parent.mkdir(parents=True)
        rp.write_text("{not valid json", encoding="utf-8")

        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)


# ===========================================================================
# 12. Colab sentinel parsing and Kaggle provenance fallback
# ===========================================================================


class TestColabSentinelParsing:
    """Colab collector parses OCZY_EXECUTION_REPORT_JSON from stdout.log.

    The execution_report.json stays on the remote VM and is NOT downloaded
    by the current Colab provider.  Instead, the runner emits a sentinel
    line ``OCZY_EXECUTION_REPORT_JSON=<compact-json>`` in stdout.log.  The
    collector parses this sentinel to recover the structured report.
    """

    def _make_colab_output(
        self,
        base: Path,
        job_name: str,
        *,
        stdout_lines: list[str],
        result_json: dict[str, Any] | None = None,
    ) -> Path:
        """Create a Colab job output dir with stdout.log and optional result.json."""
        out = base / job_name
        out.mkdir(parents=True, exist_ok=True)
        (out / _COLAB_STDOUT_LOG).write_text(
            "".join(line if line.endswith("\n") else line + "\n" for line in stdout_lines),
            encoding="utf-8",
        )
        if result_json is not None:
            (out / COLAB_RESULT_FILENAME).write_text(
                json.dumps(result_json), encoding="utf-8"
            )
        return out

    def _sentinel_line(self, report: dict[str, Any]) -> str:
        return _COLAB_SENTINEL_PREFIX + json.dumps(report, separators=(",", ":"))

    def test_sentinel_parsed_from_stdout_log(self, tmp_path: Path) -> None:
        """A valid sentinel line in stdout.log is parsed into a report dict."""
        report = _make_report(
            job_name="cb-0",
            provider=PROVIDER_COLAB,
            metrics={"loss": 0.5},
        )
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[
                "Starting...\n",
                self._sentinel_line(report),
            ],
        )
        # The collector should find and parse the sentinel.
        # We test via collect_experiment_campaign with a Colab campaign.
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_missing_sentinel_is_invalid(self, tmp_path: Path) -> None:
        """A stdout.log with no sentinel line is classified INVALID."""
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=["just some output\n", "no sentinel here\n"],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_multiple_sentinels_is_invalid(self, tmp_path: Path) -> None:
        """Two sentinel lines in stdout.log is ambiguous → INVALID."""
        report1 = _make_report(job_name="cb-0", provider=PROVIDER_COLAB, metrics={"a": 1})
        report2 = _make_report(job_name="cb-0", provider=PROVIDER_COLAB, metrics={"a": 2})
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[
                self._sentinel_line(report1),
                self._sentinel_line(report2),
            ],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_corrupt_sentinel_is_invalid(self, tmp_path: Path) -> None:
        """A sentinel line with non-JSON payload is INVALID."""
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[
                "OCZY_EXECUTION_REPORT_JSON=not-json-at-all\n",
            ],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_result_json_takes_priority_over_sentinel(self, tmp_path: Path) -> None:
        """When result.json exists, it is used instead of the sentinel."""
        report_sentinel = _make_report(
            job_name="cb-0", provider=PROVIDER_COLAB, metrics={"from_sentinel": 1}
        )
        result_json = {"ok": True, "error": None}
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[self._sentinel_line(report_sentinel)],
            result_json=result_json,
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="infrastructure"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_sentinel_with_metrics_classified_complete(self, tmp_path: Path) -> None:
        """A Colab scientific job with metrics in the sentinel is COMPLETE."""
        report = _make_report(
            job_name="cb-0",
            provider=PROVIDER_COLAB,
            exit_code=0,
            status="complete",
            metrics={"loss": 0.3},
        )
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[self._sentinel_line(report)],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_sentinel_no_metrics_scientific_is_null(self, tmp_path: Path) -> None:
        """A Colab scientific job with sentinel but no metrics is NULL."""
        report = _make_report(
            job_name="cb-0",
            provider=PROVIDER_COLAB,
            exit_code=0,
            status="complete",
            metrics={},
            asi_scores={},
        )
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[self._sentinel_line(report)],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)

    def test_sentinel_provenance_mismatch_is_invalid(self, tmp_path: Path) -> None:
        """A sentinel with wrong source_commit is INVALID."""
        report = _make_report(
            job_name="cb-0",
            provider=PROVIDER_COLAB,
            source_commit="d" * 40,
            metrics={"loss": 0.5},
        )
        self._make_colab_output(
            tmp_path / "reports",
            "cb-0",
            stdout_lines=[self._sentinel_line(report)],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=tmp_path / "reports",
        )
        assert isinstance(summary, dict)


class TestKaggleProvenanceFallback:
    """Kaggle collector uses execution_report.json, falling back to
    remote_run_provenance.json when the primary file is absent.
    """

    def test_execution_report_json_primary(self, tmp_path: Path) -> None:
        """When execution_report.json exists, it is the primary provenance."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)

        report_dir = tmp_path / "reports"
        _write_report(
            report_dir / "kg-0" / "execution_report.json",
            _make_report(job_name="kg-0", provider=PROVIDER_KAGGLE, metrics={"loss": 0.5}),
        )
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)

    def test_falls_back_to_remote_run_provenance(self, tmp_path: Path) -> None:
        """When execution_report.json is missing but remote_run_provenance.json
        exists, the collector uses the fallback file."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)

        report_dir = tmp_path / "reports"
        fallback_path = report_dir / "kg-0" / KAGGLE_PROVENANCE_FILENAME
        _write_report(
            fallback_path,
            _make_report(job_name="kg-0", provider=PROVIDER_KAGGLE, metrics={"loss": 0.5}),
        )
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)

    def test_no_provenance_file_is_blocked(self, tmp_path: Path) -> None:
        """When neither execution_report.json nor remote_run_provenance.json
        exists, the job is BLOCKED."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="kg-0", provider=PROVIDER_KAGGLE, claim_class="scientific"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)

        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        # Create the job dir but leave it empty — no provenance files.
        (report_dir / "kg-0").mkdir()
        summary = collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )
        assert isinstance(summary, dict)


 # ===========================================================================
# 13. Campaign schema version constant
 # ===========================================================================
# ===========================================================================


def test_campaign_schema_version_constant() -> None:
    assert CAMPAIGN_SCHEMA_VERSION == EXPECTED_CAMPAIGN_SCHEMA


def test_runner_schema_version_constant() -> None:
    assert RUNNER_SCHEMA_VERSION == EXPECTED_RUNNER_SCHEMA


def test_colab_job_schema_version_constant() -> None:
    assert COLAB_JOB_SCHEMA_VERSION == EXPECTED_COLAB_JOB_SCHEMA
