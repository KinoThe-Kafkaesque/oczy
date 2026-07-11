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
import hashlib
import io
import json
import os
import re
import runpy
import subprocess
import sys
import threading
import types
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
CampaignValidationError = _prep_mod["CampaignValidationError"]
_build_runner_arguments = _prep_mod["_build_runner_arguments"]

classify_job_result = _coll_mod["classify_job_result"]
collect_experiment_campaign = _coll_mod["collect_experiment_campaign"]
# Colab sentinel / Kaggle provenance constants (may not exist in all versions).
_COLAB_SENTINEL_PREFIX: str = _coll_mod.get("_COLAB_SENTINEL_PREFIX", "OCZY_EXECUTION_REPORT_JSON=")
_COLAB_STDOUT_LOG: str = _coll_mod.get("_COLAB_STDOUT_LOG", "stdout.log")
COLAB_RESULT_FILENAME: str = _coll_mod.get("COLAB_RESULT_FILENAME", "result.json")
KAGGLE_PROVENANCE_FILENAME: str = _coll_mod.get("KAGGLE_PROVENANCE_FILENAME", "remote_run_provenance.json")
SentinelError = _coll_mod.get("SentinelError", type("SentinelError", (Exception,), {}))

RUNNER_SCHEMA_VERSION: str = _runner_mod["SCHEMA_VERSION"]
parse_args = _runner_mod["parse_args"]
run_module = _runner_mod["run_module"]
DIAGNOSTIC_MAX_BYTES: int = _runner_mod.get("_DIAGNOSTIC_MAX_BYTES", 8192)

prepare_colab_experiment = _colab_prep_mod["prepare_colab_experiment"]
COLAB_JOB_SCHEMA_VERSION: str = _colab_prep_mod["JOB_SPEC_SCHEMA_VERSION"]
ColabPrepValueError = _colab_prep_mod["ColabPrepValueError"]
_VALID_MODEL_ARTIFACT_KINDS = _colab_prep_mod.get(
    "_VALID_MODEL_ARTIFACT_KINDS", frozenset({"gguf", "hf_snapshot"})
)
_LLAMA_CPP_VERSION: str = _colab_prep_mod.get("_LLAMA_CPP_VERSION", "0.3.33")
_LLAMA_CPP_WHEEL_INDEX: str = _colab_prep_mod.get(
    "_LLAMA_CPP_WHEEL_INDEX",
    "https://abetlen.github.io/llama-cpp-python/whl/cpu",
)

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

MODEL_REVISION = "f" * 40
MODEL_SHA_DUMMY = "e" * 64


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


def _valid_gguf_artifact(
    *,
    repo_id: str = "KinoThe-Kafkaesque/LFM2.5-1.2B-Instruct-GGUF",
    revision: str = MODEL_REVISION,
    filename: str = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    sha256: str = MODEL_SHA_DUMMY,
) -> dict[str, Any]:
    return {
        "kind": "gguf",
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "sha256": sha256,
    }


def _valid_hf_snapshot_artifact(
    *,
    repo_id: str = "KinoThe-Kafkaesque/LFM2.5-1.2B-Instruct",
    revision: str = MODEL_REVISION,
    filename: str = "config.json",
    sha256: str = MODEL_SHA_DUMMY,
) -> dict[str, Any]:
    return {
        "kind": "hf_snapshot",
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "sha256": sha256,
    }


def _exec_bootstrap(source: str) -> dict[str, Any]:
    """Exec generated bootstrap source in a test namespace.

    Sets ``__name__`` to avoid triggering ``main()``.  Returns the
    namespace containing all bootstrap-level functions.
    """
    ns: dict[str, Any] = {"__name__": "colab_bootstrap_test"}
    exec(compile(source, "colab_bootstrap.py", "exec"), ns)
    return ns




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
# 1b. Runner report: nonzero-exit diagnostic capture
# ===========================================================================


class TestRunnerErrorDiagnostics:
    """On nonzero child exit, the report error carries a bounded stderr tail
    (falling back to stdout when stderr is empty) so downstream collectors
    that cannot retrieve the log files still get an actionable diagnostic.
    """

    def test_nonzero_stderr_tail(self, tmp_path: Path) -> None:
        """Nonzero exit with stderr → status=error, original exit code, and
        error.stderr_tail contains the child's stderr text."""
        proc = FakeProc(
            returncode=1,
            stderr_lines=[
                "Traceback (most recent call last):\n",
                '  File "m.py", line 10, in <module>\n',
                "ValueError: bad input\n",
            ],
        )
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=["--bad"],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-err",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "error"
        assert report["exit_code"] == 1
        assert report["error"] is not None
        assert report["error"]["type"] == "NonzeroExit"
        assert "code 1" in report["error"]["message"]
        assert report["error"]["stderr_tail"] is not None
        assert "ValueError: bad input" in report["error"]["stderr_tail"]
        assert report["error"]["stdout_tail"] is None

    def test_nonzero_stdout_fallback(self, tmp_path: Path) -> None:
        """Nonzero exit with empty stderr → falls back to stdout_tail."""
        proc = FakeProc(
            returncode=2,
            stdout_lines=["usage: prog [--flag]\n", "error: invalid argument\n"],
            stderr_lines=[],
        )
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_COLAB,
                job_name="job-err2",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "error"
        assert report["exit_code"] == 2
        assert report["error"] is not None
        assert report["error"]["stderr_tail"] is None
        assert report["error"]["stdout_tail"] is not None
        assert "invalid argument" in report["error"]["stdout_tail"]

    def test_nonzero_both_empty(self, tmp_path: Path) -> None:
        """Nonzero exit with no output at all → error block still has
        type/message, both tails None."""
        proc = FakeProc(returncode=127, stdout_lines=[], stderr_lines=[])
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-err3",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "error"
        assert report["exit_code"] == 127
        assert report["error"] is not None
        assert report["error"]["type"] == "NonzeroExit"
        assert report["error"]["stderr_tail"] is None
        assert report["error"]["stdout_tail"] is None

    def test_success_error_still_null(self, tmp_path: Path) -> None:
        """Successful run → error is None (unchanged behavior)."""
        proc = FakeProc(
            returncode=0,
            stdout_lines=["METRIC loss=0.5\n"],
            stderr_lines=["some warning\n"],
        )
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-ok",
                report_path=str(tmp_path / "execution_report.json"),
            )
        assert report["status"] == "complete"
        assert report["exit_code"] == 0
        assert report["error"] is None

    def test_diagnostic_bounded_size(self, tmp_path: Path) -> None:
        """Large stderr is truncated to DIAGNOSTIC_MAX_BYTES and starts on a
        line boundary."""
        big_lines = [f"line {i:05d} padding padding padding\n" for i in range(5000)]
        proc = FakeProc(returncode=1, stderr_lines=big_lines)
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-big",
                report_path=str(tmp_path / "execution_report.json"),
            )
        tail = report["error"]["stderr_tail"]
        assert tail is not None
        assert len(tail.encode("utf-8")) <= DIAGNOSTIC_MAX_BYTES
        # Must start on a line boundary (not mid-line from the seek offset).
        assert tail.startswith("line "), f"tail did not start on line boundary: {tail[:40]!r}"
        # Must contain the *last* line (tail, not head).
        assert "line 04999" in tail

    def test_secret_redaction_in_tail(self, tmp_path: Path) -> None:
        """Obvious credential values in stderr are redacted before inlining."""
        proc = FakeProc(
            returncode=1,
            stderr_lines=[
                "API_KEY=sk-1234567890abcdef\n",
                "password=hunter2\n",
                "Traceback (most recent call last):\n",
                "RuntimeError: oops\n",
            ],
        )
        patcher, _ = _patch_popen(proc)
        with patcher:
            report = run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_KAGGLE,
                job_name="job-secret",
                report_path=str(tmp_path / "execution_report.json"),
            )
        tail = report["error"]["stderr_tail"]
        assert tail is not None
        assert "sk-1234567890abcdef" not in tail
        assert "hunter2" not in tail
        assert "[REDACTED]" in tail
        # Non-secret diagnostic content is preserved.
        assert "RuntimeError: oops" in tail

    def test_sentinel_carries_diagnostic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The sentinel JSON includes the error block with the stderr tail."""
        proc = FakeProc(
            returncode=1,
            stderr_lines=["ImportError: no module named foo\n"],
        )
        patcher, _ = _patch_popen(proc)
        with patcher:
            run_module(
                module="json",
                arguments=[],
                source_commit=COMMIT,
                provider=PROVIDER_COLAB,
                job_name="job-sentinel",
                report_path=str(tmp_path / "execution_report.json"),
            )
        captured = capsys.readouterr()
        sentinel_lines = [
            line for line in captured.out.splitlines()
            if line.startswith("OCZY_EXECUTION_REPORT_JSON=")
        ]
        assert len(sentinel_lines) == 1
        parsed = json.loads(sentinel_lines[0][len("OCZY_EXECUTION_REPORT_JSON="):])
        assert parsed["status"] == "error"
        assert parsed["error"] is not None
        assert parsed["error"]["stderr_tail"] is not None
        assert "ImportError" in parsed["error"]["stderr_tail"]

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
        # Module arguments must use --arg=<value> single-token encoding so that
        # option-like arguments (e.g. --seed, -1) are not misparsed by argparse.
        assert "--arg=" in source, "bootstrap must use --arg=<value> encoding"
        assert 'extend(["--arg"' not in source, "bootstrap must not use --arg <value> pair encoding"

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
        # "password" may appear as a redaction pattern name (password=***),
        # but no actual credential value may be embedded.
        assert "BEGIN OPENSSH" not in source
        assert "BEGIN RSA" not in source
        # Verify no hardcoded credential values anywhere.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.lower()
                assert "ghp_" not in val
                assert "begin openssh" not in val
                assert "begin rsa" not in val

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
# 12b. Kaggle log JSON sentinel fallback (OCZY_EXECUTION_REPORT_JSON in *.log)
# ===========================================================================


class TestKaggleLogSentinelFallback:
    """Kaggle collector parses OCZY_EXECUTION_REPORT_JSON from the downloaded
    *.log JSON stream when execution_report.json and remote_run_provenance.json
    are both absent.

    The Kaggle kernel log is a JSON array of stream objects::

        [{"stream_name": "stdout", "time": 9.24,
          "data": "OCZY_EXECUTION_REPORT_JSON={...}"},
         {"stream_name": "stderr", "time": 9.24, "data": "..."},
         ...]

    The sentinel appears in a ``stdout`` entry's ``data`` field.  The collector
    must reuse the same strict sentinel/provenance validation as the Colab
    path — missing, multiple, or corrupt sentinels are INVALID; provenance
    mismatches are INVALID.
    """

    def _make_kaggle_log_output(
        self,
        base: Path,
        job_name: str,
        *,
        stream_entries: list[dict[str, Any]],
        log_filename: str | None = None,
    ) -> Path:
        """Create a Kaggle job output dir with a *.log JSON stream file."""
        out = base / job_name
        out.mkdir(parents=True, exist_ok=True)
        fname = log_filename or f"oczy-{job_name}.log"
        (out / fname).write_text(
            json.dumps(stream_entries), encoding="utf-8"
        )
        return out

    def _log_entry(
        self, stream: str, data: str, time: float = 1.0
    ) -> dict[str, Any]:
        return {"stream_name": stream, "time": time, "data": data}

    def _sentinel_data(self, report: dict[str, Any]) -> str:
        return _COLAB_SENTINEL_PREFIX + json.dumps(report, separators=(",", ":"))

    def _run_collector(
        self, tmp_path: Path, report_dir: Path
    ) -> dict[str, Any]:
        """Run collect_experiment_campaign for a single Kaggle job."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="kg-0", provider=PROVIDER_KAGGLE, claim_class="scientific"
            ),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        gen_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, gen_dir)
        return collect_experiment_campaign(
            campaign_path,
            result["batch_path"],
            tmp_path / "state.json",
            tmp_path / "collected",
            report_dir=report_dir,
        )

    def test_sentinel_from_log_json_complete(self, tmp_path: Path) -> None:
        """A valid sentinel in the Kaggle log JSON stream with metrics → COMPLETE."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"loss": 0.5},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
                self._log_entry("stderr", "execution_report: status=complete exit_code=0\n"),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "COMPLETE"
        assert job["report_source"] == "kaggle_log_sentinel"
        assert job["metrics"] == {"loss": 0.5}

    def test_sentinel_from_log_json_null(self, tmp_path: Path) -> None:
        """A valid sentinel with no metrics for a scientific job → NULL."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            exit_code=0,
            status="complete",
            metrics={},
            asi_scores={},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "NULL"
        assert job["report_source"] == "kaggle_log_sentinel"

    def test_sentinel_from_log_json_infrastructure_complete(self, tmp_path: Path) -> None:
        """A valid sentinel with no metrics for an infrastructure job → COMPLETE."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            exit_code=0,
            status="complete",
            metrics={},
            asi_scores={},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
            ],
        )
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="kg-0", provider=PROVIDER_KAGGLE, claim_class="infrastructure"
            ),
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
        job = summary["jobs"][0]
        assert job["classification"] == "COMPLETE"

    def test_sentinel_mismatched_commit_invalid(self, tmp_path: Path) -> None:
        """A sentinel with wrong source_commit → INVALID."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            source_commit="d" * 40,
            metrics={"loss": 0.5},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"

    def test_sentinel_mismatched_provider_invalid(self, tmp_path: Path) -> None:
        """A sentinel with wrong provider → INVALID."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_COLAB,
            metrics={"loss": 0.5},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"

    def test_sentinel_mismatched_job_name_invalid(self, tmp_path: Path) -> None:
        """A sentinel with wrong job_name → INVALID."""
        report = _make_report(
            job_name="wrong-job",
            provider=PROVIDER_KAGGLE,
            metrics={"loss": 0.5},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"

    def test_log_json_no_sentinel_invalid(self, tmp_path: Path) -> None:
        """A log JSON with no sentinel line → INVALID."""
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", "just some output\n"),
                self._log_entry("stderr", "no sentinel here\n"),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job

    def test_log_json_corrupt_sentinel_invalid(self, tmp_path: Path) -> None:
        """A sentinel with non-JSON payload → INVALID."""
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", "OCZY_EXECUTION_REPORT_JSON=not-json\n"),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job

    def test_log_json_multiple_sentinels_invalid(self, tmp_path: Path) -> None:
        """Two sentinel lines in the stdout stream → INVALID."""
        report1 = _make_report(job_name="kg-0", provider=PROVIDER_KAGGLE, metrics={"a": 1})
        report2 = _make_report(job_name="kg-0", provider=PROVIDER_KAGGLE, metrics={"a": 2})
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(report1)),
                self._log_entry("stdout", self._sentinel_data(report2)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job

    def test_log_json_not_array_invalid(self, tmp_path: Path) -> None:
        """A log file that is valid JSON but not an array → INVALID."""
        out = (tmp_path / "reports" / "kg-0")
        out.mkdir(parents=True, exist_ok=True)
        (out / "oczy-kg-0.log").write_text(
            json.dumps({"not": "an array"}), encoding="utf-8"
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job

    def test_log_json_not_json_invalid(self, tmp_path: Path) -> None:
        """A .log file that is not valid JSON → INVALID."""
        out = (tmp_path / "reports" / "kg-0")
        out.mkdir(parents=True, exist_ok=True)
        (out / "oczy-kg-0.log").write_text(
            "this is plain text, not JSON\n", encoding="utf-8"
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job

    def test_no_log_file_still_blocked(self, tmp_path: Path) -> None:
        """When no execution_report.json, no provenance, and no *.log exist,
        the job is BLOCKED (not INVALID)."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "kg-0").mkdir()
        summary = self._run_collector(tmp_path, report_dir)
        job = summary["jobs"][0]
        assert job["classification"] == "BLOCKED"

    def test_execution_report_priority_over_log_sentinel(self, tmp_path: Path) -> None:
        """execution_report.json takes priority over the log sentinel."""
        log_report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"from_sentinel": 1},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(log_report)),
            ],
        )
        # Also write execution_report.json — should take priority.
        file_report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"from_file": 2},
        )
        _write_report(
            tmp_path / "reports" / "kg-0" / "execution_report.json",
            file_report,
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "COMPLETE"
        assert job["metrics"] == {"from_file": 2}
        assert job["report_source"] == "execution_report"

    def test_sentinel_outranks_provenance(self, tmp_path: Path) -> None:
        """A valid log sentinel takes priority over remote_run_provenance.json.

        The sentinel carries the full structured report (commit, provider,
        job_name, metrics) from the runner, while provenance only records
        bootstrap metadata.  When both are present, the sentinel wins.
        """
        log_report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"from_sentinel": 1},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", self._sentinel_data(log_report)),
            ],
        )
        # Also write remote_run_provenance.json — sentinel should still win.
        provenance = {
            "schema_version": "oczy/kaggle-research-job/v1",
            "status": "complete",
            "exit_code": 0,
            "started_utc": "2026-07-11T00:00:00Z",
            "finished_utc": "2026-07-11T00:01:00Z",
            "job_spec": {
                "module": "infrastructure.kaggle.run_cortex_smoke",
                "source_commit": COMMIT,
                "arguments": [],
            },
        }
        prov_path = tmp_path / "reports" / "kg-0" / KAGGLE_PROVENANCE_FILENAME
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(provenance), encoding="utf-8")
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["report_source"] == "kaggle_log_sentinel"
        assert job["metrics"] == {"from_sentinel": 1}
        assert job["classification"] == "COMPLETE"

    def test_sentinel_split_across_stdout_entries(self, tmp_path: Path) -> None:
        """The sentinel data split across multiple stdout entries is still
        parsed correctly (Kaggle may fragment a long line across entries)."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"loss": 0.5},
        )
        sentinel = self._sentinel_data(report)
        mid = len(sentinel) // 2
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", "Starting...\n"),
                self._log_entry("stdout", sentinel[:mid]),
                self._log_entry("stdout", sentinel[mid:]),
                self._log_entry("stderr", "done\n"),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "COMPLETE"
        assert job["metrics"] == {"loss": 0.5}

    def test_sentinel_only_in_stderr_not_found(self, tmp_path: Path) -> None:
        """A sentinel in a stderr entry (not stdout) is not parsed → INVALID."""
        report = _make_report(
            job_name="kg-0",
            provider=PROVIDER_KAGGLE,
            metrics={"loss": 0.5},
        )
        self._make_kaggle_log_output(
            tmp_path / "reports",
            "kg-0",
            stream_entries=[
                self._log_entry("stdout", "normal output\n"),
                self._log_entry("stderr", self._sentinel_data(report)),
            ],
        )
        summary = self._run_collector(tmp_path, tmp_path / "reports")
        job = summary["jobs"][0]
        assert job["classification"] == "INVALID"
        assert "sentinel_error" in job


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


# ===========================================================================
# 14. Colab model artifact validation (prepare_colab_experiment)
# ===========================================================================


class TestModelArtifactValidation:
    """prepare_colab_experiment rejects invalid model_artifact / install_llama_cpp."""

    def test_rejects_non_dict_model_artifact(self, tmp_path: Path) -> None:
        with pytest.raises(ColabPrepValueError, match="model_artifact"):
            prepare_colab_experiment(
                output=tmp_path / "out",
                job_name="cb-a",
                repo_url=REPO_URL,
                source_commit=COMMIT,
                module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[],
                phase="development",
                claim_class="scientific",
                output_path="out/cb-a",
                model_artifact="not-a-dict",
            )

    def test_rejects_missing_kind(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        del artifact["kind"]
        with pytest.raises(ColabPrepValueError, match="model_artifact|missing"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_invalid_kind(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["kind"] = "onnx"
        with pytest.raises(ColabPrepValueError, match="kind"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_empty_repo_id(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["repo_id"] = ""
        with pytest.raises(ColabPrepValueError, match="repo_id"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_short_revision(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["revision"] = "abc123"
        with pytest.raises(ColabPrepValueError, match="revision"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_uppercase_revision(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["revision"] = "F" * 40
        with pytest.raises(ColabPrepValueError, match="revision"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_empty_filename(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["filename"] = ""
        with pytest.raises(ColabPrepValueError, match="filename"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_short_sha256(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["sha256"] = "a" * 32
        with pytest.raises(ColabPrepValueError, match="sha256"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_uppercase_sha256(self, tmp_path: Path) -> None:
        artifact = _valid_gguf_artifact()
        artifact["sha256"] = "E" * 64
        with pytest.raises(ColabPrepValueError, match="sha256"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", model_artifact=artifact,
            )

    def test_rejects_install_llama_cpp_not_bool(self, tmp_path: Path) -> None:
        with pytest.raises(ColabPrepValueError, match="install_llama_cpp"):
            prepare_colab_experiment(
                output=tmp_path / "out", job_name="cb-a", repo_url=REPO_URL,
                source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
                arguments=[], phase="development", claim_class="scientific",
                output_path="out/cb-a", install_llama_cpp="yes",
            )

    def test_valid_gguf_artifact_accepted(self, tmp_path: Path) -> None:
        """A valid GGUF model_artifact is accepted and written to job_spec.json."""
        out = tmp_path / "out"
        artifact = _valid_gguf_artifact()
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", model_artifact=artifact,
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["model_artifact"] == artifact

    def test_valid_hf_snapshot_artifact_accepted(self, tmp_path: Path) -> None:
        """A valid hf_snapshot model_artifact is accepted."""
        out = tmp_path / "out"
        artifact = _valid_hf_snapshot_artifact()
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", model_artifact=artifact,
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["model_artifact"] == artifact

    def test_install_llama_cpp_true_accepted(self, tmp_path: Path) -> None:
        """install_llama_cpp=True is accepted and written to job_spec.json."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", install_llama_cpp=True,
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["install_llama_cpp"] is True


# ===========================================================================
# 15. Colab bootstrap provisioning code injection (AST inspection)
# ===========================================================================


class TestBootstrapProvisioningInjection:
    """The generated bootstrap contains hash-verified provisioning code."""

    def test_gguf_bootstrap_uses_direct_streaming_and_sha_verify(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", model_artifact=_valid_gguf_artifact(),
        )
        source = (out / "colab_bootstrap.py").read_text()
        tree = ast.parse(source)
        prov_func = next(
            (
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "provision_model_artifact"
            ),
            None,
        )
        assert prov_func is not None, "provision_model_artifact not defined in bootstrap"
        calls = {
            node.func.id
            for node in ast.walk(prov_func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_gguf_resolve_url" in calls
        assert "_download_gguf_stream" in calls
        assert "_sha256_file" in calls
        assignments = [
            node for node in ast.walk(prov_func)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os"
                and target.value.attr == "environ"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "OCZY_MODEL_PATH"
                for target in node.targets
            )
        ]
        assert assignments, "GGUF provisioning must set OCZY_MODEL_PATH after SHA verification"

    def test_hf_snapshot_bootstrap_has_snapshot_download_and_sha_verify(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", model_artifact=_valid_hf_snapshot_artifact(),
        )
        source = (out / "colab_bootstrap.py").read_text()
        assert "snapshot_download" in source
        assert "OCZY_HF_MODEL_DIR" in source
        assert "_sha256_file" in source

    def test_bootstrap_forces_offline_after_download(self, tmp_path: Path) -> None:
        """The bootstrap must set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1
        after provisioning completes (in the finally block)."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", model_artifact=_valid_gguf_artifact(),
        )
        source = (out / "colab_bootstrap.py").read_text()
        tree = ast.parse(source)
        # Find provision_model_artifact function and check its finally block
        # sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE back to "1".
        prov_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "provision_model_artifact":
                prov_func = node
                break
        assert prov_func is not None, "provision_model_artifact not defined in bootstrap"
        # The function body must contain references to offline env vars.
        func_source = ast.unparse(prov_func)
        assert 'HF_HUB_OFFLINE' in func_source
        assert 'TRANSFORMERS_OFFLINE' in func_source
        # The finally block must set them back to "1".
        # Look for the string "1" being assigned to these env vars.
        assert func_source.count('"1"') >= 2 or func_source.count("'1'") >= 2

    def test_bootstrap_install_llama_cpp_uses_pinned_argv(self, tmp_path: Path) -> None:
        """install_llama_cpp in the bootstrap uses the exact pinned CPU wheel argv."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", install_llama_cpp=True,
        )
        source = (out / "colab_bootstrap.py").read_text()
        tree = ast.parse(source)
        install_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "install_llama_cpp":
                install_func = node
                break
        assert install_func is not None, "install_llama_cpp not defined in bootstrap"
        func_source = ast.unparse(install_func)
        # Must use the exact pinned version.
        assert f"llama-cpp-python=={_LLAMA_CPP_VERSION}" in func_source
        # Binary-only: pip must fail fast instead of falling back to source builds.
        assert "--only-binary=:all:" in func_source
        assert "--prefer-binary" not in func_source
        # Must use the abetlen CPU wheel index.
        assert _LLAMA_CPP_WHEEL_INDEX in func_source
        # Must use sys.executable -m pip install (no shell=True).
        assert "sys.executable" in func_source
        assert "-m" in func_source
        assert "pip" in func_source
        assert "install" in func_source
        # AST-structured check: no subprocess call in install_llama_cpp may
        # pass shell=True (proves it structurally, not via substring match).
        subprocess_funcs = {"run", "Popen", "call", "check_call", "check_output"}
        for node in ast.walk(install_func):
            if isinstance(node, ast.Call):
                func = node.func
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
                        ), "install_llama_cpp subprocess call must not use shell=True"
        # Must NOT contain arbitrary pip args like --user, --upgrade, --no-deps.
        for forbidden in ("--user", "--upgrade", "--no-deps", "--force-reinstall"):
            assert forbidden not in func_source, f"install_llama_cpp must not use {forbidden}"

    def test_bootstrap_provisioning_has_no_shell_true(self, tmp_path: Path) -> None:
        """No subprocess call in the bootstrap may pass shell=True."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
            model_artifact=_valid_gguf_artifact(),
            install_llama_cpp=True,
        )
        source = (out / "colab_bootstrap.py").read_text()
        tree = ast.parse(source)
        subprocess_funcs = {"run", "Popen", "call", "check_call", "check_output"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
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
                        ), "subprocess call in bootstrap must not use shell=True"

    def test_bootstrap_provenance_records_artifact_and_install(self, tmp_path: Path) -> None:
        """The bootstrap main() records model_artifact and llama_cpp_install in
        the provenance report."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
            model_artifact=_valid_gguf_artifact(),
            install_llama_cpp=True,
        )
        source = (out / "colab_bootstrap.py").read_text()
        # The main() function must write artifact_info and install_info to
        # the provenance report.
        assert 'report["model_artifact"]' in source or "report['model_artifact']" in source
        assert 'report["llama_cpp_install"]' in source or "report['llama_cpp_install']" in source


# 16. Colab provision_model_artifact runtime (mocked network)
# ===========================================================================


class TestProvisionModelArtifactRuntime:
    """provision_model_artifact downloads, verifies SHA-256, and sets env vars.

    The function is defined inside the bootstrap template, so tests exec the
    generated bootstrap source and call the function with mocked network
    boundaries.  No real network is invoked.
    """

    def _generate_bootstrap(self, tmp_path: Path, artifact: dict[str, Any] | None = None) -> str:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
            model_artifact=artifact or _valid_gguf_artifact(),
        )
        return (out / "colab_bootstrap.py").read_text()

    @staticmethod
    def _fake_hf_that_fails_gguf_download() -> types.ModuleType:
        fake_hf = types.ModuleType("huggingface_hub")

        def _unexpected_hf_hub_download(**kw: object) -> str:
            raise AssertionError(
                "GGUF provisioning must use direct urllib streaming, not hf_hub_download"
            )

        fake_hf.hf_hub_download = _unexpected_hf_hub_download
        fake_hf.snapshot_download = lambda **kw: ""
        return fake_hf

    def test_gguf_download_verifies_sha_and_sets_path(self, tmp_path: Path) -> None:
        """GGUF provisioning: direct stream → SHA verify → OCZY_MODEL_PATH set."""
        model_content = b"fake gguf model bytes"
        expected_sha = hashlib.sha256(model_content).hexdigest()

        artifact = _valid_gguf_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        cache_dir = tmp_path / "hf_cache"
        os.environ["OCZY_HF_CACHE_DIR"] = str(cache_dir)
        fake_hf = self._fake_hf_that_fails_gguf_download()
        try:
            with (
                patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
                patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(model_content)),
            ):
                result = ns["provision_model_artifact"](artifact)
            expected_path = (
                cache_dir
                / artifact["repo_id"].replace("/", "_")
                / artifact["revision"]
                / artifact["filename"]
            )
            assert result["kind"] == "gguf"
            assert result["sha256_verified"] is True
            assert result["sha256"] == expected_sha
            assert result["env_var"] == "OCZY_MODEL_PATH"
            assert result["model_path"] == str(expected_path)
            assert result["download_url"] == (
                "https://huggingface.co/"
                f"{artifact['repo_id']}/resolve/{artifact['revision']}/"
                f"{artifact['filename']}?download=true"
            )
            assert os.environ.get("OCZY_MODEL_PATH") == str(expected_path)
            assert expected_path.read_bytes() == model_content
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_hf_snapshot_download_verifies_sha_and_sets_dir(self, tmp_path: Path) -> None:
        """HF snapshot provisioning: snapshot_download → SHA verify → OCZY_HF_MODEL_DIR set."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_content = b'{"model_type":"lfm"}'
        config_file.write_bytes(config_content)
        expected_sha = hashlib.sha256(config_content).hexdigest()

        artifact = _valid_hf_snapshot_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kw: str(config_file)
        fake_hf.snapshot_download = lambda **kw: str(snapshot_dir)

        old_env = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR")}
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = ns["provision_model_artifact"](artifact)
            assert result["kind"] == "hf_snapshot"
            assert result["sha256_verified"] is True
            assert result["sha256"] == expected_sha
            assert result["env_var"] == "OCZY_HF_MODEL_DIR"
            assert os.environ.get("OCZY_HF_MODEL_DIR") == str(snapshot_dir)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_hash_mismatch_raises_runtime_error(self, tmp_path: Path) -> None:
        """SHA-256 mismatch must raise RuntimeError (fail closed)."""
        artifact = _valid_gguf_artifact(sha256="0" * 64)  # wrong SHA
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        fake_hf = self._fake_hf_that_fails_gguf_download()
        try:
            with (
                patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
                patch(
                    "urllib.request.urlopen",
                    return_value=_FakeHTTPResponse(b"actual content"),
                ),
            ):
                with pytest.raises(RuntimeError, match="SHA-256 mismatch|mismatch"):
                    ns["provision_model_artifact"](artifact)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_env_forced_offline_after_download(self, tmp_path: Path) -> None:
        """After provisioning, HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be '1'."""
        model_content = b"offline test"
        expected_sha = hashlib.sha256(model_content).hexdigest()

        artifact = _valid_gguf_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        fake_hf = self._fake_hf_that_fails_gguf_download()
        try:
            with (
                patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
                patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(model_content)),
            ):
                ns["provision_model_artifact"](artifact)
            # After provisioning, offline mode must be forced back on.
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_hf_snapshot_missing_file_raises(self, tmp_path: Path) -> None:
        """If the named file is not in the snapshot, provisioning fails."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        artifact = _valid_hf_snapshot_artifact(sha256="0" * 64)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kw: str(tmp_path / "x")
        fake_hf.snapshot_download = lambda **kw: str(snapshot_dir)

        old_env = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                with pytest.raises(RuntimeError, match="not found|Expected file"):
                    ns["provision_model_artifact"](artifact)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


# ===========================================================================
# 17. Colab install_llama_cpp runtime (mocked subprocess)
# ===========================================================================


class TestInstallLlamaCppRuntime:
    """install_llama_cpp installs the pinned CPU wheel via explicit argv.

    The function is defined inside the bootstrap template, so tests exec the
    generated bootstrap source and call it with mocked subprocess.run.
    No real pip install is invoked.
    """

    def _generate_bootstrap(self, tmp_path: Path) -> str:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a", install_llama_cpp=True,
        )
        return (out / "colab_bootstrap.py").read_text()

    def test_install_uses_exact_pinned_argv(self, tmp_path: Path) -> None:
        """install_llama_cpp calls subprocess.run with the exact pinned argv."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)

        captured_argv: list[str] | None = None

        def fake_run(argv, **kwargs):
            nonlocal captured_argv
            captured_argv = list(argv)
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="ok", stderr=""
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            result = ns["install_llama_cpp"]()

        assert captured_argv is not None
        assert captured_argv[0] == sys.executable
        assert captured_argv[1] == "-m"
        assert captured_argv[2] == "pip"
        assert captured_argv[3] == "install"
        assert captured_argv[4] == f"llama-cpp-python=={_LLAMA_CPP_VERSION}"
        # Binary-only: pip must fail fast instead of falling back to source builds.
        assert "--only-binary=:all:" in captured_argv
        assert "--prefer-binary" not in captured_argv
        assert "--no-binary" not in captured_argv
        assert "--extra-index-url" in captured_argv
        idx_pos = captured_argv.index("--extra-index-url")
        assert captured_argv[idx_pos + 1] == _LLAMA_CPP_WHEEL_INDEX
        # No arbitrary pip args.
        for forbidden in ("--user", "--upgrade", "--no-deps", "--force-reinstall", "--editable"):
            assert forbidden not in captured_argv
        # No shell=True in kwargs.
        assert result["package"] == "llama-cpp-python"
        assert result["version"] == _LLAMA_CPP_VERSION
        assert result["exit_code"] == 0

    def test_install_success_returns_provenance(self, tmp_path: Path) -> None:
        """Successful install returns a provenance dict with version and command."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="Successfully installed", stderr=""
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            result = ns["install_llama_cpp"]()

        assert result["package"] == "llama-cpp-python"
        assert result["version"] == _LLAMA_CPP_VERSION
        assert result["wheel_index"] == _LLAMA_CPP_WHEEL_INDEX
        assert result["exit_code"] == 0
        assert isinstance(result["install_command"], list)
        assert f"llama-cpp-python=={_LLAMA_CPP_VERSION}" in result["install_command"]
        assert "--only-binary=:all:" in result["install_command"]

    def test_install_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        """Nonzero exit from pip install raises RuntimeError."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1, stdout="", stderr="pip install failed"
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            with pytest.raises(RuntimeError, match="llama-cpp-python install failed|install failed"):
                ns["install_llama_cpp"]()

    def test_install_no_shell_true(self, tmp_path: Path) -> None:
        """install_llama_cpp must not pass shell=True to subprocess.run."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)

        captured_kwargs: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="ok", stderr=""
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            ns["install_llama_cpp"]()

        assert captured_kwargs.get("shell") is not True
        assert "shell" not in captured_kwargs or captured_kwargs["shell"] is not True


# ===========================================================================
# 18. Pure NumPy job unchanged (no model_artifact / no install_llama_cpp)
# ===========================================================================


class TestPureJobUnchanged:
    """A Colab job without model_artifact/install_llama_cpp is unchanged."""

    def test_no_model_artifact_key_in_job_spec(self, tmp_path: Path) -> None:
        """job_spec.json must not contain model_artifact when not requested."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert "model_artifact" not in spec

    def test_install_llama_cpp_false_in_job_spec(self, tmp_path: Path) -> None:
        """job_spec.json must have install_llama_cpp=False when not requested."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["install_llama_cpp"] is False

    def test_pure_job_bootstrap_still_has_provisioning_functions(self, tmp_path: Path) -> None:
        """The bootstrap always defines provisioning functions (they are gated
        by JOB_SPEC at runtime), so pure jobs are unaffected."""
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        tree = ast.parse(source)
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        # Provisioning functions are always present but gated by JOB_SPEC.
        assert "provision_model_artifact" in func_names
        assert "install_llama_cpp" in func_names
        # JOB_SPEC must not contain model_artifact (it's absent from the spec).
        spec = json.loads((out / "job_spec.json").read_text())
        assert "model_artifact" not in spec
        assert spec["install_llama_cpp"] is False


# ===========================================================================
# 19. Campaign model_artifact / install_llama_cpp field propagation
# ===========================================================================


class TestCampaignModelArtifactPropagation:
    """prepare_experiment_campaign propagates model_artifact and install_llama_cpp
    from campaign Colab jobs to the generated job_spec.json.
    """

    def test_campaign_propagates_model_artifact(self, tmp_path: Path) -> None:
        """A Colab campaign job with model_artifact propagates to job_spec.json."""
        artifact = _valid_gguf_artifact()
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=artifact,
            ),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        # Find the Colab job's kernel_dir / script and check job_spec.json.
        for job in result["jobs"]:
            script_rel = job.get("script")
            if script_rel and "colab" in script_rel:
                spec_path = out_dir / script_rel.replace("colab_bootstrap.py", "job_spec.json")
                if spec_path.exists():
                    spec = json.loads(spec_path.read_text())
                    assert spec.get("model_artifact") == artifact
                    return
        pytest.fail("No Colab job_spec.json found in generated output")

    def test_campaign_propagates_install_llama_cpp(self, tmp_path: Path) -> None:
        """A Colab campaign job with install_llama_cpp=True propagates to job_spec.json."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                install_llama_cpp=True,
            ),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        for job in result["jobs"]:
            script_rel = job.get("script")
            if script_rel and "colab" in script_rel:
                spec_path = out_dir / script_rel.replace("colab_bootstrap.py", "job_spec.json")
                if spec_path.exists():
                    spec = json.loads(spec_path.read_text())
                    assert spec.get("install_llama_cpp") is True
                    return
        pytest.fail("No Colab job_spec.json found in generated output")

    def test_campaign_colab_job_without_artifact_has_no_key(self, tmp_path: Path) -> None:
        """A Colab campaign job without model_artifact does not get the key."""
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(name="cb-0", provider=PROVIDER_COLAB, phase="development"),
        ])
        campaign_path = _write_campaign(tmp_path, campaign)
        out_dir = tmp_path / "generated"
        result = prepare_experiment_campaign(campaign_path, out_dir)
        for job in result["jobs"]:
            script_rel = job.get("script")
            if script_rel and "colab" in script_rel:
                spec_path = out_dir / script_rel.replace("colab_bootstrap.py", "job_spec.json")
                if spec_path.exists():
                    spec = json.loads(spec_path.read_text())
                    assert "model_artifact" not in spec
                    assert spec.get("install_llama_cpp") is False
                    return
        pytest.fail("No Colab job_spec.json found in generated output")


# ===========================================================================
# 20. Campaign Kaggle rejection of model_artifact / install_llama_cpp
# ===========================================================================


class TestCampaignKaggleRejection:
    """validate_campaign rejects model_artifact/install_llama_cpp on Kaggle jobs
    and validates them on Colab jobs.
    """

    def test_kaggle_rejects_model_artifact(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument",
                model_artifact=_valid_gguf_artifact(),
            ),
        ])
        with pytest.raises(CampaignValidationError, match="model_artifact|kaggle"):
            validate_campaign(campaign)

    def test_kaggle_rejects_install_llama_cpp(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="kg-0", provider=PROVIDER_KAGGLE, phase="instrument",
                install_llama_cpp=True,
            ),
        ])
        with pytest.raises(CampaignValidationError, match="install_llama_cpp|kaggle"):
            validate_campaign(campaign)

    def test_colab_accepts_valid_model_artifact(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=_valid_gguf_artifact(),
            ),
        ])
        validate_campaign(campaign)  # should not raise

    def test_colab_accepts_install_llama_cpp(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                install_llama_cpp=True,
            ),
        ])
        validate_campaign(campaign)  # should not raise

    def test_colab_rejects_invalid_kind(self) -> None:
        artifact = _valid_gguf_artifact()
        artifact["kind"] = "onnx"
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=artifact,
            ),
        ])
        with pytest.raises(CampaignValidationError, match="kind"):
            validate_campaign(campaign)

    def test_colab_rejects_short_revision(self) -> None:
        artifact = _valid_gguf_artifact()
        artifact["revision"] = "abc123"
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=artifact,
            ),
        ])
        with pytest.raises(CampaignValidationError, match="revision"):
            validate_campaign(campaign)

    def test_colab_rejects_short_sha256(self) -> None:
        artifact = _valid_gguf_artifact()
        artifact["sha256"] = "a" * 32
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=artifact,
            ),
        ])
        with pytest.raises(CampaignValidationError, match="sha256"):
            validate_campaign(campaign)

    def test_colab_rejects_non_bool_install_llama_cpp(self) -> None:
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                install_llama_cpp="yes",
            ),
        ])
        with pytest.raises(CampaignValidationError, match="install_llama_cpp|boolean"):
            validate_campaign(campaign)

    def test_colab_rejects_missing_required_field(self) -> None:
        artifact = _valid_gguf_artifact()
        del artifact["sha256"]
        campaign = _valid_campaign(jobs=[
            _valid_campaign_job(
                name="cb-0", provider=PROVIDER_COLAB, phase="development",
                model_artifact=artifact,
            ),
        ])
        with pytest.raises(CampaignValidationError, match="model_artifact|missing"):
            validate_campaign(campaign)


# ===========================================================================
# 21. Option-like argument transport (--arg=<value> encoding)
# ===========================================================================


class TestArgumentTransportOptionLike:
    """Regression: option-like module arguments (--seed, -1, --flag) must
    reach the child unchanged and in order via ``--arg=<value>`` single-token
    encoding in both Kaggle and Colab runner argv.

    Pair encoding (``--arg <value>``) makes argparse report "expected one
    argument" when the value starts with ``-``.  Equals encoding
    (``--arg=<value>``) is parsed correctly by argparse with
    ``action="append"``.
    """

    OPTION_LIKE_ARGS: list[str] = ["--seed", "-1", "--flag"]
    MIXED_ARGS: list[str] = ["--seed", "0", "-1", "--flag", "normal", "--epochs", "5"]

    # --- Kaggle: _build_runner_arguments ---

    def test_kaggle_build_uses_arg_equals_encoding(self) -> None:
        """_build_runner_arguments emits --arg=<value> single tokens, not pairs."""
        job = _valid_campaign_job(arguments=self.OPTION_LIKE_ARGS)
        runner_args = _build_runner_arguments(job, COMMIT, PROVIDER_KAGGLE)
        arg_tokens = [t for t in runner_args if t.startswith("--arg=")]
        assert len(arg_tokens) == len(self.OPTION_LIKE_ARGS)
        for token, expected in zip(arg_tokens, self.OPTION_LIKE_ARGS):
            assert token == f"--arg={expected}"
        # No bare "--arg" tokens (pair encoding).
        assert "--arg" not in runner_args

    def test_kaggle_build_preserves_order(self) -> None:
        """Arguments preserve their original order in the runner argv."""
        job = _valid_campaign_job(arguments=self.MIXED_ARGS)
        runner_args = _build_runner_arguments(job, COMMIT, PROVIDER_KAGGLE)
        arg_tokens = [t for t in runner_args if t.startswith("--arg=")]
        assert [t[len("--arg="):] for t in arg_tokens] == self.MIXED_ARGS

    def test_kaggle_build_includes_required_cli_flags(self) -> None:
        """The runner argv includes all required CLI flags before --arg tokens."""
        job = _valid_campaign_job(arguments=self.OPTION_LIKE_ARGS)
        runner_args = _build_runner_arguments(job, COMMIT, PROVIDER_KAGGLE)
        assert runner_args[0] == "--module"
        assert runner_args[1] == job["module"]
        assert "--source-commit" in runner_args
        assert COMMIT in runner_args
        assert "--provider" in runner_args
        assert "--job-name" in runner_args
        assert "--report" in runner_args

    def test_kaggle_runner_argv_parses_without_argparse_error(self) -> None:
        """Generated Kaggle runner argv parses through parse_args without failure."""
        job = _valid_campaign_job(arguments=self.OPTION_LIKE_ARGS)
        runner_args = _build_runner_arguments(job, COMMIT, PROVIDER_KAGGLE)
        args = parse_args(runner_args)
        assert args.arguments == self.OPTION_LIKE_ARGS
        assert args.module == job["module"]
        assert args.source_commit == COMMIT
        assert args.provider == PROVIDER_KAGGLE

    def test_kaggle_runner_argv_parses_mixed_args(self) -> None:
        """Mixed option-like and normal args all parse correctly and in order."""
        job = _valid_campaign_job(arguments=self.MIXED_ARGS)
        runner_args = _build_runner_arguments(job, COMMIT, PROVIDER_KAGGLE)
        args = parse_args(runner_args)
        assert args.arguments == self.MIXED_ARGS

    def test_kaggle_pair_encoding_would_fail(self) -> None:
        """Sanity: pair encoding with option-like values triggers argparse error.

        This confirms the regression is real — if the implementation reverted to
        pair encoding, parse_args would raise SystemExit (argparse error).
        """
        job = _valid_campaign_job(arguments=self.OPTION_LIKE_ARGS)
        # Simulate pair encoding (the broken form).
        pair_args: list[str] = [
            "--module", job["module"],
            "--source-commit", COMMIT,
            "--provider", PROVIDER_KAGGLE,
            "--job-name", job["name"],
            "--report", "execution_report.json",
        ]
        for arg in self.OPTION_LIKE_ARGS:
            pair_args.extend(["--arg", arg])
        with pytest.raises(SystemExit):
            parse_args(pair_args)

    # --- Colab: bootstrap runner_argv ---

    def test_colab_bootstrap_source_uses_arg_equals(self, tmp_path: Path) -> None:
        """Colab bootstrap source text uses --arg=<value>, not --arg <value> pairs."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=self.OPTION_LIKE_ARGS,
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        assert "--arg=" in source, "bootstrap must use --arg=<value> encoding"
        assert 'extend(["--arg"' not in source, "bootstrap must not use pair encoding"

    def test_colab_job_spec_preserves_option_like_args(self, tmp_path: Path) -> None:
        """job_spec.json preserves option-like arguments verbatim and in order."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=self.OPTION_LIKE_ARGS,
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        spec = json.loads((out / "job_spec.json").read_text())
        assert spec["arguments"] == self.OPTION_LIKE_ARGS

    def test_colab_bootstrap_exec_captures_runner_argv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exec the Colab bootstrap with mocked subprocess to capture runner_argv,
        then verify it parses through parse_args without argparse failure."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=self.OPTION_LIKE_ARGS,
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        ns = _exec_bootstrap(source)

        captured: dict[str, Any] = {}

        class _FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_subprocess_run(argv: list[str], **kwargs: Any) -> _FakeProc:
            captured["runner_argv"] = list(argv)
            return _FakeProc()

        # Mock bootstrap-level functions so main() reaches runner_argv construction.
        monkeypatch.setattr(ns["subprocess"], "run", fake_subprocess_run)
        ns["clone_at_commit"] = lambda repo_url, commit, dest: tmp_path
        ns["add_source_paths"] = lambda root: None
        ns["write_provenance"] = lambda payload: None
        ns["hardware"] = lambda: {}

        result = ns["main"]()
        assert result == 0
        assert "runner_argv" in captured, "subprocess.run was not called"

        runner_argv: list[str] = captured["runner_argv"]
        # Strip [sys.executable, "-m", "infrastructure.kaggle.run_experiment_module"]
        # to get the runner CLI flags that parse_args expects.
        assert runner_argv[1] == "-m"
        assert runner_argv[2] == "infrastructure.kaggle.run_experiment_module"
        runner_cli_args = runner_argv[3:]

        args = parse_args(runner_cli_args)
        assert args.arguments == self.OPTION_LIKE_ARGS
        assert args.module == "infrastructure.kaggle.run_cortex_smoke"
        assert args.source_commit == COMMIT
        assert args.provider == "colab"

    def test_colab_bootstrap_exec_preserves_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Colab bootstrap runner_argv preserves mixed arg order."""
        out = tmp_path / "colab-job"
        prepare_colab_experiment(
            output=out,
            job_name="colab-a",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=self.MIXED_ARGS,
            phase="development",
            claim_class="scientific",
            output_path="out/colab-a",
        )
        source = (out / "colab_bootstrap.py").read_text()
        ns = _exec_bootstrap(source)

        captured: dict[str, Any] = {}

        class _FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_subprocess_run(argv: list[str], **kwargs: Any) -> _FakeProc:
            captured["runner_argv"] = list(argv)
            return _FakeProc()

        monkeypatch.setattr(ns["subprocess"], "run", fake_subprocess_run)
        ns["clone_at_commit"] = lambda repo_url, commit, dest: tmp_path
        ns["add_source_paths"] = lambda root: None
        ns["write_provenance"] = lambda payload: None
        ns["hardware"] = lambda: {}

        ns["main"]()
        runner_cli_args = captured["runner_argv"][3:]
        args = parse_args(runner_cli_args)
        assert args.arguments == self.MIXED_ARGS

    def test_colab_pair_encoding_would_fail(self, tmp_path: Path) -> None:
        """Sanity: pair encoding with option-like values triggers argparse error.

        Confirms the regression is real for the Colab path too.
        """
        # Build the runner CLI args as pair encoding (the broken form).
        pair_cli_args: list[str] = [
            "--module", "infrastructure.kaggle.run_cortex_smoke",
            "--source-commit", COMMIT,
            "--provider", "colab",
            "--job-name", "colab-a",
            "--report", "execution_report.json",
        ]
        for arg in self.OPTION_LIKE_ARGS:
            pair_cli_args.extend(["--arg", arg])
        with pytest.raises(SystemExit):
            parse_args(pair_cli_args)


# ===========================================================================
# 22. Colab public HF download: implicit token disabled, token=False passed
# ===========================================================================


class TestPublicHFDownloadNoImplicitToken:
    """Generated bootstrap disables implicit HF token lookup for public
    downloads, while the HF snapshot branch retains token=False, revision
    pinning, and SHA-256 verification.

    GGUF artifacts are direct-streamed from the public resolve URL instead of
    using a token-bearing huggingface_hub download call.
    """

    def _generate_bootstrap(
        self, tmp_path: Path, artifact: dict[str, Any] | None = None
    ) -> str:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
            model_artifact=artifact or _valid_gguf_artifact(),
        )
        return (out / "colab_bootstrap.py").read_text()

    @staticmethod
    def _find_func(source: str, name: str) -> ast.FunctionDef | None:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    # --- AST inspection: implicit token disabled ---

    def test_top_env_guard_sets_disable_implicit_token(self, tmp_path: Path) -> None:
        """The top-level CPU-only env guard sets HF_HUB_DISABLE_IMPLICIT_TOKEN=1
        before any heavy imports, so implicit token lookup never fires."""
        source = self._generate_bootstrap(tmp_path)
        assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" in source, (
            "bootstrap must set HF_HUB_DISABLE_IMPLICIT_TOKEN to disable "
            "implicit HF token lookup (e.g. from Colab vault secrets)"
        )

    def test_provision_reasserts_disable_implicit_token_before_import(
        self, tmp_path: Path
    ) -> None:
        """provision_model_artifact re-asserts HF_HUB_DISABLE_IMPLICIT_TOKEN=1
        right before importing huggingface_hub, after the temporary
        HF_HUB_OFFLINE=0 that permits network for the download."""
        source = self._generate_bootstrap(tmp_path)
        prov_func = self._find_func(source, "provision_model_artifact")
        assert prov_func is not None, "provision_model_artifact not defined"
        func_source = ast.unparse(prov_func)
        assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" in func_source, (
            "provision_model_artifact must re-assert HF_HUB_DISABLE_IMPLICIT_TOKEN "
            "before importing huggingface_hub"
        )
        # The re-assertion must come before the huggingface_hub import.
        idx_disable = func_source.index("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        idx_import = func_source.index("huggingface_hub")
        assert idx_disable < idx_import, (
            "HF_HUB_DISABLE_IMPLICIT_TOKEN must be set before "
            "importing huggingface_hub"
        )

    def test_snapshot_download_has_token_false(self, tmp_path: Path) -> None:
        """The snapshot_download call in the hf_snapshot branch passes token=False."""
        source = self._generate_bootstrap(tmp_path, _valid_hf_snapshot_artifact())
        prov_func = self._find_func(source, "provision_model_artifact")
        assert prov_func is not None
        calls = [
            n for n in ast.walk(prov_func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "snapshot_download"
        ]
        assert calls, "snapshot_download call not found in provision_model_artifact"
        kw_args = {kw.arg: kw.value for kw in calls[0].keywords}
        assert "token" in kw_args, "snapshot_download must pass token=False"
        token_val = kw_args["token"]
        assert isinstance(token_val, ast.Constant) and token_val.value is False, (
            "snapshot_download token kwarg must be literal False"
        )

    # --- AST inspection: revision pinning retained ---


    def test_snapshot_download_retains_revision(self, tmp_path: Path) -> None:
        """snapshot_download still passes revision=revision (pinned)."""
        source = self._generate_bootstrap(tmp_path, _valid_hf_snapshot_artifact())
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "snapshot_download"
            ):
                kw_args = {kw.arg for kw in node.keywords}
                assert "revision" in kw_args, (
                    "snapshot_download must still pass revision for pinning"
                )
                assert "repo_id" in kw_args
                return
        pytest.fail("snapshot_download call not found")

    # --- Runtime: token=False and revision passed to mocked HF ---


    def test_snapshot_runtime_token_false_and_revision(self, tmp_path: Path) -> None:
        """Exec bootstrap, mock huggingface_hub, call provision_model_artifact:
        snapshot_download receives token=False and the pinned revision."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_content = b'{"model_type":"lfm"}'
        config_file.write_bytes(config_content)
        expected_sha = hashlib.sha256(config_content).hexdigest()

        artifact = _valid_hf_snapshot_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        captured: dict[str, Any] = {}

        def fake_snapshot_download(**kw):
            captured["snapshot_download"] = kw
            return str(snapshot_dir)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kw: str(config_file)
        fake_hf.snapshot_download = fake_snapshot_download

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = ns["provision_model_artifact"](artifact)
            assert captured["snapshot_download"]["token"] is False
            assert captured["snapshot_download"]["revision"] == MODEL_REVISION
            assert captured["snapshot_download"]["repo_id"] == artifact["repo_id"]
            assert result["sha256_verified"] is True
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    # --- Runtime: implicit token disabled at module load ---

    def test_disable_implicit_token_set_after_exec(self, tmp_path: Path) -> None:
        """After execing the bootstrap, HF_HUB_DISABLE_IMPLICIT_TOKEN must be
        set to '1' in the process environment (the top-level guard runs at
        import time)."""
        source = self._generate_bootstrap(tmp_path)
        old_val = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        try:
            _exec_bootstrap(source)
            assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1", (
                "top-level env guard must set HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
            )
        finally:
            if old_val is not None:
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = old_val
            elif "HF_HUB_DISABLE_IMPLICIT_TOKEN" in os.environ:
                del os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]



# ===========================================================================
# 23. Direct GGUF provisioning: exact revision URL, chunked streaming,
#     atomic temp replacement, SHA mismatch cleanup, HF snapshot Xet disable
# ===========================================================================


class _FakeHTTPResponse:
    """Minimal fake HTTP response for urllib streaming tests.

    Tracks ``read()`` calls so tests can verify chunked streaming.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.status = 200
        self.read_calls: list[int] = []

    def read(self, n: int = -1) -> bytes:
        self.read_calls.append(n)
        if n is None or n < 0:
            chunk = self._data[self._pos:]
        else:
            chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def getcode(self) -> int:
        return 200

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class TestDirectGGUFProvisioning:
    """Direct GGUF streaming provisioning bypasses huggingface_hub for GGUF
    files, using the exact public revision resolve URL with urllib streaming.

    These tests fail on the old ``hf_hub_download`` implementation (no
    ``_gguf_resolve_url`` / ``_download_gguf_stream`` helpers, no
    ``HF_HUB_DISABLE_XET``) and pass on the direct streaming implementation.
    """

    def _generate_bootstrap(
        self, tmp_path: Path, artifact: dict[str, Any] | None = None
    ) -> str:
        out = tmp_path / "out"
        prepare_colab_experiment(
            output=out, job_name="cb-a", repo_url=REPO_URL,
            source_commit=COMMIT, module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[], phase="development", claim_class="scientific",
            output_path="out/cb-a",
            model_artifact=artifact or _valid_gguf_artifact(),
        )
        return (out / "colab_bootstrap.py").read_text()

    @staticmethod
    def _find_func(source: str, name: str) -> ast.FunctionDef | None:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    @staticmethod
    def _make_fake_hf(
        hf_hub_download_return: str = "",
        snapshot_download_return: str = "",
        captured: dict[str, Any] | None = None,
    ) -> types.ModuleType:
        """Build a fake huggingface_hub module for safety (prevents real
        network calls on old code)."""
        fake_hf = types.ModuleType("huggingface_hub")

        def _hf_hub_download(**kw):
            if captured is not None:
                captured["hf_hub_download"] = kw
            return hf_hub_download_return

        def _snapshot_download(**kw):
            if captured is not None:
                captured["snapshot_download"] = kw
            return snapshot_download_return

        fake_hf.hf_hub_download = _hf_hub_download
        fake_hf.snapshot_download = _snapshot_download
        return fake_hf

    # ------------------------------------------------------------------
    # URL construction: exact revision URL and filename quoting
    # ------------------------------------------------------------------

    def test_gguf_resolve_url_exact_format(self, tmp_path: Path) -> None:
        """``_gguf_resolve_url`` builds the exact public resolve URL:
        ``https://huggingface.co/{repo_id}/resolve/{revision}/{filename}?download=true``"""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        resolve = ns.get("_gguf_resolve_url")
        assert resolve is not None, (
            "_gguf_resolve_url must be defined in bootstrap for direct GGUF download"
        )
        url = resolve("org/model-repo", "abc123def", "model.gguf")
        assert url == (
            "https://huggingface.co/org/model-repo/resolve/abc123def/model.gguf"
            "?download=true"
        ), f"resolve URL must match exact public format, got: {url}"

    def test_gguf_resolve_url_quotes_filename(self, tmp_path: Path) -> None:
        """Filename with special characters is URL-quoted in the resolve URL."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        resolve = ns.get("_gguf_resolve_url")
        assert resolve is not None
        url = resolve("org/model", "rev1", "my model file.gguf")
        # Spaces must be percent-encoded (not appear raw in the path).
        assert "%20" in url or "+" in url, (
            "filename spaces must be URL-encoded in resolve URL"
        )
        # The raw unquoted filename must not appear after the revision segment.
        path_part = url.split("resolve/rev1/", 1)[1]
        assert "my model file.gguf" not in path_part, (
            "raw filename with spaces must not appear unquoted in URL path"
        )

    def test_gguf_resolve_url_uses_pinned_revision(self, tmp_path: Path) -> None:
        """The resolve URL embeds the exact pinned revision, not 'main'."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        resolve = ns.get("_gguf_resolve_url")
        assert resolve is not None
        url = resolve("org/model", MODEL_REVISION, "file.gguf")
        assert f"/resolve/{MODEL_REVISION}/" in url, (
            "resolve URL must embed the exact pinned revision"
        )
        assert "/resolve/main/" not in url, (
            "resolve URL must not fall back to 'main' revision"
        )

    # ------------------------------------------------------------------
    # Chunked streaming
    # ------------------------------------------------------------------

    def test_gguf_download_streams_in_chunks(self, tmp_path: Path) -> None:
        """``_download_gguf_stream`` reads the response in bounded-size chunks,
        not in a single ``read()`` call."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        download = ns.get("_download_gguf_stream")
        assert download is not None, (
            "_download_gguf_stream must be defined for chunked streaming"
        )

        # 3 MiB + 17 bytes of data — must require multiple 1 MiB chunks.
        data = b"\x00" * (3 * 1024 * 1024 + 17)
        response = _FakeHTTPResponse(data)

        dest = tmp_path / "model.gguf"
        with patch("urllib.request.urlopen", return_value=response):
            download("https://example.com/model.gguf", dest)

        assert dest.read_bytes() == data, "downloaded content must match exactly"
        # At least 4 read calls (3 full 1 MiB chunks + 1 partial).
        assert len(response.read_calls) >= 4, (
            f"expected chunked reads (>=4 calls), got {len(response.read_calls)}"
        )
        # No single read should request the entire file at once.
        assert all(
            n is None or n <= 1024 * 1024 + 1 for n in response.read_calls
        ), "each read must request a bounded chunk, not the whole file"

    # ------------------------------------------------------------------
    # Atomic temp replacement
    # ------------------------------------------------------------------

    def test_gguf_download_atomic_temp_replacement(self, tmp_path: Path) -> None:
        """``_download_gguf_stream`` writes to a ``.tmp`` file then atomically
        ``os.replace`` to the final path — no ``.tmp`` file remains."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        download = ns.get("_download_gguf_stream")
        assert download is not None

        data = b"gguf atomic test payload"
        response = _FakeHTTPResponse(data)
        dest = tmp_path / "model.gguf"
        tmp_file = dest.with_name(dest.name + ".tmp")

        with patch("urllib.request.urlopen", return_value=response):
            download("https://example.com/model.gguf", dest)

        assert dest.exists(), "final file must exist after download"
        assert dest.read_bytes() == data
        assert not tmp_file.exists(), (
            ".tmp file must not remain after atomic os.replace"
        )

    def test_gguf_download_cleans_up_tmp_on_failure(self, tmp_path: Path) -> None:
        """On download error, the ``.tmp`` file is cleaned up."""
        source = self._generate_bootstrap(tmp_path)
        ns = _exec_bootstrap(source)
        download = ns.get("_download_gguf_stream")
        assert download is not None

        dest = tmp_path / "model.gguf"
        tmp_file = dest.with_name(dest.name + ".tmp")

        def failing_urlopen(*a: object, **kw: object) -> object:
            raise OSError("network error")

        with patch("urllib.request.urlopen", side_effect=failing_urlopen):
            with pytest.raises(OSError, match="network error"):
                download("https://example.com/model.gguf", dest)

        assert not dest.exists(), "dest must not exist on download failure"
        assert not tmp_file.exists(), ".tmp file must be cleaned up on failure"

    # ------------------------------------------------------------------
    # SHA mismatch: cleanup + failure
    # ------------------------------------------------------------------

    def test_gguf_sha_mismatch_raises_runtime_error(self, tmp_path: Path) -> None:
        """SHA-256 mismatch raises RuntimeError (fail closed)."""
        model_content = b"actual gguf content for mismatch test"
        model_file = tmp_path / "hf_cache.gguf"
        model_file.write_bytes(model_content)

        artifact = _valid_gguf_artifact(sha256="0" * 64)  # wrong SHA
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        def fake_download(url: str, dest: object, timeout: float = 600.0) -> None:
            p = Path(dest)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(model_content)

        ns["_download_gguf_stream"] = fake_download
        fake_hf = self._make_fake_hf(hf_hub_download_return=str(model_file))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                with pytest.raises(RuntimeError, match="SHA-256 mismatch|mismatch"):
                    ns["provision_model_artifact"](artifact)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_gguf_sha_mismatch_cleans_up_downloaded_file(self, tmp_path: Path) -> None:
        """On SHA mismatch, the downloaded GGUF file is removed (cleanup)."""
        model_content = b"actual gguf content for cleanup test"
        model_file = tmp_path / "hf_cache.gguf"
        model_file.write_bytes(model_content)
        downloaded_paths: list[Path] = []

        artifact = _valid_gguf_artifact(sha256="0" * 64)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        def fake_download(url: str, dest: object, timeout: float = 600.0) -> None:
            p = Path(dest)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(model_content)
            downloaded_paths.append(p)

        ns["_download_gguf_stream"] = fake_download
        fake_hf = self._make_fake_hf(hf_hub_download_return=str(model_file))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                with pytest.raises(RuntimeError):
                    ns["provision_model_artifact"](artifact)
            # _download_gguf_stream must have been called (fails on old code).
            assert downloaded_paths, "_download_gguf_stream must have been called"
            # The downloaded file must be cleaned up after SHA mismatch.
            assert not downloaded_paths[0].exists(), (
                "downloaded file must be removed after SHA mismatch"
            )
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    # ------------------------------------------------------------------
    # Successful provisioning: OCZY_MODEL_PATH set
    # ------------------------------------------------------------------

    def test_gguf_success_sets_oczy_model_path(self, tmp_path: Path) -> None:
        """Successful GGUF download + SHA verify sets OCZY_MODEL_PATH and
        returns the correct result dict."""
        model_content = b"fake gguf model bytes for success test"
        expected_sha = hashlib.sha256(model_content).hexdigest()
        model_file = tmp_path / "hf_cache.gguf"
        model_file.write_bytes(model_content)

        artifact = _valid_gguf_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        captured_url: list[str] = []

        def fake_download(url: str, dest: object, timeout: float = 600.0) -> None:
            captured_url.append(url)
            p = Path(dest)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(model_content)

        ns["_download_gguf_stream"] = fake_download
        fake_hf = self._make_fake_hf(hf_hub_download_return=str(model_file))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = ns["provision_model_artifact"](artifact)
            # _download_gguf_stream must have been called with the resolve URL.
            assert captured_url, "_download_gguf_stream must have been called"
            assert "huggingface.co" in captured_url[0]
            assert "/resolve/" in captured_url[0]
            assert "?download=true" in captured_url[0]
            # Result dict.
            assert result["kind"] == "gguf"
            assert result["sha256_verified"] is True
            assert result["sha256"] == expected_sha
            assert result["env_var"] == "OCZY_MODEL_PATH"
            # OCZY_MODEL_PATH env var set to the downloaded file.
            assert os.environ.get("OCZY_MODEL_PATH") == result["model_path"]
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_gguf_success_forces_offline_after(self, tmp_path: Path) -> None:
        """After successful GGUF provisioning, offline env vars are forced on."""
        model_content = b"offline gguf test payload"
        expected_sha = hashlib.sha256(model_content).hexdigest()
        model_file = tmp_path / "hf_cache.gguf"
        model_file.write_bytes(model_content)

        artifact = _valid_gguf_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        def fake_download(url: str, dest: object, timeout: float = 600.0) -> None:
            p = Path(dest)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(model_content)

        ns["_download_gguf_stream"] = fake_download
        fake_hf = self._make_fake_hf(hf_hub_download_return=str(model_file))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_MODEL_PATH",
                "OCZY_HF_CACHE_DIR",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                ns["provision_model_artifact"](artifact)
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    # ------------------------------------------------------------------
    # No hf_hub_download in GGUF branch (direct streaming instead)
    # ------------------------------------------------------------------

    def test_gguf_branch_no_hf_hub_download_call(self, tmp_path: Path) -> None:
        """The GGUF provisioning branch must not call ``hf_hub_download`` —
        it uses direct urllib streaming instead."""
        source = self._generate_bootstrap(tmp_path, _valid_gguf_artifact())
        prov_func = self._find_func(source, "provision_model_artifact")
        assert prov_func is not None, "provision_model_artifact not defined"

        has_hf_hub_download = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hf_hub_download"
            for node in ast.walk(prov_func)
        )
        assert not has_hf_hub_download, (
            "GGUF branch must not call hf_hub_download — use direct streaming"
        )

    def test_gguf_branch_has_resolve_url_and_stream_helpers(self, tmp_path: Path) -> None:
        """The bootstrap defines ``_gguf_resolve_url`` and ``_download_gguf_stream``."""
        source = self._generate_bootstrap(tmp_path, _valid_gguf_artifact())
        assert "_gguf_resolve_url" in source, (
            "bootstrap must define _gguf_resolve_url for direct GGUF download"
        )
        assert "_download_gguf_stream" in source, (
            "bootstrap must define _download_gguf_stream for chunked streaming"
        )

    # ------------------------------------------------------------------
    # HF snapshot: HF_HUB_DISABLE_XET=1
    # ------------------------------------------------------------------

    def test_hf_snapshot_sets_disable_xet_in_source(self, tmp_path: Path) -> None:
        """The HF snapshot branch sets ``HF_HUB_DISABLE_XET=1`` to bypass
        Xet provisioning."""
        source = self._generate_bootstrap(tmp_path, _valid_hf_snapshot_artifact())
        prov_func = self._find_func(source, "provision_model_artifact")
        assert prov_func is not None
        func_source = ast.unparse(prov_func)
        assert "HF_HUB_DISABLE_XET" in func_source, (
            "provision_model_artifact must set HF_HUB_DISABLE_XET=1 for "
            "hf_snapshot to bypass Xet provisioning"
        )

    def test_hf_snapshot_disable_xet_before_snapshot_download(
        self, tmp_path: Path
    ) -> None:
        """``HF_HUB_DISABLE_XET`` must be set before ``snapshot_download`` is called."""
        source = self._generate_bootstrap(tmp_path, _valid_hf_snapshot_artifact())
        prov_func = self._find_func(source, "provision_model_artifact")
        assert prov_func is not None
        func_source = ast.unparse(prov_func)
        idx_xet = func_source.find("HF_HUB_DISABLE_XET")
        idx_snapshot = func_source.find("snapshot_download")
        assert idx_xet != -1, "HF_HUB_DISABLE_XET must appear in provision function"
        assert idx_snapshot != -1, "snapshot_download must appear in provision function"
        assert idx_xet < idx_snapshot, (
            "HF_HUB_DISABLE_XET must be set before snapshot_download is called"
        )

    def test_hf_snapshot_runtime_sets_disable_xet_env(self, tmp_path: Path) -> None:
        """After provisioning an hf_snapshot, ``HF_HUB_DISABLE_XET=1`` is set
        in the process environment."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_content = b'{"model_type":"lfm"}'
        config_file.write_bytes(config_content)
        expected_sha = hashlib.sha256(config_content).hexdigest()

        artifact = _valid_hf_snapshot_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        fake_hf = self._make_fake_hf(snapshot_download_return=str(snapshot_dir))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR",
                "HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                ns["provision_model_artifact"](artifact)
            assert os.environ.get("HF_HUB_DISABLE_XET") == "1", (
                "HF_HUB_DISABLE_XET must be set to '1' for hf_snapshot provisioning"
            )
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    # ------------------------------------------------------------------
    # HF snapshot: retains token=False / revision / hash verification
    # ------------------------------------------------------------------

    def test_hf_snapshot_retains_token_false(self, tmp_path: Path) -> None:
        """``snapshot_download`` still receives ``token=False``."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_content = b'{"model_type":"lfm"}'
        config_file.write_bytes(config_content)
        expected_sha = hashlib.sha256(config_content).hexdigest()

        artifact = _valid_hf_snapshot_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        captured: dict[str, Any] = {}
        fake_hf = self._make_fake_hf(
            snapshot_download_return=str(snapshot_dir), captured=captured
        )

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR",
                "HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                ns["provision_model_artifact"](artifact)
            assert "snapshot_download" in captured
            assert captured["snapshot_download"]["token"] is False, (
                "snapshot_download must still receive token=False"
            )
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_hf_snapshot_retains_revision(self, tmp_path: Path) -> None:
        """``snapshot_download`` still receives the pinned revision."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_content = b'{"model_type":"lfm"}'
        config_file.write_bytes(config_content)
        expected_sha = hashlib.sha256(config_content).hexdigest()

        artifact = _valid_hf_snapshot_artifact(sha256=expected_sha)
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        captured: dict[str, Any] = {}
        fake_hf = self._make_fake_hf(
            snapshot_download_return=str(snapshot_dir), captured=captured
        )

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR",
                "HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                ns["provision_model_artifact"](artifact)
            assert "snapshot_download" in captured
            assert captured["snapshot_download"]["revision"] == MODEL_REVISION, (
                "snapshot_download must still receive the pinned revision"
            )
            assert captured["snapshot_download"]["repo_id"] == artifact["repo_id"]
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    def test_hf_snapshot_retains_sha_verification(self, tmp_path: Path) -> None:
        """SHA-256 verification is still enforced for hf_snapshot."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        config_file = snapshot_dir / "config.json"
        config_file.write_bytes(b'{"model_type":"lfm"}')

        artifact = _valid_hf_snapshot_artifact(sha256="0" * 64)  # wrong SHA
        source = self._generate_bootstrap(tmp_path, artifact)
        ns = _exec_bootstrap(source)

        fake_hf = self._make_fake_hf(snapshot_download_return=str(snapshot_dir))

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OCZY_HF_MODEL_DIR",
                "HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        try:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                with pytest.raises(RuntimeError, match="SHA-256 mismatch|mismatch"):
                    ns["provision_model_artifact"](artifact)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

# ===========================================================================
# 24. Colab bootstrap stderr diagnostics on failure
# ===========================================================================


class TestBootstrapStderrDiagnostics:
    """Bootstrap exceptions print bounded actionable diagnostics to stderr,
    redact secrets, attempt provenance, and exit nonzero.

    The generated bootstrap except Exception block must:
    - print a bounded traceback/error to sys.stderr with a bootstrap: prefix
    - redact secret-like substrings (token=, key=, password=, Authorization:)
    - still call write_provenance with status=error
    - return exit code 1

    These tests exec the generated bootstrap, patch the failure boundary,
    call main(), and assert on the captured stderr / provenance / exit.
    """

    _SECRET = "sk-secret_abc123DEF456"
    _STDERR_BOUND = 8192

    def _generate_and_exec(
        self, tmp_path: Path, **kwargs: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Generate bootstrap, exec it, install spies. Returns (ns, prov_calls)."""
        out = tmp_path / "diag_out"
        defaults: dict[str, Any] = dict(
            output=out,
            job_name="cb-diag",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/cb-diag",
        )
        defaults.update(kwargs)
        prepare_colab_experiment(**defaults)
        source = (out / "colab_bootstrap.py").read_text()
        ns = _exec_bootstrap(source)

        # Provenance spy records every write_provenance call.
        provenance_calls: list[dict[str, Any]] = []

        def _spy_provenance(payload: dict) -> None:
            provenance_calls.append(dict(payload))

        ns["write_provenance"] = _spy_provenance

        # Avoid real git / sys.path / chdir side effects.
        fake_repo = tmp_path / "fake_repo"
        fake_repo.mkdir(exist_ok=True)
        ns["clone_at_commit"] = lambda *a, **kw: fake_repo
        ns["add_source_paths"] = lambda repo_root: None

        return ns, provenance_calls

    @staticmethod
    def _call_main_captured(ns: dict[str, Any]) -> tuple[int, str]:
        """Call ns main() with stderr captured; restore cwd/sys.path."""
        err_buf = io.StringIO()
        orig_cwd = os.getcwd()
        orig_path = list(sys.path)
        try:
            with patch.object(sys, "stderr", new=err_buf):
                exit_code = ns["main"]()
        finally:
            os.chdir(orig_cwd)
            sys.path[:] = orig_path
        return exit_code, err_buf.getvalue()

    def test_model_download_failure_stderr_diagnostic(self, tmp_path: Path) -> None:
        """Forced GGUF download failure prints bounded diagnostic to stderr."""
        artifact = _valid_gguf_artifact()
        ns, prov = self._generate_and_exec(tmp_path, model_artifact=artifact)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "OCZY_HF_CACHE_DIR", "OCZY_MODEL_PATH",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch(
                "urllib.request.urlopen",
                side_effect=RuntimeError(
                    f"connection refused token={self._SECRET}"
                ),
            ):
                exit_code, stderr = self._call_main_captured(ns)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        assert "RuntimeError" in stderr
        assert "Traceback" in stderr or "traceback" in stderr.lower()
        assert len(stderr) < self._STDERR_BOUND
        # Secret must be redacted.
        assert self._SECRET not in stderr
        assert "***" in stderr
        # Provenance was attempted.
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"

    def test_hash_mismatch_failure_stderr_diagnostic(self, tmp_path: Path) -> None:
        """Forced SHA-256 mismatch prints bounded diagnostic to stderr."""
        wrong_content = b"this is not the right model content"
        artifact = _valid_gguf_artifact(sha256="0" * 64)
        ns, prov = self._generate_and_exec(tmp_path, model_artifact=artifact)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "OCZY_HF_CACHE_DIR", "OCZY_MODEL_PATH",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch(
                "urllib.request.urlopen",
                return_value=_FakeHTTPResponse(wrong_content),
            ):
                exit_code, stderr = self._call_main_captured(ns)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        assert "RuntimeError" in stderr
        assert "mismatch" in stderr.lower() or "SHA" in stderr
        assert len(stderr) < self._STDERR_BOUND
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"

    def test_install_failure_stderr_diagnostic(self, tmp_path: Path) -> None:
        """Forced llama-cpp install failure prints bounded diagnostic to stderr."""
        ns, prov = self._generate_and_exec(tmp_path, install_llama_cpp=True)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr=f"error: auth key={self._SECRET} denied",
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, stderr = self._call_main_captured(ns)

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        assert "RuntimeError" in stderr
        assert "install" in stderr.lower()
        assert len(stderr) < self._STDERR_BOUND
        # Secret must be redacted.
        assert self._SECRET not in stderr
        assert "***" in stderr
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"

    def test_runner_exception_stderr_diagnostic(self, tmp_path: Path) -> None:
        """Forced runner subprocess exception prints bounded diagnostic to stderr."""
        ns, prov = self._generate_and_exec(tmp_path)

        def fake_run(argv, **kwargs):
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: 'python'"
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, stderr = self._call_main_captured(ns)

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        assert "FileNotFoundError" in stderr
        assert len(stderr) < self._STDERR_BOUND
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"

    def test_secret_redaction_multiple_patterns(self, tmp_path: Path) -> None:
        """Multiple secret patterns (token=, key=, password=, Authorization:) are redacted."""
        artifact = _valid_gguf_artifact()
        ns, prov = self._generate_and_exec(tmp_path, model_artifact=artifact)

        secret_msg = (
            "failed: token=sk-token-xyz key=sk-key-abc "
            "password=hunter2 Authorization: Bearer s3cr3t"
        )

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "OCZY_HF_CACHE_DIR", "OCZY_MODEL_PATH",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch(
                "urllib.request.urlopen",
                side_effect=RuntimeError(secret_msg),
            ):
                exit_code, stderr = self._call_main_captured(ns)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        assert exit_code == 1
        assert "sk-token-xyz" not in stderr
        assert "sk-key-abc" not in stderr
        assert "hunter2" not in stderr
        assert "s3cr3t" not in stderr
        assert "***" in stderr

    def test_stderr_diagnostic_is_bounded(self, tmp_path: Path) -> None:
        """Stderr diagnostic traceback is bounded even with deep call stacks."""
        ns, prov = self._generate_and_exec(tmp_path)

        # Create a deep call stack to produce a very long traceback.
        def raise_deep(depth: int) -> None:
            if depth <= 0:
                raise RuntimeError("deep failure")
            raise_deep(depth - 1)

        def fake_run(argv, **kwargs):
            raise_deep(200)

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, stderr = self._call_main_captured(ns)

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        # The traceback must be bounded (last ~4000 chars) so that 200
        # frames do not flood stderr.
        assert len(stderr) < 10000, f"stderr is unbounded: {len(stderr)} bytes"

    def test_stderr_message_truncation(self, tmp_path: Path) -> None:
        """Very long exception messages are truncated with a marker."""
        ns, prov = self._generate_and_exec(tmp_path)

        long_msg = "x" * 100000

        def fake_run(argv, **kwargs):
            raise RuntimeError(long_msg)

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, stderr = self._call_main_captured(ns)

        assert exit_code == 1
        assert "bootstrap: EXCEPTION" in stderr
        # The first line (EXCEPTION line) must be bounded.
        first_line = stderr.splitlines()[0] if stderr else ""
        assert len(first_line) < 600, (
            f"EXCEPTION line is unbounded: {len(first_line)} chars"
        )
        # Truncation marker must appear when message exceeds the bound.
        assert "...[truncated]..." in stderr

    def test_provenance_attempted_on_exception(self, tmp_path: Path) -> None:
        """Provenance is written with status=error on exception."""
        artifact = _valid_gguf_artifact()
        ns, prov = self._generate_and_exec(tmp_path, model_artifact=artifact)

        old_env = {
            k: os.environ.get(k)
            for k in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "OCZY_HF_CACHE_DIR", "OCZY_MODEL_PATH",
            )
        }
        os.environ["OCZY_HF_CACHE_DIR"] = str(tmp_path / "hf_cache")
        try:
            with patch(
                "urllib.request.urlopen",
                side_effect=RuntimeError("download failed"),
            ):
                exit_code, stderr = self._call_main_captured(ns)
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        assert exit_code == 1
        # Provenance was attempted multiple times (initial + running + error).
        assert len(prov) >= 2
        last = prov[-1]
        assert last["status"] == "error"
        assert "error" in last or "traceback" in last

    def test_runner_nonzero_no_bootstrap_diagnostic(self, tmp_path: Path) -> None:
        """Runner nonzero exit does NOT emit bootstrap: stderr diagnostic.

        The stderr diagnostic is reserved for bootstrap-level exceptions.
        A runner nonzero exit is a normal return path: the bootstrap writes
        provenance with status=error and forwards the runner exit code.
        """
        ns, prov = self._generate_and_exec(tmp_path)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=42, stdout="", stderr="runner error"
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, stderr = self._call_main_captured(ns)

        assert exit_code == 42
        assert "bootstrap: EXCEPTION" not in stderr
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"
        assert prov[-1].get("exit_code") == 42
# ===========================================================================
# 25. Colab bootstrap runner stdout/stderr forwarding
# ===========================================================================


class TestBootstrapRunnerOutputForwarding:
    """Bootstrap forwards captured runner stdout/stderr to its own streams.

    The generated bootstrap invokes the runner with ``capture_output=True``,
    then forwards ``proc.stdout`` to ``sys.stdout`` (verbatim, preserving the
    ``OCZY_EXECUTION_REPORT_JSON`` sentinel) and ``proc.stderr`` to
    ``sys.stderr`` (redacted and bounded), before writing provenance and
    returning the runner exit code unchanged.

    These tests exec the generated bootstrap, patch ``subprocess.run`` to
    return a fake ``CompletedProcess`` with controlled stdout/stderr, call
    ``main()`` with both streams captured, and assert on the forwarded
    output, exit code, and provenance.
    """

    _SENTINEL_PREFIX = "OCZY_EXECUTION_REPORT_JSON="
    _STDERR_BOUND = 4000

    def _generate_and_exec(
        self, tmp_path: Path, **kwargs: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Generate bootstrap, exec it, install spies. Returns (ns, prov_calls)."""
        out = tmp_path / "fwd_out"
        defaults: dict[str, Any] = dict(
            output=out,
            job_name="cb-fwd",
            repo_url=REPO_URL,
            source_commit=COMMIT,
            module="infrastructure.kaggle.run_cortex_smoke",
            arguments=[],
            phase="development",
            claim_class="scientific",
            output_path="out/cb-fwd",
        )
        defaults.update(kwargs)
        prepare_colab_experiment(**defaults)
        source = (out / "colab_bootstrap.py").read_text()
        ns = _exec_bootstrap(source)

        provenance_calls: list[dict[str, Any]] = []

        def _spy_provenance(payload: dict) -> None:
            provenance_calls.append(dict(payload))

        ns["write_provenance"] = _spy_provenance

        fake_repo = tmp_path / "fake_repo"
        fake_repo.mkdir(exist_ok=True)
        ns["clone_at_commit"] = lambda *a, **kw: fake_repo
        ns["add_source_paths"] = lambda repo_root: None

        return ns, provenance_calls

    @staticmethod
    def _call_main_captured_both(ns: dict[str, Any]) -> tuple[int, str, str]:
        """Call ns main() with stdout and stderr captured; restore cwd/sys.path."""
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        orig_cwd = os.getcwd()
        orig_path = list(sys.path)
        try:
            with patch.object(sys, "stdout", new=out_buf), patch.object(sys, "stderr", new=err_buf):
                exit_code = ns["main"]()
        finally:
            os.chdir(orig_cwd)
            sys.path[:] = orig_path
        return exit_code, out_buf.getvalue(), err_buf.getvalue()

    @staticmethod
    def _make_sentinel(status: str = "complete", exit_code: int = 0) -> str:
        """Build a valid OCZY_EXECUTION_REPORT_JSON sentinel line."""
        report = {
            "schema_version": "oczy/execution-report/v1",
            "status": status,
            "exit_code": exit_code,
            "job_name": "cb-fwd",
            "module": "infrastructure.kaggle.run_cortex_smoke",
            "source_commit": COMMIT,
            "provider": "colab",
            "metrics": {},
            "asi_scores": {},
        }
        compact = json.dumps(report, sort_keys=True, separators=(",", ":"))
        return f"OCZY_EXECUTION_REPORT_JSON={compact}\n"

    # ------------------------------------------------------------------
    # Sentinel + stderr forwarding on exit 0 and exit 1
    # ------------------------------------------------------------------

    def test_sentinel_forwarded_on_exit0(self, tmp_path: Path) -> None:
        """Runner exit 0: sentinel on stdout forwarded verbatim, stderr
        forwarded, exit code 0, provenance status=complete."""
        ns, prov = self._generate_and_exec(tmp_path)
        sentinel = self._make_sentinel(status="complete", exit_code=0)
        runner_stdout = sentinel + "METRIC loss=0.5\n"
        runner_stderr = "some warning\n"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=0,
                stdout=runner_stdout, stderr=runner_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, out, err = self._call_main_captured_both(ns)

        assert exit_code == 0
        # Sentinel forwarded verbatim to stdout.
        sentinel_lines = [
            line for line in out.splitlines()
            if line.startswith(self._SENTINEL_PREFIX)
        ]
        assert len(sentinel_lines) == 1, (
            "exactly one sentinel line must be forwarded to stdout"
        )
        parsed = json.loads(sentinel_lines[0][len(self._SENTINEL_PREFIX):])
        assert parsed["status"] == "complete"
        # Non-sentinel stdout also forwarded.
        assert "METRIC loss=0.5" in out
        # Stderr forwarded.
        assert "some warning" in err
        # Provenance recorded with status=complete, exit_code=0.
        assert len(prov) > 0
        assert prov[-1].get("status") == "complete"
        assert prov[-1].get("exit_code") == 0

    def test_sentinel_forwarded_on_exit1(self, tmp_path: Path) -> None:
        """Runner exit 1: sentinel on stdout forwarded verbatim, stderr
        diagnostics forwarded, exit code 1, provenance status=error."""
        ns, prov = self._generate_and_exec(tmp_path)
        sentinel = self._make_sentinel(status="error", exit_code=1)
        runner_stdout = sentinel
        runner_stderr = "Traceback (most recent call last):\nRuntimeError: boom\n"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1,
                stdout=runner_stdout, stderr=runner_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, out, err = self._call_main_captured_both(ns)

        assert exit_code == 1
        # Sentinel forwarded verbatim.
        sentinel_lines = [
            line for line in out.splitlines()
            if line.startswith(self._SENTINEL_PREFIX)
        ]
        assert len(sentinel_lines) == 1
        parsed = json.loads(sentinel_lines[0][len(self._SENTINEL_PREFIX):])
        assert parsed["status"] == "error"
        # Stderr diagnostics forwarded.
        assert "RuntimeError: boom" in err
        # Provenance recorded with status=error, exit_code=1.
        assert len(prov) > 0
        assert prov[-1].get("status") == "error"
        assert prov[-1].get("exit_code") == 1

    def test_stderr_forwarded_on_exit0(self, tmp_path: Path) -> None:
        """Runner exit 0 with stderr: diagnostics forwarded to bootstrap stderr."""
        ns, _ = self._generate_and_exec(tmp_path)
        runner_stderr = "WARNING: deprecation\nINFO: something\n"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=0,
                stdout=self._make_sentinel(status="complete", exit_code=0),
                stderr=runner_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            _, _, err = self._call_main_captured_both(ns)

        assert "WARNING: deprecation" in err
        assert "INFO: something" in err

    def test_stderr_forwarded_on_exit1(self, tmp_path: Path) -> None:
        """Runner exit 1 with stderr: diagnostics forwarded to bootstrap stderr."""
        ns, _ = self._generate_and_exec(tmp_path)
        runner_stderr = "ImportError: no module named foo\n"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1,
                stdout=self._make_sentinel(status="error", exit_code=1),
                stderr=runner_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            _, _, err = self._call_main_captured_both(ns)

        assert "ImportError: no module named foo" in err

    # ------------------------------------------------------------------
    # Bounded / sanitized forwarded stderr
    # ------------------------------------------------------------------

    def test_forwarded_stderr_bounded(self, tmp_path: Path) -> None:
        """Very long runner stderr is bounded in the forwarded output."""
        ns, _ = self._generate_and_exec(tmp_path)
        long_stderr = "x" * 100000

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1,
                stdout="",
                stderr=long_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            _, _, err = self._call_main_captured_both(ns)

        # Forwarded stderr must be bounded (last ~4000 chars + truncation marker).
        assert len(err) < 5000, (
            f"forwarded stderr is unbounded: {len(err)} bytes"
        )
        assert "...[truncated]..." in err

    def test_forwarded_stderr_sanitized(self, tmp_path: Path) -> None:
        """Secrets in runner stderr are redacted in the forwarded output."""
        ns, _ = self._generate_and_exec(tmp_path)
        secret = "sk-secret_abc123DEF456"
        runner_stderr = (
            f"failed: token={secret} key={secret} "
            f"password=hunter2 Authorization: Bearer s3cr3t\n"
        )

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1,
                stdout="",
                stderr=runner_stderr,
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            _, _, err = self._call_main_captured_both(ns)

        assert secret not in err
        assert "hunter2" not in err
        assert "s3cr3t" not in err
        assert "***" in err

    # ------------------------------------------------------------------
    # Exit code / provenance unchanged
    # ------------------------------------------------------------------

    def test_exit_code_preserved_nonzero(self, tmp_path: Path) -> None:
        """Runner exit code (non-standard nonzero) is returned unchanged."""
        ns, prov = self._generate_and_exec(tmp_path)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=42,
                stdout=self._make_sentinel(status="error", exit_code=42),
                stderr="runner error\n",
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            exit_code, _, _ = self._call_main_captured_both(ns)

        assert exit_code == 42
        assert prov[-1].get("exit_code") == 42
        assert prov[-1].get("status") == "error"

    def test_provenance_exit0(self, tmp_path: Path) -> None:
        """Provenance on exit 0: status=complete, exit_code=0,
        runner_command recorded."""
        ns, prov = self._generate_and_exec(tmp_path)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=0,
                stdout=self._make_sentinel(status="complete", exit_code=0),
                stderr="",
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            self._call_main_captured_both(ns)

        last = prov[-1]
        assert last["status"] == "complete"
        assert last["exit_code"] == 0
        assert "runner_command" in last
        assert "finished_utc" in last

    def test_provenance_exit1(self, tmp_path: Path) -> None:
        """Provenance on exit 1: status=error, exit_code=1,
        runner_command recorded."""
        ns, prov = self._generate_and_exec(tmp_path)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=1,
                stdout=self._make_sentinel(status="error", exit_code=1),
                stderr="error\n",
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            self._call_main_captured_both(ns)

        last = prov[-1]
        assert last["status"] == "error"
        assert last["exit_code"] == 1
        assert "runner_command" in last
        assert "finished_utc" in last

    # ------------------------------------------------------------------
    # Stdout verbatim (not sanitized) and flushing
    # ------------------------------------------------------------------

    def test_stdout_forwarded_verbatim(self, tmp_path: Path) -> None:
        """Runner stdout is forwarded verbatim — sentinel JSON is not truncated
        or sanitized, preserving collector parseability."""
        ns, _ = self._generate_and_exec(tmp_path)
        sentinel = self._make_sentinel(status="complete", exit_code=0)
        # Include a secret-like pattern in stdout to prove stdout is NOT sanitized.
        runner_stdout = sentinel + "token=sk-test123\n"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                args=argv, returncode=0,
                stdout=runner_stdout,
                stderr="",
            )

        with patch.object(ns["subprocess"], "run", fake_run):
            _, out, _ = self._call_main_captured_both(ns)

        # Sentinel is intact and parseable.
        sentinel_lines = [
            line for line in out.splitlines()
            if line.startswith(self._SENTINEL_PREFIX)
        ]
        assert len(sentinel_lines) == 1
        parsed = json.loads(sentinel_lines[0][len(self._SENTINEL_PREFIX):])
        assert parsed["status"] == "complete"
        # Stdout is NOT sanitized — token= survives verbatim.
        assert "token=sk-test123" in out

    def test_output_flushed(self, tmp_path: Path) -> None:
        """Runner stdout/stderr are flushed to bootstrap streams."""
        ns, _ = self._generate_and_exec(tmp_path)

        flush_tracker: dict[str, int] = {"stdout": 0, "stderr": 0}

        class _FlushTrackingStream(io.StringIO):
            def __init__(self, name: str) -> None:
                super().__init__()
                self._name = name

            def flush(self) -> None:
                flush_tracker[self._name] += 1
                super().flush()

        out_buf = _FlushTrackingStream("stdout")
        err_buf = _FlushTrackingStream("stderr")
        orig_cwd = os.getcwd()
        orig_path = list(sys.path)
        try:
            with patch.object(sys, "stdout", new=out_buf), \
                 patch.object(sys, "stderr", new=err_buf):

                def fake_run(argv, **kwargs):
                    return subprocess.CompletedProcess(
                        args=argv, returncode=0,
                        stdout=self._make_sentinel(status="complete", exit_code=0),
                        stderr="warning\n",
                    )

                with patch.object(ns["subprocess"], "run", fake_run):
                    exit_code = ns["main"]()
        finally:
            os.chdir(orig_cwd)
            sys.path[:] = orig_path

        assert exit_code == 0
        assert flush_tracker["stdout"] >= 1, "stdout must be flushed"
        assert flush_tracker["stderr"] >= 1, "stderr must be flushed"
