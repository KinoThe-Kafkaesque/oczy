"""Collect and classify results from a mixed Kaggle/Colab experiment campaign.

Reads the scheduler durable state plus each job's output directory, parses the
structured execution report (``execution_report.json`` for Kaggle; for Colab
parses the ``OCZY_EXECUTION_REPORT_JSON=<compact-json>`` sentinel from
``stdout.log`` since the provider does not download the report file, with
``result.json`` as a provider-level fallback), validates source commit /
provider / job identity / exit code, and classifies each job as ``COMPLETE``,
``NULL``, ``INVALID``, or ``BLOCKED``.

Classification rules (no thresholds are changed):

* **COMPLETE** — exit_code == 0, provenance valid (source_commit, provider,
  job_name all match the campaign), report parseable, status == "complete".
* **NULL** — exit_code == 0, provenance valid, status == "complete",
  claim_class == "scientific", and no metrics/asi_signals found.
  Infrastructure jobs are **never** NULL.
* **INVALID** — missing or corrupt provenance, source_commit/provider/job
  identity mismatch, or corrupt/unparseable report.  Missing provenance is
  INVALID, not NULL.
* **BLOCKED** — exit_code != 0, status != "complete", or report file missing
  entirely.  A failed infrastructure job is BLOCKED, never a scientific NULL.

Writes ``campaign_execution_summary.json`` to the output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- Co-located imports (same directory) -----------------------------------
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from parallel_scheduler import (  # noqa: E402  # type: ignore[import-not-found]
    PROVIDER_COLAB,
    PROVIDER_KAGGLE,
    SUCCEEDED,
    load_batch,
    load_state,
)
from prepare_experiment_campaign import (  # noqa: E402  # type: ignore[import-not-found]
    CLAIM_SCIENTIFIC,
    CampaignValidationError,
    validate_campaign,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_SCHEMA = "oczy/execution-report/v1"
DEFAULT_REPORT_FILENAME = "execution_report.json"
COLAB_RESULT_FILENAME = "result.json"
KAGGLE_PROVENANCE_FILENAME = "remote_run_provenance.json"

CLASS_COMPLETE = "COMPLETE"
CLASS_NULL = "NULL"
CLASS_INVALID = "INVALID"
CLASS_BLOCKED = "BLOCKED"

_VALID_CLASSIFICATIONS = frozenset(
    {CLASS_COMPLETE, CLASS_NULL, CLASS_INVALID, CLASS_BLOCKED}
)

# METRIC name=value  /  ASI name=value  — parsed from stdout fallback.
_METRIC_RE = re.compile(r"^METRIC\s+(\S+)\s*=\s*(.+)$")
_ASI_RE = re.compile(r"^ASI\s+(\S+)\s*=\s*(.+)$")

# Colab sentinel: the execution report stays on the remote VM and the
# current provider does not download it.  The runner emits a single-line
# sentinel ``OCZY_EXECUTION_REPORT_JSON=<compact-json>`` on stdout, which
# the provider captures into ``stdout.log``.  Missing, multiple, or
# conflicting (unparseable) sentinels are treated as INVALID provenance.
_COLAB_SENTINEL_PREFIX = "OCZY_EXECUTION_REPORT_JSON="
_COLAB_STDOUT_LOG = "stdout.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on missing/corrupt."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _parse_metric_lines(text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse METRIC/ASI lines from *text*, returning (metrics, asi_scores)."""
    metrics: dict[str, float] = {}
    asi: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if m:
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            continue
        a = _ASI_RE.match(line.strip())
        if a:
            try:
                asi[a.group(1)] = float(a.group(2))
            except ValueError:
                pass
    return metrics, asi


class SentinelError(ValueError):
    """Raised when the Colab stdout sentinel is missing, multiple, or corrupt."""


def _extract_sentinel_report(stdout_text: str) -> dict[str, Any]:
    """Extract the Colab execution report from the stdout sentinel.

    The runner emits exactly one line starting with
    ``OCZY_EXECUTION_REPORT_JSON=`` followed by compact JSON.  Missing,
    multiple, or unparseable sentinels raise :class:`SentinelError` so the
    caller can classify the job as INVALID.
    """
    sentinel_lines = [
        line for line in stdout_text.splitlines()
        if line.startswith(_COLAB_SENTINEL_PREFIX)
    ]
    if not sentinel_lines:
        raise SentinelError("no OCZY_EXECUTION_REPORT_JSON sentinel in stdout.log")
    if len(sentinel_lines) > 1:
        raise SentinelError(
            f"multiple ({len(sentinel_lines)}) "
            "OCZY_EXECUTION_REPORT_JSON sentinels in stdout.log"
        )
    json_text = sentinel_lines[0][len(_COLAB_SENTINEL_PREFIX):]
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise SentinelError(f"sentinel JSON unparseable: {exc}") from exc
    if not isinstance(raw, dict):
        raise SentinelError("sentinel JSON is not an object")
    return raw


# ---------------------------------------------------------------------------
# Report loading (provider-specific)
# ---------------------------------------------------------------------------

def _extract_kaggle_log_sentinel(log_path: Path) -> dict[str, Any]:
    """Extract the execution report from a Kaggle kernel log JSON stream.

    Kaggle kernel logs are downloaded as a JSON array of stream objects::

        [{"stream_name": "stdout", "time": 9.24,
          "data": "OCZY_EXECUTION_REPORT_JSON={...}"},
         {"stream_name": "stderr", "time": 9.24,
          "data": "execution_report: ..."},
         ...]

    The sentinel ``OCZY_EXECUTION_REPORT_JSON=<compact-json>`` appears in a
    ``stdout`` stream entry's ``data`` field.  This function concatenates all
    stdout data fragments and reuses :func:`_extract_sentinel_report` for
    strict single-sentinel validation (missing/multiple/corrupt →
    :class:`SentinelError`).
    """
    if not log_path.is_file():
        raise SentinelError(f"Kaggle log not found: {log_path.name}")
    try:
        raw = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SentinelError(f"Kaggle log not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise SentinelError(f"Kaggle log is not a JSON array: {log_path.name}")
    stdout_chunks: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if entry.get("stream_name") != "stdout":
            continue
        data = entry.get("data", "")
        if isinstance(data, str):
            stdout_chunks.append(data)
    stdout_text = "".join(stdout_chunks)
    return _extract_sentinel_report(stdout_text)


def _load_kaggle_report(output_dir: Path) -> dict[str, Any] | None:
    """Load a Kaggle execution report from *output_dir*.

    Resolution order:
    1. ``execution_report.json`` — written by the runner (primary).
    2. ``OCZY_EXECUTION_REPORT_JSON`` sentinel in a downloaded ``*.log`` JSON
       stream — the runner emits this on stdout; the Kaggle API captures it
       into a kernel log file.  A valid sentinel carries the full structured
       report (commit, provider, job_name, metrics) and outranks the
       provenance fallback because provenance only records bootstrap metadata,
       not the runner's scientific output.
    3. ``remote_run_provenance.json`` — written by the Kaggle bootstrap when
       the runner report is absent and no log sentinel is available.
    """
    report = _try_load_json(output_dir / DEFAULT_REPORT_FILENAME)
    if report is not None:
        return report
    # Try log sentinel before provenance: a valid sentinel has the full
    # structured report, while provenance only has bootstrap metadata.
    log_paths = sorted(output_dir.glob("*.log"))
    for log_path in log_paths:
        try:
            report = _extract_kaggle_log_sentinel(log_path)
        except SentinelError:
            continue
        report["_source"] = "kaggle_log_sentinel"
        return report
    # Fall back to provenance report from the bootstrap.
    provenance = _try_load_json(output_dir / KAGGLE_PROVENANCE_FILENAME)
    if provenance is not None:
        return _adapt_provenance_report(provenance)
    # Log file(s) existed but no valid sentinel, and no provenance file.
    if log_paths:
        raise SentinelError(
            "no valid OCZY_EXECUTION_REPORT_JSON sentinel in Kaggle log"
        )
    return None


def _adapt_provenance_report(provenance: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Kaggle bootstrap provenance report to the runner report shape."""
    job_spec = provenance.get("job_spec", {})
    status = provenance.get("status", "")
    exit_code = provenance.get("exit_code")
    if exit_code is None:
        if status == "complete":
            exit_code = 0
        elif status == "error":
            exit_code = 1
        else:
            exit_code = -1
    return {
        "schema_version": REPORT_SCHEMA,
        "job_name": job_spec.get("module", ""),
        "provider": PROVIDER_KAGGLE,
        "source_commit": job_spec.get("source_commit", ""),
        "module": job_spec.get("module", ""),
        "arguments": job_spec.get("arguments", []),
        "command": [],
        "exit_code": exit_code,
        "status": status,
        "started_utc": provenance.get("started_utc", ""),
        "finished_utc": provenance.get("finished_utc", ""),
        "stdout_file": "",
        "stderr_file": "",
        "metrics": {},
        "asi_scores": {},
        "_source": "provenance_fallback",
    }


def _load_colab_report(output_dir: Path) -> dict[str, Any] | None:
    """Load a Colab execution report from *output_dir*.

    The execution report normally stays on the remote VM — the current Colab
    provider does not download ``execution_report.json``.  Instead the runner
    emits a single-line sentinel ``OCZY_EXECUTION_REPORT_JSON=<compact-json>``
    which the provider captures into ``stdout.log``.  A valid sentinel carries
    the full structured report (commit, provider, job_name, metrics) and
    outranks the provider ``result.json`` fallback, which only records
    provider-level metadata (exit_code/status) without scientific provenance.

    Resolution order:
    1. ``execution_report.json`` — if the provider downloads it (primary,
       symmetric with Kaggle).
    2. ``OCZY_EXECUTION_REPORT_JSON`` sentinel in ``stdout.log`` — the full
       structured runner report.  A valid sentinel outranks result.json.
    3. ``result.json`` — provider result fallback, used only when no stdout
       sentinel source exists (no stdout.log at all).

    If ``stdout.log`` exists but the sentinel is missing, multiple, or
    unparseable, :class:`SentinelError` is raised so the caller classifies the
    job as INVALID rather than silently trusting the result.json fallback.
    """
    # 1. execution_report.json — primary if the provider downloads it.
    report = _try_load_json(output_dir / DEFAULT_REPORT_FILENAME)
    if report is not None:
        return report

    # 2. Sentinel in stdout.log — full structured report, outranks result.json.
    stdout_path = output_dir / _COLAB_STDOUT_LOG
    if stdout_path.is_file():
        try:
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SentinelError(f"cannot read stdout.log: {exc}") from exc
        report = _extract_sentinel_report(stdout_text)
        report["_source"] = "stdout_sentinel"
        return report

    # 3. result.json fallback — only when no stdout sentinel source exists.
    result = _try_load_json(output_dir / COLAB_RESULT_FILENAME)
    if result is not None:
        return _adapt_colab_result(result, output_dir)

    return None


def _adapt_colab_result(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Adapt a Colab provider result.json to the runner report shape."""
    exit_code = result.get("exit_code", -1)
    status = result.get("status", "error" if exit_code != 0 else "complete")
    # Also parse metrics/ASI from stdout.log if available.
    stdout_text = ""
    stdout_path = output_dir / _COLAB_STDOUT_LOG
    if stdout_path.is_file():
        try:
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    metrics, asi = _parse_metric_lines(stdout_text)
    return {
        "schema_version": REPORT_SCHEMA,
        "job_name": result.get("job_name", ""),
        "provider": PROVIDER_COLAB,
        "source_commit": result.get("source_commit", ""),
        "module": result.get("module", ""),
        "arguments": result.get("arguments", []),
        "command": result.get("command", []),
        "exit_code": exit_code,
        "status": status,
        "started_utc": result.get("started_utc", ""),
        "finished_utc": result.get("finished_utc", ""),
        "stdout_file": _COLAB_STDOUT_LOG,
        "stderr_file": result.get("stderr_file", ""),
        "metrics": metrics or result.get("metrics", {}),
        "asi_scores": asi or result.get("asi_scores", {}),
        "_source": "result_fallback",
    }


def _load_report(provider: str, output_dir: Path) -> dict[str, Any] | None:
    """Load the execution report for a job, dispatching by provider."""
    if provider == PROVIDER_COLAB:
        return _load_colab_report(output_dir)
    return _load_kaggle_report(output_dir)


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------

def _validate_provenance(
    report: dict[str, Any] | None,
    campaign_job: dict[str, Any],
    expected_commit: str,
) -> str | None:
    """Validate report provenance against the campaign.

    Returns an error message string if invalid, None if valid.
    """
    if report is None:
        return "execution report not found"
    if not isinstance(report, dict):
        return "execution report is not a valid object"

    # Schema version check.
    report_schema = report.get("schema_version", "")
    if report_schema and report_schema != REPORT_SCHEMA:
        return f"report schema_version mismatch: {report_schema!r}"

    # Source commit check (skipped when expected_commit is absent —
    # the caller may not have the campaign-level commit available).
    report_commit = report.get("source_commit", "")
    if not report_commit:
        return "report missing source_commit"
    if expected_commit and report_commit != expected_commit:
        return (
            f"source_commit mismatch: report={report_commit!r} "
            f"campaign={expected_commit!r}"
        )

    # Provider check.
    report_provider = report.get("provider", "")
    expected_provider = campaign_job.get("provider", "")
    if report_provider and report_provider != expected_provider:
        return (
            f"provider mismatch: report={report_provider!r} "
            f"campaign={expected_provider!r}"
        )

    # Job identity check (job_name in report should match campaign name).
    report_name = report.get("job_name", "")
    expected_name = campaign_job.get("name", "")
    if report_name and expected_name and report_name != expected_name:
        return (
            f"job identity mismatch: report={report_name!r} "
            f"campaign={expected_name!r}"
        )

    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_job_result(
    job_entry: dict[str, Any],
    report: dict[str, Any] | None,
    campaign_job: dict[str, Any],
) -> str:
    """Classify a single job result as COMPLETE/NULL/INVALID/BLOCKED.

    *job_entry* is the scheduler state job dict (contains ``state``,
    ``error``, ``provider``, etc.).  *report* is the parsed execution report
    dict or None.  *campaign_job* is the original campaign job dict (contains
    ``claim_class``, ``name``, ``provider``, etc.).

    The expected source commit is read from *campaign_job* via the
    ``_campaign_source_commit`` key set by the caller, or from the campaign
    top-level.  If absent, provenance validation is skipped for commit match
    but other checks still apply.
    """
    expected_commit = campaign_job.get("_campaign_source_commit", "")

    # --- INVALID: provenance failure ---
    provenance_error = _validate_provenance(report, campaign_job, expected_commit)
    if provenance_error is not None:
        # Distinguish "report missing entirely" (BLOCKED) from "report present
        # but provenance bad" (INVALID).  A missing report means the job never
        # produced structured output — that's an infrastructure block, not
        # invalid provenance.  BUT if the scheduler says the job succeeded
        # yet no report exists, that's INVALID (corrupt/missing provenance).
        scheduler_state = job_entry.get("state", "")
        if report is None and scheduler_state != SUCCEEDED:
            return CLASS_BLOCKED
        if report is None:
            # Scheduler says succeeded but no report — invalid provenance.
            return CLASS_INVALID
        return CLASS_INVALID

    assert report is not None  # provenance_error is None implies report exists

    # --- BLOCKED: infrastructure failure ---
    exit_code = report.get("exit_code")
    status = report.get("status", "")
    if exit_code is None or exit_code != 0:
        return CLASS_BLOCKED
    if status and status != "complete":
        return CLASS_BLOCKED

    # --- COMPLETE or NULL ---
    metrics = report.get("metrics", {})
    asi_scores = report.get("asi_scores", {})
    has_signals = bool(metrics) or bool(asi_scores)

    claim_class = campaign_job.get("claim_class", "")

    if has_signals:
        return CLASS_COMPLETE

    # No metrics/ASI signals.  Scientific jobs with no signals are NULL.
    # Infrastructure jobs with no signals are still COMPLETE (they don't
    # produce scientific metrics — their success is infrastructural).
    if claim_class == CLAIM_SCIENTIFIC:
        return CLASS_NULL
    return CLASS_COMPLETE


# ---------------------------------------------------------------------------
# Campaign collection
# ---------------------------------------------------------------------------

def collect_experiment_campaign(
    campaign_path: str | Path,
    batch_path: str | Path,
    state_path: str | Path,
    output_dir: str | Path,
    *,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Collect and classify results from a completed campaign run.

    Reads the campaign manifest, scheduler batch, and durable state, then
    walks each job's output directory for the execution report.  Writes
    ``campaign_execution_summary.json`` to *output_dir* and returns the
    summary dict.  When *report_dir* is given, reports are loaded from
    ``report_dir/<job_name>/`` instead of ``output_dir/<output_path>/``.
    """
    campaign_path = Path(campaign_path).resolve()
    batch_path = Path(batch_path).resolve()
    state_path = Path(state_path).resolve()
    report_dir_resolved = Path(report_dir).resolve() if report_dir else None

    # Load and validate campaign.
    if not campaign_path.is_file():
        raise FileNotFoundError(f"campaign manifest not found: {campaign_path}")
    try:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampaignValidationError(f"invalid campaign JSON: {exc}") from exc
    validate_campaign(campaign)

    source_commit = campaign["source_commit"]

    # Load scheduler batch + state.
    load_batch(batch_path)  # validate batch file exists
    state = load_state(state_path)
    state_jobs: dict[str, Any] = state.get("jobs", {})

    # Index campaign jobs by name.
    campaign_jobs_by_name: dict[str, dict[str, Any]] = {}
    for job in campaign["jobs"]:
        job_copy = dict(job)
        job_copy["_campaign_source_commit"] = source_commit
        campaign_jobs_by_name[job["name"]] = job_copy

    job_results: list[dict[str, Any]] = []
    classification_counts: dict[str, int] = {
        CLASS_COMPLETE: 0,
        CLASS_NULL: 0,
        CLASS_INVALID: 0,
        CLASS_BLOCKED: 0,
    }

    for job in campaign["jobs"]:
        name = job["name"]
        campaign_job = campaign_jobs_by_name[name]
        scheduler_job = state_jobs.get(name, {})

        # Resolve output directory for report loading.  When report_dir is
        # provided, reports are organized as report_dir/<job_name>/; otherwise
        # fall back to output_dir/<output_path>/.
        if report_dir_resolved is not None:
            job_output_dir = (report_dir_resolved / name).resolve()
        else:
            output_dir_rel = campaign_job.get("output_path", "")
            job_output_dir = (output_dir / output_dir_rel).resolve() if output_dir_rel else output_dir

        provider = campaign_job.get("provider", PROVIDER_KAGGLE)
        sentinel_error: str | None = None
        try:
            report = _load_report(provider, job_output_dir)
        except SentinelError as exc:
            report = None
            sentinel_error = str(exc)

        # Build a job_entry dict for classification.
        job_entry = {
            "name": name,
            "state": scheduler_job.get("state", ""),
            "error": scheduler_job.get("error"),
            "provider": provider,
        }

        if sentinel_error is not None:
            # Missing/multiple/corrupt sentinel → INVALID provenance.
            classification = CLASS_INVALID
        else:
            classification = classify_job_result(job_entry, report, campaign_job)
        classification_counts[classification] += 1

        # Build per-job result record.
        result_record: dict[str, Any] = {
            "name": name,
            "provider": provider,
            "claim_class": campaign_job.get("claim_class", ""),
            "phase": campaign_job.get("phase", ""),
            "classification": classification,
            "scheduler_state": scheduler_job.get("state", ""),
            "scheduler_error": scheduler_job.get("error"),
            "output_dir": str(job_output_dir),
        }
        if sentinel_error is not None:
            result_record["sentinel_error"] = sentinel_error

        if report is not None:
            result_record["exit_code"] = report.get("exit_code")
            result_record["report_status"] = report.get("status", "")
            result_record["source_commit"] = report.get("source_commit", "")
            result_record["metrics"] = report.get("metrics", {})
            result_record["asi_scores"] = report.get("asi_scores", {})
            result_record["report_source"] = report.get("_source", "execution_report")
            result_record["stdout_file"] = report.get("stdout_file", "")
            result_record["stderr_file"] = report.get("stderr_file", "")
            result_record["started_utc"] = report.get("started_utc", "")
            result_record["finished_utc"] = report.get("finished_utc", "")
        else:
            result_record["exit_code"] = None
            result_record["report_status"] = ""
            result_record["source_commit"] = ""
            result_record["metrics"] = {}
            result_record["asi_scores"] = {}
            result_record["report_source"] = "missing"
            result_record["stdout_file"] = ""
            result_record["stderr_file"] = ""
            result_record["started_utc"] = ""
            result_record["finished_utc"] = ""

        job_results.append(result_record)

    # Build summary.
    total = len(job_results)
    summary: dict[str, Any] = {
        "schema_version": "oczy/campaign-execution-summary/v1",
        "campaign_source_commit": source_commit,
        "campaign_path": str(campaign_path),
        "batch_path": str(batch_path),
        "state_path": str(state_path),
        "collected_at": _utc_now(),
        "total_jobs": total,
        "classifications": dict(classification_counts),
        "all_complete": (
            classification_counts[CLASS_COMPLETE] == total
            and classification_counts[CLASS_BLOCKED] == 0
            and classification_counts[CLASS_INVALID] == 0
        ),
        "has_blocked": classification_counts[CLASS_BLOCKED] > 0,
        "has_invalid": classification_counts[CLASS_INVALID] > 0,
        "has_null": classification_counts[CLASS_NULL] > 0,
        "jobs": job_results,
    }

    summary_path = output_dir / "campaign_execution_summary.json"
    _write_json(summary_path, summary)

    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_experiment_campaign",
        description=__doc__,
    )
    parser.add_argument("campaign", type=Path, help="Path to campaign manifest JSON.")
    parser.add_argument("batch", type=Path, help="Path to scheduler v2 batch JSON.")
    parser.add_argument(
        "--state", type=Path, required=True,
        help="Path to scheduler durable state file.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Base output directory containing per-job output dirs.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help="Base directory for per-job reports (default: --output/<output_path>/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = collect_experiment_campaign(
        args.campaign,
        args.batch,
        args.state,
        args.output,
        report_dir=args.report_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Exit 0 if no blocked/invalid, 1 otherwise.
    return 0 if not (summary["has_blocked"] or summary["has_invalid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
