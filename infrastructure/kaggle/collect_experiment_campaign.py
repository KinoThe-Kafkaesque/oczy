"""Collect and classify results from a mixed Kaggle/Colab experiment campaign.

Reads the scheduler durable state plus each job's output directory, parses the
structured execution report (via ``execution_report.py``), validates source
commit / provider / job identity / exit code, verifies runtime manifest identity
against the campaign and batch, and classifies each job as ``COMPLETE``,
``NULL``, ``INVALID``, or ``BLOCKED``.

Classification rules (no thresholds are changed):

* **COMPLETE** — exit_code == 0, provenance valid (source_commit, provider,
  job_name all match the campaign, schema is execution-report/v2), report
  parseable, status == "complete", runtime manifest verified.
* **NULL** — exit_code == 0, provenance valid, runtime manifest verified,
  status == "complete", claim_class == "scientific", and no metrics/asi_signals
  found.  Infrastructure jobs are **never** NULL.
* **INVALID** — missing or corrupt provenance, source_commit/provider/job
  identity mismatch, wrong report schema, corrupt/unparseable report, missing
  or mismatched runtime manifest, campaign/batch manifest divergence, or report
  ``_source`` is a diagnostic fallback.  Missing provenance is INVALID, not NULL.
* **BLOCKED** — exit_code != 0, status != "complete", or report file missing
  entirely.  A failed infrastructure job is BLOCKED, never a scientific NULL.

Writes ``campaign_execution_summary.json`` (schema
``oczy/campaign-execution-summary/v2``) to the output directory.
"""

from __future__ import annotations

import argparse
import json
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
from execution_report import (  # noqa: E402  # type: ignore[import-not-found]
    EXECUTION_REPORT_SCHEMA_VERSION,
    SentinelError,
    load_execution_report,
    validate_execution_report_runtime,
)
from runtime_manifest import (  # noqa: E402  # type: ignore[import-not-found]
    RuntimeManifestError,
    compute_manifest_sha256,
    validate_runtime_manifest,
    compare_runtime_manifests,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_COMPLETE = "COMPLETE"
CLASS_NULL = "NULL"
CLASS_INVALID = "INVALID"
CLASS_BLOCKED = "BLOCKED"

_VALID_CLASSIFICATIONS = frozenset(
    {CLASS_COMPLETE, CLASS_NULL, CLASS_INVALID, CLASS_BLOCKED}
)


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------

def _validate_provenance(
    report: dict[str, Any] | None,
    campaign_job: dict[str, Any],
    expected_commit: str,
) -> str | None:
    """Validate report provenance against the campaign.

    All checks are required — no conditional skipping.  Returns an error
    message string if invalid, None if valid.
    """
    if report is None:
        return "execution report not found"
    if not isinstance(report, dict):
        return "execution report is not a valid object"

    # Schema version — must be execution-report/v2 exactly.
    report_schema = report.get("schema_version", "")
    if report_schema != EXECUTION_REPORT_SCHEMA_VERSION:
        return f"report schema_version mismatch: {report_schema!r}"

    # Source commit — required, must match campaign.
    report_commit = report.get("source_commit", "")
    if not report_commit:
        return "report missing source_commit"
    if report_commit != expected_commit:
        return (
            f"source_commit mismatch: report={report_commit!r} "
            f"campaign={expected_commit!r}"
        )

    # Provider — required, must match campaign.
    report_provider = report.get("provider", "")
    expected_provider = campaign_job.get("provider", "")
    if not report_provider:
        return "report missing provider"
    if report_provider != expected_provider:
        return (
            f"provider mismatch: report={report_provider!r} "
            f"campaign={expected_provider!r}"
        )

    # Job identity — required, must match campaign.
    report_name = report.get("job_name", "")
    expected_name = campaign_job.get("name", "")
    if not report_name:
        return "report missing job_name"
    if expected_name and report_name != expected_name:
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
    *,
    expected_runtime_manifest: dict[str, Any] | None = None,
) -> str:
    """Classify a single job result as COMPLETE/NULL/INVALID/BLOCKED.

    *job_entry* is the scheduler state job dict (contains ``state``,
    ``error``, ``provider``, etc.).  *report* is the parsed execution report
    dict or None.  *campaign_job* is the original campaign job dict (contains
    ``claim_class``, ``name``, ``provider``, etc.).

    The expected source commit is read from *campaign_job* via the
    ``_campaign_source_commit`` key set by the caller.

    When *expected_runtime_manifest* is provided (a valid runtime manifest
    dict), the report's observed runtime manifest is independently validated
    against it via :func:`execution_report.validate_execution_report_runtime`.
    A mismatch or missing observed manifest results in ``INVALID`` before
    exit/status/metrics are inspected.
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

    # --- INVALID: runtime manifest failure ---
    # Validate observed manifest against expected before inspecting metrics.
    # Diagnostic fallbacks (provenance/result) are rejected by
    # validate_execution_report_runtime.
    if expected_runtime_manifest is not None:
        try:
            validate_execution_report_runtime(report, expected_runtime_manifest)
        except (SentinelError, RuntimeManifestError):
            return CLASS_INVALID

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
    walks each job's output directory for the execution report.  Verifies
    campaign/batch runtime manifest equality and validates each report's
    observed runtime manifest independently of scheduler state.  Writes
    ``campaign_execution_summary.json`` (schema
    ``oczy/campaign-execution-summary/v2``) to *output_dir* and returns the
    summary dict.  When *report_dir* is given, reports are loaded from
    ``report_dir/<job_name>/`` instead of ``output_dir/<output_path>/``.
    """
    campaign_path = Path(campaign_path).resolve()
    batch_path = Path(batch_path).resolve()
    state_path = Path(state_path).resolve()
    output_dir = Path(output_dir).resolve()
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

    # Load scheduler batch (returns jobs list) + state.
    batch_jobs = load_batch(batch_path)
    state = load_state(state_path)
    state_jobs: dict[str, Any] = state.get("jobs", {})

    # Index batch jobs by name for manifest cross-check.
    batch_jobs_by_name: dict[str, dict[str, Any]] = {}
    for bj in batch_jobs:
        batch_jobs_by_name[bj["name"]] = bj

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

        # --- Cross-check campaign vs batch runtime manifest ---
        manifest_error: str | None = None
        expected_manifest: dict[str, Any] | None = None
        expected_manifest_sha256: str | None = None

        campaign_rm = campaign_job.get("runtime_manifest")
        batch_job = batch_jobs_by_name.get(name, {})

        if not isinstance(campaign_rm, dict):
            manifest_error = "campaign job missing or invalid runtime_manifest"
        elif not isinstance(batch_job, dict) or not isinstance(
            batch_job.get("runtime_manifest"), dict
        ):
            manifest_error = "batch job missing or invalid runtime_manifest"
        else:
            batch_rm = batch_job["runtime_manifest"]
            try:
                validate_runtime_manifest(campaign_rm)
                validate_runtime_manifest(batch_rm)
            except RuntimeManifestError as exc:
                manifest_error = f"runtime_manifest validation failed: {exc}"
            else:
                mismatches = compare_runtime_manifests(campaign_rm, batch_rm)
                if mismatches:
                    manifest_error = (
                        f"campaign/batch runtime_manifest mismatch: {mismatches[0]}"
                    )
                    if len(mismatches) > 1:
                        manifest_error += f" (+{len(mismatches) - 1} more)"
                else:
                    expected_manifest = campaign_rm
                    expected_manifest_sha256 = compute_manifest_sha256(expected_manifest)

        # Load execution report via shared loader.
        sentinel_error: str | None = None
        report: dict[str, Any] | None = None
        try:
            report = load_execution_report(provider, job_output_dir)
        except SentinelError as exc:
            sentinel_error = str(exc)

        # Build a job_entry dict for classification.
        job_entry = {
            "name": name,
            "state": scheduler_job.get("state", ""),
            "error": scheduler_job.get("error"),
            "provider": provider,
        }

        # Classification: campaign/batch manifest divergence is INVALID
        # before the report is even loaded.  If the manifest is ok, pass
        # it to classify_job_result for observed-vs-expected validation.
        if manifest_error is not None:
            classification = CLASS_INVALID
        elif sentinel_error is not None:
            classification = CLASS_INVALID
        else:
            classification = classify_job_result(
                job_entry, report, campaign_job,
                expected_runtime_manifest=expected_manifest,
            )
        classification_counts[classification] += 1

        # --- Runtime manifest verification for the summary ---
        runtime_manifest_verified: bool = False
        observed_manifest_sha256: str | None = None
        runtime_error_detail: str | None = manifest_error

        if report is not None and expected_manifest is not None:
            observed = report.get("observed_runtime_manifest")
            if isinstance(observed, dict):
                try:
                    observed_manifest_sha256 = compute_manifest_sha256(observed)
                except (RuntimeManifestError, TypeError, ValueError):
                    pass
            if classification == CLASS_COMPLETE or classification == CLASS_NULL:
                runtime_manifest_verified = True
            elif runtime_error_detail is None:
                runtime_error_detail = (
                    f"runtime manifest mismatch "
                    f"(classification={classification})"
                )
        elif report is not None and expected_manifest is None:
            if runtime_error_detail is None:
                runtime_error_detail = (
                    "expected_runtime_manifest unavailable "
                    "(campaign/batch cross-check failed)"
                )

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
            "expected_runtime_manifest_sha256": expected_manifest_sha256,
            "observed_runtime_manifest_sha256": observed_manifest_sha256,
            "runtime_manifest_verified": runtime_manifest_verified,
        }
        if runtime_error_detail is not None:
            result_record["runtime_manifest_error"] = runtime_error_detail
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
        "schema_version": "oczy/campaign-execution-summary/v2",
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

    output_dir.mkdir(parents=True, exist_ok=True)
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
