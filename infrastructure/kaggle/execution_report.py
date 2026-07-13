"""Provider-neutral execution report loading and runtime validation.

Schema ``oczy/execution-report/v2``.  Owns the report constants, sentinel
extraction, provider dispatch, and the runtime-manifest gate so scheduler
and collector never diverge on loader logic.

Exports
-------
- ``EXECUTION_REPORT_SCHEMA_VERSION``
- ``EXECUTION_REPORT_SENTINEL_PREFIX``
- ``SentinelError``
- ``load_execution_report``
- ``validate_execution_report_runtime``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EXECUTION_REPORT_SCHEMA_VERSION: str = "oczy/execution-report/v2"
EXECUTION_REPORT_SENTINEL_PREFIX: str = "OCZY_EXECUTION_REPORT_JSON="

# Filenames used by provider loaders.
DEFAULT_REPORT_FILENAME: str = "execution_report.json"
COLAB_RESULT_FILENAME: str = "result.json"
KAGGLE_PROVENANCE_FILENAME: str = "remote_run_provenance.json"
COLAB_STDOUT_LOG: str = "stdout.log"

# Provider labels (kept in-sync with parallel_scheduler constants).
PROVIDER_KAGGLE: str = "kaggle"
PROVIDER_COLAB: str = "colab"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SentinelError(ValueError):
    """Raised when the execution-report sentinel is missing, multiple, or corrupt."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on missing/corrupt/non-object."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


# ---------------------------------------------------------------------------
# Sentinel extraction
# ---------------------------------------------------------------------------


def _extract_sentinel_report(stdout_text: str) -> dict[str, Any]:
    """Extract an execution report from the stdout sentinel line.

    The runner emits exactly one line beginning at column zero with
    ``OCZY_EXECUTION_REPORT_JSON=`` followed by compact JSON.  Missing,
    multiple, or unparseable sentinels raise :class:`SentinelError`.
    """
    prefix = EXECUTION_REPORT_SENTINEL_PREFIX
    sentinel_lines = [
        line for line in stdout_text.splitlines()
        if line.startswith(prefix)
    ]
    if not sentinel_lines:
        raise SentinelError(f"no {prefix[:-1]} sentinel in stdout text")
    if len(sentinel_lines) > 1:
        raise SentinelError(
            f"multiple ({len(sentinel_lines)}) {prefix[:-1]} sentinels"
        )
    json_text = sentinel_lines[0][len(prefix):]
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise SentinelError(f"sentinel JSON unparseable: {exc}") from exc
    if not isinstance(raw, dict):
        raise SentinelError("sentinel JSON is not an object")
    return raw


def _extract_kaggle_log_sentinel(log_path: Path) -> dict[str, Any]:
    """Extract the execution report from a Kaggle kernel log JSON stream.

    Kaggle kernel logs are a JSON array of stream objects.  The sentinel
    ``OCZY_EXECUTION_REPORT_JSON=<compact-json>`` appears in a ``stdout``
    stream entry's ``data`` field.  This function concatenates all stdout
    data fragments and reuses ``_extract_sentinel_report`` for strict
    single-sentinel validation.
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


# ---------------------------------------------------------------------------
# Provider-specific report loading
# ---------------------------------------------------------------------------


def _load_kaggle_report(output_dir: Path) -> dict[str, Any] | None:
    """Load a Kaggle execution report from *output_dir*.

    Resolution order:
    1. ``execution_report.json`` — primary.
    2. ``OCZY_EXECUTION_REPORT_JSON`` sentinel in downloaded ``*.log`` stream.
    3. ``remote_run_provenance.json`` — diagnostic fallback only (not runtime evidence).
    """
    report = _try_load_json(output_dir / DEFAULT_REPORT_FILENAME)
    if report is not None:
        return report

    log_paths = sorted(output_dir.glob("*.log"))
    for log_path in log_paths:
        try:
            report = _extract_kaggle_log_sentinel(log_path)
        except SentinelError:
            continue
        report["_source"] = "kaggle_log_sentinel"
        return report

    provenance = _try_load_json(output_dir / KAGGLE_PROVENANCE_FILENAME)
    if provenance is not None:
        provenance["_source"] = "provenance_fallback"
        return provenance

    if log_paths:
        raise SentinelError(
            "no valid OCZY_EXECUTION_REPORT_JSON sentinel in Kaggle log"
        )
    return None


def _load_colab_report(output_dir: Path) -> dict[str, Any] | None:
    """Load a Colab execution report from *output_dir*.

    Resolution order:
    1. ``execution_report.json`` — primary if downloaded.
    2. ``OCZY_EXECUTION_REPORT_JSON`` sentinel in ``stdout.log``.
    3. ``result.json`` — provider fallback (not runtime evidence).
    """
    report = _try_load_json(output_dir / DEFAULT_REPORT_FILENAME)
    if report is not None:
        return report

    stdout_path = output_dir / COLAB_STDOUT_LOG
    if stdout_path.is_file():
        try:
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SentinelError(f"cannot read stdout.log: {exc}") from exc
        report = _extract_sentinel_report(stdout_text)
        report["_source"] = "stdout_sentinel"
        return report

    result = _try_load_json(output_dir / COLAB_RESULT_FILENAME)
    if result is not None:
        result["_source"] = "result_fallback"
        return result

    return None


def load_execution_report(
    provider: str,
    output_dir: Path | str,
) -> dict[str, Any] | None:
    """Load the execution report for a job, dispatching by *provider*.

    Parameters
    ----------
    provider:
        ``"kaggle"`` or ``"colab"``.
    output_dir:
        Directory containing the job's collected output.

    Returns
    -------
    dict or None
        The loaded report dict (with ``_source`` marker), or ``None`` if no
        report could be found.
    """
    output_dir = Path(output_dir)
    if provider == PROVIDER_COLAB:
        return _load_colab_report(output_dir)
    return _load_kaggle_report(output_dir)


# ---------------------------------------------------------------------------
# Runtime manifest validation on a loaded report
# ---------------------------------------------------------------------------


def validate_execution_report_runtime(
    report: dict[str, Any],
    expected_manifest: dict[str, Any],
) -> None:
    """Validate runtime identity on a loaded execution report.

    Requires:
    * ``report`` schema is ``oczy/execution-report/v2`` (violation raises).
    * ``report`` contains a valid ``observed_runtime_manifest`` with correct
      self-hash.
    * ``report["expected_runtime_manifest_sha256"]`` matches the hash of
      *expected_manifest*.
    * ``report["observed_runtime_manifest"]`` equals *expected_manifest* via
      exact canonical comparison.

    A ``_source: "provenance_fallback"`` or ``"result_fallback"`` report
    cannot satisfy runtime verification: those are diagnostic records, not
    structured runner output.
    """
    source = report.get("_source", "")
    if source in ("provenance_fallback", "result_fallback"):
        raise SentinelError(
            f"report _source={source!r} is a diagnostic fallback, "
            f"not a structured execution-report/v2 — cannot verify runtime"
        )

    schema_version = report.get("schema_version")
    if schema_version != EXECUTION_REPORT_SCHEMA_VERSION:
        raise SentinelError(
            f"expected report schema {EXECUTION_REPORT_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )

    # Import here to avoid circular dependency at module level (runtime_manifest
    # is a sibling, not a package with a shared __init__).
    _SCRIPT_DIR = str(Path(__file__).resolve().parent)
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from runtime_manifest import (  # type: ignore[import-not-found]
        RuntimeManifestError,
        compute_manifest_sha256,
        compare_runtime_manifests,
        validate_runtime_manifest,
    )

    expected_hash = report.get("expected_runtime_manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SentinelError(
            "report missing or invalid `expected_runtime_manifest_sha256`"
        )

    actual_expected_hash = compute_manifest_sha256(expected_manifest)
    if expected_hash != actual_expected_hash:
        raise RuntimeManifestError(
            f"expected_runtime_manifest_sha256 mismatch: "
            f"report claims {expected_hash}, "
            f"computed from expected manifest {actual_expected_hash}"
        )

    observed = report.get("observed_runtime_manifest")
    if not isinstance(observed, dict):
        raise SentinelError("report missing `observed_runtime_manifest` or not an object")

    try:
        validate_runtime_manifest(observed)
    except RuntimeManifestError as exc:
        raise SentinelError(
            f"observed_runtime_manifest fails validation: {exc}"
        ) from exc

    mismatches = compare_runtime_manifests(expected_manifest, observed)
    if mismatches:
        raise RuntimeManifestError(
            f"runtime manifest mismatch: {mismatches[0]}"
            + (f" (+{len(mismatches) - 1} more)" if len(mismatches) > 1 else "")
        )
