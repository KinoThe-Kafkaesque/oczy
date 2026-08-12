#!/usr/bin/env python3
"""
Reproducible OMP throughput probe.

Wraps ``omp bench`` with a pinned prompt and all controls fixed, records
TTFT (time-to-first-token) and TPS (tokens-per-second) metrics, and emits
a versioned reproducibility envelope with full provenance.  Standard
library only — no third-party dependencies.

OMP model selectors are **positional**, not ``--model`` flags:

    omp bench devin/swe-1-7 devin/glm-5-2 devin/kimi-k2-7 --runs 3

Devin reproduction command (raw omp bench)::

    omp bench devin/swe-1-7 devin/glm-5-2 devin/kimi-k2-7 --runs 3 --max-tokens 256 --par 1 --service-tier none --json

Equivalent via this probe::

    python benchmarks/omp/throughput_probe.py devin/swe-1-7 devin/glm-5-2 devin/kimi-k2-7 \\
        --runs 3 --max-tokens 256 --par 1 --service-tier none \\
        --output /tmp/devin-throughput.json

Exit codes:
    0  — all benchmark runs succeeded
    1  — one or more benchmark runs failed (valid JSON was still captured)
    2  — probe-level error (binary not found, timeout, invalid JSON, …)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── constants ────────────────────────────────────────────────────────

SCHEMA_VERSION = "oczy/omp-throughput-probe/v1"
PROBE_VERSION = "1.0.0"

DEFAULT_RUNS = 3
DEFAULT_MAX_TOKENS = 256
DEFAULT_PAR = 1
DEFAULT_SERVICE_TIER = "none"
DEFAULT_OMP_BIN = "omp"
DEFAULT_TIMEOUT_SECONDS = 1800

# Valid --service-tier values, matching OMP's SERVICE_TIER_OPENAI_VALUES
# (src/config/service-tier.ts).  OMP's serviceTierSettingToTier() maps
# "none" to undefined (omit the wire parameter); the others pass through.
SERVICE_TIER_CHOICES = (
    "none",
    "auto",
    "default",
    "flex",
    "scale",
    "priority",
)

# ResolvedThinkingLevel values (excluding "inherit" which is resolved away
# before serialization).  The `thinking` field is optional in the JSON.
THINKING_LEVELS = frozenset(("off", "minimal", "low", "medium", "high", "xhigh"))

# The OMP bundled bench prompt, embedded verbatim (trimmed, matching OMP's
# own `benchPrompt.trim()` at src/cli/bench-cli.ts:44).  Pinning this in the
# probe makes results reproducible even if a future OMP version changes the
# bundled prompt.
BENCH_PROMPT = """\
Write a detailed, four-paragraph explanation of how a web browser renders a webpage. Cover the process from receiving the initial HTML payload to painting pixels on the screen. Include the construction of the DOM and CSSOM, the render tree, layout, and painting.

Form:
- Plain paragraphs only: no headings, no lists, no code fences, no preamble.
- Do not summarize early; keep explaining until you reach the token limit.
- Output only the explanation.
""".strip()

# ─── exit codes ───────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_BENCH_FAILURES = 1
EXIT_PROBE_ERROR = 2

# ─── status values ────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_PARTIAL_FAILURE = "partial_failure"
STATUS_ALL_FAILED = "all_failed"


# ─── errors ───────────────────────────────────────────────────────────


class ProbeError(Exception):
    """Raised when the probe itself fails (not individual benchmark runs)."""

    def __init__(self, message: str, *, exit_code: int = EXIT_PROBE_ERROR) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ─── helpers ──────────────────────────────────────────────────────────


def _is_number(v: Any) -> bool:
    """True if *v* is a real number (int or float, but not bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


# ─── validation ───────────────────────────────────────────────────────


def _validate_success_result(data: dict[str, Any], context: str) -> None:
    """Validate a successful run result: finite, nonnegative, TTFT <= duration."""
    for field in ("ttftMs", "durationMs", "outputTokens", "tokensPerSecond"):
        val = data.get(field)
        if val is None:
            raise ProbeError(f"{context}: success result missing '{field}'")
        if not _is_number(val):
            raise ProbeError(f"{context}.{field}: expected number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ProbeError(f"{context}.{field}: not finite ({val})")
        if val < 0:
            raise ProbeError(f"{context}.{field}: negative ({val})")
    if data["ttftMs"] > data["durationMs"]:
        raise ProbeError(
            f"{context}: ttftMs ({data['ttftMs']}) exceeds durationMs ({data['durationMs']})"
        )


def _validate_failure_result(data: dict[str, Any], context: str) -> None:
    """Validate a failure run result: error must be a nonempty string."""
    err = data.get("error")
    if err is None:
        raise ProbeError(f"{context}: failure result missing 'error'")
    if not isinstance(err, str):
        raise ProbeError(f"{context}.error: expected string, got {type(err).__name__}")
    if not err.strip():
        raise ProbeError(f"{context}.error: empty string")


def _validate_result(data: Any, context: str) -> None:
    if not isinstance(data, dict):
        raise ProbeError(f"{context}: expected object, got {type(data).__name__}")
    ok = data.get("ok")
    if ok is True:
        _validate_success_result(data, context)
    elif ok is False:
        _validate_failure_result(data, context)
    else:
        raise ProbeError(f"{context}.ok: expected true or false, got {ok!r}")


def _validate_average(
    data: Any, successes: list[dict[str, Any]], context: str
) -> None:
    """Validate average: null iff no successes; otherwise mean values must match."""
    if len(successes) == 0:
        if data is not None:
            raise ProbeError(
                f"{context}.average: expected null when no successful runs, got {data}"
            )
        return
    if data is None:
        raise ProbeError(f"{context}.average: expected non-null when successful runs exist")
    if not isinstance(data, dict):
        raise ProbeError(f"{context}.average: expected object, got {type(data).__name__}")
    for field in ("ttftMs", "durationMs", "outputTokens", "tokensPerSecond"):
        val = data.get(field)
        if val is None:
            raise ProbeError(f"{context}.average: missing field '{field}'")
        if not _is_number(val):
            raise ProbeError(
                f"{context}.average.{field}: expected number, got {type(val).__name__}"
            )
        expected = sum(s[field] for s in successes) / len(successes)
        if not math.isclose(val, expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ProbeError(
                f"{context}.average.{field}: expected ~{expected}, got {val}"
            )


def _validate_model_report(data: Any, index: int, expected_runs: int) -> None:
    ctx = f"models[{index}]"
    if not isinstance(data, dict):
        raise ProbeError(f"{ctx}: expected object, got {type(data).__name__}")

    selector = data.get("selector")
    if selector is None:
        raise ProbeError(f"{ctx}: missing field 'selector'")
    if not isinstance(selector, str):
        raise ProbeError(f"{ctx}.selector: expected string, got {type(selector).__name__}")

    model = data.get("model")
    if model is None:
        raise ProbeError(f"{ctx}: missing field 'model'")
    if not isinstance(model, str):
        raise ProbeError(f"{ctx}.model: expected string, got {type(model).__name__}")
    if not model.strip():
        raise ProbeError(f"{ctx}.model: resolved model string is empty")
    thinking = data.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, str):
            raise ProbeError(
                f"{ctx}.thinking: expected string or null, got {type(thinking).__name__}"
            )
        if thinking not in THINKING_LEVELS:
            raise ProbeError(f"{ctx}.thinking: unknown value {thinking!r}")

    results = data.get("results")
    if results is None:
        raise ProbeError(f"{ctx}: missing field 'results'")
    if not isinstance(results, list):
        raise ProbeError(f"{ctx}.results: expected array, got {type(results).__name__}")

    # Result count must equal expected runs, except for the preflight-failure
    # case where OMP pushes a single failure result and skips remaining runs.
    if len(results) != expected_runs:
        is_preflight = (
            len(results) == 1
            and isinstance(results[0], dict)
            and results[0].get("ok") is False
        )
        if not is_preflight:
            raise ProbeError(
                f"{ctx}.results: expected {expected_runs} result(s), got {len(results)}"
            )

    for i, result in enumerate(results):
        _validate_result(result, f"{ctx}.results[{i}]")

    successes = [r for r in results if isinstance(r, dict) and r.get("ok") is True]
    _validate_average(data.get("average"), successes, ctx)


def validate_bench_summary(
    data: Any,
    expected_models: list[str],
    expected_runs: int,
    expected_max_tokens: int,
) -> None:
    """Validate that *data* conforms to the OMP BenchSummary JSON shape and
    matches the expected probe configuration.

    Enforces:
      - raw runs/maxTokens match the values requested on the CLI
      - exact selector order and count
      - success metrics are finite, nonnegative, with TTFT <= duration
      - failure errors are nonempty strings
      - result count equals runs (except single preflight failure)
      - raw failure count matches actual ok:false results
      - average is null iff no successes, otherwise mean values match (isclose)

    Raises ProbeError with a descriptive message on any mismatch.
    Extra fields not in the schema are ignored (forward-compatible).
    """
    if not isinstance(data, dict):
        raise ProbeError(f"top-level JSON: expected object, got {type(data).__name__}")

    runs = data.get("runs")
    if runs is None:
        raise ProbeError("missing required field 'runs'")
    if not _is_positive_int(runs):
        raise ProbeError(f"runs: expected positive integer, got {runs!r}")
    if runs != expected_runs:
        raise ProbeError(f"runs: expected {expected_runs}, got {runs}")

    max_tokens = data.get("maxTokens")
    if max_tokens is None:
        raise ProbeError("missing required field 'maxTokens'")
    if not _is_positive_int(max_tokens):
        raise ProbeError(f"maxTokens: expected positive integer, got {max_tokens!r}")
    if max_tokens != expected_max_tokens:
        raise ProbeError(f"maxTokens: expected {expected_max_tokens}, got {max_tokens}")

    models = data.get("models")
    if models is None:
        raise ProbeError("missing required field 'models'")
    if not isinstance(models, list) or len(models) == 0:
        raise ProbeError("models: expected non-empty array")

    # Exact selector order and count must match what the user requested.
    actual_selectors = [
        m.get("selector") if isinstance(m, dict) else None for m in models
    ]
    if actual_selectors != expected_models:
        raise ProbeError(
            f"model selectors: expected {expected_models}, got {actual_selectors}"
        )

    failures = data.get("failures")
    if failures is None:
        raise ProbeError("missing required field 'failures'")
    if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
        raise ProbeError(f"failures: expected non-negative integer, got {failures!r}")

    stbf = data.get("serviceTierByFamily")
    if stbf is not None and not isinstance(stbf, dict):
        raise ProbeError(
            f"serviceTierByFamily: expected object or null, got {type(stbf).__name__}"
        )

    for i, report in enumerate(models):
        _validate_model_report(report, i, expected_runs)

    # Verify raw failure count matches actual ok:false results.
    actual_failures = sum(
        1
        for m in models
        for r in (m.get("results", []) if isinstance(m, dict) else [])
        if isinstance(r, dict) and r.get("ok") is False
    )
    if failures != actual_failures:
        raise ProbeError(
            f"failures: raw value ({failures}) does not match actual "
            f"failure count ({actual_failures})"
        )


# ─── binary resolution ────────────────────────────────────────────────


def resolve_omp_binary(omp_bin: str) -> Path:
    """Resolve the OMP binary to an absolute, executable Path."""
    p = Path(omp_bin)
    # Absolute path or relative path with a directory component — use as-is.
    if p.is_absolute() or len(p.parts) > 1:
        if not p.exists():
            raise ProbeError(f"omp binary not found: {p}")
        if not os.access(p, os.X_OK):
            raise ProbeError(f"omp binary not executable: {p}")
        return p.resolve()
    # Bare name — search PATH.
    resolved = shutil.which(omp_bin)
    if resolved is None:
        raise ProbeError(f"omp binary '{omp_bin}' not found in PATH")
    return Path(resolved)


# ─── version capture ──────────────────────────────────────────────────


def capture_omp_version(omp_path: Path) -> str:
    """Run ``omp --version`` and return the stripped stdout."""
    try:
        result = subprocess.run(
            [str(omp_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise ProbeError("omp --version timed out after 30s") from e
    except OSError as e:
        raise ProbeError(f"failed to execute omp --version: {e}") from e

    if result.returncode != 0:
        raise ProbeError(
            f"omp --version exited with code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    version = result.stdout.strip()
    if not version:
        raise ProbeError("omp --version produced no output")
    return version


# ─── prompt loading ───────────────────────────────────────────────────


def load_prompt(prompt_file: str | None) -> tuple[str, str, str | None]:
    """Load the bench prompt.

    Returns ``(prompt_text, source, file_path)`` where *source* is
    ``"embedded"`` or ``"file"``, and *file_path* is the resolved path
    or ``None``.
    """
    if prompt_file is not None:
        p = Path(prompt_file)
        if not p.exists():
            raise ProbeError(f"prompt file not found: {p}")
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            raise ProbeError(f"prompt file is empty: {p}")
        return content, "file", str(p.resolve())
    return BENCH_PROMPT, "embedded", None


# ─── command building ─────────────────────────────────────────────────


def build_bench_command(
    omp_path: Path,
    models: list[str],
    runs: int,
    max_tokens: int,
    par: int,
    service_tier: str,
    prompt: str,
) -> list[str]:
    """Build the argv list for ``omp bench`` (no shell, no shell=True)."""
    cmd: list[str] = [str(omp_path), "bench", *models]
    cmd += ["--runs", str(runs)]
    cmd += ["--max-tokens", str(max_tokens)]
    cmd += ["--par", str(par)]
    cmd += ["--service-tier", service_tier]
    cmd += ["--prompt", prompt]
    cmd += ["--json"]
    return cmd


# ─── descriptive statistics ───────────────────────────────────────────


def _descriptive_stats(values: list[float]) -> dict[str, Any]:
    """Compute count/mean/median/min/max/sample_stdev/coefficientOfVariationPercent.

    sampleStdev is null when count < 2 (undefined for a single sample).
    coefficientOfVariationPercent is null when count < 2 or mean is zero;
    otherwise it is (stdev / mean * 100).
    """
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "sampleStdev": None,
            "coefficientOfVariationPercent": None,
        }
    mean_val = statistics.fmean(values)
    median_val = statistics.median(values)
    stdev_val = statistics.stdev(values) if count >= 2 else None
    cv_pct = (
        (stdev_val / mean_val * 100)
        if (stdev_val is not None and mean_val != 0)
        else None
    )
    return {
        "count": count,
        "mean": mean_val,
        "median": median_val,
        "min": min(values),
        "max": max(values),
        "sampleStdev": stdev_val,
        "coefficientOfVariationPercent": cv_pct,
    }


def _model_stats(successes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-model descriptive stats from that model's successful runs only."""
    ttft_values = [s["ttftMs"] for s in successes]
    tps_values = [s["tokensPerSecond"] for s in successes]
    duration_values = [s["durationMs"] for s in successes]
    token_values = [s["outputTokens"] for s in successes]
    return {
        "ttftMs": _descriptive_stats(ttft_values),
        "tokensPerSecond": _descriptive_stats(tps_values),
        "durationMs": _descriptive_stats(duration_values),
        "outputTokens": _descriptive_stats(token_values),
    }


# ─── warnings ─────────────────────────────────────────────────────────


def _compute_warnings(raw: dict[str, Any], max_tokens: int) -> list[str]:
    """Emit factual warnings without speculation."""
    warnings: list[str] = []
    for report in raw.get("models", []):
        selector = report.get("selector", "?")
        for i, result in enumerate(report.get("results", [])):
            if result.get("ok") is True and result.get("outputTokens", 0) > max_tokens:
                warnings.append(
                    f"{selector} run {i + 1}: outputTokens ({result['outputTokens']}) "
                    f"exceeds maxTokens ({max_tokens})"
                )
    warnings.append(
        "TTFT measures time to the first non-empty streamed text, thinking, "
        "or tool-call delta; it is not guaranteed to be the first visible "
        "answer text."
    )
    return warnings


# ─── status ───────────────────────────────────────────────────────────


def _compute_status(raw: dict[str, Any], bench_exit_code: int) -> str:
    """Determine top-level status from raw bench summary and process exit code.

    A nonzero bench exit code with zero raw failures still produces a
    non-ok status (the process signalled a problem even though the JSON
    reported no failures).
    """
    total_results = 0
    total_successes = 0
    for report in raw.get("models", []):
        for result in report.get("results", []):
            total_results += 1
            if result.get("ok") is True:
                total_successes += 1
    if total_successes == 0:
        return STATUS_ALL_FAILED
    if total_successes < total_results or bench_exit_code != 0:
        return STATUS_PARTIAL_FAILURE
    return STATUS_OK


# ─── metrics extraction ───────────────────────────────────────────────


def _extract_metrics(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract clean per-model metrics from validated raw BenchSummary."""
    metrics: list[dict[str, Any]] = []
    for report in raw.get("models", []):
        results = report.get("results", [])
        successes = [r for r in results if r.get("ok") is True]
        failures = [r for r in results if r.get("ok") is False]
        avg = report.get("average")
        metrics.append(
            {
                "selector": report.get("selector"),
                "model": report.get("model"),
                "thinking": report.get("thinking"),
                "runs": len(results),
                "successfulRuns": len(successes),
                "failedRuns": len(failures),
                "average": {
                    "ttftMs": avg["ttftMs"],
                    "durationMs": avg["durationMs"],
                    "outputTokens": avg["outputTokens"],
                    "tokensPerSecond": avg["tokensPerSecond"],
                }
                if avg is not None
                else None,
                "stats": _model_stats(successes),
                "results": results,
            }
        )
    return metrics


# ─── probe file hash ──────────────────────────────────────────────────


def _probe_file_sha256() -> str:
    """Return SHA-256 of this probe script file."""
    p = Path(__file__).resolve()
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ─── envelope building ────────────────────────────────────────────────


def build_envelope(
    *,
    omp_path: Path,
    omp_version: str,
    models: list[str],
    runs: int,
    max_tokens: int,
    par: int,
    service_tier: str,
    prompt: str,
    prompt_source: str,
    prompt_file: str | None,
    timeout_seconds: int,
    command: list[str],
    raw: dict[str, Any],
    bench_exit_code: int,
    bench_duration: float,
    bench_stderr: str,
    timed_out: bool,
) -> dict[str, Any]:
    """Assemble the versioned reproducibility envelope."""
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    status = _compute_status(raw, bench_exit_code)
    return {
        "schema_version": SCHEMA_VERSION,
        "probeVersion": PROBE_VERSION,
        "status": status,
        "probeFileSha256": _probe_file_sha256(),
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - system Python 3.10
        "provenance": {
            "ompBinary": str(omp_path),
            "ompVersion": omp_version,
            "pythonVersion": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "config": {
            "models": models,
            "runs": runs,
            "maxTokens": max_tokens,
            "par": par,
            "serviceTier": service_tier,
            "prompt": prompt,
            "promptSource": prompt_source,
            "promptFile": prompt_file,
            "promptSha256": prompt_sha256,
            "timeoutSeconds": timeout_seconds,
        },
        "command": command,
        "execution": {
            "exitCode": bench_exit_code,
            "durationSeconds": round(bench_duration, 3),
            "timedOut": timed_out,
            "stderr": bench_stderr[-4096:] if bench_stderr else "",
        },
        "warnings": _compute_warnings(raw, max_tokens),
        "raw": raw,
        "metrics": _extract_metrics(raw),
        "failures": raw.get("failures", 0),
    }


# ─── output ───────────────────────────────────────────────────────────


def write_output(envelope: dict[str, Any], output_path: str | None) -> None:
    """Write the envelope as pretty JSON.

    If *output_path* is None, writes to stdout.  Otherwise writes
    atomically (temp file + os.replace) to the given path, creating
    parent directories as needed.
    """
    json_str = json.dumps(envelope, indent=2, ensure_ascii=False)

    if output_path is None:
        sys.stdout.write(json_str + "\n")
        return

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent),
        prefix=p.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="throughput_probe",
        description=(
            "Reproducible OMP throughput probe — wraps `omp bench` with "
            "pinned controls and records TTFT/TPS plus provenance."
        ),
    )
    parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="OMP model selectors (positional, e.g. devin/glm-5-2 opus sonnet)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        metavar="N",
        help=f"Requests per model (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Max output tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--par",
        type=int,
        default=DEFAULT_PAR,
        metavar="N",
        help=f"Parallel requests per model (default: {DEFAULT_PAR})",
    )
    parser.add_argument(
        "--service-tier",
        default=DEFAULT_SERVICE_TIER,
        choices=SERVICE_TIER_CHOICES,
        metavar="VALUE",
        help=f"Service tier broadcast across families (default: {DEFAULT_SERVICE_TIER})",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="Read prompt from file instead of the embedded OMP bench prompt",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write envelope JSON to file (atomic). Default: stdout",
    )
    parser.add_argument(
        "--omp-bin",
        default=DEFAULT_OMP_BIN,
        metavar="PATH",
        help=f"OMP binary name or absolute path (default: {DEFAULT_OMP_BIN})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="N",
        help=f"Timeout for omp bench subprocess (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args(argv)

    # Validate numeric constraints.
    if args.runs <= 0:
        parser.error("--runs must be a positive integer")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be a positive integer")
    if args.par <= 0:
        parser.error("--par must be a positive integer")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive integer")

    # Reject blank and duplicate model selectors.
    seen: set[str] = set()
    for m in args.models:
        if not m.strip():
            parser.error("model selector cannot be blank")
        if m in seen:
            parser.error(f"duplicate model selector: {m}")
        seen.add(m)

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        # 1. Resolve binary.
        omp_path = resolve_omp_binary(args.omp_bin)

        # 2. Capture version.
        omp_version = capture_omp_version(omp_path)

        # 3. Load prompt (embedded default or --prompt-file).
        prompt, prompt_source, prompt_file = load_prompt(args.prompt_file)

        # 4. Build command (list, never shell=True).
        command = build_bench_command(
            omp_path,
            args.models,
            args.runs,
            args.max_tokens,
            args.par,
            args.service_tier,
            prompt,
        )

        # 5. Execute omp bench.
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start
            raise ProbeError(
                f"omp bench timed out after {args.timeout_seconds}s "
                f"(elapsed {duration:.1f}s)"
            ) from e
        except OSError as e:
            raise ProbeError(f"failed to execute omp bench: {e}") from e

        duration = time.monotonic() - start

        # 6. Parse JSON from stdout.
        stdout = result.stdout.strip()
        if not stdout:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            raise ProbeError(
                f"omp bench produced no stdout (exit code {result.returncode})"
                + (f"\nstderr: {stderr_tail}" if stderr_tail else "")
            )

        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ProbeError(
                f"failed to parse omp bench JSON: {e}\n"
                f"stdout (first 500 chars): {stdout[:500]}"
            ) from e

        # 7. Validate the JSON structure against expected configuration.
        validate_bench_summary(raw, args.models, args.runs, args.max_tokens)

        # 8. Build the reproducibility envelope.
        envelope = build_envelope(
            omp_path=omp_path,
            omp_version=omp_version,
            models=args.models,
            runs=args.runs,
            max_tokens=args.max_tokens,
            par=args.par,
            service_tier=args.service_tier,
            prompt=prompt,
            prompt_source=prompt_source,
            prompt_file=prompt_file,
            timeout_seconds=args.timeout_seconds,
            command=command,
            raw=raw,
            bench_exit_code=result.returncode,
            bench_duration=duration,
            bench_stderr=result.stderr,
            timed_out=False,
        )

        # 9. Write output (atomic if --output).
        write_output(envelope, args.output)

        # Print resolved output path to stderr only when --output is used.
        if args.output is not None:
            sys.stderr.write(f"{Path(args.output).resolve()}\n")

        # 10. Determine exit code.
        failures = raw.get("failures", 0)
        if failures > 0 or result.returncode != 0:
            return EXIT_BENCH_FAILURES
        return EXIT_OK

    except ProbeError as e:
        sys.stderr.write(f"throughput_probe: error: {e}\n")
        return e.exit_code
    except Exception as e:
        sys.stderr.write(f"throughput_probe: unexpected error: {type(e).__name__}: {e}\n")
        return EXIT_PROBE_ERROR


if __name__ == "__main__":
    sys.exit(main())
