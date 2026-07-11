"""Generic remote experiment module runner.

Executes a Python module as ``[sys.executable, '-m', module, *args]`` with an
explicit argv (no ``shell=True``), captures stdout/stderr without deadlock,
mirrors both streams to sibling log files, parses ``METRIC name=value`` and
``ASI name=value`` lines, and writes a structured JSON execution report.

The report is **always** written — on success, nonzero exit, import error, or
timeout — so downstream provenance collection never loses the run record.
Scientific acceptance is never inferred from the exit code alone: consumers
must inspect ``status``, ``metrics``, and ``asi_scores``.

Designed to run from extracted source without installation.  The caller
(Kaggle bootstrap, Colab bootstrap, or scheduler) is responsible for adding
the repository ``src`` and workspace package ``*/src`` directories to
``sys.path``/``PYTHONPATH`` before invoking this runner so that ``-m module``
resolves src-layout packages remotely.

Importable both as ``infrastructure.kaggle.run_experiment_module`` and
directly as a script (``python run_experiment_module.py ...``) — no relative
imports.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "oczy/execution-report/v1"

# Lines must EXACTLY match these patterns (no leading/trailing junk).
# ``name`` is a Python-style identifier; ``value`` is an int or float.
_METRIC_RE = re.compile(r"^METRIC\s+([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")
_ASI_RE = re.compile(r"^ASI\s+([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")

_DEFAULT_REPORT = "execution_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_metric_line(line: str) -> tuple[str, float] | None:
    """Parse a ``METRIC name=value`` line, returning (name, value) or None."""
    stripped = line.rstrip("\r\n")
    m = _METRIC_RE.match(stripped)
    if m is None:
        return None
    return m.group(1), float(m.group(2))


def _parse_asi_line(line: str) -> tuple[str, float] | None:
    """Parse an ``ASI name=value`` line, returning (name, value) or None."""
    stripped = line.rstrip("\r\n")
    m = _ASI_RE.match(stripped)
    if m is None:
        return None
    return m.group(1), float(m.group(2))

# Maximum bytes of stderr/stdout inlined into the report error block.
# Keeps the sentinel JSON bounded so it stays a single manageable line
# and does not exfiltrate megabytes of child output.
_DIAGNOSTIC_MAX_BYTES = 8192

# Conservative secret redaction: catches ``KEY=value`` / ``key: value`` forms
# for common credential names.  Deliberately narrow to avoid mangling
# stack traces; false negatives are acceptable, false positives are not.
_SECRET_RE = re.compile(
    r"(?i)((?:api[_-]?key|auth[_-]?token|access[_-]?token|secret|password|passwd"
    r"|credential|authorization)\s*[:=]\s*)\S+"
)


def _redact_secrets(text: str) -> str:
    """Replace obvious credential values in *text* with ``[REDACTED]``."""
    return _SECRET_RE.sub(r"\1[REDACTED]", text)


def _read_bounded_tail(path: Path, max_bytes: int = _DIAGNOSTIC_MAX_BYTES) -> str | None:
    """Read the last *max_bytes* bytes of *path* as text, or None if empty.

    On files larger than *max_bytes*, reads from the offset and discards the
    partial first line so the result always starts on a line boundary.
    Returns None when the file is missing, empty, or contains only whitespace.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    offset = max(0, size - max_bytes)
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="replace")
    if offset > 0:
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1:]
    text = _redact_secrets(text)
    return text if text.strip() else None


def _stream_reader(
    stream: Any,
    log_file: Any,
    metrics: dict[str, float],
    asi_scores: dict[str, float],
) -> None:
    """Read a text stream line-by-line, mirror to log file, parse METRIC/ASI.

    Runs in a background thread so stdout and stderr are drained concurrently,
    preventing pipe-buffer deadlock on large output.
    """
    while True:
        line = stream.readline()
        if not line:
            break
        log_file.write(line)
        log_file.flush()
        metric = _parse_metric_line(line)
        if metric is not None:
            metrics[metric[0]] = metric[1]
            continue
        asi = _parse_asi_line(line)
        if asi is not None:
            asi_scores[asi[0]] = asi[1]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_module(
    *,
    module: str,
    arguments: list[str],
    source_commit: str,
    provider: str,
    job_name: str,
    report_path: Path | str = _DEFAULT_REPORT,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute ``python -m module *arguments`` and write a structured report.

    Parameters
    ----------
    module:
        Dotted module path passed to ``python -m``.
    arguments:
        Explicit argv list forwarded verbatim to the child.  Order preserved.
    source_commit:
        40-character Git commit SHA recorded for provenance.  Not used to
        checkout code — the caller arranges the source tree.
    provider:
        Provider label (``"kaggle"`` or ``"colab"``) recorded in the report.
    job_name:
        Human-readable job identifier; used to name the mirror log files.
    report_path:
        Path to the structured JSON report (default ``execution_report.json``).
    timeout:
        Optional wall-clock timeout in seconds.  On timeout the child is
        killed and the report is written with ``status="timeout"``.

    Returns
    -------
    dict
        The execution report dict (also written to ``report_path``).
    """
    report_path = Path(report_path)
    log_base = report_path.stem
    log_dir = report_path.parent
    stdout_log = log_dir / f"{log_base}.stdout.log"
    stderr_log = log_dir / f"{log_base}.stderr.log"

    command = [sys.executable, "-m", module, *arguments]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_name": job_name,
        "provider": provider,
        "source_commit": source_commit,
        "module": module,
        "arguments": list(arguments),
        "command": list(command),
        "status": "starting",
        "exit_code": None,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "metrics": {},
        "asi_scores": {},
        "stdout_file": stdout_log.name,
        "stderr_file": stderr_log.name,
        "timeout_seconds": timeout,
        "error": None,
    }

    # Write the initial report so provenance exists even if we crash before
    # the child finishes.
    _write_report(report_path, report)

    metrics: dict[str, float] = {}
    asi_scores: dict[str, float] = {}

    try:
        stdout_fh = stdout_log.open("w", encoding="utf-8")
        stderr_fh = stderr_log.open("w", encoding="utf-8")
    except OSError as exc:
        report["status"] = "error"
        report["exit_code"] = 1
        report["finished_utc"] = _utc_now()
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(report_path, report)
        _emit_sentinel(report)
        return report

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        # Launch failure (e.g. bad sys.executable) — write report and return.
        stdout_fh.close()
        stderr_fh.close()
        report["status"] = "error"
        report["exit_code"] = 1
        report["finished_utc"] = _utc_now()
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(report_path, report)
        _emit_sentinel(report)
        return report

    stdout_thread = threading.Thread(
        target=_stream_reader,
        args=(proc.stdout, stdout_fh, metrics, asi_scores),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_reader,
        args=(proc.stderr, stderr_fh, metrics, asi_scores),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Kill the child, then best-effort reap.  Exceptions from the
        # post-kill wait must never escape — the report is always written
        # below so downstream provenance collection never loses the record.
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.wait()
            except Exception:
                pass

    stdout_thread.join(timeout=30)
    stderr_thread.join(timeout=30)
    stdout_fh.close()
    stderr_fh.close()

    exit_code = proc.returncode if proc.returncode is not None else -1

    if timed_out:
        status = "timeout"
        exit_code = -1
        error_block: dict[str, Any] = {
            "type": "TimeoutExpired",
            "message": f"child exceeded timeout of {timeout}s",
        }
    elif exit_code == 0:
        status = "complete"
        error_block = None
    else:
        status = "error"
        stderr_tail = _read_bounded_tail(stderr_log)
        stdout_tail: str | None = None
        if stderr_tail is None:
            # stderr was empty — fall back to stdout so the report still
            # carries an actionable diagnostic (some modules write errors
            # to stdout, e.g. argparse usage on --help failures).
            stdout_tail = _read_bounded_tail(stdout_log)
        error_block = {
            "type": "NonzeroExit",
            "message": f"child exited with code {exit_code}",
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
        }

    report.update(
        {
            "status": status,
            "exit_code": exit_code,
            "finished_utc": _utc_now(),
            "metrics": dict(metrics),
            "asi_scores": dict(asi_scores),
            "error": error_block,
        }
    )
    _write_report(report_path, report)
    _emit_sentinel(report)
    return report


def _emit_sentinel(report: dict[str, Any]) -> None:
    """Emit a single-line report sentinel to the runner's own stdout.

    ``OCZY_EXECUTION_REPORT_JSON=<compact-json>``

    Downstream providers (notably Colab) that cannot download artifacts from
    the VM can recover the full structured report from the collected
    ``stdout.log``.  The JSON is compact (no newlines) so the sentinel is
    exactly one line.  The child's stdout/stderr are mirrored separately to
    log files and are not affected.
    """
    compact = json.dumps(report, sort_keys=True, separators=(",", ":"))
    sys.stdout.write(f"OCZY_EXECUTION_REPORT_JSON={compact}\n")
    sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic remote experiment module runner.",
    )
    parser.add_argument(
        "--module",
        required=True,
        help="Dotted module path to execute via `python -m`.",
    )
    parser.add_argument(
        "--arg",
        dest="arguments",
        action="append",
        default=[],
        help="Argument forwarded verbatim to the child. Repeatable; order preserved.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="40-character Git commit SHA recorded for provenance.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider label (kaggle or colab) recorded in the report.",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="Human-readable job identifier; used for log filenames.",
    )
    parser.add_argument(
        "--report",
        default=_DEFAULT_REPORT,
        help=f"Path to the structured JSON report (default: {_DEFAULT_REPORT}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_module(
        module=args.module,
        arguments=args.arguments,
        source_commit=args.source_commit,
        provider=args.provider,
        job_name=args.job_name,
        report_path=Path(args.report),
        timeout=args.timeout,
    )
    print(
        f"execution_report: {args.report} status={report['status']} "
        f"exit_code={report['exit_code']}",
        file=sys.stderr,
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
