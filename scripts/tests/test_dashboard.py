"""Tests for scripts/dashboard.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "dashboard.py"


def _run_dashboard(
    *args: str,
    reports_dir: Path | None = None,
    output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run dashboard.py as a subprocess with optional overrides."""
    cmd = [sys.executable, str(DASHBOARD_SCRIPT)]
    if reports_dir is not None:
        cmd.extend(["--reports-dir", str(reports_dir)])
    if output is not None:
        cmd.extend(["--output", str(output)])
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Synthetic report fixtures
# ---------------------------------------------------------------------------

_SINGLE_SEED_REPORT: dict = {
    "agent": "OrganismAgent",
    "use_lm": False,
    "split": "dev",
    "stages": [
        {
            "name": "Stage 0: Sense grounding",
            "description": "grounding probes",
            "memory_bytes_before": 1000,
            "memory_bytes_after": 2162672,
            "uptake_latency": 0.0,
            "pre_accuracy": {"sense": 0.0},
            "post_accuracy": 0.875,
            "episodes": [
                {
                    "id": "ep_0",
                    "initial_request": "hello",
                    "first_answer": "hi",
                    "second_answer": "hi",
                    "corrected_response": "hello",
                    "fixed": True,
                    "lm_parse_ok": None,
                }
            ],
        },
        {
            "name": "Stage 1: Transfer",
            "description": "transfer probes",
            "memory_bytes_before": 2162672,
            "memory_bytes_after": 2165912,
            "uptake_latency": 0.25,
            "pre_accuracy": {"transfer": 0.5},
            "post_accuracy": 0.75,
            "episodes": [],
        },
    ],
    "vanilla": [
        {
            "name": "Stage 0: Sense grounding",
            "description": "grounding probes",
            "post_accuracy": {"sense": 0.0},
        },
        {
            "name": "Stage 1: Transfer",
            "description": "transfer probes",
            "post_accuracy": {"transfer": 0.0},
        },
    ],
}

_MULTI_SEED_REPORT: dict = {
    "agent": "OrganismAgent",
    "use_lm": False,
    "n_seeds": 5,
    "per_seed": [[]],
    "aggregated": [
        {
            "name": "Stage 0: Sense grounding",
            "accuracy": {
                "n": 5,
                "mean": 0.625,
                "std": 0.0,
                "min": 0.625,
                "max": 0.625,
                "ci95_half": 0.0,
            },
        },
        {
            "name": "Stage 1: Transfer",
            "accuracy": {
                "n": 5,
                "mean": 0.625,
                "std": 0.0,
                "min": 0.625,
                "max": 0.625,
                "ci95_half": 0.0,
            },
        },
    ],
}


def _write_report(
    reports_dir: Path,
    name: str,
    payload: dict,
    mtime: float | None = None,
) -> Path:
    """Write a synthetic report JSON and optionally set its mtime."""
    p = reports_dir / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        os.utime(str(p), (mtime, mtime))
    return p


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


def test_single_seed_generates_table(tmp_path: Path) -> None:
    """A single-seed report produces a markdown table with expected columns."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    _write_report(reports_dir, "test_run.json", _SINGLE_SEED_REPORT)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "AUTO-GENERATED" in content
    assert "OrganismAgent (raw, split=dev)" in content
    assert "Stage 0: Sense grounding" in content
    assert "Stage 1: Transfer" in content
    assert "0.88" in content  # post accuracy
    assert "0.75" in content  # post accuracy
    assert "0.00" in content  # vanilla
    assert "+2,161,672B" in content  # mem delta


def test_multi_seed_generates_table(tmp_path: Path) -> None:
    """A multi-seed report produces a mean±std markdown table."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    _write_report(reports_dir, "multi_run.json", _MULTI_SEED_REPORT)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "AUTO-GENERATED" in content
    assert "OrganismAgent (raw, 5 seeds)" in content
    assert "mean ± std" in content
    assert "0.6250 ± 0.0000" in content
    assert "(n=5)" in content


def test_picks_newest_per_group(tmp_path: Path) -> None:
    """When multiple reports share a group key, the newest by mtime is picked."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    base_time = time.time()

    # Older report — should be ignored
    old = dict(_SINGLE_SEED_REPORT, **{"split": "dev"})
    old["stages"][0]["post_accuracy"] = 0.111
    _write_report(reports_dir, "old_run.json", old, mtime=base_time - 100)

    # Newer report — should be the one picked
    new = dict(_SINGLE_SEED_REPORT, **{"split": "dev"})
    new["stages"][0]["post_accuracy"] = 0.99
    _write_report(reports_dir, "new_run.json", new, mtime=base_time + 100)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "new_run.json" in content
    assert "old_run.json" not in content
    assert "0.99" in content
    assert "0.11" not in content


def test_different_splits_produce_separate_sections(tmp_path: Path) -> None:
    """Reports with different splits both appear in the dashboard."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    dev = dict(_SINGLE_SEED_REPORT, **{"split": "dev"})
    holdout = dict(_SINGLE_SEED_REPORT, **{"split": "holdout"})

    _write_report(reports_dir, "dev_run.json", dev)
    _write_report(reports_dir, "holdout_run.json", holdout)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "split=dev" in content
    assert "split=holdout" in content


def test_empty_reports_dir_produces_placeholder(tmp_path: Path) -> None:
    """An empty reports dir produces a placeholder message, not a crash."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "AUTO-GENERATED" in content
    assert "No reports found" in content


def test_single_seed_without_vanilla(tmp_path: Path) -> None:
    """A single-seed report without vanilla column omits that column."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    no_vanilla = dict(_SINGLE_SEED_REPORT)
    del no_vanilla["vanilla"]
    _write_report(reports_dir, "no_vanilla.json", no_vanilla)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "Vanilla" not in content


# ---------------------------------------------------------------------------
# --check staleness tests
# ---------------------------------------------------------------------------


def test_check_stale_when_dashboard_missing(tmp_path: Path) -> None:
    """--check exits 1 when DASHBOARD.md does not exist."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"
    _write_report(reports_dir, "test.json", _SINGLE_SEED_REPORT)

    result = _run_dashboard(
        "--check",
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_check_current_when_no_reports(tmp_path: Path) -> None:
    """--check exits 0 when no reports exist and DASHBOARD.md is present."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"
    dash_path.write_text("stale\n", encoding="utf-8")

    result = _run_dashboard(
        "--check",
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0


def test_check_stale_when_report_newer(tmp_path: Path) -> None:
    """--check exits 1 when a report is newer than DASHBOARD.md."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    base_time = time.time()

    # Write DASHBOARD.md first (older)
    dash_path.write_text("old dashboard\n", encoding="utf-8")
    os.utime(str(dash_path), (base_time - 200, base_time - 200))

    # Write report second (newer)
    _write_report(reports_dir, "fresh.json", _SINGLE_SEED_REPORT, mtime=base_time)

    result = _run_dashboard(
        "--check",
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 1
    assert "stale" in result.stderr.lower()


def test_check_current_when_dashboard_newer(tmp_path: Path) -> None:
    """--check exits 0 when DASHBOARD.md is newer than all reports."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    base_time = time.time()

    # Write report first (older)
    _write_report(reports_dir, "old.json", _SINGLE_SEED_REPORT, mtime=base_time - 200)

    # Write DASHBOARD.md second (newer)
    dash_path.write_text("current dashboard\n", encoding="utf-8")
    os.utime(str(dash_path), (base_time, base_time))

    result = _run_dashboard(
        "--check",
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0
    assert "current" in result.stdout.lower()


def test_mixed_multi_and_single(tmp_path: Path) -> None:
    """Both single-seed and multi-seed reports appear in the dashboard."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    dash_path = tmp_path / "DASHBOARD.md"

    _write_report(reports_dir, "single.json", _SINGLE_SEED_REPORT)
    _write_report(reports_dir, "multi.json", _MULTI_SEED_REPORT)

    result = _run_dashboard(
        reports_dir=reports_dir,
        output=dash_path,
    )
    assert result.returncode == 0, f"dashboard failed: {result.stderr}"

    content = dash_path.read_text(encoding="utf-8")
    assert "split=dev" in content
    assert "5 seeds" in content
    # Should have two sections
    assert content.count("## OrganismAgent") == 2
