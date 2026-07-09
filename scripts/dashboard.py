#!/usr/bin/env python3
"""Auto-generated headline dashboard from report JSONs.

Usage:
    uv run python scripts/dashboard.py          # regenerate DASHBOARD.md
    uv run python scripts/dashboard.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Return the repo root (parent of the scripts/ directory)."""
    return Path(__file__).resolve().parent.parent


def _reports_dir() -> Path:
    return _repo_root() / "src" / "oczy" / "experiments" / "organism_curriculum" / "reports"


def _dashboard_path() -> Path:
    return _repo_root() / "experiments_logs" / "DASHBOARD.md"


def _is_multi_seed(report: dict[str, Any]) -> bool:
    """Return True if this looks like a multi-seed report."""
    return "aggregated" in report and "per_seed" in report


def _discover_reports(reports_dir: Path) -> list[tuple[Path, dict[str, Any], float]]:
    """Scan reports dir, return (path, payload, mtime_epoch) for all valid JSON files."""
    out: list[tuple[Path, dict[str, Any], float]] = []
    if not reports_dir.is_dir():
        return out
    for p in sorted(reports_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mtime = p.stat().st_mtime
        out.append((p, payload, mtime))
    return out


def _pick_newest(reports: list[tuple[Path, dict[str, Any], float]]) -> list[tuple[Path, dict[str, Any], float]]:
    """Pick the newest report per (agent, use_lm, split/n_seeds) combination."""
    # Group single-seed by (agent, use_lm, split)
    # Group multi-seed by (agent, use_lm, n_seeds)
    groups: dict[tuple, tuple[Path, dict[str, Any], float]] = {}
    for p, payload, mtime in reports:
        agent = payload.get("agent", "unknown")
        use_lm = payload.get("use_lm", False)
        if _is_multi_seed(payload):
            key = (agent, use_lm, "multi", payload.get("n_seeds", 0))
        else:
            split = payload.get("split", "unknown")
            key = (agent, use_lm, "single", split)
        if key not in groups or mtime > groups[key][2]:
            groups[key] = (p, payload, mtime)
    return list(groups.values())


def _compute_vanilla_col(vanilla_stages: list[dict[str, Any]] | None, stage_name: str) -> str:
    """Extract vanilla accuracy for a stage name, or return '-' if absent."""
    if not vanilla_stages:
        return "-"
    for vs in vanilla_stages:
        if vs.get("name") == stage_name:
            post_acc = vs.get("post_accuracy", {})
            if isinstance(post_acc, dict):
                # Compute mean of category accuracies
                vals = [v for v in post_acc.values() if isinstance(v, (int, float))]
                if vals:
                    return "%.2f" % (sum(vals) / len(vals))
            return "-"
    return "-"


def _fmt_bytes(n: int) -> str:
    """Format a byte delta as a signed human-readable string."""
    if n >= 0:
        return "+%sB" % _human_bytes(n)
    return "-%sB" % _human_bytes(abs(n))


def _human_bytes(n: int) -> str:
    """Format a positive integer as human-readable with commas."""
    if n >= 1_000_000:
        return f"{n:,d}"
    return str(n)


def _generate_single_seed_section(
    report_path: Path,
    payload: dict[str, Any],
    mtime: float,
) -> str:
    """Generate a markdown section for a single-seed report."""
    agent = payload.get("agent", "unknown")
    use_lm = payload.get("use_lm", False)
    split = payload.get("split", "unknown")
    stages = payload.get("stages", [])
    vanilla = payload.get("vanilla")

    lm_label = "LM" if use_lm else "raw"
    ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append(f"## {agent} ({lm_label}, split={split})")
    lines.append("")
    lines.append(f"**Report:** `{report_path.name}` | **Backend:** {lm_label} | **Generated:** {ts}")
    lines.append("")

    has_vanilla = bool(vanilla)
    if has_vanilla:
        header = "| Stage | Uptake | Pre | Post | Vanilla | Mem Δ |"
        sep = "|---|---:|---:|---:|---:|---:|"
    else:
        header = "| Stage | Uptake | Pre | Post | Mem Δ |"
        sep = "|---|---:|---:|---:|---:|"

    lines.append(header)
    lines.append(sep)

    for stage in stages:
        name = stage.get("name", "?")
        uptake = stage.get("uptake_latency", 0.0)

        # Pre accuracy: mean of category accuracies
        pre_acc = stage.get("pre_accuracy", {})
        if isinstance(pre_acc, dict):
            pre_vals = [v for v in pre_acc.values() if isinstance(v, (int, float))]
            pre = sum(pre_vals) / len(pre_vals) if pre_vals else 0.0
        else:
            pre = 0.0

        post = float(stage.get("post_accuracy", 0.0))

        mem_before = int(stage.get("memory_bytes_before", 0))
        mem_after = int(stage.get("memory_bytes_after", 0))
        mem_delta = mem_after - mem_before

        if has_vanilla:
            v_col = _compute_vanilla_col(vanilla, name)
            lines.append(
                f"| {name} | {uptake:.2f} | {pre:.2f} | {post:.2f} | {v_col} | {_fmt_bytes(mem_delta)} |"
            )
        else:
            lines.append(
                f"| {name} | {uptake:.2f} | {pre:.2f} | {post:.2f} | {_fmt_bytes(mem_delta)} |"
            )

    lines.append("")
    return "\n".join(lines)


def _generate_multi_seed_section(
    report_path: Path,
    payload: dict[str, Any],
    mtime: float,
) -> str:
    """Generate a markdown section for a multi-seed report."""
    agent = payload.get("agent", "unknown")
    use_lm = payload.get("use_lm", False)
    n_seeds = payload.get("n_seeds", 0)
    aggregated = payload.get("aggregated", [])

    lm_label = "LM" if use_lm else "raw"
    ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append(f"## {agent} ({lm_label}, {n_seeds} seeds)")
    lines.append("")
    lines.append(f"**Report:** `{report_path.name}` | **Seeds:** {n_seeds} | **Backend:** {lm_label} | **Generated:** {ts}")
    lines.append("")

    lines.append(f"| Stage | Post accuracy (mean ± std, n={n_seeds}) |")
    lines.append("|---:|---|")

    for agg in aggregated:
        name = agg.get("name", "?")
        acc = agg.get("accuracy", {})
        mean = acc.get("mean", float("nan"))
        std = acc.get("std", float("nan"))
        n = int(acc.get("n", 0))
        lines.append(f"| {name} | {mean:.4f} ± {std:.4f} (n={n}) |")

    lines.append("")
    return "\n".join(lines)


def generate_dashboard(reports_dir: Path | None = None) -> str:
    """Generate full DASHBOARD.md content from reports.

    Returns the markdown string and writes DASHBOARD.md as a side effect.
    """
    if reports_dir is None:
        reports_dir = _reports_dir()

    all_reports = _discover_reports(reports_dir)
    picked = _pick_newest(all_reports)

    if not picked:
        newest_ts = datetime.now(timezone.utc)
    else:
        newest_ts = datetime.fromtimestamp(
            max(r[2] for r in picked), tz=timezone.utc
        )

    ts_fmt = newest_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED — do not hand-edit; regenerate with `uv run python scripts/dashboard.py` -->")
    lines.append(f"<!-- Generated: {ts_fmt} -->")
    lines.append("")
    lines.append("# Oczy Curriculum Dashboard")
    lines.append("")

    if not picked:
        lines.append("_No reports found in `%s`._" % reports_dir)
        return "\n".join(lines) + "\n"

    for report_path, payload, mtime in sorted(picked, key=lambda x: x[2], reverse=True):
        if _is_multi_seed(payload):
            lines.append(_generate_multi_seed_section(report_path, payload, mtime))
        else:
            lines.append(_generate_single_seed_section(report_path, payload, mtime))

    return "\n".join(lines) + "\n"


def regenerate(
    reports_dir: Path | None = None,
    dashboard_path: Path | None = None,
) -> Path:
    """Regenerate DASHBOARD.md and return the output path."""
    if reports_dir is None:
        reports_dir = _reports_dir()
    if dashboard_path is None:
        dashboard_path = _dashboard_path()

    content = generate_dashboard(reports_dir)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(content, encoding="utf-8")
    return dashboard_path


def check_staleness(
    reports_dir: Path | None = None,
    dashboard_path: Path | None = None,
) -> tuple[bool, str]:
    """Return (is_stale, reason_string).

    Stale means DASHBOARD.md doesn't exist or any picked report is newer.
    """
    if reports_dir is None:
        reports_dir = _reports_dir()
    if dashboard_path is None:
        dashboard_path = _dashboard_path()

    if not dashboard_path.exists():
        return True, "DASHBOARD.md does not exist"

    dash_mtime = dashboard_path.stat().st_mtime

    all_reports = _discover_reports(reports_dir)
    if not all_reports:
        # No reports at all → not stale (nothing to regenerate from)
        return False, "no reports found, DASHBOARD.md is current"

    picked = _pick_newest(all_reports)
    newest_report_mtime = max(r[2] for r in picked) if picked else 0.0

    if newest_report_mtime > dash_mtime:
        newest_name = None
        for p, _, mt in picked:
            if mt == newest_report_mtime:
                newest_name = p.name
                break
        ts = datetime.fromtimestamp(newest_report_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return True, (
            f"DASHBOARD.md is stale: newest report {newest_name} ({ts})"
            f" is newer than dashboard"
        )

    return False, "DASHBOARD.md is current"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the Oczy curriculum dashboard.")
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if DASHBOARD.md is stale relative to newest reports.",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Override the reports directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the DASHBOARD.md output path.",
    )
    args = p.parse_args(argv)

    reports_dir = args.reports_dir or _reports_dir()
    dashboard_path = args.output or _dashboard_path()

    if args.check:
        stale, reason = check_staleness(reports_dir, dashboard_path)
        if stale:
            print(reason, file=sys.stderr)
            return 1
        print(reason)
        return 0

    out = regenerate(reports_dir, dashboard_path)
    print("Dashboard written to: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
