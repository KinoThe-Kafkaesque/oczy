#!/usr/bin/env python3
"""A/B harness for the OrganismAgent scope-slot reranker strategies.

Launches one ``run_curriculum`` subprocess per reranker config (in parallel),
waits for them all to finish, then reads the resulting JSON reports and prints
a comparison table of Stage-5 (Cross-domain) metrics across configs.

Uses only the standard library.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Report directory is the same default used by run_curriculum.py.
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Each entry: (name, JSON config dict passed to OrganismAgent via --config).
CONFIGS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("topk3", {"scope_rerank_topk": 3}),
    ("sense_split", {"scope_rerank_sense_split": True}),
    ("multi_label", {"scope_rerank_multi_label": True}),
    (
        "combined",
        {
            "scope_rerank_topk": 3,
            "scope_rerank_sense_split": True,
            "scope_rerank_multi_label": True,
        },
    ),
]


def _report_path(name: str) -> Path:
    return REPORTS_DIR / f"run_rerank_{name}.json"


def _stage5(report: dict) -> dict | None:
    """Return the stage whose name contains 'Cross-domain', else None."""
    for stage in report.get("stages", []):
        if "Cross-domain" in stage.get("name", ""):
            return stage
    return None


def _episodes(stage: dict) -> list[dict]:
    """Accept both the documented 'episode_results' and the actual 'episodes'."""
    return stage.get("episode_results") or stage.get("episodes") or []


def _count_fixed(stage: dict) -> int:
    return sum(1 for ep in _episodes(stage) if ep.get("fixed"))


def _scope_accuracy(stage: dict) -> float:
    """Extract the scope pre-accuracy scalar for the stage.

    The report stores ``pre_accuracy`` as a dict with ``retention`` and
    ``scope`` keys.  We return ``scope`` specifically because it measures
    proactive cross-domain disambiguation — the quantity the reranker targets.
    """
    pre = stage.get("pre_accuracy")
    if pre is None:
        return 0.0
    if isinstance(pre, dict):
        return float(pre.get("scope", 0.0))
    if isinstance(pre, (int, float)):
        return float(pre)
    return 0.0


def _total_memory(report: dict) -> int:
    """Final memory footprint = last stage's memory_bytes_after."""
    stages = report.get("stages", [])
    if not stages:
        return 0
    return int(stages[-1].get("memory_bytes_after", 0) or 0)


def _launch(name: str, config: dict) -> subprocess.Popen[bytes]:
    """Launch one curriculum run for the given config; return the Popen handle."""
    config_json = json.dumps(config)
    report_name = f"run_rerank_{name}.json"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "oczy.experiments.organism_curriculum.run_curriculum",
        "--no-validate",
        "--use-real-driver",
        "--semantic",
        "--config",
        config_json,
        "--report-name",
        report_name,
    ]
    # Stream each subprocess's output to its own log so parallel runs never
    # interleave or deadlock on a shared pipe.
    log_path = REPORTS_DIR / f"run_rerank_{name}.log"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    print("Launching [%s] -> %s" % (name, " ".join(cmd)))
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)


def _read_report(name: str) -> dict | None:
    path = _report_path(name)
    if not path.exists():
        print("WARNING: report missing for config '%s': %s" % (name, path))
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print("WARNING: could not read report for '%s': %s" % (name, exc))
        return None


def main() -> int:
    # 1. Launch every config in parallel.
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    for name, config in CONFIGS:
        procs.append((name, _launch(name, config)))

    # 2. Wait for all subprocesses to complete.
    failed: list[str] = []
    for name, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append(name)
            print("ERROR: [%s] exited with code %d" % (name, rc))
        else:
            print("Done    [%s]" % name)

    # 3. Read reports and extract Stage-5 metrics.
    rows: list[tuple[str, int, float, int]] = []
    for name, _ in CONFIGS:
        report = _read_report(name)
        if report is None:
            rows.append((name, -1, 0.0, 0))
            continue
        stage = _stage5(report)
        if stage is None:
            print("WARNING: no Stage 5 (Cross-domain) found for '%s'" % name)
            rows.append((name, -1, 0.0, _total_memory(report)))
            continue
        rows.append(
            (
                name,
                _count_fixed(stage),
                _scope_accuracy(stage),
                _total_memory(report),
            )
        )

    # 4. Print the comparison table.
    print("")
    print("=== Reranker A/B comparison ===")
    header = "%-14s %14s %16s %12s" % ("Config", "Stage5 Fixed", "Stage5 Scope", "Total Mem")
    print(header)
    print("-" * len(header))
    for name, fixed, post_acc, total_mem in rows:
        print(
            "%-14s %14d %16.3f %12d"
            % (name, fixed, post_acc, total_mem)
        )

    if failed:
        print("\nFailed configs: %s" % ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
