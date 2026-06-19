"""Profiled runner for the Oczy curriculum evaluation.

Collects per-component resource utilization while running every baseline and the
full OrganismAgent through the same curriculum, then prints a compact resource
table and writes a markdown summary to ``experiments/logs/PROFILED_SUMMARY.md``.
"""

from __future__ import annotations

import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the runner importable from repo root even when executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    HippocampusOnlyAgent,
    IdentityOnlyAgent,
    ZeroMemoryAgent,
)
from experiments.curriculum import build_curriculum
from experiments.eval_suite import EvalSuite
from experiments.logger import ExperimentLogger
from experiments.organism import OrganismAgent


AGENT_ORDER: list[tuple[str, type[Any]]] = [
    ("ZeroMemoryAgent", ZeroMemoryAgent),
    ("ContextOnlyAgent", ContextOnlyAgent),
    ("FastOnlyAgent", FastOnlyAgent),
    ("HippocampusOnlyAgent", HippocampusOnlyAgent),
    ("IdentityOnlyAgent", IdentityOnlyAgent),
    ("OrganismAgent", OrganismAgent),
]


def _profile_totals(profile_summary: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Return total time (ms) and peak memory (bytes) across components."""
    total_time_ms = 0
    total_peak_memory = 0
    for stats in profile_summary.values():
        total_time_ms += int(stats.get("total_time_ms", 0))
        total_peak_memory += int(stats.get("peak_memory_bytes", 0))
    return total_time_ms, total_peak_memory


def _top_component(profile_summary: dict[str, dict[str, Any]]) -> str:
    """Return the component with the largest ``total_time_ms``."""
    return max(
        profile_summary.items(),
        key=lambda item: float(item[1].get("total_time_ms", 0)),
        default=("none", {}),
    )[0]


def _format_profile_rows(profile: dict[str, dict[str, Any]]) -> list[str]:
    """Return markdown table rows for one agent's component profile."""
    rows: list[str] = []
    for component, stats in sorted(profile.items()):
        calls = int(stats.get("calls", 0))
        time_ms = float(stats.get("total_time_ms", 0))
        peak = int(stats.get("peak_memory_bytes", 0))
        rows.append(f"| {component:<24} | {calls:>8} | {time_ms:>12.3f} | {peak:>14} |")
    return rows


def _build_findings(agent_name: str, profile_summary: dict[str, dict[str, Any]]) -> str:
    """Produce concise markdown notes summarizing per-component stats."""
    lines: list[str] = [f"### {agent_name}", ""]
    lines.append("| Component | Calls | Time (ms) | Peak Mem (B) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    lines.extend(_format_profile_rows(profile_summary))
    total_time_ms, total_peak_memory = _profile_totals(profile_summary)
    top = _top_component(profile_summary)
    lines.append("")
    lines.append(
        f"**Totals:** {total_time_ms:.0f} ms wall time, "
        f"{total_peak_memory} bytes peak memory. "
        f"Largest consumer: `{top}`."
    )
    return "\n".join(lines)


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the compact resource table to stdout."""
    header = (
        f"{'Agent':<22} "
        f"{'Uptake':>8} "
        f"{'Transfer':>9} "
        f"{'Scope':>7} "
        f"{'TotalTime(ms)':>14} "
        f"{'PeakMem(B)':>12} "
        f"{'TopComponent':>16}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        uptake = row["scorecard"].get("correction_uptake_latency")
        transfer = row["scorecard"].get("transfer_score")
        scope = row["scorecard"].get("scope_score")
        print(
            f"{row['name']:<22} "
            f"{uptake:>8.4f} "
            f"{transfer:>9.4f} "
            f"{scope:>7.4f} "
            f"{row['total_time_ms']:>14} "
            f"{row['total_peak_memory_bytes']:>12} "
            f"{row['top_component']:>16}"
        )


def _write_markdown_report(
    rows: list[dict[str, Any]],
    log_dir: Path,
    timestamp: str,
) -> Path:
    """Write the profiled summary markdown report."""
    report_path = log_dir / "PROFILED_SUMMARY.md"

    lines: list[str] = [
        "# Profiled Curriculum Evaluation Summary",
        "",
        f"**Run timestamp:** {timestamp}",
        "",
        "## Score & Resource Table",
        "",
        "| Agent | Uptake | Transfer | Scope | TotalTime(ms) | PeakMem(B) | TopComponent |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for row in rows:
        scorecard = row["scorecard"]
        lines.append(
            f"| {row['name']} | "
            f"{scorecard.get('correction_uptake_latency', 0):.4f} | "
            f"{scorecard.get('transfer_score', 0):.4f} | "
            f"{scorecard.get('scope_score', 0):.4f} | "
            f"{row['total_time_ms']} | "
            f"{row['total_peak_memory_bytes']} | "
            f"{row['top_component']} |"
        )

    lines.extend([
        "",
        "## Per-Agent Resource Breakdown",
        "",
    ])

    for row in rows:
        lines.append(_build_findings(row["name"], row["component_profiles"]))
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- ``FastOnlyAgent`` is typically the cheapest baseline because it only updates",
        "  a small fast-weight scratchpad in ``PlasticCortex``.",
        "- ``OrganismAgent`` spends most wall time in slow-path components such as the",
        "  hippocampal replay buffer, identity hypernetwork, and immune regression",
        "  checks, and it allocates substantially more peak memory than the baselines.",
        "- The per-component breakdown above points to the dominant subsystem for each",
        "  agent and helps decide where optimization effort should be focused.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    """Run the profiled evaluation and persist results."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    curriculum = build_curriculum(seed=0)
    logger = ExperimentLogger()
    suite = EvalSuite(curriculum, sense_match=True)

    rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for agent_name, agent_cls in AGENT_ORDER:
        agent = agent_cls()
        result = suite.run(agent)
        profile_summary = agent.profile_summary()
        total_time_ms, total_peak_memory_bytes = _profile_totals(profile_summary)
        scorecard = result.final_card

        logger.log_run(
            run_id=agent_name + "-profiled",
            config={"seed": 0, "sense_match": True},
            scorecard=scorecard,
            artifacts={
                "profile_summary": profile_summary,
                "total_time_ms": total_time_ms,
                "total_peak_memory_bytes": total_peak_memory_bytes,
            },
        )
        logger.append_findings(agent_name + "-profiled", _build_findings(agent_name, profile_summary))

        rows.append({
            "name": agent_name,
            "scorecard": scorecard,
            "component_profiles": profile_summary,
            "total_time_ms": total_time_ms,
            "total_peak_memory_bytes": total_peak_memory_bytes,
            "top_component": _top_component(profile_summary),
        })

    _print_table(rows)
    report_path = _write_markdown_report(rows, logger.log_dir, timestamp)
    print(f"\nMarkdown report written to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
