"""Timestamped JSON logger for Oczy experiment runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Writes timestamped JSON scorecards to ``experiments/logs/``.

    Runs are stored as ``<ISO-timestamp>_<run_id>.json`` so they are
    naturally ordered and collision-resistant.
    """

    def __init__(self, log_dir: Path | str | None = None) -> None:
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        gitkeep = self.log_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    def _run_path(self, run_id: str, timestamp: str) -> Path:
        safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")
        return self.log_dir / f"{timestamp}_{safe_run_id}.json"

    def log_run(
        self,
        run_id: str,
        config: dict[str, Any],
        scorecard: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a single experimental run and return the written file path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        record: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": config,
            "scorecard": scorecard,
            "artifacts": artifacts or {},
            "findings": "",
        }
        path = self._run_path(run_id, timestamp)
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path

    def append_findings(self, run_id: str, findings: str) -> Path:
        """Append free-form markdown notes to the most recent run matching ``run_id``."""
        runs = [p for p in self.list_runs() if p.stem.endswith(f"_{run_id}")]
        if not runs:
            raise FileNotFoundError(f"No log file found for run_id={run_id!r}")
        # Sorted ascending; take latest.
        latest = sorted(runs)[-1]
        record = json.loads(latest.read_text(encoding="utf-8"))
        existing = record.get("findings", "")
        separator = "\n\n---\n\n" if existing else ""
        record["findings"] = existing + separator + findings
        latest.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return latest

    def list_runs(self) -> list[Path]:
        """Return all JSON log files in the log directory, sorted."""
        if not self.log_dir.exists():
            return []
        return sorted(p for p in self.log_dir.iterdir() if p.suffix == ".json")


if __name__ == "__main__":
    logger = ExperimentLogger()
    path = logger.log_run(
        run_id="smoke-test",
        config={"curriculum": "dummy", "seed": 0},
        scorecard={
            "correction_uptake_latency": 1.0,
            "transfer_score": 0.8,
            "scope_score": 0.75,
            "forgetting_score": 0.9,
            "consolidation_score": 0.85,
            "memory_bytes_per_behavior_delta": 128,
            "identity_drift": 0.02,
        },
        artifacts={"note": "dummy run"},
    )
    logger.append_findings(
        "smoke-test",
        "## Smoke-test findings\n\nLogger successfully wrote, re-read, and extended a run record.",
    )
    print(path.read_text(encoding="utf-8"))
    print("\n\nAll runs on disk:", [p.name for p in logger.list_runs()])
