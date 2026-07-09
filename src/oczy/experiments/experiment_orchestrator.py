"""Combined experiment orchestrator.

Runs all seven curriculum experiment modules, parses their headline METRIC
lines, and reports how many satisfy their acceptance thresholds.

Use:
    uv run python -m oczy.experiments.experiment_orchestrator [--driver real|mock]

The mock path skips heavy real-driver loads but still exercises import/wiring.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    module: str
    driver: str | None
    metric_name: str
    accepted: Callable[[float], bool]


_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "exp01_v2",
        "oczy.experiments.correction_competence_v2",
        "real",
        "v2_desaturation_count",
        lambda v: v >= 3.0,
    ),
    Experiment(
        "exp02_kv_slot",
        "oczy.experiments.kv_slot_injection",
        "real",
        "kv_slot_rank1_count",
        lambda v: v >= 2.0,
    ),
    Experiment(
        "exp03_layer_l",
        "oczy.experiments.layer_l_probe",
        "real",
        "layer_l_silhouette_gap",
        lambda v: v >= 0.10,
    ),
    Experiment(
        "exp04_scope",
        "oczy.experiments.scope_selectivity_stressor",
        "real",
        "scope_selectivity_index",
        lambda v: v >= 0.5,
    ),
    Experiment(
        "exp05_metabolism",
        "oczy.experiments.metabolism_loop",
        "real",
        "metabolism_drift_delta",
        lambda v: v > 0.0,
    ),
    Experiment(
        "exp06_bounded_growth",
        "oczy.experiments.bounded_growth.bounded_growth_eval",
        None,
        "bounded_growth_m1_ratio",
        lambda v: v <= 0.10,
    ),
    Experiment(
        "exp07_world_model",
        "oczy.experiments.conversation_world_model",
        "real",
        "marker_free_uptake_gap",
        lambda v: v >= 0.40,
    ),
)

_METRIC_RE = re.compile(r"^METRIC\s+([\w_]+)=([\d.eE+-]+|nan)$")


def _run_experiment(exp: Experiment) -> float:
    """Run one experiment module in a subprocess and return its headline metric."""
    cmd = ["uv", "run", "python", "-m", exp.module]
    if exp.driver is not None:
        cmd += ["--driver", exp.driver]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return float("nan")

    for line in proc.stdout.splitlines():
        m = _METRIC_RE.match(line.strip())
        if m and m.group(1) == exp.metric_name:
            try:
                return float(m.group(2))
            except ValueError:
                return float("nan")
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combined curriculum experiment orchestrator"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
        help="mock skips heavy LM loads; real runs all experiments with real drivers",
    )
    args = parser.parse_args(argv)

    results: dict[str, tuple[float, bool]] = {}
    for exp in _EXPERIMENTS:
        if args.driver == "mock" and exp.module == "oczy.experiments.scope_selectivity_stressor":
            # Scope module has no mock behavioral semantics; report 0.
            value = 0.0
        else:
            value = _run_experiment(exp)
        ok = exp.accepted(value) if not math.isnan(value) else False
        results[exp.name] = (value, ok)

    accepted_count = sum(1 for _, ok in results.values() if ok)
    total = len(_EXPERIMENTS)

    print(f"METRIC experiments_accepted_count={accepted_count}")
    print(f"ASI experiments_total={total}")
    for exp in _EXPERIMENTS:
        value, ok = results[exp.name]
        print(f"ASI {exp.metric_name}={value}")
        print(f"ASI {exp.name}_accepted={int(ok)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
