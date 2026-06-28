"""Aggregator: runs every lanes/lane_NN.py module, emits METRIC lines.

Primary metric: lanes_with_signal = count of lanes producing a finite, non-NaN float.
Each lane's own metric is emitted as a secondary METRIC line.

Designed for autoresearch Phase 2: each iter that wires up a previously-NaN lane
increments lanes_with_signal by 1 (monotone optimization target).
"""
from __future__ import annotations

import importlib
import math
import sys
import traceback


def _run_lane(module_name: str) -> tuple[str, float]:
    """Import lanes.<module_name>, call name() and measure(). Return (name, value).

    On any failure, returns (module_name, float('nan')) so the aggregator still
    counts the lane as "no signal" rather than crashing the whole harness.
    """
    try:
        mod = importlib.import_module(f"lanes.{module_name}")
        metric_name = mod.name()
        value = float(mod.measure())
        if math.isnan(value) or math.isinf(value):
            return metric_name, float("nan")
        return metric_name, value
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return module_name, float("nan")


def main() -> int:
    lane_modules = [
        "lane_01",
        "lane_02",
        "lane_03",
        "lane_04",
        "lane_05",
        "lane_06",
        "lane_07",
    ]

    results: list[tuple[str, float]] = []
    for m in lane_modules:
        results.append(_run_lane(m))

    # Primary metric: count of lanes with finite, non-NaN signal.
    lanes_with_signal = sum(1 for _, v in results if not math.isnan(v))
    print(f"METRIC lanes_with_signal={lanes_with_signal}")

    # Secondary metrics: each lane's own value (or nan).
    for name, value in results:
        if math.isnan(value):
            print(f"METRIC {name}=nan")
        else:
            print(f"METRIC {name}={value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())