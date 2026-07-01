"""Aggregator: runs every lanes/lane_NN.py module, emits METRIC lines.

Primary metric: lanes_with_signal = count of metrics producing a finite,
non-NaN float. Each lane's metric(s) emitted as secondary METRIC lines.

Lanes may return either a single float (via name() + measure()) or a
dict[str, float] (via measure()) for multi-metric reporting. The
aggregator handles both forms transparently.

Designed for autoresearch Phase 2: each iter that wires up a previously-NaN
lane increments lanes_with_signal (monotone optimization target).
"""
from __future__ import annotations

import importlib
import math
import sys
import traceback

SPEC_THRESHOLD = 0.75


def _run_lane(module_name: str) -> list[tuple[str, float]]:
    """Import lanes.<module_name>, call measure(). Return [(name, value), ...].

    If measure() returns a dict[str, float], each key/value pair becomes a
    separate metric entry. If it returns a single float, name() provides the
    metric name (legacy single-metric lanes).

    On any failure, returns [(module_name, float('nan'))] so the aggregator
    still counts the lane as "no signal" rather than crashing the harness.
    """
    try:
        mod = importlib.import_module(f"lanes.{module_name}")
        result = mod.measure()
        if isinstance(result, dict):
            entries: list[tuple[str, float]] = []
            for k, v in result.items():
                fv = float(v)
                entries.append((str(k), float("nan") if math.isnan(fv) or math.isinf(fv) else fv))
            return entries
        value = float(result)
        if math.isnan(value) or math.isinf(value):
            return [(mod.name(), float("nan"))]
        return [(mod.name(), value)]
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return [(module_name, float("nan"))]


def main() -> int:
    lane_modules = [
        "lane_01",
        "lane_02",
        "lane_03",
        "lane_04",
        "lane_05",
        "lane_06",
        "lane_07",
        "lane_08",
    ]

    results: list[tuple[str, float]] = []
    for m in lane_modules:
        results.extend(_run_lane(m))

    # Primary metric: count of lanes with finite, non-NaN signal.
    lanes_with_signal = sum(1 for _, v in results if not math.isnan(v))
    print(f"METRIC lanes_with_signal={lanes_with_signal}")

    # Spec-threshold metric: count of lanes at or above the spec threshold.
    lanes_above_spec = sum(1 for _, v in results if not math.isnan(v) and v >= SPEC_THRESHOLD)
    print(f"METRIC lanes_above_spec={lanes_above_spec}")

    # Secondary metrics: each lane's own value (or nan).
    for name, value in results:
        if math.isnan(value):
            print(f"METRIC {name}=nan")
        else:
            print(f"METRIC {name}={value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
