"""Statistical helpers for multi-seed experiment reporting.

Every reported metric should carry n, mean, std, and a CI, never
a bare point estimate.  This module provides the building blocks.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable

# ---------------------------------------------------------------------------
# t-distribution critical values for 95% CI (two-tailed, alpha=0.05).
# Covers n-1 up to 30 then a few large-n values + infinity (1.96).
# ---------------------------------------------------------------------------
_T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980, 10**9: 1.960,
}


def _t_critical(df: int) -> float:
    """Return the two-tailed 95% t-critical value for ``df`` degrees of freedom.

    Tries scipy.stats first (most accurate), then a hard-coded table,
    and falls back to the normal approximation (1.96).
    """
    try:
        from scipy.stats import t  # type: ignore[import-untyped]
    except ImportError:
        pass
    else:
        return float(t.ppf(0.975, df))

    for threshold, val in sorted(_T95.items()):
        if df <= threshold:
            return val
    return _T95[10**9]


def summarize(values: list[float]) -> dict[str, float]:
    """Return summary statistics for a list of numeric observations.

    Returns a dict with keys:
        n, mean, std, min, max, ci95_half

    ``ci95_half`` is the half-width of a two-sided 95% confidence interval
    for the mean (using the t-distribution when possible, falling back to
    a normal approximation).

    For ``n == 1``, ``std`` is reported as 0.0 (population-safe) and
    ``ci95_half`` is ``nan`` (a CI requires at least two observations).
    """
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "ci95_half": float("nan"),
        }

    m = statistics.mean(values)
    s = statistics.stdev(values) if n >= 2 else 0.0
    lo = min(values)
    hi = max(values)

    if n <= 1:
        ci = float("nan")
    else:
        sem = s / math.sqrt(n)
        ci = _t_critical(n - 1) * sem

    return {"n": n, "mean": m, "std": s, "min": lo, "max": hi, "ci95_half": ci}


def run_seeded(
    fn: Callable[[int], float | dict[str, float]],
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    """Run ``fn(seed)`` once per seed and return per-key summaries.

    ``fn`` must accept a single ``int`` seed argument.  It may return:

    * a ``float`` — treated as a single metric keyed ``"_all"``.
    * a ``dict[str, float]`` — each key becomes an independent metric.

    Returns a dict mapping each metric key to the output of
    :func:`summarize` for the collected per-seed values.
    """
    if not seeds:
        return {}

    per_key: dict[str, list[float]] = {}
    for s in seeds:
        result = fn(s)
        if isinstance(result, (int, float)):
            per_key.setdefault("_all", []).append(float(result))
        else:
            for k, v in result.items():
                per_key.setdefault(k, []).append(v)

    return {k: summarize(vals) for k, vals in per_key.items()}


def format_row(name: str, summary: dict[str, float]) -> str:
    """Format a summary row for console tables.

    Produces strings like ``"my_metric: 0.62 ± 0.08 (n=5)"``.
    If the CI half-width is ``nan`` (n=1) the uncertainty field is omitted.
    """
    n = int(summary.get("n", 0))
    mean = summary.get("mean", float("nan"))
    ci = summary.get("ci95_half", float("nan"))

    if math.isnan(ci) or n <= 1:
        return "%s: %.4f (n=%d)" % (name, mean, n)
    return "%s: %.4f ± %.4f (n=%d)" % (name, mean, ci, n)
