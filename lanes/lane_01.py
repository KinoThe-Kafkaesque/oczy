"""Lane 01: Correction-to-Competence Benchmark v2 de-saturation count.

Counts how many of the 7 eval-suite sub-metrics produce a spread
(max - min) greater than 0.2 across the baseline agent set
{ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent} on a one-level
deterministic subset of the seed=0 curriculum. A count of 0 or 1
means the eval cannot discriminate between architectures -- the
saturation problem diagnosed in
research/01-correction-to-competence-benchmark.md.

The 7 sub-metrics are ``EvalSuite.run(...).final_card`` values:
correction_uptake_latency, transfer_score, scope_score,
forgetting_score, consolidation_score, memory_bytes_per_behavior_delta,
identity_drift_score.

No real LM is required: all three baselines are pure-python stubs
(ZeroMemory / ContextOnly) or run on the mock PlasticCortex. If imports
fail, or fewer than 2 agents survive, returns float('nan').
"""

from __future__ import annotations


_SUB_METRICS = (
    "correction_uptake_latency",
    "transfer_score",
    "scope_score",
    "forgetting_score",
    "consolidation_score",
    "memory_bytes_per_behavior_delta",
    "identity_drift_score",
)


class _SubsetCurriculum:
    """Adapter exposing only the first N levels of a base Curriculum."""

    def __init__(self, base, n: int = 1) -> None:
        self._base = base
        self._levels = base.levels()[:n]

    @property
    def seed(self) -> int:
        return self._base.seed

    def levels(self):
        return self._levels

    def __len__(self) -> int:
        return len(self._levels)


def name() -> str:
    return "lane_01_desaturation_count"


def measure() -> float:
    try:
        from src.oczy.experiments.baselines import (
            ContextOnlyAgent,
            FastOnlyAgent,
            ZeroMemoryAgent,
        )
        from src.oczy.experiments.curriculum import build_curriculum
        from src.oczy.experiments.eval_suite import EvalSuite
    except Exception:
        return float("nan")

    try:
        # Deterministic seed=0 curriculum; one level only (no LM calls).
        full = build_curriculum(seed=0)
        if not full.levels():
            return float("nan")
        suite = EvalSuite(_SubsetCurriculum(full, n=1))
        cards = []
        for factory in (ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent):
            try:
                cards.append(suite.run(factory()).final_card)
            except Exception:
                continue  # per-agent failure (e.g. real-LM path): skip
        if len(cards) < 2:
            return float("nan")
        count = 0
        for metric in _SUB_METRICS:
            try:
                values = [float(c.get(metric, 0.0)) for c in cards]
            except (TypeError, ValueError):
                continue
            if values and (max(values) - min(values)) > 0.2:
                count += 1
        return float(count)
    except Exception:
        return float("nan")