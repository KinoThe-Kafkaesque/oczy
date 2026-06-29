"""Correction-to-Competence Benchmark v2.

A behavior-only scorecard built on top of EvalSuite snapshots.  It exports a
new V2 evaluation that separates architectures the legacy EvalSuite ties at
1.0, plus a small CLI that emits METRIC/ASI lines for the autoresearch
harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oczy.experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    ZeroMemoryAgent,
)
from oczy.experiments.curriculum import Curriculum, build_curriculum
from oczy.experiments.eval_suite import (
    EvalResult,
    EvalSuite,
    _memory_bytes,
    _normalize,
    _respond,
    _token_set,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


_BASE_SUB_METRICS = (
    "correction_uptake_latency",
    "transfer_score",
    "scope_score",
    "forgetting_score",
    "consolidation_score",
    "memory_bytes_per_behavior_delta",
    "identity_drift_score",
)

_NEW_SUB_METRICS = (
    "signed_interference_forgetting",
    "separated_exact_vs_domain_recall",
    "behavior_delta_per_byte",
)


@dataclass(frozen=True)
class V2Scorecard:
    """Full V2 scorecard for one agent/trajectory.

    The 10 de-saturation sub-metrics are stored in ``card`` exactly as the
    lane_01 naming convention expects, plus the original EvalSuite metrics.
    """

    card: dict[str, float]
    raw_trace_size: int
    consolidated_size: int
    successful_lessons: int
    delta_persistent_bytes: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "card": self.card,
            "raw_trace_size": self.raw_trace_size,
            "consolidated_size": self.consolidated_size,
            "successful_lessons": self.successful_lessons,
            "delta_persistent_bytes": self.delta_persistent_bytes,
        }


class _SubsetCurriculum:
    """Adapter exposing only the first N levels of a base Curriculum."""

    def __init__(self, base: Curriculum, n: int = 1) -> None:
        self._base = base
        self._levels = base.levels()[:n]

    @property
    def seed(self) -> int:
        return self._base.seed

    def levels(self):
        return self._levels

    def __len__(self) -> int:
        return len(self._levels)


# ---------------------------------------------------------------------------
# V2 scorer
# ---------------------------------------------------------------------------


class CorrectionCompetenceV2:
    """Compute the V2 scorecard from an EvalSuite run.

    Mirrors the behavior definitions in ``lanes/lane_01.py`` so that
    ``v2_desaturation_count`` is directly comparable to
    ``lane_01_desaturation_count``.
    """

    def __init__(self, curriculum: Curriculum) -> None:
        self.suite = EvalSuite(curriculum, sense_match=False)
        self._transfer_probes = getattr(self.suite, "_transfer_probes", ())

    def score(self, agent: Any) -> V2Scorecard:
        """Run the full protocol and return a V2 scorecard."""
        result = self.suite.run(agent)
        return self.score_from_result(result, agent)

    def score_from_result(
        self, result: EvalResult, agent: Any
    ) -> V2Scorecard:
        """Compute the V2 scorecard from an already-produced EvalResult."""
        tprobes = self._transfer_probes
        n_probes = len(tprobes)
        if n_probes == 0:
            card = dict(result.final_card)
            return V2Scorecard(
                card=card,
                raw_trace_size=result.raw_trace_size,
                consolidated_size=result.consolidated_size,
                successful_lessons=0,
                delta_persistent_bytes=0.0,
            )

        pre_ans = [_respond(agent, p.request) for p in tprobes]
        pre_bytes = _memory_bytes(agent)
        post_ans = [_respond(agent, p.request) for p in tprobes]
        delta_bytes = max(1, int(result.consolidated_size) - pre_bytes)

        card = dict(result.final_card)

        # 1. signed_interference_forgetting: signed transfer shift.
        shifts: list[float] = []
        for pre, post, p in zip(pre_ans, post_ans, tprobes):
            if _normalize(pre) == _normalize(post):
                shifts.append(0.0)
            else:
                shifts.append(float(_shift_dir(pre, post, p.expected)))
        card["signed_interference_forgetting"] = (
            sum(shifts) / n_probes if n_probes else 0.0
        )

        # 2. separated_exact_vs_domain_recall: |exact - domain| gap.
        exact_recall = (
            sum(_exact(a, p.expected) for a, p in zip(post_ans, tprobes))
            / n_probes
        )
        domain_recall = (
            sum(_domain(a, p.expected) for a, p in zip(post_ans, tprobes))
            / n_probes
        )
        card["separated_exact_vs_domain_recall"] = abs(
            exact_recall - domain_recall
        )

        # 3. behavior_delta_per_byte: un-inverted north star.
        pre_domain_recall = (
            sum(_domain(a, p.expected) for a, p in zip(pre_ans, tprobes))
            / n_probes
        )
        behavior_delta = domain_recall - pre_domain_recall
        card["behavior_delta_per_byte"] = behavior_delta / delta_bytes

        return V2Scorecard(
            card=card,
            raw_trace_size=result.raw_trace_size,
            consolidated_size=result.consolidated_size,
            successful_lessons=self._successful_lessons(result),
            delta_persistent_bytes=float(delta_bytes),
        )

    @staticmethod
    def _successful_lessons(result: EvalResult) -> int:
        return sum(
            1
            for level in result.per_level_results
            for ep in level["episodes"]
            if ep["fixed_after_correction"]
        )


# ---------------------------------------------------------------------------
# Scoring primitives aligned with lane_01.py
# ---------------------------------------------------------------------------


def _exact(a: str, e: str) -> bool:
    return _normalize(a) == _normalize(e)


def _domain(a: str, e: str) -> bool:
    """Substring match OR >=50% shared non-stopword overlap."""
    a_n = _normalize(a)
    e_n = _normalize(e)
    if not a_n or not e_n:
        return False
    if e_n in a_n:
        return True
    at = _token_set(a)
    et = _token_set(e)
    return bool(et) and len(at & et) / len(et) >= 0.5


def _shift_dir(pre: str, post: str, expected: str) -> int:
    pre_o = len(_token_set(pre) & _token_set(expected))
    post_o = len(_token_set(post) & _token_set(expected))
    if post_o > pre_o:
        return 1
    if post_o < pre_o:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Bootstrap + discrimination
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap percentile CI."""
    if not values or len(values) < 2:
        mean = float(values[0]) if values else 0.0
        return mean, mean, mean
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    means_arr = np.asarray(means)
    lower = float(np.percentile(means_arr, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means_arr, (1 + ci) / 2 * 100))
    return float(arr.mean()), lower, upper


def ci_disjoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Return True if the two (mean, lower, upper) intervals do not overlap."""
    return a[1] > b[2] or b[1] > a[2]


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def _extract_spread(
    cards: list[dict[str, float]], threshold: float
) -> tuple[int, dict[str, float]]:
    spreads: dict[str, float] = {}
    for metric in _BASE_SUB_METRICS + _NEW_SUB_METRICS:
        try:
            values = [float(c.get(metric, 0.0)) for c in cards]
        except (TypeError, ValueError):
            continue
        spreads[metric] = (max(values) - min(values)) if values else 0.0
    count = sum(1 for s in spreads.values() if s > threshold)
    return count, spreads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correction-to-Competence Benchmark v2"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
        help="Mock driver is fast and driver-free; real driver loads LFM2.5 GGUF.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of curriculum seeds to average over.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Use only the first N curriculum levels (1 matches lane_01).",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        default=0.2,
        help="Minimum spread across the baseline agent set to count as de-saturated.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args(argv)

    if args.driver == "real":
        print("ASI real_driver=not_yet_implemented_for_v2")
        return 0

    full = build_curriculum(seed=0)
    if args.levels <= 0 or not full.levels():
        print("METRIC v2_desaturation_count=0")
        print("METRIC v2_discrimination=0")
        print("METRIC v2_behavior_delta_mock=0")
        return 0

    curriculum: Any = _SubsetCurriculum(full, n=args.levels)
    threshold = args.spread_threshold

    cards: list[dict[str, float]] = []
    behavior_delta_mock = 0.0
    for factory in (ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent):
        agent = factory()
        scorer = CorrectionCompetenceV2(curriculum)
        card = scorer.score(agent).card
        cards.append(card)
        if isinstance(agent, FastOnlyAgent):
            behavior_delta_mock = card.get("behavior_delta_per_byte", 0.0)

    desaturation_count, spreads = _extract_spread(cards, threshold)
    print(f"METRIC v2_desaturation_count={desaturation_count}")
    print("METRIC v2_discrimination=0")
    print(f"METRIC v2_behavior_delta_mock={behavior_delta_mock}")

    for metric, spread in spreads.items():
        print(f"ASI spread_{metric}={spread}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "agents": cards,
                    "spreads": spreads,
                    "desaturation_count": desaturation_count,
                },
                fh,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
