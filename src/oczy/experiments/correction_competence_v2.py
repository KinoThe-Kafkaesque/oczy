"""Correction-to-Competence Benchmark v2.

A behavior-only scorecard built on top of EvalSuite snapshots.  It exports a
new V2 evaluation that separates architectures the legacy EvalSuite ties at
1.0, plus a small CLI that emits METRIC/ASI lines for the autoresearch
harness.

The sub-metrics follow the Experiment 01 spec and the lane_01 definitions:
  - Core behavior metrics from the legacy EvalSuite final_card.
  - Spec v2 metrics: exact_recall, domain_recall, uptake_gain,
    forgetting_delta, identity_drift, compression_ratio,
    delta_persistent_bytes, behavior_delta_per_byte.
  - New de-saturating metrics: separated_exact_vs_domain_recall,
    signed_interference_forgetting, behavior_delta_per_byte (again as the
    un-inverted north star).
"""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class V2SubMetrics:
    """All unique v2 sub-metrics used for the de-saturation count."""

    # Legacy final_card metrics.
    correction_uptake_latency: float
    transfer_score: float
    scope_score: float
    forgetting_score: float
    consolidation_score: float
    memory_bytes_per_behavior_delta: float
    identity_drift_score: float

    # Spec v2 metrics.
    exact_recall: float
    domain_recall: float
    uptake_gain: float
    forgetting_delta: float
    identity_drift: float
    compression_ratio: float
    delta_persistent_bytes: float
    behavior_delta_per_byte: float

    # Additional spec-lane metrics.
    separated_exact_vs_domain_recall: float
    signed_interference_forgetting: float

    def as_dict(self) -> dict[str, float]:
        return {
            "correction_uptake_latency": self.correction_uptake_latency,
            "transfer_score": self.transfer_score,
            "scope_score": self.scope_score,
            "forgetting_score": self.forgetting_score,
            "consolidation_score": self.consolidation_score,
            "memory_bytes_per_behavior_delta": self.memory_bytes_per_behavior_delta,
            "identity_drift_score": self.identity_drift_score,
            "exact_recall": self.exact_recall,
            "domain_recall": self.domain_recall,
            "uptake_gain": self.uptake_gain,
            "forgetting_delta": self.forgetting_delta,
            "identity_drift": self.identity_drift,
            "compression_ratio": self.compression_ratio,
            "delta_persistent_bytes": self.delta_persistent_bytes,
            "behavior_delta_per_byte": self.behavior_delta_per_byte,
            "separated_exact_vs_domain_recall": self.separated_exact_vs_domain_recall,
            "signed_interference_forgetting": self.signed_interference_forgetting,
        }


@dataclass(frozen=True)
class V2Scorecard:
    """Full V2 scorecard for one agent/trajectory."""

    metrics: V2SubMetrics
    raw_trace_size: int
    consolidated_size: int
    successful_lessons: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "raw_trace_size": self.raw_trace_size,
            "consolidated_size": self.consolidated_size,
            "successful_lessons": self.successful_lessons,
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
# Scoring primitives aligned with lane_01.py and the spec
# ---------------------------------------------------------------------------


def exact_recall(expected: str, answer: str) -> int:
    """Exact normalized match."""
    return 1 if _normalize(answer) == _normalize(expected) else 0


def domain_recall(expected: str, answer: str) -> int:
    """Relaxed domain/sense match: substring or >=50% shared tokens."""
    a_n = _normalize(answer)
    e_n = _normalize(expected)
    if not a_n or not e_n:
        return 0
    if e_n in a_n:
        return 1
    at = _token_set(answer)
    et = _token_set(expected)
    if not et:
        return 0
    return 1 if len(at & et) / len(et) >= 0.5 else 0


def _shift_dir(pre: str, post: str, expected: str) -> int:
    pre_o = len(_token_set(pre) & _token_set(expected))
    post_o = len(_token_set(post) & _token_set(expected))
    if post_o > pre_o:
        return 1
    if post_o < pre_o:
        return -1
    return 0


# ---------------------------------------------------------------------------
# V2 scorer
# ---------------------------------------------------------------------------


class CorrectionCompetenceV2:
    """Compute the V2 scorecard from an EvalSuite run."""

    def __init__(self, curriculum: Curriculum) -> None:
        self.suite = EvalSuite(curriculum, sense_match=False)
        self._transfer_probes = getattr(self.suite, "_transfer_probes", ())

    def score(self, agent: Any) -> V2Scorecard:
        """Run the full protocol and return a V2 scorecard."""
        tprobes = self._transfer_probes
        n_probes = len(tprobes)
        pre_ans = [_respond(agent, p.request) for p in tprobes] if n_probes else []
        pre_bytes = _memory_bytes(agent)
        result = self.suite.run(agent)
        return self.score_from_result(result, agent, pre_ans, pre_bytes)

    def score_from_result(
        self,
        result: EvalResult,
        agent: Any,
        pre_ans: list[str],
        pre_bytes: int = 0,
    ) -> V2Scorecard:
        """Compute the V2 scorecard from a produced EvalResult."""
        sub = self._compute_sub_metrics(result, agent, pre_ans, pre_bytes)
        return V2Scorecard(
            metrics=sub,
            raw_trace_size=result.raw_trace_size,
            consolidated_size=result.consolidated_size,
            successful_lessons=self._successful_lessons(result),
        )

    def _compute_sub_metrics(
        self,
        result: EvalResult,
        agent: Any,
        pre_ans: list[str],
        pre_bytes: int,
    ) -> V2SubMetrics:
        legacy = result.final_card
        pre = result.pre_test
        post = result.post_test
        cons = result.consolidation_test
        tprobes = self._transfer_probes
        n_probes = len(tprobes)

        post_ans = [_respond(agent, p.request) for p in tprobes] if n_probes else []

        exact_recall_val = (
            sum(
                exact_recall(p.expected, a) for a, p in zip(post_ans, tprobes)
            )
            / n_probes
            if n_probes
            else 0.0
        )
        domain_recall_val = (
            sum(domain_recall(p.expected, a) for a, p in zip(post_ans, tprobes))
            / n_probes
            if n_probes
            else 0.0
        )
        pre_domain_recall = (
            sum(domain_recall(p.expected, a) for a, p in zip(pre_ans, tprobes))
            / n_probes
            if n_probes
            else 0.0
        )

        pre_overall = self._overall_accuracy(pre)
        post_overall = self._overall_accuracy(post)
        uptake_gain = post_overall - pre_overall

        pre_forget = self._battery_accuracy(pre, "forgetting", mode="exact")
        post_forget = self._battery_accuracy(cons, "forgetting", mode="exact")
        forgetting_delta = pre_forget - post_forget

        pre_identity = self._battery_accuracy(pre, "identity", mode="exact")
        post_identity = self._battery_accuracy(cons, "identity", mode="exact")
        identity_drift = abs(post_identity - pre_identity)

        compression_ratio = result.raw_trace_size / max(
            1, result.consolidated_size
        )
        delta_persistent_bytes = float(
            max(1, result.consolidated_size - pre_bytes)
        )

        transfer_gain = (
            self._battery_accuracy(post, "transfer", mode="exact")
            - self._battery_accuracy(pre, "transfer", mode="exact")
        )
        behavior_delta_raw = (
            uptake_gain
            + transfer_gain
            - max(0.0, forgetting_delta)
            - identity_drift
        )
        behavior_delta_per_byte = behavior_delta_raw / delta_persistent_bytes

        # Signed interference forgetting on transfer probes.
        shifts: list[float] = []
        if n_probes:
            for pre, post, p in zip(pre_ans, post_ans, tprobes):
                if _normalize(pre) == _normalize(post):
                    shifts.append(0.0)
                else:
                    shifts.append(float(_shift_dir(pre, post, p.expected)))
            signed_interference = sum(shifts) / n_probes
        else:
            signed_interference = 0.0

        return V2SubMetrics(
            correction_uptake_latency=float(
                legacy.get("correction_uptake_latency", 0.0)
            ),
            transfer_score=float(legacy.get("transfer_score", 0.0)),
            scope_score=float(legacy.get("scope_score", 0.0)),
            forgetting_score=float(legacy.get("forgetting_score", 0.0)),
            consolidation_score=float(legacy.get("consolidation_score", 0.0)),
            memory_bytes_per_behavior_delta=float(
                legacy.get("memory_bytes_per_behavior_delta", 0.0)
            ),
            identity_drift_score=float(legacy.get("identity_drift_score", 0.0)),
            exact_recall=exact_recall_val,
            domain_recall=domain_recall_val,
            uptake_gain=uptake_gain,
            forgetting_delta=forgetting_delta,
            identity_drift=identity_drift,
            compression_ratio=compression_ratio,
            delta_persistent_bytes=delta_persistent_bytes,
            behavior_delta_per_byte=behavior_delta_per_byte,
            separated_exact_vs_domain_recall=abs(
                exact_recall_val - domain_recall_val
            ),
            signed_interference_forgetting=signed_interference,
        )

    def _battery_accuracy(
        self,
        snapshot: Any,
        category: str,
        *,
        mode: str = "exact",
    ) -> float:
        if snapshot is None:
            return 0.0
        battery = getattr(snapshot, category, [])
        if not battery:
            return 0.0
        scorer = exact_recall if mode == "exact" else domain_recall
        correct = 0
        for item in battery:
            expected = getattr(item.probe, "expected", "")
            if scorer(expected, item.answer):
                correct += 1
        return correct / len(battery)

    def _overall_accuracy(self, snapshot: Any) -> float:
        if snapshot is None:
            return 0.0
        accs = [
            self._battery_accuracy(snapshot, cat, mode="exact")
            for cat in ("transfer", "scope", "forgetting", "identity")
        ]
        return sum(accs) / len(accs)

    @staticmethod
    def _successful_lessons(result: EvalResult) -> int:
        return sum(
            1
            for level in result.per_level_results
            for ep in level["episodes"]
            if ep["fixed_after_correction"]
        )


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

    curriculum: Any = (
        _SubsetCurriculum(full, n=args.levels)
        if args.levels < len(full.levels())
        else full
    )
    threshold = args.spread_threshold

    scores: list[V2SubMetrics] = []
    for factory in (ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent):
        agent = factory()
        scorer = CorrectionCompetenceV2(curriculum)
        scores.append(scorer.score(agent).metrics)

    spreads: dict[str, float] = {}
    for field in V2SubMetrics.__dataclass_fields__:
        vals = [getattr(s, field) for s in scores]
        spreads[field] = (max(vals) - min(vals)) if vals else 0.0

    desaturation_count = sum(1 for s in spreads.values() if s > threshold)
    behavior_delta_mock = scores[2].behavior_delta_per_byte

    print(f"METRIC v2_desaturation_count={desaturation_count}")
    print("METRIC v2_discrimination=0")
    print(f"METRIC v2_behavior_delta_mock={behavior_delta_mock}")

    for field, spread in spreads.items():
        print(f"ASI spread_{field}={spread}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "agents": [s.as_dict() for s in scores],
                    "spreads": spreads,
                    "desaturation_count": desaturation_count,
                },
                fh,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
