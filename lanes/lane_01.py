"""Lane 01: Correction-to-Competence Benchmark v2 de-saturation count.

Counts how many of the FROZEN set of ``EvalSuite.final_card`` sub-metrics
produce a spread (max - min) > 0.2 across {ZeroMemoryAgent, ContextOnlyAgent,
FastOnlyAgent} on a one-level seed=0 subset of the curriculum. Spec:
research/01-correction-to-competence-benchmark.md.

The sub-metric set is FROZEN per eval version: see ``_FROZEN_SUB_METRICS``.
Adding new sub-metrics requires an eval version bump; appending entries to
inflate the desaturation count is metric gaming.

If imports fail or fewer than 2 agents survive, returns float('nan').
"""

from __future__ import annotations

from typing import Any

from lanes._common import lane_measure

_FROZEN_SUB_METRICS = (
    "correction_uptake_latency",
    "transfer_score",
    "scope_score",
    "forgetting_score",
    "consolidation_score",
    "memory_bytes_per_behavior_delta",
    "identity_drift_score",
    "signed_interference_forgetting",
    "separated_exact_vs_domain_recall",
    "behavior_delta_per_byte",
)
# WARNING: This set is FROZEN per eval version. Adding new sub-metrics
# requires an eval version bump. Do not append entries here to inflate
# the desaturation count — that is metric gaming.

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
    return "lane_01_frozen_desaturation_count"


@lane_measure
def measure() -> float:
    try:
        from oczy.experiments.baselines import (
            ContextOnlyAgent,
            FastOnlyAgent,
            ZeroMemoryAgent,
        )
        from oczy.experiments.curriculum import build_curriculum
        from oczy.experiments.eval_suite import (
            EvalSuite,
            _memory_bytes,
            _normalize,
            _respond,
            _token_set,
        )
    except Exception:
        return float("nan")

    full = build_curriculum(seed=0)
    if not full.levels():
        return float("nan")
    suite = EvalSuite(_SubsetCurriculum(full, n=1))
    tprobes = tuple(getattr(suite, "_transfer_probes", ()))
    n_probes = len(tprobes)

    def _exact(a: str, e: str) -> bool:
        return _normalize(a) == _normalize(e)

    def _domain(a: str, e: str) -> bool:
        """Substring match OR >=50% shared non-stopword overlap."""
        a_n, e_n = _normalize(a), _normalize(e)
        if not a_n or not e_n:
            return False
        if e_n in a_n:
            return True
        at, et = _token_set(a), _token_set(e)
        return bool(et) and len(at & et) / len(et) >= 0.5

    def _shift_dir(pre: str, post: str, expected: str) -> int:
        pre_o = len(_token_set(pre) & _token_set(expected))
        post_o = len(_token_set(post) & _token_set(expected))
        return 1 if post_o > pre_o else (-1 if post_o < pre_o else 0)

    cards: list[dict[str, Any]] = []
    for factory in (ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent):
        try:
            agent = factory()
            pre_ans = [_respond(agent, p.request) for p in tprobes]
            pre_bytes = _memory_bytes(agent)
            result = suite.run(agent)
            post_ans = [_respond(agent, p.request) for p in tprobes]
            delta_bytes = max(1, int(result.consolidated_size) - pre_bytes)
            card = dict(result.final_card)

            # 1. signed_interference_forgetting: signed transfer shift.
            shifts = []
            for pre, post, p in zip(pre_ans, post_ans, tprobes):
                if _normalize(pre) == _normalize(post):
                    shifts.append(0.0)
                else:
                    shifts.append(float(_shift_dir(pre, post, p.expected)))
            card["signed_interference_forgetting"] = (
                sum(shifts) / n_probes if n_probes else 0.0
            )

            # 2. separated_exact_vs_domain_recall: |exact - domain| gap.
            if n_probes:
                exact_recall = sum(
                    _exact(a, p.expected) for a, p in zip(post_ans, tprobes)
                ) / n_probes
                domain_recall = sum(
                    _domain(a, p.expected) for a, p in zip(post_ans, tprobes)
                ) / n_probes
                pre_domain_recall = sum(
                    _domain(a, p.expected) for a, p in zip(pre_ans, tprobes)
                ) / n_probes
            else:
                exact_recall = domain_recall = pre_domain_recall = 0.0
            card["separated_exact_vs_domain_recall"] = abs(
                exact_recall - domain_recall
            )

            # 3. behavior_delta_per_byte: un-inverted north star.
            behavior_delta = domain_recall - pre_domain_recall
            card["behavior_delta_per_byte"] = behavior_delta / delta_bytes

            cards.append(card)
        except Exception:
            continue
    if len(cards) < 2:
        return float("nan")
    count = 0
    for metric in _FROZEN_SUB_METRICS:
        try:
            values = [float(c.get(metric, 0.0)) for c in cards]
        except (TypeError, ValueError):
            continue
        if values and (max(values) - min(values)) > 0.2:
            count += 1
    return float(count)
