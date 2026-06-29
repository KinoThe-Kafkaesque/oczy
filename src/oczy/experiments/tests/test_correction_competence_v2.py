"""Tests for Correction-to-Competence Benchmark v2.

These tests validate that the new scorecard:
1. Does not collapse to the old saturated tie across the toy baseline set.
2. Is a behavior-only function of agent outputs (no internal-mechanic leakage).
3. Treats ZeroMemoryAgent as the negative control (no delta for doing nothing).
"""

from __future__ import annotations

from oczy.experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    ZeroMemoryAgent,
)
from oczy.experiments.correction_competence_v2 import CorrectionCompetenceV2
from oczy.experiments.curriculum import build_curriculum
from oczy.experiments.eval_suite import EvalSuite


class _SubsetCurriculum:
    """Adapter matching lanes/lane_01.py."""

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


def _score_agent(agent: object) -> dict[str, float]:
    curriculum = _SubsetCurriculum(build_curriculum(seed=0), n=1)
    return CorrectionCompetenceV2(curriculum).score(agent).card


def test_v2_desaturates_some_baselines() -> None:
    """At least one V2 sub-metric must spread across the baseline set."""
    cards = [
        _score_agent(ZeroMemoryAgent()),
        _score_agent(ContextOnlyAgent()),
        _score_agent(FastOnlyAgent()),
    ]
    metrics = list(cards[0].keys())
    spread = max(
        max(c.get(m, 0.0) for c in cards) - min(c.get(m, 0.0) for c in cards)
        for m in metrics
    )
    assert spread > 0.0, "V2 scorecard should not tie every baseline"


def test_v2_behavior_delta_zero_for_zero_memory() -> None:
    """ZeroMemoryAgent learns nothing, so its behavior_delta_per_byte must be ~0."""
    card = _score_agent(ZeroMemoryAgent())
    assert card["behavior_delta_per_byte"] == 0.0


def test_v2_internal_mechanic_hygiene() -> None:
    """Changing PlasticCortex internal config should not change the V2 card.

    The V2 scorecard is a function of observable agent behavior, not the
    mechanism that produced it.
    """
    card_a = _score_agent(FastOnlyAgent({"consolidation_strength": 1.0}))
    card_b = _score_agent(FastOnlyAgent({"consolidation_strength": 10.0}))
    assert card_a == card_b


def test_v2_sub_metric_names_match_lane_01() -> None:
    """The 10 reported sub-metrics must include the lane_01 set."""
    card = _score_agent(FastOnlyAgent())
    for m in (
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
    ):
        assert m in card


def test_eval_suite_snapshot_fields_preserved() -> None:
    """EvalResult must carry raw pre/post/consolidation snapshots for V2."""
    curriculum = _SubsetCurriculum(build_curriculum(seed=0), n=1)
    suite = EvalSuite(curriculum)
    agent = ContextOnlyAgent()
    result = suite.run(agent)
    assert result.pre_test is not None
    assert result.post_test is not None
    assert result.consolidation_test is not None


def test_separated_exact_vs_domain_gap_nonnegative() -> None:
    """|exact - domain| should be >= 0 for every agent."""
    for factory in (ZeroMemoryAgent, ContextOnlyAgent, FastOnlyAgent):
        card = _score_agent(factory())
        assert card["separated_exact_vs_domain_recall"] >= 0.0
