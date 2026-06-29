"""Tests for Correction-to-Competence Benchmark v2.

These tests validate that the new scorecard:
1. Does not collapse to the old saturated tie across the toy baseline set.
2. Is a behavior-only function of agent outputs (no internal-mechanic leakage).
3. Treats ZeroMemoryAgent as the negative control (its deltas are ~0).
"""

from __future__ import annotations

from oczy.experiments.baselines import (
    ContextOnlyAgent,
    FastOnlyAgent,
    ZeroMemoryAgent,
)
from oczy.experiments.correction_competence_v2 import (
    CorrectionCompetenceV2,
    V2SubMetrics,
)
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


def _score_agent(agent: object) -> V2SubMetrics:
    curriculum = _SubsetCurriculum(build_curriculum(seed=0), n=1)
    return CorrectionCompetenceV2(curriculum).score(agent).metrics


def test_v2_desaturates_some_baselines() -> None:
    """At least one V2 sub-metric must spread across the baseline set."""
    scores = [
        _score_agent(ZeroMemoryAgent()),
        _score_agent(ContextOnlyAgent()),
        _score_agent(FastOnlyAgent()),
    ]
    fields = list(V2SubMetrics.__dataclass_fields__)
    spread = max(
        max(getattr(s, f) for s in scores) - min(getattr(s, f) for s in scores)
        for f in fields
    )
    assert spread > 0.0, "V2 scorecard should not tie every baseline"


def test_v2_behavior_delta_zero_for_zero_memory() -> None:
    """ZeroMemoryAgent learns nothing, so its behavior_delta_per_byte must be ~0."""
    metrics = _score_agent(ZeroMemoryAgent())
    assert metrics.behavior_delta_per_byte == 0.0


def test_v2_internal_mechanic_hygiene() -> None:
    """Changing PlasticCortex internal config should not change the V2 metrics.

    The V2 scorecard is a function of observable agent behavior, not the
    mechanism that produced it.
    """
    a = _score_agent(FastOnlyAgent({"consolidation_strength": 1.0}))
    b = _score_agent(FastOnlyAgent({"consolidation_strength": 10.0}))
    assert a == b


def test_v2_sub_metric_names_present() -> None:
    """All de-saturation sub-metrics must be present on a scorecard."""
    metrics = _score_agent(FastOnlyAgent())
    for f in V2SubMetrics.__dataclass_fields__:
        assert hasattr(metrics, f)


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
        metrics = _score_agent(factory())
        assert metrics.separated_exact_vs_domain_recall >= 0.0


def test_domain_recall_at_least_as_high_as_exact() -> None:
    """Domain is a relaxed version of exact, so it should never be lower."""
    metrics = _score_agent(FastOnlyAgent())
    assert metrics.domain_recall >= metrics.exact_recall
