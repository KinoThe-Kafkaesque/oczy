"""Tests for OrganismAgent cortex policy-head scoring in _rank_answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from oczy.experiments.organism import OrganismAgent


@dataclass
class _MockCortexConfig:
    use_policy_head: bool = True


class _MockCortexAgent:
    """Minimal CortexAgent stand-in for policy-head scoring tests."""

    def __init__(self, policy_scores: np.ndarray | None = None) -> None:
        self._last_utterance: str | None = None
        self.config = _MockCortexConfig()
        self._policy_scores = policy_scores
        self.policy_update_calls: list[dict[str, Any]] = []

    def policy_update(
        self,
        candidates: list[str],
        chosen_idx: int,
        reward: float,
        baseline: float,
    ) -> None:
        self.policy_update_calls.append(
            {
                "candidates": candidates,
                "chosen_idx": chosen_idx,
                "reward": reward,
                "baseline": baseline,
            }
        )

    def perceive(self, request: str) -> None:
        self._last_utterance = request
        self.warm_state = True

    def policy_score(self, candidates: list[str]) -> np.ndarray:
        if self._policy_scores is not None:
            return self._policy_scores
        return np.array([10.0] + [0.0] * (len(candidates) - 1))


def test_cortex_policy_default_off_uses_legacy_ranking() -> None:
    organism = OrganismAgent({})
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    assert organism.answer("x") == "a"


def test_cortex_policy_boosts_preferred_candidate() -> None:
    """Policy head favours 'b' even though the fast organ returned 'a'."""
    # Matching candidate order ["a", "b"]: low for "a", high for "b".
    mock_cortex = _MockCortexAgent(policy_scores=np.array([0.0, 10.0]))
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"

    assert organism.answer("x") == "b"
    assert mock_cortex._last_utterance == "x"


def test_cortex_policy_warning_without_cortex_agent() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_cortex_policy": True})
    assert organism.cortex_agent is None


def test_policy_update_called_on_correction() -> None:
    """A real correction trains the CortexAgent policy head."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 1
    call = mock_cortex.policy_update_calls[0]
    assert call["candidates"] == ["a", "b"]
    assert call["chosen_idx"] == 0
    assert call["reward"] == -1.0


def test_policy_update_skipped_when_disabled() -> None:
    """Default use_cortex_policy=False keeps the correction path unchanged."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent({"cortex_agent": mock_cortex})
    assert not organism.use_cortex_policy
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 0


def test_policy_update_adds_expected_answer_to_candidates() -> None:
    """Policy update receives the expected label even if not a prior candidate."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is c.")

    assert len(mock_cortex.policy_update_calls) == 1
    call = mock_cortex.policy_update_calls[0]
    assert "c" in call["candidates"]

