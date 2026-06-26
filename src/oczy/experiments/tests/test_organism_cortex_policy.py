"""Tests for OrganismAgent cortex policy-head scoring in _rank_answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.organism import OrganismAgent
from plastic_cortex.kv_cortex import KVCortexConfig


@dataclass
class _MockCortexConfig:
    use_policy_head: bool = True


class _MockCortexAgent:
    """Minimal CortexAgent stand-in for policy-head scoring tests."""

    def __init__(
        self,
        policy_scores: np.ndarray | None = None,
        last_hidden: np.ndarray | None = None,
        value_critic: Any | None = None,
    ) -> None:
        self._last_utterance: str | None = None
        self.config = _MockCortexConfig()
        self._policy_scores = policy_scores
        self._last_hidden = last_hidden
        self.world_model_critic = value_critic
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


class _MockValueCritic:
    """Returns a fixed value estimate for baseline tests."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict_value(
        self,
        query: str,
        proposed_answer: str,
        lm_hidden: Any,
    ) -> float:
        return self._value


def test_cortex_policy_default_off_uses_legacy_ranking() -> None:
    organism = OrganismAgent({})
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    assert organism.answer("x") == "a"


def test_cortex_policy_boosts_preferred_candidate() -> None:
    """Policy head favours 'b' even though the fast organ returned 'a'."""
    # Matching candidate order ["a", "b"]: low logit for "a", high logit for "b".
    # Softmax turns these into probabilities; weight=2.0 keeps "b" ahead of the
    # fast-answer bias (+1.0) plus token overlap (0).
    mock_cortex = _MockCortexAgent(policy_scores=np.array([0.0, 10.0]))
    organism = OrganismAgent(
        {
            "use_cortex_policy": True,
            "cortex_policy_weight": 2.0,
            "cortex_agent": mock_cortex,
        }
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

    assert len(mock_cortex.policy_update_calls) == 2
    negative = [c for c in mock_cortex.policy_update_calls if c["reward"] == -1.0]
    positive = [c for c in mock_cortex.policy_update_calls if c["reward"] == 1.0]
    assert len(negative) == 1
    assert negative[0]["candidates"] == ["a", "b"]
    assert negative[0]["chosen_idx"] == 0
    assert len(positive) == 1
    assert positive[0]["candidates"] == ["a", "b"]
    assert positive[0]["chosen_idx"] == 1


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

class _LowSurpriseCritic:
    """Returns a low prediction error so the slow-memory gate stays closed."""

    def predict_acceptance(self, query: str, proposed_answer: str) -> dict[str, float]:
        return {"accepted_prob": 0.1}

    def record_outcome(
        self, query: str, proposed_answer: str, correction: str
    ) -> None:
        pass


def test_policy_update_fires_even_when_critic_not_surprised() -> None:
    """Policy update must not be blocked by the critic-surprise gate."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism.world_model_critic = _LowSurpriseCritic()
    organism._surprise_threshold = 0.5

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 2
    negative = [c for c in mock_cortex.policy_update_calls if c["reward"] == -1.0]
    positive = [c for c in mock_cortex.policy_update_calls if c["reward"] == 1.0]
    assert len(negative) == 1
    assert len(positive) == 1
    assert positive[0]["chosen_idx"] == positive[0]["candidates"].index("b")


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

    assert len(mock_cortex.policy_update_calls) == 2
    positive = [c for c in mock_cortex.policy_update_calls if c["reward"] == 1.0]
    assert len(positive) == 1
    assert "c" in positive[0]["candidates"]
    assert positive[0]["candidates"].index("c") == positive[0]["chosen_idx"]



def test_policy_update_rewards_correct_action() -> None:
    """Both the mistaken and corrected actions receive symmetric rewards."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 2
    negative = [c for c in mock_cortex.policy_update_calls if c["reward"] == -1.0]
    positive = [c for c in mock_cortex.policy_update_calls if c["reward"] == 1.0]
    assert len(negative) == 1
    assert negative[0]["chosen_idx"] == negative[0]["candidates"].index("a")
    assert len(positive) == 1
    assert positive[0]["chosen_idx"] == positive[0]["candidates"].index("b")


def test_positive_update_skipped_when_expected_missing() -> None:
    """No positive reinforcement when the correction text has no target label."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._extract_expected_from_correction = lambda correction: ""
    organism.learn("x", "No, that is wrong.")

    assert len(mock_cortex.policy_update_calls) == 1
    assert mock_cortex.policy_update_calls[0]["reward"] == -1.0


def test_value_baseline_disabled_by_default() -> None:
    """Without use_value_baseline, policy updates receive a zero baseline."""
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 2
    assert all(c["baseline"] == 0.0 for c in mock_cortex.policy_update_calls)


def test_value_baseline_uses_predict_value() -> None:
    """When enabled, the CortexAgent value estimate feeds the REINFORCE baseline."""
    mock_cortex = _MockCortexAgent(
        last_hidden=np.array([1.0, 2.0, 3.0]),
        value_critic=_MockValueCritic(0.42),
    )
    organism = OrganismAgent(
        {
            "use_cortex_policy": True,
            "use_value_baseline": True,
            "cortex_agent": mock_cortex,
        }
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    organism._surprise_threshold = 0.0

    organism.learn("x", "No, it is b.")

    assert len(mock_cortex.policy_update_calls) == 2
    assert all(
        c["baseline"] == pytest.approx(0.42)
        for c in mock_cortex.policy_update_calls
    )


def test_value_baseline_warning_without_cortex_agent() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_value_baseline": True})
    assert organism.cortex_agent is None


def test_acceptance_policy_reward_disabled_by_default() -> None:
    """Default config does not emit an acceptance policy update."""
    mock = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock}
    )
    organism.plastic_cortex.labels = ["a", "b", "c"]
    organism.plastic_cortex.answer = lambda request: "a"

    assert organism.answer("x") == "a"
    assert len(mock.policy_update_calls) == 0


def test_acceptance_policy_reward_trains_accepted_answer() -> None:
    """A high-confidence accepted answer triggers a +1 policy update."""
    mock = _MockCortexAgent()
    organism = OrganismAgent(
        {
            "use_acceptance_policy_reward": True,
            "use_cortex_policy": True,
            "cortex_agent": mock,
            "low_confidence_threshold": 0.0,
            "high_correction_threshold": 2.0,
        }
    )
    organism.plastic_cortex.labels = ["a", "b", "c"]
    organism.plastic_cortex.answer = lambda request: "a"

    answer = organism.answer("x")
    assert len(mock.policy_update_calls) == 1
    call = mock.policy_update_calls[0]
    assert call["reward"] == 1.0
    assert call["candidates"] == ["a", "b", "c"]
    assert call["chosen_idx"] == call["candidates"].index(answer)
    assert call["baseline"] == 0.0


def test_acceptance_policy_reward_uses_value_baseline() -> None:
    """When value baseline is enabled, acceptance update uses the critic estimate."""
    mock = _MockCortexAgent(
        last_hidden=np.array([1.0, 2.0, 3.0]),
        value_critic=_MockValueCritic(0.42),
    )
    organism = OrganismAgent(
        {
            "use_acceptance_policy_reward": True,
            "use_cortex_policy": True,
            "use_value_baseline": True,
            "cortex_agent": mock,
            "low_confidence_threshold": 0.0,
            "high_correction_threshold": 2.0,
        }
    )
    organism.plastic_cortex.labels = ["a", "b", "c"]
    organism.plastic_cortex.answer = lambda request: "a"

    organism.answer("x")
    assert len(mock.policy_update_calls) == 1
    assert mock.policy_update_calls[0]["baseline"] == pytest.approx(0.42)


def test_acceptance_policy_reward_skips_low_confidence_answer() -> None:
    """Low-confidence answers must not receive acceptance reinforcement."""
    mock = _MockCortexAgent()
    organism = OrganismAgent(
        {
            "use_acceptance_policy_reward": True,
            "use_cortex_policy": True,
            "cortex_agent": mock,
            "low_confidence_threshold": 1.0,
            "high_correction_threshold": 2.0,
        }
    )
    organism.plastic_cortex.labels = ["a", "b", "c"]
    organism.plastic_cortex.answer = lambda request: "a"

    organism.answer("x")
    assert len(mock.policy_update_calls) == 0


def test_acceptance_policy_reward_warning_without_cortex_agent() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_acceptance_policy_reward": True})
    assert organism.cortex_agent is None

class _RequestContextDriver:
    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers

    def peek_embedding(
        self, text: str, last_token_only: bool = True
    ) -> np.ndarray:
        seed = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=self.n_embd).astype(np.float32)


def test_request_context_expands_policy_features() -> None:
    driver = _RequestContextDriver(n_embd=8)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_policy_head=True,
        use_policy_request_context=True,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    agent.perceive("what is the answer")
    candidates = ["x", "y", "z"]
    scores = agent.policy_score(candidates)
    features = agent._policy_features(candidates)
    expected_dim = 4 + 4 + 8  # warm_state + request_context + candidate_hidden
    assert features.shape == (3, expected_dim)
    assert agent._policy_W.shape == (expected_dim,)
    assert scores.shape == (3,)


def test_request_context_disabled_keeps_original_feature_count() -> None:
    driver = _RequestContextDriver(n_embd=8)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_policy_head=True,
        use_policy_request_context=False,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    agent.perceive("what is the answer")
    candidates = ["x", "y", "z"]
    scores = agent.policy_score(candidates)
    features = agent._policy_features(candidates)
    expected_dim = 4 + 8  # warm_state + candidate_hidden
    assert features.shape == (3, expected_dim)
    assert agent._policy_W.shape == (expected_dim,)
    assert scores.shape == (3,)
