"""Fast unit tests verifying CortexAgent passes lm_hidden into the critic."""

from __future__ import annotations

from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Minimal LlamaCVecDriver stand-in for CortexAgent construction."""

    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers
        self._reserved_position = None
        self._cvec_active = False

    def peek_embedding(self, text: str, last_token_only: bool = True) -> np.ndarray:
        return np.ones(self.n_embd, dtype=np.float32)

    def set_reserved_position(self, position: Any) -> None:
        pass

    def clear_reserved_position(self) -> None:
        pass

    def set_cvec_uniform(self, vec: np.ndarray, scale: float = 1.0) -> int:
        self._cvec_active = True
        return 0

    def set_cvecs_per_layer(
        self, vectors: list[np.ndarray], scale: float = 1.0
    ) -> int:
        self._cvec_active = True
        return 0

    def clear_cvec(self) -> int:
        self._cvec_active = False
        return 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        return "mock"


class _RecordingCritic:
    """Spy that records every predict/record call and its hidden argument."""

    def __init__(self) -> None:
        self.predictions: list[tuple[str, str, np.ndarray | None]] = []
        self.records: list[tuple[str, str, str | None, np.ndarray | None]] = []
        self._last_correction_prob: float | None = None

    def predict_acceptance(
        self, query: str, proposed_answer: str, lm_hidden: np.ndarray | None = None
    ) -> dict[str, float]:
        self.predictions.append((query, proposed_answer, lm_hidden))
        prob = 0.5
        self._last_correction_prob = prob
        return {
            "accepted_prob": 1.0 - prob,
            "correction_likelihood": prob,
            "key_uncertainty": 0.5,
        }

    def record_outcome(
        self,
        query: str,
        proposed_answer: str,
        correction: str | None,
        lm_hidden: np.ndarray | None = None,
    ) -> None:
        self.records.append((query, proposed_answer, correction, lm_hidden))

    def prediction_error(self, actual_was_correction: bool) -> float:
        return 0.0

    def status(self, include_size: bool = False) -> dict[str, Any]:
        return {"record_count": len(self.records)}


def test_critic_use_hidden_default() -> None:
    """CortexAgent now constructs WorldModelCritic with use_hidden=True."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    assert agent.world_model_critic.use_hidden is True
    assert agent.world_model_critic.mlp_hidden_units == 16
    assert agent.world_model_critic.d_hidden == 0


def test_metabolize_trains_critic_mlp() -> None:
    """One correction turn lazy-initializes the critic MLP from the hidden vector."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    assert agent.world_model_critic.W1 is None

    agent.perceive("No, 'profile' here means business vertical.")
    assert agent._last_hidden is not None
    agent.metabolize()

    critic = agent.world_model_critic
    assert critic.d_hidden == driver.n_embd
    assert critic.W1 is not None
    assert critic.W1.shape == (critic.mlp_hidden_units, 4 + driver.n_embd)
    assert critic.b1 is not None
    assert critic.W2 is not None


def test_predict_hidden_changes_with_similar_input() -> None:
    """After one correction update, close hidden vectors yield closer predictions."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    query = "What does profile mean here?"
    agent.perceive("No, 'profile' here means business vertical.")
    agent.metabolize()

    base = agent._last_hidden.copy()
    rng = np.random.RandomState(7)
    close_1 = base + rng.randn(driver.n_embd).astype(np.float32) * 0.05
    close_2 = base + rng.randn(driver.n_embd).astype(np.float32) * 0.05
    distant = base + rng.randn(driver.n_embd).astype(np.float32) * 5.0

    critic = agent.world_model_critic
    p_close_1 = critic.predict_acceptance(query, "", lm_hidden=close_1)[
        "correction_likelihood"
    ]
    p_close_2 = critic.predict_acceptance(query, "", lm_hidden=close_2)[
        "correction_likelihood"
    ]
    p_distant = critic.predict_acceptance(query, "", lm_hidden=distant)[
        "correction_likelihood"
    ]

    assert abs(p_close_1 - p_close_2) < abs(p_close_1 - p_distant)


def test_metabolize_passes_lm_hidden_to_critic() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    spy = _RecordingCritic()
    agent.world_model_critic = spy

    agent.perceive("No, 'profile' here means business vertical.")
    assert agent._last_hidden is not None
    hidden_snapshot = agent._last_hidden.copy()

    agent.metabolize()

    assert len(spy.predictions) == 1
    assert len(spy.records) == 1

    _, _, pred_hidden = spy.predictions[0]
    _, _, _, record_hidden = spy.records[0]

    assert pred_hidden is not None
    assert record_hidden is not None
    np.testing.assert_array_equal(pred_hidden, hidden_snapshot)
    np.testing.assert_array_equal(record_hidden, hidden_snapshot)


def test_metabolize_skips_critic_when_no_hidden() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    spy = _RecordingCritic()
    agent.world_model_critic = spy

    # Bypass perceive() so _last_hidden stays None.
    agent._last_utterance = "hello"
    agent._last_correction_signal = 1.0
    result = agent.metabolize()

    assert result["metabolized"] is False
    assert len(spy.predictions) == 0
    assert len(spy.records) == 0
