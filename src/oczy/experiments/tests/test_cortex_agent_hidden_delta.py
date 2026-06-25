"""Test hidden-delta wiring into CortexAgent's autoencoder path."""

from __future__ import annotations

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Deterministic LlamaCVecDriver stand-in returning distinct embeddings."""

    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers

    def peek_embedding(self, text: str, last_token_only: bool = True) -> np.ndarray:
        seed = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=self.n_embd).astype(np.float32)

    def set_cvec_uniform(self, vec: np.ndarray, scale: float = 1.0) -> int:
        return 0

    def set_cvecs_per_layer(
        self, vectors: list[np.ndarray], scale: float = 1.0
    ) -> int:
        return 0

    def clear_cvec(self) -> int:
        return 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        return "mock"


def _make_agent(driver: _MockDriver | None = None) -> CortexAgent:
    driver = driver or _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    return CortexAgent(cfg, driver=driver)


def test_perceive_stores_prev_hidden() -> None:
    agent = _make_agent()
    agent.boot()

    agent.perceive("first utterance")
    assert agent._last_hidden is not None
    assert agent._prev_hidden is None

    first_hidden = agent._last_hidden.copy()

    agent.perceive("second utterance")
    assert agent._prev_hidden is not None
    np.testing.assert_array_equal(agent._prev_hidden, first_hidden)
    assert agent._last_hidden is not None
    assert not np.array_equal(agent._last_hidden, first_hidden)


def test_metabolize_passes_hidden_delta_when_present() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    agent = _make_agent(driver=driver)
    agent.boot()

    agent.perceive("alpha")
    agent.perceive("beta")

    assert agent.experience_autoencoder.config.get("use_hidden_delta") is True

    result = agent.metabolize()
    assert result["metabolized"] is True

    ae = agent.experience_autoencoder
    assert getattr(ae, "_d_hidden", None) == driver.n_embd
    assert getattr(ae, "_A_hidden", None) is not None
    assert ae._A_hidden is not None
    assert ae._A_hidden.shape[-1] == driver.n_embd


def test_hidden_delta_turn_does_not_crash() -> None:
    agent = _make_agent()
    agent.boot()

    agent.perceive("That was wrong, the answer is vertical.")
    result = agent.metabolize()
    assert result["metabolized"] is True
    assert "autoencoder_error" in result
    assert isinstance(result["autoencoder_error"], float)
