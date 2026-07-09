"""Tests for CortexAgent.answer() one-shot LM answer path."""

from __future__ import annotations

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Minimal LlamaCVecDriver stand-in for CortexAgent construction."""

    def __init__(self, n_embd: int = 8, n_layers: int = 2, reply: str = "mock") -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers
        self._reply = reply
        self._cvec_active = False

    def peek_embedding(self, text: str, last_token_only: bool = True) -> np.ndarray:
        return np.ones(self.n_embd, dtype=np.float32)

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
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        return self._reply


def _make_agent(reply: str = "mock") -> CortexAgent:
    driver = _MockDriver(n_embd=8, n_layers=2, reply=reply)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    return agent


def test_answer_generates_text() -> None:
    agent = _make_agent(reply="mock generated text")
    result = agent.answer("hello")
    assert result["answer"] == "mock generated text"


def test_answer_perceives_request() -> None:
    agent = _make_agent()
    agent.answer("hello")
    assert agent._last_utterance == "hello"
    assert agent._last_hidden is not None


def test_answer_with_metabolize_returns_error() -> None:
    agent = _make_agent()
    result = agent.answer("hello", metabolize=True)
    assert result["metabolized"] is True
    assert result["autoencoder_error"] is not None
    assert np.isfinite(result["autoencoder_error"])
