"""Test that CortexAgent.metabolize() passes the critic's last correction
probability into DigestiveGate.ingest() and surfaces it in the status dict.
"""

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
        return "mock"


class _UnavailableProbCritic:
    """Critic stand-in that never publishes _last_correction_prob.

    This simulates an uninitialized probability (e.g. hidden unavailable or
    an older critic without the attribute) so metabolize() must fall back to
    ``None`` without crashing.
    """

    def predict_acceptance(
        self,
        query: str,
        proposed_answer: str,
        lm_hidden: np.ndarray | None = None,
    ) -> dict[str, float]:
        return {
            "accepted_prob": 0.5,
            "correction_likelihood": 0.5,
            "key_uncertainty": 0.5,
        }

    def record_outcome(
        self,
        query: str,
        proposed_answer: str,
        correction: str | None,
        lm_hidden: np.ndarray | None = None,
    ) -> None:
        pass

    def prediction_error(self, actual_was_correction: bool) -> float:
        return 0.0

    def status(self, include_size: bool = False) -> dict[str, Any]:
        return {"record_count": 0}


def test_metabolize_passes_critic_prob_to_gate() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    agent.perceive("hello")
    result = agent.metabolize()

    assert result["metabolized"] is True
    assert "critic_correction_prob" in result
    prob = result["critic_correction_prob"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert agent.digestive_gate._ema > 0.0


def test_critic_prob_none_when_not_initialized() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    # Replace the real critic with a stand-in that never sets
    # _last_correction_prob, forcing the gate fallback path.
    agent.world_model_critic = _UnavailableProbCritic()

    agent.perceive("hello")
    result = agent.metabolize()

    assert result["metabolized"] is True
    assert result.get("critic_correction_prob") is None


if __name__ == "__main__":
    raise SystemExit(0)
