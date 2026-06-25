"""Test wiring WorldModelCritic value head into CortexAgent.metabolize()."""

from __future__ import annotations

from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Minimal LlamaCVecDriver stand-in that returns deterministic text embeddings."""

    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers
        self._cvec_active = False

    def peek_embedding(self, text: str, last_token_only: bool = True) -> np.ndarray:
        rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
        return rng.randn(self.n_embd).astype(np.float32)

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
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        return "mock"


def test_critic_has_value_head_enabled() -> None:
    """CortexAgent now constructs WorldModelCritic with use_value_head=True."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    assert agent.world_model_critic.use_value_head is True


def test_metabolize_trains_value_head() -> None:
    """A correction turn produces a finite TD error in the value head."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    agent.perceive("hello")
    agent.perceive("no, wrong")
    result = agent.metabolize("no, wrong")

    assert result["metabolized"] is True
    assert agent.world_model_critic._last_td_error is not None
    assert np.isfinite(agent.world_model_critic._last_td_error)


def test_metabolize_passes_prev_hidden_as_value_state() -> None:
    """record_outcome receives prev hidden as value state and last as next state."""
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    agent.perceive("hello")
    hello_hidden = agent._last_hidden.copy()

    agent.perceive("no, wrong")
    now_hidden = agent._last_hidden.copy()
    assert agent._prev_hidden is not None

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_record = agent.world_model_critic.record_outcome

    def spy_record(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        original_record(*args, **kwargs)

    agent.world_model_critic.record_outcome = spy_record

    agent.metabolize("no, wrong")

    assert calls, "record_outcome was not called"
    kwargs = calls[-1][1]
    assert np.array_equal(kwargs["value_lm_hidden"], hello_hidden)
    assert np.array_equal(kwargs["next_value_lm_hidden"], now_hidden)
