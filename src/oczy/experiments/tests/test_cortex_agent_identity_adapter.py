"""Test wiring IdentityHypernetwork state adapter into CortexAgent.metabolize()."""

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


class _MockIdentityHypernetwork:
    """Records calls and returns a deterministic state adapter."""

    def __init__(self, adapter_value: float = 0.5) -> None:
        self.update_calls: list[dict[str, Any]] = []
        self.generate_calls: list[int] = []
        self.adapter_value = adapter_value

    def update_identity(self, update: dict[str, Any]) -> None:
        self.update_calls.append(update)

    def generate_state_adapter(self, d_cortex: int) -> np.ndarray:
        self.generate_calls.append(d_cortex)
        return np.full(d_cortex, self.adapter_value, dtype=np.float32)


class _LegacyIdentityHypernetwork:
    """Older IdentityHypernetwork without generate_state_adapter."""

    def update_identity(self, update: dict[str, Any]) -> None:
        pass


def test_metabolize_applies_identity_state_adapter() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    mock_identity = _MockIdentityHypernetwork(adapter_value=0.3)
    # Override the real identity organ so we can inspect/intercept calls.
    agent.identity_hypernetwork = mock_identity

    agent.perceive("No, 'profile' here means business vertical.", correction_signal=1.0)
    assert agent._last_hidden is not None

    # Snapshot the cvecs produced from warm_state with zero state_bias.
    cvec_before = agent.cortex.emit_all_cvecs()

    result = agent.metabolize()
    assert result["metabolized"] is True

    assert len(mock_identity.update_calls) == 1
    assert len(mock_identity.generate_calls) == 1
    assert mock_identity.generate_calls[0] == agent.cortex.config.d_cortex

    np.testing.assert_array_equal(
        agent.cortex.state_bias,
        np.full(agent.cortex.config.d_cortex, 0.3, dtype=np.float32),
    )

    cvec_after = agent.cortex.emit_all_cvecs()
    for i, (before, after) in enumerate(zip(cvec_before, cvec_after, strict=True)):
        assert not np.allclose(before, after), (
            f"layer {i} cvec did not change after applying identity state bias"
        )


def test_metabolize_handles_missing_generate_state_adapter() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        driver=driver,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    agent.identity_hypernetwork = _LegacyIdentityHypernetwork()

    agent.perceive("Wrong answer.", correction_signal=1.0)
    # Older hypernetworks lack generate_state_adapter; metabolize must not
    # raise AttributeError.
    result = agent.metabolize()
    assert result["metabolized"] is True
    assert np.allclose(agent.cortex.state_bias, 0.0)
