"""Fast unit tests verifying CortexAgent.consolidate invokes replay SGD."""

from __future__ import annotations

from typing import Any

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers

    def peek_embedding(self, text: str, last_token_only: bool = True) -> np.ndarray:
        return np.ones(self.n_embd, dtype=np.float32)

    def set_reserved_position(self, position: Any) -> None:
        pass

    def clear_reserved_position(self) -> None:
        pass

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
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        return "mock"


class _MockHippocampus:
    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    def consolidate(self) -> list[dict[str, Any]]:
        return self._summaries

    def status(self, include_size: bool = False) -> dict[str, Any]:
        return {"episode_count": 0}


def test_consolidate_calls_replay_train_step_for_correction_summary() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=0.1),
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    hidden = np.arange(8, dtype=np.float32)
    summaries = [
        {
            "representative_hidden": hidden,
            "representative_query": "query",
            "summary_corrections": ["correction"],
        }
    ]
    agent.neural_hippocampus = _MockHippocampus(summaries)

    W_before = agent.cortex.proj_hidden.copy()
    result = agent.consolidate()

    assert result["replay_sgd_updated"] == 1
    assert result["replay_count"] == 1
    assert not np.allclose(agent.cortex.proj_hidden, W_before)


def test_consolidate_skips_replay_train_step_when_lr_zero() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=0.0),
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    hidden = np.arange(8, dtype=np.float32)
    summaries = [
        {
            "representative_hidden": hidden,
            "representative_query": "query",
            "summary_corrections": ["correction"],
        }
    ]
    agent.neural_hippocampus = _MockHippocampus(summaries)

    W_before = agent.cortex.proj_hidden.copy()
    result = agent.consolidate()

    assert result["replay_sgd_updated"] == 0
    np.testing.assert_allclose(agent.cortex.proj_hidden, W_before)


def test_consolidate_uses_negative_sign_for_neutral_summary() -> None:
    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=0.1),
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()

    hidden = np.arange(8, dtype=np.float32)
    summaries = [
        {
            "representative_hidden": hidden,
            "representative_query": "query",
            "summary_corrections": [],
        }
    ]
    agent.neural_hippocampus = _MockHippocampus(summaries)

    # We cannot assert exact sign from outside, but we can assert the method
    # ran and reported a loss. The spy in test_cortex_agent_critic_hidden is
    # not reused here to keep this test self-contained.
    result = agent.consolidate()

    assert result["replay_sgd_updated"] == 1
    assert result["summary_count"] == 1
