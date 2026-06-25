"""Fast unit tests for CortexAgent reserved-position recall wiring.

These tests mock the LlamaCVecDriver so they do not load the GGUF model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oczy.experiments.codebase_qa.knowledge_store import KnowledgeStore
from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.lm import ReservedPosition
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Minimal stand-in for LlamaCVecDriver that records steering calls."""

    def __init__(self, n_embd: int = 64, n_layers: int = 4) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.calls: list[tuple[str, Any]] = []
        self._reserved_position: ReservedPosition | None = None
        self._cvec_active = False

    def set_reserved_position(self, position: ReservedPosition | None) -> None:
        self._reserved_position = position
        self.calls.append(("set_reserved_position", position.text if position else None))

    def clear_reserved_position(self) -> None:
        self._reserved_position = None
        self.calls.append(("clear_reserved_position", None))

    def set_cvec_uniform(self, vec: np.ndarray, scale: float = 1.0) -> int:
        self.calls.append(("set_cvec_uniform", scale))
        self._cvec_active = True
        return 0

    def set_cvecs_per_layer(
        self, vectors: list[np.ndarray], scale: float = 1.0
    ) -> int:
        self.calls.append(("set_cvecs_per_layer", scale))
        self._cvec_active = True
        return 0

    def clear_cvec(self) -> int:
        self.calls.append(("clear_cvec", None))
        self._cvec_active = False
        return 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        self.calls.append(("generate", prompt[:40]))
        return "mocked answer"
@pytest.fixture
def agent_with_store() -> tuple[CortexAgent, _MockDriver, KnowledgeStore]:
    driver = _MockDriver(n_embd=64, n_layers=4)
    store = KnowledgeStore(embed_fn=None)
    # Several filler facts ensure the keyword IDF scores for "profile" stay
    # above the default min_score threshold in the small test corpus.
    for i in range(5):
        store.add_fact(f"filler topic {i}", f"Unrelated content about topic {i}.")
    store.add_fact(
        "profile business vertical",
        "In Oczy the word 'profile' here means business vertical.",
        metadata={"reserved_token": "vertical", "source": "poc"},
    )
    store.add_fact(
        "python version",
        "Requires Python 3.10 or newer.",
        metadata={"source": "pyproject.toml"},
    )
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8),
    )
    agent = CortexAgent(cfg, driver=driver, knowledge_store=store)
    return agent, driver, store


def test_articulate_sets_reserved_position_from_store(agent_with_store):
    agent, driver, _ = agent_with_store
    agent.articulate(
        prompt="Answer briefly.\nQuestion: What does 'profile' mean here?\nAnswer:",
        recall_query="what does profile mean here",
        apply_steering=False,
    )

    assert any(call[0] == "set_reserved_position" and call[1] == "vertical" for call in driver.calls)
    assert any(call[0] == "clear_reserved_position" for call in driver.calls)
    # No cvec calls because apply_steering was False.
    assert not any(call[0].startswith("set_cvec") for call in driver.calls)


def test_articulate_skips_cvec_when_reserved_position_active(agent_with_store):
    agent, driver, _ = agent_with_store
    agent.articulate(
        prompt="Answer briefly.\nQuestion: What does 'profile' mean here?\nAnswer:",
        recall_query="what does profile mean here",
        apply_steering=True,
    )

    # Reserved prefix was set.
    assert any(call[0] == "set_reserved_position" and call[1] == "vertical" for call in driver.calls)
    # cvec steering was skipped to avoid interference.
    assert not any(call[0].startswith("set_cvec") for call in driver.calls)
    assert not any(call[0] == "clear_cvec" for call in driver.calls)
    # Prefix was cleaned up.
    assert any(call[0] == "clear_reserved_position" for call in driver.calls)


def test_articulate_uses_cvec_when_no_reserved_position(agent_with_store):
    agent, driver, _ = agent_with_store
    # The "python version" fact has no reserved_token, so only cvec steering
    # should apply when apply_steering=True.
    agent.articulate(
        prompt="Answer briefly.\nQuestion: What Python version is required?\nAnswer:",
        recall_query="python version",
        apply_steering=True,
    )

    assert not any(call[0] == "set_reserved_position" for call in driver.calls)
    assert any(
        call[0] in ("set_cvec_uniform", "set_cvecs_per_layer") for call in driver.calls
    )
    assert any(call[0] == "clear_cvec" for call in driver.calls)
    assert not any(call[0] == "clear_reserved_position" for call in driver.calls)


def test_articulate_can_disable_reserved_position(agent_with_store):
    agent, driver, _ = agent_with_store
    agent.articulate(
        prompt="Answer briefly.\nQuestion: What does 'profile' mean here?\nAnswer:",
        recall_query="what does profile mean here",
        apply_steering=True,
        use_reserved_position=False,
    )

    assert not any(call[0] == "set_reserved_position" for call in driver.calls)
    # cvec steering should run normally.
    assert any(
        call[0] in ("set_cvec_uniform", "set_cvecs_per_layer") for call in driver.calls
    )
    assert any(call[0] == "clear_cvec" for call in driver.calls)
