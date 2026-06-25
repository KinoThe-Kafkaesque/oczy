"""Tests for the gated response-policy head on CortexAgent."""

from __future__ import annotations

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig


class _MockDriver:
    """Minimal driver that returns deterministic hidden vectors."""

    def __init__(self, n_embd: int = 8) -> None:
        self.n_embd = n_embd
        self.n_layers = 2

    def peek_embedding(
        self, text: str, last_token_only: bool = True
    ) -> np.ndarray:
        base = float(len(text)) + (ord(text[0]) if text else 0) * 0.1
        ramp = np.arange(self.n_embd, dtype=np.float32) * 0.05
        return np.full(self.n_embd, base, dtype=np.float32) + ramp

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        del prompt, max_tokens, temperature, stop
        return "mock"


def _make_agent(hidden_dim: int = 8, d_cortex: int = 4, use_policy_head: bool = False) -> CortexAgent:
    driver = _MockDriver(n_embd=hidden_dim)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=d_cortex),
        use_policy_head=use_policy_head,
    )
    agent = CortexAgent(cfg, driver=driver)
    agent.boot()
    return agent


def test_policy_head_disabled_by_default() -> None:
    agent = _make_agent(use_policy_head=False)
    try:
        agent.policy_score(["a"])
    except RuntimeError as exc:
        assert "not enabled" in str(exc).lower()
    else:
        raise AssertionError("policy_score should raise when disabled")


def test_policy_score_enabled_lazy_init() -> None:
    agent = _make_agent(use_policy_head=True)
    scores = agent.policy_score(["a", "bb"])

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert scores.dtype == np.float64

    expected_dim = agent.cortex.config.d_cortex + agent.driver.n_embd
    assert agent._policy_W is not None
    assert agent._policy_W.shape == (expected_dim,)


def test_policy_score_changes_with_warm_state() -> None:
    agent = _make_agent(use_policy_head=True)
    scores_before = agent.policy_score(["x", "yy"])
    agent.perceive("hello world")
    scores_after = agent.policy_score(["x", "yy"])

    assert not np.allclose(scores_before, scores_after)


def test_policy_update_increases_chosen_score() -> None:
    agent = _make_agent(use_policy_head=True)
    candidates = ["a", "bbbbbb"]
    scores_before = agent.policy_score(candidates)

    agent.policy_update(candidates, chosen_idx=1, reward=1.0, baseline=0.0)
    scores_after = agent.policy_score(candidates)

    assert scores_after[1] > scores_before[1]


def test_policy_select_chooses_argmax_at_zero_temp() -> None:
    agent = _make_agent(use_policy_head=True)
    candidates = ["short", "much longer candidate text"]
    selected = agent.policy_select(candidates, temperature=0.0)

    expected = int(np.argmax(selected["scores"]))
    assert selected["index"] == expected
    assert selected["candidate"] == candidates[expected]
    np.testing.assert_array_equal(selected["scores"], selected["logits"])
