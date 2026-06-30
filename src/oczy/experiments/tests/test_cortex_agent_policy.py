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


def test_bilinear_head_initialized() -> None:
    """The bilinear interaction matrix must be created when the policy
    head is initialized, with shape (d_cortex, hidden_dim)."""
    agent = _make_agent(use_policy_head=True, d_cortex=6, hidden_dim=8)
    agent.policy_score(["a", "b"])  # trigger lazy init
    assert agent._policy_W_bilinear is not None
    assert agent._policy_W_bilinear.shape == (6, 8)


def test_bilinear_term_discriminates_candidates() -> None:
    """The bilinear score (warm @ W_bilinear @ hidden_i) must vary across
    candidates — this is the term that lets warm_state DISCRIMINATE,
    unlike the linear warm portion which is constant for all candidates."""
    agent = _make_agent(use_policy_head=True, d_cortex=8, hidden_dim=16)
    # Set a non-zero warm_state so the bilinear term is active.
    agent.cortex.warm_state = np.random.default_rng(42).normal(
        0, 1, size=agent.cortex.warm_state.shape
    ).astype(np.float32)
    candidates = ["alpha", "beta", "gamma", "delta"]
    agent.policy_score(candidates)  # trigger lazy init
    bilinear_scores = agent._policy_bilinear_score(candidates)
    # The bilinear scores must NOT all be identical — warm_state
    # modulates different candidates differently.
    assert bilinear_scores.shape == (4,)
    score_range = float(bilinear_scores.max() - bilinear_scores.min())
    assert score_range > 1e-6, (
        f"bilinear scores are uniform (range={score_range}), "
        "warm_state cannot discriminate candidates"
    )


def test_bilinear_score_varies_with_d_cortex() -> None:
    """Different d_cortex values must produce different bilinear score
    patterns, confirming that the warm_state dimension actually matters
    for candidate discrimination."""
    candidates = ["alpha", "beta", "gamma", "delta"]
    score_patterns: list[np.ndarray] = []
    for d in [2, 8, 32]:
        agent = _make_agent(use_policy_head=True, d_cortex=d, hidden_dim=16)
        agent.cortex.warm_state = np.random.default_rng(99).normal(
            0, 1, size=agent.cortex.warm_state.shape
        ).astype(np.float32)
        agent.policy_score(candidates)  # trigger lazy init
        bilinear_scores = agent._policy_bilinear_score(candidates)
        score_patterns.append(bilinear_scores)
    # At least two of the three patterns must differ — different d_cortex
    # produces different warm_state dimensions and different W_bilinear.
    diffs = 0
    for i in range(len(score_patterns)):
        for j in range(i + 1, len(score_patterns)):
            if not np.allclose(score_patterns[i], score_patterns[j], atol=1e-8):
                diffs += 1
    assert diffs >= 2, (
        "bilinear score patterns are identical across d_cortex values, "
        "d_cortex has no effect on candidate discrimination"
    )


def test_policy_update_trains_bilinear_weights() -> None:
    """REINFORCE update must move the bilinear weights, not just the
    linear weights."""
    agent = _make_agent(use_policy_head=True, d_cortex=4, hidden_dim=8)
    # Set non-zero warm_state so the bilinear gradient is non-zero.
    agent.cortex.warm_state = np.random.default_rng(7).normal(
        0, 1, size=agent.cortex.warm_state.shape
    ).astype(np.float32)
    candidates = ["a", "bbbbbb"]
    agent.policy_score(candidates)  # lazy init
    bilinear_before = agent._policy_W_bilinear.copy()

    agent.policy_update(candidates, chosen_idx=1, reward=1.0, baseline=0.0)

    assert not np.allclose(bilinear_before, agent._policy_W_bilinear), (
        "bilinear weights did not change after policy_update"
    )

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
