"""Tests for the WorldModelCritic learned value head."""

from __future__ import annotations

import pickle

import numpy as np

from world_model_critic import WorldModelCritic

_Q = "What is the capital of France?"
_A = "I think it might be Paris, maybe."


def test_default_value_head_disabled() -> None:
    """Default config disables the value head and predict_value returns 0.0."""
    critic = WorldModelCritic()
    assert critic.use_value_head is False
    hidden = np.ones((8,), dtype=np.float32)
    assert critic.predict_value(_Q, _A, hidden) == 0.0


def test_value_head_lazy_initializes() -> None:
    """Enabling the value head lazily creates Wv when predict_value is called."""
    critic = WorldModelCritic(
        {
            "use_value_head": True,
            "use_hidden": True,
            "mlp_hidden_units": 16,
        }
    )
    assert critic.Wv is None
    hidden = np.ones((8,), dtype=np.float32)
    value = critic.predict_value(_Q, _A, hidden)
    assert critic.Wv is not None
    assert critic.Wv.shape == (16,)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_record_outcome_trains_value_head() -> None:
    """Repeated corrections with the same hidden vector should lower its value."""
    rng = np.random.RandomState(42)
    hidden_dim = 8
    hidden = rng.randn(hidden_dim).astype(np.float32)
    critic = WorldModelCritic(
        {
            "use_value_head": True,
            "use_hidden": True,
            "mlp_hidden_units": 16,
            "value_learning_rate": 0.5,
            "learning_rate": 0.0,
            "seed": 42,
        }
    )
    # Deterministic hidden vector for the update.
    first_value = critic.predict_value(_Q, _A, hidden)
    for _ in range(5):
        critic.record_outcome(_Q, _A, "correction", lm_hidden=hidden)
    final_value = critic.predict_value(_Q, _A, hidden)
    assert final_value < first_value
    assert critic._last_value is not None
    assert critic._last_td_error is not None


def test_predict_value_changes_with_hidden() -> None:
    """Different hidden vectors must generally produce different value estimates."""
    critic = WorldModelCritic(
        {
            "use_value_head": True,
            "use_hidden": True,
            "mlp_hidden_units": 16,
        }
    )
    hidden_a = np.ones((8,), dtype=np.float32)
    hidden_b = -np.ones((8,), dtype=np.float32)
    value_a = critic.predict_value(_Q, _A, hidden_a)
    value_b = critic.predict_value(_Q, _A, hidden_b)
    assert value_a != value_b


def test_pickle_roundtrip_value_head() -> None:
    """Pickle preserves value-head config and learned weights."""
    critic = WorldModelCritic(
        {
            "use_value_head": True,
            "use_hidden": True,
            "mlp_hidden_units": 16,
        }
    )
    hidden = np.ones((8,), dtype=np.float32)
    critic.predict_value(_Q, _A, hidden)
    data = pickle.dumps(critic, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(data)
    assert restored.use_value_head is True
    assert restored.Wv is not None
    np.testing.assert_array_equal(restored.Wv, critic.Wv)
    assert restored.bv == critic.bv


def test_next_lm_hidden_used_in_td_target() -> None:
    """Supplying next_lm_hidden makes the TD target non-trivial and trains Wv."""
    rng = np.random.RandomState(0)
    hidden_dim = 8
    hidden = rng.randn(hidden_dim).astype(np.float32)
    next_hidden = rng.randn(hidden_dim).astype(np.float32)
    critic = WorldModelCritic(
        {
            "use_value_head": True,
            "use_hidden": True,
            "mlp_hidden_units": 16,
            "value_learning_rate": 0.5,
            "learning_rate": 0.0,
            "gamma": 0.95,
        }
    )
    critic.record_outcome(_Q, _A, "correction", lm_hidden=hidden, next_lm_hidden=next_hidden)
    assert critic._last_td_error != 0.0
    assert critic._last_value is not None
    assert not np.allclose(critic.Wv, 0.0) or critic.bv != 0.0
