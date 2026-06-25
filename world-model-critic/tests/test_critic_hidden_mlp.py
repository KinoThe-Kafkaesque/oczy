"""Tests for WorldModelCritic hidden-vector MLP mode."""

from __future__ import annotations

import pickle

import numpy as np

from world_model_critic import WorldModelCritic

_Q = "What is the capital of France?"
_A = "I think it might be Paris, maybe."


def test_hidden_mlp_instantiation():
    """MLP-mode config options are accepted and weights stay lazy."""
    critic = WorldModelCritic(
        config={"d_hidden": 8, "mlp_hidden_units": 16, "use_hidden": True}
    )
    assert critic.d_hidden == 8
    assert critic.mlp_hidden_units == 16
    assert critic.use_hidden is True
    assert critic.W1 is None
    assert critic.b1 is None
    assert critic.W2 is None
    assert critic.b2 == 0.0


def test_hidden_vector_changes_prediction():
    """Providing a hidden vector must change the MLP prediction vs string-only."""
    critic = WorldModelCritic(
        config={"d_hidden": 8, "mlp_hidden_units": 16, "use_hidden": True}
    )
    string_only = critic.predict_acceptance(_Q, _A)["correction_likelihood"]
    hidden = np.zeros(8, dtype=float)
    with_hidden = critic.predict_acceptance(_Q, _A, lm_hidden=hidden)[
        "correction_likelihood"
    ]
    assert with_hidden != string_only


def test_hidden_mode_respects_gates():
    """use_hidden=False or missing lm_hidden falls back to string behavior."""
    critic = WorldModelCritic(config={"d_hidden": 8, "use_hidden": False})
    hidden = np.zeros(8, dtype=float)
    pred_no_hidden = critic.predict_acceptance(_Q, _A)["correction_likelihood"]
    pred_with_hidden = critic.predict_acceptance(_Q, _A, lm_hidden=hidden)[
        "correction_likelihood"
    ]
    assert pred_no_hidden == pred_with_hidden
    assert critic.W1 is None

    critic_on = WorldModelCritic(config={"d_hidden": 8, "use_hidden": True})
    critic_on.predict_acceptance(_Q, _A)["correction_likelihood"]
    assert critic_on.W1 is None


def test_mlp_lazy_initializes() -> None:
    """Calling predict_acceptance with a hidden vector creates W1/W2 on demand."""
    critic = WorldModelCritic(config={"use_hidden": True, "mlp_hidden_units": 16})
    assert critic.d_hidden == 0
    assert critic.W1 is None

    hidden = np.zeros(8, dtype=float)
    critic.predict_acceptance(_Q, _A, lm_hidden=hidden)

    assert critic.d_hidden == 8
    assert critic.W1 is not None
    assert critic.b1 is not None
    assert critic.W2 is not None
    assert critic.W1.shape == (16, 4 + 8)
    assert critic.b1.shape == (16,)
    assert critic.W2.shape == (16,)


def test_record_outcome_updates_mlp_weights():
    """Recording an outcome with lm_hidden must move at least W2 from its init."""
    critic = WorldModelCritic(
        config={
            "d_hidden": 8,
            "mlp_hidden_units": 16,
            "use_hidden": True,
            "learning_rate": 0.1,
        }
    )
    hidden = np.random.randn(8).astype(float)
    # Trigger lazy init.
    critic.predict_acceptance(_Q, _A, lm_hidden=hidden)
    assert critic.W2 is not None
    w2_before = critic.W2.copy()

    critic.record_outcome(_Q, _A, "Actually, it's Paris.", lm_hidden=hidden)
    assert not np.allclose(critic.W2, w2_before)
    assert critic.b1 is not None
    assert not np.allclose(critic.b1, 0.0) or not np.allclose(critic.b2, 0.0)


def test_mlp_trains_on_outcome() -> None:
    """Recording a correction outcome moves the same-hidden prediction toward correction."""
    critic = WorldModelCritic(
        config={
            "d_hidden": 8,
            "use_hidden": True,
            "mlp_hidden_units": 16,
            "learning_rate": 1.0,
        }
    )
    hidden = np.random.RandomState(0).randn(8).astype(float)
    before = critic.predict_acceptance(_Q, _A, lm_hidden=hidden)["correction_likelihood"]
    critic.record_outcome(_Q, _A, "Actually, it's Paris.", lm_hidden=hidden)
    after = critic.predict_acceptance(_Q, _A, lm_hidden=hidden)["correction_likelihood"]
    assert after > before


def test_pickle_roundtrip_preserves_mlp_weights():
    """Saving and loading a hidden-mode critic keeps the MLP state."""
    critic = WorldModelCritic(
        config={"d_hidden": 8, "mlp_hidden_units": 16, "use_hidden": True}
    )
    hidden = np.random.randn(8).astype(float)
    critic.predict_acceptance(_Q, _A, lm_hidden=hidden)
    critic.record_outcome(_Q, _A, "Correction text", lm_hidden=hidden)

    restored = pickle.loads(pickle.dumps(critic, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored.d_hidden == critic.d_hidden
    assert restored.mlp_hidden_units == critic.mlp_hidden_units
    assert restored.use_hidden == critic.use_hidden
    np.testing.assert_array_equal(restored.W1, critic.W1)
    np.testing.assert_array_equal(restored.b1, critic.b1)
    np.testing.assert_array_equal(restored.W2, critic.W2)
    assert restored.b2 == critic.b2


def test_string_only_api_unchanged():
    """Default instantiation continues to behave exactly like v1."""
    critic = WorldModelCritic()
    pred = critic.predict_acceptance(_Q, _A)
    assert 0.0 <= pred["accepted_prob"] <= 1.0
    assert 0.0 <= pred["correction_likelihood"] <= 1.0
    assert critic.W1 is None
    critic.record_outcome(_Q, _A, None)
    assert len(critic.records) == 1
