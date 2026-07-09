"""Tests for the optional hidden-state delta path in ExperienceAutoencoder."""

import pickle

import numpy as np

from experience_autoencoder import ExperienceAutoencoder
from experience_autoencoder.autoencoder import (
    HEBBIAN_LR,
    LATENT_DIM,
    OUTCOME_DIM,
    RESIDUAL_DIM,
)


def _text_episode(outcome: str = "corrected") -> dict:
    return {
        "situation": "What is 2 + 2?",
        "model_answer": "5",
        "correction": "The sum is 4.",
        "revised_answer": "4",
        "outcome": outcome,
    }


def _hidden_delta(d: int = 16, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(d) * scale


def test_default_uses_text_path():
    """An episode without hidden_delta must use the legacy text-token path."""
    ae = ExperienceAutoencoder()
    delta_z = ae.encode(_text_episode())

    assert delta_z.shape == (LATENT_DIM,)
    assert delta_z.dtype == float
    assert not ae.config["use_hidden_delta"]
    assert ae._A_hidden is None
    assert ae._d_hidden is None


def test_hidden_delta_lazy_initializes():
    """encode_hidden_delta creates _A_hidden with the correct shape."""
    ae = ExperienceAutoencoder(config={"use_hidden_delta": True})
    assert ae._A_hidden is None

    d = 16
    delta_h = _hidden_delta(d)
    ae.encode({"hidden_delta": delta_h, "outcome": "corrected"})

    assert ae._A_hidden is not None
    assert ae._d_hidden == d
    assert ae._A_hidden.shape == (RESIDUAL_DIM, d)
    norms = np.linalg.norm(ae._A_hidden, axis=0)
    np.testing.assert_allclose(norms, np.ones(d), atol=1e-6)


def test_hidden_delta_changes_with_input():
    """Different hidden deltas produce different Δz vectors."""
    ae = ExperienceAutoencoder(config={"use_hidden_delta": True})
    z1 = ae.encode({"hidden_delta": _hidden_delta(d=16, scale=1.0)})
    z2 = ae.encode({"hidden_delta": _hidden_delta(d=16, scale=3.0)})
    assert z2.shape == (LATENT_DIM,)
    assert not np.allclose(z1, z2)
    assert z1[:OUTCOME_DIM].argmax() == z2[:OUTCOME_DIM].argmax()
def test_train_step_hidden_delta_updates_matrix():
    """Repeated similar deltas reduce reconstruction error."""
    ae = ExperienceAutoencoder(
        config={"use_hidden_delta": True, "hidden_delta_lr": HEBBIAN_LR}
    )
    d = 16
    delta_h = _hidden_delta(d)

    errors = []
    for _ in range(30):
        err = ae.train_step({"hidden_delta": delta_h, "outcome": "corrected"})
        errors.append(err)

    assert len(errors) == 30
    # Error should trend downward over repeated presentations.
    assert np.mean(errors[-5:]) < np.mean(errors[:5])


def test_pickle_roundtrip_hidden_delta():
    """Pickle preserves _A_hidden shape/config."""
    ae = ExperienceAutoencoder(config={"use_hidden_delta": True})
    ae.encode({"hidden_delta": _hidden_delta(16), "outcome": "failed"})

    serialized = pickle.dumps(ae, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(serialized)

    assert restored.config["use_hidden_delta"] is True
    assert restored._d_hidden == 16
    assert restored._A_hidden is not None
    assert restored._A_hidden.shape == ae._A_hidden.shape
    np.testing.assert_allclose(restored._A_hidden, ae._A_hidden)

    # A fresh encode on the restored object must succeed.
    z = restored.encode({"hidden_delta": _hidden_delta(16), "outcome": "accepted"})
    assert z.shape == (LATENT_DIM,)


def test_old_pickle_without_hidden_state_attrs():
    """Objects pickled before the hidden-delta additions can load."""
    ae = ExperienceAutoencoder(config={"use_hidden_delta": False})
    ae.encode(_text_episode())

    # Simulate the pre-hidden-delta state by deleting new attributes.
    state = ae.__dict__.copy()
    del state["_A_hidden"]
    del state["_d_hidden"]
    del state["_hidden_delta_stats"]
    # Also simulate an older config dict.
    state["config"] = {"seed": 42}

    restored = ExperienceAutoencoder.__new__(ExperienceAutoencoder)
    restored.__setstate__(state)

    assert restored.config["use_hidden_delta"] is False
    assert restored.config["hidden_delta_lr"] == HEBBIAN_LR
    assert restored._A_hidden is None
    assert restored._d_hidden is None
    assert restored._hidden_delta_stats["count"] == 0.0
    # Legacy text path still works.
    z = restored.encode(_text_episode())
    assert z.shape == (LATENT_DIM,)
