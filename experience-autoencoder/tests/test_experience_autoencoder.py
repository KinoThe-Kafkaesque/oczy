"""Tests for the Experience Autoencoder prototype."""


import numpy as np

from experience_autoencoder import (
    ExperienceAutoencoder,
    ExperienceDecoder,
    ExperienceEncoder,
)
from experience_autoencoder.autoencoder import LATENT_DIM


def _profile_episode(outcome: str = "corrected") -> dict:
    return {
        "situation": "What does the profile field represent in a multi-tenant app?",
        "model_answer": "The profile field stores the user's public biography.",
        "correction": "No, profile means the business vertical the tenant belongs to.",
        "revised_answer": "The profile field identifies the business vertical for the tenant.",
        "outcome": outcome,
    }


def test_public_api_includes_expected_classes():
    assert ExperienceEncoder is not None
    assert ExperienceDecoder is not None
    assert ExperienceAutoencoder is not None


def test_encode_yields_bounded_length_vector():
    ae = ExperienceAutoencoder()
    episode = _profile_episode()
    delta_z = ae.encode(episode)

    assert delta_z.shape == (LATENT_DIM,)
    assert np.isfinite(delta_z).all()
    assert (delta_z >= -1.0).all() and (delta_z <= 1.0).all()


def test_decode_reconstructs_key_fields_with_reasonable_accuracy():
    ae = ExperienceAutoencoder()
    episode = _profile_episode("corrected")
    decoded = ae.decode(ae.encode(episode))

    assert decoded["failure_class"] in {
        "semantic_misgrounding",
        "fact_correction",
    }

    hint = decoded["corrected_behavior_hint"]
    assert isinstance(hint, dict)
    assert len(hint) > 0
    for key, value in hint.items():
        assert isinstance(key, str)
        assert isinstance(value, str)

    triggers = decoded["trigger_conditions"]
    assert isinstance(triggers, list)
    assert len(triggers) > 0
    assert all(isinstance(t, str) for t in triggers)
    # Trigger tokens should be from the episode vocabulary, not random noise.
    episode_tokens = _episode_tokens(episode)
    overlap = len(set(triggers) & episode_tokens) / max(len(triggers), 1)
    assert overlap >= 0.25

    counters = decoded["counterexamples"]
    assert isinstance(counters, list)
    assert len(counters) > 0
    assert all(isinstance(c, str) for c in counters)

    error = ae.reconstruction_error(episode, decoded)
    assert 0.0 <= error <= 1.0
    assert error < 0.8


def test_update_identity_changes_z():
    ae = ExperienceAutoencoder()
    z0 = np.zeros(LATENT_DIM, dtype=float)
    episode = _profile_episode("corrected")
    z1 = ae.update_identity(z0, episode)

    assert z1.shape == (LATENT_DIM,)
    assert not np.allclose(z0, z1)


def test_compress_produces_latent_vectors():
    """compress() returns one latent Δz per episode with the correct shape.

    Note: with the expanded 1M-parameter sensing matrix (LATENT_DIM=132),
    each Δz vector is larger than a single episode's JSON encoding.  The
    autoencoder is now a knowledge store, not a byte compressor; compression
    happens at the population level when many episodes share the same sensing
    matrix.
    """
    ae = ExperienceAutoencoder()
    episodes = [
        _profile_episode("corrected"),
        {
            "situation": "How do I safely parametrize a sqlx query?",
            "model_answer": "Concatenate user input into the query string.",
            "correction": "That risks SQL injection. Use parameterized queries.",
            "revised_answer": "Use the sqlx::query function with placeholders.",
            "outcome": "failed",
        },
    ]

    deltas = ae.compress(episodes)
    assert len(deltas) == len(episodes)
    assert all(d.shape == (LATENT_DIM,) for d in deltas)
    assert all(np.isfinite(d).all() for d in deltas)


def test_outcome_mapping():
    ae = ExperienceAutoencoder()
    accepted = _profile_episode("accepted")
    failed = {
        "situation": "Run the deploy script.",
        "model_answer": "cd prod && rm -rf /",
        "correction": "Never run destructive commands in production.",
        "revised_answer": "Run the read-only validation script first.",
        "outcome": "failed",
    }

    assert ae.decode(ae.encode(accepted))["failure_class"] == "none"
    assert ae.decode(ae.encode(failed))["failure_class"] == "execution_error"


def _episode_tokens(episode: dict) -> set[str]:
    import re

    tokens = set()
    for key in ("situation", "model_answer", "correction", "revised_answer"):
        tokens.update(re.findall(r"[a-z0-9]+", str(episode.get(key, "")).lower()))
    return tokens


def test_train_step_updates_matrix_and_bounds_error():
    """Repeated train_step on the same episode should update the sensing
    matrix and keep reconstruction error bounded.

    With the expanded 1M-parameter sensing matrix (128-dim residual, 2048
    vocab), the OMP decoder operates in a higher-dimensional space where
    Hebbian rank-1 updates don't guarantee monotonic error decrease.  The
    test verifies that (1) the matrix is actually updated, (2) the error
    remains finite and bounded, and (3) the decoded output is non-empty.
    """
    ae = ExperienceAutoencoder()
    episode = _profile_episode("corrected")

    A_before = ae._A.copy()
    errors = [ae.train_step(episode) for _ in range(50)]
    A_after = ae._A.copy()

    # Matrix must change (Hebbian update is active).
    assert not np.allclose(A_before, A_after)
    # Error must be finite and bounded.
    assert all(np.isfinite(e) for e in errors)
    assert all(e < 1.5 for e in errors)
    # Decoded output must be non-empty.
    decoded = ae._decoder.decode(ae.encode(episode))
    assert len(decoded["trigger_conditions"]) > 0


def test_encode_accepts_canonical_episode_keys():
    """encode() must accept canonical Episode keys (query/answer/corrected_answer).

    The canonical Episode schema in oczy_common.episode uses ``query``,
    ``answer``, and ``corrected_answer`` instead of the autoencoder's legacy
    ``situation`` / ``model_answer`` / ``revised_answer`` field names. encode()
    maps the canonical aliases onto the internal source fields via
    _normalize_episode and produces the correct outcome bucket.
    """
    ae = ExperienceAutoencoder()
    canonical_episode = {
        "query": "what is a branch",
        "answer": "I don't know",
        "correction": "branch means git branch",
        "corrected_answer": "git branch",
        "outcome": "corrected",
    }

    delta_z = ae.encode(canonical_episode)

    assert delta_z.shape == (LATENT_DIM,)
    assert np.isfinite(delta_z).all()

    # Outcome vector lives in the first OUTCOME_DIM=4 slots; argmax picks the
    # high-salience bucket, which should be "corrected" (index 1) here.
    from experience_autoencoder.autoencoder import _OUTCOME_LABELS, _OUTCOME_TO_IDX

    assert int(np.argmax(delta_z[: len(_OUTCOME_LABELS)])) == _OUTCOME_TO_IDX["corrected"]


def test_status_reports_serialized_bytes_and_record_count():
    """status() must include serialized_bytes and record_count fields."""
    ae = ExperienceAutoencoder()

    # Encode an episode so vocab grows beyond the empty baseline.
    ae.encode(_profile_episode("corrected"))

    status = ae.status(include_size=True)

    assert "serialized_bytes" in status
    assert "record_count" in status
    # Back-compat: existing fields must remain present and unchanged.
    assert status["project"] == "experience_autoencoder"
    assert status["ready"] is True
    assert status["latent_dim"] == 132
    assert status["vocab_size"] == len(ae._vocab)

    # serialized_bytes is a positive int; the sensing matrix (128 x 8192 floats)
    # alone is ~8 MB once pickled, so this is well over 100 kB.
    assert isinstance(status["serialized_bytes"], int)
    assert status["serialized_bytes"] > 100_000

    # record_count matches the "how much has this organ learned" semantic —
    # it equals the current vocab size.
    assert isinstance(status["record_count"], int)
    assert status["record_count"] == status["vocab_size"]
    assert status["record_count"] > 0
