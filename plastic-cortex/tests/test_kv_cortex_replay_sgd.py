"""Tests for KVCortex.replay_train_step (differentiable hippocampal replay)."""

from __future__ import annotations

import numpy as np
import pytest

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig


def _cortex(replay_sgd_step: float = 0.0) -> KVCortex:
    return KVCortex(
        KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=replay_sgd_step)
    )


def test_replay_train_step_reinforcement_moves_proj_hidden() -> None:
    cortex = _cortex(replay_sgd_step=0.1)
    h = np.random.randn(8).astype(np.float32)
    W_before = cortex.proj_hidden.copy()

    res = cortex.replay_train_step(h, target_response_sign=1.0)

    assert res["updated"] is True
    assert res["loss"] >= 0.0
    assert not np.allclose(cortex.proj_hidden, W_before)


def test_replay_train_step_suppression_moves_opposite_direction() -> None:
    cortex = _cortex(replay_sgd_step=0.1)
    h = np.random.randn(8).astype(np.float32)

    # Make a copy so the two runs are independent.
    cortex_plus = KVCortex(
        KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=0.1)
    )
    cortex_minus = KVCortex(
        KVCortexConfig(d_cortex=4, d_embd=8, n_layers=2, replay_sgd_step=0.1)
    )
    # Sync initial weights for a fair direction comparison.
    cortex_plus.proj_hidden = cortex.proj_hidden.copy()
    cortex_minus.proj_hidden = cortex.proj_hidden.copy()

    cortex_plus.replay_train_step(h, target_response_sign=1.0)
    cortex_minus.replay_train_step(h, target_response_sign=-1.0)

    plus_delta = cortex_plus.proj_hidden - cortex.proj_hidden
    minus_delta = cortex_minus.proj_hidden - cortex.proj_hidden

    # The updates should not be identical; at least one row diverges.
    assert not np.allclose(plus_delta, minus_delta)


def test_replay_train_step_disabled_when_lr_zero() -> None:
    cortex = _cortex(replay_sgd_step=0.0)
    h = np.random.randn(8).astype(np.float32)
    W_before = cortex.proj_hidden.copy()

    res = cortex.replay_train_step(h, target_response_sign=1.0)

    assert res["updated"] is False
    assert res["loss"] == 0.0
    np.testing.assert_allclose(cortex.proj_hidden, W_before)


def test_replay_train_step_invalid_hidden_dim_raises() -> None:
    cortex = _cortex(replay_sgd_step=0.1)
    with pytest.raises(ValueError):
        cortex.replay_train_step(np.ones(4, dtype=np.float32), target_response_sign=1.0)


def test_replay_train_step_lr_override() -> None:
    cortex = _cortex(replay_sgd_step=0.0)
    h = np.random.randn(8).astype(np.float32)
    W_before = cortex.proj_hidden.copy()

    res = cortex.replay_train_step(h, target_response_sign=1.0, lr=0.1)

    assert res["updated"] is True
    assert not np.allclose(cortex.proj_hidden, W_before)
