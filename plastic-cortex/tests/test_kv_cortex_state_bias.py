"""Tests for KVCortex articulation state bias."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLASTIC_CORTEX_SRC = _REPO_ROOT / "plastic-cortex" / "src"
if str(_PLASTIC_CORTEX_SRC) not in sys.path:
    sys.path.insert(0, str(_PLASTIC_CORTEX_SRC))

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig  # noqa: E402


def _cfg(**kw) -> KVCortexConfig:
    return KVCortexConfig(d_cortex=16, d_embd=32, n_layers=3, seed=0, **kw)


def _rand_hidden(d_embd: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(d_embd).astype(np.float32)


def test_set_state_bias_changes_emitted_cvecs() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(1)

    cortex.observe(_rand_hidden(cortex.config.d_embd, rng))

    before = cortex.emit_all_cvecs()
    warm_before = cortex.warm_state.copy()

    bias = rng.standard_normal(cortex.config.d_cortex).astype(np.float32)
    assert not np.allclose(bias, 0)
    cortex.set_state_bias(bias)

    after = cortex.emit_all_cvecs()
    assert not np.array_equal(cortex.warm_state, bias)  # warm_state unchanged
    assert np.array_equal(cortex.warm_state, warm_before)

    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        assert not np.allclose(b, a), f"cvec {i} did not change after setting bias"

    # Resetting to zero should restore the original cvecs.
    cortex.set_state_bias(np.zeros(cortex.config.d_cortex, dtype=np.float32))
    reset = cortex.emit_all_cvecs()
    for i, (b, r) in enumerate(zip(before, reset, strict=True)):
        np.testing.assert_array_equal(
            b, r, err_msg=f"layer {i}: zero-bias cvec differs from baseline"
        )


def test_zero_state_bias_leaves_cvecs_unchanged() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(2)

    cortex.observe(_rand_hidden(cortex.config.d_embd, rng))
    before = cortex.emit_all_cvecs()

    # set_state_bias with an explicitly-typed zero vector.
    cortex.set_state_bias(np.zeros(cortex.config.d_cortex, dtype=np.float32))
    after = cortex.emit_all_cvecs()

    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        np.testing.assert_array_equal(
            b, a, err_msg=f"layer {i}: zero bias unexpectedly altered the cvec"
        )


def test_set_state_bias_validates_shape_and_casts_dtype() -> None:
    cortex = KVCortex(_cfg())

    wrong_shape = np.zeros(cortex.config.d_cortex + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        cortex.set_state_bias(wrong_shape)

    float64_bias = np.zeros(cortex.config.d_cortex, dtype=np.float64)
    cortex.set_state_bias(float64_bias)
    assert cortex.state_bias.dtype == np.float32


def test_set_state_bias_copies_input() -> None:
    cortex = KVCortex(_cfg())
    bias = np.ones(cortex.config.d_cortex, dtype=np.float32) * 0.5
    cortex.set_state_bias(bias)
    bias[:] = 0.0
    assert np.allclose(cortex.state_bias, 0.5)


def test_status_reports_state_bias() -> None:
    cortex = KVCortex(_cfg())
    status = cortex.status()
    assert status["state_bias_norm"] == 0.0
    assert status["has_state_bias"] is False

    bias = np.ones(cortex.config.d_cortex, dtype=np.float32)
    cortex.set_state_bias(bias)
    status = cortex.status()
    assert status["state_bias_norm"] == float(np.linalg.norm(bias))
    assert status["has_state_bias"] is True
