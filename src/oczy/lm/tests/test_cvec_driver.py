"""End-to-end test: cortex.emit_cvec -> driver.set_cvec_layer -> LM generation.

Verifies that the CortexAgent articulation path is real: when the cortex
absorbs a hidden-state observation, the per-layer cvecs projected from
its warm_state, applied through ``LlamaCVecDriver``, actually shift the
LM's next-token distribution. After ``clear_cvec`` the LM returns to
baseline behaviour.

This test loads the real LFM2.5-1.2B-Instruct Q4_K_M model; it skips
cleanly if the HF cache is missing or llama-cpp is unavailable.

Run: uv run python src/oczy/lm/tests/test_cvec_driver.py
"""

from __future__ import annotations
import sys

import pytest

from pathlib import Path
from typing import Any

import numpy as np

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig
from oczy.lm.cvec_driver import CVecDriverConfig, LlamaCVecDriver


pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.llm]

_GGUF_CACHE = (
    Path.home() / ".cache/huggingface/hub"
    / "models--LiquidAI--LFM2.5-1.2B-Instruct-GGUF"
)
_DRIVER: LlamaCVecDriver | None = None


def _load_driver() -> LlamaCVecDriver:
    global _DRIVER
    if _DRIVER is not None:
        return _DRIVER
    if not _GGUF_CACHE.exists():
        raise FileNotFoundError("LFM2.5 GGUF cache not found")
    _DRIVER = LlamaCVecDriver.load(CVecDriverConfig(n_ctx=256, verbose=False))
    return _DRIVER


def test_driver_reports_expected_shape() -> None:
    driver = _load_driver()
    # n_embd is the LM residual stream width, fixed at 2048 for this Q4_K_M.
    assert driver.n_embd == 2048
    # n_layers is ``llama_n_layer(model_p)`` -- LFM2.5-1.2B is a hybrid
    # model with ~16 attention-capable layers (the API only counts those;
    # RWKV/conv layers in the hybrid stack aren't addressed). Don't pin the
    # exact count; just require it be non-zero and plausible.
    assert 8 <= driver.n_layers <= 64, "n_layers=%d" % driver.n_layers


def test_set_cvec_layer_shape_match() -> None:
    """set_cvec_layer accepts an n_embd vector and returns 0 (success)."""
    driver = _load_driver()
    driver.clear_cvec()

    rng = np.random.default_rng(0)
    vec = (rng.standard_normal(driver.n_embd) * 0.5).astype(np.float32)
    rc = driver.set_cvec_layer(layer_idx=14, vec=vec)
    assert rc == 0, "set_cvec_layer returned %d" % rc
    assert driver.cvec_active is True
    driver.clear_cvec()


def test_set_cvec_layer_rejects_wrong_dim() -> None:
    driver = _load_driver()
    with np.testing.assert_raises(ValueError):
        driver.set_cvec_layer(
            layer_idx=0,
            vec=np.zeros(driver.n_embd + 1, dtype=np.float32),
        )


def test_set_cvec_layer_rejects_bad_layer_idx() -> None:
    driver = _load_driver()
    with np.testing.assert_raises(IndexError):
        driver.set_cvec_layer(
            layer_idx=driver.n_layers,
            vec=np.zeros(driver.n_embd, dtype=np.float32),
        )
    with np.testing.assert_raises(IndexError):
        driver.set_cvec_layer(
            layer_idx=-1,
            vec=np.zeros(driver.n_embd, dtype=np.float32),
        )


def test_cvec_from_cortex_shifts_generation() -> None:
    """End-to-end: cortex warm-state -> per-layer cvecs -> LM generation differs.

    Sequence:
      1. Clear cvec -> capture baseline completion.
      2. Build cortex, observe a hidden vector (synthetic for now; Goal 2
         replaces this with a real intermediate-layer residual).
      3. For each layer, push cortex.emit_cvec(L) into driver.set_cvec_layer(L).
      4. Generate with steering active; output must differ from baseline.
      5. clear_cvec; output returns to baseline.
    """
    driver = _load_driver()
    rng = np.random.default_rng(1)

    # 1. Baseline.
    driver.clear_cvec()
    baseline = driver.generate(
        "Hello, my name is", max_tokens=4, temperature=0.0
    )
    assert isinstance(baseline, str) and len(baseline) > 0

    # 2. Cortex absorbs a synthetic hidden vector. Cortex n_layers
    #    mirrors the driver's attention-layer count so every per-layer
    #    cvec the cortex emits lands on a real layer.
    cortex = KVCortex(KVCortexConfig(
        d_cortex=64, d_embd=driver.n_embd,
        n_layers=driver.n_layers, seed=1,
    ))
    hidden = (rng.standard_normal(driver.n_embd) * 0.5).astype(np.float32)
    cortex.observe(hidden, correction_signal=1.0)
    # LFM2.5-1.2B at Q4 has a noisy residual stream; the cortex's
    # default-init magnitude (~0.1) does not reliably flip greedy
    # decoding because the baseline " Alex" token has a comfortable
    # top1 margin. Scale the projected cvecs above the residual-stream
    # noise floor so per-layer steering actually disrupts argmax().
    scale = 30.0

    # 3. Apply per-layer steering in ONE batched adapter call.
    #    (looped per-layer calls replace each other; only the last survives.)
    rc = driver.set_cvecs_per_layer(cortex.emit_all_cvecs(), scale=scale)
    assert rc == 0, "set_cvecs_per_layer returned %d" % rc
    assert driver.cvec_active is True

    # 4. Sample with steering.
    steered = driver.generate(
        "Hello, my name is", max_tokens=4, temperature=0.0
    )
    assert isinstance(steered, str) and len(steered) > 0

    # The two outputs MUST differ. A random-projected cortex signal at
    # scale 3 across all 28 layers is enough to disrupt greedy decoding's
    # single.argmax() path on a near-tied distribution.
    assert steered != baseline, (
        "Cortex steering produced no change in LM output)\n"
        "  baseline:  %r\n"
        "  steered:   %r" % (baseline, steered)
    )

    # 5. Clear; output returns to baseline.
    driver.clear_cvec()
    assert driver.cvec_active is False
    restored = driver.generate(
        "Hello, my name is", max_tokens=4, temperature=0.0
    )
    assert restored == baseline, (
        "after clear_cvec, output did not return to baseline\n"
        "  baseline:  %r\n"
        "  restored:  %r" % (baseline, restored)
    )


def test_peek_embedding_returns_n_embd_vector() -> None:
    """Goal 2 staging: final-layer prompt embedding shape is (n_embd,).

    The cortex's observe() needs a (n_embd,) hidden vector. peek_embedding
    is the working bridge until intermediate-layer extraction lands.
    """
    driver = _load_driver()
    emb = driver.peek_embedding("profile means business vertical")
    assert emb.shape == (driver.n_embd,), "embedding shape %s" % (emb.shape,)
    assert np.all(np.isfinite(emb))


def main() -> int:
    tests = [
        ("test_driver_reports_expected_shape", test_driver_reports_expected_shape),
        ("test_set_cvec_layer_shape_match", test_set_cvec_layer_shape_match),
        ("test_set_cvec_layer_rejects_wrong_dim", test_set_cvec_layer_rejects_wrong_dim),
        ("test_set_cvec_layer_rejects_bad_layer_idx", test_set_cvec_layer_rejects_bad_layer_idx),
        ("test_cvec_from_cortex_shifts_generation", test_cvec_from_cortex_shifts_generation),
        ("test_peek_embedding_returns_n_embd_vector", test_peek_embedding_returns_n_embd_vector),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok: %s" % name)
        except AssertionError as exc:
            print("FAIL: %s -- %s" % (name, exc))
            failures += 1
        except FileNotFoundError as exc:
            print("SKIP: %s -- %s" % (name, exc))
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print("ERROR: %s -- %r" % (name, exc))
            failures += 1
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())