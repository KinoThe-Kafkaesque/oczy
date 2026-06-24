"""Smoke tests for KVCortex: shape contract, warm/cold lifecycle, Hebbian training.

Verifies the cortex contract on its own: input/output shapes consolidate, warm
state mutates per observe(), cold state only mutates on consolidate(), pickle
round-trips, Hebbian training converges on a stability check. Does NOT exercise
the LM driver binding (that's ``test_cvec_driver.py``).

Run: uv run python plastic-cortex/tests/test_kv_cortex.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLASTIC_CORTEX_SRC = _REPO_ROOT / "plastic-cortex" / "src"
if str(_PLASTIC_CORTEX_SRC) not in sys.path:
    sys.path.insert(0, str(_PLASTIC_CORTEX_SRC))

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig


def _cfg(**kw) -> KVCortexConfig:
    """Build a small-rng-fitting KVCortexConfig with overrides applied."""
    defaults = dict(d_cortex=32, d_embd=64, n_layers=4, seed=0)
    defaults.update(kw)
    return KVCortexConfig(**defaults)


def _rand_hidden(d_embd: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(d_embd).astype(np.float32)


def test_shapes() -> None:
    cfg = _cfg()
    cortex = KVCortex(cfg)
    rng = np.random.default_rng(1)

    h = _rand_hidden(cfg.d_embd, rng)
    w = cortex.observe(h)
    assert w.shape == (cfg.d_cortex,), "warm state shape mismatch"

    for layer_idx in range(cfg.n_layers):
        cvec = cortex.emit_cvec(layer_idx)
        assert cvec.shape == (cfg.d_embd,), (
            "cvec shape mismatch at layer %d (got %s)"
            % (layer_idx, cvec.shape)
        )

    all_cvecs = cortex.emit_all_cvecs()
    assert len(all_cvecs) == cfg.n_layers
    for i, v in enumerate(all_cvecs):
        assert v.shape == (cfg.d_embd,), "all_cvecs[%d] shape mismatch" % i


def test_warm_mutates_cold_does_not() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(2)

    cold_before = cortex.cold_state.copy()
    warm_before = cortex.warm_state.copy()

    for _ in range(10):
        cortex.observe(_rand_hidden(cortex.config.d_embd, rng))

    assert np.array_equal(cortex.cold_state, cold_before), \
        "cold_state mutated by observe()"
    assert not np.array_equal(cortex.warm_state, warm_before), \
        "warm_state did not mutate"


def test_consolidate_moves_warm_into_cold() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(3)

    for _ in range(15):
        cortex.observe(_rand_hidden(cortex.config.d_embd, rng))

    cold_before = cortex.cold_state.copy()
    cortex.consolidate()
    assert not np.array_equal(cortex.cold_state, cold_before), \
        "consolidate() did not move cold_state"
    assert cortex.consolidate_count == 1


def test_consolidate_replay_absorption() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(4)
    replays = [_rand_hidden(cortex.config.d_embd, rng) for _ in range(5)]

    cold_without = cortex.cold_state.copy()
    cortex.consolidate()
    cold_no_replays = cortex.cold_state.copy()

    cortex.cold_state = cold_without.copy()
    cortex.consolidate(replays=replays)
    assert not np.allclose(cortex.cold_state, cold_no_replays), \
        "replays had no effect on consolidation"


def test_correction_signal_raises_plasticity() -> None:
    cfg = _cfg()
    rng = np.random.default_rng(5)
    h = _rand_hidden(cfg.d_embd, rng)

    cortex_low = KVCortex(cfg)
    cortex_low.observe(h.copy(), correction_signal=0.0)
    low_norm = float(np.linalg.norm(cortex_low.warm_state))

    cortex_high = KVCortex(cfg)
    cortex_high.observe(h.copy(), correction_signal=1.0)
    high_norm = float(np.linalg.norm(cortex_high.warm_state))

    assert high_norm > low_norm, \
        "correction_signal=1.0 did not raise plasticity over 0.0"


def test_reset_warm_from_cold() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(6)

    for _ in range(8):
        cortex.observe(_rand_hidden(cortex.config.d_embd, rng))
    cortex.consolidate()
    cortex.reset_warm_from_cold()
    assert np.array_equal(cortex.warm_state, cortex.cold_state), \
        "cold-boot did not sync warm to cold"


def test_hebbian_training_changes_projector() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(7)

    proj_before = cortex.proj_hidden.copy()
    norms_before = np.linalg.norm(cortex.proj_hidden, axis=1)

    for _ in range(50):
        cortex.train_step(_rand_hidden(cortex.config.d_embd, rng), lr=0.01)

    assert not np.allclose(cortex.proj_hidden, proj_before), \
        "Hebbian training left proj_hidden unchanged"

    norms_after = np.linalg.norm(cortex.proj_hidden, axis=1)
    assert np.allclose(norms_after, norms_after[0], rtol=1e-3), \
        "Per-row projector norms diverged after training"
    assert 0.9 < norms_after[0] < 1.1, \
        "Renormalised norms drifted outside [0.9, 1.1]"


def test_pickle_round_trip() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(8)

    for _ in range(5):
        cortex.observe(_rand_hidden(cortex.config.d_embd, rng))
    cortex.consolidate()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cortex.pkl"
        cortex.save(path)
        loaded = KVCortex.load(path)

    assert np.array_equal(loaded.warm_state, cortex.warm_state)
    assert np.array_equal(loaded.cold_state, cortex.cold_state)
    assert np.array_equal(loaded.proj_hidden, cortex.proj_hidden)
    assert np.array_equal(loaded.proj_c, cortex.proj_c)


def test_status_contract() -> None:
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(9)
    cortex.observe(_rand_hidden(cortex.config.d_embd, rng), correction_signal=1.0)

    status = cortex.status()
    for key in (
        "project", "d_cortex", "n_layers", "warm_norm", "cold_norm",
        "warm_cold_drift", "update_count", "correction_count",
        "consolidate_count", "serialized_bytes", "record_count",
    ):
        assert key in status, "status() missing %s" % key
    assert status["project"] == "plastic_cortex.kv"
    assert status["update_count"] == 1
    assert status["correction_count"] == 1


def test_emit_cvec_cached_until_observe() -> None:
    """emit_cvec must return stable references across calls until observe()."""
    cortex = KVCortex(_cfg())
    rng = np.random.default_rng(10)
    cortex.observe(_rand_hidden(cortex.config.d_embd, rng))

    # First call: populates the cache.
    v_first = cortex.emit_cvec(0)
    raw_first = v_first.tobytes()
    # Without intervening observe, second call must return the same buffer.
    v_again = cortex.emit_cvec(0)
    assert v_again is v_first or v_again.tobytes() == raw_first, \
        "emit_cvec re-derived without observe()"

    # After observe, cached payload must change.
    cortex.observe(_rand_hidden(cortex.config.d_embd, rng))
    v_after = cortex.emit_cvec(0)
    assert v_after.tobytes() != raw_first, \
        "emit_cvec did not update after observe()"


def main() -> int:
    tests = [
        ("shapes", test_shapes),
        ("warm_mutates_cold_does_not", test_warm_mutates_cold_does_not),
        ("consolidate_moves_warm_into_cold", test_consolidate_moves_warm_into_cold),
        ("consolidate_replay_absorption", test_consolidate_replay_absorption),
        ("correction_signal_raises_plasticity", test_correction_signal_raises_plasticity),
        ("reset_warm_from_cold", test_reset_warm_from_cold),
        ("hebbian_training_changes_projector", test_hebbian_training_changes_projector),
        ("pickle_round_trip", test_pickle_round_trip),
        ("status_contract", test_status_contract),
        ("emit_cvec_cached_until_observe", test_emit_cvec_cached_until_observe),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok: %s" % name)
        except AssertionError as exc:
            print("FAIL: %s -- %s" % (name, exc))
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print("ERROR: %s -- %r" % (name, exc))
            failures += 1
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())