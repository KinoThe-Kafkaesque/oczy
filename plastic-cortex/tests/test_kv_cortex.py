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


def test_svd_init_proj_c_structure() -> None:
    """init_proj_c_from_svd lands proj_c on the leading singular directions
    of the supplied hiddens, broadcast identically across all layers, and
    round-trips byte-for-byte through pickle (so SVD-init'd direction
    survives cold boot -- the persistence fix this method exists for)."""
    cfg = _cfg()
    cortex = KVCortex(cfg)
    rng = np.random.default_rng(11)

    # Build hiddens with a strong rank-1 structure along a known direction
    # so the leading right singular vector is well-defined and checkable.
    leading = rng.standard_normal(cfg.d_embd).astype(np.float32)
    leading /= np.linalg.norm(leading)
    n = max(cfg.d_cortex * 2, 64)
    hiddens = (
        np.outer(rng.standard_normal(n), leading) * 10.0
        + rng.standard_normal((n, cfg.d_embd)) * 0.1
    ).astype(np.float32)

    proj_before = cortex.proj_c.copy()
    cortex.init_proj_c_from_svd(hiddens)

    # 1. projector actually changed.
    assert not np.allclose(cortex.proj_c, proj_before), \
        "init_proj_c_from_svd left proj_c unchanged"

    # 2. all layers share the same slab (broadcast condition).
    for i in range(1, cfg.n_layers):
        assert np.array_equal(cortex.proj_c[0], cortex.proj_c[i]), \
            "proj_c slab at layer %d differs from layer 0" % i

    # 3. columns are the top-d_cortex right singular vectors of the
    # centered hiddens, scaled by 1/sqrt(d_cortex). Recompute locally
    # and compare. proj_c[0] has shape (d_embd, d_cortex); Vt[:d] has
    # shape (d, d_embd), so proj_c[0].T should match Vt/sqrt(d).
    centered = hiddens - hiddens.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    expected = (Vt[: cfg.d_cortex] / np.sqrt(cfg.d_cortex)).astype(np.float32)
    assert np.allclose(cortex.proj_c[0].T, expected, atol=1e-5), \
        "proj_c slab does not match the top-d_cortex right singular vectors"

    # 4. column norms are 1/sqrt(d_cortex) (matches proj_random bound
    # convention so emit_cvec magnitudes are comparable across modes).
    col_norms = np.linalg.norm(cortex.proj_c[0], axis=0)
    expected_norm = 1.0 / np.sqrt(cfg.d_cortex)
    assert np.allclose(col_norms, expected_norm, atol=1e-5), \
        "column norms deviated from 1/sqrt(d_cortex)"

    # 5. byte-for-byte round-trip: SVD-init'd projector survives
    # save/load exactly (this is the persistence fix, exercised at the
    # contract level -- CortexAgent.load restores proj_c unmodified).
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cortex_svd.pkl"
        cortex.save(path)
        loaded = KVCortex.load(path)
    assert np.array_equal(loaded.proj_c, cortex.proj_c), \
        "SVD-init'd proj_c did not round-trip through pickle"

    # 6. emit_cvec still produces the expected shape after SVD-init.
    cortex.observe(_rand_hidden(cfg.d_embd, rng), correction_signal=1.0)
    for layer_idx in range(cfg.n_layers):
        assert cortex.emit_cvec(layer_idx).shape == (cfg.d_embd,), \
            "emit_cvec shape broke at layer %d after SVD-init" % layer_idx


def test_svd_init_rejects_undersized_hiddens() -> None:
    """SVD needs N >= d_cortex; fewer should raise, not silently degrade."""
    cfg = _cfg()
    cortex = KVCortex(cfg)
    rng = np.random.default_rng(12)
    too_few = rng.standard_normal((cfg.d_cortex - 1, cfg.d_embd)).astype(np.float32)
    try:
        cortex.init_proj_c_from_svd(too_few)
    except ValueError:
        return
    raise AssertionError("init_proj_c_from_svd accepted N < d_cortex without error")


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
        ("svd_init_proj_c_structure", test_svd_init_proj_c_structure),
        ("svd_init_rejects_undersized_hiddens", test_svd_init_rejects_undersized_hiddens),
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