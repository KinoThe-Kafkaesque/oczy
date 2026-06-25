"""Tests for KVCortex versioned, non-pickle state persistence.

These cover ``save_state_dict`` / ``load_state_dict``. The legacy pickle
path is intentionally exercised elsewhere and left unchanged.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

def _cfg(**kw) -> KVCortexConfig:
    defaults = {
        "d_cortex": 8,
        "d_embd": 16,
        "n_layers": 4,
        "seed": 42,
        "consolidate_replay_threshold": 2,
    }
    defaults.update(kw)
    return KVCortexConfig(**defaults)


def test_state_dict_roundtrip_preserves_arrays_and_behavior() -> None:
    rng = np.random.default_rng(7)
    cortex = KVCortex(_cfg())

    # Mutate the cortex through a realistic sequence.
    for _ in range(2):
        cortex.observe(rng.standard_normal(cortex.config.d_embd).astype(np.float32))
    cortex.observe(
        rng.standard_normal(cortex.config.d_embd).astype(np.float32),
        correction_signal=1.0,
    )
    cortex.consolidate(replays=[])

    original_cold = cortex.cold_state.copy()
    original_warm = cortex.warm_state.copy()
    original_cvec = cortex.emit_cvec(0).copy()
    original_counts = {
        "update_count": cortex.update_count,
        "correction_count": cortex.correction_count,
        "consolidate_count": cortex.consolidate_count,
    }

    tmpdir = Path(tempfile.mkdtemp())
    try:
        cortex.save_state_dict(tmpdir)

        manifest_path = tmpdir / "manifest.json"
        arrays_path = tmpdir / "arrays.npz"
        assert manifest_path.exists()
        assert arrays_path.exists()

        loaded = KVCortex.load_state_dict(tmpdir)

        assert np.allclose(loaded.cold_state, original_cold)
        assert np.allclose(loaded.warm_state, original_warm)
        assert loaded.emit_cvec(0).shape == (cortex.config.d_embd,)
        assert np.allclose(loaded.emit_cvec(0), original_cvec)

        for key, expected in original_counts.items():
            assert getattr(loaded, key) == expected

        # Loaded arrays must be writable (e.g. train_step does in-place work).
        loaded.train_step(rng.standard_normal(cortex.config.d_embd).astype(np.float32), lr=0.01)
    finally:
        shutil.rmtree(tmpdir)


def test_manifest_has_expected_version_and_class() -> None:
    cortex = KVCortex(_cfg())
    tmpdir = Path(tempfile.mkdtemp())
    try:
        cortex.save_state_dict(tmpdir)
        manifest = json.loads((tmpdir / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["version"] == 1
        assert manifest["class"] == "KVCortex"
        assert manifest["config"]["d_cortex"] == cortex.config.d_cortex
        assert manifest["config"]["d_embd"] == cortex.config.d_embd
        assert "arrays" in manifest
        assert "cold_state" in manifest["arrays"]
        assert "warm_state" in manifest["arrays"]
        assert "proj_c" in manifest["arrays"]
    finally:
        shutil.rmtree(tmpdir)


def test_state_dict_rejects_bad_class_and_version() -> None:
    tmpdir = Path(tempfile.mkdtemp())
    (tmpdir / "arrays.npz").write_bytes(b"")  # not loaded for these checks
    try:
        (tmpdir / "manifest.json").write_text(
            json.dumps({"version": 1, "class": "NotKVCortex", "config": {}}),
            encoding="utf-8",
        )
        try:
            KVCortex.load_state_dict(tmpdir)
        except ValueError as exc:
            assert "Expected class" in str(exc)
        else:
            raise AssertionError("load_state_dict accepted wrong class")

        (tmpdir / "manifest.json").write_text(
            json.dumps({"version": 0, "class": "KVCortex", "config": {}}),
            encoding="utf-8",
        )
        try:
            KVCortex.load_state_dict(tmpdir)
        except ValueError as exc:
            assert "version must be >= 1" in str(exc)
        else:
            raise AssertionError("load_state_dict accepted version 0")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_state_dict_roundtrip_preserves_arrays_and_behavior()
    test_manifest_has_expected_version_and_class()
    test_state_dict_rejects_bad_class_and_version()
    print("ok")
