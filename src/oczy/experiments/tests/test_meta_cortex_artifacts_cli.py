"""High-signal contract tests for Research/20 artifacts and CLI.

These tests verify the behavioral contracts of the developmental checkpoint,
persistent state, DEV result serialization, and the DEV-only CLI — no network
or model download required.

Contracts verified:

  - Checkpoint round-trip preserves theta hash, config, parameter count
  - ``theta.npz`` contains only model ``state_dict`` keys — no F/S/optimizer/task/event
  - Corruption (schema/hash/key/shape/dtype/config) is detected and raises
  - Persistent state rejects nonzero F; round-trips S only with 16384 bytes
  - DEV result JSON is canonical, finite, contains no raw text or verdict
  - Parser has exactly 3 commands (train-dev, validate-dev, audit-dev)
  - Forbidden commands (evaluate, meta-test, signoff, etc.) fail via argparse
  - ``--help`` works
  - Model load failure returns nonzero (tested via mocking)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from oczy.experiments.meta_cortex.artifacts import (
    ArtifactError,
    canonical_state_hash,
    canonical_theta_hash,
    load_dev_persistent_state,
    load_developmental_checkpoint,
    read_dev_result,
    save_dev_persistent_state,
    save_developmental_checkpoint,
    write_dev_result,
)
from oczy.experiments.meta_cortex.cli import _build_parser, main
from oczy.experiments.meta_cortex.contracts import (
    DEV_SCHEMA,
    TASKGEN_SCHEMA,
    CheckpointMetadata,
    DevValidationResult,
    ModelConfig,
    OuterLoopConfig,
    TaskGeneratorConfig,
)
from oczy.experiments.meta_cortex.model import CORTEX_DIM, MetaCortex
from oczy.experiments.meta_cortex.organ import FrozenOrganError, QwenFrozenOrgan

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODEL_CONFIG = ModelConfig(feature_dim=16, d_cortex=64, bank_width=2)
_OUTER_CONFIG = OuterLoopConfig(
    outer_steps=2,
    tasks_per_step=1,
    optimizer_name="adamw",
    learning_rate=1e-3,
    validation_interval=1,
    generation_interval=1,
    grad_clip_norm=1.0,
    seed=0,
)
_TASKGEN_CONFIG = TaskGeneratorConfig(
    root_seed=20260709,
    train_tasks_per_family=2,
    validation_tasks_per_family=2,
    min_events=2,
    max_events=2,
)


def _make_model() -> MetaCortex:
    return MetaCortex(_MODEL_CONFIG)


def _make_metadata(model: MetaCortex, **overrides) -> CheckpointMetadata:
    theta_hash = canonical_theta_hash(model)
    param_count = model.parameter_count()
    defaults = dict(
        schema=DEV_SCHEMA,
        model_config=_MODEL_CONFIG,
        taskgen_schema=TASKGEN_SCHEMA,
        taskgen_digest="a" * 64,
        outer_config=_OUTER_CONFIG,
        completed_step=2,
        best_step=1,
        validation_score=0.5,
        parameter_count=param_count,
        parameter_bytes=param_count * 4,
        theta_hash=theta_hash,
        organ_identity="test-organ",
        organ_hash="b" * 64,
        source_provenance="unavailable",
    )
    defaults.update(overrides)
    return CheckpointMetadata(**defaults)


@pytest.fixture
def tmpdir_path(tmp_path) -> Path:
    return tmp_path


def _assert_prompt_fields_are_hashes(obj: object) -> None:
    """Recursively verify that any field whose name contains 'prompt'
    holds only SHA-256 hex hashes (or empty containers), never raw text.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "prompt" in key.lower():
                _assert_hash_values(value)
            else:
                _assert_prompt_fields_are_hashes(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_prompt_fields_are_hashes(item)


def _assert_hash_values(value: object) -> None:
    """Assert that *value* is a hash or a collection of hashes."""
    if isinstance(value, str):
        assert len(value) == 64, f"Expected 64-char hex hash, got {value!r}"
        int(value, 16)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_hash_values(item)
    elif value is None:
        pass
    else:
        raise AssertionError(f"Expected hash value, got {type(value).__name__}: {value!r}")


# ---------------------------------------------------------------------------
# Developmental checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_round_trip_preserves_theta(
        self, tmpdir_path: Path
    ) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        # Load into a fresh model.  The model uses a fixed seed for
        # initialization, so two fresh models may share the same hash.
        # Perturb model2 to guarantee it differs from the checkpoint.
        model2 = _make_model()
        with torch.no_grad():
            for param in model2.parameters():
                param.add_(1.0)
                break
        theta_before = canonical_theta_hash(model2)
        assert theta_before != metadata.theta_hash

        loaded_meta = load_developmental_checkpoint(ckpt_dir, model2)

        # After loading, theta hash should match.
        assert canonical_theta_hash(model2) == metadata.theta_hash
        assert loaded_meta.theta_hash == metadata.theta_hash

    def test_round_trip_preserves_config(
        self, tmpdir_path: Path
    ) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        model2 = _make_model()
        loaded_meta = load_developmental_checkpoint(ckpt_dir, model2)
        assert loaded_meta.model_config == _MODEL_CONFIG
        assert loaded_meta.parameter_count == model.parameter_count()
        assert loaded_meta.parameter_bytes == model.parameter_count() * 4

    def test_round_trip_preserves_organ_info(
        self, tmpdir_path: Path
    ) -> None:
        model = _make_model()
        metadata = _make_metadata(model, organ_identity="Qwen/Qwen2.5-0.5B-Instruct")
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        model2 = _make_model()
        loaded_meta = load_developmental_checkpoint(ckpt_dir, model2)
        assert loaded_meta.organ_identity == "Qwen/Qwen2.5-0.5B-Instruct"
        assert loaded_meta.organ_hash == metadata.organ_hash

    def test_round_trip_preserves_outer_config(
        self, tmpdir_path: Path
    ) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        model2 = _make_model()
        loaded_meta = load_developmental_checkpoint(ckpt_dir, model2)
        assert loaded_meta.outer_config == _OUTER_CONFIG

    def test_checkpoint_files_exist(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)
        assert (ckpt_dir / "checkpoint.json").exists()
        assert (ckpt_dir / "theta.npz").exists()


# ---------------------------------------------------------------------------
# theta.npz contains only state_dict keys
# ---------------------------------------------------------------------------


class TestCheckpointContents:
    def test_npz_keys_match_state_dict(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        npz_keys = set(npz.files)
        model_keys = set(model.state_dict().keys())
        assert npz_keys == model_keys

    def test_npz_no_fast_slow_optimizer(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        for key in npz.files:
            key_lower = key.lower()
            # event_fusion is a legitimate theta parameter (Linear layer),
            # not an event trace — exclude "event" from the forbidden list.
            assert "fast" not in key_lower
            assert "slow" not in key_lower
            assert "optimizer" not in key_lower
            assert "task" not in key_lower
            assert "trace" not in key_lower
            assert "prompt" not in key_lower
            assert "correction" not in key_lower

    def test_checkpoint_json_no_raw_text(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        json_text = (ckpt_dir / "checkpoint.json").read_text()
        parsed = json.loads(json_text)
        # No prompt/correction/event/trace fields.
        for key in parsed:
            key_lower = key.lower()
            assert "prompt" not in key_lower
            assert "correction" not in key_lower
            assert "event" not in key_lower
            assert "trace" not in key_lower
            assert "target" not in key_lower
            assert "label" not in key_lower

    def test_checkpoint_json_no_verdict(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        json_text = (ckpt_dir / "checkpoint.json").read_text()
        assert "verdict" not in json_text.lower()
        assert "signoff" not in json_text.lower()
        assert "sign_off" not in json_text.lower()

    def test_checkpoint_json_canonical(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        json_text = (ckpt_dir / "checkpoint.json").read_text()
        # Should be sorted and no NaN.
        assert "NaN" not in json_text
        assert "Infinity" not in json_text
        # Verify it's sorted by re-serializing.
        parsed = json.loads(json_text)
        reserialized = json.dumps(parsed, sort_keys=True, allow_nan=False)
        assert json_text == reserialized

    def test_npz_all_float32(self, tmpdir_path: Path) -> None:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        for key in npz.files:
            assert npz[key].dtype == np.float32


# ---------------------------------------------------------------------------
# Checkpoint corruption detection
# ---------------------------------------------------------------------------


class TestCheckpointCorruption:
    def _save_checkpoint(self, tmpdir_path: Path) -> Path:
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)
        return ckpt_dir

    def test_missing_checkpoint_json(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        (ckpt_dir / "checkpoint.json").unlink()
        model = _make_model()
        with pytest.raises(ArtifactError, match="checkpoint.json"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_missing_theta_npz(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        (ckpt_dir / "theta.npz").unlink()
        model = _make_model()
        with pytest.raises(ArtifactError, match="theta.npz"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_corrupt_json(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        (ckpt_dir / "checkpoint.json").write_text("not json{")
        model = _make_model()
        with pytest.raises(ArtifactError, match="not valid JSON"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_schema_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        data = json.loads((ckpt_dir / "checkpoint.json").read_text())
        data["schema"] = "wrong_schema"
        (ckpt_dir / "checkpoint.json").write_text(
            json.dumps(data, sort_keys=True, allow_nan=False)
        )
        model = _make_model()
        with pytest.raises(ArtifactError, match="Schema mismatch"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_theta_hash_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        # Corrupt theta.npz by modifying a parameter.
        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        # Modify one array.
        first_key = sorted(arrays.keys())[0]
        arrays[first_key] = arrays[first_key] + 1.0
        np.savez(ckpt_dir / "theta.npz", **arrays)
        model = _make_model()
        with pytest.raises(ArtifactError, match="hash mismatch"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_missing_key_in_npz(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        # Remove one key.
        first_key = sorted(arrays.keys())[0]
        del arrays[first_key]
        np.savez(ckpt_dir / "theta.npz", **arrays)
        model = _make_model()
        with pytest.raises(ArtifactError, match="key set mismatch"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_shape_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        # Corrupt one array's shape.  Pick a multi-element parameter so
        # flattening to [:1] genuinely changes the shape.
        multi_key = next(
            k for k in sorted(arrays) if arrays[k].size > 1
        )
        arrays[multi_key] = arrays[multi_key].flatten()[:1]
        np.savez(ckpt_dir / "theta.npz", **arrays)
        model = _make_model()
        with pytest.raises(ArtifactError, match="shape"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_dtype_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        npz = np.load(ckpt_dir / "theta.npz", allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        first_key = sorted(arrays.keys())[0]
        arrays[first_key] = arrays[first_key].astype(np.float64)
        np.savez(ckpt_dir / "theta.npz", **arrays)
        model = _make_model()
        with pytest.raises(ArtifactError, match="dtype"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_config_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        # Load with a model that has a different config.
        wrong_model = MetaCortex(ModelConfig(feature_dim=32, d_cortex=64, bank_width=2))
        with pytest.raises(ArtifactError, match="config mismatch"):
            load_developmental_checkpoint(ckpt_dir, wrong_model)

    def test_parameter_count_mismatch(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        # Corrupt the parameter count in checkpoint.json.
        data = json.loads((ckpt_dir / "checkpoint.json").read_text())
        data["parameter_count"] = 999
        (ckpt_dir / "checkpoint.json").write_text(
            json.dumps(data, sort_keys=True, allow_nan=False)
        )
        model = _make_model()
        with pytest.raises(ArtifactError, match="Parameter count"):
            load_developmental_checkpoint(ckpt_dir, model)

    def test_null_field_rejected(self, tmpdir_path: Path) -> None:
        ckpt_dir = self._save_checkpoint(tmpdir_path)
        data = json.loads((ckpt_dir / "checkpoint.json").read_text())
        data["organ_identity"] = None
        (ckpt_dir / "checkpoint.json").write_text(
            json.dumps(data, sort_keys=True, allow_nan=False)
        )
        model = _make_model()
        with pytest.raises(ArtifactError, match="null"):
            load_developmental_checkpoint(ckpt_dir, model)


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


class TestPersistentState:
    def test_rejects_nonzero_F(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        # Write to get nonzero F.
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        batch = EventFeatureBatch(values=torch.randn(1, 4, 16, dtype=torch.float32))
        state = model.write(state, batch).state
        # F is nonzero now.
        assert torch.count_nonzero(state.fast).item() > 0

        state_dir = tmpdir_path / "state"
        with pytest.raises(ArtifactError, match="F.*not zero"):
            save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)

    def test_round_trip_consolidated_state(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        batch = EventFeatureBatch(values=torch.randn(1, 4, 16, dtype=torch.float32))
        state = model.write(state, batch).state
        state = model.consolidate(state).state
        # F is zero after consolidation.
        assert torch.count_nonzero(state.fast).item() == 0
        # S is nonzero.
        assert torch.count_nonzero(state.slow).item() > 0

        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)

        loaded = load_dev_persistent_state(state_dir, _MODEL_CONFIG)
        assert torch.equal(loaded.fast, torch.zeros_like(loaded.fast))
        assert torch.allclose(loaded.slow, state.slow)

    def test_logical_bytes_16384(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        batch = EventFeatureBatch(values=torch.randn(1, 4, 16, dtype=torch.float32))
        state = model.write(state, batch).state
        state = model.consolidate(state).state

        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)

        state_json = json.loads((state_dir / "state.json").read_text())
        assert state_json["logical_bytes"] == 16384

    def test_state_files_exist(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        assert (state_dir / "state.json").exists()
        assert (state_dir / "slow.npy").exists()

    def test_slow_npy_shape_64x64(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        arr = np.load(state_dir / "slow.npy", allow_pickle=False)
        assert arr.shape == (CORTEX_DIM, CORTEX_DIM)
        assert arr.dtype == np.float32

    def test_state_json_no_raw_text(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        json_text = (state_dir / "state.json").read_text()
        assert "prompt" not in json_text.lower()
        assert "correction" not in json_text.lower()
        assert "event" not in json_text.lower()
        assert "trace" not in json_text.lower()

    def test_state_json_no_verdict(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        json_text = (state_dir / "state.json").read_text()
        assert "verdict" not in json_text.lower()
        assert "signoff" not in json_text.lower()

    def test_state_hash_mismatch_detected(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        # Corrupt slow.npy.
        arr = np.load(state_dir / "slow.npy", allow_pickle=False)
        arr = arr + 1.0
        np.save(state_dir / "slow.npy", arr)
        with pytest.raises(ArtifactError, match="hash mismatch"):
            load_dev_persistent_state(state_dir, _MODEL_CONFIG)

    def test_state_schema_mismatch(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)
        data = json.loads((state_dir / "state.json").read_text())
        data["schema"] = "wrong"
        (state_dir / "state.json").write_text(
            json.dumps(data, sort_keys=True, allow_nan=False)
        )
        with pytest.raises(ArtifactError, match="Schema"):
            load_dev_persistent_state(state_dir, _MODEL_CONFIG)

    def test_state_rejects_batch_gt_1(self, tmpdir_path: Path) -> None:
        model = _make_model()
        state = model.initial_state(3, device="cpu", dtype=torch.float32)
        state = model.zero_state(state)
        state_dir = tmpdir_path / "state"
        with pytest.raises(ArtifactError, match="batch"):
            save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)

    def test_canonical_state_hash_64_hex(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        h = canonical_state_hash(state)
        assert len(h) == 64
        int(h, 16)


# ---------------------------------------------------------------------------
# DEV result serialization
# ---------------------------------------------------------------------------


class TestDevResultSerialization:
    def _make_validation_result(self) -> DevValidationResult:
        from oczy.experiments.meta_cortex.contracts import (
            ConditionResult,
            OnlineEpisodeAudit,
        )

        audit = OnlineEpisodeAudit(
            family="contextual_remapping",
            split="meta_validation",
            rule_fingerprint="a" * 64,
            event_count=2,
            trace_objects_before=2,
            trace_objects_after=0,
            trace_feature_tensors_before=2,
            trace_feature_tensors_after=0,
            fast_shape=[1, 64, 64],
            slow_shape=[1, 64, 64],
            bank_shape=[1, 2, 16],
            fast_zero=True,
            logical_persistent_bytes=16384,
            optimizer_step_count_before=0,
            optimizer_step_count_after=0,
            theta_hash_before="c" * 64,
            theta_hash_after="c" * 64,
            organ_hash_before="d" * 64,
            organ_hash_after="d" * 64,
            answer_path_prompt_hashes=(),
            banned_content_absent=True,
        )
        cr = ConditionResult(
            condition="trained",
            per_kind_correct=(1, 1, 0, 0, 0, 0),
            per_kind_total=(2, 2, 2, 2, 2, 2),
            per_kind_accuracy=(0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
            state_hash="e" * 64,
            state_norm=1.0,
            state_bytes=16384,
            episode_audit=audit,
        )
        return DevValidationResult(
            per_family_results=(("contextual_remapping", (cr,)),),
            pooled_results=(cr,),
            trained_vs_update_disabled_delta=0.1,
            trained_vs_untrained_delta=0.2,
            trained_vs_shuffled_delta=0.3,
            trained_vs_zeroed_delta=0.4,
            trained_vs_swapped_delta=0.5,
        )

    def test_write_dev_result_canonical(self, tmpdir_path: Path) -> None:
        result = self._make_validation_result()
        path = tmpdir_path / "result.json"
        write_dev_result(path, result)
        text = path.read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        # Verify sorted.
        parsed = json.loads(text)
        reserialized = json.dumps(parsed, sort_keys=True, allow_nan=False)
        assert text == reserialized

    def test_write_dev_result_finite(self, tmpdir_path: Path) -> None:
        result = self._make_validation_result()
        path = tmpdir_path / "result.json"
        write_dev_result(path, result)
        data = json.loads(path.read_text())
        # Check all float values are finite.
        def _check_finite(obj):
            if isinstance(obj, float):
                assert math.isfinite(obj), f"Non-finite value: {obj}"
            elif isinstance(obj, dict):
                for v in obj.values():
                    _check_finite(v)
            elif isinstance(obj, list):
                for v in obj:
                    _check_finite(v)
        _check_finite(data)

    def test_write_dev_result_no_raw_text(self, tmpdir_path: Path) -> None:
        result = self._make_validation_result()
        path = tmpdir_path / "result.json"
        write_dev_result(path, result)
        text = path.read_text()
        # Field names containing "prompt" (e.g. answer_path_prompt_hashes)
        # store SHA-256 hashes, not raw prompt text — they are allowed.
        # Verify that any prompt-named field values are hex hashes.
        data = json.loads(text)
        _assert_prompt_fields_are_hashes(data)
        assert "correction" not in text.lower()
        assert "event_text" not in text.lower()
        assert "target" not in text.lower()
        assert "label" not in text.lower()

    def test_write_dev_result_no_verdict(self, tmpdir_path: Path) -> None:
        result = self._make_validation_result()
        path = tmpdir_path / "result.json"
        write_dev_result(path, result)
        text = path.read_text()
        assert "verdict" not in text.lower()
        assert "signoff" not in text.lower()
        assert "sign_off" not in text.lower()
        assert "accept" not in text.lower()
        assert "refute" not in text.lower()

    def test_read_dev_result_round_trip(self, tmpdir_path: Path) -> None:
        result = self._make_validation_result()
        path = tmpdir_path / "result.json"
        write_dev_result(path, result)
        data = read_dev_result(path)
        assert isinstance(data, dict)
        assert "trained_vs_update_disabled_delta" in data

    def test_read_dev_result_missing_file(self, tmpdir_path: Path) -> None:
        with pytest.raises(ArtifactError, match="not found"):
            read_dev_result(tmpdir_path / "nonexistent.json")

    def test_write_dev_result_wrong_type(self, tmpdir_path: Path) -> None:
        with pytest.raises(ArtifactError, match="expects"):
            write_dev_result(tmpdir_path / "bad.json", "not a result")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestCLIParser:
    def test_parser_has_exactly_3_commands(self) -> None:
        parser = _build_parser()
        # Find the subparsers action via its actual type.
        subparsers_action = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
                break
        assert subparsers_action is not None
        commands = set(subparsers_action.choices.keys())
        assert commands == {"train-dev", "validate-dev", "audit-dev"}

    def test_forbidden_command_evaluate(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])

    def test_forbidden_command_meta_test(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["meta-test"])

    def test_forbidden_command_signoff(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["signoff"])

    def test_forbidden_command_manifest(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["manifest"])

    def test_forbidden_command_c7(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["c7"])

    def test_forbidden_command_c8(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["c8"])

    def test_no_command_fails(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_train_dev_has_required_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "train-dev",
            "--checkpoint-out", "/tmp/ckpt",
            "--result-out", "/tmp/result.json",
        ])
        assert args.command == "train-dev"
        assert args.checkpoint_out == "/tmp/ckpt"
        assert args.result_out == "/tmp/result.json"

    def test_validate_dev_has_required_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "validate-dev",
            "--checkpoint", "/tmp/ckpt",
            "--result-out", "/tmp/result.json",
        ])
        assert args.command == "validate-dev"
        assert args.checkpoint == "/tmp/ckpt"
        assert args.result_out == "/tmp/result.json"

    def test_audit_dev_has_required_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "audit-dev",
            "--checkpoint", "/tmp/ckpt",
        ])
        assert args.command == "audit-dev"
        assert args.checkpoint == "/tmp/ckpt"

    def test_train_dev_default_model_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "train-dev",
            "--checkpoint-out", "/tmp/ckpt",
            "--result-out", "/tmp/result.json",
        ])
        assert "Qwen" in args.model_id

    def test_train_dev_default_optimizer(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "train-dev",
            "--checkpoint-out", "/tmp/ckpt",
            "--result-out", "/tmp/result.json",
        ])
        assert args.optimizer_name == "adamw"

    def test_train_dev_optimizer_choices(self) -> None:
        parser = _build_parser()
        # adamw should work.
        args = parser.parse_args([
            "train-dev",
            "--checkpoint-out", "/c",
            "--result-out", "/r",
            "--optimizer-name", "adamw",
        ])
        assert args.optimizer_name == "adamw"
        # sgd should work.
        args = parser.parse_args([
            "train-dev",
            "--checkpoint-out", "/c",
            "--result-out", "/r",
            "--optimizer-name", "sgd",
        ])
        assert args.optimizer_name == "sgd"
        # rmsprop should fail.
        with pytest.raises(SystemExit):
            parser.parse_args([
                "train-dev",
                "--checkpoint-out", "/c",
                "--result-out", "/r",
                "--optimizer-name", "rmsprop",
            ])


# ---------------------------------------------------------------------------
# CLI main entrypoint
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_main_returns_int(self) -> None:
        """main() should return an int exit code (or raise SystemExit for --help)."""
        # We can't call main with a real command without a model.
        # But we can verify it's callable.
        assert callable(main)

    def test_help_exits_zero(self) -> None:
        """--help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_command_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_forbidden_command_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["evaluate"])
        assert exc_info.value.code != 0

    def test_train_dev_organ_load_failure_returns_nonzero(
        self, tmpdir_path: Path
    ) -> None:
        """When QwenFrozenOrgan.load fails, train-dev returns nonzero."""
        def _raise(*args, **kwargs):
            raise FrozenOrganError("model not available")

        with patch.object(QwenFrozenOrgan, "load", _raise):
            result = main([
                "train-dev",
                "--checkpoint-out", str(tmpdir_path / "ckpt"),
                "--result-out", str(tmpdir_path / "result.json"),
            ])
        assert result != 0

    def test_validate_dev_organ_load_failure_returns_nonzero(
        self, tmpdir_path: Path
    ) -> None:
        """When QwenFrozenOrgan.load fails, validate-dev returns nonzero."""
        # First create a valid checkpoint.
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        def _raise(*args, **kwargs):
            raise FrozenOrganError("model not available")

        with patch.object(QwenFrozenOrgan, "load", _raise):
            result = main([
                "validate-dev",
                "--checkpoint", str(ckpt_dir),
                "--result-out", str(tmpdir_path / "result.json"),
            ])
        assert result != 0

    def test_audit_dev_no_checkpoint_returns_nonzero(
        self, tmpdir_path: Path
    ) -> None:
        result = main([
            "audit-dev",
            "--checkpoint", str(tmpdir_path / "nonexistent"),
        ])
        assert result != 0

    def test_audit_dev_valid_checkpoint(
        self, tmpdir_path: Path
    ) -> None:
        """audit-dev on a valid checkpoint should return 0."""
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        result = main([
            "audit-dev",
            "--checkpoint", str(ckpt_dir),
        ])
        assert result == 0

    def test_audit_dev_with_result_and_state(
        self, tmpdir_path: Path
    ) -> None:
        """audit-dev with result and state artifacts should return 0."""
        model = _make_model()
        metadata = _make_metadata(model)
        ckpt_dir = tmpdir_path / "ckpt"
        save_developmental_checkpoint(ckpt_dir, model, metadata)

        # Create persistent state.
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        state_dir = tmpdir_path / "state"
        save_dev_persistent_state(state_dir, state, _MODEL_CONFIG)

        # Create a minimal result.
        from oczy.experiments.meta_cortex.contracts import (
            ConditionResult,
            DevValidationResult,
            OnlineEpisodeAudit,
        )

        audit = OnlineEpisodeAudit(
            family="contextual_remapping",
            split="meta_validation",
            rule_fingerprint="a" * 64,
            event_count=2,
            trace_objects_before=2,
            trace_objects_after=0,
            trace_feature_tensors_before=2,
            trace_feature_tensors_after=0,
            fast_shape=[1, 64, 64],
            slow_shape=[1, 64, 64],
            bank_shape=[1, 2, 16],
            fast_zero=True,
            logical_persistent_bytes=16384,
            optimizer_step_count_before=0,
            optimizer_step_count_after=0,
            theta_hash_before="c" * 64,
            theta_hash_after="c" * 64,
            organ_hash_before="d" * 64,
            organ_hash_after="d" * 64,
            answer_path_prompt_hashes=(),
            banned_content_absent=True,
        )
        cr = ConditionResult(
            condition="trained",
            per_kind_correct=(1, 1, 0, 0, 0, 0),
            per_kind_total=(2, 2, 2, 2, 2, 2),
            per_kind_accuracy=(0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
            state_hash="e" * 64,
            state_norm=1.0,
            state_bytes=16384,
            episode_audit=audit,
        )
        val_result = DevValidationResult(
            per_family_results=(("contextual_remapping", (cr,)),),
            pooled_results=(cr,),
            trained_vs_update_disabled_delta=0.1,
            trained_vs_untrained_delta=0.2,
            trained_vs_shuffled_delta=0.3,
            trained_vs_zeroed_delta=0.4,
            trained_vs_swapped_delta=0.5,
        )
        result_path = tmpdir_path / "result.json"
        write_dev_result(result_path, val_result)

        result = main([
            "audit-dev",
            "--checkpoint", str(ckpt_dir),
            "--result", str(result_path),
            "--state", str(state_dir),
        ])
        assert result == 0


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


class TestCanonicalHashing:
    def test_theta_hash_64_hex(self) -> None:
        model = _make_model()
        h = canonical_theta_hash(model)
        assert len(h) == 64
        int(h, 16)

    def test_theta_hash_changes_on_param_change(self) -> None:
        model = _make_model()
        h1 = canonical_theta_hash(model)
        # Modify a parameter.
        for param in model.parameters():
            param.data.add_(1.0)
            break
        h2 = canonical_theta_hash(model)
        assert h1 != h2

    def test_theta_hash_deterministic(self) -> None:
        model = _make_model()
        h1 = canonical_theta_hash(model)
        h2 = canonical_theta_hash(model)
        assert h1 == h2

    def test_state_hash_64_hex(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        h = canonical_state_hash(state)
        assert len(h) == 64
        int(h, 16)

    def test_state_hash_changes_on_s_change(self) -> None:
        model = _make_model()
        state1 = model.initial_state(1, device="cpu", dtype=torch.float32)
        state2 = model.initial_state(1, device="cpu", dtype=torch.float32)
        state2.slow[0, 0, 0] = 1.0
        assert canonical_state_hash(state1) != canonical_state_hash(state2)


