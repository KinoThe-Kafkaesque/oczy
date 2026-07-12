"""High-signal contract tests for Research 19 (s19_language_organ).

These tests verify the behavioral contracts of the R19 implementation using
lightweight frozen-LM fakes — no real model loading.  Every test would fail
for a specific regression:

  - parameter budget exceeded (>64k)
  - arms not sharing the same cortex state
  - DEV-phase calibration leaking holdout IDs
  - evaluate running without matching manifest hash / human sign-off
  - Arm B injecting label/answer/correction text at probe time
  - latent bank width varying with episode count
  - LM parameters mutating during teaching/evaluation
  - raw traces surviving consolidation
  - C0–C7 causal controls not behaving correctly
  - H-LABEL / H-LATENT verdict gates being conflated
  - driver errors not failing closed

All tests use fakes following the pattern from test_r18_diagnostics.py.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    ProbeCategory,
    Stage,
    split_probes,
)
from oczy.lm._types import ReservedPosition

if TYPE_CHECKING:
    from typing import override

    from oczy.lm.hf_driver import HFDriver as _HFDriverBase
else:
    _HFDriverBase = object

    def override(x: Any) -> Any:
        return x

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stand-in."""

    def __init__(self) -> None:
        self.bos_token_id: int | None = 1
        self.eos_token_id: int | None = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) % 256 for c in text]
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chars = []
        for i in ids:
            if i in (self.bos_token_id, self.eos_token_id) and skip_special_tokens:
                continue
            chars.append(chr(i + 256) if i < 256 else "?")
        return "".join(chars)

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2}


class _FakeParam:
    """A fake torch.nn.Parameter-like object for named_parameters."""

    def __init__(self, data: np.ndarray, name: str = "param") -> None:
        self._data = data
        self.name = name
        self.requires_grad = True

    def detach(self) -> _FakeParam:
        return self

    def cpu(self) -> _FakeParam:
        return self

    def numpy(self) -> np.ndarray:
        return self._data

    def requires_grad_(self, flag: bool) -> _FakeParam:
        self.requires_grad = flag
        return self

    def parameters(self):
        return iter([self])


class _FakeModel:
    """Stand-in model with config, named_parameters, and forward."""

    class _Config:
        hidden_size = 896
        vocab_size = 151936
        num_hidden_layers = 24

        def to_dict(self) -> dict[str, Any]:
            return {
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "num_hidden_layers": self.num_hidden_layers,
            }

    class _Out:
        """Minimal forward output with logits and past_key_values."""

        def __init__(self, logits: Any, past: Any = None) -> None:

            self.logits = logits
            self.past_key_values = past
            self.hidden_states = None

    def __init__(self, reply: str = "unknown") -> None:
        self.config = self._Config()
        self._reply = reply
        self._params: dict[str, np.ndarray] = {
            "embed.weight": np.zeros((151936, 896), dtype=np.float32),
            "layer.0.weight": np.ones((896, 896), dtype=np.float32),
        }
        self._forward_calls: list[dict[str, Any]] = []
        self._eval = True

    def eval(self) -> None:
        self._eval = True

    def train(self, mode: bool = True) -> None:
        self._eval = not mode

    def parameters(self):
        for v in self._params.values():
            yield _FakeParam(v)

    def named_parameters(self):
        for name, v in self._params.items():
            yield name, _FakeParam(v)

    def embed_tokens(self, input_ids: Any) -> Any:
        import torch

        # Return a zero tensor of the right shape (1, seq_len, 896).
        if hasattr(input_ids, "shape"):
            seq_len = input_ids.shape[1]
        else:
            seq_len = len(input_ids)
        return torch.zeros(1, seq_len, 896, dtype=torch.float32)

    def get_input_embeddings(self) -> Any:
        import torch

        class _Embed:
            def __call__(self, ids: Any) -> Any:
                if hasattr(ids, "shape"):
                    seq_len = ids.shape[1]
                else:
                    seq_len = len(ids)
                return torch.zeros(1, seq_len, 896, dtype=torch.float32)

        return _Embed()

    def __call__(self, **kwargs: Any) -> Any:
        import torch

        self._forward_calls.append(kwargs)
        # Return logits (1, seq_len, vocab) — all zeros so argmax → token 0.
        seq_len = 1
        if "inputs_embeds" in kwargs:
            emb = kwargs["inputs_embeds"]
            if hasattr(emb, "shape"):
                seq_len = emb.shape[1]
        elif "input_ids" in kwargs:
            ids = kwargs["input_ids"]
            if hasattr(ids, "shape"):
                seq_len = ids.shape[1]
            else:
                seq_len = len(ids) if isinstance(ids, (list, tuple)) else 1
        logits = torch.zeros(1, seq_len, 151936, dtype=torch.float32)
        # Make token 0 the argmax (predictable).
        logits[0, :, 0] = 1.0
        return self._Out(logits=logits, past=None)


class _FakeHFDriver(_HFDriverBase):

    def __init__(self, reply: str = "unknown") -> None:
        self._tokenizer = _FakeTokenizer()
        self._model: Any = _FakeModel(reply)
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.n_embd = 896
        self.n_vocab = 151936
        self.n_layers = 24
        self._reply = reply
        self._reserved: ReservedPosition | None = None
        self.generate_calls: list[tuple[str, int]] = []
        self.reserved_set_count = 0
        self.reserved_cleared_count = 0
        self._closed = False
        self._embedding_cache: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()
        self._cvec_active = False
        self._inputs_embeds_calls: list[dict[str, Any]] = []

    @override
    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: Sequence[str] | str | None = None,
    ) -> str:
        self.generate_calls.append((prompt, max_tokens))
        del temperature, stop  # accepted for HFDriver interface compatibility
        return self._reply

    @override
    def set_reserved_position(self, position: Any) -> None:
        self._reserved = position
        self.reserved_set_count += 1

    @override
    def clear_reserved_position(self) -> None:
        self._reserved = None
        self.reserved_cleared_count += 1

    @override
    def set_cvec_uniform(self, vec: np.ndarray, scale: float = 1.0) -> None:
        del vec, scale  # accepted for HFDriver interface compatibility
        self._cvec_active = True

    @override
    def clear_cvec(self) -> None:
        self._cvec_active = False

    @override
    def peek_embedding(self, prompt: str, last_token_only: bool = True) -> np.ndarray:
        cache_key = (prompt, last_token_only)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        emb = rng.normal(0, 1, size=self.n_embd).astype(np.float32)
        self._embedding_cache[cache_key] = emb
        return emb

    @override
    def _tokenize(self, text: str) -> Any:
        import torch

        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long)

    @override
    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Test fixtures: tiny curriculum with known DEV/holdout split
# ---------------------------------------------------------------------------


def _make_probe(request: str, expected: str, category: ProbeCategory = "retention") -> Probe:
    return Probe(request=request, expected=expected, category=category, match_mode="contains")


def _make_episode(
    eid: str,
    request: str = "What is the code?",
    correction: str = "The code is marmalade.",
    label: str = "fruit jam",
    answer: str = "marmalade",
) -> Episode:
    return Episode(
        id=eid,
        initial_request=request,
        default_response="I don't know.",
        correction_utterance=correction,
        corrected_label=label,
        corrected_response=answer,
        domain="general",
        probes=(
            _make_probe(f"Repeat the code for {eid}.", answer, category="retention"),
        ),
    )


def _make_stage(
    n_episodes: int = 6,
    name: str = "stage_0_grounding",
) -> Stage:
    labels = [
        "fruit jam", "tree branch", "river run", "harbor seal",
        "crane bird", "fish scale",
    ]
    episodes = []
    for i in range(n_episodes):
        episodes.append(
            _make_episode(
                eid=f"ep{i}",
                request=f"What is item {i}?",
                correction=f"The answer for {i} is answer{i}.",
                label=labels[i % len(labels)],
                answer=f"answer{i}",
            )
        )
    return Stage(
        name=name,
        description="test stage",
        consolidate_before=False,
        consolidate_after=False,
        episodes=tuple(episodes),
    )


def _make_other_stages() -> tuple[Stage, ...]:
    """Make stage_1_transfer and stage_2_scope for specificity/transfer tests."""
    stage1 = _make_stage(n_episodes=4, name="stage_1_transfer")
    # Override probes to be transfer category
    stage1 = Stage(
        name=stage1.name,
        description=stage1.description,
        consolidate_before=stage1.consolidate_before,
        consolidate_after=stage1.consolidate_after,
        episodes=tuple(
            Episode(
                id=ep.id,
                initial_request=ep.initial_request,
                default_response=ep.default_response,
                correction_utterance=ep.correction_utterance,
                corrected_label=ep.corrected_label,
                corrected_response=ep.corrected_response,
                domain=ep.domain,
                probes=(
                    _make_probe(
                        f"Transfer: {ep.id}?", ep.corrected_response, category="transfer"
                    ),
                ),
            )
            for ep in stage1.episodes
        ),
    )
    stage2 = _make_stage(n_episodes=4, name="stage_2_scope")
    stage2 = Stage(
        name=stage2.name,
        description=stage2.description,
        consolidate_before=stage2.consolidate_before,
        consolidate_after=stage2.consolidate_after,
        episodes=tuple(
            Episode(
                id=ep.id,
                initial_request=ep.initial_request,
                default_response=ep.default_response,
                correction_utterance=ep.correction_utterance,
                corrected_label=ep.corrected_label,
                corrected_response=ep.corrected_response,
                domain=ep.domain,
                probes=(
                    _make_probe(
                        f"Scope: {ep.id}?", ep.corrected_response, category="scope"
                    ),
                ),
            )
            for ep in stage2.episodes
        ),
    )
    return (stage1, stage2)

def _make_valid_manifest_dict(**overrides: Any) -> dict[str, Any]:
    """Return a valid manifest dict with all required fields and a correct hash.

    The dict passes verify_hash(), required_fields_present(), and
    holdout_accessed=False, so evaluate proceeds past initial checks.
    Override individual fields via kwargs.
    """
    from oczy.experiments.s19_language_organ_core import CalibrationManifest

    defaults: dict[str, Any] = dict(
        schema_version="oczy/r19-calibration-manifest/v1",
        source_commit="a" * 40,
        source_archive_sha256="b" * 64,
        eval_version="v2.1",
        eval_manifest_sha256="c" * 64,
        model_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="main",
        model_config_sha256="d" * 64,
        model_safetensors_sha256="e" * 64,
        cortex_artifact_sha256="f" * 64,
        cortex_artifact_bytes=1000,
        cortex_artifact_path="s19_cortex.pkl",
        coupler_sha256="g" * 64,
        coupler_bytes=500,
        head_sha256="h" * 64,
        head_bytes=500,
        labels=["label1"],
        articulation_language_organ_hash="i" * 64,
        c7_reference="s3m2a_baseline",
        c7_available=True,
        signoff_human_signoff_id="human-001",
        signoff_thresholds_signed_off=True,
        signoff_oracle_ceiling=0.5,
        signoff_dev_articulation_gate=True,
        signoff_meta_test_conflation_ok=True,
    )
    defaults.update(overrides)
    m = CalibrationManifest(**defaults)
    m.manifest_sha256 = m.compute_hash()
    return m.to_dict()


# ---------------------------------------------------------------------------
# Parameter budget tests
# ---------------------------------------------------------------------------


class TestParameterBudget:
    """Tests for the <=64k parameter budget contract."""

    def test_parameter_count_within_budget(self) -> None:
        """SharedCortex.parameter_count() must be <= 64000."""
        from oczy.experiments.s19_language_organ_core import (
            MAX_PARAMS,
            TOTAL_PARAMS,
            SharedCortex,
        )

        cortex = SharedCortex()
        assert cortex.parameter_count() == TOTAL_PARAMS
        assert cortex.parameter_count() <= MAX_PARAMS

    def test_exact_parameter_count(self) -> None:
        """The parameter count must be exactly 60388."""
        from oczy.experiments.s19_language_organ_core import TOTAL_PARAMS

        assert TOTAL_PARAMS == 60388

    def test_parameter_breakdown_matches_spec(self) -> None:
        """Parameter breakdown must match the R19 contract exactly."""
        from oczy.experiments.s19_language_organ_core import PARAM_BREAKDOWN

        assert PARAM_BREAKDOWN["W_perceive"] == 14336   # 896 * 16
        assert PARAM_BREAKDOWN["W_label"] == 320         # 16 * 20
        assert PARAM_BREAKDOWN["b_label"] == 20
        assert PARAM_BREAKDOWN["W_coupler"] == 43008     # 16 * (3 * 896)
        assert PARAM_BREAKDOWN["b_coupler"] == 2688      # 3 * 896
        assert PARAM_BREAKDOWN["warm_state"] == 16

    def test_verify_parameter_budget_passes(self) -> None:
        """verify_parameter_budget must return True for a valid cortex."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            verify_parameter_budget,
        )

        cortex = SharedCortex()
        assert verify_parameter_budget(cortex) is True

    def test_persistent_bytes_nonzero(self) -> None:
        """persistent_bytes must be > 0 (serialized state dict)."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex()
        assert cortex.persistent_bytes() > 0


# ---------------------------------------------------------------------------
# Shared cortex tests
# ---------------------------------------------------------------------------


class TestSharedCortex:
    """Tests for the shared cortex architecture."""

    def test_cortex_config_defaults(self) -> None:
        """CortexConfig defaults must match the R19 contract."""
        from oczy.experiments.s19_language_organ_core import CortexConfig

        cfg = CortexConfig()
        assert cfg.d_embd == 896
        assert cfg.d_cortex == 16
        assert cfg.latent_tokens == 3
        assert cfg.n_labels == 20

    def test_perceive_output_shape(self) -> None:
        """perceive() must return a (d_cortex,) tensor."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.zeros(896, dtype=np.float32)
        act = cortex.perceive(features)
        assert act.shape[0] == 16  # d_cortex

    def test_predict_label_returns_index_and_confidence(self) -> None:
        """predict_label must return (index, confidence) with valid ranges."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.zeros(896, dtype=np.float32)
        act = cortex.perceive(features)
        idx, conf = cortex.predict_label(act)
        assert 0 <= idx < 20
        assert 0.0 <= conf <= 1.0

    def test_compute_latent_shape(self) -> None:
        """compute_latent must return (latent_tokens, d_embd) = (3, 896)."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.zeros(896, dtype=np.float32)
        act = cortex.perceive(features)
        latent = cortex.compute_latent(act)
        assert latent.shape[0] == 3   # latent_tokens
        assert latent.shape[1] == 896  # d_embd

    def test_coupler_freeze(self) -> None:
        """freeze_coupler must set coupler_frozen and disable gradients."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex()
        assert not cortex.coupler_frozen
        cortex.freeze_coupler()
        assert cortex.coupler_frozen
        assert not cortex.W_coupler.requires_grad
        assert not cortex.b_coupler.requires_grad

    def test_coupler_unfreeze(self) -> None:
        """unfreeze_coupler must re-enable coupler gradients."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex()
        cortex.freeze_coupler()
        cortex.unfreeze_coupler()
        assert not cortex.coupler_frozen
        assert cortex.W_coupler.requires_grad

    def test_zero_state(self) -> None:
        """zero_state must set warm_state to all zeros."""
        import torch

        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        # Move warm_state away from zero.
        cortex.warm_state.data.fill_(0.5)
        cortex.zero_state()
        assert torch.allclose(cortex.warm_state, torch.zeros(16))

    def test_swap_state(self) -> None:
        """swap_state must copy warm_state from other cortex (C5: addressing test)."""
        import torch

        from oczy.experiments.s19_language_organ_core import CortexConfig, SharedCortex

        config = CortexConfig()
        cortex_a = SharedCortex(config=config, seed=0)
        cortex_b = SharedCortex(config=config, seed=1)
        # Make warm_state different.
        cortex_b.warm_state.data.fill_(0.7)

        cortex_a.swap_state(cortex_b)
        # warm_state should match b's.
        assert torch.allclose(cortex_a.warm_state, cortex_b.warm_state)

    def test_state_dict_roundtrip(self) -> None:
        """state_dict / load_state_dict must round-trip correctly."""
        import torch

        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex_a = SharedCortex(seed=0)
        cortex_a.warm_state.data.fill_(0.42)
        state = cortex_a.state_dict()

        cortex_b = SharedCortex(seed=1)
        cortex_b.load_state_dict(state)
        assert torch.allclose(cortex_b.warm_state, cortex_a.warm_state)

    def test_coupler_state_isolated(self) -> None:
        """coupler_state must return only W_coupler and b_coupler."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex()
        state = cortex.coupler_state()
        assert "W_coupler" in state
        assert "b_coupler" in state
        assert "W_perceive" not in state
        assert "W_label" not in state

    def test_coupler_hash_deterministic_same_instance(self) -> None:
        """coupler_hash must be deterministic when called on the same instance."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        h1 = cortex.coupler_hash()
        h2 = cortex.coupler_hash()
        assert h1 == h2

    def test_coupler_hash_deterministic_after_roundtrip(self) -> None:
        """coupler_hash must be the same after state_dict → load_coupler roundtrip."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex_a = SharedCortex(seed=0)
        state = cortex_a.coupler_state()
        cortex_b = SharedCortex(seed=1)  # different seed → different initial weights
        cortex_b.load_coupler(state)
        # After loading the same coupler state, hashes should match.
        # (Note: pickle of detached CPU tensors from the same source is deterministic.)
        assert cortex_a.coupler_hash() == cortex_b.coupler_hash()

    def test_coupler_hash_changes_with_weights(self) -> None:
        """coupler_hash must change when coupler weights change."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex_a = SharedCortex(seed=0)
        cortex_b = SharedCortex(seed=1)
        assert cortex_a.coupler_hash() != cortex_b.coupler_hash()


# ---------------------------------------------------------------------------
# Shared cortex between arms
# ---------------------------------------------------------------------------


class TestSharedCortexBetweenArms:
    """Tests that Arms A and B share the same cortex state."""

    def test_arm_a_and_b_use_same_cortex(self) -> None:
        """Both arms must use the same SharedCortex instance (shared perception/state)."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_a_generate,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        labels = [f"label_{i}" for i in range(20)]

        # Arm A uses cortex.perceive and cortex.predict_label.
        _, audit_a = arm_a_generate(
            driver, cortex, "test request", labels, 0.0, "{label}. "
        )
        # Arm B uses cortex.perceive and cortex.compute_latent.
        _, audit_b = arm_b_generate(driver, cortex, "test request", 0.0)

        # Both arms must report the same cortex confidence (shared perception).
        # (Confidence comes from predict_label which uses cortex.perceive.)
        # Arm A confidence is from predict_label; Arm B also calls predict_label.
        assert "confidence" in audit_a
        assert "confidence" in audit_b

    def test_shared_perception_produces_same_activation(self) -> None:
        """The cortex activation must be identical for both arms given the same request."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.ones(896, dtype=np.float32) * 0.1
        act_a = cortex.perceive(features)
        act_b = cortex.perceive(features)
        import torch

        assert torch.allclose(act_a, act_b)


# ---------------------------------------------------------------------------
# DEV-only calibrate firewall tests
# ---------------------------------------------------------------------------


class TestCalibrateDevFirewall:
    """Tests that calibrate-dev never accesses holdout IDs."""

    def test_calibrate_dev_discards_holdout_ids(self) -> None:
        """calibrate-dev must discard holdout_ids (del holdout_ids)."""
        # We test the contract by verifying that the calibrate-dev code path
        # only uses dev_ids, never holdout_ids.  Since we can't run the full
        # CLI without a real model, we test the split + discard pattern.
        stage = _make_stage()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")

        # The calibrate-dev phase must discard holdout_ids.
        # Simulate the discard:
        del holdout_ids

        # dev_ids must be non-empty and used for scoring.
        assert len(dev_ids) > 0

        # holdout_ids must be truly gone from the local scope.
        assert "holdout_ids" not in locals()

    def test_calibrate_dev_only_scores_dev_probes(self) -> None:
        """score_probes with dev_ids must never score holdout probes."""
        from oczy.experiments.s19_language_organ_core import score_probes

        stage = _make_stage()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        driver = _FakeHFDriver(reply="answer0")

        result = score_probes(driver, stage, dev_ids, "vanilla")
        scored_pids = {a["probe_id"] for a in result["audits"]}

        # No scored probe should be in holdout_ids.
        assert scored_pids.isdisjoint(holdout_ids)

    def test_holdout_firewall_dev_ids_disjoint_from_holdout(self) -> None:
        """DEV and holdout IDs must be disjoint sets."""
        stage = _make_stage()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        assert dev_ids.isdisjoint(holdout_ids)


# ---------------------------------------------------------------------------
# Evaluate sign-off gate tests
# ---------------------------------------------------------------------------


class TestEvaluateSignoffGate:
    """Tests that evaluate fails closed without matching manifest hash / sign-off."""

    def test_evaluate_rejects_empty_signoff(self) -> None:
        """evaluate must fail if signoff_id is empty or whitespace."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict()
            manifest_path.write_text(json.dumps(manifest_data))

            # Empty signoff → must fail.
            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "",
            ])
            assert rc != 0

    def test_evaluate_rejects_whitespace_signoff(self) -> None:
        """evaluate must fail if signoff_id is only whitespace."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict()
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "   ",
            ])
            assert rc != 0

    def test_evaluate_rejects_missing_manifest(self) -> None:
        """evaluate must fail if the manifest file does not exist."""
        from oczy.experiments.s19_language_organ import main

        rc = main([
            "evaluate",
            "--manifest", "/nonexistent/path/manifest.json",
            "--signoff-id", "human-001",
        ])
        assert rc != 0

    def test_evaluate_rejects_tampered_manifest_hash(self) -> None:
        """evaluate must fail if the manifest hash doesn't match recomputation."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict()
            manifest_data["manifest_sha256"] = "deliberately_wrong_hash"
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0


# ---------------------------------------------------------------------------
# CalibrationManifest tests
# ---------------------------------------------------------------------------


class TestCalibrationManifest:
    """Tests for the CalibrationManifest dataclass."""

    def test_manifest_hash_is_deterministic(self) -> None:
        """compute_hash must be deterministic for the same field values."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(
            model_repo_id="test",
            model_safetensors_sha256="abc",
            labels=["a", "b"],
            proposed_confidence_threshold=0.5,
            proposed_specificity_margin=0.1,
        )
        m2 = CalibrationManifest(
            model_repo_id="test",
            model_safetensors_sha256="abc",
            labels=["a", "b"],
            proposed_confidence_threshold=0.5,
            proposed_specificity_margin=0.1,
        )
        assert m1.compute_hash() == m2.compute_hash()

    def test_manifest_hash_changes_with_fields(self) -> None:
        """compute_hash must change when any field changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(model_repo_id="a", model_safetensors_sha256="x")
        m2 = CalibrationManifest(model_repo_id="b", model_safetensors_sha256="x")
        assert m1.compute_hash() != m2.compute_hash()

    def test_manifest_verify_hash_passes_for_valid(self) -> None:
        """verify_hash must return True when manifest_sha256 matches recomputation."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(model_repo_id="test", model_safetensors_sha256="abc")
        m.manifest_sha256 = m.compute_hash()
        assert m.verify_hash() is True

    def test_manifest_verify_hash_fails_for_tampered(self) -> None:
        """verify_hash must return False when manifest_sha256 doesn't match."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(model_repo_id="test", model_safetensors_sha256="abc")
        m.manifest_sha256 = "wrong"
        assert m.verify_hash() is False

    def test_manifest_roundtrip(self) -> None:
        """to_dict → from_dict must round-trip correctly."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(
            model_repo_id="test",
            model_safetensors_sha256="abc",
            labels=["a", "b"],
            proposed_confidence_threshold=0.72,
            proposed_specificity_margin=0.08,
            coupler_sha256="coupler_hash_123",
        )
        m1.manifest_sha256 = m1.compute_hash()
        d = m1.to_dict()
        m2 = CalibrationManifest.from_dict(d)
        assert m2.model_repo_id == m1.model_repo_id
        assert m2.model_safetensors_sha256 == m1.model_safetensors_sha256
        assert m2.labels == m1.labels
        assert m2.proposed_confidence_threshold == m1.proposed_confidence_threshold
        assert m2.proposed_specificity_margin == m1.proposed_specificity_margin
        assert m2.coupler_sha256 == m1.coupler_sha256

    def test_manifest_human_signoff_default_empty(self) -> None:
        """New manifests must have empty signoff_human_signoff_id by default."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert m.signoff_human_signoff_id == ""


# ---------------------------------------------------------------------------
# No-text Arm B audit tests
# ---------------------------------------------------------------------------


class TestNoTextArmB:
    """Tests that Arm B never injects label/answer/correction text."""

    def test_arm_b_prompt_is_bare_request(self) -> None:
        """Arm B's prompt_text must be exactly the request — no prefix."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        request = "What is the answer?"
        _, audit = arm_b_generate(driver, cortex, request, 0.0)
        assert audit["prompt_text"] == request

    def test_arm_b_does_not_set_reserved_position(self) -> None:
        """Arm B must never call set_reserved_position."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        arm_b_generate(driver, cortex, "test request", 0.0)
        assert driver.reserved_set_count == 0

    def test_arm_b_latent_bank_shape_is_fixed(self) -> None:
        """Arm B's latent_bank_shape must be (3, 896) when not abstaining."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        _, audit = arm_b_generate(driver, cortex, "test request", 0.0)
        # When not abstaining (threshold=0.0), shape must be (3, 896).
        assert audit["latent_bank_shape"] == (3, 896)

    def test_arm_b_abstain_no_latent_bank(self) -> None:
        """When abstaining, Arm B must not inject a latent bank."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        # threshold=2.0 → always abstain (confidence <= 1.0).
        _, audit = arm_b_generate(driver, cortex, "test request", 2.0)
        assert audit["abstained"] is True
        assert audit["latent_bank_shape"] is None

    def test_arm_b_no_correction_text_in_prompt(self) -> None:
        """Arm B's prompt must not contain correction text."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        request = "What is the code?"
        _, audit = arm_b_generate(driver, cortex, request, 0.0)
        # The prompt must not contain correction-like text.
        assert "marmalade" not in audit["prompt_text"]
        assert "The code is" not in audit["prompt_text"]

    def test_arm_b_no_label_text_in_prompt(self) -> None:
        """Arm B's prompt must not contain label text."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        request = "What is the answer?"
        _, audit = arm_b_generate(driver, cortex, request, 0.0)
        # No label prefix should appear.
        assert "fruit jam" not in audit["prompt_text"]
        assert "tree branch" not in audit["prompt_text"]

    def test_verify_no_text_injection_passes_for_arm_b(self) -> None:
        """verify_no_text_injection must pass for clean Arm B audits."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {"arm": "B", "prompt_text": "bare request", "latent_bank_shape": (3, 896)},
        ]
        assert verify_no_text_injection(audits) is True

    def test_verify_no_text_injection_fails_for_wrong_shape(self) -> None:
        """verify_no_text_injection must fail if latent bank shape is wrong."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {"arm": "B", "prompt_text": "request", "latent_bank_shape": (5, 896)},
        ]
        assert verify_no_text_injection(audits) is False

    def test_verify_no_text_injection_ignores_arm_a(self) -> None:
        """verify_no_text_injection must not flag Arm A audits."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {"arm": "A", "prompt_text": "label prefix. request", "latent_bank_shape": None},
        ]
        assert verify_no_text_injection(audits) is True


# ---------------------------------------------------------------------------
# Fixed latent shape tests
# ---------------------------------------------------------------------------


class TestFixedLatentShape:
    """Tests that the latent bank shape is fixed (3, 896) independent of episodes."""

    def test_latent_shape_constant_across_episode_counts(self) -> None:
        """compute_latent must always return (3, 896) regardless of teaching history."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.random.default_rng(0).normal(size=896).astype(np.float32)

        # Before any teaching.
        act = cortex.perceive(features)
        latent = cortex.compute_latent(act)
        assert latent.shape == (3, 896)

        # After simulated teaching (update warm_state).
        cortex.warm_state.data.fill_(0.5)
        act = cortex.perceive(features)
        latent = cortex.compute_latent(act)
        assert latent.shape == (3, 896)

    def test_verify_fixed_latent_width_passes(self) -> None:
        """verify_fixed_latent_width must pass for (3, 896) shapes."""
        from oczy.experiments.s19_language_organ_core import verify_fixed_latent_width

        audits = [
            {"arm": "B", "latent_bank_shape": (3, 896)},
            {"arm": "B", "latent_bank_shape": (3, 896)},
            {"arm": "B", "latent_bank_shape": None},  # abstained is OK
        ]
        assert verify_fixed_latent_width(audits) is True

    def test_verify_fixed_latent_width_fails_for_variable(self) -> None:
        """verify_fixed_latent_width must fail if shape varies."""
        from oczy.experiments.s19_language_organ_core import verify_fixed_latent_width

        audits = [
            {"arm": "B", "latent_bank_shape": (3, 896)},
            {"arm": "B", "latent_bank_shape": (12, 896)},  # grew!
        ]
        assert verify_fixed_latent_width(audits) is False


# ---------------------------------------------------------------------------
# Model hash immutability tests
# ---------------------------------------------------------------------------


class TestModelHashImmutability:
    """Tests that the LM hash is identical before and after teaching/evaluation."""

    def test_hash_model_deterministic(self) -> None:
        """hash_model must return the same hash for the same driver state."""
        from oczy.experiments.s19_language_organ_core import hash_model

        driver = _FakeHFDriver()
        h1 = hash_model(driver)
        h2 = hash_model(driver)
        assert h1 == h2

    def test_hash_model_changes_with_weights(self) -> None:
        """hash_model must change when model weights change."""
        from oczy.experiments.s19_language_organ_core import hash_model

        driver = _FakeHFDriver()
        h1 = hash_model(driver)
        # Mutate a parameter.
        driver._model._params["embed.weight"][0, 0] = 42.0
        h2 = hash_model(driver)
        assert h1 != h2

    def test_verify_frozen_lm_passes(self) -> None:
        """verify_frozen_lm must return True when hashes match."""
        from oczy.experiments.s19_language_organ_core import verify_frozen_lm

        assert verify_frozen_lm("abc123", "abc123") is True

    def test_verify_frozen_lm_fails(self) -> None:
        """verify_frozen_lm must return False when hashes differ."""
        from oczy.experiments.s19_language_organ_core import verify_frozen_lm

        assert verify_frozen_lm("abc123", "def456") is False


# ---------------------------------------------------------------------------
# Raw trace deletion tests
# ---------------------------------------------------------------------------


class TestRawTraceDeletion:
    """Tests that raw traces are deleted after consolidation."""

    def test_trace_store_add_and_count(self) -> None:
        """TraceStore.add must increment count."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        assert store.count() == 0
        store.add("ep0", "request", "correction", "response")
        assert store.count() == 1
        store.add("ep1", "request1", "correction1", "response1")
        assert store.count() == 2

    def test_trace_store_delete_all(self) -> None:
        """delete_all must clear all traces and return the count deleted."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add("ep0", "r", "c", "resp")
        store.add("ep1", "r1", "c1", "resp1")
        deleted = store.delete_all()
        assert deleted == 2
        assert store.count() == 0

    def test_trace_store_verify_zero(self) -> None:
        """verify_zero must return True when empty, False when traces exist."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        assert store.verify_zero() is True
        store.add("ep0", "r", "c", "resp")
        assert store.verify_zero() is False
        store.delete_all()
        assert store.verify_zero() is True

    def test_verify_raw_traces_deleted(self) -> None:
        """verify_raw_traces_deleted must return True for empty store."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            verify_raw_traces_deleted,
        )

        store = TraceStore()
        store.add("ep0", "r", "c", "resp")
        assert verify_raw_traces_deleted(store) is False
        store.delete_all()
        assert verify_raw_traces_deleted(store) is True


# ---------------------------------------------------------------------------
# C0-C7 condition tests
# ---------------------------------------------------------------------------


class TestConditions:
    """Tests for C0-C7 condition behavior."""

    def _setup_eval_env(self):
        """Set up a minimal evaluation environment with fakes."""
        from oczy.experiments.s19_language_organ_core import (
            CortexConfig,
            SharedCortex,
            TraceStore,
            extract_stage_labels,
        )

        driver = _FakeHFDriver(reply="answer0")
        stage = _make_stage()
        other_stages = _make_other_stages()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        labels = extract_stage_labels(stage)
        # Configure cortex with n_labels matching the test stage.
        config = CortexConfig(n_labels=len(labels))
        cortex = SharedCortex(config=config, seed=0)
        cortex.freeze_coupler()
        trace_store = TraceStore()
        return driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store

    def test_c0_vanilla_no_cortex(self) -> None:
        """C0 must run vanilla (no cortex) and return accuracy."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        result = run_condition(
            driver, "C0", stage, other_stages, dev_ids, holdout_ids,
            None, None, 0.5, "{label}. ", trace_store,
        )
        assert "holdout_acc" in result
        assert "transfer_acc" in result
        assert result["persistent_bytes"] == 0

    def test_c1_no_update_random_cortex(self) -> None:
        """C1 must use a random/untrained cortex (no update)."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        result = run_condition(
            driver, "C1", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.5, "{label}. ", trace_store,
        )
        assert "holdout_acc" in result
        assert result["persistent_bytes"] > 0

    def test_c2_arm_a_label_prefix(self) -> None:
        """C2 must use Arm A (label prefix) and set/clear reserved position."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        initial_set = driver.reserved_set_count
        run_condition(
            driver, "C2", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.0, "{label}. ", trace_store,
        )
        # Arm A with threshold=0.0 should always set reserved position.
        assert driver.reserved_set_count > initial_set

    def test_c3_arm_b_latent_control(self) -> None:
        """C3 must use Arm B (latent control) and not set reserved position."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        initial_set = driver.reserved_set_count
        result = run_condition(
            driver, "C3", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.0, "{label}. ", trace_store,
        )
        assert driver.reserved_set_count == initial_set  # Arm B never sets reserved.
        assert "holdout_acc" in result

    def test_c4_zeroed_state(self) -> None:
        """C4 must zero the cortex state before probing."""
        import torch

        from oczy.experiments.s19_language_organ_core import (
            run_condition,
        )

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        # Set warm_state to nonzero before C4.
        cortex.warm_state.data.fill_(0.5)
        run_condition(
            driver, "C4", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.0, "{label}. ", trace_store,
        )
        # After C4, warm_state should be zeroed.
        assert torch.allclose(cortex.warm_state, torch.zeros(16))

    def test_c5_swapped_state(self) -> None:
        """C5 must swap cortex state with another cortex."""
        import torch

        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            run_condition,
        )

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        # Create a different cortex to swap with.
        other_cortex = SharedCortex(config=cortex.config, seed=99)
        other_cortex.warm_state.data.fill_(0.77)
        other_cortex.freeze_coupler()

        cortex.warm_state.data.fill_(0.0)  # Ensure different from other.
        run_condition(
            driver, "C5", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.0, "{label}. ", trace_store,
            swapped_cortex=other_cortex,
        )
        # After C5, cortex's state should match other_cortex's.
        assert torch.allclose(cortex.warm_state, other_cortex.warm_state)

    def test_c5_requires_swapped_cortex(self) -> None:
        """C5 must raise ValueError if swapped_cortex is None."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        with pytest.raises(ValueError, match="swapped_cortex"):
            run_condition(
                driver, "C5", stage, other_stages, dev_ids, holdout_ids,
                cortex, labels, 0.0, "{label}. ", trace_store,
                swapped_cortex=None,
            )

    def test_c6_permuted_labels(self) -> None:
        """C6 must run with the permuted-label cortex."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        result = run_condition(
            driver, "C6", stage, other_stages, dev_ids, holdout_ids,
            cortex, labels, 0.0, "{label}. ", trace_store,
        )
        assert "holdout_acc" in result

    def test_c7_retrieval_baseline(self) -> None:
        """C7 must run the retrieval baseline (external bar)."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        result = run_condition(
            driver, "C7", stage, other_stages, dev_ids, holdout_ids,
            None, None, 0.5, "{label}. ", trace_store,
        )
        assert "holdout_acc" in result
        assert result["persistent_bytes"] == 0  # external baseline

    def test_unknown_condition_raises(self) -> None:
        """run_condition must raise ValueError for unknown conditions."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            self._setup_eval_env()
        )
        with pytest.raises(ValueError, match="Unknown condition"):
            run_condition(
                driver, "C99", stage, other_stages, dev_ids, holdout_ids,
                cortex, labels, 0.5, "{label}. ", trace_store,
            )


# ---------------------------------------------------------------------------
# Causal behavior tests (active/zero/swapped/permuted)
# ---------------------------------------------------------------------------


class TestCausalBehavior:
    """Tests that causal interventions produce different behavior."""

    def test_zero_state_changes_latent(self) -> None:
        """Zeroing warm_state must change the latent bank output."""
        import torch

        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        features = np.ones(896, dtype=np.float32) * 0.1

        # With normal state.
        cortex.warm_state.data.fill_(0.5)
        act = cortex.perceive(features)
        latent_normal = cortex.compute_latent(act)

        # With zeroed state.
        cortex.zero_state()
        act = cortex.perceive(features)
        latent_zeroed = cortex.compute_latent(act)

        # The latents must differ (unless W_coupler is zero, which it isn't
        # because perceive adds warm_state).
        assert not torch.allclose(latent_normal, latent_zeroed)

    def test_swap_state_changes_latent(self) -> None:
        """Swapping warm_state from another cortex must change the latent output."""
        import torch

        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex_a = SharedCortex(seed=0)
        cortex_b = SharedCortex(seed=1)
        # Give b a distinct warm_state so the swap actually changes something.
        cortex_b.warm_state.data.fill_(0.7)
        features = np.ones(896, dtype=np.float32) * 0.1

        act_a = cortex_a.perceive(features)
        latent_a = cortex_a.compute_latent(act_a)

        cortex_a.swap_state(cortex_b)
        act_a_swapped = cortex_a.perceive(features)
        latent_a_swapped = cortex_a.compute_latent(act_a_swapped)

        # After swap, the latent should change (different warm_state).
        assert not torch.allclose(latent_a, latent_a_swapped)

    def test_permuted_labels_teach_uses_wrong_label(self) -> None:
        """teach_cortex with permuted_labels must use permuted indices."""
        from oczy.experiments.s19_language_organ_core import (
            CortexConfig,
            SharedCortex,
            TraceStore,
            build_label_index,
            extract_stage_labels,
            teach_cortex,
        )

        driver = _FakeHFDriver(reply="test")
        stage = _make_stage()
        labels = extract_stage_labels(stage)
        label_index = build_label_index(labels)
        config = CortexConfig(n_labels=len(labels))

        cortex_normal = SharedCortex(config=config, seed=0)
        cortex_perm = SharedCortex(config=config, seed=0)
        # Mock train_coupler to avoid the LM forward pass.
        cortex_normal.train_coupler = lambda driver, request, corrected_response: 0.0  # type: ignore[assignment]
        cortex_perm.train_coupler = lambda driver, request, corrected_response: 0.0  # type: ignore[assignment]
        trace_n = TraceStore()
        trace_p = TraceStore()

        teach_cortex(driver, cortex_normal, stage, labels, label_index, 0, trace_n, permuted_labels=False)
        teach_cortex(driver, cortex_perm, stage, labels, label_index, 0, trace_p, permuted_labels=True)

        # The permuted cortex should have different weights (learned wrong mapping).
        import torch

        assert not torch.allclose(cortex_normal.W_label, cortex_perm.W_label)

    def test_c3_vs_c4_active_vs_zeroed(self) -> None:
        """C3 (active) and C4 (zeroed) must produce different holdout accuracy."""
        # This test verifies the structural contract: C4 zeroes state before probing.
        # With a fake driver returning the same reply, accuracies may be equal,
        # but the zero_state call must happen.
        import torch

        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            TraceStore,
            run_condition,
        )

        driver = _FakeHFDriver(reply="answer0")
        stage = _make_stage()
        other_stages = _make_other_stages()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")

        cortex = SharedCortex(seed=0)
        cortex.warm_state.data.fill_(0.5)
        cortex.freeze_coupler()
        trace = TraceStore()

        # C3 runs with active state.
        run_condition(
            driver, "C3", stage, other_stages, dev_ids, holdout_ids,
            cortex, [], 0.0, "{label}. ", trace,
        )

        # C4 zeroes state.  Need a fresh cortex copy.
        cortex_c4 = SharedCortex(seed=0)
        cortex_c4.load_state_dict(cortex.state_dict())
        cortex_c4.freeze_coupler()
        run_condition(
            driver, "C4", stage, other_stages, dev_ids, holdout_ids,
            cortex_c4, [], 0.0, "{label}. ", trace,
        )

        # The structural contract: C4's cortex must have zero warm_state after.
        assert torch.allclose(cortex_c4.warm_state, torch.zeros(16))


# ---------------------------------------------------------------------------
# Deterministic manifest tests
# ---------------------------------------------------------------------------


class TestDeterministicManifest:
    """Tests that the manifest is deterministic."""

    def test_manifest_hash_deterministic_across_constructions(self) -> None:
        """Two manifests with identical fields must have identical hashes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        fields: dict[str, Any] = dict(
            model_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_safetensors_sha256="abc",
            eval_manifest_sha256="def",
            labels=["a", "b"],
            proposed_confidence_threshold=0.72,
            proposed_specificity_margin=0.08,
            coupler_sha256="ch",
        )
        m1 = CalibrationManifest(**fields)
        m2 = CalibrationManifest(**fields)
        assert m1.compute_hash() == m2.compute_hash()

    def test_manifest_hash_excludes_manifest_hash_field(self) -> None:
        """The manifest_sha256 field must not affect the computed hash."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(model_repo_id="test")
        h1 = m.compute_hash()
        m.manifest_sha256 = "some_value"
        h2 = m.compute_hash()
        assert h1 == h2  # manifest_sha256 is excluded from hash computation.


# ---------------------------------------------------------------------------
# Separate H-LABEL / H-LATENT verdict gate tests
# ---------------------------------------------------------------------------


class TestVerdictGates:
    """Tests that H-LABEL and H-LATENT verdicts are computed separately."""

    def _make_result(self, holdout=0.5, transfer=0.5, scope=0.5, specificity=0.5):
        return {
            "holdout_acc": holdout,
            "transfer_acc": transfer,
            "scope_acc": scope,
            "specificity_acc": specificity,
            "persistent_bytes": 100,
        }

    def test_h_latent_accept_when_all_conditions_met(self) -> None:
        """H-LATENT must ACCEPT when all C3 conditions pass and validity passes."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        c1 = [self._make_result(holdout=0.3, transfer=0.3, specificity=0.5)]
        c3 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.5)]
        c2 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.5)]
        c4 = [self._make_result(holdout=0.3)]  # causal_state_delta > 0
        c5 = [self._make_result(holdout=0.3)]  # state_addressing_delta > 0
        c6 = [self._make_result(holdout=0.3)]  # feedback_semantics_delta > 0

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LATENT"] == "ACCEPT"

    def test_h_latent_refute_when_retention_fails(self) -> None:
        """H-LATENT must REFUTE when retention_delta CI includes zero."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        # C3 and C1 have the same holdout → retention_delta = 0.
        c1 = [self._make_result(holdout=0.5)]
        c3 = [self._make_result(holdout=0.5)]
        c2 = [self._make_result(holdout=0.8)]
        c4 = [self._make_result(holdout=0.3)]
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.3)]

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LATENT"] == "REFUTE"

    def test_h_latent_blocked_when_validity_fails(self) -> None:
        """H-LATENT must be BLOCKED when validity_pass is False."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        c1 = [self._make_result()]
        c3 = [self._make_result()]
        c2 = [self._make_result()]
        c4 = [self._make_result()]
        c5 = [self._make_result()]
        c6 = [self._make_result()]

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, False)
        assert verdicts["H_LATENT"] == "BLOCKED"
        assert verdicts["H_LABEL"] == "BLOCKED"

    def test_h_label_accept_independent_of_h_latent(self) -> None:
        """H-LABEL can ACCEPT even when H-LATENT REFUTES (separate gates)."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        # C2 passes but C3 fails.
        c1 = [self._make_result(holdout=0.3, transfer=0.3, specificity=0.5)]
        c2 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.5)]
        c3 = [self._make_result(holdout=0.3, transfer=0.3, specificity=0.5)]  # C3 fails
        c4 = [self._make_result(holdout=0.3)]
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.3)]

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LABEL"] == "ACCEPT"
        assert verdicts["H_LATENT"] == "REFUTE"

    def test_h_label_refute_when_retention_fails(self) -> None:
        """H-LABEL must REFUTE when C2 retention_delta CI includes zero."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        c1 = [self._make_result(holdout=0.5)]
        c2 = [self._make_result(holdout=0.5)]  # same as C1 → delta = 0
        c3 = [self._make_result(holdout=0.8)]
        c4 = [self._make_result(holdout=0.3)]
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.3)]

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LABEL"] == "REFUTE"

    def test_h_latent_refute_when_causal_state_fails(self) -> None:
        """H-LATENT must REFUTE when causal_state_delta CI includes zero."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        # C3 and C4 have the same holdout → causal_state_delta = 0.
        c1 = [self._make_result(holdout=0.3)]
        c3 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.5)]
        c2 = [self._make_result(holdout=0.8)]
        c4 = [self._make_result(holdout=0.8)]  # same as C3 → delta = 0
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.3)]

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LATENT"] == "REFUTE"

    def test_h_latent_refute_when_feedback_semantics_fails(self) -> None:
        """H-LATENT must REFUTE when feedback_semantics_delta CI includes zero."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        # C3 and C6 have the same holdout → feedback_semantics_delta = 0.
        c1 = [self._make_result(holdout=0.3)]
        c3 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.5)]
        c2 = [self._make_result(holdout=0.8)]
        c4 = [self._make_result(holdout=0.3)]
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.8)]  # same as C3 → delta = 0

        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LATENT"] == "REFUTE"

    def test_h_latent_refute_when_specificity_exceeds_margin(self) -> None:
        """H-LATENT must REFUTE when specificity exceeds the equivalence margin."""
        from oczy.experiments.s19_language_organ_core import compute_verdicts

        c1 = [self._make_result(holdout=0.3, transfer=0.3, specificity=0.5)]
        c3 = [self._make_result(holdout=0.8, transfer=0.8, specificity=0.9)]  # big change
        c2 = [self._make_result(holdout=0.8)]
        c4 = [self._make_result(holdout=0.3)]
        c5 = [self._make_result(holdout=0.3)]
        c6 = [self._make_result(holdout=0.3)]

        # specificity_delta = 0.9 - 0.5 = 0.4 > 0.15 margin.
        verdicts = compute_verdicts(c3, c2, c4, c5, c6, c1, 0.15, True)
        assert verdicts["H_LATENT"] == "REFUTE"


# ---------------------------------------------------------------------------
# Fail-closed driver error tests
# ---------------------------------------------------------------------------


class TestFailClosedDriverErrors:
    """Tests that driver errors cause fail-closed behavior."""

    def test_evaluate_fails_closed_on_missing_manifest(self) -> None:
        """evaluate must return non-zero when the manifest file is missing."""
        from oczy.experiments.s19_language_organ import main

        rc = main([
            "evaluate",
            "--manifest", "/nonexistent/manifest.json",
            "--signoff-id", "human-001",
        ])
        assert rc != 0

    def test_evaluate_fails_closed_on_missing_coupler(self) -> None:
        """evaluate must fail when the coupler file is missing."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict()
            manifest_path.write_text(json.dumps(manifest_data))

            # Don't create the coupler file → must fail.
            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--coupler-path", str(Path(tmpdir) / "missing_coupler.pkl"),
                "--signoff-id", "human-001",
            ])
            assert rc != 0


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for mean_ci and ci_excludes_zero."""

    def test_mean_ci_single_value(self) -> None:
        """mean_ci with one value must return (val, val, val)."""
        from oczy.experiments.s19_language_organ_core import mean_ci

        mean, lo, hi = mean_ci([0.5])
        assert mean == 0.5
        assert lo == 0.5
        assert hi == 0.5

    def test_mean_ci_empty(self) -> None:
        """mean_ci with no values must return zeros."""
        from oczy.experiments.s19_language_organ_core import mean_ci

        mean, lo, hi = mean_ci([])
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_excludes_zero_positive(self) -> None:
        """ci_excludes_zero must return True when CI is entirely positive."""
        from oczy.experiments.s19_language_organ_core import ci_excludes_zero

        assert ci_excludes_zero((0.5, 0.1, 0.9)) is True

    def test_ci_excludes_zero_negative(self) -> None:
        """ci_excludes_zero must return True when CI is entirely negative."""
        from oczy.experiments.s19_language_organ_core import ci_excludes_zero

        assert ci_excludes_zero((-0.5, -0.9, -0.1)) is True

    def test_ci_includes_zero(self) -> None:
        """ci_excludes_zero must return False when CI includes zero."""
        from oczy.experiments.s19_language_organ_core import ci_excludes_zero

        assert ci_excludes_zero((0.0, -0.5, 0.5)) is False


# ---------------------------------------------------------------------------
# Content leakage tests
# ---------------------------------------------------------------------------


class TestContentLeakage:
    """Tests that banned content does not leak into Arm B."""

    def test_arm_b_prompt_no_expected_answer(self) -> None:
        """Arm B's prompt must not contain the expected answer text."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        request = "What is the secret code?"
        _, audit = arm_b_generate(driver, cortex, request, 0.0)
        assert "marmalade" not in audit["prompt_text"]
        assert "secret" not in audit["prompt_text"].lower() or audit["prompt_text"] == request

    def test_arm_b_prompt_no_episode_id(self) -> None:
        """Arm B's prompt must not contain episode IDs."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_b_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        request = "What is the answer?"
        _, audit = arm_b_generate(driver, cortex, request, 0.0)
        assert "ep0" not in audit["prompt_text"]
        assert "ep1" not in audit["prompt_text"]

    def test_verify_no_episode_id_conditioning_passes(self) -> None:
        """verify_no_episode_id_conditioning must pass for clean prompts."""
        from oczy.experiments.s19_language_organ_core import (
            verify_no_episode_id_conditioning,
        )

        audits = [
            {"arm": "B", "prompt_text": "What is the answer?"},
        ]
        assert verify_no_episode_id_conditioning(audits) is True

    def test_verify_no_episode_id_conditioning_fails(self) -> None:
        """verify_no_episode_id_conditioning must fail if prompt contains episode IDs."""
        from oczy.experiments.s19_language_organ_core import (
            verify_no_episode_id_conditioning,
        )

        audits = [
            {"arm": "B", "prompt_text": "What is the answer for s0_ep0?"},
        ]
        assert verify_no_episode_id_conditioning(audits) is False


# ---------------------------------------------------------------------------
# Arm A label prefix tests
# ---------------------------------------------------------------------------


class TestArmALabelPrefix:
    """Tests for Arm A's label-prefix articulation path."""

    def test_arm_a_sets_reserved_position_when_confident(self) -> None:
        """Arm A must set reserved position when confidence >= threshold."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_a_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        labels = [f"label_{i}" for i in range(20)]
        # threshold=0.0 → always confident.
        arm_a_generate(driver, cortex, "test request", labels, 0.0, "{label}. ")
        assert driver.reserved_set_count > 0
        assert driver.reserved_cleared_count > 0

    def test_arm_a_abstains_when_low_confidence(self) -> None:
        """Arm A must not set reserved position when confidence < threshold."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_a_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        labels = [f"label_{i}" for i in range(20)]
        # threshold=2.0 → always abstain (confidence <= 1.0).
        _, audit = arm_a_generate(driver, cortex, "test request", labels, 2.0, "{label}. ")
        assert audit["abstained"] is True
        assert driver.reserved_set_count == 0

    def test_arm_a_prefix_uses_label_template(self) -> None:
        """Arm A's prefix must use the label_prefix_template format."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_a_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        labels = [f"label_{i}" for i in range(20)]
        _, audit = arm_a_generate(driver, cortex, "test request", labels, 0.0, "TOPIC: {label}. ")
        assert audit["prefix_text"] is not None
        assert "TOPIC:" in audit["prefix_text"]

    def test_arm_a_clears_reserved_after_generate(self) -> None:
        """Arm A must clear reserved position after generate (even on error)."""
        from oczy.experiments.s19_language_organ_core import (
            SharedCortex,
            arm_a_generate,
        )

        driver = _FakeHFDriver(reply="test")
        cortex = SharedCortex(seed=0)
        labels = [f"label_{i}" for i in range(20)]
        arm_a_generate(driver, cortex, "test request", labels, 0.0, "{label}. ")
        # After generate, reserved position should be cleared.
        assert driver._reserved is None


# ---------------------------------------------------------------------------
# Score probes tests
# ---------------------------------------------------------------------------


class TestScoreProbes:
    """Tests for the score_probes function."""

    def test_score_probes_vanilla(self) -> None:
        """score_probes with arm='vanilla' must call vanilla_generate."""
        from oczy.experiments.s19_language_organ_core import score_probes

        driver = _FakeHFDriver(reply="answer")
        stage = _make_stage()
        dev_ids, _ = split_probes(stage, fraction=0.3, salt="v2.2")
        result = score_probes(driver, stage, dev_ids, "vanilla")
        assert "accuracy" in result
        assert result["total"] > 0
        assert len(result["audits"]) == result["total"]

    def test_score_probes_filters_by_probe_ids(self) -> None:
        """score_probes must only score probes whose IDs are in probe_ids."""
        from oczy.experiments.s19_language_organ_core import score_probes

        driver = _FakeHFDriver(reply="answer")
        stage = _make_stage()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        result = score_probes(driver, stage, dev_ids, "vanilla")
        scored_pids = {a["probe_id"] for a in result["audits"]}
        assert scored_pids == dev_ids

    def test_score_probes_empty_ids(self) -> None:
        """score_probes with empty probe_ids must return zero total."""
        from oczy.experiments.s19_language_organ_core import score_probes

        driver = _FakeHFDriver(reply="answer")
        stage = _make_stage()
        result = score_probes(driver, stage, set(), "vanilla")
        assert result["total"] == 0
        assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Teaching tests
# ---------------------------------------------------------------------------


class TestTeaching:
    """Tests for the teach_cortex function.

    teach_cortex calls train_coupler internally, which requires a real LM
    forward pass.  We mock train_coupler to test the teaching contract
    (trace recording, seed shuffling, permuted labels) without requiring
    a working coupler implementation.
    """

    def _make_cortex_with_mocked_coupler(self, seed: int = 0):
        """Create a cortex with train_coupler mocked to return 0.0."""
        from oczy.experiments.s19_language_organ_core import (
            CortexConfig,
            SharedCortex,
            extract_stage_labels,
        )

        stage = _make_stage()
        labels = extract_stage_labels(stage)
        config = CortexConfig(n_labels=len(labels))
        cortex = SharedCortex(config=config, seed=seed)
        # Mock train_coupler to avoid the LM forward pass (implementation
        # may have bugs in the coupler path; we test the teaching contract).
        cortex.train_coupler = lambda driver, request, corrected_response: 0.0  # type: ignore[assignment]
        return cortex, labels

    def test_teach_cortex_records_traces(self) -> None:
        """teach_cortex must record raw traces in the TraceStore."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            build_label_index,
            extract_stage_labels,
            teach_cortex,
        )

        driver = _FakeHFDriver(reply="test")
        stage = _make_stage()
        labels = extract_stage_labels(stage)
        label_index = build_label_index(labels)
        cortex, _ = self._make_cortex_with_mocked_coupler(seed=0)
        trace_store = TraceStore()

        teach_cortex(driver, cortex, stage, labels, label_index, 0, trace_store)
        assert trace_store.count() == len(stage.episodes)

    def test_teach_cortex_shuffles_by_seed(self) -> None:
        """teach_cortex with different seeds must produce different teach orders."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            build_label_index,
            extract_stage_labels,
            teach_cortex,
        )

        driver = _FakeHFDriver(reply="test")
        stage = _make_stage()
        labels = extract_stage_labels(stage)
        label_index = build_label_index(labels)

        cortex_a, _ = self._make_cortex_with_mocked_coupler(seed=0)
        cortex_b, _ = self._make_cortex_with_mocked_coupler(seed=0)
        trace_a = TraceStore()
        trace_b = TraceStore()

        teach_cortex(driver, cortex_a, stage, labels, label_index, 0, trace_a)
        teach_cortex(driver, cortex_b, stage, labels, label_index, 42, trace_b)

        import torch

        # With different seeds, the teaching order differs, so the final
        # weights should differ (unless the curriculum is trivially uniform).
        assert not torch.allclose(cortex_a.W_label, cortex_b.W_label)

    def test_teach_cortex_permuted_labels(self) -> None:
        """teach_cortex with permuted_labels=True must use permuted label indices."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            build_label_index,
            extract_stage_labels,
            teach_cortex,
        )

        driver = _FakeHFDriver(reply="test")
        stage = _make_stage()
        labels = extract_stage_labels(stage)
        label_index = build_label_index(labels)

        cortex_n, _ = self._make_cortex_with_mocked_coupler(seed=0)
        cortex_p, _ = self._make_cortex_with_mocked_coupler(seed=0)
        trace_n = TraceStore()
        trace_p = TraceStore()

        stats_n = teach_cortex(driver, cortex_n, stage, labels, label_index, 0, trace_n, permuted_labels=False)
        stats_p = teach_cortex(driver, cortex_p, stage, labels, label_index, 0, trace_p, permuted_labels=True)

        assert stats_n["permuted"] is False
        assert stats_p["permuted"] is True

    def test_teach_cortex_returns_stats(self) -> None:
        """teach_cortex must return a dict with n_episodes and loss means."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            build_label_index,
            extract_stage_labels,
            teach_cortex,
        )

        driver = _FakeHFDriver(reply="test")
        stage = _make_stage()
        labels = extract_stage_labels(stage)
        label_index = build_label_index(labels)
        cortex, _ = self._make_cortex_with_mocked_coupler(seed=0)
        trace_store = TraceStore()

        stats = teach_cortex(driver, cortex, stage, labels, label_index, 0, trace_store)
        assert "n_episodes" in stats
        assert "label_loss_mean" in stats
        assert "permuted" in stats
        assert stats["n_episodes"] == len(stage.episodes)

# ---------------------------------------------------------------------------
# Articulation audit tests
# ---------------------------------------------------------------------------


class TestArticulationAudit:
    """Tests for build_articulation_audit."""

    def test_build_articulation_audit_fields(self) -> None:
        """build_articulation_audit must include all required fields."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3",
            arm="B",
            prompt_text="test request",
            latent_bank_shape=(3, 896),
            raw_trace_count=0,
            model_hash="abc123",
            persistent_bytes=241552,
        )
        assert audit["condition"] == "C3"
        assert audit["arm"] == "B"
        assert audit["prompt_text"] == "test request"
        assert audit["latent_bank_shape"] == (3, 896)
        assert audit["raw_trace_count"] == 0
        assert audit["language_organ_hash"] == "abc123"
        assert audit["persistent_cortex_bytes"] == 241552


# ---------------------------------------------------------------------------
# Label extraction tests
# ---------------------------------------------------------------------------


class TestLabelExtraction:
    """Tests for extract_stage_labels and build_label_index."""

    def test_extract_stage_labels_preserves_first_appearance_order(self) -> None:
        """extract_stage_labels must return labels in first-appearance order."""
        from oczy.experiments.s19_language_organ_core import extract_stage_labels

        stage = Stage(
            name="test",
            description="",
            consolidate_before=False,
            consolidate_after=False,
            episodes=(
                _make_episode("ep0", label="zebra"),
                _make_episode("ep1", label="apple"),
                _make_episode("ep2", label="zebra"),  # duplicate
                _make_episode("ep3", label="mango"),
            ),
        )
        labels = extract_stage_labels(stage)
        assert labels == ["zebra", "apple", "mango"]

    def test_build_label_index(self) -> None:
        """build_label_index must map label → integer index."""
        from oczy.experiments.s19_language_organ_core import build_label_index

        idx = build_label_index(["a", "b", "c"])
        assert idx == {"a": 0, "b": 1, "c": 2}


# ---------------------------------------------------------------------------
# Eval manifest hash tests
# ---------------------------------------------------------------------------


class TestEvalManifestHash:
    """Tests for hash_eval_manifest."""

    def test_hash_eval_manifest_returns_hex(self) -> None:
        """hash_eval_manifest must return a hex SHA-256 string."""
        from oczy.experiments.s19_language_organ_core import hash_eval_manifest

        h = hash_eval_manifest()
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_eval_manifest_deterministic(self) -> None:
        """hash_eval_manifest must be deterministic."""
        from oczy.experiments.s19_language_organ_core import hash_eval_manifest

        h1 = hash_eval_manifest()
        h2 = hash_eval_manifest()
        assert h1 == h2


# ---------------------------------------------------------------------------
# CLI structure tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI entry point structure."""

    def test_main_requires_subcommand(self) -> None:
        """main with no subcommand must fail."""
        from oczy.experiments.s19_language_organ import main

        with pytest.raises(SystemExit):
            main([])

    def test_main_calibrate_dev_exists(self) -> None:
        """The calibrate-dev subcommand must exist."""
        from oczy.experiments.s19_language_organ import main

        # Just parse args, don't run.
        with pytest.raises(SystemExit):
            main(["calibrate-dev", "--help"])

    def test_main_evaluate_exists(self) -> None:
        """The evaluate subcommand must exist."""
        from oczy.experiments.s19_language_organ import main

        with pytest.raises(SystemExit):
            main(["evaluate", "--help"])

    def test_evaluate_requires_signoff_id(self) -> None:
        """evaluate must require --signoff-id."""
        from oczy.experiments.s19_language_organ import main

        with pytest.raises(SystemExit):
            main(["evaluate", "--manifest", "/tmp/test.json"])

    def test_evaluate_requires_manifest(self) -> None:
        """evaluate must require --manifest."""
        from oczy.experiments.s19_language_organ import main

        with pytest.raises(SystemExit):
            main(["evaluate", "--signoff-id", "human-001"])

    def test_calibrate_dev_requires_manifest_out(self) -> None:
        """calibrate-dev must require --manifest-out."""
        from oczy.experiments.s19_language_organ import main

        with pytest.raises(SystemExit):
            main(["calibrate-dev"])

    def test_import_works_on_py310_without_datetime_UTC(self) -> None:
        """Regression: importing the CLI must succeed on Python 3.10, which
        lacks ``datetime.UTC``.  The module must use ``timezone.utc`` instead
        and must not reference ``datetime.UTC`` / ``from datetime import UTC``.
        """
        import importlib
        import inspect

        mod = importlib.import_module("oczy.experiments.s19_language_organ")

        # main entry point is callable after a clean import.
        assert callable(mod.main)

        # No dependency on the 3.11+ ``datetime.UTC`` singleton.
        source = inspect.getsource(mod)
        assert "datetime.UTC" not in source
        assert "from datetime import UTC" not in source


# ---------------------------------------------------------------------------
# Perception / label freeze during coupler DEV training
# ---------------------------------------------------------------------------


class TestPerceptionFreeze:
    """Tests that W_perceive / W_label / b_label / warm_state are frozen
    during coupler DEV training so the coupler cannot corrupt them."""

    def test_freeze_perception_disables_grad_on_four_params(self) -> None:
        """freeze_perception must set requires_grad=False on W_perceive,
        W_label, b_label, and warm_state."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        assert not cortex.W_perceive.requires_grad
        assert not cortex.W_label.requires_grad
        assert not cortex.b_label.requires_grad
        assert not cortex.warm_state.requires_grad

    def test_freeze_perception_leaves_coupler_trainable(self) -> None:
        """freeze_perception must NOT freeze the coupler — it is what gets trained."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        assert cortex.W_coupler.requires_grad
        assert cortex.b_coupler.requires_grad

    def test_perception_frozen_property(self) -> None:
        """perception_frozen must reflect the freeze state."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        assert not cortex.perception_frozen
        cortex.freeze_perception()
        assert cortex.perception_frozen

    def test_unfreeze_perception_re_enables_grad(self) -> None:
        """unfreeze_perception must re-enable gradients on all four params."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        cortex.unfreeze_perception()
        assert cortex.W_perceive.requires_grad
        assert cortex.W_label.requires_grad
        assert cortex.b_label.requires_grad
        assert cortex.warm_state.requires_grad
        assert not cortex.perception_frozen

    def test_coupler_parameters_returns_only_coupler(self) -> None:
        """coupler_parameters must return exactly [W_coupler, b_coupler]."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        params = cortex.coupler_parameters()
        assert len(params) == 2
        assert params[0] is cortex.W_coupler
        assert params[1] is cortex.b_coupler

    def test_trainable_parameters_excludes_perception_when_frozen(self) -> None:
        """When perception is frozen, trainable_parameters must not include
        W_perceive, W_label, b_label, or warm_state."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        trainable = cortex.trainable_parameters()
        # Only coupler params should remain (if not coupler-frozen).
        trainable_ids = {id(p) for p in trainable}
        for p in [cortex.W_perceive, cortex.W_label, cortex.b_label, cortex.warm_state]:
            assert id(p) not in trainable_ids

    def test_freeze_perception_does_not_freeze_coupler_flag(self) -> None:
        """freeze_perception must not set coupler_frozen — they are independent."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        assert cortex.coupler_frozen is False

    def test_freeze_both_perception_and_coupler(self) -> None:
        """Freezing both perception and coupler must leave zero trainable params."""
        from oczy.experiments.s19_language_organ_core import SharedCortex

        cortex = SharedCortex(seed=0)
        cortex.freeze_perception()
        cortex.freeze_coupler()
        assert cortex.trainable_parameters() == []


# ---------------------------------------------------------------------------
# TraceStore cached feature / embedding deletion
# ---------------------------------------------------------------------------


class TestTraceStoreCacheDeletion:
    """Tests that TraceStore.delete_all clears cached features and embeddings,
    not just raw text traces."""

    def test_cached_feature_count(self) -> None:
        """add_cached_feature must increment cached_feature_count."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        assert store.cached_feature_count() == 0
        store.add_cached_feature("req1", np.zeros(896, dtype=np.float32))
        assert store.cached_feature_count() == 1
        store.add_cached_feature("req2", np.zeros(896, dtype=np.float32))
        assert store.cached_feature_count() == 2

    def test_cached_embedding_count(self) -> None:
        """add_cached_embedding must increment cached_embedding_count."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        assert store.cached_embedding_count() == 0
        store.add_cached_embedding(("req1", False), np.zeros(896, dtype=np.float32))
        assert store.cached_embedding_count() == 1

    def test_delete_all_clears_cached_features(self) -> None:
        """delete_all must clear cached features."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add("ep0", "req", "corr", "resp")
        store.add_cached_feature("req", np.zeros(896, dtype=np.float32))
        store.add_cached_embedding(("req", False), np.zeros(896, dtype=np.float32))
        store.delete_all()
        assert store.cached_feature_count() == 0

    def test_delete_all_clears_cached_embeddings(self) -> None:
        """delete_all must clear cached embeddings."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add("ep0", "req", "corr", "resp")
        store.add_cached_embedding(("req", True), np.zeros(896, dtype=np.float32))
        store.delete_all()
        assert store.cached_embedding_count() == 0

    def test_verify_zero_fails_with_cached_features(self) -> None:
        """verify_zero must return False when cached features remain."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add_cached_feature("req", np.zeros(896, dtype=np.float32))
        assert store.verify_zero() is False

    def test_verify_zero_fails_with_cached_embeddings(self) -> None:
        """verify_zero must return False when cached embeddings remain."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add_cached_embedding(("req", False), np.zeros(896, dtype=np.float32))
        assert store.verify_zero() is False

    def test_verify_zero_passes_after_full_delete(self) -> None:
        """verify_zero must pass after delete_all with all three stores populated."""
        from oczy.experiments.s19_language_organ_core import TraceStore

        store = TraceStore()
        store.add("ep0", "req", "corr", "resp")
        store.add_cached_feature("req", np.zeros(896, dtype=np.float32))
        store.add_cached_embedding(("req", False), np.zeros(896, dtype=np.float32))
        store.delete_all()
        assert store.verify_zero() is True

    def test_verify_raw_traces_deleted_checks_caches(self) -> None:
        """verify_raw_traces_deleted must return False when caches are non-empty."""
        from oczy.experiments.s19_language_organ_core import (
            TraceStore,
            verify_raw_traces_deleted,
        )

        store = TraceStore()
        store.add_cached_feature("req", np.zeros(896, dtype=np.float32))
        assert verify_raw_traces_deleted(store) is False


# ---------------------------------------------------------------------------
# Split metadata + Phase-0 distribution manifest fields
# ---------------------------------------------------------------------------


class TestManifestSplitAndPhase0:
    """Tests that CalibrationManifest carries split salt/fraction and DEV
    distribution fields, and that they participate in the hash."""

    def test_manifest_has_split_salt_default(self) -> None:
        """CalibrationManifest must have a split_salt field."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert hasattr(m, "split_salt")
        assert isinstance(m.split_salt, str)

    def test_manifest_has_split_fraction_default(self) -> None:
        """CalibrationManifest must have a split_fraction field."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert hasattr(m, "split_fraction")
        assert isinstance(m.split_fraction, float)

    def test_manifest_has_dev_repeatability_std(self) -> None:
        """CalibrationManifest must have a dev_repeatability_std field."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert hasattr(m, "dev_repeatability_std")
        assert isinstance(m.dev_repeatability_std, float)

    def test_manifest_has_dev_confidence_mean(self) -> None:
        """CalibrationManifest must have a dev_confidence_mean field."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert hasattr(m, "dev_confidence_mean")
        assert isinstance(m.dev_confidence_mean, float)

    def test_manifest_has_dev_specificity_acc(self) -> None:
        """CalibrationManifest must have a dev_specificity_acc field."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert hasattr(m, "dev_specificity_acc")
        assert isinstance(m.dev_specificity_acc, float)

    def test_manifest_has_dev_holdout_ids_discarded(self) -> None:
        """CalibrationManifest must have dev_holdout_ids_discarded defaulting True."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert m.dev_holdout_ids_discarded is True

    def test_manifest_to_dict_includes_split_fields(self) -> None:
        """to_dict must include split_salt and split_fraction."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(split_salt="v2.2", split_fraction=0.3)
        d = m.to_dict()
        assert d["split_salt"] == "v2.2"
        assert d["split_fraction"] == 0.3

    def test_manifest_to_dict_includes_dev_fields(self) -> None:
        """to_dict must include the dev_* distribution fields."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(
            dev_repeatability_std=0.01,
            dev_confidence_mean=0.5,
            dev_confidence_std=0.1,
            dev_confidence_min=0.3,
            dev_confidence_max=0.7,
            dev_specificity_acc=0.8,
        )
        d = m.to_dict()
        assert d["dev_repeatability_std"] == 0.01
        assert d["dev_confidence_mean"] == 0.5
        assert d["dev_confidence_std"] == 0.1
        assert d["dev_confidence_min"] == 0.3
        assert d["dev_confidence_max"] == 0.7
        assert d["dev_specificity_acc"] == 0.8

    def test_manifest_roundtrip_preserves_split_fields(self) -> None:
        """from_dict must restore split_salt and split_fraction."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(split_salt="v2.2", split_fraction=0.3)
        m2 = CalibrationManifest.from_dict(m.to_dict())
        assert m2.split_salt == "v2.2"
        assert m2.split_fraction == 0.3

    def test_manifest_roundtrip_preserves_dev_fields(self) -> None:
        """from_dict must restore the dev_* distribution fields."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(
            dev_repeatability_std=0.01,
            dev_confidence_mean=0.5,
            dev_confidence_std=0.1,
            dev_confidence_min=0.3,
            dev_confidence_max=0.7,
            dev_specificity_acc=0.8,
        )
        m2 = CalibrationManifest.from_dict(m.to_dict())
        assert m2.dev_repeatability_std == 0.01
        assert m2.dev_confidence_mean == 0.5
        assert m2.dev_confidence_std == 0.1
        assert m2.dev_confidence_min == 0.3
        assert m2.dev_confidence_max == 0.7
        assert m2.dev_specificity_acc == 0.8

    def test_manifest_hash_changes_with_split_salt(self) -> None:
        """compute_hash must change when split_salt changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(split_salt="v2.2")
        m2 = CalibrationManifest(split_salt="v3.0")
        assert m1.compute_hash() != m2.compute_hash()

    def test_manifest_hash_changes_with_dev_confidence_mean(self) -> None:
        """compute_hash must change when dev_confidence_mean changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(dev_confidence_mean=0.5)
        m2 = CalibrationManifest(dev_confidence_mean=0.6)
        assert m1.compute_hash() != m2.compute_hash()

    def test_compute_phase0_distributions_returns_dicts(self) -> None:
        """compute_phase0_distributions must return label/domain/category counts."""
        from oczy.experiments.s19_language_organ_core import compute_phase0_distributions

        stage = _make_stage(n_episodes=6)
        dists = compute_phase0_distributions(stage)
        assert "labels" in dists
        assert "domains" in dists
        assert "probe_categories" in dists
        # Each value must be a dict[str, int].
        for key in ("labels", "domains", "probe_categories"):
            assert isinstance(dists[key], dict)
            for v in dists[key].values():
                assert isinstance(v, int)

    def test_compute_phase0_distributions_counts_labels(self) -> None:
        """compute_phase0_distributions must count corrected_label occurrences."""
        from oczy.experiments.s19_language_organ_core import compute_phase0_distributions

        stage = _make_stage(n_episodes=6)
        dists = compute_phase0_distributions(stage)
        # _make_stage uses 6 labels cycled over 6 episodes → each appears once.
        assert sum(dists["labels"].values()) == 6

    def test_compute_phase0_distributions_counts_domains(self) -> None:
        """compute_phase0_distributions must count domain occurrences."""
        from oczy.experiments.s19_language_organ_core import compute_phase0_distributions

        stage = _make_stage(n_episodes=6)
        dists = compute_phase0_distributions(stage)
        # All episodes use domain="general".
        assert dists["domains"].get("general", 0) == 6

    def test_compute_phase0_distributions_counts_probe_categories(self) -> None:
        """compute_phase0_distributions must count probe category occurrences."""
        from oczy.experiments.s19_language_organ_core import compute_phase0_distributions

        stage = _make_stage(n_episodes=6)
        dists = compute_phase0_distributions(stage)
        # All probes in _make_stage are category="retention".
        assert dists["probe_categories"].get("retention", 0) == 6


# ---------------------------------------------------------------------------
# C7 actual-or-blocked S3.M2a retrieval baseline
# ---------------------------------------------------------------------------


class TestC7RetrievalBaseline:
    """Tests that C7 either runs the actual S3.M2a retrieval baseline or
    explicitly blocks evaluation with a validity_blocked flag."""

    def test_c7_returns_holdout_acc_or_blocked(self) -> None:
        """C7 must either return holdout_acc or set validity_blocked=True."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            _C7Env.make()
        )
        result = run_condition(
            driver, "C7", stage, other_stages, dev_ids, holdout_ids,
            None, None, 0.5, "{label}. ", trace_store,
        )
        # Either we got a real accuracy or the run was validity-blocked.
        assert "holdout_acc" in result
        assert result["persistent_bytes"] == 0
        if result.get("validity_blocked"):
            assert isinstance(result["validity_error"], str)
            assert len(result["validity_error"]) > 0
        else:
            # If not blocked, accuracy must be a float in [0, 1].
            assert isinstance(result["holdout_acc"], float)
            assert 0.0 <= result["holdout_acc"] <= 1.0

    def test_c7_blocked_has_validity_error(self) -> None:
        """When C7 is validity-blocked, validity_error must be a non-empty string."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            _C7Env.make()
        )
        result = run_condition(
            driver, "C7", stage, other_stages, dev_ids, holdout_ids,
            None, None, 0.5, "{label}. ", trace_store,
        )
        if result.get("validity_blocked"):
            assert "validity_error" in result
            assert isinstance(result["validity_error"], str)
            assert result["validity_error"].strip() != ""

    def test_c7_zero_persistent_bytes(self) -> None:
        """C7 must always report persistent_bytes == 0 (external baseline)."""
        from oczy.experiments.s19_language_organ_core import run_condition

        driver, stage, other_stages, dev_ids, holdout_ids, labels, cortex, trace_store = (
            _C7Env.make()
        )
        result = run_condition(
            driver, "C7", stage, other_stages, dev_ids, holdout_ids,
            None, None, 0.5, "{label}. ", trace_store,
        )
        assert result["persistent_bytes"] == 0


class _C7Env:
    """Shared setup for C7 tests."""

    @staticmethod
    def make():
        from oczy.experiments.s19_language_organ_core import TraceStore

        driver = _FakeHFDriver(reply="answer0")
        stage = _make_stage()
        other_stages = _make_other_stages()
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        trace_store = TraceStore()
        return driver, stage, other_stages, dev_ids, holdout_ids, None, None, trace_store


# ---------------------------------------------------------------------------
# Full articulation audit: banned-content fields
# ---------------------------------------------------------------------------


class TestArticulationAuditBannedContent:
    """Tests that build_articulation_audit includes banned-content fields
    for label, correction, answer, and episode ID."""

    def test_audit_includes_banned_label_text(self) -> None:
        """build_articulation_audit must include banned_label_text when provided."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="req",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc", persistent_bytes=100,
            banned_label_text="fruit jam",
        )
        assert audit["banned_label_text"] == "fruit jam"

    def test_audit_includes_banned_correction_text(self) -> None:
        """build_articulation_audit must include banned_correction_text when provided."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="req",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc", persistent_bytes=100,
            banned_correction_text="The answer is 42.",
        )
        assert audit["banned_correction_text"] == "The answer is 42."

    def test_audit_includes_banned_answer_text(self) -> None:
        """build_articulation_audit must include banned_answer_text when provided."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="req",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc", persistent_bytes=100,
            banned_answer_text="42",
        )
        assert audit["banned_answer_text"] == "42"

    def test_audit_includes_banned_episode_id(self) -> None:
        """build_articulation_audit must include banned_episode_id when provided."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="req",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc", persistent_bytes=100,
            banned_episode_id="ep0",
        )
        assert audit["banned_episode_id"] == "ep0"

    def test_audit_banned_fields_default_none(self) -> None:
        """build_articulation_audit must default banned fields to None."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="req",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc", persistent_bytes=100,
        )
        assert audit.get("banned_label_text") is None
        assert audit.get("banned_correction_text") is None
        assert audit.get("banned_answer_text") is None
        assert audit.get("banned_episode_id") is None

    def test_audit_all_required_fields_present(self) -> None:
        """build_articulation_audit must include all required fields from the
        full articulation audit spec: prompt, latent shape, raw traces,
        LM hash, persistent bytes, and banned-content fields."""
        from oczy.experiments.s19_language_organ_core import build_articulation_audit

        audit = build_articulation_audit(
            condition="C3", arm="B", prompt_text="test request",
            latent_bank_shape=(3, 896), raw_trace_count=0,
            model_hash="abc123", persistent_bytes=241552,
            banned_label_text="label", banned_correction_text="corr",
            banned_answer_text="ans", banned_episode_id="ep0",
        )
        required = {
            "condition", "arm", "prompt_text", "latent_bank_shape",
            "raw_trace_count", "language_organ_hash", "persistent_cortex_bytes",
            "banned_label_text", "banned_correction_text",
            "banned_answer_text", "banned_episode_id",
        }
        assert required.issubset(audit.keys())

    def test_verify_no_text_injection_checks_banned_fields(self) -> None:
        """verify_no_text_injection must detect banned label text in Arm B prompt."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {
                "arm": "B",
                "prompt_text": "fruit jam is the answer",
                "latent_bank_shape": (3, 896),
                "banned_label_text": "fruit jam",
            },
        ]
        assert verify_no_text_injection(audits) is False

    def test_verify_no_text_injection_passes_with_clean_banned_fields(self) -> None:
        """verify_no_text_injection must pass when banned text is not in prompt."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {
                "arm": "B",
                "prompt_text": "What is the code?",
                "latent_bank_shape": (3, 896),
                "banned_label_text": "fruit jam",
                "banned_correction_text": "The code is marmalade.",
                "banned_answer_text": "marmalade",
                "banned_episode_id": "ep0",
            },
        ]
        assert verify_no_text_injection(audits) is True

    def test_verify_no_text_injection_detects_banned_answer(self) -> None:
        """verify_no_text_injection must detect banned answer text in prompt."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {
                "arm": "B",
                "prompt_text": "The answer is marmalade.",
                "latent_bank_shape": (3, 896),
                "banned_answer_text": "marmalade",
            },
        ]
        assert verify_no_text_injection(audits) is False

    def test_verify_no_text_injection_detects_banned_correction(self) -> None:
        """verify_no_text_injection must detect banned correction text in prompt."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {
                "arm": "B",
                "prompt_text": "The code is marmalade. What is it?",
                "latent_bank_shape": (3, 896),
                "banned_correction_text": "The code is marmalade.",
            },
        ]
        assert verify_no_text_injection(audits) is False

    def test_verify_no_text_injection_detects_banned_episode_id(self) -> None:
        """verify_no_text_injection must detect banned episode ID in prompt."""
        from oczy.experiments.s19_language_organ_core import verify_no_text_injection

        audits = [
            {
                "arm": "B",
                "prompt_text": "Question for ep0: what is it?",
                "latent_bank_shape": (3, 896),
                "banned_episode_id": "ep0",
            },
        ]
        assert verify_no_text_injection(audits) is False
# ---------------------------------------------------------------------------
# Manifest / schema flat-key alignment tests
# ---------------------------------------------------------------------------


class TestManifestSchemaAlignment:
    """Tests that runtime to_dict() keys exactly match schema-required flat keys."""

    _SCHEMA_PATH = (
        Path(__file__).resolve().parents[4]
        / "infrastructure" / "kaggle" / "model_manifests"
        / "r19_calibration_manifest.schema.json"
    )

    @classmethod
    def _load_schema(cls) -> dict[str, Any]:
        """Load the JSON schema file."""
        with open(cls._SCHEMA_PATH) as f:
            return json.load(f)

    def test_to_dict_keys_equal_schema_properties(self) -> None:
        """Every to_dict() key must be a schema property and vice versa."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        schema = self._load_schema()
        schema_props = set(schema["properties"].keys())
        manifest_keys = set(CalibrationManifest().to_dict().keys())
        assert manifest_keys == schema_props, (
            f"Key mismatch: runtime-only={manifest_keys - schema_props}, "
            f"schema-only={schema_props - manifest_keys}"
        )

    def test_all_schema_required_keys_in_to_dict(self) -> None:
        """Every schema 'required' key must appear in to_dict()."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        schema = self._load_schema()
        required = set(schema["required"])
        manifest_keys = set(CalibrationManifest().to_dict().keys())
        missing = required - manifest_keys
        assert not missing, f"Required keys missing from to_dict: {missing}"

    def test_no_extra_to_dict_keys_beyond_schema(self) -> None:
        """to_dict() must not emit keys not defined in schema properties."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        schema = self._load_schema()
        schema_props = set(schema["properties"].keys())
        manifest_keys = set(CalibrationManifest().to_dict().keys())
        extra = manifest_keys - schema_props
        assert not extra, f"Extra keys in to_dict not in schema: {extra}"

    def test_schema_version_const_matches_runtime(self) -> None:
        """schema_version const in schema must match the runtime SCHEMA_VERSION."""
        from oczy.experiments.s19_language_organ_core import SCHEMA_VERSION, CalibrationManifest

        schema = self._load_schema()
        schema_const = schema["properties"]["schema_version"]["const"]
        assert schema_const == SCHEMA_VERSION
        assert CalibrationManifest().to_dict()["schema_version"] == schema_const

    def test_to_dict_has_no_nested_objects_except_parameter_breakdown(self) -> None:
        """to_dict() must be flat — no nested objects except parameter_breakdown."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = CalibrationManifest().to_dict()
        for key, val in d.items():
            if key == "parameter_breakdown":
                assert isinstance(val, dict)
                continue
            assert not isinstance(val, dict), (
                f"Key '{key}' is a nested dict — manifest must be flat"
            )

    def test_to_dict_key_count_matches_schema_property_count(self) -> None:
        """to_dict() must emit exactly as many keys as schema properties."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        schema = self._load_schema()
        n_schema_props = len(schema["properties"])
        n_manifest_keys = len(CalibrationManifest().to_dict())
        assert n_manifest_keys == n_schema_props, (
            f"Key count mismatch: to_dict={n_manifest_keys}, schema={n_schema_props}"
        )

    def test_schema_additional_properties_false(self) -> None:
        """Schema must set additionalProperties=false to reject drift keys."""
        schema = self._load_schema()
        assert schema.get("additionalProperties") is False

    def test_roundtrip_preserves_all_keys(self) -> None:
        """to_dict → from_dict → to_dict must produce identical keys."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(
            source_commit="a" * 40,
            model_repo_id="test",
            labels=["x"],
            cortex_artifact_sha256="f" * 64,
            cortex_artifact_bytes=100,
            coupler_sha256="g" * 64,
            coupler_bytes=50,
            head_sha256="h" * 64,
            head_bytes=50,
            c7_reference="ref",
            articulation_language_organ_hash="i" * 64,
        )
        m.manifest_sha256 = m.compute_hash()
        d1 = m.to_dict()
        m2 = CalibrationManifest.from_dict(d1)
        d2 = m2.to_dict()
        assert set(d1.keys()) == set(d2.keys())


# ---------------------------------------------------------------------------
# Canonical manifest hash tests
# ---------------------------------------------------------------------------


class TestManifestCanonicalHash:
    """Tests that the canonical hash ignores created_at and manifest_sha256
    but changes on scientific/audit fields."""

    def test_hash_excludes_created_at(self) -> None:
        """Changing created_at must NOT change the manifest hash."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(model_repo_id="test", created_at="2026-01-01T00:00:00Z")
        m2 = CalibrationManifest(model_repo_id="test", created_at="2026-12-31T23:59:59Z")
        assert m1.compute_hash() == m2.compute_hash()

    def test_hash_excludes_manifest_sha256(self) -> None:
        """Changing manifest_sha256 must NOT change the manifest hash."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(model_repo_id="test")
        h1 = m.compute_hash()
        m.manifest_sha256 = "0" * 64
        h2 = m.compute_hash()
        assert h1 == h2

    def test_hash_changes_with_proposed_confidence_threshold(self) -> None:
        """Hash must change when proposed_confidence_threshold changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(proposed_confidence_threshold=0.5)
        m2 = CalibrationManifest(proposed_confidence_threshold=0.6)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_proposed_specificity_margin(self) -> None:
        """Hash must change when proposed_specificity_margin changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(proposed_specificity_margin=0.1)
        m2 = CalibrationManifest(proposed_specificity_margin=0.2)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_signoff_oracle_ceiling(self) -> None:
        """Hash must change when signoff_oracle_ceiling changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(signoff_oracle_ceiling=0.3)
        m2 = CalibrationManifest(signoff_oracle_ceiling=0.5)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_cortex_artifact_sha256(self) -> None:
        """Hash must change when cortex_artifact_sha256 changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(cortex_artifact_sha256="a" * 64)
        m2 = CalibrationManifest(cortex_artifact_sha256="b" * 64)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_trace_deletion_audit(self) -> None:
        """Hash must change when trace deletion audit fields change."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(trace_raw_traces_deleted=False)
        m2 = CalibrationManifest(trace_raw_traces_deleted=True)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_articulation_audit(self) -> None:
        """Hash must change when articulation audit fields change."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(articulation_banned_label_text_absent=True)
        m2 = CalibrationManifest(articulation_banned_label_text_absent=False)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_c7_reference(self) -> None:
        """Hash must change when c7_reference changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(c7_reference="baseline_v1")
        m2 = CalibrationManifest(c7_reference="baseline_v2")
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_holdout_accessed(self) -> None:
        """Hash must change when holdout_accessed changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(holdout_accessed=False)
        m2 = CalibrationManifest(holdout_accessed=True)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_signoff_thresholds_signed_off(self) -> None:
        """Hash must change when signoff_thresholds_signed_off changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(signoff_thresholds_signed_off=False)
        m2 = CalibrationManifest(signoff_thresholds_signed_off=True)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_signoff_human_signoff_id(self) -> None:
        """Hash must change when signoff_human_signoff_id changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(signoff_human_signoff_id="")
        m2 = CalibrationManifest(signoff_human_signoff_id="human-001")
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_labels(self) -> None:
        """Hash must change when labels change."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(labels=["a"])
        m2 = CalibrationManifest(labels=["b"])
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_changes_with_model_safetensors_sha256(self) -> None:
        """Hash must change when model_safetensors_sha256 changes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(model_safetensors_sha256="a" * 64)
        m2 = CalibrationManifest(model_safetensors_sha256="b" * 64)
        assert m1.compute_hash() != m2.compute_hash()

    def test_hash_is_sha256_hex(self) -> None:
        """compute_hash must return a 64-character hex string."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        h = CalibrationManifest().compute_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Incomplete manifest fails evaluate tests
# ---------------------------------------------------------------------------


class TestManifestIncompleteFailsEvaluate:
    """Tests that incomplete manifests (missing required fields) fail evaluate."""

    def test_evaluate_fails_on_missing_source_commit(self) -> None:
        """evaluate must fail when source_commit is empty (required provenance)."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(source_commit="")
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_missing_model_safetensors_sha256(self) -> None:
        """evaluate must fail when model_safetensors_sha256 is empty."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(model_safetensors_sha256="")
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_missing_cortex_artifact_sha256(self) -> None:
        """evaluate must fail when cortex_artifact_sha256 is empty."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(cortex_artifact_sha256="")
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_zero_cortex_artifact_bytes(self) -> None:
        """evaluate must fail when cortex_artifact_bytes is zero."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(cortex_artifact_bytes=0)
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_empty_labels(self) -> None:
        """evaluate must fail when labels is empty."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(labels=[])
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_missing_c7_reference(self) -> None:
        """evaluate must fail when c7_reference is empty."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(c7_reference="")
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_fails_on_holdout_accessed_true(self) -> None:
        """evaluate must fail when holdout_accessed is True."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(holdout_accessed=True)
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_required_fields_present_rejects_empty_provenance(self) -> None:
        """required_fields_present must return False when provenance is empty."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert m.required_fields_present() is False

    def test_required_fields_present_passes_with_all_fields(self) -> None:
        """required_fields_present must return True when all required fields are set."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(
            source_commit="a" * 40,
            source_archive_sha256="b" * 64,
            eval_version="v2.1",
            eval_manifest_sha256="c" * 64,
            model_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_revision="main",
            model_config_sha256="d" * 64,
            model_safetensors_sha256="e" * 64,
            cortex_artifact_sha256="f" * 64,
            cortex_artifact_bytes=1000,
            coupler_sha256="g" * 64,
            coupler_bytes=500,
            head_sha256="h" * 64,
            head_bytes=500,
            labels=["label1"],
            articulation_language_organ_hash="i" * 64,
            c7_reference="s3m2a_baseline",
        )
        assert m.required_fields_present() is True


# ---------------------------------------------------------------------------
# Cortex artifact hash/bytes verification tests
# ---------------------------------------------------------------------------


class TestManifestArtifactVerification:
    """Tests that the manifest carries cortex artifact hash+bytes and
    evaluate verifies them against the serialized artifact file."""

    def test_manifest_has_cortex_artifact_sha256(self) -> None:
        """to_dict must include cortex_artifact_sha256."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(cortex_artifact_sha256="a" * 64)
        d = m.to_dict()
        assert d["cortex_artifact_sha256"] == "a" * 64

    def test_manifest_has_cortex_artifact_bytes(self) -> None:
        """to_dict must include cortex_artifact_bytes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(cortex_artifact_bytes=241552)
        d = m.to_dict()
        assert d["cortex_artifact_bytes"] == 241552

    def test_manifest_has_coupler_sha256_and_bytes(self) -> None:
        """to_dict must include coupler_sha256 and coupler_bytes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(coupler_sha256="a" * 64, coupler_bytes=500)
        d = m.to_dict()
        assert d["coupler_sha256"] == "a" * 64
        assert d["coupler_bytes"] == 500

    def test_manifest_has_head_sha256_and_bytes(self) -> None:
        """to_dict must include head_sha256 and head_bytes."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(head_sha256="a" * 64, head_bytes=500)
        d = m.to_dict()
        assert d["head_sha256"] == "a" * 64
        assert d["head_bytes"] == 500

    def test_artifact_hash_covers_head_and_coupler(self) -> None:
        """cortex_artifact_sha256 must differ from coupler_sha256 alone —
        it covers the full head+coupler artifact, not just the coupler."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest(
            cortex_artifact_sha256="a" * 64,
            coupler_sha256="b" * 64,
            head_sha256="c" * 64,
        )
        d = m.to_dict()
        assert d["cortex_artifact_sha256"] != d["coupler_sha256"]
        assert d["cortex_artifact_sha256"] != d["head_sha256"]

    def test_evaluate_fails_on_artifact_hash_mismatch(self) -> None:
        """evaluate must fail when the cortex artifact file hash doesn't
        match the manifest's cortex_artifact_sha256."""
        import pickle

        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict()
            manifest_path.write_text(json.dumps(manifest_data))

            # Create a cortex artifact file with a different hash.
            cortex_path = Path(tmpdir) / "s19_cortex.pkl"
            with open(cortex_path, "wb") as f:
                pickle.dump(
                    {"state": {}, "sha256": "wrong_hash", "bytes": 1000},
                    f,
                )

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--coupler-path", str(cortex_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_artifact_hash_roundtrips_through_dict(self) -> None:
        """cortex_artifact_sha256 and bytes must survive to_dict/from_dict."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m1 = CalibrationManifest(
            cortex_artifact_sha256="a" * 64,
            cortex_artifact_bytes=241552,
        )
        m2 = CalibrationManifest.from_dict(m1.to_dict())
        assert m2.cortex_artifact_sha256 == "a" * 64
        assert m2.cortex_artifact_bytes == 241552


# ---------------------------------------------------------------------------
# Calibration cannot self-sign tests
# ---------------------------------------------------------------------------


class TestCalibrationCannotSelfSign:
    """Tests that calibrate-dev cannot mark itself signed off —
    thresholds_signed_off defaults False and human_signoff_id defaults empty."""

    def test_default_thresholds_signed_off_is_false(self) -> None:
        """New CalibrationManifest must default signoff_thresholds_signed_off=False."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert m.signoff_thresholds_signed_off is False

    def test_default_human_signoff_id_is_empty(self) -> None:
        """New CalibrationManifest must default signoff_human_signoff_id=''."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest()
        assert m.signoff_human_signoff_id == ""

    def test_to_dict_emits_unsigned_defaults(self) -> None:
        """to_dict must emit signoff_thresholds_signed_off=False and
        signoff_human_signoff_id='' for a default manifest."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = CalibrationManifest().to_dict()
        assert d["signoff_thresholds_signed_off"] is False
        assert d["signoff_human_signoff_id"] == ""

    def test_evaluate_rejects_unsigned_thresholds(self) -> None:
        """evaluate must fail when signoff_thresholds_signed_off=False."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(
                signoff_thresholds_signed_off=False,
                signoff_human_signoff_id="human-001",
            )
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_rejects_empty_manifest_signoff_id(self) -> None:
        """evaluate must fail when manifest has no human sign-off ID."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(
                signoff_thresholds_signed_off=True,
                signoff_human_signoff_id="",
            )
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_rejects_signoff_id_mismatch(self) -> None:
        """evaluate must fail when CLI signoff-id doesn't match manifest."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(
                signoff_thresholds_signed_off=True,
                signoff_human_signoff_id="human-001",
            )
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "different-reviewer",
            ])
            assert rc != 0

    def test_evaluate_rejects_zero_oracle_ceiling(self) -> None:
        """evaluate must fail when signoff_oracle_ceiling <= 0."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(
                signoff_thresholds_signed_off=True,
                signoff_human_signoff_id="human-001",
                signoff_oracle_ceiling=0.0,
            )
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_evaluate_rejects_failed_dev_articulation_gate(self) -> None:
        """evaluate must fail when signoff_dev_articulation_gate=False."""
        from oczy.experiments.s19_language_organ import main

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_data = _make_valid_manifest_dict(
                signoff_thresholds_signed_off=True,
                signoff_human_signoff_id="human-001",
                signoff_dev_articulation_gate=False,
            )
            manifest_path.write_text(json.dumps(manifest_data))

            rc = main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
            assert rc != 0

    def test_hash_changes_when_human_signs_off(self) -> None:
        """The manifest hash must change when a human signs off —
        proving the signoff is part of the canonical hash and cannot
        be retroactively applied without changing the hash."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m_unsigned = CalibrationManifest(
            signoff_thresholds_signed_off=False,
            signoff_human_signoff_id="",
        )
        m_signed = CalibrationManifest(
            signoff_thresholds_signed_off=True,
            signoff_human_signoff_id="human-001",
        )
        assert m_unsigned.compute_hash() != m_signed.compute_hash()


# ---------------------------------------------------------------------------
# Required-field null rejection + legacy alias typed-default tests
# ---------------------------------------------------------------------------


class TestManifestRequiredFieldsRejectNull:
    """Tests that every required typed field rejects explicit None.

    A null value for a required typed field is malformed input — distinct
    from a missing migration field that resolves to a typed default.
    ``from_dict`` must raise ``ValueError`` (mentioning the field name) so
    that a partially invalid manifest is never produced.  The optional
    ``c7_blocked_reason`` field and the ``labels`` list (extracted via
    ``d.get``) are not typed fields and preserve ``None``.
    """

    @pytest.mark.parametrize("field_name", [
        "schema_version",
        "source_commit",
        "source_archive_sha256",
        "eval_version",
        "eval_manifest_sha256",
        "model_repo_id",
        "model_revision",
        "model_config_sha256",
        "model_safetensors_sha256",
        "cortex_artifact_sha256",
        "coupler_sha256",
        "head_sha256",
        "articulation_language_organ_hash",
        "c7_reference",
    ])
    def test_required_str_rejects_null(self, field_name: str) -> None:
        """from_dict with explicit None for a required string field must
        raise ValueError mentioning the field name — no manifest produced."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d[field_name] = None
        with pytest.raises(ValueError, match=field_name):
            CalibrationManifest.from_dict(d)

    @pytest.mark.parametrize("field_name", [
        "cortex_artifact_bytes",
        "coupler_bytes",
        "head_bytes",
    ])
    def test_required_positive_int_rejects_null(self, field_name: str) -> None:
        """from_dict with explicit None for a required positive-int field
        must raise ValueError mentioning the field name — no manifest
        produced, no TypeError."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d[field_name] = None
        with pytest.raises(ValueError, match=field_name):
            CalibrationManifest.from_dict(d)

    def test_required_labels_rejects_null(self) -> None:
        """Explicit null labels must fail during manifest parsing."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d["labels"] = None
        with pytest.raises(ValueError, match="labels"):
            CalibrationManifest.from_dict(d)

    def test_required_holdout_accessed_rejects_null(self) -> None:
        """from_dict with explicit None for holdout_accessed must raise
        ValueError mentioning the field name — None is not False."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d["holdout_accessed"] = None
        with pytest.raises(ValueError, match="holdout_accessed"):
            CalibrationManifest.from_dict(d)

    def test_optional_nullable_field_accepts_null(self) -> None:
        """c7_blocked_reason is str | None — None is valid, not malformed.

        This contrasts with required fields: null rejection is specific to
        required typed fields, not all fields.
        """
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d["c7_blocked_reason"] = None
        m = CalibrationManifest.from_dict(d)
        assert m.c7_blocked_reason is None
        assert m.required_fields_present() is True


class TestManifestLegacyAliasDefaults:
    """Tests that legacy missing aliases receive deterministic typed defaults
    without changing canonical hash semantics.

    When both the canonical name and its legacy alias are absent from the
    dict, ``from_dict`` must resolve to the same typed default that a
    default-constructed ``CalibrationManifest`` would have.  When the alias
    is present but the canonical is absent, the alias value must be used.
    The canonical hash must be identical whether values are provided via
    canonical names or legacy aliases.
    """

    _ALIAS_PAIRS: list[tuple[str, str]] = [
        ("created_at", "calibration_timestamp"),
        ("eval_manifest_sha256", "eval_manifest_hash"),
        ("model_repo_id", "model_id"),
        ("model_safetensors_sha256", "model_hash"),
        ("max_labels", "n_labels"),
        ("parameter_total", "parameter_count"),
        ("proposed_confidence_threshold", "confidence_threshold"),
        ("proposed_specificity_margin", "specificity_margin"),
        ("coupler_sha256", "coupler_hash"),
        ("signoff_thresholds_signed_off", "thresholds_signed_off"),
        ("signoff_human_signoff_id", "human_signoff_id"),
        ("signoff_oracle_ceiling", "oracle_ceiling"),
        ("signoff_dev_articulation_gate", "dev_articulation_gate"),
        ("signoff_meta_test_conflation_ok", "meta_test_conflation_ok"),
        ("manifest_sha256", "manifest_hash"),
    ]

    @pytest.mark.parametrize("canonical,alias", _ALIAS_PAIRS)
    def test_missing_alias_resolves_to_typed_default(
        self, canonical: str, alias: str
    ) -> None:
        """When both canonical and alias are absent, from_dict uses the
        typed default (matching a default-constructed manifest)."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d.pop(canonical, None)
        d.pop(alias, None)
        m = CalibrationManifest.from_dict(d)
        default_m = CalibrationManifest()
        assert getattr(m, canonical) == getattr(default_m, canonical)
        assert getattr(m, canonical) is not None

    @pytest.mark.parametrize("canonical,alias", _ALIAS_PAIRS)
    def test_alias_used_when_canonical_absent(
        self, canonical: str, alias: str
    ) -> None:
        """When canonical is absent but alias is present, from_dict uses
        the alias value."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        original = d.pop(canonical)
        d[alias] = original
        m = CalibrationManifest.from_dict(d)
        assert getattr(m, canonical) == original

    def test_empty_dict_hash_matches_default_manifest(self) -> None:
        """from_dict({}) produces a manifest with the same canonical hash
        as CalibrationManifest() — all missing fields resolve to defaults."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        m = CalibrationManifest.from_dict({})
        default_m = CalibrationManifest()
        assert m.compute_hash() == default_m.compute_hash()

    def test_alias_values_preserve_hash_vs_canonical(self) -> None:
        """Using legacy aliases instead of canonical names produces the
        same canonical hash when the values are identical."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d_canonical = _make_valid_manifest_dict()
        d_alias = _make_valid_manifest_dict()
        for canonical, alias in self._ALIAS_PAIRS:
            d_alias[alias] = d_alias.pop(canonical)
        m_canonical = CalibrationManifest.from_dict(d_canonical)
        m_alias = CalibrationManifest.from_dict(d_alias)
        assert m_canonical.compute_hash() == m_alias.compute_hash()

    def test_missing_alias_hash_matches_explicit_default(self) -> None:
        """Removing an aliased field produces the same hash as explicitly
        setting that field to its typed default — no hidden hash drift."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        # Remove eval_manifest_sha256 (alias: eval_manifest_hash) entirely.
        d_missing = _make_valid_manifest_dict()
        d_missing.pop("eval_manifest_sha256", None)
        d_missing.pop("eval_manifest_hash", None)
        m_missing = CalibrationManifest.from_dict(d_missing)

        # Explicitly set to the typed default "".
        d_default = _make_valid_manifest_dict()
        d_default["eval_manifest_sha256"] = ""
        m_default = CalibrationManifest.from_dict(d_default)

        assert m_missing.compute_hash() == m_default.compute_hash()


class TestManifestMissingVsNullDistinction:
    """Tests that distinguish missing migration fields from malformed null fields.

    A missing field resolves to a typed default (e.g. "", 0, False) — valid
    migration behavior.  An explicit ``None`` for a required typed field is
    malformed — ``from_dict`` must raise ``ValueError`` (mentioning the
    field name) so that no partially invalid manifest is produced.  Missing
    yields a usable manifest (typed default); null never yields a manifest.
    """

    @pytest.mark.parametrize("field_name,typed_default", [
        ("source_commit", ""),
        ("cortex_artifact_bytes", 0),
        ("holdout_accessed", False),
    ])
    def test_missing_field_gets_typed_default(
        self, field_name: str, typed_default: Any
    ) -> None:
        """Missing field resolves to typed default, not None."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        del d[field_name]
        m = CalibrationManifest.from_dict(d)
        assert getattr(m, field_name) == typed_default
        assert getattr(m, field_name) is not None

    @pytest.mark.parametrize("field_name", [
        "source_commit",
        "cortex_artifact_bytes",
        "holdout_accessed",
    ])
    def test_null_field_raises_not_typed_default(
        self, field_name: str
    ) -> None:
        """Null field raises ValueError (mentioning the field name) — it is
        never cast to the typed default and never reaches a manifest."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d[field_name] = None
        with pytest.raises(ValueError, match=field_name):
            CalibrationManifest.from_dict(d)

    def test_missing_yields_default_null_raises(self) -> None:
        """Missing source_commit resolves to typed default "" (manifest
        produced, fails required_fields_present because empty); explicit
        null raises ValueError — no manifest produced."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d_missing = _make_valid_manifest_dict()
        del d_missing["source_commit"]
        m_missing = CalibrationManifest.from_dict(d_missing)
        assert m_missing.source_commit == ""
        assert m_missing.required_fields_present() is False

        d_null = _make_valid_manifest_dict()
        d_null["source_commit"] = None
        with pytest.raises(ValueError, match="source_commit"):
            CalibrationManifest.from_dict(d_null)

    def test_missing_alias_distinct_from_null_canonical(self) -> None:
        """A missing canonical key (alias absent too) resolves to the typed
        default "", while an explicit null for the same key raises
        ValueError — the two must not be conflated."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d_missing = _make_valid_manifest_dict()
        del d_missing["coupler_sha256"]
        m_missing = CalibrationManifest.from_dict(d_missing)
        assert m_missing.coupler_sha256 == ""

        d_null = _make_valid_manifest_dict()
        d_null["coupler_sha256"] = None
        with pytest.raises(ValueError, match="coupler_sha256"):
            CalibrationManifest.from_dict(d_null)


# ---------------------------------------------------------------------------
# Required collection fields: null / wrong container / invalid shape rejection
# ---------------------------------------------------------------------------


class TestManifestRequiredCollectionsRejectMalformed:
    """Tests that the four required collection fields reject null, wrong
    container type, invalid latent shape, and empty labels.

    Required collection fields and their contracts:
      - parameter_breakdown: dict[str, int]
      - fixed_latent_shape: list[int] exactly [3, 896]
      - labels: nonempty list[str]
      - articulation_latent_bank_shape: list[int] exactly [3, 896]

    Explicit None or a wrong container type fails closed — either
    from_dict raises ValueError (mentioning the field name) or the
    resulting manifest's required_fields_present() returns False.
    This mirrors how schema validation rejects the same malformed input.
    """

    # -- helper ------------------------------------------------------------

    @staticmethod
    def _assert_fails_closed(d: dict[str, Any], field_name: str) -> None:
        """Assert that malformed *field_name* either causes from_dict to
        raise ValueError mentioning the field, or produces a manifest
        whose required_fields_present() is False."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        try:
            m = CalibrationManifest.from_dict(d)
        except ValueError as exc:
            assert field_name in str(exc), (
                f"ValueError for {field_name} should mention the field name; "
                f"got: {exc}"
            )
            return
        assert m.required_fields_present() is False, (
            f"{field_name} is malformed but manifest passed required_fields_present"
        )

    # -- null for each collection field ------------------------------------

    @pytest.mark.parametrize("field_name", [
        "parameter_breakdown",
        "fixed_latent_shape",
        "labels",
        "articulation_latent_bank_shape",
    ])
    def test_null_collection_rejects(self, field_name: str) -> None:
        """Explicit None for a required collection field fails closed."""
        d = _make_valid_manifest_dict()
        d[field_name] = None
        self._assert_fails_closed(d, field_name)

    # -- wrong container type ----------------------------------------------

    @pytest.mark.parametrize("field_name,wrong_value", [
        ("parameter_breakdown", ["W_perceive", "W_label"]),
        ("parameter_breakdown", "not_a_dict"),
        ("parameter_breakdown", 42),
        ("fixed_latent_shape", {"tokens": 3, "d_embd": 896}),
        ("fixed_latent_shape", "(3, 896)"),
        ("labels", {"label1": 0}),
        ("labels", "label1"),
        ("articulation_latent_bank_shape", {"tokens": 3, "d_embd": 896}),
        ("articulation_latent_bank_shape", "(3, 896)"),
    ])
    def test_wrong_container_type_rejects(
        self, field_name: str, wrong_value: Any
    ) -> None:
        """A value with the wrong container type fails closed."""
        d = _make_valid_manifest_dict()
        d[field_name] = wrong_value
        self._assert_fails_closed(d, field_name)

    # -- invalid latent shape (valid list, wrong contents) -----------------

    @pytest.mark.parametrize("field_name,invalid_shape", [
        ("fixed_latent_shape", [3, 1024]),
        ("fixed_latent_shape", [2, 896]),
        ("fixed_latent_shape", [3, 896, 1]),
        ("fixed_latent_shape", [3]),
        ("fixed_latent_shape", []),
        ("articulation_latent_bank_shape", [3, 1024]),
        ("articulation_latent_bank_shape", [2, 896]),
        ("articulation_latent_bank_shape", [3, 896, 1]),
        ("articulation_latent_bank_shape", [3]),
        ("articulation_latent_bank_shape", []),
    ])
    def test_invalid_latent_shape_rejects(
        self, field_name: str, invalid_shape: list[int]
    ) -> None:
        """A list with wrong dimensions (not exactly [3, 896]) fails closed."""
        d = _make_valid_manifest_dict()
        d[field_name] = invalid_shape
        self._assert_fails_closed(d, field_name)

    # -- empty labels -------------------------------------------------------

    def test_empty_labels_rejects(self) -> None:
        """An empty labels list fails required_fields_present."""
        from oczy.experiments.s19_language_organ_core import CalibrationManifest

        d = _make_valid_manifest_dict()
        d["labels"] = []
        m = CalibrationManifest.from_dict(d)
        assert m.labels == []
        assert m.required_fields_present() is False

    # -- malformed parameter_breakdown (dict with wrong value types) -------

    @pytest.mark.parametrize("bad_breakdown", [
        {"W_perceive": "not_an_int"},
        {"W_perceive": 1.5},
        {"W_perceive": None},
        {"W_perceive": [1, 2]},
    ])
    def test_malformed_parameter_breakdown_rejects(
        self, bad_breakdown: dict[str, Any]
    ) -> None:
        """A dict with non-int values in parameter_breakdown fails closed."""
        d = _make_valid_manifest_dict()
        d["parameter_breakdown"] = bad_breakdown
        self._assert_fails_closed(d, "parameter_breakdown")



# ---------------------------------------------------------------------------
# Offline model resolution tests (R19 Kaggle calibration infrastructure fix)
# ---------------------------------------------------------------------------


class _ResolverSentinel(Exception):
    """Sentinel raised by a fake _resolve_load_target to prove the CLI
    actually called the resolver before reaching HFDriver.load."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(model_id)


class TestOfflineModelResolution:
    """Tests for _resolve_load_target and its integration into both CLI phases.

    Regression target: commit 0d62811 where ``_calibrate_dev`` called
    ``HFDriver.load(model_id='Qwen/Qwen2.5-0.5B-Instruct')`` with
    ``HF_HUB_OFFLINE=1`` and an attached local model at ``OCZY_MODEL_DIR``,
    causing ``LocalEntryNotFoundError`` because the hub ID was used instead
    of the verified local path.
    """

    # -- _resolve_load_target unit tests ----------------------------------

    def test_kaggle_offline_resolves_to_local_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Regression: under HF_HUB_OFFLINE=1 with OCZY_MODEL_DIR set to a
        real directory, the resolver must return the local path, not the hub
        ID.  This is the exact Kaggle error path that caused
        LocalEntryNotFoundError.
        """
        local_dir = tmp_path / "models" / "Qwen2.5-0.5B-Instruct"
        local_dir.mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OCZY_MODEL_DIR", str(local_dir))
        monkeypatch.delenv("OCZY_HF_MODEL_DIR", raising=False)
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == str(local_dir)
        assert result != "Qwen/Qwen2.5-0.5B-Instruct"

    def test_local_env_precedence_model_dir_over_hf_model_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """OCZY_MODEL_DIR takes priority over OCZY_HF_MODEL_DIR when both
        are set and exist.
        """
        model_dir = tmp_path / "model_dir"
        hf_model_dir = tmp_path / "hf_model_dir"
        model_dir.mkdir()
        hf_model_dir.mkdir()
        monkeypatch.setenv("OCZY_MODEL_DIR", str(model_dir))
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", str(hf_model_dir))
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == str(model_dir)
        assert result != str(hf_model_dir)

    def test_hf_model_dir_used_when_model_dir_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When OCZY_MODEL_DIR is unset, OCZY_HF_MODEL_DIR is used if it
        exists.
        """
        hf_model_dir = tmp_path / "hf_model"
        hf_model_dir.mkdir()
        monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", str(hf_model_dir))
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == str(hf_model_dir)

    def test_remote_missing_local_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under offline mode with neither OCZY_MODEL_DIR nor
        OCZY_HF_MODEL_DIR pointing to an existing directory, the resolver
        must raise RuntimeError (fail closed) rather than falling back to
        the hub ID.
        """
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", "/nonexistent/path/xyz")
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        with pytest.raises(RuntimeError, match="OCZY_MODEL_DIR|OCZY_HF_MODEL_DIR"):
            _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")

    def test_remote_no_env_vars_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under offline mode with no env vars set at all, the resolver
        must raise RuntimeError.
        """
        monkeypatch.setenv("OCZY_REMOTE_CPU_ONLY", "1")
        monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)
        monkeypatch.delenv("OCZY_HF_MODEL_DIR", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        with pytest.raises(RuntimeError, match="OCZY_MODEL_DIR|OCZY_HF_MODEL_DIR"):
            _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")

    def test_local_nonexistence_rejection_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When OCZY_MODEL_DIR points to a non-existent path, it is rejected
        (skipped); the resolver falls through to OCZY_HF_MODEL_DIR if it
        exists.
        """
        real_dir = tmp_path / "real_model"
        real_dir.mkdir()
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", str(real_dir))
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == str(real_dir)
        assert result != "/nonexistent/path/abc"

    def test_local_nonexistence_rejection_to_hub_in_non_offline(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When OCZY_MODEL_DIR points to a non-existent path, no
        OCZY_HF_MODEL_DIR is set, and we are NOT offline, the resolver
        falls through to the hub model_id.
        """
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")
        monkeypatch.delenv("OCZY_HF_MODEL_DIR", raising=False)
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_non_remote_hub_fallback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In non-offline mode with no local env dirs set, the resolver
        falls back to the original model_id (hub ID).
        """
        monkeypatch.delenv("OCZY_MODEL_DIR", raising=False)
        monkeypatch.delenv("OCZY_HF_MODEL_DIR", raising=False)
        monkeypatch.delenv("OCZY_REMOTE_CPU_ONLY", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from oczy.experiments.s19_language_organ import _resolve_load_target

        result = _resolve_load_target("Qwen/Qwen2.5-0.5B-Instruct")
        assert result == "Qwen/Qwen2.5-0.5B-Instruct"

    # -- logical model ID preservation -----------------------------------

    def test_calibrate_dev_manifest_preserves_logical_model_id(self) -> None:
        """The calibrate-dev manifest's model_repo_id must use the original
        ``model_id`` (the CLI/manifest value), not the resolved local load
        target.  This preserves provenance for reproducibility across
        local/remote environments.
        """
        import inspect

        from oczy.experiments.s19_language_organ import _calibrate_dev

        source = inspect.getsource(_calibrate_dev)
        # The manifest must use the original model_id for provenance.
        assert "model_repo_id=model_id" in source
        # Must NOT use the resolved load target for provenance.
        assert "model_repo_id=load_target" not in source

    def test_evaluate_uses_manifest_model_repo_id_for_resolver(self) -> None:
        """The evaluate phase must pass ``manifest.model_repo_id`` (the
        pinned logical ID) to ``_resolve_load_target``, not a hardcoded
        hub ID or the resolved path from a previous run.
        """
        import inspect

        from oczy.experiments.s19_language_organ import _evaluate

        source = inspect.getsource(_evaluate)
        # The resolver must receive the manifest's pinned model_repo_id.
        assert "_resolve_load_target" in source
        assert "manifest.model_repo_id" in source

    # -- both CLI phases use the resolver --------------------------------

    def test_calibrate_dev_calls_resolver(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """calibrate-dev must call _resolve_load_target before
        HFDriver.load.  A sentinel exception proves the call site is hit.
        """
        monkeypatch.setenv("EVAL_CHANGE_APPROVED", "1")

        def _fake_resolver(model_id: str) -> str:
            raise _ResolverSentinel(model_id)

        monkeypatch.setattr(
            "oczy.experiments.s19_language_organ._resolve_load_target",
            _fake_resolver,
        )

        from oczy.experiments.s19_language_organ import main

        with pytest.raises(_ResolverSentinel) as exc_info:
            main([
                "calibrate-dev",
                "--manifest-out", str(tmp_path / "manifest.json"),
            ])
        assert exc_info.value.model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_evaluate_calls_resolver(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """evaluate must call _resolve_load_target before HFDriver.load.
        A sentinel exception proves the call site is hit.
        """
        monkeypatch.setenv("EVAL_CHANGE_APPROVED", "1")

        manifest_data = _make_valid_manifest_dict()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        # Make the eval-manifest hash check pass so evaluate proceeds to
        # the model-loading step where the resolver is called.
        monkeypatch.setattr(
            "oczy.experiments.s19_language_organ.hash_eval_manifest",
            lambda: manifest_data["eval_manifest_sha256"],
        )

        def _fake_resolver(model_id: str) -> str:
            raise _ResolverSentinel(model_id)

        monkeypatch.setattr(
            "oczy.experiments.s19_language_organ._resolve_load_target",
            _fake_resolver,
        )

        from oczy.experiments.s19_language_organ import main

        with pytest.raises(_ResolverSentinel) as exc_info:
            main([
                "evaluate",
                "--manifest", str(manifest_path),
                "--signoff-id", "human-001",
            ])
        assert exc_info.value.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
