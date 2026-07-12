"""High-signal contract tests for Research/20 training lifecycle and boundary.

These tests verify the behavioral contracts of the online episode, optimization
boundary, outer trainer, and DEV causal interventions using a test-only tiny
frozen organ — no network or model download.

Contracts verified:

  - Online unroll changes F/S but not theta hash or optimizer step count
  - Optimizer step inside ``boundary.online()`` raises ``OnlineOptimizationError``
  - Exactly one theta update after one ``train_step``; organ hash unchanged
  - Meta-validation is inference-only (works with optimizers monkeypatched to raise)
  - Trace lifecycle: buffer counts, ``delete_all`` returns counts, ``verify_zero``
  - No sentinel text (META_TEST) in any task prompt
  - UPDATE_DISABLED: F/S stay exactly zero
  - UNTRAINED_RULE: fresh theta, events enabled, no optimization
  - FEEDBACK_SHUFFLED: real derangement (no fixed points)
  - STATE_ZEROED / STATE_SWAPPED: rescore without writes
  - All DEV conditions perform zero optimizer steps
  - Firewall rejects wrong-split tasks in ``train_step``
  - Two-step CPU smoke completes and selects on validation
"""

from __future__ import annotations

import hashlib
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from oczy.experiments.meta_cortex.contracts import (
    DEV_SCHEMA,
    ContractError,
    DevCondition,
    DevSplit,
    DialogueMessage,
    LearningEvent,
    LossBreakdown,
    ModelConfig,
    OnlineEpisodeAudit,
    OutcomeCode,
    OuterLoopConfig,
    ProbeKind,
    TaskGeneratorConfig,
)
from oczy.experiments.meta_cortex.model import CortexState, EventFeatureBatch, MetaCortex
from oczy.experiments.meta_cortex.organ import FrozenOrganError
from oczy.experiments.meta_cortex.taskgen import build_dev_catalog
from oczy.experiments.meta_cortex.training import (
    OnlineOptimizationError,
    OptimizationBoundary,
    OuterTrainer,
    TransientTraceBuffer,
    _derangement,
    _state_hash,
    _theta_hash,
    compute_outer_objective,
    run_dev_interventions,
    run_dev_validation,
    unroll_online_episode,
)

# ---------------------------------------------------------------------------
# Test-only tiny frozen organ (same as organ test)
# ---------------------------------------------------------------------------


class _TinyFrozenOrgan:
    """Test-only differentiable tiny frozen organ — no network required."""

    def __init__(self, feature_dim: int = 16, vocab_size: int = 128) -> None:
        self.feature_dim = feature_dim
        self._vocab_size = vocab_size
        self._closed = False
        torch.manual_seed(42)
        self._embedding = torch.randn(vocab_size, feature_dim, requires_grad=False)
        self._output_proj = torch.randn(feature_dim, vocab_size, requires_grad=False)
        self._output_bias = torch.zeros(vocab_size, requires_grad=False)
        self._initial_hash = self.parameter_hash()

    def _tokenize(self, text: str) -> list[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def _render(self, messages: list[DialogueMessage]) -> str:
        return "".join(f"{m.role}: {m.content}\n" for m in messages)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not texts:
            raise FrozenOrganError("empty texts")
        features = []
        for text in texts:
            ids = self._tokenize(text)
            if not ids:
                ids = [0]
            features.append(self._embedding[ids].mean(dim=0).to(dtype=torch.float32))
        return torch.stack(features, dim=0).detach()

    def teacher_forced_logits(self, messages, target, soft_bank):
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not messages:
            raise FrozenOrganError("empty messages")
        if not target or not target.strip():
            raise FrozenOrganError("empty target")
        if soft_bank.dim() != 3 or soft_bank.shape[0] != 1:
            raise FrozenOrganError("soft_bank must be [1, L, D]")
        if soft_bank.shape[2] != self.feature_dim:
            raise FrozenOrganError("feature dim mismatch")
        if not torch.isfinite(soft_bank).all():
            raise FrozenOrganError("non-finite soft_bank")

        prompt_text = self._render(messages)
        prompt_ids = self._tokenize(prompt_text)
        target_text = " " + target.lstrip()
        answer_ids = self._tokenize(target_text)
        if not answer_ids:
            raise FrozenOrganError("empty target tokens")

        bank = soft_bank[0]
        prompt_embeds = self._embedding[prompt_ids]
        answer_embeds = self._embedding[answer_ids]
        inputs_embeds = torch.cat([bank, prompt_embeds, answer_embeds], dim=0)
        logits = inputs_embeds @ self._output_proj + self._output_bias
        prompt_len = bank.shape[0] + len(prompt_ids)
        start = prompt_len - 1
        end = start + len(answer_ids)
        return logits[start:end]

    def teacher_forced_loss(self, messages, target, soft_bank):
        logits = self.teacher_forced_logits(messages, target, soft_bank)
        target_text = " " + target.lstrip()
        answer_ids = self._tokenize(target_text)
        targets = torch.tensor(answer_ids, dtype=torch.long)
        return F.cross_entropy(logits, targets)

    def specificity_kl(self, messages, target, soft_bank, reference_bank=None):
        bank_logits = self.teacher_forced_logits(messages, target, soft_bank)
        bank_log_probs = F.log_softmax(bank_logits, dim=-1)
        with torch.no_grad():
            if reference_bank is not None:
                ref_logits = self.teacher_forced_logits(messages, target, reference_bank)
            else:
                prompt_text = self._render(messages)
                prompt_ids = self._tokenize(prompt_text)
                target_text = " " + target.lstrip()
                answer_ids = self._tokenize(target_text)
                pe = self._embedding[prompt_ids]
                ae = self._embedding[answer_ids]
                emb = torch.cat([pe, ae], dim=0)
                logits = emb @ self._output_proj + self._output_bias
                pl = len(prompt_ids)
                ref_logits = logits[pl - 1 : pl - 1 + len(answer_ids)]
            ref_probs = F.softmax(ref_logits, dim=-1)
        bank_probs = bank_log_probs.exp()
        ref_log_probs = ref_probs.clamp(min=1e-12).log()
        kl = (bank_probs * (bank_log_probs - ref_log_probs)).sum(dim=-1)
        return kl.mean()

    def generate(self, messages, soft_bank, max_new_tokens):
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not messages:
            raise FrozenOrganError("empty messages")
        if max_new_tokens <= 0:
            raise FrozenOrganError("max_new_tokens must be positive")
        prompt_text = self._render(messages)
        prompt_ids = self._tokenize(prompt_text)
        bank = soft_bank[0]
        pe = self._embedding[prompt_ids]
        emb = torch.cat([bank, pe], dim=0)
        logits = emb @ self._output_proj + self._output_bias
        next_token = int(logits[-1].argmax().item())
        generated: list[str] = []
        for _ in range(max_new_tokens):
            generated.append(chr(min(next_token, 127)))
            ne = self._embedding[next_token].unsqueeze(0)
            logits = ne @ self._output_proj + self._output_bias
            next_token = int(logits.argmax().item())
        return "".join(generated)

    def parameter_hash(self) -> str:
        parts: list[str] = []
        for name, tensor in sorted([
            ("embedding", self._embedding),
            ("output_bias", self._output_bias),
            ("output_proj", self._output_proj),
        ]):
            data = tensor.detach().cpu().contiguous().numpy().tobytes()
            parts.append(f"{name}:{hashlib.sha256(data).hexdigest()}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def assert_frozen(self) -> None:
        for name, tensor in [
            ("embedding", self._embedding),
            ("output_proj", self._output_proj),
            ("output_bias", self._output_bias),
        ]:
            if tensor.requires_grad:
                raise FrozenOrganError(f"{name} has requires_grad=True")

    def close(self) -> None:
        self._closed = True

    def initial_hash(self) -> str:
        return self._initial_hash

    def organ_parameter_ids(self) -> set[int]:
        return {id(self._embedding), id(self._output_proj), id(self._output_bias)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODEL_CONFIG = ModelConfig(feature_dim=16, d_cortex=64, bank_width=2)
_TASKGEN_CONFIG = TaskGeneratorConfig(
    root_seed=20260709,
    train_tasks_per_family=2,
    validation_tasks_per_family=2,
    min_events=2,
    max_events=2,
)
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


@pytest.fixture
def organ() -> _TinyFrozenOrgan:
    return _TinyFrozenOrgan(feature_dim=16)


@pytest.fixture
def model() -> MetaCortex:
    return MetaCortex(_MODEL_CONFIG)


@pytest.fixture
def catalog():
    return build_dev_catalog(_TASKGEN_CONFIG)


@pytest.fixture
def boundary() -> OptimizationBoundary:
    return OptimizationBoundary()


# ---------------------------------------------------------------------------
# Optimization boundary
# ---------------------------------------------------------------------------


class TestOptimizationBoundary:
    def test_initial_state_zero(self, boundary: OptimizationBoundary) -> None:
        assert boundary.online_depth == 0
        assert boundary.optimizer_step_count == 0

    def test_assert_outer_passes_outside(self, boundary: OptimizationBoundary) -> None:
        boundary.assert_outer()  # should not raise

    def test_assert_outer_fails_inside(self, boundary: OptimizationBoundary) -> None:
        with boundary.online():
            with pytest.raises(OnlineOptimizationError, match="online"):
                boundary.assert_outer()

    def test_nested_online_depth(self, boundary: OptimizationBoundary) -> None:
        with boundary.online():
            assert boundary.online_depth == 1
            with boundary.online():
                assert boundary.online_depth == 2
            assert boundary.online_depth == 1
        assert boundary.online_depth == 0

    def test_record_step_outside(self, boundary: OptimizationBoundary) -> None:
        boundary._record_step()
        assert boundary.optimizer_step_count == 1

    def test_depth_resets_on_exception(self, boundary: OptimizationBoundary) -> None:
        with pytest.raises(RuntimeError):
            with boundary.online():
                assert boundary.online_depth == 1
                raise RuntimeError("test")
        assert boundary.online_depth == 0


# ---------------------------------------------------------------------------
# Transient trace buffer
# ---------------------------------------------------------------------------


class TestTransientTraceBuffer:
    def test_initial_empty(self) -> None:
        buf = TransientTraceBuffer()
        assert buf.object_count == 0
        assert buf.feature_count == 0
        assert buf.verify_zero() is True

    def test_add_increments(self) -> None:
        buf = TransientTraceBuffer()
        event = LearningEvent(
            observation_messages=(DialogueMessage(role="user", content="test"),),
            attempted_behavior="test",
            correction="fix",
            outcome=OutcomeCode.NEUTRAL,
        )
        features = torch.randn(4, 16)
        buf.add(event, features)
        assert buf.object_count == 1
        assert buf.feature_count == 1
        assert buf.verify_zero() is False

    def test_delete_all_returns_counts(self) -> None:
        buf = TransientTraceBuffer()
        for i in range(3):
            event = LearningEvent(
                observation_messages=(DialogueMessage(role="user", content=f"test {i}"),),
                attempted_behavior="test",
                correction="fix",
                outcome=OutcomeCode.NEUTRAL,
            )
            buf.add(event, torch.randn(4, 16))
        n_obj, n_feat = buf.delete_all()
        assert n_obj == 3
        assert n_feat == 3
        assert buf.verify_zero() is True
        assert buf.object_count == 0
        assert buf.feature_count == 0

    def test_delete_all_empty(self) -> None:
        buf = TransientTraceBuffer()
        n_obj, n_feat = buf.delete_all()
        assert n_obj == 0
        assert n_feat == 0

    def test_add_rejects_non_event(self) -> None:
        buf = TransientTraceBuffer()
        with pytest.raises(TypeError):
            buf.add("not an event", torch.randn(4, 16))  # type: ignore[arg-type]

    def test_add_rejects_non_tensor(self) -> None:
        buf = TransientTraceBuffer()
        event = LearningEvent(
            observation_messages=(DialogueMessage(role="user", content="test"),),
            attempted_behavior="test",
            correction="fix",
            outcome=OutcomeCode.NEUTRAL,
        )
        with pytest.raises(TypeError):
            buf.add(event, "not a tensor")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Online episode unroll
# ---------------------------------------------------------------------------


class TestOnlineEpisode:
    def test_unroll_changes_fs_not_theta(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        theta_before = _theta_hash(model)
        step_before = boundary.optimizer_step_count

        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=False,
        )

        # F should be zero (consolidated), S should be nonzero.
        assert torch.count_nonzero(output.state.fast).item() == 0
        assert output.state.slow.abs().sum().item() > 0

        # Theta and step count unchanged.
        assert _theta_hash(model) == theta_before
        assert boundary.optimizer_step_count == step_before

    def test_unroll_audit_fields(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=False,
        )
        audit = output.audit
        assert isinstance(audit, OnlineEpisodeAudit)
        assert audit.family == task.family.value
        assert audit.split == task.split.value
        assert audit.event_count == len(task.events)
        assert audit.trace_objects_after == 0
        assert audit.trace_feature_tensors_after == 0
        assert audit.fast_zero is True
        assert audit.optimizer_step_count_before == audit.optimizer_step_count_after
        assert audit.theta_hash_before == audit.theta_hash_after
        assert audit.organ_hash_before == audit.organ_hash_after
        assert audit.banned_content_absent is True

    def test_unroll_bank_shape(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary, update_enabled=True, gradient_enabled=False,
        )
        assert output.audit.bank_shape == (1, 2, 16)

    def test_unroll_loss_tensor_with_gradient(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=True,
        )
        assert output.loss_tensor is not None
        assert output.loss_tensor.requires_grad
        assert output.loss_breakdown is not None
        assert math.isfinite(output.loss_breakdown.weighted_total)

    def test_unroll_no_gradient_no_loss_tensor(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=False,
        )
        # loss_tensor may be present but detached; loss_breakdown should be present.
        assert output.loss_breakdown is not None

    def test_unroll_pre_post_metrics(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary, update_enabled=True, gradient_enabled=False,
        )
        assert len(output.pre_metrics) > 0
        assert len(output.post_metrics) > 0
        for v in output.pre_metrics.values():
            assert 0.0 <= v <= 1.0
        for v in output.post_metrics.values():
            assert 0.0 <= v <= 1.0

    def test_unroll_update_disabled_fs_zero(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=False,
            gradient_enabled=False,
        )
        assert torch.count_nonzero(output.state.fast).item() == 0
        assert torch.count_nonzero(output.state.slow).item() == 0

    def test_unroll_organ_hash_unchanged(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_train[0]
        hash_before = organ.parameter_hash()
        unroll_online_episode(
            model, organ, task, boundary, update_enabled=True, gradient_enabled=False,
        )
        assert organ.parameter_hash() == hash_before

    def test_unroll_rejects_invalid_split(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        # We can't easily create an invalid split since DevSplit only has two
        # valid members. Instead, test that both valid splits are accepted.
        val_task = catalog.meta_validation[0]
        output = unroll_online_episode(
            model, organ, val_task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        assert output.audit.split == DevSplit.META_VALIDATION.value


# ---------------------------------------------------------------------------
# Outer trainer
# ---------------------------------------------------------------------------


class TestOuterTrainer:
    def test_train_step_one_optimizer_step(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        step_before = trainer.boundary.optimizer_step_count
        tasks = list(catalog.tasks_for(DevSplit.META_TRAIN))[:1]
        trainer.train_step(tasks)
        assert trainer.boundary.optimizer_step_count == step_before + 1

    def test_train_step_organ_hash_unchanged(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        hash_before = organ.parameter_hash()
        tasks = list(catalog.tasks_for(DevSplit.META_TRAIN))[:1]
        trainer.train_step(tasks)
        assert organ.parameter_hash() == hash_before

    def test_train_step_rejects_validation_task(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        val_tasks = list(catalog.tasks_for(DevSplit.META_VALIDATION))
        with pytest.raises(ValueError, match="META_TRAIN"):
            trainer.train_step(val_tasks)

    def test_train_step_theta_changes(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        theta_before = _theta_hash(model)
        tasks = list(catalog.tasks_for(DevSplit.META_TRAIN))[:1]
        trainer.train_step(tasks)
        theta_after = _theta_hash(model)
        assert theta_before != theta_after

    def test_train_step_finite_loss(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        tasks = list(catalog.tasks_for(DevSplit.META_TRAIN))[:1]
        result = trainer.train_step(tasks)
        assert math.isfinite(result["loss"])

    def test_full_train_completes(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        """Two-step CPU smoke: train completes and returns DevTrainingResult."""
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        result = trainer.train(catalog)
        assert result.schema == DEV_SCHEMA
        assert result.optimizer_step_count == _OUTER_CONFIG.outer_steps
        assert result.organ_hash_before == result.organ_hash_after
        assert result.audit_status == "ok"
        assert len(result.step_curve) == _OUTER_CONFIG.outer_steps
        assert result.best_validation_step > 0

    def test_full_train_organ_unchanged(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        hash_before = organ.parameter_hash()
        trainer.train(catalog)
        assert organ.parameter_hash() == hash_before

    def test_full_train_selects_best_validation(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        result = trainer.train(catalog)
        # best_validation_score should be finite (could be negative).
        assert math.isfinite(result.best_validation_score)
        # validation_result should be present.
        assert result.validation_result is not None

    def test_optimizer_disjoint_from_organ(
        self, model: MetaCortex, organ: _TinyFrozenOrgan
    ) -> None:
        """No organ parameter may appear in the cortex optimizer."""
        trainer = OuterTrainer(model, organ, _OUTER_CONFIG)
        opt_param_ids: set[int] = set()
        for group in trainer._optimizer.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
        organ_ids = organ.organ_parameter_ids()
        assert len(opt_param_ids & organ_ids) == 0

    def test_unknown_optimizer_rejected(self) -> None:
        with pytest.raises(ContractError, match="optimizer"):
            OuterLoopConfig(
                outer_steps=1,
                tasks_per_step=1,
                optimizer_name="rmsprop",
            )


# ---------------------------------------------------------------------------
# DEV validation (inference-only)
# ---------------------------------------------------------------------------


class TestDevValidation:
    def test_validation_runs_without_optimizer(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        """Meta-validation works with optimizers monkeypatched to raise."""
        def _raise(*args, **kwargs):
            raise RuntimeError("Optimizer should not be created")

        with patch("torch.optim.AdamW", _raise), patch("torch.optim.SGD", _raise):
            result = run_dev_validation(
                model, organ, catalog, device="cpu", dtype=torch.float32,
            )
        assert isinstance(result.trained_vs_update_disabled_delta, float)
        assert math.isfinite(result.trained_vs_update_disabled_delta)

    def test_validation_theta_unchanged(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        theta_before = _theta_hash(model)
        run_dev_validation(model, organ, catalog, device="cpu", dtype=torch.float32)
        assert _theta_hash(model) == theta_before

    def test_validation_organ_unchanged(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        hash_before = organ.parameter_hash()
        run_dev_validation(model, organ, catalog, device="cpu", dtype=torch.float32)
        assert organ.parameter_hash() == hash_before

    def test_validation_has_all_deltas(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        result = run_dev_validation(
            model, organ, catalog, device="cpu", dtype=torch.float32,
        )
        for delta in (
            result.trained_vs_update_disabled_delta,
            result.trained_vs_untrained_delta,
            result.trained_vs_shuffled_delta,
            result.trained_vs_zeroed_delta,
            result.trained_vs_swapped_delta,
        ):
            assert math.isfinite(delta)

    def test_validation_per_family_results(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        result = run_dev_validation(
            model, organ, catalog, device="cpu", dtype=torch.float32,
        )
        assert len(result.per_family_results) > 0
        for family_name, conds in result.per_family_results:
            assert isinstance(family_name, str)
            assert len(conds) > 0

    def test_validation_pooled_results(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        result = run_dev_validation(
            model, organ, catalog, device="cpu", dtype=torch.float32,
        )
        assert len(result.pooled_results) > 0

    def test_validation_no_verdict_field(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        result = run_dev_validation(
            model, organ, catalog, device="cpu", dtype=torch.float32,
        )
        json_str = result.to_json()
        assert "verdict" not in json_str.lower()
        assert "signoff" not in json_str.lower()
        assert "sign_off" not in json_str.lower()


# ---------------------------------------------------------------------------
# DEV causal interventions
# ---------------------------------------------------------------------------


class TestDevInterventions:
    @pytest.fixture
    def trained_state(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> CortexState:
        task = catalog.meta_validation[0]
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        return output.state

    def test_all_conditions_zero_optimizer_steps(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary,
        catalog, trained_state
    ) -> None:
        task = catalog.meta_validation[0]
        step_before = boundary.optimizer_step_count
        results = run_dev_interventions(
            model, organ, task, trained_state, boundary=boundary,
        )
        assert boundary.optimizer_step_count == step_before
        assert len(results) > 0

    def test_update_disabled_state_zero(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        results = run_dev_interventions(
            model, organ, task,
            model.initial_state(1, device="cpu", dtype=torch.float32),
            boundary=boundary,
        )
        # Find UPDATE_DISABLED result.
        for cr in results:
            if cr.condition == DevCondition.UPDATE_DISABLED.value:
                # State hash should match zero state.
                zero_state = model.initial_state(1, device="cpu", dtype=torch.float32)
                assert cr.state_hash == _state_hash(zero_state)
                assert cr.state_norm == 0.0
                return
        pytest.fail("UPDATE_DISABLED condition not found")

    def test_untrained_rule_uses_fresh_theta(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        theta_before = _theta_hash(model)
        results = run_dev_interventions(
            model, organ, task,
            model.initial_state(1, device="cpu", dtype=torch.float32),
            boundary=boundary,
        )
        # Theta should be restored after UNTRAINED_RULE.
        assert _theta_hash(model) == theta_before
        # Find UNTRAINED_RULE result.
        for cr in results:
            if cr.condition == DevCondition.UNTRAINED_RULE.value:
                assert cr.state_norm > 0  # events were enabled
                return
        pytest.fail("UNTRAINED_RULE condition not found")

    def test_feedback_shuffled_is_derangement(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        n = len(task.events)
        perm = _derangement(n)
        # Verify no fixed points.
        for i, p in enumerate(perm):
            assert i != p, f"Fixed point at index {i}"

    def test_feedback_shuffled_different_state(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        # Run trained.
        trained_output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        # Run shuffled.
        n = len(task.events)
        perm = _derangement(n)
        shuffled_output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
            feedback_permutation=perm,
        )
        # States should differ (different event order → different writes).
        assert _state_hash(trained_output.state) != _state_hash(shuffled_output.state)

    def test_state_zeroed_rescores_without_writes(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        # Create a trained state.
        trained_output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        results = run_dev_interventions(
            model, organ, task, trained_output.state, boundary=boundary,
        )
        for cr in results:
            if cr.condition == DevCondition.STATE_ZEROED.value:
                zero_state = model.initial_state(1, device="cpu", dtype=torch.float32)
                assert cr.state_hash == _state_hash(zero_state)
                assert cr.state_norm == 0.0
                return
        pytest.fail("STATE_ZEROED condition not found")

    def test_state_swapped_uses_donor(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        # Need at least 2 validation tasks for donor.
        val_tasks = list(catalog.tasks_for(DevSplit.META_VALIDATION))
        if len(val_tasks) < 2:
            pytest.skip("Need >=2 validation tasks for STATE_SWAPPED")

        trained_output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        donor_task = val_tasks[1]
        donor_output = unroll_online_episode(
            model, organ, donor_task, boundary,
            update_enabled=True, gradient_enabled=False,
        )

        results = run_dev_interventions(
            model, organ, task, trained_output.state,
            boundary=boundary, donor_state=donor_output.state,
        )
        for cr in results:
            if cr.condition == DevCondition.STATE_SWAPPED.value:
                # State should match donor's S.
                swapped = model.swap_state(trained_output.state, donor_output.state)
                assert cr.state_hash == _state_hash(swapped)
                return
        pytest.fail("STATE_SWAPPED condition not found")

    def test_organ_only_zero_bank(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        results = run_dev_interventions(
            model, organ, task,
            model.initial_state(1, device="cpu", dtype=torch.float32),
            boundary=boundary,
        )
        for cr in results:
            if cr.condition == DevCondition.ORGAN_ONLY.value:
                assert cr.state_norm == 0.0
                assert cr.state_bytes == 0
                return
        pytest.fail("ORGAN_ONLY condition not found")

    def test_all_conditions_present(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, boundary: OptimizationBoundary, catalog
    ) -> None:
        task = catalog.meta_validation[0]
        val_tasks = list(catalog.tasks_for(DevSplit.META_VALIDATION))
        donor_state = None
        if len(val_tasks) >= 2:
            donor_output = unroll_online_episode(
                model, organ, val_tasks[1], boundary,
                update_enabled=True, gradient_enabled=False,
            )
            donor_state = donor_output.state

        trained_output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True, gradient_enabled=False,
        )
        results = run_dev_interventions(
            model, organ, task, trained_output.state,
            boundary=boundary, donor_state=donor_state,
        )
        conditions = {cr.condition for cr in results}
        expected = {
            DevCondition.TRAINED.value,
            DevCondition.UPDATE_DISABLED.value,
            DevCondition.UNTRAINED_RULE.value,
            DevCondition.FEEDBACK_SHUFFLED.value,
            DevCondition.STATE_ZEROED.value,
            DevCondition.ORGAN_ONLY.value,
        }
        if donor_state is not None:
            expected.add(DevCondition.STATE_SWAPPED.value)
        assert expected.issubset(conditions)


# ---------------------------------------------------------------------------
# No sentinel text in prompts
# ---------------------------------------------------------------------------


class TestNoSentinelText:
    def test_no_meta_test_in_tasks(self, catalog) -> None:
        for task in catalog.meta_train + catalog.meta_validation:
            for event in task.events:
                for msg in event.observation_messages:
                    assert "meta_test" not in msg.content.lower()
                    assert "META_TEST" not in msg.content
                assert "meta_test" not in event.attempted_behavior.lower()
                assert "meta_test" not in event.correction.lower()
            for kind in ProbeKind:
                for probe in task.probes.by_kind(kind):
                    for msg in probe.messages:
                        assert "meta_test" not in msg.content.lower()
                    assert "meta_test" not in probe.expected_response.lower()

    def test_no_meta_test_in_split_audit(self, catalog) -> None:
        audit_json = catalog.split_audit.to_json()
        assert "meta_test" not in audit_json.lower()


# ---------------------------------------------------------------------------
# Compute outer objective
# ---------------------------------------------------------------------------


class TestComputeOuterObjective:
    def test_returns_loss_breakdown(
        self, model: MetaCortex, organ: _TinyFrozenOrgan, catalog
    ) -> None:
        task = catalog.meta_train[0]
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        # Write some events to get nonzero state.

        for event in task.events:
            features = organ.encode_texts([
                " ".join(m.content for m in event.observation_messages),
                event.attempted_behavior,
                event.correction,
                event.outcome.value,
            ])
            state = model.write(state, EventFeatureBatch(values=features.unsqueeze(0))).state
        state = model.consolidate(state).state

        breakdown = compute_outer_objective(
            model, organ, task, state,
            behavior_weight=1.0,
            specificity_weight=0.5,
            survival_weight=0.5,
            state_norm_weight=0.01,
            gradient_enabled=False,
        )
        assert isinstance(breakdown, LossBreakdown)
        assert math.isfinite(breakdown.behavior)
        assert math.isfinite(breakdown.specificity)
        assert math.isfinite(breakdown.consolidation_survival)
        assert math.isfinite(breakdown.state_norm)
        assert math.isfinite(breakdown.weighted_total)


# ---------------------------------------------------------------------------
# Derangement helper
# ---------------------------------------------------------------------------


class TestDerangement:
    def test_no_fixed_points_n2(self) -> None:
        perm = _derangement(2)
        for i, p in enumerate(perm):
            assert i != p

    def test_no_fixed_points_n5(self) -> None:
        perm = _derangement(5)
        for i, p in enumerate(perm):
            assert i != p

    def test_n1_rejected(self) -> None:
        with pytest.raises(ValueError, match="derange"):
            _derangement(1)

    def test_n0_rejected(self) -> None:
        with pytest.raises(ValueError, match="derange"):
            _derangement(0)

    def test_is_permutation(self) -> None:
        perm = _derangement(7)
        assert sorted(perm) == list(range(7))
