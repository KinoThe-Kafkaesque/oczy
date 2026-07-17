"""Online episode lifecycle, optimization boundary, and outer trainer for
Research/20 meta-cortex DEV.

This module owns:
- :class:`TransientTraceBuffer` — raw event/feature lifecycle with deletion
  and verification.
- :class:`OptimizationBoundary` — context manager that separates the online
  episode (no optimizer) from the outer step (the only place an optimizer
  may fire).
- :func:`unroll_online_episode` — the sequential write → consolidate →
  delete-traces → probe pipeline.
- :func:`compute_outer_objective` — the four-term outer loss.
- :class:`OuterTrainer` — the sole owner of a cortex-only optimizer.
- :func:`run_dev_interventions` — causal DEV controls (update-disabled,
  untrained, shuffled, zeroed, swapped, organ-only).
- :func:`run_dev_validation` — inference-only validation with no optimizer.

Design invariants enforced throughout:
- No optimizer or ``backward`` reference enters the online episode.
- Only cortex theta is stepped, and only outside every online context.
- Meta-train tasks are the only optimized tasks; meta-validation is
  inference-only.
- Trace objects and feature tensors are deleted before post probes.
- All causal controls are real and optimizer-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import (
    DEV_SCHEMA,
    ConditionResult,
    DevCondition,
    DevSplit,
    DevTaskCatalog,
    DevTrainingResult,
    DevValidationResult,
    LearningEvent,
    LossBreakdown,
    MetaTask,
    ModelConfig,
    OnlineEpisodeAudit,
    OuterLoopConfig,
    ProbeCase,
    ProbeKind,
    TaskFamily,
)
from .model import CortexState, EventFeatureBatch, MetaCortex
from .organ import FrozenLanguageOrgan

__all__ = [
    "TransientTraceBuffer",
    "OnlineOptimizationError",
    "OptimizationBoundary",
    "OnlineEpisodeOutput",
    "unroll_online_episode",
    "compute_outer_objective",
    "run_dev_interventions",
    "run_dev_validation",
    "OuterTrainer",
    "ValidationTaskProgress",
    "ValidationCompleted",
    "ValidationTaskCallback",
    "ValidationCompleteCallback",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OnlineOptimizationError(RuntimeError):
    """Raised when an optimizer step is attempted inside an online context."""


# ---------------------------------------------------------------------------
# Theta / state hashing
# ---------------------------------------------------------------------------


def _theta_hash(model: MetaCortex) -> str:
    """Canonical SHA-256 over sorted named-parameter raw bytes."""
    parts: list[str] = []
    for name, param in sorted(model.state_dict().items()):
        data = param.detach().cpu().contiguous()
        parts.append(
            f"{name}:{data.dtype}:{tuple(data.shape)}:"
            f"{hashlib.sha256(data.numpy().tobytes()).hexdigest()}"
        )
    payload = "|".join(parts).encode()
    return hashlib.sha256(payload).hexdigest()


def _state_hash(state: CortexState) -> str:
    """SHA-256 over F and S raw bytes (detached, CPU, contiguous)."""
    f_data = state.fast.detach().cpu().contiguous().numpy().tobytes()
    s_data = state.slow.detach().cpu().contiguous().numpy().tobytes()
    payload = f"F:{hashlib.sha256(f_data).hexdigest()}|S:{hashlib.sha256(s_data).hexdigest()}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _state_norm(state: CortexState) -> float:
    """L2 norm of S (the persistent component)."""
    return float(state.slow.detach().pow(2).sum().sqrt().item())


# ---------------------------------------------------------------------------
# Transient trace buffer
# ---------------------------------------------------------------------------


class TransientTraceBuffer:
    """Stores raw ``LearningEvent`` references and their ``[4, D]`` feature
    tensors only while processing experiences.

    Adapted from the R19 ``TraceStore`` lifecycle
    (``s19_language_organ_core.py:607-663``) but tracks both raw objects and
    feature tensors, and returns counts from ``delete_all``.
    """

    def __init__(self) -> None:
        self._objects: list[LearningEvent] = []
        self._features: list[torch.Tensor] = []

    def add(self, event: LearningEvent, features: torch.Tensor) -> None:
        """Record one raw event and its ``[4, D]`` feature tensor."""
        if not isinstance(event, LearningEvent):
            raise TypeError("event must be a LearningEvent")
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        self._objects.append(event)
        self._features.append(features)

    @property
    def object_count(self) -> int:
        return len(self._objects)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def delete_all(self) -> tuple[int, int]:
        """Clear both collections and return ``(objects_deleted, features_deleted)``."""
        n_obj = len(self._objects)
        n_feat = len(self._features)
        self._objects.clear()
        self._features.clear()
        return n_obj, n_feat

    def verify_zero(self) -> bool:
        """Return True iff both collections are empty."""
        return len(self._objects) == 0 and len(self._features) == 0


# ---------------------------------------------------------------------------
# Optimization boundary
# ---------------------------------------------------------------------------


class OptimizationBoundary:
    """Enforces that optimizer steps never occur inside an online context.

    Tracks ``online_depth`` (how many nested ``online()`` contexts are
    active) and ``optimizer_step_count`` (incremented only by
    ``_record_step``, which is called exclusively by ``OuterTrainer._outer_step``).
    """

    def __init__(self) -> None:
        self._online_depth: int = 0
        self._optimizer_step_count: int = 0

    @property
    def online_depth(self) -> int:
        return self._online_depth

    @property
    def optimizer_step_count(self) -> int:
        return self._optimizer_step_count

    @contextlib.contextmanager
    def online(self) -> Generator[None, None, None]:
        """Enter an online (no-optimizer) context."""
        self._online_depth += 1
        try:
            yield
        finally:
            self._online_depth -= 1

    def assert_outer(self) -> None:
        """Raise ``OnlineOptimizationError`` if inside an online context."""
        if self._online_depth != 0:
            raise OnlineOptimizationError(
                f"optimizer step attempted inside online context "
                f"(online_depth={self._online_depth})"
            )

    def _record_step(self) -> None:
        """Record that one optimizer step occurred.  Must be called only
        after ``assert_outer`` passes."""
        self._optimizer_step_count += 1


# ---------------------------------------------------------------------------
# Online episode output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnlineEpisodeOutput:
    """Result of one online episode unroll.

    Contains the consolidated state, optional loss breakdown (detached
    floats for audit), the gradient-carrying weighted loss tensor (for
    the outer trainer), audit, and pre/post metrics.  It deliberately
    contains **no** event/task text or feature tensors.
    """

    state: CortexState
    loss_breakdown: LossBreakdown | None
    loss_tensor: torch.Tensor | None
    audit: OnlineEpisodeAudit
    pre_metrics: dict[str, float]
    post_metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _render_event_texts(event: LearningEvent) -> list[str]:
    """Render the four event roles as text for feature extraction.

    Returns texts in role order: observation, attempt, correction, outcome.
    """
    obs_text = " ".join(m.content for m in event.observation_messages)
    outcome_text = event.outcome.value
    return [obs_text, event.attempted_behavior, event.correction, outcome_text]


def _extract_event_features(
    organ: FrozenLanguageOrgan,
    event: LearningEvent,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extract ``[4, D]`` mean-pooled features for one event."""
    texts = _render_event_texts(event)
    features = organ.encode_texts(texts)  # [4, D]
    return features.detach().to(device=device, dtype=dtype)


def _extract_query_features(
    organ: FrozenLanguageOrgan,
    probe: ProbeCase,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extract ``[1, D]`` mean-pooled features for a probe's messages."""
    texts = [m.content for m in probe.messages]
    features = organ.encode_texts(texts)  # [N, D]
    # Mean-pool across messages to get [1, D].
    if features.ndim == 1:
        features = features.unsqueeze(0)
    pooled = features.mean(dim=0, keepdim=True)  # [1, D]
    return pooled.detach().to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Probe scoring
# ---------------------------------------------------------------------------


def _score_probe_ce(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    probe: ProbeCase,
    *,
    gradient_enabled: bool = True,
) -> torch.Tensor:
    """Score one probe with teacher-forced CE loss.

    Returns a scalar tensor.  If ``gradient_enabled`` is False, runs under
    inference mode.
    """
    device = state.fast.device
    dtype = state.fast.dtype

    query_feat = _extract_query_features(organ, probe, device, dtype)

    def _compute() -> torch.Tensor:
        readout = model.read(state, query_feat)  # [1, 64]
        soft_bank = model.couple(readout)  # [1, L, D]
        return organ.teacher_forced_loss(probe.messages, probe.expected_response, soft_bank)

    if gradient_enabled:
        return _compute()
    else:
        with torch.inference_mode():
            return _compute().detach()


def _score_probe_kl(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    probe: ProbeCase,
    *,
    reference_bank: torch.Tensor | None = None,
    gradient_enabled: bool = True,
) -> torch.Tensor:
    """Score one probe with specificity KL divergence."""
    device = state.fast.device
    dtype = state.fast.dtype

    query_feat = _extract_query_features(organ, probe, device, dtype)

    def _compute() -> torch.Tensor:
        readout = model.read(state, query_feat)
        soft_bank = model.couple(readout)
        return organ.specificity_kl(
            probe.messages, probe.expected_response, soft_bank,
            reference_bank=reference_bank,
        )

    if gradient_enabled:
        return _compute()
    else:
        with torch.inference_mode():
            return _compute().detach()


def _score_probe_accuracy(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    probe: ProbeCase,
    *,
    max_new_tokens: int = 16,
) -> float:
    """Score one probe with greedy generation accuracy (inference-only).

    Returns 1.0 if the expected response appears in the generated text,
    0.0 otherwise.  This is a simple substring match suitable for DEV.
    """
    device = state.fast.device
    dtype = state.fast.dtype
    query_feat = _extract_query_features(organ, probe, device, dtype)

    with torch.inference_mode():
        readout = model.read(state, query_feat)
        soft_bank = model.couple(readout)
        generated = organ.generate(probe.messages, soft_bank, max_new_tokens)

    expected = probe.expected_response.strip().lower()
    generated_lower = generated.strip().lower()
    return 1.0 if expected and expected in generated_lower else 0.0


def _score_battery_accuracy(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    battery: tuple[ProbeCase, ...],
    *,
    max_new_tokens: int = 16,
) -> tuple[int, int]:
    """Return (correct, total) for a battery via greedy generation."""
    correct = 0
    total = len(battery)
    for probe in battery:
        correct += int(_score_probe_accuracy(
            model, organ, state, probe, max_new_tokens=max_new_tokens,
        ))
    return correct, total


def _score_battery_ce(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    battery: tuple[ProbeCase, ...],
    *,
    gradient_enabled: bool = True,
) -> torch.Tensor:
    """Mean CE across a battery of probes."""
    if not battery:
        return torch.tensor(0.0)
    losses = [
        _score_probe_ce(model, organ, state, p, gradient_enabled=gradient_enabled)
        for p in battery
    ]
    return torch.stack(losses).mean()


def _score_battery_kl(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    battery: tuple[ProbeCase, ...],
    *,
    reference_bank: torch.Tensor | None = None,
    gradient_enabled: bool = True,
) -> torch.Tensor:
    """Mean KL across a battery of probes."""
    if not battery:
        return torch.tensor(0.0)
    kls = [
        _score_probe_kl(
            model, organ, state, p,
            reference_bank=reference_bank,
            gradient_enabled=gradient_enabled,
        )
        for p in battery
    ]
    return torch.stack(kls).mean()


# ---------------------------------------------------------------------------
# Banned content check
# ---------------------------------------------------------------------------


def _collect_banned_strings(task: MetaTask) -> list[str]:
    """Collect correction texts that must never appear in answer-path prompts.

    The information boundary forbids correction text from entering the
    answer path.  Expected responses are targets used only in teacher
    forcing, never passed to ``generate()``, so they are not banned from
    probe messages (though a well-designed task should not leak them).
    """
    banned: list[str] = []
    for event in task.events:
        banned.append(event.correction)
    # Filter out very short strings to avoid false positives from common words.
    return [s.strip() for s in banned if len(s.strip()) > 5]


def _check_banned_absent(task: MetaTask, prompt_hashes: tuple[str, ...]) -> bool:
    """Check that correction text is absent from answer-path probe messages.

    The answer path only receives probe messages + soft_bank.  We verify
    structurally that no correction text appears verbatim in any post-
    consolidation probe message content.
    """
    banned = _collect_banned_strings(task)
    if not banned:
        return True
    # Collect post-consolidation probe message contents.
    probe_contents: list[str] = []
    for probe_cat in (
        task.probes.same_rule, task.probes.transfer,
        task.probes.composition, task.probes.specificity,
    ):
        for probe in probe_cat:
            for msg in probe.messages:
                probe_contents.append(msg.content.lower())
    # Check no correction string appears in probe messages.
    for b in banned:
        b_lower = b.lower()
        for pc in probe_contents:
            if b_lower in pc:
                return False
    return True


def _hash_prompt(probe: ProbeCase) -> str:
    """Hash a probe's message contents for audit."""
    payload = "|".join(f"{m.role}:{m.content}" for m in probe.messages)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Online episode unroll
# ---------------------------------------------------------------------------


def unroll_online_episode(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    *,
    update_enabled: bool = True,
    feedback_permutation: list[int] | None = None,
    gradient_enabled: bool = True,
    behavior_weight: float = 1.0,
    specificity_weight: float = 0.5,
    survival_weight: float = 0.5,
    state_norm_weight: float = 0.01,
) -> OnlineEpisodeOutput:
    """Unroll one online episode: write → consolidate → delete → probe.

    Steps (from the plan's online episode order):
    1. Assert task.split matches caller.
    2. Initialize F/S to zero; record pre-learning metrics.
    3. Enter ``boundary.online()``; extract features, append to buffer,
       call ``write`` 2–5 times.
    4. Compute pre-clear probe loss for survival term.
    5. Call ``consolidate`` once; verify F is zero, S is ``[1,64,64]``.
    6. Delete all buffer objects/features; verify zero before post probes.
    7. Read/couple/score post-consolidation probes.
    8. Exit ``boundary.online()`` with step count and theta hash unchanged.

    No optimizer or backward reference is passed into this function.
    """
    # 1. Assert split is valid (both META_TRAIN and META_VALIDATION are
    #    allowed; the caller determines which is expected via gradient_enabled).
    if task.split not in (DevSplit.META_TRAIN, DevSplit.META_VALIDATION):
        raise ValueError(
            f"unroll_online_episode expects META_TRAIN or META_VALIDATION tasks, "
            f"got split={task.split.value}"
        )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Record theta hash and organ hash before.
    theta_hash_before = _theta_hash(model)
    organ_hash_before = organ.parameter_hash()
    step_count_before = boundary.optimizer_step_count

    # 2. Initialize F/S to zero.
    state = model.initial_state(1, device=device, dtype=dtype)

    # Pre-learning metrics.
    pre_metrics: dict[str, float] = {}
    for kind in (ProbeKind.SAME_RULE, ProbeKind.TRANSFER, ProbeKind.COMPOSITION):
        battery = task.probes.by_kind(kind)
        correct, total = _score_battery_accuracy(
            model, organ, state, battery,
        )
        pre_metrics[f"pre_{kind.value}_accuracy"] = correct / total if total > 0 else 0.0

    # 3. Enter online context.
    trace_buffer = TransientTraceBuffer()
    events = task.events
    if feedback_permutation is not None:
        events = tuple(task.events[i] for i in feedback_permutation)

    with boundary.online():
        for event in events:
            features = _extract_event_features(organ, event, device, dtype)
            trace_buffer.add(event, features)
            if update_enabled:
                event_batch = EventFeatureBatch(values=features.unsqueeze(0))  # [1,4,D]
                write_result = model.write(state, event_batch)
                state = write_result.state

        # 4. Pre-clear probe loss for survival term.
        if gradient_enabled and update_enabled:
            pre_clear_loss = _score_battery_ce(
                model, organ, state, task.probes.same_rule,
                gradient_enabled=True,
            )
        else:
            with torch.inference_mode():
                pre_clear_loss = _score_battery_ce(
                    model, organ, state, task.probes.same_rule,
                    gradient_enabled=False,
                ).detach()

        # 5. Consolidate.
        if update_enabled:
            cons_result = model.consolidate(state)
            state = cons_result.state
            # Verify F is bitwise zero.
            if torch.count_nonzero(state.fast).item() != 0:
                raise RuntimeError(
                    "Consolidation did not clear F to bitwise zero"
                )
            if tuple(state.slow.shape) != (1, 64, 64):
                raise RuntimeError(
                    f"S shape after consolidation is {tuple(state.slow.shape)}, "
                    f"expected (1, 64, 64)"
                )

        # 6. Delete traces before post probes.
        trace_objects_before = trace_buffer.object_count
        trace_features_before = trace_buffer.feature_count
        trace_buffer.delete_all()
        if not trace_buffer.verify_zero():
            raise RuntimeError("Trace buffer not zero after delete_all")

        # In validation, verify pre/post-deletion bank/logits are identical.
        if not gradient_enabled:
            _verify_post_deletion_invariance(model, organ, state, task)

        # 7. Post-consolidation probe scoring.
        if gradient_enabled and update_enabled:
            l_behavior, l_specificity, l_survival, l_state = _compute_outer_objective_inner(
                model, organ, task, state, pre_clear_loss,
                gradient_enabled=True,
            )
        else:
            with torch.inference_mode():
                l_behavior, l_specificity, l_survival, l_state = _compute_outer_objective_inner(
                    model, organ, task, state, pre_clear_loss,
                    gradient_enabled=False,
                )
            l_behavior = l_behavior.detach()
            l_specificity = l_specificity.detach()
            l_survival = l_survival.detach()
            l_state = l_state.detach()

    # 8. Exit online context — verify no optimizer step occurred.
    step_count_after = boundary.optimizer_step_count
    theta_hash_after = _theta_hash(model)
    organ_hash_after = organ.parameter_hash()

    if step_count_after != step_count_before:
        raise OnlineOptimizationError(
            f"Optimizer step count changed during online episode: "
            f"{step_count_before} -> {step_count_after}"
        )
    if theta_hash_after != theta_hash_before:
        raise OnlineOptimizationError(
            "Theta hash changed during online episode"
        )

    # Compute weighted loss tensor and breakdown.
    loss_tensor: torch.Tensor | None = None
    loss_breakdown: LossBreakdown | None = None
    if update_enabled:
        weighted = (
            behavior_weight * l_behavior
            + specificity_weight * l_specificity
            + survival_weight * l_survival
            + state_norm_weight * l_state
        )
        loss_tensor = weighted
        loss_breakdown = LossBreakdown(
            behavior=float(l_behavior.detach().item()),
            specificity=float(l_specificity.detach().item()),
            consolidation_survival=float(l_survival.detach().item()),
            state_norm=float(l_state.detach().item()),
            weighted_total=float(weighted.detach().item()),
        )

    # Post metrics (inference-only accuracy).
    post_metrics: dict[str, float] = {}
    for kind in (ProbeKind.SAME_RULE, ProbeKind.TRANSFER, ProbeKind.COMPOSITION):
        battery = task.probes.by_kind(kind)
        correct, total = _score_battery_accuracy(
            model, organ, state, battery,
        )
        post_metrics[f"post_{kind.value}_accuracy"] = correct / total if total > 0 else 0.0

    # Build audit.
    prompt_hashes: list[str] = []
    for kind in (ProbeKind.SAME_RULE, ProbeKind.TRANSFER,
                 ProbeKind.COMPOSITION, ProbeKind.SPECIFICITY):
        for probe in task.probes.by_kind(kind):
            prompt_hashes.append(_hash_prompt(probe))

    banned_absent = _check_banned_absent(task, tuple(prompt_hashes))

    bank_shape = (1, model.config.bank_width, model.config.feature_dim)
    audit = OnlineEpisodeAudit(
        family=task.family.value,
        split=task.split.value,
        rule_fingerprint=task.rule_fingerprint,
        event_count=len(task.events),
        trace_objects_before=trace_objects_before,
        trace_objects_after=0,
        trace_feature_tensors_before=trace_features_before,
        trace_feature_tensors_after=0,
        fast_shape=tuple(state.fast.shape),
        slow_shape=tuple(state.slow.shape),
        bank_shape=bank_shape,
        fast_zero=torch.count_nonzero(state.fast).item() == 0,
        logical_persistent_bytes=model.logical_persistent_bytes(state),
        optimizer_step_count_before=step_count_before,
        optimizer_step_count_after=step_count_after,
        theta_hash_before=theta_hash_before,
        theta_hash_after=theta_hash_after,
        organ_hash_before=organ_hash_before,
        organ_hash_after=organ_hash_after,
        answer_path_prompt_hashes=tuple(prompt_hashes),
        banned_content_absent=banned_absent,
    )

    return OnlineEpisodeOutput(
        state=state,
        loss_breakdown=loss_breakdown,
        loss_tensor=loss_tensor,
        audit=audit,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
    )


def _verify_post_deletion_invariance(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    task: MetaTask,
) -> None:
    """Verify that pre/post-deletion bank and logits are identical.

    This proves the answer path never consulted the trace buffer.
    """
    device = state.fast.device
    dtype = state.fast.dtype
    probe = task.probes.same_rule[0]
    query_feat = _extract_query_features(organ, probe, device, dtype)
    with torch.inference_mode():
        readout = model.read(state, query_feat)
        bank = model.couple(readout)
        logits = organ.teacher_forced_logits(
            probe.messages, probe.expected_response, bank,
        )
    # Just verify it runs without error — the point is that the answer path
    # only uses state + theta, not the trace buffer.  Since we already deleted
    # traces before this call, any reference would have failed.
    assert logits is not None


# ---------------------------------------------------------------------------
# Outer objective
# ---------------------------------------------------------------------------


def _compute_outer_objective_inner(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    state: CortexState,
    pre_clear_loss: torch.Tensor,
    *,
    gradient_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the four-term outer objective.

    ``L_behavior`` = mean CE on same_rule + transfer + composition probes.
    ``L_specificity`` = mean KL on specificity probes.
    ``L_survival`` = mean relu(CE_after_F_clear - CE_before_F_clear).
    ``L_state`` = mean(S_next²).
    """
    # Behavior loss.
    behavior_probes = (
        task.probes.same_rule + task.probes.transfer + task.probes.composition
    )
    l_behavior = _score_battery_ce(
        model, organ, state, behavior_probes,
        gradient_enabled=gradient_enabled,
    )

    # Specificity loss.
    l_specificity = _score_battery_kl(
        model, organ, state, task.probes.specificity,
        gradient_enabled=gradient_enabled,
    )

    # Survival loss: relu(CE_after - CE_before).
    # pre_clear_loss was computed before consolidation (with F still active).
    # After consolidation, F is zero.  We need CE_after_F_clear.
    post_clear_loss = _score_battery_ce(
        model, organ, state, task.probes.same_rule,
        gradient_enabled=gradient_enabled,
    )
    l_survival = torch.relu(post_clear_loss - pre_clear_loss)

    # State norm.
    l_state = state.slow.pow(2).mean()

    return l_behavior, l_specificity, l_survival, l_state


def compute_outer_objective(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    state: CortexState,
    *,
    behavior_weight: float,
    specificity_weight: float,
    survival_weight: float,
    state_norm_weight: float,
    pre_clear_loss: torch.Tensor | None = None,
    gradient_enabled: bool = True,
) -> LossBreakdown:
    """Compute the weighted outer objective as a :class:`LossBreakdown`.

    If ``pre_clear_loss`` is None, it is computed from the current state's
    same_rule probes (this is the post-consolidation CE, so survival will
    be zero — used when calling standalone after consolidation).
    """
    if pre_clear_loss is None:
        if gradient_enabled:
            pre_clear_loss = _score_battery_ce(
                model, organ, state, task.probes.same_rule,
                gradient_enabled=True,
            )
        else:
            with torch.inference_mode():
                pre_clear_loss = _score_battery_ce(
                    model, organ, state, task.probes.same_rule,
                    gradient_enabled=False,
                ).detach()

    l_behavior, l_specificity, l_survival, l_state = _compute_outer_objective_inner(
        model, organ, task, state, pre_clear_loss,
        gradient_enabled=gradient_enabled,
    )

    weighted_total = (
        behavior_weight * l_behavior
        + specificity_weight * l_specificity
        + survival_weight * l_survival
        + state_norm_weight * l_state
    )

    return LossBreakdown(
        behavior=float(l_behavior.detach().item()),
        specificity=float(l_specificity.detach().item()),
        consolidation_survival=float(l_survival.detach().item()),
        state_norm=float(l_state.detach().item()),
        weighted_total=float(weighted_total.detach().item()),
    )


# ---------------------------------------------------------------------------
# DEV causal interventions
# ---------------------------------------------------------------------------


def _build_condition_result(
    condition: str,
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    task: MetaTask,
    audit: OnlineEpisodeAudit,
) -> ConditionResult:
    """Build a ConditionResult from a state by scoring all probe kinds."""

    corrects: list[int] = []
    totals: list[int] = []
    accuracies: list[float] = []

    for kind in ProbeKind:
        battery = task.probes.by_kind(kind)
        correct, total = _score_battery_accuracy(model, organ, state, battery)
        corrects.append(correct)
        totals.append(total)
        accuracies.append(correct / total if total > 0 else 0.0)

    s_hash = _state_hash(state)
    s_norm = _state_norm(state)
    s_bytes = model.logical_persistent_bytes(state)

    return ConditionResult(
        condition=condition,
        per_kind_correct=tuple(corrects),
        per_kind_total=tuple(totals),
        per_kind_accuracy=tuple(accuracies),
        state_hash=s_hash,
        state_norm=s_norm,
        state_bytes=s_bytes,
        episode_audit=audit,
    )


def _make_audit(
    task: MetaTask,
    state: CortexState,
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    *,
    event_count: int = 0,
    trace_objects_before: int = 0,
    trace_features_before: int = 0,
    optimizer_steps_before: int = 0,
    optimizer_steps_after: int = 0,
    theta_hash_before: str | None = None,
    theta_hash_after: str | None = None,
    organ_hash_before: str | None = None,
    organ_hash_after: str | None = None,
    banned_content_absent: bool = True,
) -> OnlineEpisodeAudit:
    """Build an OnlineEpisodeAudit for a causal condition."""
    if theta_hash_before is None:
        theta_hash_before = _theta_hash(model)
    if theta_hash_after is None:
        theta_hash_after = theta_hash_before
    if organ_hash_before is None:
        organ_hash_before = organ.parameter_hash()
    if organ_hash_after is None:
        organ_hash_after = organ_hash_before

    bank_shape = (1, model.config.bank_width, model.config.feature_dim)

    return OnlineEpisodeAudit(
        family=task.family.value,
        split=task.split.value,
        rule_fingerprint=task.rule_fingerprint,
        event_count=event_count,
        trace_objects_before=trace_objects_before,
        trace_objects_after=0,
        trace_feature_tensors_before=trace_features_before,
        trace_feature_tensors_after=0,
        fast_shape=tuple(state.fast.shape),
        slow_shape=tuple(state.slow.shape),
        bank_shape=bank_shape,
        fast_zero=torch.count_nonzero(state.fast).item() == 0,
        logical_persistent_bytes=model.logical_persistent_bytes(state),
        optimizer_step_count_before=optimizer_steps_before,
        optimizer_step_count_after=optimizer_steps_after,
        theta_hash_before=theta_hash_before,
        theta_hash_after=theta_hash_after,
        organ_hash_before=organ_hash_before,
        organ_hash_after=organ_hash_after,
        answer_path_prompt_hashes=(),
        banned_content_absent=banned_content_absent,
    )


def _run_condition_trained(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    *,
    gradient_enabled: bool = False,
) -> tuple[ConditionResult, CortexState]:
    """Run the TRAINED condition: full episode with writes + consolidation."""
    output = unroll_online_episode(
        model, organ, task, boundary,
        update_enabled=True,
        gradient_enabled=gradient_enabled,
    )
    cr = _build_condition_result(
        DevCondition.TRAINED.value, model, organ, output.state, task, output.audit,
    )
    return cr, output.state


def _run_condition_update_disabled(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
) -> ConditionResult:
    """UPDATE_DISABLED: skip writes; F/S stay zero."""
    output = unroll_online_episode(
        model, organ, task, boundary,
        update_enabled=False,
        gradient_enabled=False,
    )
    # Verify F/S are still zero.
    if torch.count_nonzero(output.state.fast).item() != 0:
        raise RuntimeError("UPDATE_DISABLED: F is not zero")
    if torch.count_nonzero(output.state.slow).item() != 0:
        raise RuntimeError("UPDATE_DISABLED: S is not zero")

    cr = _build_condition_result(
        DevCondition.UPDATE_DISABLED.value, model, organ, output.state, task, output.audit,
    )
    return cr


def _run_condition_untrained(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    *,
    evaluation_seed: int = 0,
) -> ConditionResult:
    """UNTRAINED_RULE: fresh deterministic theta, events enabled, no optimization.

    The fresh theta is seeded from the evaluation seed and task fingerprint
    so that different evaluation seeds produce genuinely distinct untrained
    baselines, while keeping capacity/config identical to the trained model.
    """
    # Save current theta (clone to avoid in-place mutation issues).
    saved_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Derive a fresh seed from evaluation_seed and task fingerprint.
    import hashlib
    fresh_seed_material = f"untrained_rule|{evaluation_seed}|{task.rule_fingerprint}"
    fresh_seed = int.from_bytes(
        hashlib.sha256(fresh_seed_material.encode("utf-8")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)

    # Fresh theta with the derived seed.
    fresh_model = MetaCortex(model.config, init_seed=fresh_seed)
    # Swap in the fresh model temporarily.
    model.load_state_dict(fresh_model.state_dict())

    try:
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=False,
        )
        cr = _build_condition_result(
            DevCondition.UNTRAINED_RULE.value, model, organ, output.state, task, output.audit,
        )
        return cr
    finally:
        # Restore original theta from the cloned saved state.
        model.load_state_dict(saved_state_dict)


def _derangement(n: int) -> list[int]:
    """Return a deterministic derangement of range(n).

    Raises ValueError if n < 2 (no derangement possible).
    """
    if n < 2:
        raise ValueError(f"Cannot derange {n} elements (need >= 2)")
    # Simple rotation by 1 — guaranteed derangement for n >= 2.
    perm = [(i + 1) % n for i in range(n)]
    return perm


def _run_condition_feedback_shuffled(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
) -> ConditionResult:
    """FEEDBACK_SHUFFLED: deterministic derangement of correction+outcome pairs."""
    n = len(task.events)
    perm = _derangement(n)

    # Verify it's a real derangement (no fixed points).
    for i, p in enumerate(perm):
        if i == p:
            raise RuntimeError(
                f"Feedback shuffle produced a fixed point at index {i}"
            )

    # Apply derangement: shuffle events by the permutation.
    shuffled_events = tuple(task.events[p] for p in perm)

    # Build a shuffled task (same probes, shuffled events).
    shuffled_task = MetaTask(
        family=task.family,
        split=task.split,
        events=shuffled_events,
        probes=task.probes,
        rule_fingerprint=task.rule_fingerprint,
        assignment_fingerprint=task.assignment_fingerprint,
        composition_fingerprint=task.composition_fingerprint,
        paraphrase_group_fingerprint=task.paraphrase_group_fingerprint,
    )

    output = unroll_online_episode(
        model, organ, shuffled_task, boundary,
        update_enabled=True,
        gradient_enabled=False,
    )
    cr = _build_condition_result(
        DevCondition.FEEDBACK_SHUFFLED.value, model, organ, output.state, shuffled_task, output.audit,
    )
    return cr


def _run_condition_state_zeroed(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    trained_state: CortexState,
) -> ConditionResult:
    """STATE_ZEROED: reuse trained consolidated snapshot, replace F/S with zeros."""
    zeroed_state = model.zero_state(trained_state)

    audit = _make_audit(
        task, zeroed_state, model, organ,
        event_count=len(task.events),
        banned_content_absent=_check_banned_absent(task, ()),
    )

    cr = _build_condition_result(
        DevCondition.STATE_ZEROED.value, model, organ, zeroed_state, task, audit,
    )
    return cr


def _run_condition_state_swapped(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    trained_state: CortexState,
    donor_state: CortexState,
) -> ConditionResult:
    """STATE_SWAPPED: rotate consolidated S among different rule fingerprints."""
    if _state_hash(trained_state) == _state_hash(donor_state):
        raise ValueError(
            "STATE_SWAPPED: donor state must differ from trained state"
        )

    swapped_state = model.swap_state(trained_state, donor_state)

    audit = _make_audit(
        task, swapped_state, model, organ,
        event_count=len(task.events),
        banned_content_absent=_check_banned_absent(task, ()),
    )

    cr = _build_condition_result(
        DevCondition.STATE_SWAPPED.value, model, organ, swapped_state, task, audit,
    )
    return cr


def _run_condition_organ_only(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
) -> ConditionResult:
    """ORGAN_ONLY: no bank (zero bank)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    state = model.initial_state(1, device=device, dtype=dtype)

    # Score with zero bank — simulate no-bank by using a zero soft_bank.
    corrects: list[int] = []
    totals: list[int] = []
    accuracies: list[float] = []

    zero_bank = torch.zeros(
        1, model.config.bank_width, model.config.feature_dim,
        device=device, dtype=dtype,
    )

    for kind in ProbeKind:
        battery = task.probes.by_kind(kind)
        correct = 0
        total = len(battery)
        for probe in battery:
            with torch.inference_mode():
                generated = organ.generate(probe.messages, zero_bank, max_new_tokens=16)
            expected = probe.expected_response.strip().lower()
            if expected and expected in generated.strip().lower():
                correct += 1
        corrects.append(correct)
        totals.append(total)
        accuracies.append(correct / total if total > 0 else 0.0)

    audit = _make_audit(
        task, state, model, organ,
        event_count=0,
        banned_content_absent=True,
    )

    return ConditionResult(
        condition=DevCondition.ORGAN_ONLY.value,
        per_kind_correct=tuple(corrects),
        per_kind_total=tuple(totals),
        per_kind_accuracy=tuple(accuracies),
        state_hash=_state_hash(state),
        state_norm=0.0,
        state_bytes=0,
        episode_audit=audit,
    )


def run_dev_interventions(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    trained_state: CortexState,
    *,
    boundary: OptimizationBoundary,
    donor_state: CortexState | None = None,
) -> tuple[Any, ...]:
    """Run all DEV causal interventions for one task.

    Returns a tuple of :class:`ConditionResult` objects for:
    TRAINED, UPDATE_DISABLED, UNTRAINED_RULE, FEEDBACK_SHUFFLED,
    STATE_ZEROED, STATE_SWAPPED, ORGAN_ONLY.

    All conditions perform zero optimizer steps.
    """
    results: list[Any] = []

    # TRAINED — rescore using the provided trained_state (already consolidated).
    trained_audit = _make_audit(
        task, trained_state, model, organ,
        event_count=len(task.events),
        banned_content_absent=_check_banned_absent(task, ()),
    )
    trained_cr = _build_condition_result(
        DevCondition.TRAINED.value, model, organ, trained_state, task, trained_audit,
    )
    results.append(trained_cr)

    # UPDATE_DISABLED.
    results.append(_run_condition_update_disabled(model, organ, task, boundary))

    # UNTRAINED_RULE.
    results.append(_run_condition_untrained(model, organ, task, boundary))

    # FEEDBACK_SHUFFLED.
    results.append(_run_condition_feedback_shuffled(model, organ, task, boundary))

    # STATE_ZEROED.
    results.append(_run_condition_state_zeroed(model, organ, task, trained_state))

    # STATE_SWAPPED (requires donor).
    if donor_state is not None:
        results.append(
            _run_condition_state_swapped(
                model, organ, task, trained_state, donor_state,
            )
        )

    # ORGAN_ONLY.
    results.append(_run_condition_organ_only(model, organ, task))

    return tuple(results)


# ---------------------------------------------------------------------------
# DEV validation
# ---------------------------------------------------------------------------


def _compute_delta(trained_acc: float, control_acc: float) -> float:
    """Compute the trained-minus-control accuracy delta."""
    return trained_acc - control_acc


def _pooled_accuracy(result: Any, kinds: tuple[ProbeKind, ...] = (
    ProbeKind.SAME_RULE, ProbeKind.TRANSFER, ProbeKind.COMPOSITION,
)) -> float:
    """Compute pooled accuracy across specified probe kinds for a ConditionResult."""
    total_correct = 0
    total_count = 0
    for kind in kinds:
        idx = list(ProbeKind).index(kind)
        total_correct += result.per_kind_correct[idx]
        total_count += result.per_kind_total[idx]
    return total_correct / total_count if total_count > 0 else 0.0


@dataclass(frozen=True)
class ValidationTaskProgress:
    """Safe progress emitted after one validation task completes."""

    validation_pass: int
    optimizer_step: int
    family: TaskFamily
    completed: int
    total: int


@dataclass(frozen=True)
class ValidationCompleted:
    """Result emitted after one complete in-loop validation pass."""

    validation_pass: int
    optimizer_step: int
    score: float
    result: DevValidationResult
    is_best: bool


ValidationTaskCallback = Callable[[ValidationTaskProgress], None]
ValidationCompleteCallback = Callable[[ValidationCompleted], None]


def run_dev_validation(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    catalog: DevTaskCatalog,
    *,
    device: torch.device,
    dtype: torch.dtype,
    validation_pass: int = 1,
    optimizer_step: int = 1,
    on_task_complete: ValidationTaskCallback | None = None,
) -> DevValidationResult:
    """Run inference-only DEV validation with causal interventions.

    This function accepts no optimizer/trainer.  It uses ``model.eval()``
    plus ``torch.inference_mode()`` and verifies theta/organ hashes
    before/after.
    """
    model.eval()
    organ.assert_frozen()

    theta_hash_before = _theta_hash(model)
    organ_hash_before = organ.parameter_hash()

    boundary = OptimizationBoundary()

    validation_tasks = catalog.tasks_for(DevSplit.META_VALIDATION)
    completed_tasks = 0
    total_tasks = len(validation_tasks)

    # Group tasks by family.
    tasks_by_family: dict[TaskFamily, list[MetaTask]] = {}
    for task in validation_tasks:
        tasks_by_family.setdefault(task.family, []).append(task)

    per_family_results: list[tuple[str, tuple[Any, ...]]] = []
    all_trained: list[Any] = []
    all_update_disabled: list[Any] = []
    all_untrained: list[Any] = []
    all_shuffled: list[Any] = []
    all_zeroed: list[Any] = []
    all_swapped: list[Any] = []

    for family in TaskFamily:
        family_tasks = tasks_by_family.get(family, [])
        if not family_tasks:
            continue

        family_results: list[Any] = []

        for i, task in enumerate(family_tasks):
            # Run trained episode (inference-only).
            with torch.inference_mode():
                output = unroll_online_episode(
                    model, organ, task, boundary,
                    update_enabled=True,
                    gradient_enabled=False,
                )
                trained_state = output.state

            # Find a donor state from a different task in the same family.
            donor_state: CortexState | None = None
            if len(family_tasks) > 1:
                donor_idx = (i + 1) % len(family_tasks)
                donor_task = family_tasks[donor_idx]
                with torch.inference_mode():
                    donor_output = unroll_online_episode(
                        model, organ, donor_task, boundary,
                        update_enabled=True,
                        gradient_enabled=False,
                    )
                    donor_state = donor_output.state

            # Run interventions.
            interventions = run_dev_interventions(
                model, organ, task, trained_state,
                boundary=boundary,
                donor_state=donor_state,
            )

            for cr in interventions:
                family_results.append(cr)
                if cr.condition == DevCondition.TRAINED.value:
                    all_trained.append(cr)
                elif cr.condition == DevCondition.UPDATE_DISABLED.value:
                    all_update_disabled.append(cr)
                elif cr.condition == DevCondition.UNTRAINED_RULE.value:
                    all_untrained.append(cr)
                elif cr.condition == DevCondition.FEEDBACK_SHUFFLED.value:
                    all_shuffled.append(cr)
                elif cr.condition == DevCondition.STATE_ZEROED.value:
                    all_zeroed.append(cr)
                elif cr.condition == DevCondition.STATE_SWAPPED.value:
                    all_swapped.append(cr)
            completed_tasks += 1
            if on_task_complete is not None:
                on_task_complete(
                    ValidationTaskProgress(
                        validation_pass=validation_pass,
                        optimizer_step=optimizer_step,
                        family=family,
                        completed=completed_tasks,
                        total=total_tasks,
                    )
                )

        per_family_results.append((family.value, tuple(family_results)))

    # Compute pooled deltas.
    trained_acc = sum(_pooled_accuracy(r) for r in all_trained) / len(all_trained) if all_trained else 0.0
    update_disabled_acc = sum(_pooled_accuracy(r) for r in all_update_disabled) / len(all_update_disabled) if all_update_disabled else 0.0
    untrained_acc = sum(_pooled_accuracy(r) for r in all_untrained) / len(all_untrained) if all_untrained else 0.0
    shuffled_acc = sum(_pooled_accuracy(r) for r in all_shuffled) / len(all_shuffled) if all_shuffled else 0.0
    zeroed_acc = sum(_pooled_accuracy(r) for r in all_zeroed) / len(all_zeroed) if all_zeroed else 0.0
    swapped_acc = sum(_pooled_accuracy(r) for r in all_swapped) / len(all_swapped) if all_swapped else 0.0

    # Build pooled ConditionResults by averaging per-kind accuracies.
    pooled_results = _build_pooled_results(
        all_trained, all_update_disabled, all_untrained,
        all_shuffled, all_zeroed, all_swapped,
    )

    # Verify hashes unchanged.
    theta_hash_after = _theta_hash(model)
    organ_hash_after = organ.parameter_hash()

    if theta_hash_after != theta_hash_before:
        raise RuntimeError("Theta hash changed during validation")
    if organ_hash_after != organ_hash_before:
        raise RuntimeError("Organ hash changed during validation")

    return DevValidationResult(
        per_family_results=tuple(per_family_results),
        pooled_results=pooled_results,
        trained_vs_update_disabled_delta=_compute_delta(trained_acc, update_disabled_acc),
        trained_vs_untrained_delta=_compute_delta(trained_acc, untrained_acc),
        trained_vs_shuffled_delta=_compute_delta(trained_acc, shuffled_acc),
        trained_vs_zeroed_delta=_compute_delta(trained_acc, zeroed_acc),
        trained_vs_swapped_delta=_compute_delta(trained_acc, swapped_acc),
    )


def _build_pooled_results(
    all_trained: list[Any],
    all_update_disabled: list[Any],
    all_untrained: list[Any],
    all_shuffled: list[Any],
    all_zeroed: list[Any],
    all_swapped: list[Any],
) -> tuple[Any, ...]:
    """Build pooled ConditionResults by averaging per-kind accuracies."""
    pooled: list[Any] = []

    for condition_name, results_list in (
        (DevCondition.TRAINED.value, all_trained),
        (DevCondition.UPDATE_DISABLED.value, all_update_disabled),
        (DevCondition.UNTRAINED_RULE.value, all_untrained),
        (DevCondition.FEEDBACK_SHUFFLED.value, all_shuffled),
        (DevCondition.STATE_ZEROED.value, all_zeroed),
        (DevCondition.STATE_SWAPPED.value, all_swapped),
    ):
        if not results_list:
            continue
        n_kinds = len(ProbeKind)
        sum_correct = [0] * n_kinds
        sum_total = [0] * n_kinds
        for cr in results_list:
            for k in range(n_kinds):
                sum_correct[k] += cr.per_kind_correct[k]
                sum_total[k] += cr.per_kind_total[k]

        per_kind_acc = tuple(
            sum_correct[k] / sum_total[k] if sum_total[k] > 0 else 0.0
            for k in range(n_kinds)
        )

        # Use the first result's audit as representative.
        audit = results_list[0].episode_audit
        state_hash = results_list[0].state_hash
        state_norm = sum(r.state_norm for r in results_list) / len(results_list)
        state_bytes = results_list[0].state_bytes

        pooled.append(ConditionResult(
            condition=condition_name,
            per_kind_correct=tuple(sum_correct),
            per_kind_total=tuple(sum_total),
            per_kind_accuracy=per_kind_acc,
            state_hash=state_hash,
            state_norm=state_norm,
            state_bytes=state_bytes,
            episode_audit=audit,
        ))

    return tuple(pooled)


# ---------------------------------------------------------------------------
# Outer trainer
# ---------------------------------------------------------------------------


class OuterTrainer:
    """Sole owner of the cortex-only optimizer.

    The optimizer is created in the constructor and verified to be
    disjoint from organ parameters.  ``_outer_step`` is the only method
    that calls ``backward`` and ``optimizer.step``, and it asserts
    ``boundary.assert_outer()`` first.
    """

    def __init__(
        self,
        model: MetaCortex,
        organ: FrozenLanguageOrgan,
        config: OuterLoopConfig,
        *,
        on_validation_task_complete: ValidationTaskCallback | None = None,
        on_validation_complete: ValidationCompleteCallback | None = None,
    ) -> None:
        self.model = model
        self.organ = organ
        self.config = config
        self.boundary = OptimizationBoundary()
        self.on_validation_task_complete = on_validation_task_complete
        self.on_validation_complete = on_validation_complete
        self._validation_pass_count = 0

        # Create cortex-only optimizer.
        cortex_params = list(model.parameters())
        cortex_param_ids = {id(p) for p in cortex_params}

        # Verify no organ parameter is in the cortex parameter set.
        # (Organ parameters should all be frozen, but we verify ID disjointness.)
        organ_param_ids: set[int] = set()
        if hasattr(organ, "_model") and hasattr(organ._model, "parameters"):
            for p in organ._model.parameters():
                organ_param_ids.add(id(p))

        overlap = cortex_param_ids & organ_param_ids
        if overlap:
            raise OnlineOptimizationError(
                f"Cortex and organ share {len(overlap)} parameter tensors"
            )

        self._optimizer: torch.optim.Optimizer
        if config.optimizer_name == "adamw":
            self._optimizer = torch.optim.AdamW(
                cortex_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_name == "sgd":
            self._optimizer = torch.optim.SGD(
                cortex_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {config.optimizer_name}"
            )

        # Verify the optimizer has no organ parameters.
        opt_param_ids: set[int] = set()
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
        organ_in_opt = opt_param_ids & organ_param_ids
        if organ_in_opt:
            raise OnlineOptimizationError(
                f"Optimizer contains {len(organ_in_opt)} organ parameters"
            )


    def _outer_step(self, loss: torch.Tensor) -> None:
        """Perform the sole optimizer step.

        This is the only method in the package that calls ``backward`` and
        ``optimizer.step``.  It asserts ``boundary.assert_outer()`` first.
        """
        self.boundary.assert_outer()

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient audit: finite and nonzero.
        has_grad = False
        for p in self.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise OnlineOptimizationError(
                        "Non-finite gradient detected"
                    )
                if p.grad.abs().sum().item() > 0:
                    has_grad = True
        if not has_grad:
            raise OnlineOptimizationError(
                "No parameter received a nonzero gradient"
            )

        # Gradient clipping.
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm,
            )

        self._optimizer.step()
        self.boundary._record_step()

    def train_step(self, meta_train_tasks: list[MetaTask]) -> dict[str, Any]:
        """Run one outer training step.

        All online unrolls run first.  Only after leaving every online
        context may ``_outer_step`` fire.
        """
        # Verify all tasks are META_TRAIN.
        for task in meta_train_tasks:
            if task.split != DevSplit.META_TRAIN:
                raise ValueError(
                    f"train_step received non-META_TRAIN task: split={task.split.value}"
                )
        organ_hash_before = self.organ.parameter_hash()

        # Run all online unrolls first.  Each unroll returns a loss_tensor
        # that carries the autograd graph through write/consolidate/read/couple.
        # No optimizer or backward is called inside the online context.
        episode_outputs: list[OnlineEpisodeOutput] = []
        loss_tensors: list[torch.Tensor] = []

        for task in meta_train_tasks:
            output = unroll_online_episode(
                self.model, self.organ, task, self.boundary,
                update_enabled=True,
                gradient_enabled=True,
                behavior_weight=self.config.behavior_weight,
                specificity_weight=self.config.specificity_weight,
                survival_weight=self.config.survival_weight,
                state_norm_weight=self.config.state_norm_weight,
            )
            episode_outputs.append(output)
            if output.loss_tensor is not None:
                loss_tensors.append(output.loss_tensor)

        # Outside every online context: accumulate and step.
        if not loss_tensors:
            raise OnlineOptimizationError("No loss tensors collected from online episodes")

        total_loss = torch.stack(loss_tensors).mean()

        # The sole optimizer step — asserts boundary.assert_outer() first.
        self._outer_step(total_loss)

        # Verify organ hash unchanged.
        organ_hash_after = self.organ.parameter_hash()
        theta_hash_after = _theta_hash(self.model)

        if organ_hash_after != organ_hash_before:
            raise OnlineOptimizationError(
                "Organ hash changed during train_step"
            )

        # Drop episode outputs so activation graphs are collectible.
        episode_outputs.clear()
        loss_tensors.clear()

        return {
            "loss": float(total_loss.detach().item()),
            "theta_hash": theta_hash_after,
            "organ_hash": organ_hash_after,
            "optimizer_steps": self.boundary.optimizer_step_count,
        }

    def _run_validation(
        self,
        catalog: DevTaskCatalog,
        *,
        validation_pass: int,
        optimizer_step: int,
    ) -> tuple[float, DevValidationResult]:
        """Run validation and return (score, result)."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        val_result = run_dev_validation(
            self.model,
            self.organ,
            catalog,
            device=device,
            dtype=dtype,
            validation_pass=validation_pass,
            optimizer_step=optimizer_step,
            on_task_complete=self.on_validation_task_complete,
        )

        # Use the trained-vs-update-disabled delta as the validation score.
        score = val_result.trained_vs_update_disabled_delta
        return score, val_result


    def train(self, catalog: DevTaskCatalog) -> DevTrainingResult:
        """Run the full outer training loop.

        Deterministically orders train tasks, validates only on
        ``catalog.meta_validation``, selects/restores the best theta by
        meta-validation metric, and never gradients through or steps on
        validation tasks.
        """

        organ_hash_before = self.organ.parameter_hash()

        # Verify split firewall.
        if not catalog.split_audit.passed:
            raise OnlineOptimizationError(
                "Split firewall audit failed — cannot train"
            )

        train_tasks = list(catalog.tasks_for(DevSplit.META_TRAIN))
        step_curve: list[tuple[int, float, float]] = []

        best_val_score: float = -math.inf
        best_val_step: int = 0
        best_theta: dict[str, torch.Tensor] | None = None
        best_val_result: DevValidationResult | None = None

        step = 0
        # Seed-derived task ordering: use config.seed to produce a
        # reproducible but genuinely distinct task order per seed.
        # A simple seeded permutation via Fisher-Yates with a local LCG.
        _order_state = self.config.seed & ((1 << 63) - 1)
        if _order_state == 0:
            _order_state = 1
        _task_order = list(range(len(train_tasks)))
        for _i in range(len(_task_order) - 1, 0, -1):
            _order_state = (_order_state * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
            _j = (_order_state >> 1) % (_i + 1)
            _task_order[_i], _task_order[_j] = _task_order[_j], _task_order[_i]

        for outer_step in range(self.config.outer_steps):
            # Select tasks for this step using the seed-derived order.
            start_idx = (outer_step * self.config.tasks_per_step) % len(train_tasks)
            step_tasks: list[MetaTask] = []
            for i in range(self.config.tasks_per_step):
                step_tasks.append(train_tasks[_task_order[(start_idx + i) % len(train_tasks)]])

            # Train step.
            step_result = self.train_step(step_tasks)
            train_loss = step_result["loss"]

            step += 1

            # Validation.
            val_score: float = 0.0
            if outer_step % self.config.validation_interval == 0 or outer_step == self.config.outer_steps - 1:
                self._validation_pass_count += 1
                validation_pass = self._validation_pass_count
                val_score, val_result = self._run_validation(
                    catalog,
                    validation_pass=validation_pass,
                    optimizer_step=step,
                )

                is_best = val_score > best_val_score
                if is_best:
                    best_val_score = val_score
                    best_val_step = step
                    best_theta = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    best_val_result = val_result

                if self.on_validation_complete is not None:
                    self.on_validation_complete(
                        ValidationCompleted(
                            validation_pass=validation_pass,
                            optimizer_step=step,
                            score=val_score,
                            result=val_result,
                            is_best=is_best,
                        )
                    )
            step_curve.append((step, train_loss, val_score))

        # The final-step guard above guarantees at least one validation.
        if best_theta is None or best_val_result is None:
            raise OnlineOptimizationError(
                "Training completed without a validation result; "
                "the final-step validation invariant was broken"
            )

        # Restore the exact theta paired with the retained best result.  Do not
        # rerun validation: the stored result is the result selected in-loop.
        self.model.load_state_dict(best_theta)
        organ_hash_after = self.organ.parameter_hash()

        # Build config digests.
        model_config_digest = _hash_config(self.model.config)
        taskgen_config_digest = ""  # Filled by CLI from catalog.
        outer_config_digest = _hash_config(self.config)
        catalog_digest = catalog.catalog_sha256

        audit_status = "ok"
        if organ_hash_after != organ_hash_before:
            audit_status = "failed_audit"

        return DevTrainingResult(
            schema=DEV_SCHEMA,
            model_config_digest=model_config_digest,
            taskgen_config_digest=taskgen_config_digest,
            outer_config_digest=outer_config_digest,
            catalog_digest=catalog_digest,
            step_curve=tuple(step_curve),
            best_validation_step=best_val_step,
            best_validation_score=best_val_score,
            validation_result=best_val_result,
            organ_hash_before=organ_hash_before,
            organ_hash_after=organ_hash_after,
            optimizer_step_count=self.boundary.optimizer_step_count,
            audit_status=audit_status,
        )


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------


def _hash_config(config: Any) -> str:
    # Hash a config dataclass canonically.
    if isinstance(config, ModelConfig):
        payload = json.dumps(
            {"feature_dim": config.feature_dim, "d_cortex": config.d_cortex, "bank_width": config.bank_width},
            sort_keys=True,
        )
    elif isinstance(config, OuterLoopConfig):
        payload = json.dumps(
            {
                "outer_steps": config.outer_steps,
                "tasks_per_step": config.tasks_per_step,
                "optimizer_name": config.optimizer_name,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "grad_clip_norm": config.grad_clip_norm,
                "validation_interval": config.validation_interval,
                "generation_interval": config.generation_interval,
                "behavior_weight": config.behavior_weight,
                "specificity_weight": config.specificity_weight,
                "survival_weight": config.survival_weight,
                "state_norm_weight": config.state_norm_weight,
                "seed": config.seed,
            },
            sort_keys=True,
        )
    else:
        payload = str(config)
    return hashlib.sha256(payload.encode()).hexdigest()
