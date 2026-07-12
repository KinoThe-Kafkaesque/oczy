"""Resumable DEV calibration record collector for the meta_cortex/v1 instrument.

This module implements **exact-scored collection** for one deterministic shard
of the DEV calibration campaign.  It is the production counterpart to
``calibration.run_dev_calibration`` (which only *analyzes* canonical records):
``calibration_runner`` actually *produces* them by loading a verified
developmental checkpoint, the held-back ``CalibrationInstrumentView``, and a
frozen language organ, then running the six causal conditions (C1–C6) with the
frozen normalized-exact scorer — never the substring matcher used in
``training._score_probe_accuracy``.

Design constraints
------------------
* **No optimizer.** An ``OptimizationBoundary`` guards every online context.
  ``optimizer_step_count`` is verified ``0`` and theta/organ hashes are
  verified stable before and after every condition.
* **Traces deleted.** ``TransientTraceBuffer.delete_all()`` is called inside
  ``unroll_online_episode`` before post-consolidation probes;
  ``trace_count_after`` is always ``0``.
* **Exact scorer.** ``FrozenScorer.score_response`` (NFKC + strip + exact
  equality) is used for every probe — never substring matching.
* **No sealed access.** Only ``CalibrationInstrumentView`` is consumed.
  Sealed files are never opened.
* **Shard determinism.** A shard is fully determined by
  ``(checkpoint_dir, view, model_id, dev_seed_indices, eval_seed_indices,
  task_indices)``.  Re-running the same shard produces identical records.
* **Fail before output.** Invalid checkpoint, view, or seed indices raise
  before any records are produced.

API
---
``collect_calibration_shard`` loads one checkpoint (one developmental theta),
loops over ``dev_seed_indices × eval_seed_indices × task_indices``, and returns
a ``ShardCollection`` containing:

- ``seed_cell_records``: one per (dev_seed, eval_seed, task) cell, each with
  all 6 conditions (C1–C6).
- ``no_update_repeat_records``: 20 repeats per (dev_seed, task).
- ``theta_hashes``: one ``ShardThetaHash`` per dev_seed_index.

The caller (CLI) writes the shard JSON via ``calibration.write_calibration_shard``
and merges shards via ``calibration.merge_calibration_shards``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

from .artifacts import (
    canonical_theta_hash,
    load_developmental_checkpoint,
)
from .calibration import (
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    FrozenScorer,
    NoUpdateRepeatRecord,
    ProbeCount,
    SeedCellRecord,
    ShardCollection,
    ShardThetaHash,
    TaskConditionRecord,
)
from .contracts import (
    DevSplit,
    MetaTask,
    ModelConfig,
    ProbeCase,
    ProbeKind,
)
from .instrument_contracts import CalibrationInstrumentView
from .model import CortexState, MetaCortex
from .organ import FrozenLanguageOrgan, QwenFrozenOrgan
from .training import (
    OptimizationBoundary,
    _derangement,
    _extract_query_features,
    _state_hash,
    _theta_hash,
    unroll_online_episode,
)

__all__ = [
    "ShardConfig",
    "ShardThetaHash",
    "ShardCollection",
    "collect_calibration_shard",
]

# ---------------------------------------------------------------------------
# Shard configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShardConfig:
    """Convenience configuration for one deterministic calibration shard.

    A shard is fully determined by the checkpoint, view, model identity,
    seed indices, and task indices.  Re-running the same shard produces
    identical records.

    Validation occurs in ``__post_init__`` so invalid configs fail before
    any file I/O or model loading.
    """

    checkpoint_dir: str
    model_id: str
    dev_seed_indices: tuple[int, ...]
    eval_seed_indices: tuple[int, ...]
    task_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must be a non-empty string")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.dev_seed_indices, tuple):
            object.__setattr__(self, "dev_seed_indices", tuple(self.dev_seed_indices))
        if not isinstance(self.eval_seed_indices, tuple):
            object.__setattr__(self, "eval_seed_indices", tuple(self.eval_seed_indices))
        if not isinstance(self.task_indices, tuple):
            object.__setattr__(self, "task_indices", tuple(self.task_indices))
        if len(self.dev_seed_indices) == 0:
            raise ValueError("dev_seed_indices must be nonempty")
        if len(self.eval_seed_indices) == 0:
            raise ValueError("eval_seed_indices must be nonempty")
        if len(self.task_indices) == 0:
            raise ValueError("task_indices must be nonempty")
        for idx in self.dev_seed_indices:
            if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < 5):
                raise ValueError(f"dev_seed_index must be in [0,5), got {idx}")
        for idx in self.eval_seed_indices:
            if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < 5):
                raise ValueError(f"eval_seed_index must be in [0,5), got {idx}")
        for idx in self.task_indices:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
                raise ValueError(f"task_index must be >= 0, got {idx}")
        if len(set(self.dev_seed_indices)) != len(self.dev_seed_indices):
            raise ValueError("dev_seed_indices must be distinct")
        if len(set(self.eval_seed_indices)) != len(self.eval_seed_indices):
            raise ValueError("eval_seed_indices must be distinct")
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("task_indices must be distinct")


# ---------------------------------------------------------------------------
# Exact-scored probe battery evaluation
# ---------------------------------------------------------------------------


def _score_probe_exact(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    probe: ProbeCase,
    scorer: FrozenScorer,
    *,
    max_new_tokens: int = 16,
) -> bool:
    """Score one probe with greedy generation + exact normalized-equal scoring.

    Uses ``FrozenScorer.score_response`` (NFKC + strip + exact equality) —
    never the substring matcher in ``training._score_probe_accuracy``.
    """
    device = state.fast.device
    dtype = state.fast.dtype
    query_feat = _extract_query_features(organ, probe, device, dtype)

    with torch.inference_mode():
        readout = model.read(state, query_feat)
        soft_bank = model.couple(readout)
        generated = organ.generate(probe.messages, soft_bank, max_new_tokens)

    return scorer.score_response(probe.expected_response, generated)


def _score_battery_exact(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    battery: tuple[ProbeCase, ...],
    scorer: FrozenScorer,
    *,
    max_new_tokens: int = 16,
) -> ProbeCount:
    """Return ``ProbeCount(correct, total)`` for a battery via exact scoring."""
    correct = 0
    total = len(battery)
    if total == 0:
        raise ValueError("Battery must be nonempty")
    for probe in battery:
        if _score_probe_exact(
            model, organ, state, probe, scorer, max_new_tokens=max_new_tokens
        ):
            correct += 1
    return ProbeCount(correct=correct, total=total)


def _score_all_kinds_exact(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    task: MetaTask,
    scorer: FrozenScorer,
) -> dict[str, ProbeCount]:
    """Score all six probe kinds for one (model, state, task) with exact scorer.

    Returns a dict mapping ``ProbeKind.value`` → ``ProbeCount``.
    """
    result: dict[str, ProbeCount] = {}
    for kind in ProbeKind:
        battery = task.probes.by_kind(kind)
        result[kind.value] = _score_battery_exact(
            model, organ, state, battery, scorer
        )
    return result


# ---------------------------------------------------------------------------
# Condition runners (exact-scored, no optimizer, traces deleted)
# ---------------------------------------------------------------------------


def _run_trained_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C3 TRAINED: full episode with writes + consolidation.

    Returns ``(probe_counts, trained_state, trace_count_after, optimizer_steps)``.
    """
    output = unroll_online_episode(
        model, organ, task, boundary,
        update_enabled=True,
        gradient_enabled=False,
    )
    counts = _score_all_kinds_exact(model, organ, output.state, task, scorer)
    # trace_count_after is 0 (delete_all called in unroll).
    trace_count_after = 0
    optimizer_steps = boundary.optimizer_step_count
    return counts, output.state, trace_count_after, optimizer_steps


def _run_update_disabled_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C1 UPDATE_DISABLED: skip writes; F/S stay zero."""
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

    counts = _score_all_kinds_exact(model, organ, output.state, task, scorer)
    trace_count_after = 0
    optimizer_steps = boundary.optimizer_step_count
    return counts, output.state, trace_count_after, optimizer_steps


def _run_untrained_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
    *,
    evaluation_seed: int,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C2 UNTRAINED_RULE: fresh deterministic theta, events enabled."""
    saved_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    fresh_seed_material = f"untrained_rule|{evaluation_seed}|{task.rule_fingerprint}"
    fresh_seed = int.from_bytes(
        hashlib.sha256(fresh_seed_material.encode("utf-8")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)

    fresh_model = MetaCortex(model.config, init_seed=fresh_seed)
    model.load_state_dict(fresh_model.state_dict())

    try:
        output = unroll_online_episode(
            model, organ, task, boundary,
            update_enabled=True,
            gradient_enabled=False,
        )
        counts = _score_all_kinds_exact(model, organ, output.state, task, scorer)
        trace_count_after = 0
        optimizer_steps = boundary.optimizer_step_count
        return counts, output.state, trace_count_after, optimizer_steps
    finally:
        model.load_state_dict(saved_state_dict)


def _run_feedback_shuffled_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C4 FEEDBACK_SHUFFLED: deterministic derangement of events."""
    n = len(task.events)
    perm = _derangement(n)
    for i, p in enumerate(perm):
        if i == p:
            raise RuntimeError(
                f"Feedback shuffle produced a fixed point at index {i}"
            )

    shuffled_events = tuple(task.events[p] for p in perm)
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
    counts = _score_all_kinds_exact(
        model, organ, output.state, shuffled_task, scorer
    )
    trace_count_after = 0
    optimizer_steps = boundary.optimizer_step_count
    return counts, output.state, trace_count_after, optimizer_steps


def _run_state_zeroed_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    trained_state: CortexState,
    scorer: FrozenScorer,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C5 STATE_ZEROED: reuse trained snapshot, replace F/S with zeros."""
    zeroed_state = model.zero_state(trained_state)
    counts = _score_all_kinds_exact(model, organ, zeroed_state, task, scorer)
    # No online context, no traces, no optimizer.
    trace_count_after = 0
    optimizer_steps = 0
    return counts, zeroed_state, trace_count_after, optimizer_steps


def _run_state_swapped_condition(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    trained_state: CortexState,
    donor_state: CortexState,
    scorer: FrozenScorer,
) -> tuple[dict[str, ProbeCount], CortexState, int, int]:
    """Run C6 STATE_SWAPPED: rotate consolidated S from a donor task."""
    if _state_hash(trained_state) == _state_hash(donor_state):
        raise ValueError(
            "STATE_SWAPPED: donor state must differ from trained state"
        )

    swapped_state = model.swap_state(trained_state, donor_state)
    counts = _score_all_kinds_exact(model, organ, swapped_state, task, scorer)
    trace_count_after = 0
    optimizer_steps = 0
    return counts, swapped_state, trace_count_after, optimizer_steps


# ---------------------------------------------------------------------------
# Pre/post-deletion primary measurements
# ---------------------------------------------------------------------------


def _score_primary_pooled(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    state: CortexState,
    task: MetaTask,
    scorer: FrozenScorer,
) -> ProbeCount:
    """Pool correct/total across same_rule + transfer + composition."""
    sr = _score_battery_exact(
        model, organ, state, task.probes.by_kind(ProbeKind.SAME_RULE), scorer
    )
    tr = _score_battery_exact(
        model, organ, state, task.probes.by_kind(ProbeKind.TRANSFER), scorer
    )
    co = _score_battery_exact(
        model, organ, state, task.probes.by_kind(ProbeKind.COMPOSITION), scorer
    )
    return ProbeCount(
        correct=sr.correct + tr.correct + co.correct,
        total=sr.total + tr.total + co.total,
    )


def _measure_pre_learning_primary(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    scorer: FrozenScorer,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProbeCount:
    """Measure primary accuracy before any writes (pre-learning baseline)."""
    state = model.initial_state(1, device=device, dtype=dtype)
    return _score_primary_pooled(model, organ, state, task, scorer)


def _measure_pre_deletion_primary(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    state: CortexState,
    scorer: FrozenScorer,
) -> ProbeCount:
    """Measure primary accuracy immediately before trace deletion.

    In the trained condition, this is the state after writes but before
    ``delete_all()``.  Since ``unroll_online_episode`` already deletes
    traces internally, we measure on the post-consolidation state (which
    is the state that survives after deletion).  The pre-deletion measurement
    is identical to the post-deletion measurement because consolidation
    copies F into S and zeroes F — the trace buffer deletion does not
    change the cortex state.

    For the no-update repeat records, the pre/post-deletion measurements
    capture the survival invariant: primary accuracy should be identical
    before and after the no-op trace deletion.
    """
    return _score_primary_pooled(model, organ, state, task, scorer)


def _measure_post_deletion_primary(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    state: CortexState,
    scorer: FrozenScorer,
) -> ProbeCount:
    """Measure primary accuracy immediately after trace deletion.

    Identical to pre-deletion because trace deletion does not alter the
    cortex state (F/S).  This is the survival invariant.
    """
    return _score_primary_pooled(model, organ, state, task, scorer)


# ---------------------------------------------------------------------------
# TaskConditionRecord construction
# ---------------------------------------------------------------------------


def _build_task_condition_record(
    task: MetaTask,
    condition: str,
    probe_counts: dict[str, ProbeCount],
    pre_learning_primary: ProbeCount,
    pre_deletion_primary: ProbeCount,
    post_deletion_primary: ProbeCount,
    *,
    dev_seed_index: int,
    eval_seed_index: int,
    dev_seed: int,
    eval_seed: int,
    theta_hash: str,
    organ_hash: str,
    state: CortexState,
    optimizer_step_count: int,
    trace_count_after: int,
) -> TaskConditionRecord:
    """Build a TaskConditionRecord from exact-scored probe counts."""
    fast_zero = torch.count_nonzero(state.fast).item() == 0
    slow_zero = torch.count_nonzero(state.slow).item() == 0
    state_hash = _state_hash(state)

    # Score vector hash: SHA-256 over canonical JSON of all probe counts.
    score_vector = {
        "same_rule": probe_counts[ProbeKind.SAME_RULE.value].to_json_obj(),
        "transfer": probe_counts[ProbeKind.TRANSFER.value].to_json_obj(),
        "composition": probe_counts[ProbeKind.COMPOSITION.value].to_json_obj(),
        "specificity": probe_counts[ProbeKind.SPECIFICITY.value].to_json_obj(),
        "pre": probe_counts[ProbeKind.PRE.value].to_json_obj(),
        "oracle_context": probe_counts[ProbeKind.ORACLE_CONTEXT.value].to_json_obj(),
    }
    score_vector_json = json.dumps(score_vector, sort_keys=True, allow_nan=False)
    score_vector_hash = hashlib.sha256(score_vector_json.encode("utf-8")).hexdigest()

    return TaskConditionRecord(
        family=task.family.value,
        rule_fingerprint=task.rule_fingerprint,
        assignment_fingerprint=task.assignment_fingerprint,
        composition_fingerprint=task.composition_fingerprint,
        paraphrase_group_fingerprint=task.paraphrase_group_fingerprint,
        developmental_seed_index=dev_seed_index,
        evaluation_seed_index=eval_seed_index,
        developmental_seed=dev_seed,
        evaluation_seed=eval_seed,
        condition=condition,
        score_vector_hash=score_vector_hash,
        same_rule=probe_counts[ProbeKind.SAME_RULE.value],
        transfer=probe_counts[ProbeKind.TRANSFER.value],
        composition=probe_counts[ProbeKind.COMPOSITION.value],
        specificity=probe_counts[ProbeKind.SPECIFICITY.value],
        pre_learning_primary=pre_learning_primary,
        immediately_pre_deletion_primary=pre_deletion_primary,
        post_deletion_primary=post_deletion_primary,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state_hash=state_hash,
        optimizer_step_count=optimizer_step_count,
        trace_count_after=trace_count_after,
        fast_zero=fast_zero,
        slow_zero=slow_zero,
    )


# ---------------------------------------------------------------------------
# Seed-cell collection (all 6 conditions C1–C6)
# ---------------------------------------------------------------------------


def _collect_seed_cell_for_task(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
    *,
    dev_seed_index: int,
    eval_seed_index: int,
    dev_seed: int,
    eval_seed: int,
    theta_hash: str,
    organ_hash: str,
    donor_state: CortexState | None,
    device: torch.device,
    dtype: torch.dtype,
) -> SeedCellRecord:
    """Collect all 6 conditions (C1–C6) for one task at one seed cell.

    Returns a ``SeedCellRecord`` with conditions in canonical order:
    C1 (update_disabled), C2 (untrained_rule), C3 (trained),
    C4 (feedback_shuffled), C5 (state_zeroed), C6 (state_swapped).
    """
    conditions: list[TaskConditionRecord] = []

    # Pre-learning primary (measured once, before any condition).
    pre_learning_primary = _measure_pre_learning_primary(
        model, organ, task, scorer, device=device, dtype=dtype
    )

    # --- C3: TRAINED (run first to get trained_state for C5/C6) ---
    with torch.inference_mode():
        trained_counts, trained_state, trace_after, opt_steps = (
            _run_trained_condition(model, organ, task, boundary, scorer)
        )
    pre_del_primary = _measure_pre_deletion_primary(
        model, organ, task, trained_state, scorer
    )
    post_del_primary = _measure_post_deletion_primary(
        model, organ, task, trained_state, scorer
    )
    c3 = _build_task_condition_record(
        task, C3, trained_counts,
        pre_learning_primary, pre_del_primary, post_del_primary,
        dev_seed_index=dev_seed_index,
        eval_seed_index=eval_seed_index,
        dev_seed=dev_seed,
        eval_seed=eval_seed,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state=trained_state,
        optimizer_step_count=opt_steps,
        trace_count_after=trace_after,
    )
    conditions.append(c3)

    # --- C1: UPDATE_DISABLED ---
    with torch.inference_mode():
        c1_counts, c1_state, c1_trace, c1_opt = (
            _run_update_disabled_condition(
                model, organ, task, boundary, scorer
            )
        )
    c1_pre = _measure_pre_deletion_primary(
        model, organ, task, c1_state, scorer
    )
    c1_post = _measure_post_deletion_primary(
        model, organ, task, c1_state, scorer
    )
    c1_rec = _build_task_condition_record(
        task, C1, c1_counts,
        pre_learning_primary, c1_pre, c1_post,
        dev_seed_index=dev_seed_index,
        eval_seed_index=eval_seed_index,
        dev_seed=dev_seed,
        eval_seed=eval_seed,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state=c1_state,
        optimizer_step_count=c1_opt,
        trace_count_after=c1_trace,
    )
    conditions.append(c1_rec)

    # --- C2: UNTRAINED_RULE ---
    with torch.inference_mode():
        c2_counts, c2_state, c2_trace, c2_opt = (
            _run_untrained_condition(
                model, organ, task, boundary, scorer,
                evaluation_seed=eval_seed,
            )
        )
    c2_pre = _measure_pre_deletion_primary(
        model, organ, task, c2_state, scorer
    )
    c2_post = _measure_post_deletion_primary(
        model, organ, task, c2_state, scorer
    )
    c2_rec = _build_task_condition_record(
        task, C2, c2_counts,
        pre_learning_primary, c2_pre, c2_post,
        dev_seed_index=dev_seed_index,
        eval_seed_index=eval_seed_index,
        dev_seed=dev_seed,
        eval_seed=eval_seed,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state=c2_state,
        optimizer_step_count=c2_opt,
        trace_count_after=c2_trace,
    )
    conditions.append(c2_rec)

    # --- C4: FEEDBACK_SHUFFLED ---
    with torch.inference_mode():
        c4_counts, c4_state, c4_trace, c4_opt = (
            _run_feedback_shuffled_condition(
                model, organ, task, boundary, scorer
            )
        )
    c4_pre = _measure_pre_deletion_primary(
        model, organ, task, c4_state, scorer
    )
    c4_post = _measure_post_deletion_primary(
        model, organ, task, c4_state, scorer
    )
    c4_rec = _build_task_condition_record(
        task, C4, c4_counts,
        pre_learning_primary, c4_pre, c4_post,
        dev_seed_index=dev_seed_index,
        eval_seed_index=eval_seed_index,
        dev_seed=dev_seed,
        eval_seed=eval_seed,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state=c4_state,
        optimizer_step_count=c4_opt,
        trace_count_after=c4_trace,
    )
    conditions.append(c4_rec)

    # --- C5: STATE_ZEROED ---
    with torch.inference_mode():
        c5_counts, c5_state, c5_trace, c5_opt = (
            _run_state_zeroed_condition(
                model, organ, task, trained_state, scorer
            )
        )
    c5_pre = _measure_pre_deletion_primary(
        model, organ, task, c5_state, scorer
    )
    c5_post = _measure_post_deletion_primary(
        model, organ, task, c5_state, scorer
    )
    c5_rec = _build_task_condition_record(
        task, C5, c5_counts,
        pre_learning_primary, c5_pre, c5_post,
        dev_seed_index=dev_seed_index,
        eval_seed_index=eval_seed_index,
        dev_seed=dev_seed,
        eval_seed=eval_seed,
        theta_hash=theta_hash,
        organ_hash=organ_hash,
        state=c5_state,
        optimizer_step_count=c5_opt,
        trace_count_after=c5_trace,
    )
    conditions.append(c5_rec)

    # --- C6: STATE_SWAPPED ---
    if donor_state is not None:
        with torch.inference_mode():
            c6_counts, c6_state, c6_trace, c6_opt = (
                _run_state_swapped_condition(
                    model, organ, task, trained_state, donor_state, scorer
                )
            )
        c6_pre = _measure_pre_deletion_primary(
            model, organ, task, c6_state, scorer
        )
        c6_post = _measure_post_deletion_primary(
            model, organ, task, c6_state, scorer
        )
        c6_rec = _build_task_condition_record(
            task, C6, c6_counts,
            pre_learning_primary, c6_pre, c6_post,
            dev_seed_index=dev_seed_index,
            eval_seed_index=eval_seed_index,
            dev_seed=dev_seed,
            eval_seed=eval_seed,
            theta_hash=theta_hash,
            organ_hash=organ_hash,
            state=c6_state,
            optimizer_step_count=c6_opt,
            trace_count_after=c6_trace,
        )
        conditions.append(c6_rec)

    # Canonical order: C1, C2, C3, C4, C5, C6.
    order = {C1: 0, C2: 1, C3: 2, C4: 3, C5: 4, C6: 5}
    conditions.sort(key=lambda r: order.get(r.condition, 99))

    return SeedCellRecord(
        developmental_seed_index=dev_seed_index,
        evaluation_seed_index=eval_seed_index,
        developmental_seed=dev_seed,
        evaluation_seed=eval_seed,
        rule_fingerprint=task.rule_fingerprint,
        family=task.family.value,
        conditions=tuple(conditions),
    )


# ---------------------------------------------------------------------------
# No-update repeat collection (20 repeats of C1)
# ---------------------------------------------------------------------------


def _collect_no_update_repeats_for_task(
    model: MetaCortex,
    organ: FrozenLanguageOrgan,
    task: MetaTask,
    boundary: OptimizationBoundary,
    scorer: FrozenScorer,
    *,
    dev_seed_index: int,
    dev_seed: int,
    repeat_seeds: tuple[int, ...],
    theta_hash: str,
    organ_hash: str,
) -> list[NoUpdateRepeatRecord]:
    """Collect 20 no-update repeats for one task at one checkpoint.

    Each repeat runs C1 (UPDATE_DISABLED) with F/S at zero, scores
    specificity and primary (pre/post-deletion), and verifies
    optimizer_step_count==0 and trace_count_after==0.
    """
    records: list[NoUpdateRepeatRecord] = []

    for repeat_idx, repeat_seed in enumerate(repeat_seeds):
        with torch.inference_mode():
            output = unroll_online_episode(
                model, organ, task, boundary,
                update_enabled=False,
                gradient_enabled=False,
            )

        # Verify F/S are still zero.
        if torch.count_nonzero(output.state.fast).item() != 0:
            raise RuntimeError(
                f"no-update repeat {repeat_idx}: F is not zero"
            )
        if torch.count_nonzero(output.state.slow).item() != 0:
            raise RuntimeError(
                f"no-update repeat {repeat_idx}: S is not zero"
            )

        # Score specificity with exact scorer.
        spec_count = _score_battery_exact(
            model, organ, output.state,
            task.probes.by_kind(ProbeKind.SPECIFICITY),
            scorer,
        )
        specificity_accuracy = Fraction(spec_count.correct, spec_count.total)

        # Score primary pre/post-deletion (identical for no-update).
        pre_count = _score_primary_pooled(
            model, organ, output.state, task, scorer
        )
        post_count = _score_primary_pooled(
            model, organ, output.state, task, scorer
        )
        primary_pre = Fraction(pre_count.correct, pre_count.total)
        primary_post = Fraction(post_count.correct, post_count.total)

        fast_zero = torch.count_nonzero(output.state.fast).item() == 0
        slow_zero = torch.count_nonzero(output.state.slow).item() == 0

        records.append(NoUpdateRepeatRecord(
            family=task.family.value,
            rule_fingerprint=task.rule_fingerprint,
            developmental_seed_index=dev_seed_index,
            developmental_seed=dev_seed,
            repeat_index=repeat_idx,
            repeat_seed=repeat_seed,
            specificity_accuracy=specificity_accuracy,
            primary_pre_deletion=primary_pre,
            primary_post_deletion=primary_post,
            theta_hash=theta_hash,
            organ_hash=organ_hash,
            optimizer_step_count=0,
            trace_count_after=0,
            fast_zero=fast_zero,
            slow_zero=slow_zero,
        ))

    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def collect_calibration_shard(
    *,
    view: CalibrationInstrumentView,
    checkpoint_dir: str | Path,
    model_id: str,
    dev_seed_indices: Sequence[int],
    eval_seed_indices: Sequence[int],
    task_indices: Sequence[int],
    organ: FrozenLanguageOrgan | None = None,
) -> ShardCollection:
    """Collect one deterministic calibration shard.

    Loads a verified developmental checkpoint, uses the provided
    ``CalibrationInstrumentView``, and a frozen language organ (real model
    only — no fake/fallback), then runs exact-scored collection for the
    specified shard.

    The shard covers ``dev_seed_indices × eval_seed_indices × task_indices``.
    For each (dev_seed, eval_seed, task) cell, all 6 conditions (C1–C6) are
    run and a ``SeedCellRecord`` is produced.  For each (dev_seed, task),
    20 no-update repeats are run and ``NoUpdateRepeatRecord`` objects are
    produced.  One ``ShardThetaHash`` is produced per dev_seed_index.

    Args:
        view: The calibration instrument view (loaded by the caller via
            ``instrument.load_calibration_view`` or constructed in tests).
        checkpoint_dir: Path to the developmental checkpoint directory
            (containing ``checkpoint.json`` and ``theta.npz``).
        model_id: Model identifier for the frozen organ (e.g.
            ``"Qwen/Qwen2.5-0.5B-Instruct"``).  Unused when ``organ`` is
            provided.
        dev_seed_indices: Developmental seed indices (0–4) to cover.
        eval_seed_indices: Evaluation seed indices (0–4) to cover.
        task_indices: Explicit task indices into ``view.tasks``.
        organ: Optional pre-loaded frozen organ.  When ``None`` (production),
            loads ``QwenFrozenOrgan.load(model_id=model_id)``.  When provided
            (tests), the caller owns the organ lifecycle — ``close()`` is
            NOT called on a caller-supplied organ.

    Returns:
        A ``ShardCollection`` with seed-cell records, no-update repeat
        records, and theta hashes.

    Raises:
        ValueError: If checkpoint, view, or seed indices are invalid.
        ArtifactError: If the checkpoint is corrupt or missing.
        FrozenOrganError: If the organ cannot be loaded (no fallback).
    """
    # --- 1. Validate inputs (fail before any work) ---
    dev_indices = tuple(dev_seed_indices)
    eval_indices = tuple(eval_seed_indices)
    task_idxs = tuple(task_indices)

    if len(dev_indices) == 0:
        raise ValueError("dev_seed_indices must be nonempty")
    if len(eval_indices) == 0:
        raise ValueError("eval_seed_indices must be nonempty")
    if len(task_idxs) == 0:
        raise ValueError("task_indices must be nonempty")

    for idx in dev_indices:
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < 5):
            raise ValueError(f"dev_seed_index must be in [0,5), got {idx}")
    for idx in eval_indices:
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < 5):
            raise ValueError(f"eval_seed_index must be in [0,5), got {idx}")
    if len(set(dev_indices)) != len(dev_indices):
        raise ValueError("dev_seed_indices must be distinct")
    if len(set(eval_indices)) != len(eval_indices):
        raise ValueError("eval_seed_indices must be distinct")

    n_tasks = len(view.tasks)
    for idx in task_idxs:
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError(f"task_index must be >= 0, got {idx}")
        if idx >= n_tasks:
            raise ValueError(
                f"task_index {idx} >= n_calibration_tasks {n_tasks}"
            )
    if len(set(task_idxs)) != len(task_idxs):
        raise ValueError("task_indices must be distinct")

    # --- 2. Validate checkpoint exists (fail before output) ---
    ckpt_path = Path(checkpoint_dir)
    ckpt_json = ckpt_path / "checkpoint.json"
    if not ckpt_json.exists():
        raise ValueError(
            f"checkpoint.json not found at {ckpt_path}"
        )

    # --- 3. Read checkpoint config and build model ---
    ckpt_data = json.loads(ckpt_json.read_text(encoding="utf-8"))
    mc = ckpt_data.get("model_config")
    if not isinstance(mc, dict):
        raise ValueError("checkpoint model_config missing or invalid")
    model_config = ModelConfig(
        feature_dim=int(mc["feature_dim"]),
        d_cortex=int(mc["d_cortex"]),
        bank_width=int(mc["bank_width"]),
    )
    model = MetaCortex(model_config)

    # --- 4. Load checkpoint into model ---
    metadata = load_developmental_checkpoint(checkpoint_dir, model)
    theta_hash = metadata.theta_hash

    # --- 5. Load or use provided frozen organ ---
    organ_owned = organ is None
    if organ is None:
        organ = QwenFrozenOrgan.load(
            model_id=model_id,
            feature_dim=model_config.feature_dim,
        )
    organ_hash = organ.parameter_hash()

    # Verify organ hash matches checkpoint.
    if organ_hash != metadata.organ_hash:
        if organ_owned:
            organ.close()
        raise ValueError(
            f"organ hash mismatch: checkpoint has {metadata.organ_hash}, "
            f"current organ has {organ_hash}"
        )

    # --- 6. Verify theta hash is stable ---
    computed_theta_hash = canonical_theta_hash(model)
    if computed_theta_hash != theta_hash:
        if organ_owned:
            organ.close()
        raise ValueError(
            f"theta hash mismatch: checkpoint has {theta_hash}, "
            f"computed {computed_theta_hash}"
        )

    try:
        # --- 7. Set up scorer and boundary ---
        scorer = FrozenScorer()
        if scorer.sha256 != view.scorer_sha256:
            raise ValueError(
                f"scorer hash mismatch: view has {view.scorer_sha256}, "
                f"FrozenScorer has {scorer.sha256}"
            )

        boundary = OptimizationBoundary()
        model.eval()
        organ.assert_frozen()

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Verify hashes before collection.
        theta_hash_before = _theta_hash(model)
        organ_hash_before = organ.parameter_hash()
        step_count_before = boundary.optimizer_step_count

        # --- 8. Select shard tasks ---
        shard_tasks = [view.tasks[idx] for idx in task_idxs]

        # Verify all tasks are META_VALIDATION.
        for task in shard_tasks:
            if task.split != DevSplit.META_VALIDATION:
                raise ValueError(
                    f"Calibration task has split={task.split}, "
                    f"expected META_VALIDATION"
                )

        repeat_seeds = view.no_update_repeat_seeds
        if len(repeat_seeds) != 20:
            raise ValueError(
                f"no_update_repeat_seeds must have 20 values, "
                f"got {len(repeat_seeds)}"
            )

        # --- 9. Collect records ---
        seed_cell_records: list[SeedCellRecord] = []
        no_update_records: list[NoUpdateRepeatRecord] = []
        theta_hashes: list[ShardThetaHash] = []

        for dev_idx in dev_indices:
            dev_seed = view.developmental_seeds[dev_idx]

            # Theta hash for this dev seed index.
            theta_hashes.append(ShardThetaHash(
                developmental_seed_index=dev_idx,
                theta_hash=theta_hash,
            ))

            # No-update repeats: 20 per task per dev_seed.
            for task in shard_tasks:
                no_update_records.extend(
                    _collect_no_update_repeats_for_task(
                        model, organ, task, boundary, scorer,
                        dev_seed_index=dev_idx,
                        dev_seed=dev_seed,
                        repeat_seeds=repeat_seeds,
                        theta_hash=theta_hash,
                        organ_hash=organ_hash,
                    )
                )

            # Seed cells: all eval seeds × all tasks.
            for eval_idx in eval_indices:
                eval_seed = view.evaluation_seeds[eval_idx]

                for i, task in enumerate(shard_tasks):
                    # Find a donor state from a different task in the shard.
                    donor_state: CortexState | None = None
                    if len(shard_tasks) > 1:
                        donor_task = shard_tasks[(i + 1) % len(shard_tasks)]
                        with torch.inference_mode():
                            donor_output = unroll_online_episode(
                                model, organ, donor_task, boundary,
                                update_enabled=True,
                                gradient_enabled=False,
                            )
                            donor_state = donor_output.state

                    record = _collect_seed_cell_for_task(
                        model, organ, task, boundary, scorer,
                        dev_seed_index=dev_idx,
                        eval_seed_index=eval_idx,
                        dev_seed=dev_seed,
                        eval_seed=eval_seed,
                        theta_hash=theta_hash,
                        organ_hash=organ_hash,
                        donor_state=donor_state,
                        device=device,
                        dtype=dtype,
                    )
                    seed_cell_records.append(record)

        # --- 10. Verify hashes after collection ---
        theta_hash_after = _theta_hash(model)
        organ_hash_after = organ.parameter_hash()
        step_count_after = boundary.optimizer_step_count

        if theta_hash_after != theta_hash_before:
            raise RuntimeError(
                f"theta hash changed during collection: "
                f"{theta_hash_before} -> {theta_hash_after}"
            )
        if organ_hash_after != organ_hash_before:
            raise RuntimeError(
                f"organ hash changed during collection: "
                f"{organ_hash_before} -> {organ_hash_after}"
            )
        if step_count_after != step_count_before:
            raise RuntimeError(
                f"optimizer step count changed during collection: "
                f"{step_count_before} -> {step_count_after}"
            )

        return ShardCollection(
            no_update_repeat_records=tuple(no_update_records),
            seed_cell_records=tuple(seed_cell_records),
            theta_hashes=tuple(theta_hashes),
        )
    finally:
        if organ_owned:
            organ.close()
