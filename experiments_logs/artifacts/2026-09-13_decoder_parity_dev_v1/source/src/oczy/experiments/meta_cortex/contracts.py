"""Pure-Python contracts for the DEV-only meta-cortex experiment.

This module owns every enum, dataclass, config, and result type so that
task generation, model code, training, artifacts, and CLI share one schema
without import cycles.  It imports nothing from ``eval/v2``,
``organism_curriculum``, PyTorch, or any driver.

All result serialization uses ``json.dumps(..., allow_nan=False,
sort_keys=True)`` and rejects explicit ``None``/wrong-type values on load
rather than silently defaulting.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from dataclasses import fields as dc_fields
from typing import Any

__all__ = [
    # Constants
    "DEV_SCHEMA",
    "TASKGEN_SCHEMA",
    "CORTEX_DIM",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_BANK_WIDTH",
    # Enums
    "TaskFamily",
    "DevSplit",
    "ProbeKind",
    "OutcomeCode",
    "DevCondition",
    # Frozen task/config dataclasses
    "DialogueMessage",
    "LearningEvent",
    "ProbeCase",
    "ProbeBattery",
    "MetaTask",
    "DevTaskCatalog",
    "TaskGeneratorConfig",
    "ModelConfig",
    "OuterLoopConfig",
    # JSON-safe audit/result dataclasses
    "SplitFirewallAudit",
    "OnlineEpisodeAudit",
    "LossBreakdown",
    "ConditionResult",
    "DevValidationResult",
    "DevTrainingResult",
    "CheckpointMetadata",
    # Exceptions
    "ContractError",
    # Serialization helpers
    "canonical_json",
    "from_json_obj",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEV_SCHEMA = "oczy/meta-cortex/dev/v1"
TASKGEN_SCHEMA = "oczy/meta-cortex/taskgen/v1-dev"
CORTEX_DIM = 64
DEFAULT_FEATURE_DIM = 896  # Qwen2.5-0.5B hidden size
DEFAULT_BANK_WIDTH = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContractError(ValueError):
    """Raised when a contract is violated (bad config, bad split, etc.)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskFamily(enum.Enum):
    """The three immutable v1 learning families."""

    CONTEXTUAL_REMAP = "contextual_remap"
    RULE_TRANSFORMATION = "rule_transformation"
    FINITE_STATE = "finite_state"


class DevSplit(enum.Enum):
    """DEV-only splits.

    There is intentionally **no** test member.  Any string other than
    ``"meta_train"`` or ``"meta_validation"`` fails at parse time before any
    RNG is instantiated.
    """

    META_TRAIN = "meta_train"
    META_VALIDATION = "meta_validation"


class ProbeKind(enum.Enum):
    """The six nonempty probe categories."""

    PRE = "pre"
    SAME_RULE = "same_rule"
    TRANSFER = "transfer"
    COMPOSITION = "composition"
    SPECIFICITY = "specificity"
    ORACLE_CONTEXT = "oracle_context"


class OutcomeCode(enum.Enum):
    """Canonical finite outcomes used only to render the writer's outcome feature."""

    NEUTRAL = "neutral"
    CORRECTED = "corrected"
    CONFIRMED = "confirmed"


class DevCondition(enum.Enum):
    """DEV diagnostic conditions — not C0–C9 scientific results."""

    ORGAN_ONLY = "organ_only"
    UPDATE_DISABLED = "update_disabled"
    UNTRAINED_RULE = "untrained_rule"
    TRAINED = "trained"
    FEEDBACK_SHUFFLED = "feedback_shuffled"
    STATE_ZEROED = "state_zeroed"
    STATE_SWAPPED = "state_swapped"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Serialize *obj* with ``allow_nan=False`` and ``sort_keys=True``.

    Raises ``ValueError`` if the object contains ``NaN``/``Infinity``.
    """
    return json.dumps(obj, allow_nan=False, sort_keys=True, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Default handler for non-JSON-native types."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "__dict__") and obj.__dict__:
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return str(obj)


def _validate_no_nulls(data: Mapping[str, Any], context: str = "") -> None:
    """Reject explicit ``None`` values in *data*."""
    for key, value in data.items():
        if value is None:
            raise ContractError(
                f"Explicit null for field '{key}'{f' in {context}' if context else ''} "
                "is not allowed — contracts are fail-closed."
            )


def from_json_obj(cls: type, data: Mapping[str, Any]) -> Any:
    """Reconstruct a dataclass from a JSON-safe mapping, rejecting nulls/wrong types.

    This adapts the typed manifest parsing discipline in
    ``s19_language_orn_core.py`` without copying its sign-off/eval fields.
    """
    if not isinstance(data, Mapping):
        raise ContractError(
            f"Cannot load {cls.__name__}: expected mapping, got {type(data).__name__}"
        )
    _validate_no_nulls(data, context=cls.__name__)
    field_names = {f.name for f in dc_fields(cls)}
    unknown = set(data.keys()) - field_names
    if unknown:
        raise ContractError(
            f"Unknown fields for {cls.__name__}: {sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for f in dc_fields(cls):
        if f.name not in data:
            if f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            continue
        kwargs[f.name] = data[f.name]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Frozen task/config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DialogueMessage:
    """A single chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ContractError("DialogueMessage.role must be a non-empty string")
        if not isinstance(self.content, str):
            raise ContractError("DialogueMessage.content must be a string")


@dataclass(frozen=True, slots=True)
class LearningEvent:
    """One experience/feedback event.

    Deliberately has **no task ID field** — the information boundary forbids
    episode/task identifiers from entering the model path.
    """

    observation_messages: tuple[DialogueMessage, ...]
    attempted_behavior: str
    correction: str
    outcome: OutcomeCode

    def __post_init__(self) -> None:
        if not isinstance(self.observation_messages, tuple):
            # Allow list input by converting.
            object.__setattr__(
                self, "observation_messages", tuple(self.observation_messages)
            )
        if len(self.observation_messages) == 0:
            raise ContractError("LearningEvent requires at least one observation message")
        for msg in self.observation_messages:
            if not isinstance(msg, DialogueMessage):
                raise ContractError(
                    "LearningEvent.observation_messages must contain DialogueMessage instances"
                )
        if not isinstance(self.attempted_behavior, str) or not self.attempted_behavior:
            raise ContractError("LearningEvent.attempted_behavior must be a non-empty string")
        if not isinstance(self.correction, str) or not self.correction:
            raise ContractError("LearningEvent.correction must be a non-empty string")
        if not isinstance(self.outcome, OutcomeCode):
            raise ContractError("LearningEvent.outcome must be an OutcomeCode")


@dataclass(frozen=True, slots=True)
class ProbeCase:
    """A single probe with expected response and category."""

    messages: tuple[DialogueMessage, ...]
    expected_response: str
    kind: ProbeKind

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if len(self.messages) == 0:
            raise ContractError("ProbeCase requires at least one message")
        for msg in self.messages:
            if not isinstance(msg, DialogueMessage):
                raise ContractError("ProbeCase.messages must contain DialogueMessage instances")
        if not isinstance(self.expected_response, str) or not self.expected_response:
            raise ContractError("ProbeCase.expected_response must be a non-empty string")
        if not isinstance(self.kind, ProbeKind):
            raise ContractError("ProbeCase.kind must be a ProbeKind")


@dataclass(frozen=True, slots=True)
class ProbeBattery:
    """All six nonempty probe categories for one task.

    The constructor rejects an empty required category.
    """

    pre: tuple[ProbeCase, ...]
    same_rule: tuple[ProbeCase, ...]
    transfer: tuple[ProbeCase, ...]
    composition: tuple[ProbeCase, ...]
    specificity: tuple[ProbeCase, ...]
    oracle_context: tuple[ProbeCase, ...]

    def __post_init__(self) -> None:
        for cat_name in (
            "pre",
            "same_rule",
            "transfer",
            "composition",
            "specificity",
            "oracle_context",
        ):
            cat = getattr(self, cat_name)
            if not isinstance(cat, tuple):
                object.__setattr__(self, cat_name, tuple(cat))
            if len(getattr(self, cat_name)) == 0:
                raise ContractError(
                    f"ProbeBattery.{cat_name} must be nonempty"
                )
            for case in getattr(self, cat_name):
                if not isinstance(case, ProbeCase):
                    raise ContractError(
                        f"ProbeBattery.{cat_name} must contain ProbeCase instances"
                    )

    def by_kind(self, kind: ProbeKind) -> tuple[ProbeCase, ...]:
        """Return the probe tuple for *kind*."""
        return getattr(self, kind.value)  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class MetaTask:
    """A complete meta-learning task.

    Fingerprints are audit metadata and must **never** be accepted by model
    methods — no model-facing API takes a fingerprint argument.
    """

    family: TaskFamily
    split: DevSplit
    events: tuple[LearningEvent, ...]
    probes: ProbeBattery
    rule_fingerprint: str
    assignment_fingerprint: str
    composition_fingerprint: str
    paraphrase_group_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, TaskFamily):
            raise ContractError("MetaTask.family must be a TaskFamily")
        if not isinstance(self.split, DevSplit):
            raise ContractError("MetaTask.split must be a DevSplit")
        if not isinstance(self.events, tuple):
            object.__setattr__(self, "events", tuple(self.events))
        n = len(self.events)
        if n < 2 or n > 5:
            raise ContractError(
                f"MetaTask must have 2–5 events, got {n}"
            )
        for ev in self.events:
            if not isinstance(ev, LearningEvent):
                raise ContractError("MetaTask.events must contain LearningEvent instances")
        if not isinstance(self.probes, ProbeBattery):
            raise ContractError("MetaTask.probes must be a ProbeBattery")
        for fp_name in (
            "rule_fingerprint",
            "assignment_fingerprint",
            "composition_fingerprint",
            "paraphrase_group_fingerprint",
        ):
            fp = getattr(self, fp_name)
            if not isinstance(fp, str) or not fp:
                raise ContractError(f"MetaTask.{fp_name} must be a non-empty string")
            if len(fp) != 64:
                raise ContractError(
                    f"MetaTask.{fp_name} must be a 64-char SHA-256 hex digest"
                )


@dataclass(frozen=True, slots=True)
class TaskGeneratorConfig:
    """Configuration for deterministic task generation.

    Enforces ``1 <= min_events <= max_events <= 5``.
    """

    root_seed: int
    train_tasks_per_family: int
    validation_tasks_per_family: int
    min_events: int = 2
    max_events: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ContractError("root_seed must be a non-negative int")
        if not isinstance(self.train_tasks_per_family, int) or self.train_tasks_per_family <= 0:
            raise ContractError("train_tasks_per_family must be a positive int")
        if not isinstance(self.validation_tasks_per_family, int) or self.validation_tasks_per_family <= 0:
            raise ContractError("validation_tasks_per_family must be a positive int")
        if not isinstance(self.min_events, int) or not isinstance(self.max_events, int):
            raise ContractError("min_events and max_events must be ints")
        if self.min_events < 1:
            raise ContractError("min_events must be >= 1")
        if self.min_events > self.max_events:
            raise ContractError("min_events must be <= max_events")
        if self.max_events > 5:
            raise ContractError("max_events must be <= 5")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model architecture configuration.

    Rejects ``d_cortex != 64``, nonpositive dimensions, non-float32
    execution, or dynamic bank-width changes.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    d_cortex: int = CORTEX_DIM
    bank_width: int = DEFAULT_BANK_WIDTH

    def __post_init__(self) -> None:
        if not isinstance(self.feature_dim, int) or self.feature_dim <= 0:
            raise ContractError("feature_dim must be a positive int")
        if not isinstance(self.d_cortex, int) or self.d_cortex <= 0:
            raise ContractError("d_cortex must be a positive int")
        if self.d_cortex != CORTEX_DIM:
            raise ContractError(
                f"d_cortex must be {CORTEX_DIM} (got {self.d_cortex}); "
                "v1 cortex is fixed at 64×64"
            )
        if not isinstance(self.bank_width, int) or self.bank_width <= 0:
            raise ContractError("bank_width must be a positive int")


@dataclass(frozen=True, slots=True)
class OuterLoopConfig:
    """Outer-loop training configuration.

    Optimizer choices: ``"adamw"`` or ``"sgd"``, compared only on
    meta-validation.
    """

    outer_steps: int
    tasks_per_step: int
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    validation_interval: int = 1
    generation_interval: int = 1
    behavior_weight: float = 1.0
    specificity_weight: float = 0.5
    survival_weight: float = 0.5
    state_norm_weight: float = 0.01
    seed: int = 0

    _VALID_OPTIMIZERS = ("adamw", "sgd")

    def __post_init__(self) -> None:
        if not isinstance(self.outer_steps, int) or self.outer_steps <= 0:
            raise ContractError("outer_steps must be a positive int")
        if not isinstance(self.tasks_per_step, int) or self.tasks_per_step <= 0:
            raise ContractError("tasks_per_step must be a positive int")
        if self.optimizer_name not in self._VALID_OPTIMIZERS:
            raise ContractError(
                f"optimizer_name must be one of {self._VALID_OPTIMIZERS}, "
                f"got '{self.optimizer_name}'"
            )
        for fname in ("learning_rate", "weight_decay", "grad_clip_norm",
                       "behavior_weight", "specificity_weight",
                       "survival_weight", "state_norm_weight"):
            val = getattr(self, fname)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ContractError(f"{fname} must be a real number")
            if not math.isfinite(val):
                raise ContractError(f"{fname} must be finite")
        if self.learning_rate <= 0:
            raise ContractError("learning_rate must be positive")
        if self.grad_clip_norm <= 0:
            raise ContractError("grad_clip_norm must be positive")
        if not isinstance(self.validation_interval, int) or self.validation_interval <= 0:
            raise ContractError("validation_interval must be a positive int")
        if not isinstance(self.generation_interval, int) or self.generation_interval <= 0:
            raise ContractError("generation_interval must be a positive int")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError("seed must be a non-negative int")


# ---------------------------------------------------------------------------
# DevTaskCatalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevTaskCatalog:
    """The complete DEV task catalog with split firewall audit.

    ``tasks_for`` accepts only a ``DevSplit`` — there is no arbitrary
    string/test accessor.
    """

    meta_train: tuple[MetaTask, ...]
    meta_validation: tuple[MetaTask, ...]
    catalog_sha256: str
    split_audit: SplitFirewallAudit

    def __post_init__(self) -> None:
        if not isinstance(self.meta_train, tuple):
            object.__setattr__(self, "meta_train", tuple(self.meta_train))
        if not isinstance(self.meta_validation, tuple):
            object.__setattr__(self, "meta_validation", tuple(self.meta_validation))
        for task in self.meta_train:
            if not isinstance(task, MetaTask):
                raise ContractError("meta_train must contain MetaTask instances")
            if task.split != DevSplit.META_TRAIN:
                raise ContractError(
                    f"meta_train contains a task with split={task.split.value}"
                )
        for task in self.meta_validation:
            if not isinstance(task, MetaTask):
                raise ContractError("meta_validation must contain MetaTask instances")
            if task.split != DevSplit.META_VALIDATION:
                raise ContractError(
                    f"meta_validation contains a task with split={task.split.value}"
                )
        if not isinstance(self.catalog_sha256, str) or len(self.catalog_sha256) != 64:
            raise ContractError("catalog_sha256 must be a 64-char hex digest")
        if not isinstance(self.split_audit, SplitFirewallAudit):
            raise ContractError("split_audit must be a SplitFirewallAudit")

    def tasks_for(self, split: DevSplit) -> tuple[MetaTask, ...]:
        """Return tasks for *split*.  Accepts only a ``DevSplit`` enum."""
        if not isinstance(split, DevSplit):
            raise ContractError(
                f"tasks_for accepts only DevSplit, got {type(split).__name__}"
            )
        if split is DevSplit.META_TRAIN:
            return self.meta_train
        return self.meta_validation


# ---------------------------------------------------------------------------
# JSON-safe audit/result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitFirewallAudit:
    """Split firewall audit: counts/digests per family/split plus overlap counts.

    ``passed`` is ``True`` only when all four overlap counts are zero.
    """

    train_rule_digests: tuple[str, ...]
    validation_rule_digests: tuple[str, ...]
    train_assignment_digests: tuple[str, ...]
    validation_assignment_digests: tuple[str, ...]
    train_composition_digests: tuple[str, ...]
    validation_composition_digests: tuple[str, ...]
    train_paraphrase_digests: tuple[str, ...]
    validation_paraphrase_digests: tuple[str, ...]
    rule_overlap: int
    assignment_overlap: int
    composition_overlap: int
    paraphrase_overlap: int
    train_task_count: int
    validation_task_count: int

    @property
    def passed(self) -> bool:
        """True only when all four overlap counts are zero."""
        return (
            self.rule_overlap == 0
            and self.assignment_overlap == 0
            and self.composition_overlap == 0
            and self.paraphrase_overlap == 0
        )

    def to_json(self) -> str:
        return canonical_json(
            {
                "train_rule_digests": list(self.train_rule_digests),
                "validation_rule_digests": list(self.validation_rule_digests),
                "train_assignment_digests": list(self.train_assignment_digests),
                "validation_assignment_digests": list(self.validation_assignment_digests),
                "train_composition_digests": list(self.train_composition_digests),
                "validation_composition_digests": list(self.validation_composition_digests),
                "train_paraphrase_digests": list(self.train_paraphrase_digests),
                "validation_paraphrase_digests": list(self.validation_paraphrase_digests),
                "rule_overlap": self.rule_overlap,
                "assignment_overlap": self.assignment_overlap,
                "composition_overlap": self.composition_overlap,
                "paraphrase_overlap": self.paraphrase_overlap,
                "train_task_count": self.train_task_count,
                "validation_task_count": self.validation_task_count,
                "passed": self.passed,
            }
        )


@dataclass(frozen=True, slots=True)
class OnlineEpisodeAudit:
    """Audit of a single online episode.

    Contains no event/task text or feature tensors — only hashes, counts,
    shapes, and flags.
    """

    family: str
    split: str
    rule_fingerprint: str
    event_count: int
    trace_objects_before: int
    trace_objects_after: int
    trace_feature_tensors_before: int
    trace_feature_tensors_after: int
    fast_shape: tuple[int, ...]
    slow_shape: tuple[int, ...]
    bank_shape: tuple[int, ...]
    fast_zero: bool
    logical_persistent_bytes: int
    optimizer_step_count_before: int
    optimizer_step_count_after: int
    theta_hash_before: str
    theta_hash_after: str
    organ_hash_before: str
    organ_hash_after: str
    answer_path_prompt_hashes: tuple[str, ...]
    banned_content_absent: bool

    def to_json(self) -> str:
        return canonical_json(
            {
                "family": self.family,
                "split": self.split,
                "rule_fingerprint": self.rule_fingerprint,
                "event_count": self.event_count,
                "trace_objects_before": self.trace_objects_before,
                "trace_objects_after": self.trace_objects_after,
                "trace_feature_tensors_before": self.trace_feature_tensors_before,
                "trace_feature_tensors_after": self.trace_feature_tensors_after,
                "fast_shape": list(self.fast_shape),
                "slow_shape": list(self.slow_shape),
                "bank_shape": list(self.bank_shape),
                "fast_zero": self.fast_zero,
                "logical_persistent_bytes": self.logical_persistent_bytes,
                "optimizer_step_count_before": self.optimizer_step_count_before,
                "optimizer_step_count_after": self.optimizer_step_count_after,
                "theta_hash_before": self.theta_hash_before,
                "theta_hash_after": self.theta_hash_after,
                "organ_hash_before": self.organ_hash_before,
                "organ_hash_after": self.organ_hash_after,
                "answer_path_prompt_hashes": list(self.answer_path_prompt_hashes),
                "banned_content_absent": self.banned_content_absent,
            }
        )


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Loss breakdown for the outer objective."""

    behavior: float
    specificity: float
    consolidation_survival: float
    state_norm: float
    weighted_total: float

    def __post_init__(self) -> None:
        for fname in ("behavior", "specificity", "consolidation_survival",
                       "state_norm", "weighted_total"):
            val = getattr(self, fname)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ContractError(f"LossBreakdown.{fname} must be a real number")
            if not math.isfinite(val):
                raise ContractError(f"LossBreakdown.{fname} must be finite")

    def to_json(self) -> str:
        return canonical_json(
            {
                "behavior": self.behavior,
                "specificity": self.specificity,
                "consolidation_survival": self.consolidation_survival,
                "state_norm": self.state_norm,
                "weighted_total": self.weighted_total,
            }
        )


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """Result of one DEV causal condition.

    Contains per-probe-kind correct/total/accuracy, state hash/norm/bytes,
    and the episode audit.  No verdict or sign-off field.
    """

    condition: str
    per_kind_correct: tuple[int, ...]  # ordered by ProbeKind enum
    per_kind_total: tuple[int, ...]
    per_kind_accuracy: tuple[float, ...]
    state_hash: str
    state_norm: float
    state_bytes: int
    episode_audit: OnlineEpisodeAudit

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or not self.condition:
            raise ContractError("ConditionResult.condition must be a non-empty string")
        n = len(ProbeKind)
        if len(self.per_kind_correct) != n or len(self.per_kind_total) != n or len(self.per_kind_accuracy) != n:
            raise ContractError(
                f"per_kind_* tuples must have length {n} (one per ProbeKind)"
            )
        if not isinstance(self.episode_audit, OnlineEpisodeAudit):
            raise ContractError("ConditionResult.episode_audit must be an OnlineEpisodeAudit")

    def to_json(self) -> str:
        return canonical_json(
            {
                "condition": self.condition,
                "per_kind_correct": list(self.per_kind_correct),
                "per_kind_total": list(self.per_kind_total),
                "per_kind_accuracy": list(self.per_kind_accuracy),
                "state_hash": self.state_hash,
                "state_norm": self.state_norm,
                "state_bytes": self.state_bytes,
                "episode_audit": json.loads(self.episode_audit.to_json()),
            }
        )


@dataclass(frozen=True, slots=True)
class DevValidationResult:
    """DEV validation result: per-family and pooled condition results plus
    the four causal DEV deltas.

    It has **no** verdict or sign-off field.
    """

    per_family_results: tuple[tuple[str, tuple[ConditionResult, ...]], ...]
    pooled_results: tuple[ConditionResult, ...]
    trained_vs_update_disabled_delta: float
    trained_vs_untrained_delta: float
    trained_vs_shuffled_delta: float
    trained_vs_zeroed_delta: float
    trained_vs_swapped_delta: float

    def to_json(self) -> str:
        return canonical_json(
            {
                "per_family_results": [
                    {
                        "family": fam,
                        "conditions": [json.loads(cr.to_json()) for cr in conds],
                    }
                    for fam, conds in self.per_family_results
                ],
                "pooled_results": [json.loads(cr.to_json()) for cr in self.pooled_results],
                "trained_vs_update_disabled_delta": self.trained_vs_update_disabled_delta,
                "trained_vs_untrained_delta": self.trained_vs_untrained_delta,
                "trained_vs_shuffled_delta": self.trained_vs_shuffled_delta,
                "trained_vs_zeroed_delta": self.trained_vs_zeroed_delta,
                "trained_vs_swapped_delta": self.trained_vs_swapped_delta,
            }
        )


@dataclass(frozen=True, slots=True)
class DevTrainingResult:
    """DEV training result.

    Contains fingerprints/metrics only, never prompts, corrections, labels,
    targets, events, or feature tensors.
    """

    schema: str
    model_config_digest: str
    taskgen_config_digest: str
    outer_config_digest: str
    catalog_digest: str
    step_curve: tuple[tuple[int, float, float], ...]  # (step, train_loss, val_score)
    best_validation_step: int
    best_validation_score: float
    validation_result: DevValidationResult
    organ_hash_before: str
    organ_hash_after: str
    optimizer_step_count: int
    audit_status: str  # "ok" or "failed_audit"

    def __post_init__(self) -> None:
        if self.schema != DEV_SCHEMA:
            raise ContractError(f"DevTrainingResult.schema must be '{DEV_SCHEMA}'")
        if self.audit_status not in ("ok", "failed_audit"):
            raise ContractError("audit_status must be 'ok' or 'failed_audit'")
        if not isinstance(self.validation_result, DevValidationResult):
            raise ContractError("validation_result must be a DevValidationResult")

    def to_json(self) -> str:
        return canonical_json(
            {
                "schema": self.schema,
                "model_config_digest": self.model_config_digest,
                "taskgen_config_digest": self.taskgen_config_digest,
                "outer_config_digest": self.outer_config_digest,
                "catalog_digest": self.catalog_digest,
                "step_curve": [list(row) for row in self.step_curve],
                "best_validation_step": self.best_validation_step,
                "best_validation_score": self.best_validation_score,
                "validation_result": json.loads(self.validation_result.to_json()),
                "organ_hash_before": self.organ_hash_before,
                "organ_hash_after": self.organ_hash_after,
                "optimizer_step_count": self.optimizer_step_count,
                "audit_status": self.audit_status,
            }
        )


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Developmental checkpoint metadata.

    No fabricated provenance default; use explicit ``"unavailable"`` where
    necessary, following the fail-honestly intent of R19.
    """

    schema: str
    model_config: ModelConfig
    taskgen_schema: str
    taskgen_digest: str
    outer_config: OuterLoopConfig
    completed_step: int
    best_step: int
    validation_score: float
    parameter_count: int
    parameter_bytes: int
    theta_hash: str
    organ_identity: str
    organ_hash: str
    source_provenance: str = "unavailable"

    def __post_init__(self) -> None:
        if self.schema != DEV_SCHEMA:
            raise ContractError(f"CheckpointMetadata.schema must be '{DEV_SCHEMA}'")
        if not isinstance(self.model_config, ModelConfig):
            raise ContractError("model_config must be a ModelConfig")
        if self.taskgen_schema != TASKGEN_SCHEMA:
            raise ContractError(f"taskgen_schema must be '{TASKGEN_SCHEMA}'")
        if not isinstance(self.outer_config, OuterLoopConfig):
            raise ContractError("outer_config must be an OuterLoopConfig")
        if not isinstance(self.completed_step, int) or self.completed_step < 0:
            raise ContractError("completed_step must be a non-negative int")
        if not isinstance(self.best_step, int) or self.best_step < 0:
            raise ContractError("best_step must be a non-negative int")
        if not isinstance(self.parameter_count, int) or self.parameter_count <= 0:
            raise ContractError("parameter_count must be a positive int")
        if not isinstance(self.parameter_bytes, int) or self.parameter_bytes <= 0:
            raise ContractError("parameter_bytes must be a positive int")
        if not isinstance(self.validation_score, (int, float)) or isinstance(self.validation_score, bool):
            raise ContractError("validation_score must be a real number")
        if not math.isfinite(self.validation_score):
            raise ContractError("validation_score must be finite")
        if not isinstance(self.source_provenance, str) or not self.source_provenance:
            raise ContractError("source_provenance must be a non-empty string")

    def to_json(self) -> str:
        return canonical_json(
            {
                "schema": self.schema,
                "model_config": {
                    "feature_dim": self.model_config.feature_dim,
                    "d_cortex": self.model_config.d_cortex,
                    "bank_width": self.model_config.bank_width,
                },
                "taskgen_schema": self.taskgen_schema,
                "taskgen_digest": self.taskgen_digest,
                "outer_config": {
                    "outer_steps": self.outer_config.outer_steps,
                    "tasks_per_step": self.outer_config.tasks_per_step,
                    "optimizer_name": self.outer_config.optimizer_name,
                    "learning_rate": self.outer_config.learning_rate,
                    "weight_decay": self.outer_config.weight_decay,
                    "grad_clip_norm": self.outer_config.grad_clip_norm,
                    "validation_interval": self.outer_config.validation_interval,
                    "generation_interval": self.outer_config.generation_interval,
                    "behavior_weight": self.outer_config.behavior_weight,
                    "specificity_weight": self.outer_config.specificity_weight,
                    "survival_weight": self.outer_config.survival_weight,
                    "state_norm_weight": self.outer_config.state_norm_weight,
                    "seed": self.outer_config.seed,
                },
                "completed_step": self.completed_step,
                "best_step": self.best_step,
                "validation_score": self.validation_score,
                "parameter_count": self.parameter_count,
                "parameter_bytes": self.parameter_bytes,
                "theta_hash": self.theta_hash,
                "organ_identity": self.organ_identity,
                "organ_hash": self.organ_hash,
                "source_provenance": self.source_provenance,
            }
        )
