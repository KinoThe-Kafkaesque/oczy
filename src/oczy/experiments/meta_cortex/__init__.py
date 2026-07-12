"""DEV-only meta-cortex experiment package (Research/20).

This package implements the DEV-only developmental training and validation
pipeline for the meta-trained cortex with a frozen language organ.  It is
intentionally scoped to ``train-dev``, ``validate-dev``, and ``audit-dev``
commands only — no meta-test generator, manifest, sign-off, or scientific
verdict is provided.
"""

from __future__ import annotations

from .artifacts import (
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
from .contracts import (
    CORTEX_DIM,
    DEFAULT_BANK_WIDTH,
    DEFAULT_FEATURE_DIM,
    DEV_SCHEMA,
    TASKGEN_SCHEMA,
    CheckpointMetadata,
    ConditionResult,
    ContractError,
    DevCondition,
    DevSplit,
    DevTaskCatalog,
    DevTrainingResult,
    DevValidationResult,
    DialogueMessage,
    LearningEvent,
    LossBreakdown,
    MetaTask,
    ModelConfig,
    OnlineEpisodeAudit,
    OutcomeCode,
    OuterLoopConfig,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    SplitFirewallAudit,
    TaskFamily,
    TaskGeneratorConfig,
)
from .model import CortexState, EventFeatureBatch, MetaCortex
from .organ import FrozenLanguageOrgan, FrozenOrganError, QwenFrozenOrgan
from .taskgen import build_dev_catalog
from .training import OuterTrainer, run_dev_validation

__all__ = [
    # Constants
    "CORTEX_DIM",
    "DEFAULT_BANK_WIDTH",
    "DEFAULT_FEATURE_DIM",
    "DEV_SCHEMA",
    "TASKGEN_SCHEMA",
    # Enums
    "DevCondition",
    "DevSplit",
    "OutcomeCode",
    "ProbeKind",
    "TaskFamily",
    # Contracts
    "CheckpointMetadata",
    "ConditionResult",
    "ContractError",
    "DevTaskCatalog",
    "DevTrainingResult",
    "DevValidationResult",
    "DialogueMessage",
    "LearningEvent",
    "LossBreakdown",
    "MetaTask",
    "ModelConfig",
    "OnlineEpisodeAudit",
    "OuterLoopConfig",
    "ProbeBattery",
    "ProbeCase",
    "SplitFirewallAudit",
    "TaskGeneratorConfig",
    # Model
    "CortexState",
    "EventFeatureBatch",
    "MetaCortex",
    # Organ
    "FrozenLanguageOrgan",
    "FrozenOrganError",
    "QwenFrozenOrgan",
    # Taskgen
    "build_dev_catalog",
    # Training
    "OuterTrainer",
    "run_dev_validation",
    # Artifacts
    "ArtifactError",
    "canonical_state_hash",
    "canonical_theta_hash",
    "load_dev_persistent_state",
    "load_developmental_checkpoint",
    "read_dev_result",
    "save_dev_persistent_state",
    "save_developmental_checkpoint",
    "write_dev_result",
]
