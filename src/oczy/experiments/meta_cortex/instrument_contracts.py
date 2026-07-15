"""Instrument contracts for the INT8 ``meta_cortex/v2`` evaluation instrument.

This module is separate from ``contracts.py`` (which stays DEV-only).
It owns the schema constants, strict serialization helpers, and frozen
value types needed by ``instrument.py`` (candidate materialization) and
``authorization.py`` (detached signoff).

Nothing here is exported through the package ``__init__.py`` DEV surface.
The module imports only from ``contracts.py`` (enums, ``ContractError``,
``from_json_obj``) and the standard library — no PyTorch, model, or driver
imports.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass
from dataclasses import fields as dc_fields
from typing import Any

from .contracts import (
    ContractError,
    DevSplit,
    DevTaskCatalog,
    MetaTask,
)

__all__ = [
    # Schema constants
    "INSTRUMENT_DEFINITION_SCHEMA",
    "DEV_VIEW_SCHEMA",
    "CALIBRATION_VIEW_SCHEMA",
    "CANDIDATE_MANIFEST_SCHEMA",
    "SIGNOFF_SCHEMA",
    "TASK_RECORD_SCHEMA",
    "PROMPT_SCHEMA",
    "SCORER_SCHEMA",
    "ENDPOINT_SCHEMA",
    "INSTRUMENT_ID",
    "INSTRUMENT_VERSION",
    # Strict serialization
    "strict_canonical_json",
    "strict_json_loads",
    "canonical_decimal",
    "validate_sha256_hex",
    "validate_relative_path",
    "validate_signoff_id",
    # Frozen types
    "InstrumentFileEntry",
    "InstrumentBinding",
    "DevInstrumentView",
    "CalibrationInstrumentView",
    "InstrumentDefinitionConfig",
    "InstrumentDefinition",
    "CandidateManifest",
    "SignoffAttestation",
]


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

INSTRUMENT_ID = "meta_cortex/v2"
INSTRUMENT_VERSION = "v2"

# Serialization layouts are unchanged from v1.  The instrument identity is
# v2 because the frozen organ changed; schema versions describe JSON shape,
# not the scientific candidate version.

INSTRUMENT_DEFINITION_SCHEMA = "oczy/meta-cortex/instrument-definition/v1"
DEV_VIEW_SCHEMA = "oczy/meta-cortex/instrument-dev-view/v1"
CALIBRATION_VIEW_SCHEMA = "oczy/meta-cortex/instrument-calibration-view/v1"
CANDIDATE_MANIFEST_SCHEMA = "oczy/meta-cortex/instrument-candidate-manifest/v1"
SIGNOFF_SCHEMA = "oczy/meta-cortex/instrument-signoff/v1"
TASK_RECORD_SCHEMA = "oczy/meta-cortex/task-record/v1"
PROMPT_SCHEMA = "oczy/meta-cortex/prompts/v1"
SCORER_SCHEMA = "oczy/meta-cortex/scorers/v1"
ENDPOINT_SCHEMA = "oczy/meta-cortex/endpoints/v1"


# ---------------------------------------------------------------------------
# Strict serialization helpers
# ---------------------------------------------------------------------------

def _strict_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys in JSON objects."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ContractError(f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _strict_parse_constant(value: str) -> None:
    """Reject NaN, Infinity, -Infinity in JSON."""
    raise ContractError(f"JSON constant {value!r} is not allowed (NaN/Infinity rejected)")


def strict_canonical_json(obj: Any) -> bytes:
    """Serialize *obj* to canonical UTF-8 bytes.

    Uses ``sort_keys=True``, compact separators, ``ensure_ascii=False``,
    ``allow_nan=False``, no ``default=`` fallback.  This is the one
    canonical encoding for all instrument artifacts.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_json_loads(data: bytes | str) -> dict[str, Any]:
    """Parse JSON with strict discipline: no duplicates, no NaN/Inf, no defaults.

    Returns a plain ``dict``.  Rejects:
    - Duplicate keys (via ``object_pairs_hook``)
    - NaN / Infinity (via ``parse_constant``)
    - Non-object top-level (must be a mapping)
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    result = json.loads(
        data,
        object_pairs_hook=_strict_object_pairs_hook,
        parse_constant=_strict_parse_constant,
    )
    if not isinstance(result, dict):
        raise ContractError(
            f"Expected JSON object, got {type(result).__name__}"
        )
    return result


def canonical_decimal(value: str | float) -> str:
    """Return a canonical decimal string for *value*.

    Uses ``Decimal.normalize()`` with trailing fractional zeros/dot removed.
    Zero is represented as ``"0"``.  Rejects exponent notation, signs,
    whitespace, noncanonical equivalents, values outside ``[0, 1]``,
    NaN, and Infinity.
    """
    from decimal import Decimal, InvalidOperation

    if isinstance(value, str):
        s = value.strip()
        if "e" in s.lower() or "E" in s:
            raise ContractError(f"Exponent notation not allowed: {value!r}")
        if s.startswith("+"):
            raise ContractError(f"Leading '+' not allowed: {value!r}")
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError) as exc:
            raise ContractError(f"Invalid decimal: {value!r}") from exc
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float):
            import math
            if not math.isfinite(value):
                raise ContractError(f"Non-finite value: {value!r}")
        d = Decimal(str(value))
    else:
        raise ContractError(f"Cannot canonicalize {type(value).__name__} as decimal")

    if d != d:  # NaN
        raise ContractError(f"NaN not allowed: {value!r}")
    if d < 0 or d > 1:
        raise ContractError(f"Decimal must be in [0, 1], got {value!r}")

    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0" or s == "":
        s = "0"
    return s


def validate_sha256_hex(value: str, *, field_name: str = "sha256") -> str:
    """Validate that *value* is a 64-char lowercase hex SHA-256 digest."""
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string, got {type(value).__name__}")
    if len(value) != 64:
        raise ContractError(f"{field_name} must be 64 chars, got {len(value)}")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError(f"{field_name} must be lowercase hex, got: {value[:16]}...")
    return value


def validate_relative_path(value: str, *, field_name: str = "path") -> str:
    """Validate that *value* is a safe POSIX relative path."""
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field_name} must be a non-empty string")
    if "\\" in value:
        raise ContractError(f"{field_name} contains backslash: {value!r}")
    if value.startswith("/"):
        raise ContractError(f"{field_name} is absolute: {value!r}")
    if "//" in value:
        raise ContractError(f"{field_name} has duplicate separator (empty component): {value!r}")
    parts = value.split("/")
    for part in parts:
        if part == "" or part == "." or part == "..":
            raise ContractError(
                f"{field_name} has empty/dot/parent component: {value!r}"
            )
    return value


def validate_signoff_id(value: str) -> str:
    """Validate structured signoff ID format.

    Must match ``r20-meta-cortex-v<version>/<lowercase UUIDv4>`` where
    version is a positive integer and the UUID confirms version 4 and
    RFC 4122 variant.
    """
    if not isinstance(value, str) or not value:
        raise ContractError("human_signoff_id must be a non-empty string")
    pattern = r"^r20-meta-cortex-v(\d+)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
    m = re.fullmatch(pattern, value)
    if m is None:
        raise ContractError(
            f"human_signoff_id must match r20-meta-cortex-v<version>/<lowercase-uuidv4>, "
            f"got: {value!r}"
        )
    version_str, uuid_str = m.group(1), m.group(2)
    version = int(version_str)
    if version < 1:
        raise ContractError(f"signoff ID version must be positive, got {version}")
    uuid_parts = uuid_str.split("-")
    version_nibble = uuid_parts[2][0]
    if version_nibble != "4":
        raise ContractError(
            f"signoff ID UUID must be version 4, got version {version_nibble}"
        )
    variant_nibble = uuid_parts[3][0]
    if variant_nibble not in "89ab":
        raise ContractError(
            f"signoff ID UUID must be RFC 4122 variant, got variant nibble {variant_nibble!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Frozen value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InstrumentFileEntry:
    """One file in the instrument bundle."""

    path: str
    sha256: str
    size_bytes: int
    visibility: str
    role: str

    _VALID_VISIBILITY = ("public", "calibration", "evidence", "sealed")

    def __post_init__(self) -> None:
        validate_relative_path(self.path)
        validate_sha256_hex(self.sha256, field_name="sha256")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise ContractError("size_bytes must be an int")
        if self.size_bytes < 0:
            raise ContractError(f"size_bytes must be >= 0, got {self.size_bytes}")
        if self.visibility not in self._VALID_VISIBILITY:
            raise ContractError(
                f"visibility must be one of {self._VALID_VISIBILITY}, "
                f"got {self.visibility!r}"
            )
        if not isinstance(self.role, str) or not self.role:
            raise ContractError("role must be a non-empty string")

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "visibility": self.visibility,
            "role": self.role,
        }

    @classmethod
    def from_json_obj(cls, data: Mapping[str, Any]) -> InstrumentFileEntry:
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            visibility=data["visibility"],
            role=data["role"],
        )


@dataclass(frozen=True, slots=True)
class InstrumentBinding:
    """Hash bindings that tie a checkpoint/result to a specific instrument."""

    instrument_version: str
    instrument_manifest_sha256: str
    definition_sha256: str
    dev_view_sha256: str
    prompt_registry_sha256: str
    scorer_registry_sha256: str
    endpoint_registry_sha256: str
    organ_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_version, str) or not self.instrument_version:
            raise ContractError("instrument_version must be a non-empty string")
        for fname in (
            "definition_sha256",
            "dev_view_sha256",
            "prompt_registry_sha256",
            "scorer_registry_sha256",
            "endpoint_registry_sha256",
            "organ_hash",
        ):
            validate_sha256_hex(getattr(self, fname), field_name=fname)
        # instrument_manifest_sha256 may be empty at definition stage
        # (filled when the candidate is finalized).  If non-empty, it
        # must be a valid SHA-256 hex.
        if self.instrument_manifest_sha256:
            validate_sha256_hex(
                self.instrument_manifest_sha256,
                field_name="instrument_manifest_sha256",
            )

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "instrument_version": self.instrument_version,
            "instrument_manifest_sha256": self.instrument_manifest_sha256,
            "definition_sha256": self.definition_sha256,
            "dev_view_sha256": self.dev_view_sha256,
            "prompt_registry_sha256": self.prompt_registry_sha256,
            "scorer_registry_sha256": self.scorer_registry_sha256,
            "endpoint_registry_sha256": self.endpoint_registry_sha256,
            "organ_hash": self.organ_hash,
        }


@dataclass(frozen=True, slots=True)
class DevInstrumentView:
    """Public DEV instrument view: binding + train/tuning-validation catalog.

    The catalog's ``meta_validation`` contains only ``purpose=tuning``
    rules.  Calibration rules are in ``CalibrationInstrumentView``.
    Sealed meta-test data is never accessible through this type.
    """

    binding: InstrumentBinding
    catalog: DevTaskCatalog

    def __post_init__(self) -> None:
        if not isinstance(self.binding, InstrumentBinding):
            raise ContractError("binding must be an InstrumentBinding")
        if not isinstance(self.catalog, DevTaskCatalog):
            raise ContractError("catalog must be a DevTaskCatalog")


@dataclass(frozen=True, slots=True)
class CalibrationInstrumentView:
    """Held-back calibration instrument view.

    Contains only calibration-validation tasks.  Not accepted by
    ``OuterTrainer`` — calibration is a separate process.

    This is a data container satisfying the duck-typed
    ``CalibrationViewProtocol`` used by ``calibration.py``.  All hash
    fields are injected after byte verification — they are never
    self-referential.
    """

    schema: str
    instrument_id: str
    instrument_version: str
    definition_sha256: str
    calibration_view_sha256: str
    scorer_sha256: str
    endpoint_schema_sha256: str
    confidence_level: float
    target_power: float
    minimum_tasks_per_family: int
    developmental_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    no_update_repeat_seeds: tuple[int, ...]
    task_cluster_bootstrap_seed: int
    tasks: Sequence[MetaTask]
    calibration_tasks_per_family: dict[str, int]

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_VIEW_SCHEMA:
            raise ContractError(
                f"schema must be {CALIBRATION_VIEW_SCHEMA!r}, got {self.schema!r}"
            )
        if self.instrument_id != INSTRUMENT_ID:
            raise ContractError(f"instrument_id must be {INSTRUMENT_ID!r}")
        if self.instrument_version != INSTRUMENT_VERSION:
            raise ContractError(
                f"instrument_version must be {INSTRUMENT_VERSION!r}"
            )
        validate_sha256_hex(self.definition_sha256, field_name="definition_sha256")
        validate_sha256_hex(
            self.calibration_view_sha256, field_name="calibration_view_sha256"
        )
        validate_sha256_hex(self.scorer_sha256, field_name="scorer_sha256")
        validate_sha256_hex(
            self.endpoint_schema_sha256, field_name="endpoint_schema_sha256"
        )
        if not isinstance(self.confidence_level, (int, float)) or isinstance(
            self.confidence_level, bool
        ):
            raise ContractError("confidence_level must be a real number")
        if self.confidence_level != 0.95:
            raise ContractError(f"confidence_level must be 0.95, got {self.confidence_level}")
        if not isinstance(self.target_power, (int, float)) or isinstance(
            self.target_power, bool
        ):
            raise ContractError("target_power must be a real number")
        if self.target_power != 0.80:
            raise ContractError(f"target_power must be 0.80, got {self.target_power}")
        if not isinstance(self.minimum_tasks_per_family, int) or isinstance(
            self.minimum_tasks_per_family, bool
        ):
            raise ContractError("minimum_tasks_per_family must be an int")
        if self.minimum_tasks_per_family < 30:
            raise ContractError(
                f"minimum_tasks_per_family must be >= 30, got {self.minimum_tasks_per_family}"
            )
        if not isinstance(self.developmental_seeds, tuple):
            object.__setattr__(self, "developmental_seeds", tuple(self.developmental_seeds))
        if len(self.developmental_seeds) != 5:
            raise ContractError(
                f"developmental_seeds must have exactly 5 values, got {len(self.developmental_seeds)}"
            )
        if len(set(self.developmental_seeds)) != 5:
            raise ContractError("developmental_seeds must be distinct")
        if not isinstance(self.evaluation_seeds, tuple):
            object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))
        if len(self.evaluation_seeds) != 5:
            raise ContractError(
                f"evaluation_seeds must have exactly 5 values, got {len(self.evaluation_seeds)}"
            )
        if len(set(self.evaluation_seeds)) != 5:
            raise ContractError("evaluation_seeds must be distinct")
        if not isinstance(self.no_update_repeat_seeds, tuple):
            object.__setattr__(self, "no_update_repeat_seeds", tuple(self.no_update_repeat_seeds))
        if len(self.no_update_repeat_seeds) != 20:
            raise ContractError(
                f"no_update_repeat_seeds must have exactly 20 values, got {len(self.no_update_repeat_seeds)}"
            )
        if not isinstance(self.task_cluster_bootstrap_seed, int) or isinstance(
            self.task_cluster_bootstrap_seed, bool
        ):
            raise ContractError("task_cluster_bootstrap_seed must be an int")
        # Verify all tasks have split=META_VALIDATION
        for task in self.tasks:
            if not isinstance(task, MetaTask):
                raise ContractError("tasks must contain MetaTask instances")
            if task.split != DevSplit.META_VALIDATION:
                raise ContractError(
                    f"CalibrationInstrumentView tasks must have split=META_VALIDATION, "
                    f"got {task.split}"
                )


@dataclass(frozen=True, slots=True)
class InstrumentDefinitionConfig:
    """Configuration for deterministic instrument definition materialization."""

    instrument_id: str
    instrument_version: str
    root_seed: int
    train_tasks_per_family: int
    tuning_tasks_per_family: int
    calibration_tasks_per_family: int
    developmental_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    organ_model_id: str
    organ_revision: str
    organ_parameter_sha256: str
    chat_template_sha256: str
    feature_mode: str
    decoding_mode: str
    max_new_tokens: int
    feature_dim: int
    d_cortex: int
    soft_bank_width: int
    event_min: int
    event_max: int
    abstain_threshold: str
    source_commit: str
    source_archive_sha256: str

    def __post_init__(self) -> None:
        if self.instrument_id != INSTRUMENT_ID:
            raise ContractError(
                f"instrument_id must be {INSTRUMENT_ID!r}, "
                f"got {self.instrument_id!r}"
            )
        if self.instrument_version != INSTRUMENT_VERSION:
            raise ContractError(
                f"instrument_version must be {INSTRUMENT_VERSION!r}, "
                f"got {self.instrument_version!r}"
            )
        if not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ContractError("root_seed must be a non-negative int")
        for fname in (
            "train_tasks_per_family",
            "tuning_tasks_per_family",
            "calibration_tasks_per_family",
        ):
            val = getattr(self, fname)
            if not isinstance(val, int) or val <= 0:
                raise ContractError(f"{fname} must be a positive int")
        if self.calibration_tasks_per_family < 30:
            raise ContractError(
                f"calibration_tasks_per_family must be >= 30, got {self.calibration_tasks_per_family}"
            )
        if not isinstance(self.developmental_seeds, tuple):
            object.__setattr__(self, "developmental_seeds", tuple(self.developmental_seeds))
        if len(self.developmental_seeds) < 5:
            raise ContractError(
                f"developmental_seeds must have >= 5 values, got {len(self.developmental_seeds)}"
            )
        if len(set(self.developmental_seeds)) != len(self.developmental_seeds):
            raise ContractError("developmental_seeds must be distinct")
        if not isinstance(self.evaluation_seeds, tuple):
            object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))
        if len(self.evaluation_seeds) < 5:
            raise ContractError(
                f"evaluation_seeds must have >= 5 values, got {len(self.evaluation_seeds)}"
            )
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ContractError("evaluation_seeds must be distinct")
        for fname in ("organ_model_id", "organ_revision", "source_commit"):
            val = getattr(self, fname)
            if not isinstance(val, str) or not val:
                raise ContractError(f"{fname} must be a non-empty string")
        validate_sha256_hex(
            self.organ_parameter_sha256, field_name="organ_parameter_sha256"
        )
        validate_sha256_hex(
            self.chat_template_sha256, field_name="chat_template_sha256"
        )
        validate_sha256_hex(
            self.source_archive_sha256, field_name="source_archive_sha256"
        )
        if self.feature_mode not in ("final_layer_mean_pool",):
            raise ContractError(
                f"feature_mode must be 'final_layer_mean_pool', got {self.feature_mode!r}"
            )
        if self.decoding_mode not in ("greedy",):
            raise ContractError(
                f"decoding_mode must be 'greedy', got {self.decoding_mode!r}"
            )
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ContractError("max_new_tokens must be a positive int")
        if not isinstance(self.feature_dim, int) or self.feature_dim <= 0:
            raise ContractError("feature_dim must be a positive int")
        if not isinstance(self.d_cortex, int) or self.d_cortex <= 0:
            raise ContractError("d_cortex must be a positive int")
        if not isinstance(self.soft_bank_width, int) or self.soft_bank_width <= 0:
            raise ContractError("soft_bank_width must be a positive int")
        if not isinstance(self.event_min, int) or not isinstance(self.event_max, int):
            raise ContractError("event_min and event_max must be ints")
        if self.event_min < 2 or self.event_min > self.event_max or self.event_max > 5:
            raise ContractError("event_min must be >= 2, <= event_max, event_max <= 5")
        canonical_decimal(self.abstain_threshold)

    @classmethod
    def from_json_obj(cls, data: Mapping[str, Any]) -> InstrumentDefinitionConfig:
        """Construct from a JSON-safe mapping, rejecting nulls/unknowns."""
        if not isinstance(data, Mapping):
            raise ContractError(
                f"Expected mapping, got {type(data).__name__}"
            )
        field_names = {f.name for f in dc_fields(cls)}
        unknown = set(data.keys()) - field_names
        if unknown:
            raise ContractError(f"Unknown fields: {sorted(unknown)}")
        for key, value in data.items():
            if value is None:
                raise ContractError(f"Explicit null for {key!r} is not allowed")
        kwargs: dict[str, Any] = {}
        for f in dc_fields(cls):
            if f.name not in data:
                if f.default is not MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory is not MISSING:
                    kwargs[f.name] = f.default_factory()
                continue
            val = data[f.name]
            if f.name in ("developmental_seeds", "evaluation_seeds"):
                val = tuple(val)
            kwargs[f.name] = val
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class InstrumentDefinition:
    """Materialized instrument definition with all hashes and file entries."""

    schema: str
    instrument_id: str
    instrument_version: str
    lifecycle_state: str
    source_commit: str
    source_archive_sha256: str
    taskgen_schema: str
    generator_algorithm: str
    generator_source_sha256: str
    prompt_schema: str
    prompt_registry_sha256: str
    scorer_schema: str
    scorer_registry_sha256: str
    endpoint_schema: str
    endpoint_registry_sha256: str
    organ_model_id: str
    organ_revision: str
    organ_parameter_sha256: str
    chat_template_sha256: str
    feature_mode: str
    decoding_mode: str
    max_new_tokens: int
    feature_dim: int
    d_cortex: int
    soft_bank_width: int
    event_min: int
    event_max: int
    family_order: tuple[str, ...]
    train_tasks_per_family: int
    tuning_tasks_per_family: int
    calibration_tasks_per_family: int
    developmental_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    dev_seed_table_sha256: str
    meta_test_seed_sha256: str
    probe_counts_sha256: str
    dev_view_sha256: str
    calibration_view_sha256: str
    public_files: tuple[InstrumentFileEntry, ...]
    definition_sha256: str

    def __post_init__(self) -> None:
        if self.schema != INSTRUMENT_DEFINITION_SCHEMA:
            raise ContractError(
                f"schema must be {INSTRUMENT_DEFINITION_SCHEMA!r}, got {self.schema!r}"
            )
        if self.instrument_id != INSTRUMENT_ID:
            raise ContractError(f"instrument_id must be {INSTRUMENT_ID!r}")
        if self.instrument_version != INSTRUMENT_VERSION:
            raise ContractError(
                f"instrument_version must be {INSTRUMENT_VERSION!r}"
            )
        if self.lifecycle_state != "definition":
            raise ContractError(
                f"lifecycle_state must be 'definition', got {self.lifecycle_state!r}"
            )
        if self.generator_algorithm != "sha256-counter-rejection/v1":
            raise ContractError(
                "generator_algorithm must be 'sha256-counter-rejection/v1'"
            )
        for fname in (
            "source_archive_sha256",
            "generator_source_sha256",
            "prompt_registry_sha256",
            "scorer_registry_sha256",
            "endpoint_registry_sha256",
            "organ_parameter_sha256",
            "chat_template_sha256",
            "dev_seed_table_sha256",
            "meta_test_seed_sha256",
            "probe_counts_sha256",
            "dev_view_sha256",
            "calibration_view_sha256",
            "definition_sha256",
        ):
            validate_sha256_hex(getattr(self, fname), field_name=fname)
        if not isinstance(self.public_files, tuple):
            object.__setattr__(self, "public_files", tuple(self.public_files))
        for entry in self.public_files:
            if not isinstance(entry, InstrumentFileEntry):
                raise ContractError("public_files must contain InstrumentFileEntry")
        if not isinstance(self.family_order, tuple):
            object.__setattr__(self, "family_order", tuple(self.family_order))
        if not isinstance(self.developmental_seeds, tuple):
            object.__setattr__(self, "developmental_seeds", tuple(self.developmental_seeds))
        if not isinstance(self.evaluation_seeds, tuple):
            object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instrument_id": self.instrument_id,
            "instrument_version": self.instrument_version,
            "lifecycle_state": self.lifecycle_state,
            "source_commit": self.source_commit,
            "source_archive_sha256": self.source_archive_sha256,
            "taskgen_schema": self.taskgen_schema,
            "generator_algorithm": self.generator_algorithm,
            "generator_source_sha256": self.generator_source_sha256,
            "prompt_schema": self.prompt_schema,
            "prompt_registry_sha256": self.prompt_registry_sha256,
            "scorer_schema": self.scorer_schema,
            "scorer_registry_sha256": self.scorer_registry_sha256,
            "endpoint_schema": self.endpoint_schema,
            "endpoint_registry_sha256": self.endpoint_registry_sha256,
            "organ_model_id": self.organ_model_id,
            "organ_revision": self.organ_revision,
            "organ_parameter_sha256": self.organ_parameter_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "feature_mode": self.feature_mode,
            "decoding_mode": self.decoding_mode,
            "max_new_tokens": self.max_new_tokens,
            "feature_dim": self.feature_dim,
            "d_cortex": self.d_cortex,
            "soft_bank_width": self.soft_bank_width,
            "event_min": self.event_min,
            "event_max": self.event_max,
            "family_order": list(self.family_order),
            "train_tasks_per_family": self.train_tasks_per_family,
            "tuning_tasks_per_family": self.tuning_tasks_per_family,
            "calibration_tasks_per_family": self.calibration_tasks_per_family,
            "developmental_seeds": list(self.developmental_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "dev_seed_table_sha256": self.dev_seed_table_sha256,
            "meta_test_seed_sha256": self.meta_test_seed_sha256,
            "probe_counts_sha256": self.probe_counts_sha256,
            "dev_view_sha256": self.dev_view_sha256,
            "calibration_view_sha256": self.calibration_view_sha256,
            "public_files": [e.to_json_obj() for e in self.public_files],
        }


@dataclass(frozen=True, slots=True)
class CandidateManifest:
    """Finalized candidate manifest binding all instrument bytes.

    Created by ``finalize_candidate``.  Contains no signoff fields.
    The self-hash ``manifest_sha256`` excludes only itself.
    """

    schema: str
    instrument_id: str
    instrument_version: str
    lifecycle_state: str
    definition_sha256: str
    dev_view_sha256: str
    calibration_view_sha256: str
    source_commit: str
    source_archive_sha256: str
    taskgen_schema: str
    generator_algorithm: str
    generator_source_sha256: str
    prompt_schema: str
    prompt_registry_sha256: str
    scorer_schema: str
    scorer_registry_sha256: str
    endpoint_schema: str
    endpoint_registry_sha256: str
    organ_model_id: str
    organ_revision: str
    organ_parameter_sha256: str
    chat_template_sha256: str
    feature_mode: str
    decoding_mode: str
    max_new_tokens: int
    feature_dim: int
    d_cortex: int
    soft_bank_width: int
    abstain_threshold: str
    event_min: int
    event_max: int
    family_order: tuple[str, ...]
    probe_counts_by_split_family_kind: dict[str, dict[str, dict[str, int]]]
    train_tasks_per_family: int
    tuning_tasks_per_family: int
    calibration_tasks_per_family: int
    sample_size_tasks_per_family: int
    meta_test_tasks_by_family: dict[str, int]
    developmental_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    equivalence_margin: str
    calibration_report_sha256: str
    power_report_sha256: str
    calibration_holdout_accessed: bool
    independent_sample_unit: str
    leakage_audit_sha256: str
    leakage_audit_passed: bool
    meta_test_seed_sha256: str
    files: tuple[InstrumentFileEntry, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_MANIFEST_SCHEMA:
            raise ContractError(
                f"schema must be {CANDIDATE_MANIFEST_SCHEMA!r}, got {self.schema!r}"
            )
        if self.instrument_id != INSTRUMENT_ID:
            raise ContractError(f"instrument_id must be {INSTRUMENT_ID!r}")
        if self.instrument_version != INSTRUMENT_VERSION:
            raise ContractError(
                f"instrument_version must be {INSTRUMENT_VERSION!r}"
            )
        if self.lifecycle_state != "candidate":
            raise ContractError(
                f"lifecycle_state must be 'candidate', got {self.lifecycle_state!r}"
            )
        if self.generator_algorithm != "sha256-counter-rejection/v1":
            raise ContractError(
                "generator_algorithm must be 'sha256-counter-rejection/v1'"
            )
        for fname in (
            "definition_sha256",
            "dev_view_sha256",
            "calibration_view_sha256",
            "source_archive_sha256",
            "generator_source_sha256",
            "prompt_registry_sha256",
            "scorer_registry_sha256",
            "endpoint_registry_sha256",
            "organ_parameter_sha256",
            "chat_template_sha256",
            "calibration_report_sha256",
            "power_report_sha256",
            "leakage_audit_sha256",
            "meta_test_seed_sha256",
            "manifest_sha256",
        ):
            validate_sha256_hex(getattr(self, fname), field_name=fname)
        canonical_abstain = canonical_decimal(self.abstain_threshold)
        if canonical_abstain != self.abstain_threshold:
            raise ContractError(
                f"abstain_threshold must be canonical decimal, "
                f"got {self.abstain_threshold!r} (canonical: {canonical_abstain!r})"
            )
        canonical_margin = canonical_decimal(self.equivalence_margin)
        if canonical_margin != self.equivalence_margin:
            raise ContractError(
                f"equivalence_margin must be canonical decimal, "
                f"got {self.equivalence_margin!r} (canonical: {canonical_margin!r})"
            )
        if not isinstance(self.sample_size_tasks_per_family, int) or isinstance(
            self.sample_size_tasks_per_family, bool
        ):
            raise ContractError("sample_size_tasks_per_family must be an int")
        if self.sample_size_tasks_per_family < 30:
            raise ContractError(
                f"sample_size_tasks_per_family must be >= 30, "
                f"got {self.sample_size_tasks_per_family}"
            )
        if self.calibration_holdout_accessed is not False:
            raise ContractError("calibration_holdout_accessed must be False")
        if self.leakage_audit_passed is not True:
            raise ContractError("leakage_audit_passed must be True")
        if self.independent_sample_unit != "task_rule":
            raise ContractError(
                f"independent_sample_unit must be 'task_rule', "
                f"got {self.independent_sample_unit!r}"
            )
        n = self.sample_size_tasks_per_family
        _known_families = ("contextual_remap", "rule_transformation", "finite_state")
        for fam in _known_families:
            if fam not in self.meta_test_tasks_by_family:
                raise ContractError(
                    f"meta_test_tasks_by_family missing family {fam!r}"
                )
            if self.meta_test_tasks_by_family[fam] != n:
                raise ContractError(
                    f"meta_test_tasks_by_family[{fam!r}] must equal "
                    f"sample_size_tasks_per_family ({n}), "
                    f"got {self.meta_test_tasks_by_family[fam]}"
                )
        extra_families = set(self.meta_test_tasks_by_family.keys()) - set(_known_families)
        if extra_families:
            raise ContractError(
                f"meta_test_tasks_by_family has unknown family(s): "
                f"{sorted(extra_families)}"
            )
        if not isinstance(self.files, tuple):
            object.__setattr__(self, "files", tuple(self.files))
        for entry in self.files:
            if not isinstance(entry, InstrumentFileEntry):
                raise ContractError("files must contain InstrumentFileEntry")
        if not isinstance(self.family_order, tuple):
            object.__setattr__(self, "family_order", tuple(self.family_order))
        if not isinstance(self.developmental_seeds, tuple):
            object.__setattr__(self, "developmental_seeds", tuple(self.developmental_seeds))
        if not isinstance(self.evaluation_seeds, tuple):
            object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instrument_id": self.instrument_id,
            "instrument_version": self.instrument_version,
            "lifecycle_state": self.lifecycle_state,
            "definition_sha256": self.definition_sha256,
            "dev_view_sha256": self.dev_view_sha256,
            "calibration_view_sha256": self.calibration_view_sha256,
            "source_commit": self.source_commit,
            "source_archive_sha256": self.source_archive_sha256,
            "taskgen_schema": self.taskgen_schema,
            "generator_algorithm": self.generator_algorithm,
            "generator_source_sha256": self.generator_source_sha256,
            "prompt_schema": self.prompt_schema,
            "prompt_registry_sha256": self.prompt_registry_sha256,
            "scorer_schema": self.scorer_schema,
            "scorer_registry_sha256": self.scorer_registry_sha256,
            "endpoint_schema": self.endpoint_schema,
            "endpoint_registry_sha256": self.endpoint_registry_sha256,
            "organ_model_id": self.organ_model_id,
            "organ_revision": self.organ_revision,
            "organ_parameter_sha256": self.organ_parameter_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "feature_mode": self.feature_mode,
            "decoding_mode": self.decoding_mode,
            "max_new_tokens": self.max_new_tokens,
            "feature_dim": self.feature_dim,
            "d_cortex": self.d_cortex,
            "soft_bank_width": self.soft_bank_width,
            "abstain_threshold": self.abstain_threshold,
            "event_min": self.event_min,
            "event_max": self.event_max,
            "family_order": list(self.family_order),
            "probe_counts_by_split_family_kind": self.probe_counts_by_split_family_kind,
            "train_tasks_per_family": self.train_tasks_per_family,
            "tuning_tasks_per_family": self.tuning_tasks_per_family,
            "calibration_tasks_per_family": self.calibration_tasks_per_family,
            "sample_size_tasks_per_family": self.sample_size_tasks_per_family,
            "meta_test_tasks_by_family": self.meta_test_tasks_by_family,
            "developmental_seeds": list(self.developmental_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "equivalence_margin": self.equivalence_margin,
            "calibration_report_sha256": self.calibration_report_sha256,
            "power_report_sha256": self.power_report_sha256,
            "calibration_holdout_accessed": self.calibration_holdout_accessed,
            "independent_sample_unit": self.independent_sample_unit,
            "leakage_audit_sha256": self.leakage_audit_sha256,
            "leakage_audit_passed": self.leakage_audit_passed,
            "meta_test_seed_sha256": self.meta_test_seed_sha256,
            "files": [e.to_json_obj() for e in self.files],
        }


@dataclass(frozen=True, slots=True)
class SignoffAttestation:
    """Detached human signoff attestation.

    Created by ``promote_and_sign`` in ``authorization.py``.  The
    self-hash ``signoff_sha256`` excludes only itself.
    """

    schema: str
    instrument_id: str
    instrument_version: str
    lifecycle_state: str
    instrument_manifest_sha256: str
    equivalence_margin: str
    sample_size_tasks_per_family: int
    meta_test_tasks_by_family: dict[str, int]
    human_signoff_id: str
    signed_at_utc: str
    signoff_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SIGNOFF_SCHEMA:
            raise ContractError(
                f"schema must be {SIGNOFF_SCHEMA!r}, got {self.schema!r}"
            )
        if self.instrument_id != INSTRUMENT_ID:
            raise ContractError(f"instrument_id must be {INSTRUMENT_ID!r}")
        if self.instrument_version != INSTRUMENT_VERSION:
            raise ContractError(
                f"instrument_version must be {INSTRUMENT_VERSION!r}"
            )
        if self.lifecycle_state != "signed":
            raise ContractError(
                f"lifecycle_state must be 'signed', got {self.lifecycle_state!r}"
            )
        validate_sha256_hex(
            self.instrument_manifest_sha256,
            field_name="instrument_manifest_sha256",
        )
        canonical_decimal(self.equivalence_margin)
        if not isinstance(self.sample_size_tasks_per_family, int) or isinstance(
            self.sample_size_tasks_per_family, bool
        ):
            raise ContractError("sample_size_tasks_per_family must be an int")
        if self.sample_size_tasks_per_family < 30:
            raise ContractError(
                f"sample_size_tasks_per_family must be >= 30, "
                f"got {self.sample_size_tasks_per_family}"
            )
        validate_signoff_id(self.human_signoff_id)
        if not isinstance(self.signed_at_utc, str) or not self.signed_at_utc:
            raise ContractError("signed_at_utc must be a non-empty string")
        validate_sha256_hex(self.signoff_sha256, field_name="signoff_sha256")
        n = self.sample_size_tasks_per_family
        for fam in ("contextual_remap", "rule_transformation", "finite_state"):
            if fam not in self.meta_test_tasks_by_family:
                raise ContractError(
                    f"meta_test_tasks_by_family missing family {fam!r}"
                )
            if self.meta_test_tasks_by_family[fam] != n:
                raise ContractError(
                    f"meta_test_tasks_by_family[{fam!r}] must equal "
                    f"sample_size_tasks_per_family ({n}), "
                    f"got {self.meta_test_tasks_by_family[fam]}"
                )

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instrument_id": self.instrument_id,
            "instrument_version": self.instrument_version,
            "lifecycle_state": self.lifecycle_state,
            "instrument_manifest_sha256": self.instrument_manifest_sha256,
            "equivalence_margin": self.equivalence_margin,
            "sample_size_tasks_per_family": self.sample_size_tasks_per_family,
            "meta_test_tasks_by_family": self.meta_test_tasks_by_family,
            "human_signoff_id": self.human_signoff_id,
            "signed_at_utc": self.signed_at_utc,
        }
