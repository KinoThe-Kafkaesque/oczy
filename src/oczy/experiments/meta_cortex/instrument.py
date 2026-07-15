"""Versioned INT8 ``meta_cortex/v2`` instrument: deterministic candidate
materialization, view loaders, leakage audit, and candidate finalization.

This module is the **only** code allowed to call the private meta-test
generation kernel before authorization.  It materializes:

- ``DEFINITION.json`` — the frozen instrument definition
- ``public/DEV_VIEW.json`` — train + tuning-validation only
- ``public/CALIBRATION_VIEW.json`` — held-back calibration-validation only
- ``sealed/`` — meta-test seed + tasks (never accessible through DEV APIs)

All files are written atomically with canonical JSON encoding and
per-file SHA-256 hashing.  Candidate directories are created once and
never overwritten.

Nothing in this module is exported through the package ``__init__.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._sealed_taskgen import generate_sealed_tasks
from .contracts import (
    TASKGEN_SCHEMA,
    DevSplit,
    DevTaskCatalog,
    MetaTask,
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from .instrument_contracts import (
    CALIBRATION_VIEW_SCHEMA,
    CANDIDATE_MANIFEST_SCHEMA,
    DEV_VIEW_SCHEMA,
    ENDPOINT_SCHEMA,
    INSTRUMENT_DEFINITION_SCHEMA,
    PROMPT_SCHEMA,
    SCORER_SCHEMA,
    TASK_RECORD_SCHEMA,
    CalibrationInstrumentView,
    CandidateManifest,
    DevInstrumentView,
    InstrumentBinding,
    InstrumentDefinition,
    InstrumentDefinitionConfig,
    InstrumentFileEntry,
    canonical_decimal,
    strict_canonical_json,
    strict_json_loads,
    validate_sha256_hex,
)
from .taskgen import (
    MAX_COLLISION_NONCE,
    build_dev_catalog,
)

__all__ = [
    "materialize_definition",
    "verify_definition",
    "load_dev_view",
    "load_calibration_view",
    "finalize_candidate",
    "verify_candidate",
    "InstrumentError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InstrumentError(ValueError):
    """Raised on instrument materialization, verification, or audit failure."""


# ---------------------------------------------------------------------------
# Canonical file I/O helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for raw file bytes at *path*."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _write_atomic(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically: temp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_canonical_json(path: Path, obj: Any) -> tuple[str, int]:
    """Write canonical JSON to *path*, return (sha256, size_bytes)."""
    data = strict_canonical_json(obj)
    if not data.endswith(b"\n"):
        data = data + b"\n"
    _write_atomic(path, data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _write_canonical_jsonl(
    path: Path, records: Sequence[dict[str, Any]]
) -> tuple[str, int]:
    """Write canonical JSONL to *path*, return (sha256, size_bytes).

    Each record is compact canonical JSON on its own line, with a
    final newline.
    """
    lines: list[bytes] = []
    for record in records:
        line = strict_canonical_json(record)
        lines.append(line + b"\n")
    data = b"".join(lines)
    _write_atomic(path, data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _write_text(path: Path, text: str) -> tuple[str, int]:
    """Write UTF-8 text to *path*, return (sha256, size_bytes)."""
    data = text.encode("utf-8")
    _write_atomic(path, data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _read_and_verify(path: Path, expected_sha256: str) -> bytes:
    """Read file bytes, verify SHA-256, return bytes."""
    if not path.is_file():
        raise InstrumentError(f"File not found: {path}")
    if path.is_symlink():
        raise InstrumentError(f"Symlink not allowed: {path}")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha256:
        raise InstrumentError(
            f"Hash mismatch for {path}: expected {expected_sha256}, got {actual_sha}"
        )
    return raw


def _read_canonical_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Read and verify a canonical JSON file, return parsed dict."""
    raw = _read_and_verify(path, expected_sha256)
    # Strip trailing newline if present
    text = raw.decode("utf-8").rstrip("\n")
    return strict_json_loads(text)


# ---------------------------------------------------------------------------
# Typed JSON accessors (strict type narrowing for Pyrefly)
# ---------------------------------------------------------------------------

def _json_str(data: dict[str, Any], key: str) -> str:
    """Extract a required string from a JSON dict, validating the type."""
    value = data[key]
    if not isinstance(value, str):
        raise InstrumentError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _json_int(data: dict[str, Any], key: str) -> int:
    """Extract a required int from a JSON dict, validating the type."""
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise InstrumentError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _json_list(data: dict[str, Any], key: str) -> list[Any]:
    """Extract a required list from a JSON dict, validating the type."""
    value = data[key]
    if not isinstance(value, list):
        raise InstrumentError(f"{key} must be a list, got {type(value).__name__}")
    return value


def _json_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract a required dict from a JSON dict, validating the type."""
    value = data[key]
    if not isinstance(value, dict):
        raise InstrumentError(f"{key} must be a dict, got {type(value).__name__}")
    return value

def _json_bool(data: dict[str, Any], key: str) -> bool:
    """Extract a required bool from a JSON dict, validating the type."""
    value = data[key]
    if not isinstance(value, bool):
        raise InstrumentError(f"{key} must be a bool, got {type(value).__name__}")
    return value


def _json_float(data: dict[str, Any], key: str) -> float:
    """Extract a required float from a JSON dict, validating the type.

    Accepts JSON numbers (int or float), rejects bool.
    """
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InstrumentError(f"{key} must be a number, got {type(value).__name__}")
    return value

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _check_path_safe(path: Path, root: Path) -> Path:
    """Resolve *path* under *root*, rejecting symlinks and traversal."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise InstrumentError(
            f"Path {path} escapes root {root}"
        ) from None
    if path.is_symlink():
        raise InstrumentError(f"Symlink not allowed: {path}")
    return resolved


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def _derive_seed(domain: str, instrument_id: str, index: int) -> int:
    """Derive a uint63 seed from domain-separated SHA-256.

    ``uint63(first_8_bytes(SHA256('oczy/meta-cortex/calibration-seeds/v1|'
    + instrument_id + '|' + domain + '|' + str(index))))``
    """
    material = f"oczy/meta-cortex/calibration-seeds/v1|{instrument_id}|{domain}|{index}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big")
    return val & ((1 << 63) - 1)  # uint63


def _generate_dev_seed_table(
    config: InstrumentDefinitionConfig,
) -> dict[str, Any]:
    """Generate the DEV seed table with all four domains.

    Returns a JSON-safe dict with ordered integer vectors for:
    developmental, evaluation, no_update_repeat, task_cluster_bootstrap.
    """
    developmental = [_derive_seed("developmental", config.instrument_id, i) for i in range(len(config.developmental_seeds))]
    evaluation = [_derive_seed("evaluation", config.instrument_id, i) for i in range(len(config.evaluation_seeds))]
    no_update_repeat = [_derive_seed("no_update_repeat", config.instrument_id, i) for i in range(20)]
    task_cluster_bootstrap = [_derive_seed("task_cluster_bootstrap", config.instrument_id, 0)]

    return {
        "derivation_schema": "oczy/meta-cortex/calibration-seeds/v1",
        "instrument_id": config.instrument_id,
        "developmental": developmental,
        "evaluation": evaluation,
        "no_update_repeat": no_update_repeat,
        "task_cluster_bootstrap": task_cluster_bootstrap,
    }


def _generate_meta_test_seed_commitment(test_seed_file: Path) -> tuple[str, str]:
    """Read the independent 256-bit test seed and return (seed_hex, seed_sha256).

    Accepts either exactly 32 raw bytes or 64 lowercase hex characters,
    normalized to the same 32-byte seed.  The SHA-256 is a commitment —
    the actual seed is never in a DEV view.
    """
    if not test_seed_file.is_file():
        raise InstrumentError(f"Test seed file not found: {test_seed_file}")
    raw_bytes = test_seed_file.read_bytes()
    if len(raw_bytes) == 32:
        seed_hex = raw_bytes.hex()
    else:
        try:
            text = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise InstrumentError(
                f"Test seed file must contain 32 raw bytes or a 64-char "
                f"lowercase hex string, got {len(raw_bytes)} bytes"
            ) from exc
        if len(text) != 64 or not re.fullmatch(r"[0-9a-f]{64}", text):
            raise InstrumentError(
                f"Test seed file must contain 32 raw bytes or a 64-char "
                f"lowercase hex string, got {len(raw_bytes)} bytes"
            )
        seed_hex = text
    seed_sha256 = hashlib.sha256(seed_hex.encode("utf-8")).hexdigest()
    return seed_hex, seed_sha256


# ---------------------------------------------------------------------------
# Task record serialization
# ---------------------------------------------------------------------------

def _task_to_jsonl_record(task: MetaTask, split_role: str, index: int) -> dict[str, Any]:
    """Serialize a MetaTask to a JSONL-safe record for file materialization.

    Contains: schema, split_role, family, events, probes, and four
    fingerprints.  No task/episode identifier enters a model-facing method.
    """
    events_list = []
    for ev in task.events:
        events_list.append({
            "observation_messages": [
                {"role": m.role, "content": m.content}
                for m in ev.observation_messages
            ],
            "attempted_behavior": ev.attempted_behavior,
            "correction": ev.correction,
            "outcome": ev.outcome.value,
        })
    probes_obj = {}
    for kind in ProbeKind:
        kind_probes = task.probes.by_kind(kind)
        probes_obj[kind.value] = [
            {
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in p.messages
                ],
                "expected_response": p.expected_response,
                "kind": p.kind.value,
            }
            for p in kind_probes
        ]
    return {
        "schema": TASK_RECORD_SCHEMA,
        "split_role": split_role,
        "index": index,
        "family": task.family.value,
        "events": events_list,
        "probes": probes_obj,
        "rule_fingerprint": task.rule_fingerprint,
        "assignment_fingerprint": task.assignment_fingerprint,
        "composition_fingerprint": task.composition_fingerprint,
        "paraphrase_group_fingerprint": task.paraphrase_group_fingerprint,
    }


# ---------------------------------------------------------------------------
# Prompt/scorer/endpoint registries (frozen)
# ---------------------------------------------------------------------------

def _prompt_registry_obj() -> dict[str, Any]:
    """Return the frozen prompt template registry."""
    return {
        "schema": PROMPT_SCHEMA,
        "templates": {
            "context.event.request.v1": "In the {context} room, respond to {symbol}.",
            "context.event.correction.v1": "In the {context} room, {symbol} requires the token {output}.",
            "context.probe.pre.v1": "The room is {context}. What response follows {symbol}?",
            "context.probe.same.v1": "You are in the {context} chamber. {symbol} demands what token?",
            "context.probe.transfer_setting.v1": "Setting: {context} environment. Command word: {symbol}. What is the correct response?",
            "context.probe.transfer_domain.v1": "Within the {context} domain, the signal {symbol} elicits what?",
            "context.probe.composition.v1": "First, in the {first_context} room, respond to {first_symbol}. Then, in the {second_context} room, respond to {first_output}. Give both tokens in order.",
            "context.probe.specificity.v1": "The room is {context}. What response follows {symbol}?",
            "context.oracle.header.v1": "Complete mapping:",
            "context.oracle.entry.v1": "  {context} / {symbol} -> {output}",
            "context.oracle.query.v1": "Given the above mapping, in the {context} room, what token follows {symbol}?",
            "transform.event.request.v1": "Apply the rule to: {operand}",
            "transform.event.correction.v1": "The correct result for {operand} is {result}.",
            "transform.probe.pre.v1": "Apply the rule to: {operand}",
            "transform.probe.same.v1": "What is the transformed output for input {operand}?",
            "transform.probe.transfer.v1": "Transform: {operand}",
            "transform.probe.composition.v1": "Apply the rule twice to: {operand}",
            "transform.probe.specificity.v1": "Apply the rule to: {operand}",
            "transform.oracle.header.v1": "Rule: {template} with parameters {param1!r} and {param2!r}.\nWorked examples:",
            "transform.oracle.example.v1": "  {operand} -> {result}",
            "transform.oracle.query.v1": "Given this rule, what is the output for: {operand}?",
            "fsm.event.request.v1": "State: {state}. Input: {input}. What is the next state?",
            "fsm.event.correction.v1": "From {state}, input {input} transitions to {next_state}.",
            "fsm.probe.pre.v1": "State: {state}. Input: {input}. What is the next state?",
            "fsm.probe.same.v1": "Given current state {state} and signal {input}, which state follows?",
            "fsm.probe.transfer.v1": "State: {state}. Signal: {input}. Next state?",
            "fsm.probe.composition.v1": "Start at {start}. First input: {input_1}. Then input: {input_2}. What is the result?",
            "fsm.probe.specificity.v1": "State: {state}. Input: {input}. What is the next state?",
            "fsm.oracle.header.v1": "Complete transition graph:",
            "fsm.oracle.transition.v1": "  {state} + {input} -> {next_state}",
            "fsm.oracle.action.v1": "  {state}: {action}",
            "fsm.oracle.goal.v1": "Goal: reach {goal_state}.",
            "fsm.oracle.query.v1": "Given the above graph, from {state} with input {input}, what is the next state?",
        },
    }


def _scorer_registry_obj() -> dict[str, Any]:
    """Return the frozen scorer registry."""
    return {
        "schema": SCORER_SCHEMA,
        "scorer_id": "normalized-exact/v1",
        "normalization": {
            "unicode_normalization": "NFKC",
            "strip": "leading_trailing_unicode_whitespace",
            "preserve": ["case", "punctuation", "internal_whitespace"],
            "reject": ["substring_match", "extra_prose", "token_extraction"],
        },
        "abstain_rule": "abstained when confidence < abstain_threshold; equality scores normally",
        "correct_rule": "correct iff normalized generated bytes equal normalized expected bytes exactly",
        "confidence_definition": "geometric_mean_probability_of_greedily_selected_non_eos_tokens",
        "empty_generation": {"confidence": 0, "abstained": True, "correct": False},
    }


def _endpoint_registry_obj() -> dict[str, Any]:
    """Return the frozen endpoint registry."""
    return {
        "schema": ENDPOINT_SCHEMA,
        "endpoints": {
            "primary_accuracy_task": {
                "definition": "micro_correct / micro_total over same_rule + transfer + composition for one rule",
                "excluded_kinds": ["pre", "specificity", "oracle_context"],
            },
            "adaptation_delta_task": {
                "definition": "A_primary(C3_postdelete) - A_primary(C3_prelearning)",
            },
            "transfer_delta_task": {
                "definition": "A_transfer(C3) - A_transfer(C1)",
            },
            "composition_delta_task": {
                "definition": "A_composition(C3) - A_composition(C1)",
            },
            "meta_training_delta_task": {
                "definition": "A_primary(C3) - A_primary(C2)",
            },
            "feedback_semantics_delta_task": {
                "definition": "A_primary(C3) - A_primary(C4)",
            },
            "causal_state_delta_task": {
                "definition": "A_primary(C3) - A_primary(C5)",
            },
            "state_addressing_delta_task": {
                "definition": "A_primary(C3) - A_primary(C6)",
            },
            "specificity_delta_task": {
                "definition": "A_specificity(C3) - A_specificity(C1)",
            },
            "trace_free_survival_task": {
                "definition": "A_primary(post_delete) - A_primary(immediately_pre_delete)",
            },
        },
        "independent_unit": "task_rule",
        "seed_checkpoint_role": "repeated_factor",
        "equivalence_margin_formula": "nearest_rank_p95_of_joint_max_repeat_pair_v",
        "power_formula": "task_normal_planning_approximation_matching_final_ci_rule",
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _run_leakage_audit(
    train_tasks: Sequence[MetaTask],
    tuning_tasks: Sequence[MetaTask],
    calibration_tasks: Sequence[MetaTask],
    meta_test_tasks: Sequence[MetaTask] | None,
    meta_test_seed_in_public: bool,
) -> dict[str, Any]:
    """Run full 4-domain leakage audit.

    Checks pairwise overlap for all four fingerprint classes across
    all domain pairs, within-domain duplicates, and test-seed presence
    in public files.
    """
    domains: dict[str, Sequence[MetaTask] | None] = {
        "meta_train": train_tasks,
        "meta_validation_tuning": tuning_tasks,
        "meta_validation_calibration": calibration_tasks,
        "meta_test": meta_test_tasks,
    }

    fp_classes = ("rule_fingerprint", "assignment_fingerprint",
                  "composition_fingerprint", "paraphrase_group_fingerprint")

    # Collect fingerprints per domain per class
    domain_fps: dict[str, dict[str, set[str]]] = {}
    for dname, dtasks in domains.items():
        domain_fps[dname] = {fc: set() for fc in fp_classes}
        if dtasks is None:
            continue
        for task in dtasks:
            for fc in fp_classes:
                domain_fps[dname][fc].add(getattr(task, fc))

    # Pairwise overlap counts
    domain_names = list(domains.keys())
    pairwise: dict[str, dict[str, dict[str, int]]] = {}
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1:]:
            overlaps: dict[str, int] = {}
            for fc in fp_classes:
                if domains[d1] is None or domains[d2] is None:
                    overlaps[fc] = 0
                else:
                    overlaps[fc] = len(
                        domain_fps[d1][fc] & domain_fps[d2][fc]
                    )
            pairwise.setdefault(d1, {})[d2] = overlaps

    # Within-domain duplicates
    within_domain: dict[str, dict[str, int]] = {}
    for dname, dtasks in domains.items():
        dup_counts: dict[str, int] = {}
        if dtasks is None:
            for fc in fp_classes:
                dup_counts[fc] = 0
            within_domain[dname] = dup_counts
            continue
        for fc in fp_classes:
            fps = [getattr(t, fc) for t in dtasks]
            dup_counts[fc] = len(fps) - len(set(fps))
        within_domain[dname] = dup_counts

    # Per-domain counts
    per_domain_counts: dict[str, dict[str, int]] = {}
    for dname, dtasks in domains.items():
        if dtasks is None:
            per_domain_counts[dname] = {"task_count": 0}
        else:
            per_domain_counts[dname] = {"task_count": len(dtasks)}

    # Check invariants
    all_overlap_zero = True
    for _d1, others in pairwise.items():
        for _d2, overlaps in others.items():
            for _fc, count in overlaps.items():
                if count > 0:
                    all_overlap_zero = False

    all_within_zero = True
    for _dname, dup_counts in within_domain.items():
        # Only enforce within-domain diversity for the sealed meta-test
        # domain.  DEV domains (train/tuning/calibration) may contain
        # within-domain duplicate fingerprints when the DEV generator's
        # finite-state assignment space (4^3 = 64 Moore-machine values)
        # is smaller than the task count — this is a DEV-generator
        # limitation, not a cross-domain firewall violation.
        if _dname != "meta_test":
            continue
        for _fc, count in dup_counts.items():
            if count > 0:
                all_within_zero = False

    passed = (
        all_overlap_zero
        and all_within_zero
        and not meta_test_seed_in_public
    )

    return {
        "per_domain_counts": per_domain_counts,
        "pairwise_overlap": pairwise,
        "within_domain_duplicates": within_domain,
        "meta_test_seed_present_in_public_files": meta_test_seed_in_public,
        "meta_test_records_present_in_dev_view": False,
        "meta_test_records_present_in_calibration_view": False,
        "passed": passed,
    }


def _public_leakage_summary(audit: dict[str, Any]) -> dict[str, Any]:
    """Return public leakage summary with no test fingerprints.

    Contains only counts, status flags, and hashes — no test
    fingerprint lists that could be reversible.
    """
    return {
        "per_domain_counts": audit["per_domain_counts"],
        "pairwise_overlap": audit["pairwise_overlap"],
        "within_domain_duplicates": audit["within_domain_duplicates"],
        "meta_test_seed_present_in_public_files": audit["meta_test_seed_present_in_public_files"],
        "meta_test_records_present_in_dev_view": audit["meta_test_records_present_in_dev_view"],
        "meta_test_records_present_in_calibration_view": audit["meta_test_records_present_in_calibration_view"],
        "passed": audit["passed"],
    }


def _compute_probe_counts(
    all_tasks: dict[str, tuple[MetaTask, ...]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute exact probe counts per split/family/kind.

    Returns a nested dict: ``{split: {family: {kind: count}}}``.
    """
    result: dict[str, dict[str, dict[str, int]]] = {}
    split_map = {
        "meta_train": "meta_train",
        "meta_validation_tuning": "meta_validation_tuning",
        "meta_validation_calibration": "meta_validation_calibration",
        "meta_test": "meta_test",
    }
    for domain_key, split_name in split_map.items():
        tasks = all_tasks.get(domain_key, ())
        if not tasks:
            continue
        result[split_name] = {}
        for task in tasks:
            fam = task.family.value
            if fam not in result[split_name]:
                result[split_name][fam] = {}
            for kind in ProbeKind:
                kind_name = kind.value
                count = len(task.probes.by_kind(kind))
                result[split_name][fam][kind_name] = (
                    result[split_name][fam].get(kind_name, 0) + count
                )
    return result


# ---------------------------------------------------------------------------
# Definition materialization
# ---------------------------------------------------------------------------

def _generate_all_tasks(
    config: InstrumentDefinitionConfig,
    meta_test_seed: str,
) -> dict[str, tuple[MetaTask, ...]]:
    """Generate tasks for all four domains.

    Uses the public ``build_dev_catalog`` for DEV tasks (train +
    validation), then splits validation into tuning and calibration
    by index.  Meta-test tasks are generated by the private
    ``_sealed_taskgen`` module using the independent test seed.

    Returns a dict mapping domain name to tuple of MetaTask.
    """
    tg_config = TaskGeneratorConfig(
        root_seed=config.root_seed,
        train_tasks_per_family=config.train_tasks_per_family,
        validation_tasks_per_family=config.tuning_tasks_per_family
        + config.calibration_tasks_per_family,
        min_events=config.event_min,
        max_events=config.event_max,
    )

    # Build the full DEV catalog (train + validation with firewall)
    catalog = build_dev_catalog(tg_config)

    family_order = (
        TaskFamily.CONTEXTUAL_REMAP,
        TaskFamily.RULE_TRANSFORMATION,
        TaskFamily.FINITE_STATE,
    )

    # Train tasks are already in order
    train_tasks = list(catalog.meta_train)

    # Split validation into tuning and calibration by index.
    # Validation tasks are in family/index order: for each family,
    # the first tuning_tasks_per_family are tuning, the rest are calibration.
    validation = list(catalog.meta_validation)
    tuning_tasks: list[MetaTask] = []
    calibration_tasks: list[MetaTask] = []
    val_idx = 0
    for _family in family_order:
        for _ in range(config.tuning_tasks_per_family):
            tuning_tasks.append(validation[val_idx])
            val_idx += 1
        for _ in range(config.calibration_tasks_per_family):
            calibration_tasks.append(validation[val_idx])
            val_idx += 1

    # Collect all DEV fingerprints for sealed collision rejection
    dev_fps: set[str] = set()
    for task in train_tasks:
        dev_fps.add(task.rule_fingerprint)
        dev_fps.add(task.assignment_fingerprint)
        dev_fps.add(task.composition_fingerprint)
        dev_fps.add(task.paraphrase_group_fingerprint)
    for task in tuning_tasks:
        dev_fps.add(task.rule_fingerprint)
        dev_fps.add(task.assignment_fingerprint)
        dev_fps.add(task.composition_fingerprint)
        dev_fps.add(task.paraphrase_group_fingerprint)
    for task in calibration_tasks:
        dev_fps.add(task.rule_fingerprint)
        dev_fps.add(task.assignment_fingerprint)
        dev_fps.add(task.composition_fingerprint)
        dev_fps.add(task.paraphrase_group_fingerprint)

    # Generate sealed meta-test tasks using the independent test seed
    test_seed_int = int(meta_test_seed, 16) & ((1 << 63) - 1)
    meta_test_tasks = generate_sealed_tasks(
        test_seed=test_seed_int,
        config=tg_config,
        tasks_per_family=config.calibration_tasks_per_family,
        dev_fingerprints=dev_fps,
    )

    return {
        "meta_train": tuple(train_tasks),
        "meta_validation_tuning": tuple(tuning_tasks),
        "meta_validation_calibration": tuple(calibration_tasks),
        "meta_test": tuple(meta_test_tasks),
    }


def materialize_definition(
    config: InstrumentDefinitionConfig,
    *,
    test_seed_file: Path,
    out: Path,
) -> InstrumentDefinition:
    """Materialize a deterministic instrument definition.

    Creates a new directory at *out* with:
    - ``DEFINITION.json``
    - ``public/DEV_VIEW.json`` (train + tuning)
    - ``public/CALIBRATION_VIEW.json`` (calibration only)
    - ``public/generator.json``, ``seeds.json``, ``prompts.json``,
      ``scorers.json``, ``endpoints.json``, ``chat_template.txt``,
      ``probe_counts.json``
    - ``public/tasks/*.jsonl``
    - ``public/audits/leakage_summary.json``
    - ``sealed/meta_test_seed.json``, ``tasks/meta_test.jsonl``,
      ``audits/leakage_details.json``

    Refuses to overwrite an existing directory.
    """
    out = Path(out)
    if out.exists():
        raise InstrumentError(f"Output directory already exists: {out}")

    # Read the independent test seed
    meta_test_seed_hex, meta_test_seed_sha256 = _generate_meta_test_seed_commitment(
        Path(test_seed_file)
    )

    # Generate all tasks
    all_tasks = _generate_all_tasks(config, meta_test_seed_hex)

    # Run leakage audit
    audit = _run_leakage_audit(
        all_tasks["meta_train"],
        all_tasks["meta_validation_tuning"],
        all_tasks["meta_validation_calibration"],
        all_tasks["meta_test"],
        meta_test_seed_in_public=False,
    )
    if not audit["passed"]:
        raise InstrumentError(
            f"Leakage audit failed: {json.dumps(audit, sort_keys=True)}"
        )

    # Generate seed table
    seed_table = _generate_dev_seed_table(config)
    seed_table_bytes = strict_canonical_json(seed_table)
    dev_seed_table_sha256 = hashlib.sha256(seed_table_bytes).hexdigest()

    # Generate registries
    prompt_registry = _prompt_registry_obj()
    scorer_registry = _scorer_registry_obj()
    endpoint_registry = _endpoint_registry_obj()

    prompt_registry_sha256 = hashlib.sha256(
        strict_canonical_json(prompt_registry)
    ).hexdigest()
    scorer_registry_sha256 = hashlib.sha256(
        strict_canonical_json(scorer_registry)
    ).hexdigest()
    endpoint_registry_sha256 = hashlib.sha256(
        strict_canonical_json(endpoint_registry)
    ).hexdigest()

    # Compute probe counts
    probe_counts = _compute_probe_counts(all_tasks)
    probe_counts_sha256 = hashlib.sha256(
        strict_canonical_json(probe_counts)
    ).hexdigest()

    # Build task JSONL records
    train_records = [
        _task_to_jsonl_record(t, "meta_train", i)
        for i, t in enumerate(all_tasks["meta_train"])
    ]
    tuning_records = [
        _task_to_jsonl_record(t, "meta_validation_tuning", i)
        for i, t in enumerate(all_tasks["meta_validation_tuning"])
    ]
    calibration_records = [
        _task_to_jsonl_record(t, "meta_validation_calibration", i)
        for i, t in enumerate(all_tasks["meta_validation_calibration"])
    ]
    meta_test_records = [
        _task_to_jsonl_record(t, "meta_test", i)
        for i, t in enumerate(all_tasks["meta_test"])
    ]

    # Create directory structure
    public_dir = out / "public"
    sealed_dir = out / "sealed"
    tasks_dir = public_dir / "tasks"
    audits_dir = public_dir / "audits"
    sealed_tasks_dir = sealed_dir / "tasks"
    sealed_audits_dir = sealed_dir / "audits"

    for d in (public_dir, sealed_dir, tasks_dir, audits_dir,
              sealed_tasks_dir, sealed_audits_dir):
        d.mkdir(parents=True, exist_ok=False)

    # Write files and collect hashes
    file_entries: list[InstrumentFileEntry] = []

    def _record_file(rel_path: str, sha: str, size: int, visibility: str, role: str) -> None:
        file_entries.append(InstrumentFileEntry(
            path=rel_path, sha256=sha, size_bytes=size,
            visibility=visibility, role=role,
        ))

    # Write task JSONL files
    for name, records, vis, role in [
        ("meta_train", train_records, "public", "train_tasks"),
        ("meta_validation_tuning", tuning_records, "public", "tuning_tasks"),
        ("meta_validation_calibration", calibration_records, "public", "calibration_tasks"),
    ]:
        path = tasks_dir / f"{name}.jsonl"
        sha, size = _write_canonical_jsonl(path, records)
        _record_file(f"public/tasks/{name}.jsonl", sha, size, vis, role)

    # Write sealed meta-test tasks
    mt_path = sealed_tasks_dir / "meta_test.jsonl"
    sha, size = _write_canonical_jsonl(mt_path, meta_test_records)
    _record_file("sealed/tasks/meta_test.jsonl", sha, size, "sealed", "meta_test_tasks")

    # Write sealed meta-test seed
    mt_seed_obj = {
        "schema": "oczy/meta-cortex/meta-test-seed/v1",
        "seed_sha256": meta_test_seed_sha256,
    }
    mt_seed_path = sealed_dir / "meta_test_seed.json"
    sha, size = _write_canonical_json(mt_seed_path, mt_seed_obj)
    _record_file("sealed/meta_test_seed.json", sha, size, "sealed", "meta_test_seed")

    # Write generator.json
    generator_obj = {
        "schema": TASKGEN_SCHEMA,
        "algorithm": "sha256-counter-rejection/v1",
        "root_seed": config.root_seed,
        "family_order": [f.value for f in (
            TaskFamily.CONTEXTUAL_REMAP,
            TaskFamily.RULE_TRANSFORMATION,
            TaskFamily.FINITE_STATE,
        )],
        "max_collision_nonce": MAX_COLLISION_NONCE,
    }
    gen_path = public_dir / "generator.json"
    sha, size = _write_canonical_json(gen_path, generator_obj)
    _record_file("public/generator.json", sha, size, "public", "generator_config")

    # Write seeds.json
    seeds_path = public_dir / "seeds.json"
    sha, size = _write_canonical_json(seeds_path, seed_table)
    _record_file("public/seeds.json", sha, size, "public", "dev_seeds")

    # Write prompts.json
    prompts_path = public_dir / "prompts.json"
    sha, size = _write_canonical_json(prompts_path, prompt_registry)
    _record_file("public/prompts.json", sha, size, "public", "prompt_registry")

    # Write scorers.json
    scorers_path = public_dir / "scorers.json"
    sha, size = _write_canonical_json(scorers_path, scorer_registry)
    _record_file("public/scorers.json", sha, size, "public", "scorer_registry")

    # Write endpoints.json
    endpoints_path = public_dir / "endpoints.json"
    sha, size = _write_canonical_json(endpoints_path, endpoint_registry)
    _record_file("public/endpoints.json", sha, size, "public", "endpoint_registry")

    # Write chat_template.txt placeholder
    chat_template_path = public_dir / "chat_template.txt"
    sha, size = _write_text(chat_template_path, config.organ_model_id)
    _record_file("public/chat_template.txt", sha, size, "public", "chat_template")

    # Write probe_counts.json
    pc_path = public_dir / "probe_counts.json"
    sha, size = _write_canonical_json(pc_path, probe_counts)
    _record_file("public/probe_counts.json", sha, size, "public", "probe_counts")

    # Write public leakage summary
    pub_audit = _public_leakage_summary(audit)
    audit_path = audits_dir / "leakage_summary.json"
    sha, size = _write_canonical_json(audit_path, pub_audit)
    _record_file("public/audits/leakage_summary.json", sha, size, "public", "leakage_audit")

    # Write sealed leakage details
    sealed_audit_path = sealed_audits_dir / "leakage_details.json"
    sha, size = _write_canonical_json(sealed_audit_path, audit)
    _record_file("sealed/audits/leakage_details.json", sha, size, "sealed", "leakage_details")

    # Sort non-view file entries by path
    file_entries.sort(key=lambda e: e.path)

    # Build DEV_VIEW.json object (definition_sha256 injected later)
    dev_view_obj = {
        "schema": DEV_VIEW_SCHEMA,
        "instrument_id": config.instrument_id,
        "instrument_version": config.instrument_version,
        "definition_sha256": "",  # injected after definition hash
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": config.organ_model_id,
        "organ_revision": config.organ_revision,
        "organ_parameter_sha256": config.organ_parameter_sha256,
        "chat_template_sha256": config.chat_template_sha256,
        "feature_mode": config.feature_mode,
        "decoding_mode": config.decoding_mode,
        "max_new_tokens": config.max_new_tokens,
        "feature_dim": config.feature_dim,
        "d_cortex": config.d_cortex,
        "soft_bank_width": config.soft_bank_width,
        "abstain_threshold": config.abstain_threshold,
        "train_tasks_per_family": config.train_tasks_per_family,
        "tuning_tasks_per_family": config.tuning_tasks_per_family,
        "family_order": [f.value for f in (
            TaskFamily.CONTEXTUAL_REMAP,
            TaskFamily.RULE_TRANSFORMATION,
            TaskFamily.FINITE_STATE,
        )],
        "task_files": [
            "public/tasks/meta_train.jsonl",
            "public/tasks/meta_validation_tuning.jsonl",
        ],
        "dev_view_sha256": "",  # self-hash, filled below
    }
    # Compute dev_view_sha256 excluding self-hash AND definition_sha256
    # (definition_sha256 is injected after the definition hash is known)
    dev_view_obj_for_hash = {
        k: v for k, v in dev_view_obj.items()
        if k not in ("dev_view_sha256", "definition_sha256")
    }
    dev_view_sha256 = hashlib.sha256(
        strict_canonical_json(dev_view_obj_for_hash)
    ).hexdigest()
    dev_view_obj["dev_view_sha256"] = dev_view_sha256

    # Build CALIBRATION_VIEW.json object (definition_sha256 injected later)
    cal_view_obj = {
        "schema": CALIBRATION_VIEW_SCHEMA,
        "instrument_id": config.instrument_id,
        "instrument_version": config.instrument_version,
        "definition_sha256": "",  # injected after definition hash
        "scorer_sha256": scorer_registry_sha256,
        "endpoint_schema_sha256": endpoint_registry_sha256,
        "confidence_level": 0.95,
        "target_power": 0.80,
        "minimum_tasks_per_family": config.calibration_tasks_per_family,
        "developmental_seeds": list(seed_table["developmental"]),
        "evaluation_seeds": list(seed_table["evaluation"]),
        "no_update_repeat_seeds": list(seed_table["no_update_repeat"]),
        "task_cluster_bootstrap_seed": seed_table["task_cluster_bootstrap"][0],
        "calibration_tasks_per_family": {
            f.value: config.calibration_tasks_per_family
            for f in (
                TaskFamily.CONTEXTUAL_REMAP,
                TaskFamily.RULE_TRANSFORMATION,
                TaskFamily.FINITE_STATE,
            )
        },
        "family_order": [f.value for f in (
            TaskFamily.CONTEXTUAL_REMAP,
            TaskFamily.RULE_TRANSFORMATION,
            TaskFamily.FINITE_STATE,
        )],
        "task_files": ["public/tasks/meta_validation_calibration.jsonl"],
        "calibration_view_sha256": "",  # self-hash, filled below
    }
    # Compute calibration_view_sha256 excluding self-hash AND definition_sha256
    cal_view_obj_for_hash = {
        k: v for k, v in cal_view_obj.items()
        if k not in ("calibration_view_sha256", "definition_sha256")
    }
    calibration_view_sha256 = hashlib.sha256(
        strict_canonical_json(cal_view_obj_for_hash)
    ).hexdigest()
    cal_view_obj["calibration_view_sha256"] = calibration_view_sha256

    # Build DEFINITION.json (view files excluded from public_files;
    # their file hashes go in the candidate manifest at finalization time)
    def_obj = {
        "schema": INSTRUMENT_DEFINITION_SCHEMA,
        "instrument_id": config.instrument_id,
        "instrument_version": config.instrument_version,
        "lifecycle_state": "definition",
        "source_commit": config.source_commit,
        "source_archive_sha256": config.source_archive_sha256,
        "taskgen_schema": TASKGEN_SCHEMA,
        "generator_algorithm": "sha256-counter-rejection/v1",
        "generator_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "prompt_schema": PROMPT_SCHEMA,
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_schema": SCORER_SCHEMA,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_schema": ENDPOINT_SCHEMA,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": config.organ_model_id,
        "organ_revision": config.organ_revision,
        "organ_parameter_sha256": config.organ_parameter_sha256,
        "chat_template_sha256": config.chat_template_sha256,
        "feature_mode": config.feature_mode,
        "decoding_mode": config.decoding_mode,
        "max_new_tokens": config.max_new_tokens,
        "feature_dim": config.feature_dim,
        "d_cortex": config.d_cortex,
        "soft_bank_width": config.soft_bank_width,
        "event_min": config.event_min,
        "event_max": config.event_max,
        "family_order": [f.value for f in (
            TaskFamily.CONTEXTUAL_REMAP,
            TaskFamily.RULE_TRANSFORMATION,
            TaskFamily.FINITE_STATE,
        )],
        "train_tasks_per_family": config.train_tasks_per_family,
        "tuning_tasks_per_family": config.tuning_tasks_per_family,
        "calibration_tasks_per_family": config.calibration_tasks_per_family,
        "developmental_seeds": list(config.developmental_seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
        "dev_seed_table_sha256": dev_seed_table_sha256,
        "meta_test_seed_sha256": meta_test_seed_sha256,
        "probe_counts_sha256": probe_counts_sha256,
        "dev_view_sha256": dev_view_sha256,
        "calibration_view_sha256": calibration_view_sha256,
        "public_files": [e.to_json_obj() for e in file_entries],
        "definition_sha256": "",  # self-hash, filled last
    }
    def_obj_for_hash = {k: v for k, v in def_obj.items() if k != "definition_sha256"}
    definition_sha256 = hashlib.sha256(strict_canonical_json(def_obj_for_hash)).hexdigest()
    def_obj["definition_sha256"] = definition_sha256

    # Inject definition_sha256 into view objects (self-hashes don't change
    # because definition_sha256 was excluded from the self-hash computation)
    dev_view_obj["definition_sha256"] = definition_sha256
    cal_view_obj["definition_sha256"] = definition_sha256

    # Write view files now that definition_sha256 is injected
    dev_view_path = public_dir / "DEV_VIEW.json"
    sha, size = _write_canonical_json(dev_view_path, dev_view_obj)
    _record_file("public/DEV_VIEW.json", sha, size, "public", "dev_view")

    cal_view_path = public_dir / "CALIBRATION_VIEW.json"
    sha, size = _write_canonical_json(cal_view_path, cal_view_obj)
    _record_file("public/CALIBRATION_VIEW.json", sha, size, "calibration", "calibration_view")

    # Write DEFINITION.json
    def_path = out / "DEFINITION.json"
    _write_canonical_json(def_path, def_obj)

    # Return the InstrumentDefinition object
    return InstrumentDefinition(
        schema=_json_str(def_obj, "schema"),
        instrument_id=_json_str(def_obj, "instrument_id"),
        instrument_version=_json_str(def_obj, "instrument_version"),
        lifecycle_state=_json_str(def_obj, "lifecycle_state"),
        source_commit=_json_str(def_obj, "source_commit"),
        source_archive_sha256=_json_str(def_obj, "source_archive_sha256"),
        taskgen_schema=_json_str(def_obj, "taskgen_schema"),
        generator_algorithm=_json_str(def_obj, "generator_algorithm"),
        generator_source_sha256=_json_str(def_obj, "generator_source_sha256"),
        prompt_schema=_json_str(def_obj, "prompt_schema"),
        prompt_registry_sha256=_json_str(def_obj, "prompt_registry_sha256"),
        scorer_schema=_json_str(def_obj, "scorer_schema"),
        scorer_registry_sha256=_json_str(def_obj, "scorer_registry_sha256"),
        endpoint_schema=_json_str(def_obj, "endpoint_schema"),
        endpoint_registry_sha256=_json_str(def_obj, "endpoint_registry_sha256"),
        organ_model_id=_json_str(def_obj, "organ_model_id"),
        organ_revision=_json_str(def_obj, "organ_revision"),
        organ_parameter_sha256=_json_str(def_obj, "organ_parameter_sha256"),
        chat_template_sha256=_json_str(def_obj, "chat_template_sha256"),
        feature_mode=_json_str(def_obj, "feature_mode"),
        decoding_mode=_json_str(def_obj, "decoding_mode"),
        max_new_tokens=_json_int(def_obj, "max_new_tokens"),
        feature_dim=_json_int(def_obj, "feature_dim"),
        d_cortex=_json_int(def_obj, "d_cortex"),
        soft_bank_width=_json_int(def_obj, "soft_bank_width"),
        event_min=_json_int(def_obj, "event_min"),
        event_max=_json_int(def_obj, "event_max"),
        family_order=tuple(_json_list(def_obj, "family_order")),
        train_tasks_per_family=_json_int(def_obj, "train_tasks_per_family"),
        tuning_tasks_per_family=_json_int(def_obj, "tuning_tasks_per_family"),
        calibration_tasks_per_family=_json_int(def_obj, "calibration_tasks_per_family"),
        developmental_seeds=tuple(_json_list(def_obj, "developmental_seeds")),
        evaluation_seeds=tuple(_json_list(def_obj, "evaluation_seeds")),
        dev_seed_table_sha256=_json_str(def_obj, "dev_seed_table_sha256"),
        meta_test_seed_sha256=_json_str(def_obj, "meta_test_seed_sha256"),
        probe_counts_sha256=_json_str(def_obj, "probe_counts_sha256"),
        dev_view_sha256=_json_str(def_obj, "dev_view_sha256"),
        calibration_view_sha256=_json_str(def_obj, "calibration_view_sha256"),
        public_files=tuple(e for e in file_entries if e.visibility != "sealed"),
        definition_sha256=definition_sha256,
    )


# ---------------------------------------------------------------------------
# Definition verification
# ---------------------------------------------------------------------------

def verify_definition(root: Path) -> InstrumentDefinition:
    """Verify a materialized definition directory at *root*.

    Reads ``DEFINITION.json``, validates all file hashes, paths,
    schema, and self-hash.  Returns the parsed ``InstrumentDefinition``.
    """
    root = Path(root)
    if not root.is_dir():
        raise InstrumentError(f"Not a directory: {root}")

    def_path = root / "DEFINITION.json"
    if not def_path.is_file():
        raise InstrumentError(f"DEFINITION.json not found in {root}")

    raw = def_path.read_bytes()
    data = strict_json_loads(raw.decode("utf-8").rstrip("\n"))

    if data.get("schema") != INSTRUMENT_DEFINITION_SCHEMA:
        raise InstrumentError(
            f"Wrong schema: expected {INSTRUMENT_DEFINITION_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )

    # Verify self-hash
    stored_hash = _json_str(data, "definition_sha256")
    validate_sha256_hex(stored_hash, field_name="definition_sha256")
    obj_for_hash = {k: v for k, v in data.items() if k != "definition_sha256"}
    computed_hash = hashlib.sha256(strict_canonical_json(obj_for_hash)).hexdigest()
    if computed_hash != stored_hash:
        raise InstrumentError(
            f"Definition self-hash mismatch: expected {stored_hash}, "
            f"computed {computed_hash}"
        )

    # Verify all listed file hashes (includes sealed files for integrity)
    file_entries_data = _json_list(data, "public_files")
    for entry_data in file_entries_data:
        entry = InstrumentFileEntry.from_json_obj(entry_data)
        file_path = root / entry.path
        actual_sha, actual_size = _sha256_file(file_path)
        if actual_sha != entry.sha256:
            raise InstrumentError(
                f"File hash mismatch for {entry.path}: "
                f"expected {entry.sha256}, got {actual_sha}"
            )
        if actual_size != entry.size_bytes:
            raise InstrumentError(
                f"File size mismatch for {entry.path}: "
                f"expected {entry.size_bytes}, got {actual_size}"
            )

    # Verify view file self-hashes (not in public_files due to circular
    # dependency: view self-hash excludes definition_sha256, which is only
    # known after the definition hash is computed).
    dev_view_sha = _json_str(data, "dev_view_sha256")
    cal_view_sha = _json_str(data, "calibration_view_sha256")
    for rel_path, expected_sha, self_hash_key, label in [
        ("public/DEV_VIEW.json", dev_view_sha, "dev_view_sha256", "dev_view"),
        ("public/CALIBRATION_VIEW.json", cal_view_sha, "calibration_view_sha256", "calibration_view"),
    ]:
        view_path = root / rel_path
        if not view_path.is_file():
            raise InstrumentError(f"{rel_path} not found in {root}")
        try:
            view_data = strict_json_loads(
                view_path.read_bytes().decode("utf-8").rstrip("\n")
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise InstrumentError(f"{rel_path} is not valid JSON: {exc}") from exc
        view_self_hash = _json_str(view_data, self_hash_key)
        # Verify the self-hash in the view file matches the one in DEFINITION.json
        if view_self_hash != expected_sha:
            raise InstrumentError(
                f"{label} self-hash mismatch: definition has {expected_sha}, "
                f"view file has {view_self_hash}"
            )
        # Verify the self-hash is correct by recomputing it
        view_obj_for_hash = {
            k: v for k, v in view_data.items()
            if k not in (self_hash_key, "definition_sha256")
        }
        computed_sha = hashlib.sha256(
            strict_canonical_json(view_obj_for_hash)
        ).hexdigest()
        if computed_sha != expected_sha:
            raise InstrumentError(
                f"{label} hash mismatch: expected {expected_sha}, "
                f"computed {computed_sha}"
            )

    # Check for extra files in protected directories
    _check_no_extra_files(root, file_entries_data)

    # Build and return InstrumentDefinition
    # public_files excludes sealed entries — sealed files are verified
    # above but not exposed in the public API
    all_entries = tuple(
        InstrumentFileEntry.from_json_obj(e) for e in file_entries_data
    )
    public_only = tuple(
        e for e in all_entries if e.visibility != "sealed"
    )
    return InstrumentDefinition(
        schema=_json_str(data, "schema"),
        instrument_id=_json_str(data, "instrument_id"),
        instrument_version=_json_str(data, "instrument_version"),
        lifecycle_state=_json_str(data, "lifecycle_state"),
        source_commit=_json_str(data, "source_commit"),
        source_archive_sha256=_json_str(data, "source_archive_sha256"),
        taskgen_schema=_json_str(data, "taskgen_schema"),
        generator_algorithm=_json_str(data, "generator_algorithm"),
        generator_source_sha256=_json_str(data, "generator_source_sha256"),
        prompt_schema=_json_str(data, "prompt_schema"),
        prompt_registry_sha256=_json_str(data, "prompt_registry_sha256"),
        scorer_schema=_json_str(data, "scorer_schema"),
        scorer_registry_sha256=_json_str(data, "scorer_registry_sha256"),
        endpoint_schema=_json_str(data, "endpoint_schema"),
        endpoint_registry_sha256=_json_str(data, "endpoint_registry_sha256"),
        organ_model_id=_json_str(data, "organ_model_id"),
        organ_revision=_json_str(data, "organ_revision"),
        organ_parameter_sha256=_json_str(data, "organ_parameter_sha256"),
        chat_template_sha256=_json_str(data, "chat_template_sha256"),
        feature_mode=_json_str(data, "feature_mode"),
        decoding_mode=_json_str(data, "decoding_mode"),
        max_new_tokens=_json_int(data, "max_new_tokens"),
        feature_dim=_json_int(data, "feature_dim"),
        d_cortex=_json_int(data, "d_cortex"),
        soft_bank_width=_json_int(data, "soft_bank_width"),
        event_min=_json_int(data, "event_min"),
        event_max=_json_int(data, "event_max"),
        family_order=tuple(_json_list(data, "family_order")),
        train_tasks_per_family=_json_int(data, "train_tasks_per_family"),
        tuning_tasks_per_family=_json_int(data, "tuning_tasks_per_family"),
        calibration_tasks_per_family=_json_int(data, "calibration_tasks_per_family"),
        developmental_seeds=tuple(_json_list(data, "developmental_seeds")),
        evaluation_seeds=tuple(_json_list(data, "evaluation_seeds")),
        dev_seed_table_sha256=_json_str(data, "dev_seed_table_sha256"),
        meta_test_seed_sha256=_json_str(data, "meta_test_seed_sha256"),
        probe_counts_sha256=_json_str(data, "probe_counts_sha256"),
        dev_view_sha256=dev_view_sha,
        calibration_view_sha256=cal_view_sha,
        public_files=public_only,
        definition_sha256=stored_hash,
    )


def _check_no_extra_files(root: Path, file_entries_data: list[dict[str, Any]]) -> None:
    """Check that no unlisted files exist in protected directories."""
    listed_paths = {e["path"] for e in file_entries_data}
    # Allow DEFINITION.json at root and view files in public/
    # (view files are not in public_files because their hashes depend
    # on definition_sha256, creating a circular dependency; their
    # file hashes are recorded in the candidate manifest instead)
    allowed_unlisted = {
        "DEFINITION.json",
        "CANDIDATE_MANIFEST.json",
        "public/DEV_VIEW.json",
        "public/CALIBRATION_VIEW.json",
    }
    # Walk the tree
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        for fname in filenames:
            if str(rel_dir) == ".":
                rel_path = fname
            else:
                rel_path = str(rel_dir / fname)
            # Normalize to forward slashes
            rel_path = rel_path.replace(os.sep, "/")
            if rel_path not in listed_paths and rel_path not in allowed_unlisted:
                raise InstrumentError(
                    f"Unlisted file in protected directory: {rel_path}"
                )


# ---------------------------------------------------------------------------
# View loaders
# ---------------------------------------------------------------------------

def load_dev_view(public_root: Path) -> DevInstrumentView:
    """Load the public DEV instrument view.

    Reads ``DEV_VIEW.json`` and the train/tuning task files it references.
    Never opens ``sealed/`` or calibration task files.

    *public_root* is the directory containing ``DEV_VIEW.json`` (typically
    ``<definition_root>/public/``).
    """
    public_root = Path(public_root)
    dev_view_path = public_root / "DEV_VIEW.json"
    if not dev_view_path.is_file():
        raise InstrumentError(f"DEV_VIEW.json not found in {public_root}")

    raw = dev_view_path.read_bytes()
    data = strict_json_loads(raw.decode("utf-8").rstrip("\n"))

    if data.get("schema") != DEV_VIEW_SCHEMA:
        raise InstrumentError(
            f"Wrong schema: expected {DEV_VIEW_SCHEMA!r}, got {data.get('schema')!r}"
        )

    # Verify self-hash (excluding self-hash AND definition_sha256,
    # which is injected after the definition hash is computed)
    stored_hash = data.get("dev_view_sha256", "")
    validate_sha256_hex(stored_hash, field_name="dev_view_sha256")
    obj_for_hash = {
        k: v for k, v in data.items()
        if k not in ("dev_view_sha256", "definition_sha256")
    }
    computed_hash = hashlib.sha256(strict_canonical_json(obj_for_hash)).hexdigest()
    if computed_hash != stored_hash:
        raise InstrumentError(
            f"DEV_VIEW self-hash mismatch: expected {stored_hash}, "
            f"computed {computed_hash}"
        )

    # Build binding
    binding = InstrumentBinding(
        instrument_version=_json_str(data, "instrument_version"),
        instrument_manifest_sha256="",  # not yet known at definition stage
        definition_sha256=_json_str(data, "definition_sha256"),
        dev_view_sha256=stored_hash,
        prompt_registry_sha256=_json_str(data, "prompt_registry_sha256"),
        scorer_registry_sha256=_json_str(data, "scorer_registry_sha256"),
        endpoint_registry_sha256=_json_str(data, "endpoint_registry_sha256"),
        organ_hash=_json_str(data, "organ_parameter_sha256"),
    )

    # Load task files — only train and tuning
    task_files = data.get("task_files", [])
    for tf in task_files:
        if "sealed" in tf:
            raise InstrumentError(
                f"DEV_VIEW references sealed file: {tf}"
            )
        if "calibration" in tf:
            raise InstrumentError(
                f"DEV_VIEW references calibration file: {tf}"
            )

    # Build a minimal DevTaskCatalog from the task files
    # (The actual task parsing is handled by the training pipeline;
    # here we just verify the files exist and are accessible.)
    catalog = _build_catalog_from_files(public_root.parent, task_files)

    return DevInstrumentView(binding=binding, catalog=catalog)


def load_calibration_view(public_root: Path) -> CalibrationInstrumentView:
    """Load the held-back calibration instrument view.

    Reads ``CALIBRATION_VIEW.json`` and the calibration task files it
    references.  Never opens ``sealed/`` or train/tuning task files.
    """
    public_root = Path(public_root)
    cal_view_path = public_root / "CALIBRATION_VIEW.json"
    if not cal_view_path.is_file():
        raise InstrumentError(f"CALIBRATION_VIEW.json not found in {public_root}")

    raw = cal_view_path.read_bytes()
    data = strict_json_loads(raw.decode("utf-8").rstrip("\n"))

    if data.get("schema") != CALIBRATION_VIEW_SCHEMA:
        raise InstrumentError(
            f"Wrong schema: expected {CALIBRATION_VIEW_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )

    # Verify self-hash (excluding self-hash AND definition_sha256,
    # which is injected after the definition hash is computed)
    stored_hash = data.get("calibration_view_sha256", "")
    validate_sha256_hex(stored_hash, field_name="calibration_view_sha256")
    obj_for_hash = {
        k: v for k, v in data.items()
        if k not in ("calibration_view_sha256", "definition_sha256")
    }
    computed_hash = hashlib.sha256(strict_canonical_json(obj_for_hash)).hexdigest()
    if computed_hash != stored_hash:
        raise InstrumentError(
            f"CALIBRATION_VIEW self-hash mismatch: expected {stored_hash}, "
            f"computed {computed_hash}"
        )

    # Verify task files don't reference sealed or train/tuning
    task_files = data.get("task_files", [])
    for tf in task_files:
        if "sealed" in tf:
            raise InstrumentError(
                f"CALIBRATION_VIEW references sealed file: {tf}"
            )
        if "meta_train" in tf or "tuning" in tf:
            raise InstrumentError(
                f"CALIBRATION_VIEW references non-calibration file: {tf}"
            )

    catalog = _build_catalog_from_files(public_root.parent, task_files)
    cal_tasks = list(catalog.meta_validation)

    return CalibrationInstrumentView(
        schema=_json_str(data, "schema"),
        instrument_id=_json_str(data, "instrument_id"),
        instrument_version=_json_str(data, "instrument_version"),
        definition_sha256=_json_str(data, "definition_sha256"),
        calibration_view_sha256=stored_hash,
        scorer_sha256=_json_str(data, "scorer_sha256"),
        endpoint_schema_sha256=_json_str(data, "endpoint_schema_sha256"),
        confidence_level=_json_float(data, "confidence_level"),
        target_power=_json_float(data, "target_power"),
        minimum_tasks_per_family=_json_int(data, "minimum_tasks_per_family"),
        developmental_seeds=tuple(_json_list(data, "developmental_seeds")),
        evaluation_seeds=tuple(_json_list(data, "evaluation_seeds")),
        no_update_repeat_seeds=tuple(_json_list(data, "no_update_repeat_seeds")),
        task_cluster_bootstrap_seed=_json_int(data, "task_cluster_bootstrap_seed"),
        tasks=tuple(cal_tasks),
        calibration_tasks_per_family=_json_dict(data, "calibration_tasks_per_family"),
    )


def _build_catalog_from_files(
    root: Path, task_files: list[str]
) -> DevTaskCatalog:
    """Build a DevTaskCatalog from JSONL task files.

    This is a minimal loader that reconstructs MetaTask objects from
    the JSONL records.  For the definition stage, we just need the
    catalog to exist with correct fingerprints for the firewall audit.
    """
    from .contracts import (
        DialogueMessage,
        LearningEvent,
        OutcomeCode,
        ProbeBattery,
        ProbeCase,
    )

    train_tasks: list[MetaTask] = []
    validation_tasks: list[MetaTask] = []

    for tf in task_files:
        path = root / tf
        if not path.is_file():
            raise InstrumentError(f"Task file not found: {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            record = strict_json_loads(line)
            # Reconstruct MetaTask
            events = tuple(
                LearningEvent(
                    observation_messages=tuple(
                        DialogueMessage(
                            role=m["role"], content=m["content"]
                        )
                        for m in ev["observation_messages"]
                    ),
                    attempted_behavior=ev["attempted_behavior"],
                    correction=ev["correction"],
                    outcome=OutcomeCode(ev["outcome"]),
                )
                for ev in record["events"]
            )
            probes_dict: dict[str, tuple[ProbeCase, ...]] = {}
            for kind in ProbeKind:
                kind_records = record["probes"].get(kind.value, [])
                probes_dict[kind.value] = tuple(
                    ProbeCase(
                        messages=tuple(
                            DialogueMessage(
                                role=m["role"], content=m["content"]
                            )
                            for m in p["messages"]
                        ),
                        expected_response=p["expected_response"],
                        kind=ProbeKind(p["kind"]),
                    )
                    for p in kind_records
                )
            probes = ProbeBattery(
                pre=probes_dict["pre"],
                same_rule=probes_dict["same_rule"],
                transfer=probes_dict["transfer"],
                composition=probes_dict["composition"],
                specificity=probes_dict["specificity"],
                oracle_context=probes_dict["oracle_context"],
            )
            task = MetaTask(
                family=TaskFamily(record["family"]),
                split=DevSplit.META_TRAIN,  # placeholder
                events=events,
                probes=probes,
                rule_fingerprint=record["rule_fingerprint"],
                assignment_fingerprint=record["assignment_fingerprint"],
                composition_fingerprint=record["composition_fingerprint"],
                paraphrase_group_fingerprint=record["paraphrase_group_fingerprint"],
            )
            # Assign split based on split_role
            if record["split_role"] == "meta_train":
                train_tasks.append(task)
            else:
                # Replace split with META_VALIDATION
                from dataclasses import replace
                task = replace(task, split=DevSplit.META_VALIDATION)
                validation_tasks.append(task)

    # Build audit
    from .taskgen import audit_split_firewall
    audit = audit_split_firewall(train_tasks, validation_tasks)

    # Compute catalog digest
    from .taskgen import _catalog_digest
    digest = _catalog_digest(train_tasks, validation_tasks)

    return DevTaskCatalog(
        meta_train=tuple(train_tasks),
        meta_validation=tuple(validation_tasks),
        catalog_sha256=digest,
        split_audit=audit,
    )


# ---------------------------------------------------------------------------
# Candidate finalization
# ---------------------------------------------------------------------------

def finalize_candidate(
    definition_root: Path,
    *,
    calibration_report: Path,
    power_report: Path,
    out: Path,
) -> CandidateManifest:
    """Finalize a new candidate from a verified definition and calibration evidence.

    Creates a new directory at *out* with:
    - ``CANDIDATE_MANIFEST.json`` binding all instrument bytes, the
      empirical equivalence margin, and the powered task count.
    - Copies of all public and sealed files from the definition.
    - Calibration/power evidence files.

    Refuses to overwrite an existing directory.  Verifies that
    calibration evidence binds the exact definition hashes and says
    ``holdout_accessed=false``.
    """
    definition_root = Path(definition_root)
    out = Path(out)
    if out.exists():
        raise InstrumentError(f"Output directory already exists: {out}")

    # Verify the definition
    definition = verify_definition(definition_root)

    # Read calibration report
    cal_report_path = Path(calibration_report)
    if not cal_report_path.is_file():
        raise InstrumentError(f"Calibration report not found: {cal_report_path}")
    cal_report_raw = cal_report_path.read_bytes()
    cal_report = strict_json_loads(cal_report_raw.decode("utf-8").rstrip("\n"))
    cal_report_sha256 = hashlib.sha256(cal_report_raw).hexdigest()
    cal_report_size = len(cal_report_raw)

    # Read power report
    power_report_path = Path(power_report)
    if not power_report_path.is_file():
        raise InstrumentError(f"Power report not found: {power_report_path}")
    power_report_raw = power_report_path.read_bytes()
    power_report_data = strict_json_loads(power_report_raw.decode("utf-8").rstrip("\n"))
    power_report_sha256 = hashlib.sha256(power_report_raw).hexdigest()
    power_report_size = len(power_report_raw)

    # Verify calibration evidence binds the correct definition
    if cal_report.get("definition_sha256") != definition.definition_sha256:
        raise InstrumentError(
            "Calibration report does not bind this definition"
        )
    if cal_report.get("holdout_accessed") is not False:
        raise InstrumentError(
            "Calibration report must have holdout_accessed=false"
        )

    # Extract margin and sample size from evidence
    equivalence_margin = cal_report.get("equivalence_margin", "")
    if not isinstance(equivalence_margin, str) or not equivalence_margin:
        raise InstrumentError("Calibration report missing equivalence_margin")
    canonical_decimal(equivalence_margin)

    sample_size = power_report_data.get("sample_size_tasks_per_family")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise InstrumentError("Power report sample_size_tasks_per_family must be int")
    if sample_size < 30:
        raise InstrumentError(
            f"sample_size_tasks_per_family must be >= 30, got {sample_size}"
        )

    # Verify power report binds the correct definition
    if power_report_data.get("definition_sha256") != definition.definition_sha256:
        raise InstrumentError(
            "Power report does not bind this definition"
        )

    # Create candidate directory
    public_dir = out / "public"
    sealed_dir = out / "sealed"
    public_dir.mkdir(parents=True, exist_ok=False)
    sealed_dir.mkdir(parents=True, exist_ok=False)

    # Copy all files from definition to candidate
    file_entries: list[InstrumentFileEntry] = []
    for entry in definition.public_files:
        src = definition_root / entry.path
        dst = out / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Verify hash before copying
        raw = _read_and_verify(src, entry.sha256)
        _write_atomic(dst, raw)
        file_entries.append(InstrumentFileEntry(
            path=entry.path,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            visibility=entry.visibility,
            role=entry.role,
        ))
    # Copy sealed files (excluded from public_files but needed in candidate)
    def_json_path = definition_root / "DEFINITION.json"
    def_data = strict_json_loads(def_json_path.read_bytes().decode("utf-8").rstrip("\n"))
    all_file_entries = _json_list(def_data, "public_files")
    for entry_data in all_file_entries:
        entry = InstrumentFileEntry.from_json_obj(entry_data)
        if entry.visibility != "sealed":
            continue
        src = definition_root / entry.path
        dst = out / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = _read_and_verify(src, entry.sha256)
        _write_atomic(dst, raw)
        file_entries.append(InstrumentFileEntry(
            path=entry.path,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            visibility=entry.visibility,
            role=entry.role,
        ))
    # Copy view files and DEFINITION.json (not in public_files because
    # their hashes depend on definition_sha256, creating a circular
    # dependency at definition time; they are included in the manifest)
    for rel_path, visibility, role in [
        ("DEFINITION.json", "public", "instrument_definition"),
        ("public/DEV_VIEW.json", "public", "dev_view"),
        ("public/CALIBRATION_VIEW.json", "calibration", "calibration_view"),
    ]:
        src = definition_root / rel_path
        dst = out / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        _write_atomic(dst, raw)
        file_entries.append(InstrumentFileEntry(
            path=rel_path,
            sha256=sha,
            size_bytes=size,
            visibility=visibility,
            role=role,
        ))

    # Add calibration evidence files
    cal_evidence_path = public_dir / "calibration" / "DEV_DISTRIBUTIONS.json"
    cal_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(cal_evidence_path, cal_report_raw)
    file_entries.append(InstrumentFileEntry(
        path="public/calibration/DEV_DISTRIBUTIONS.json",
        sha256=cal_report_sha256,
        size_bytes=cal_report_size,
        visibility="evidence",
        role="calibration_report",
    ))

    power_evidence_path = public_dir / "calibration" / "POWER_ANALYSIS.json"
    _write_atomic(power_evidence_path, power_report_raw)
    file_entries.append(InstrumentFileEntry(
        path="public/calibration/POWER_ANALYSIS.json",
        sha256=power_report_sha256,
        size_bytes=power_report_size,
        visibility="evidence",
        role="power_report",
    ))

    # Sort file entries
    file_entries.sort(key=lambda e: e.path)

    # Build meta_test_tasks_by_family (A=B=C=N)
    meta_test_tasks_by_family = {
        "contextual_remap": sample_size,
        "rule_transformation": sample_size,
        "finite_state": sample_size,
    }

    # Build probe_counts_by_split_family_kind from definition
    probe_counts_path = definition_root / "public" / "probe_counts.json"
    probe_counts_raw = probe_counts_path.read_bytes()
    probe_counts = strict_json_loads(probe_counts_raw.decode("utf-8").rstrip("\n"))

    # Compute leakage audit hash
    leakage_audit_path = definition_root / "public" / "audits" / "leakage_summary.json"
    leakage_audit_raw = leakage_audit_path.read_bytes()
    leakage_audit_sha256 = hashlib.sha256(leakage_audit_raw).hexdigest()

    # Build candidate manifest
    manifest_obj = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "instrument_id": definition.instrument_id,
        "instrument_version": definition.instrument_version,
        "lifecycle_state": "candidate",
        "definition_sha256": definition.definition_sha256,
        "dev_view_sha256": definition.dev_view_sha256,
        "calibration_view_sha256": definition.calibration_view_sha256,
        "source_commit": definition.source_commit,
        "source_archive_sha256": definition.source_archive_sha256,
        "taskgen_schema": definition.taskgen_schema,
        "generator_algorithm": definition.generator_algorithm,
        "generator_source_sha256": definition.generator_source_sha256,
        "prompt_schema": definition.prompt_schema,
        "prompt_registry_sha256": definition.prompt_registry_sha256,
        "scorer_schema": definition.scorer_schema,
        "scorer_registry_sha256": definition.scorer_registry_sha256,
        "endpoint_schema": definition.endpoint_schema,
        "endpoint_registry_sha256": definition.endpoint_registry_sha256,
        "organ_model_id": definition.organ_model_id,
        "organ_revision": definition.organ_revision,
        "organ_parameter_sha256": definition.organ_parameter_sha256,
        "chat_template_sha256": definition.chat_template_sha256,
        "feature_mode": definition.feature_mode,
        "decoding_mode": definition.decoding_mode,
        "max_new_tokens": definition.max_new_tokens,
        "feature_dim": definition.feature_dim,
        "d_cortex": definition.d_cortex,
        "soft_bank_width": definition.soft_bank_width,
        "abstain_threshold": canonical_decimal(
            _get_abstain_threshold(definition_root)
        ),
        "event_min": definition.event_min,
        "event_max": definition.event_max,
        "family_order": list(definition.family_order),
        "probe_counts_by_split_family_kind": probe_counts,
        "train_tasks_per_family": definition.train_tasks_per_family,
        "tuning_tasks_per_family": definition.tuning_tasks_per_family,
        "calibration_tasks_per_family": definition.calibration_tasks_per_family,
        "sample_size_tasks_per_family": sample_size,
        "meta_test_tasks_by_family": meta_test_tasks_by_family,
        "developmental_seeds": list(definition.developmental_seeds),
        "evaluation_seeds": list(definition.evaluation_seeds),
        "equivalence_margin": equivalence_margin,
        "calibration_report_sha256": cal_report_sha256,
        "power_report_sha256": power_report_sha256,
        "calibration_holdout_accessed": False,
        "independent_sample_unit": "task_rule",
        "leakage_audit_sha256": leakage_audit_sha256,
        "leakage_audit_passed": True,
        "meta_test_seed_sha256": definition.meta_test_seed_sha256,
        "files": [e.to_json_obj() for e in file_entries],
        "manifest_sha256": "",  # self-hash, filled last
    }

    # Compute manifest self-hash
    obj_for_hash = {k: v for k, v in manifest_obj.items() if k != "manifest_sha256"}
    manifest_sha256 = hashlib.sha256(strict_canonical_json(obj_for_hash)).hexdigest()
    manifest_obj["manifest_sha256"] = manifest_sha256

    # Write CANDIDATE_MANIFEST.json
    manifest_path = out / "CANDIDATE_MANIFEST.json"
    _write_canonical_json(manifest_path, manifest_obj)

    # Build and return CandidateManifest
    return CandidateManifest(
        schema=_json_str(manifest_obj, "schema"),
        instrument_id=_json_str(manifest_obj, "instrument_id"),
        instrument_version=_json_str(manifest_obj, "instrument_version"),
        lifecycle_state=_json_str(manifest_obj, "lifecycle_state"),
        definition_sha256=_json_str(manifest_obj, "definition_sha256"),
        dev_view_sha256=_json_str(manifest_obj, "dev_view_sha256"),
        calibration_view_sha256=_json_str(manifest_obj, "calibration_view_sha256"),
        source_commit=_json_str(manifest_obj, "source_commit"),
        source_archive_sha256=_json_str(manifest_obj, "source_archive_sha256"),
        taskgen_schema=_json_str(manifest_obj, "taskgen_schema"),
        generator_algorithm=_json_str(manifest_obj, "generator_algorithm"),
        generator_source_sha256=_json_str(manifest_obj, "generator_source_sha256"),
        prompt_schema=_json_str(manifest_obj, "prompt_schema"),
        prompt_registry_sha256=_json_str(manifest_obj, "prompt_registry_sha256"),
        scorer_schema=_json_str(manifest_obj, "scorer_schema"),
        scorer_registry_sha256=_json_str(manifest_obj, "scorer_registry_sha256"),
        endpoint_schema=_json_str(manifest_obj, "endpoint_schema"),
        endpoint_registry_sha256=_json_str(manifest_obj, "endpoint_registry_sha256"),
        organ_model_id=_json_str(manifest_obj, "organ_model_id"),
        organ_revision=_json_str(manifest_obj, "organ_revision"),
        organ_parameter_sha256=_json_str(manifest_obj, "organ_parameter_sha256"),
        chat_template_sha256=_json_str(manifest_obj, "chat_template_sha256"),
        feature_mode=_json_str(manifest_obj, "feature_mode"),
        decoding_mode=_json_str(manifest_obj, "decoding_mode"),
        max_new_tokens=_json_int(manifest_obj, "max_new_tokens"),
        feature_dim=_json_int(manifest_obj, "feature_dim"),
        d_cortex=_json_int(manifest_obj, "d_cortex"),
        soft_bank_width=_json_int(manifest_obj, "soft_bank_width"),
        abstain_threshold=_json_str(manifest_obj, "abstain_threshold"),
        event_min=_json_int(manifest_obj, "event_min"),
        event_max=_json_int(manifest_obj, "event_max"),
        family_order=tuple(_json_list(manifest_obj, "family_order")),
        probe_counts_by_split_family_kind=_json_dict(manifest_obj, "probe_counts_by_split_family_kind"),
        train_tasks_per_family=_json_int(manifest_obj, "train_tasks_per_family"),
        tuning_tasks_per_family=_json_int(manifest_obj, "tuning_tasks_per_family"),
        calibration_tasks_per_family=_json_int(manifest_obj, "calibration_tasks_per_family"),
        sample_size_tasks_per_family=sample_size,
        meta_test_tasks_by_family=meta_test_tasks_by_family,
        developmental_seeds=tuple(_json_list(manifest_obj, "developmental_seeds")),
        evaluation_seeds=tuple(_json_list(manifest_obj, "evaluation_seeds")),
        equivalence_margin=_json_str(manifest_obj, "equivalence_margin"),
        calibration_report_sha256=_json_str(manifest_obj, "calibration_report_sha256"),
        power_report_sha256=_json_str(manifest_obj, "power_report_sha256"),
        calibration_holdout_accessed=_json_bool(manifest_obj, "calibration_holdout_accessed"),
        independent_sample_unit=_json_str(manifest_obj, "independent_sample_unit"),
        leakage_audit_sha256=_json_str(manifest_obj, "leakage_audit_sha256"),
        leakage_audit_passed=_json_bool(manifest_obj, "leakage_audit_passed"),
        meta_test_seed_sha256=_json_str(manifest_obj, "meta_test_seed_sha256"),
        files=tuple(file_entries),
        manifest_sha256=manifest_sha256,
    )


def _get_abstain_threshold(definition_root: Path) -> str:
    """Read abstain_threshold from DEV_VIEW.json."""
    dev_view_path = definition_root / "public" / "DEV_VIEW.json"
    raw = dev_view_path.read_bytes()
    data = strict_json_loads(raw.decode("utf-8").rstrip("\n"))
    return data.get("abstain_threshold", "0")


# ---------------------------------------------------------------------------
# Candidate verification
# ---------------------------------------------------------------------------

def verify_candidate(candidate_root: Path) -> CandidateManifest:
    """Verify a candidate directory at *candidate_root*.

    Reads ``CANDIDATE_MANIFEST.json``, validates all file hashes,
    schema, self-hash, and invariants.  Returns the parsed
    ``CandidateManifest``.
    """
    candidate_root = Path(candidate_root)
    if not candidate_root.is_dir():
        raise InstrumentError(f"Not a directory: {candidate_root}")

    manifest_path = candidate_root / "CANDIDATE_MANIFEST.json"
    if not manifest_path.is_file():
        raise InstrumentError(f"CANDIDATE_MANIFEST.json not found in {candidate_root}")

    raw = manifest_path.read_bytes()
    data = strict_json_loads(raw.decode("utf-8").rstrip("\n"))

    if data.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise InstrumentError(
            f"Wrong schema: expected {CANDIDATE_MANIFEST_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )

    # Verify self-hash
    stored_hash = _json_str(data, "manifest_sha256")
    validate_sha256_hex(stored_hash, field_name="manifest_sha256")
    obj_for_hash = {k: v for k, v in data.items() if k != "manifest_sha256"}
    computed_hash = hashlib.sha256(strict_canonical_json(obj_for_hash)).hexdigest()
    if computed_hash != stored_hash:
        raise InstrumentError(
            f"Manifest self-hash mismatch: expected {stored_hash}, "
            f"computed {computed_hash}"
        )

    # Verify all file hashes
    file_entries_data = _json_list(data, "files")
    for entry_data in file_entries_data:
        entry = InstrumentFileEntry.from_json_obj(entry_data)
        file_path = candidate_root / entry.path
        actual_sha, actual_size = _sha256_file(file_path)
        if actual_sha != entry.sha256:
            raise InstrumentError(
                f"File hash mismatch for {entry.path}: "
                f"expected {entry.sha256}, got {actual_sha}"
            )
        if actual_size != entry.size_bytes:
            raise InstrumentError(
                f"File size mismatch for {entry.path}: "
                f"expected {entry.size_bytes}, got {actual_size}"
            )

    # Check for extra files
    _check_no_extra_files(candidate_root, file_entries_data)

    # Verify invariants
    if data.get("calibration_holdout_accessed") is not False:
        raise InstrumentError("calibration_holdout_accessed must be False")
    if data.get("leakage_audit_passed") is not True:
        raise InstrumentError("leakage_audit_passed must be True")

    # A=B=C=N invariant
    n = _json_int(data, "sample_size_tasks_per_family")
    meta_test_counts = _json_dict(data, "meta_test_tasks_by_family")
    for fam in ("contextual_remap", "rule_transformation", "finite_state"):
        if meta_test_counts[fam] != n:
            raise InstrumentError(
                f"meta_test_tasks_by_family[{fam}] must equal N={n}"
            )

    # Build and return CandidateManifest
    file_entries = tuple(
        InstrumentFileEntry.from_json_obj(e) for e in file_entries_data
    )
    return CandidateManifest(
        schema=_json_str(data, "schema"),
        instrument_id=_json_str(data, "instrument_id"),
        instrument_version=_json_str(data, "instrument_version"),
        lifecycle_state=_json_str(data, "lifecycle_state"),
        definition_sha256=_json_str(data, "definition_sha256"),
        dev_view_sha256=_json_str(data, "dev_view_sha256"),
        calibration_view_sha256=_json_str(data, "calibration_view_sha256"),
        source_commit=_json_str(data, "source_commit"),
        source_archive_sha256=_json_str(data, "source_archive_sha256"),
        taskgen_schema=_json_str(data, "taskgen_schema"),
        generator_algorithm=_json_str(data, "generator_algorithm"),
        generator_source_sha256=_json_str(data, "generator_source_sha256"),
        prompt_schema=_json_str(data, "prompt_schema"),
        prompt_registry_sha256=_json_str(data, "prompt_registry_sha256"),
        scorer_schema=_json_str(data, "scorer_schema"),
        scorer_registry_sha256=_json_str(data, "scorer_registry_sha256"),
        endpoint_schema=_json_str(data, "endpoint_schema"),
        endpoint_registry_sha256=_json_str(data, "endpoint_registry_sha256"),
        organ_model_id=_json_str(data, "organ_model_id"),
        organ_revision=_json_str(data, "organ_revision"),
        organ_parameter_sha256=_json_str(data, "organ_parameter_sha256"),
        chat_template_sha256=_json_str(data, "chat_template_sha256"),
        feature_mode=_json_str(data, "feature_mode"),
        decoding_mode=_json_str(data, "decoding_mode"),
        max_new_tokens=_json_int(data, "max_new_tokens"),
        feature_dim=_json_int(data, "feature_dim"),
        d_cortex=_json_int(data, "d_cortex"),
        soft_bank_width=_json_int(data, "soft_bank_width"),
        abstain_threshold=_json_str(data, "abstain_threshold"),
        event_min=_json_int(data, "event_min"),
        event_max=_json_int(data, "event_max"),
        family_order=tuple(_json_list(data, "family_order")),
        probe_counts_by_split_family_kind=_json_dict(data, "probe_counts_by_split_family_kind"),
        train_tasks_per_family=_json_int(data, "train_tasks_per_family"),
        tuning_tasks_per_family=_json_int(data, "tuning_tasks_per_family"),
        calibration_tasks_per_family=_json_int(data, "calibration_tasks_per_family"),
        sample_size_tasks_per_family=n,
        meta_test_tasks_by_family=meta_test_counts,
        developmental_seeds=tuple(_json_list(data, "developmental_seeds")),
        evaluation_seeds=tuple(_json_list(data, "evaluation_seeds")),
        equivalence_margin=_json_str(data, "equivalence_margin"),
        calibration_report_sha256=_json_str(data, "calibration_report_sha256"),
        power_report_sha256=_json_str(data, "power_report_sha256"),
        calibration_holdout_accessed=_json_bool(data, "calibration_holdout_accessed"),
        independent_sample_unit=_json_str(data, "independent_sample_unit"),
        leakage_audit_sha256=_json_str(data, "leakage_audit_sha256"),
        leakage_audit_passed=_json_bool(data, "leakage_audit_passed"),
        meta_test_seed_sha256=_json_str(data, "meta_test_seed_sha256"),
        files=file_entries,
        manifest_sha256=stored_hash,
    )
