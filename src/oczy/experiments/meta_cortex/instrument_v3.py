"""Versioned ``meta_cortex/v3`` DEV instrument: the task-support-repaired lineage.

This module is the **G1 freeze** of the ``oczy/meta-cortex/taskgen/v2-dev``
generator lineage, approved for instrument construction and adoption by the
human sign-off recorded in
``experiments/r20-task-support-repair-v1/SIGNOFF_CHAIN.md`` (S1, 2026-10-04).

``meta_cortex/v3`` differs from ``meta_cortex/v2`` in exactly two ways:

1. **Task semantics.** Public DEV tasks are built by ``taskgen_v2.build_dev_catalog_v2``
   (the task-support repair) instead of ``taskgen.build_dev_catalog``.  v1
   remains frozen and untouched; its rendered tasks, thresholds and recorded
   scores are not inputs to this module and are not comparable to v3 results.
2. **Decoder.** Generated ids are decoded with ``skip_special_tokens=True``
   (sign-off S3, adopted with a version bump after its comparability condition
   was discharged in ``experiments_logs/2026-10-05_r20_signoff_s3_comparability.md``).

Everything else — the scorer, endpoint registry, prompt registry, seed
derivation, cortex geometry, organ identity and view schemas — is **byte-identical
to v2**, which this module asserts by hashing the registries and comparing them
to the frozen v2 hashes rather than by re-deriving new ones.

What this module deliberately does NOT do:

- It does not generate or open any sealed meta-test task, seed or generator.  v3
  is a DEV instrument: there is no ``sealed/`` directory, no ``meta_test_seed``
  commitment, and no meta-test entry point.  Materializing v3 cannot access the
  meta-test, by construction.
- It does not read the calibration view of any existing instrument.
- It does not choose, recompute or import any threshold, margin or power number.
  Those are gate G4, and they require explicit human sign-off at G5.

Registry hashes carried unchanged from the frozen v2 instrument:

- prompt registry ``db624922bf4f67e3e1011b5530ade4111b479ef05391ca0a61e375cac2339735``
- scorer registry ``e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742``
- endpoint registry ``669d6130075e429264a6c9eec470bbf82663559433781adcffa6ed942c9ca796``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    DevSplit,
    DevTaskCatalog,
    MetaTask,
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from .instrument import (
    _compute_probe_counts,
    _endpoint_registry_obj,
    _prompt_registry_obj,
    _run_leakage_audit,
    _scorer_registry_obj,
    _task_to_jsonl_record,
    _write_canonical_json,
    _write_canonical_jsonl,
)
from .instrument_contracts import (
    CALIBRATION_VIEW_SCHEMA,
    DEV_VIEW_SCHEMA,
    ENDPOINT_SCHEMA,
    INSTRUMENT_DEFINITION_SCHEMA,
    PROMPT_SCHEMA,
    SCORER_SCHEMA,
    TASK_RECORD_SCHEMA,
    strict_canonical_json,
    strict_json_loads,
)
from .taskgen import MAX_COLLISION_NONCE, assert_split_firewall, audit_split_firewall
from .taskgen_v2 import TASKGEN_SCHEMA_V2, build_dev_catalog_v2

__all__ = [
    "INSTRUMENT_ID",
    "INSTRUMENT_VERSION",
    "GENERATOR_ALGORITHM",
    "V2_FROZEN_HASHES",
    "V3DefinitionError",
    "V3Definition",
    "materialize_v3_definition",
    "verify_v3_definition",
]


INSTRUMENT_ID = "meta_cortex/v3"
INSTRUMENT_VERSION = "v3"
GENERATOR_ALGORITHM = "sha256-counter-rejection/v1"

#: Registries that v3 must inherit byte-for-byte from the frozen v2
#: instrument.  Any drift is a hard failure, not a warning.
V2_FROZEN_HASHES = {
    "prompt_registry_sha256": "db624922bf4f67e3e1011b5530ade4111b479ef05391ca0a61e375cac2339735",
    "scorer_registry_sha256": "e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742",
    "endpoint_registry_sha256": "669d6130075e429264a6c9eec470bbf82663559433781adcffa6ed942c9ca796",
}

_FAMILY_ORDER = (
    TaskFamily.CONTEXTUAL_REMAP,
    TaskFamily.RULE_TRANSFORMATION,
    TaskFamily.FINITE_STATE,
)


class V3DefinitionError(ValueError):
    """Raised when the v3 instrument cannot be frozen or verified."""


class V3Definition:
    """A frozen v3 DEV instrument definition and its self-verifying hashes."""

    __slots__ = (
        "definition_sha256",
        "dev_view_sha256",
        "calibration_view_sha256",
        "dev_seed_table_sha256",
        "probe_counts_sha256",
        "catalog_sha256",
        "support_bundle_sha256",
        "leakage_audit",
        "support_certificate_summary",
        "file_entries",
        "task_counts",
        "probe_counts",
        "root",
    )

    def __init__(
        self,
        *,
        definition_sha256: str,
        dev_view_sha256: str,
        calibration_view_sha256: str,
        dev_seed_table_sha256: str,
        probe_counts_sha256: str,
        catalog_sha256: str,
        support_bundle_sha256: str,
        leakage_audit: dict[str, Any],
        support_certificate_summary: dict[str, Any],
        file_entries: tuple[dict[str, Any], ...],
        task_counts: dict[str, Any],
        probe_counts: dict[str, Any],
        root: Path,
    ) -> None:
        self.definition_sha256 = definition_sha256
        self.dev_view_sha256 = dev_view_sha256
        self.calibration_view_sha256 = calibration_view_sha256
        self.dev_seed_table_sha256 = dev_seed_table_sha256
        self.probe_counts_sha256 = probe_counts_sha256
        self.catalog_sha256 = catalog_sha256
        self.support_bundle_sha256 = support_bundle_sha256
        self.leakage_audit = leakage_audit
        self.support_certificate_summary = support_certificate_summary
        self.file_entries = file_entries
        self.task_counts = task_counts
        self.probe_counts = probe_counts
        self.root = root

    def as_json_obj(self) -> dict[str, Any]:
        return {
            "definition_sha256": self.definition_sha256,
            "dev_view_sha256": self.dev_view_sha256,
            "calibration_view_sha256": self.calibration_view_sha256,
            "dev_seed_table_sha256": self.dev_seed_table_sha256,
            "probe_counts_sha256": self.probe_counts_sha256,
            "catalog_sha256": self.catalog_sha256,
            "support_bundle_sha256": self.support_bundle_sha256,
            "task_counts": self.task_counts,
        }


# ---------------------------------------------------------------------------
# Seed table
# ---------------------------------------------------------------------------


def _dev_seed_table() -> dict[str, Any]:
    """Derive the v3 DEV seed table, versioned by ``meta_cortex/v3``."""
    from .calibration import derive_seed_table

    return derive_seed_table(
        instrument_id=INSTRUMENT_ID,
        instrument_version=INSTRUMENT_VERSION,
    )


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------


def _split_validation(
    catalog: DevTaskCatalog, *, tuning_tasks_per_family: int, calibration_tasks_per_family: int
) -> tuple[tuple[MetaTask, ...], tuple[MetaTask, ...]]:
    """Split validation tasks into tuning then calibration, by family/index order.

    Mirrors the v2 ``_generate_all_tasks`` index walk exactly: validation is
    built in ``_FAMILY_ORDER``, so the first *tuning* entries of each family are
    tuning and the rest are calibration.
    """
    validation = list(catalog.meta_validation)
    tuning: list[MetaTask] = []
    calibration: list[MetaTask] = []
    idx = 0
    for _family in _FAMILY_ORDER:
        for _ in range(tuning_tasks_per_family):
            tuning.append(validation[idx])
            idx += 1
        for _ in range(calibration_tasks_per_family):
            calibration.append(validation[idx])
            idx += 1
    if idx != len(validation):
        raise V3DefinitionError(
            f"validation split sizes {tuning_tasks_per_family}+"
            f"{calibration_tasks_per_family} per family do not consume "
            f"{len(validation)} validation tasks"
        )
    return tuple(tuning), tuple(calibration)


def _support_bundle_digest(bundle: Any) -> str:
    """Hash the support bundle deterministically (audit metadata only).

    Uses each record's own canonical ``as_dict`` so the digest cannot drift from
    the lineage's own serialization contract.
    """
    payload = {
        "lineage": bundle.lineage,
        "catalog_sha256": bundle.catalog_sha256,
        "tasks": [entry.as_dict() for entry in bundle.tasks],
    }
    return hashlib.sha256(strict_canonical_json(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def materialize_v3_definition(
    *,
    out: Path,
    organ_model_id: str,
    organ_revision: str,
    organ_parameter_sha256: str,
    chat_template_sha256: str,
    source_commit: str,
    source_archive_sha256: str,
    root_seed: int,
    train_tasks_per_family: int,
    tuning_tasks_per_family: int,
    calibration_tasks_per_family: int,
    event_min: int,
    event_max: int,
    feature_dim: int,
    d_cortex: int,
    soft_bank_width: int,
    max_new_tokens: int,
    abstain_threshold: str,
    support_certificate_summary: dict[str, Any] | None = None,
) -> V3Definition:
    """Freeze the v3 DEV instrument at *out*.

    Writes ``DEFINITION.json``, ``public/DEV_VIEW.json``,
    ``public/CALIBRATION_VIEW.json`` and the public task/registry files, each
    atomically and each recorded with its SHA-256 and byte size.  There is no
    ``sealed/`` directory: v3 carries no meta-test payload of any kind.

    Refuses to overwrite an existing directory.
    """
    out = Path(out)
    if out.exists():
        raise V3DefinitionError(f"Output directory already exists: {out}")

    # -- Registries: hash, then assert they are unchanged from v2 ------------
    prompt_registry = _prompt_registry_obj()
    scorer_registry = _scorer_registry_obj()
    endpoint_registry = _endpoint_registry_obj()
    prompt_registry_sha256 = hashlib.sha256(strict_canonical_json(prompt_registry)).hexdigest()
    scorer_registry_sha256 = hashlib.sha256(strict_canonical_json(scorer_registry)).hexdigest()
    endpoint_registry_sha256 = hashlib.sha256(strict_canonical_json(endpoint_registry)).hexdigest()
    actual = {
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_registry_sha256": endpoint_registry_sha256,
    }
    drifted = sorted(k for k, v in actual.items() if v != V2_FROZEN_HASHES[k])
    if drifted:
        raise V3DefinitionError(
            f"v3 must inherit the v2 registries byte-for-byte; drifted: {drifted}. "
            f"expected {V2_FROZEN_HASHES}, computed {actual}. A registry change "
            f"requires a new human sign-off, not a v3 freeze."
        )

    # -- Tasks --------------------------------------------------------------
    tg_config = TaskGeneratorConfig(
        root_seed=root_seed,
        train_tasks_per_family=train_tasks_per_family,
        validation_tasks_per_family=tuning_tasks_per_family + calibration_tasks_per_family,
        min_events=event_min,
        max_events=event_max,
    )
    catalog, bundle = build_dev_catalog_v2(tg_config)
    if bundle.lineage != TASKGEN_SCHEMA_V2:
        raise V3DefinitionError(f"unexpected support lineage {bundle.lineage!r}")
    if bundle.catalog_sha256 != catalog.catalog_sha256:
        raise V3DefinitionError("support bundle does not bind the built catalog")

    # The v2 lineage already asserts its own train/validation firewall.
    assert_split_firewall(catalog.split_audit)

    train_tasks = tuple(catalog.meta_train)
    tuning_tasks, calibration_tasks = _split_validation(
        catalog,
        tuning_tasks_per_family=tuning_tasks_per_family,
        calibration_tasks_per_family=calibration_tasks_per_family,
    )

    # -- Four-domain leakage audit over the three DEV domains ----------------
    leakage_audit = _run_leakage_audit(
        train_tasks, tuning_tasks, calibration_tasks, None, meta_test_seed_in_public=False
    )
    if not leakage_audit["passed"]:
        raise V3DefinitionError(
            f"DEV leakage/support audit failed: {json.dumps(leakage_audit, sort_keys=True)}"
        )

    # -- Seeds --------------------------------------------------------------
    seed_table = _dev_seed_table()
    dev_seed_table_sha256 = hashlib.sha256(strict_canonical_json(seed_table)).hexdigest()

    # -- Probe counts over the three DEV domains ----------------------------
    all_tasks = {
        "meta_train": train_tasks,
        "meta_validation_tuning": tuning_tasks,
        "meta_validation_calibration": calibration_tasks,
    }
    probe_counts = _compute_probe_counts(all_tasks)
    probe_counts_sha256 = hashlib.sha256(strict_canonical_json(probe_counts)).hexdigest()

    # -- Records ------------------------------------------------------------
    train_records = [
        _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(train_tasks)
    ]
    tuning_records = [
        _task_to_jsonl_record(t, "meta_validation_tuning", i)
        for i, t in enumerate(tuning_tasks)
    ]
    calibration_records = [
        _task_to_jsonl_record(t, "meta_validation_calibration", i)
        for i, t in enumerate(calibration_tasks)
    ]

    public_dir = out / "public"
    tasks_dir = public_dir / "tasks"
    audits_dir = public_dir / "audits"
    for directory in (public_dir, tasks_dir, audits_dir):
        directory.mkdir(parents=True, exist_ok=False)

    file_entries: list[dict[str, Any]] = []

    def _record(rel_path: str, sha: str, size: int, visibility: str, role: str) -> None:
        file_entries.append(
            {
                "path": rel_path,
                "sha256": sha,
                "size_bytes": size,
                "visibility": visibility,
                "role": role,
            }
        )

    for name, records, visibility, role in (
        ("meta_train", train_records, "public", "train_tasks"),
        ("meta_validation_tuning", tuning_records, "public", "tuning_tasks"),
        ("meta_validation_calibration", calibration_records, "calibration", "calibration_tasks"),
    ):
        path = tasks_dir / f"{name}.jsonl"
        sha, size = _write_canonical_jsonl(path, records)
        _record(f"public/tasks/{name}.jsonl", sha, size, visibility, role)

    # generator.json — names the v2 lineage and binds the generator source.
    generator_source = Path(__file__).resolve().parent / "taskgen_v2.py"
    generator_source_sha256 = hashlib.sha256(generator_source.read_bytes()).hexdigest()
    generator_obj = {
        "schema": TASKGEN_SCHEMA_V2,
        "algorithm": GENERATOR_ALGORITHM,
        "root_seed": root_seed,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "max_collision_nonce": MAX_COLLISION_NONCE,
        "generator_source_sha256": generator_source_sha256,
        "superseded_lineage": "oczy/meta-cortex/taskgen/v1-dev",
        "comparability": "v1 and v3 task scores are not comparable (new task semantics)",
    }
    gen_path = public_dir / "generator.json"
    sha, size = _write_canonical_json(gen_path, generator_obj)
    _record("public/generator.json", sha, size, "public", "generator_config")

    seeds_path = public_dir / "seeds.json"
    sha, size = _write_canonical_json(seeds_path, seed_table)
    _record("public/seeds.json", sha, size, "public", "dev_seeds")

    prompts_path = public_dir / "prompts.json"
    sha, size = _write_canonical_json(prompts_path, prompt_registry)
    _record("public/prompts.json", sha, size, "public", "prompt_registry")

    scorers_path = public_dir / "scorers.json"
    sha, size = _write_canonical_json(scorers_path, scorer_registry)
    _record("public/scorers.json", sha, size, "public", "scorer_registry")

    endpoints_path = public_dir / "endpoints.json"
    sha, size = _write_canonical_json(endpoints_path, endpoint_registry)
    _record("public/endpoints.json", sha, size, "public", "endpoint_registry")

    chat_path = public_dir / "chat_template.txt"
    sha, size = _write_canonical_json(chat_path, {"organ_model_id": organ_model_id})
    _record("public/chat_template.txt", sha, size, "public", "chat_template")

    pc_path = public_dir / "probe_counts.json"
    sha, size = _write_canonical_json(pc_path, probe_counts)
    _record("public/probe_counts.json", sha, size, "public", "probe_counts")

    leakage_summary = {
        "per_domain_counts": leakage_audit["per_domain_counts"],
        "pairwise_overlap": leakage_audit["pairwise_overlap"],
        "within_domain_duplicates": leakage_audit["within_domain_duplicates"],
        "meta_test_seed_present_in_public_files": False,
        "meta_test_records_present_in_dev_view": False,
        "meta_test_records_present_in_calibration_view": False,
        "passed": leakage_audit["passed"],
        "audit_scope": "three DEV domains only; v3 carries no sealed meta-test payload",
    }
    audit_path = audits_dir / "leakage_summary.json"
    sha, size = _write_canonical_json(audit_path, leakage_summary)
    _record("public/audits/leakage_summary.json", sha, size, "public", "leakage_audit")

    # -- Views --------------------------------------------------------------
    dev_view_body = {
        "schema": DEV_VIEW_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "decoding_mode": "greedy_skip_special_tokens",
        "superceded_instrument_id": "meta_cortex/v2",
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": organ_model_id,
        "organ_revision": organ_revision,
        "organ_parameter_sha256": organ_parameter_sha256,
        "chat_template_sha256": chat_template_sha256,
        "feature_mode": "final_layer_mean_pool",
        "max_new_tokens": max_new_tokens,
        "feature_dim": feature_dim,
        "d_cortex": d_cortex,
        "soft_bank_width": soft_bank_width,
        "abstain_threshold": abstain_threshold,
        "train_tasks_per_family": train_tasks_per_family,
        "tuning_tasks_per_family": tuning_tasks_per_family,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "task_files": [
            "public/tasks/meta_train.jsonl",
            "public/tasks/meta_validation_tuning.jsonl",
        ],
        "catalog_sha256": catalog.catalog_sha256,
        "dev_view_sha256": "",
    }
    dev_view_sha256 = hashlib.sha256(
        strict_canonical_json(
            {
                k: v
                for k, v in dev_view_body.items()
                if k not in ("dev_view_sha256", "definition_sha256")
            }
        )
    ).hexdigest()
    dev_view_body["dev_view_sha256"] = dev_view_sha256

    cal_view_body = {
        "schema": CALIBRATION_VIEW_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "definition_sha256": "",
        "scorer_sha256": scorer_registry_sha256,
        "endpoint_schema_sha256": endpoint_registry_sha256,
        "confidence_level": 0.95,
        "target_power": 0.80,
        "minimum_tasks_per_family": calibration_tasks_per_family,
        "developmental_seeds": list(seed_table["developmental"]),
        "evaluation_seeds": list(seed_table["evaluation"]),
        "no_update_repeat_seeds": list(seed_table["no_update_repeat"]),
        "task_cluster_bootstrap_seed": seed_table["task_cluster_bootstrap"],
        "calibration_tasks_per_family": {
            f.value: calibration_tasks_per_family for f in _FAMILY_ORDER
        },
        "family_order": [f.value for f in _FAMILY_ORDER],
        "task_files": ["public/tasks/meta_validation_calibration.jsonl"],
        "calibration_tasks_per_family_note": (
            "counts only; no margin, threshold or power value is set or implied here"
        ),
        "calibration_view_sha256": "",
    }
    calibration_view_sha256 = hashlib.sha256(
        strict_canonical_json(
            {
                k: v
                for k, v in cal_view_body.items()
                if k not in ("calibration_view_sha256", "definition_sha256")
            }
        )
    ).hexdigest()
    cal_view_body["calibration_view_sha256"] = calibration_view_sha256

    # -- DEFINITION.json ----------------------------------------------------
    support_bundle_sha256 = _support_bundle_digest(bundle)
    task_counts = {
        "meta_train": len(train_tasks),
        "meta_validation_tuning": len(tuning_tasks),
        "meta_validation_calibration": len(calibration_tasks),
    }
    file_entries.sort(key=lambda e: e["path"])

    def_body = {
        "schema": INSTRUMENT_DEFINITION_SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "instrument_version": INSTRUMENT_VERSION,
        "lifecycle_state": "definition",
        "dev_only": True,
        "sealed_payload_present": False,
        "meta_test_authorized": False,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "taskgen_schema": TASKGEN_SCHEMA_V2,
        "generator_algorithm": GENERATOR_ALGORITHM,
        "generator_source_sha256": generator_source_sha256,
        "prompt_schema": PROMPT_SCHEMA,
        "prompt_registry_sha256": prompt_registry_sha256,
        "scorer_schema": SCORER_SCHEMA,
        "scorer_registry_sha256": scorer_registry_sha256,
        "endpoint_schema": ENDPOINT_SCHEMA,
        "endpoint_registry_sha256": endpoint_registry_sha256,
        "organ_model_id": organ_model_id,
        "organ_revision": organ_revision,
        "organ_parameter_sha256": organ_parameter_sha256,
        "chat_template_sha256": chat_template_sha256,
        "feature_mode": "final_layer_mean_pool",
        "decoding_mode": "greedy_skip_special_tokens",
        "max_new_tokens": max_new_tokens,
        "feature_dim": feature_dim,
        "d_cortex": d_cortex,
        "soft_bank_width": soft_bank_width,
        "event_min": event_min,
        "event_max": event_max,
        "family_order": [f.value for f in _FAMILY_ORDER],
        "train_tasks_per_family": train_tasks_per_family,
        "tuning_tasks_per_family": tuning_tasks_per_family,
        "calibration_tasks_per_family": calibration_tasks_per_family,
        "task_counts": task_counts,
        "developmental_seeds": list(seed_table["developmental"]),
        "evaluation_seeds": list(seed_table["evaluation"]),
        "dev_seed_table_sha256": dev_seed_table_sha256,
        "probe_counts_sha256": probe_counts_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "support_bundle_sha256": support_bundle_sha256,
        "dev_view_sha256": dev_view_sha256,
        "calibration_view_sha256": calibration_view_sha256,
        "public_files": file_entries,
        "definition_sha256": "",
    }
    definition_sha256 = hashlib.sha256(
        strict_canonical_json({k: v for k, v in def_body.items() if k != "definition_sha256"})
    ).hexdigest()
    def_body["definition_sha256"] = definition_sha256

    dev_view_body["definition_sha256"] = definition_sha256
    cal_view_body["definition_sha256"] = definition_sha256

    _write_canonical_json(public_dir / "DEV_VIEW.json", dev_view_body)
    _write_canonical_json(public_dir / "CALIBRATION_VIEW.json", cal_view_body)
    _write_canonical_json(out / "DEFINITION.json", def_body)

    return V3Definition(
        definition_sha256=definition_sha256,
        dev_view_sha256=dev_view_sha256,
        calibration_view_sha256=calibration_view_sha256,
        dev_seed_table_sha256=dev_seed_table_sha256,
        probe_counts_sha256=probe_counts_sha256,
        catalog_sha256=catalog.catalog_sha256,
        support_bundle_sha256=support_bundle_sha256,
        leakage_audit=leakage_summary,
        support_certificate_summary=support_certificate_summary or {},
        file_entries=tuple(file_entries),
        task_counts=task_counts,
        probe_counts=probe_counts,
        root=out,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_v3_definition(root: Path) -> V3Definition:
    """Verify a frozen v3 DEV instrument at *root*, hashes included.

    Fails closed on: a wrong schema or identity, a self-hash mismatch, a file
    whose bytes do not match its recorded SHA-256 or size, a registry that
    drifted from v2, an unlisted file in the tree, or any sign of a sealed
    payload.
    """
    root = Path(root)
    if not root.is_dir():
        raise V3DefinitionError(f"Not a directory: {root}")

    def_path = root / "DEFINITION.json"
    if not def_path.is_file():
        raise V3DefinitionError(f"DEFINITION.json not found in {root}")

    data = strict_json_loads(def_path.read_bytes().decode("utf-8").rstrip("\n"))

    if data.get("schema") != INSTRUMENT_DEFINITION_SCHEMA:
        raise V3DefinitionError(
            f"Wrong schema: expected {INSTRUMENT_DEFINITION_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )
    if data.get("instrument_id") != INSTRUMENT_ID:
        raise V3DefinitionError(f"instrument_id must be {INSTRUMENT_ID!r}")
    if data.get("instrument_version") != INSTRUMENT_VERSION:
        raise V3DefinitionError(f"instrument_version must be {INSTRUMENT_VERSION!r}")
    if data.get("taskgen_schema") != TASKGEN_SCHEMA_V2:
        raise V3DefinitionError(
            f"v3 must bind the v2 taskgen lineage, got {data.get('taskgen_schema')!r}"
        )

    stored_hash = data.get("definition_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise V3DefinitionError("definition_sha256 missing or malformed")
    computed = hashlib.sha256(
        strict_canonical_json({k: v for k, v in data.items() if k != "definition_sha256"})
    ).hexdigest()
    if computed != stored_hash:
        raise V3DefinitionError(
            f"Definition self-hash mismatch: expected {stored_hash}, computed {computed}"
        )

    # v3 must carry no sealed payload.
    if data.get("sealed_payload_present") is not False:
        raise V3DefinitionError("v3 definition must declare sealed_payload_present=false")
    if (root / "sealed").exists():
        raise V3DefinitionError("v3 definition directory must not contain sealed/")

    entries = data.get("public_files")
    if not isinstance(entries, list) or not entries:
        raise V3DefinitionError("public_files missing from definition")
    for entry in entries:
        if not isinstance(entry, dict):
            raise V3DefinitionError("public_files entries must be objects")
        if entry.get("visibility") == "sealed":
            raise V3DefinitionError(
                f"v3 must not list a sealed file: {entry.get('path')!r}"
            )
        path = root / entry["path"]
        if not path.is_file():
            raise V3DefinitionError(f"File not found: {path}")
        if path.is_symlink():
            raise V3DefinitionError(f"Symlink not allowed: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise V3DefinitionError(f"File hash mismatch for {entry['path']}")
        if len(raw) != entry["size_bytes"]:
            raise V3DefinitionError(f"File size mismatch for {entry['path']}")

    # Unlisted files anywhere in the tree are a hard failure.
    listed = {entry["path"] for entry in entries}
    allowed_unlisted = {
        "DEFINITION.json",
        "public/DEV_VIEW.json",
        "public/CALIBRATION_VIEW.json",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in listed and rel not in allowed_unlisted:
            raise V3DefinitionError(f"Unlisted file in frozen instrument: {rel}")

    # Registries must still match v2 exactly.
    for key, expected in V2_FROZEN_HASHES.items():
        if data.get(key) != expected:
            raise V3DefinitionError(
                f"registry drift: {key} is {data.get(key)!r}, v2 frozen value is {expected!r}"
            )

    # View self-hashes must recompute, and both must name this definition.
    for rel_path, key in (
        ("public/DEV_VIEW.json", "dev_view_sha256"),
        ("public/CALIBRATION_VIEW.json", "calibration_view_sha256"),
    ):
        view = strict_json_loads((root / rel_path).read_bytes().decode("utf-8").rstrip("\n"))
        if view.get(key) != data.get(key):
            raise V3DefinitionError(f"{rel_path} self-hash disagrees with DEFINITION.json")
        recomputed = hashlib.sha256(
            strict_canonical_json(
                {k: v for k, v in view.items() if k not in (key, "definition_sha256")}
            )
        ).hexdigest()
        if recomputed != data.get(key):
            raise V3DefinitionError(f"{rel_path} self-hash mismatch")
        if view.get("definition_sha256") != stored_hash:
            raise V3DefinitionError(f"{rel_path} does not bind this definition")
        if view.get("instrument_id") != INSTRUMENT_ID:
            raise V3DefinitionError(f"{rel_path} has the wrong instrument_id")

    # The DEV view may not reference the calibration file, and vice versa.
    dev_view = strict_json_loads(
        (root / "public" / "DEV_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    for task_file in dev_view.get("task_files", []):
        if "calibration" in task_file or "sealed" in task_file:
            raise V3DefinitionError(f"DEV view references a held-back file: {task_file}")
    cal_view = strict_json_loads(
        (root / "public" / "CALIBRATION_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    for task_file in cal_view.get("task_files", []):
        if "meta_train" in task_file or "tuning" in task_file or "sealed" in task_file:
            raise V3DefinitionError(
                f"calibration view references a non-calibration file: {task_file}"
            )

    # Materialized tasks must be byte-identical to a fresh in-memory rebuild
    # of the same lineage — the freeze must be a *record* of the generator,
    # not a snapshot that can drift from it.
    _rebuild_and_compare(root, data)

    audit = strict_json_loads(
        (root / "public" / "audits" / "leakage_summary.json").read_bytes()
        .decode("utf-8")
        .rstrip("\n")
    )
    if audit.get("passed") is not True:
        raise V3DefinitionError("recorded leakage audit did not pass")

    probe_counts = strict_json_loads(
        (root / "public" / "probe_counts.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    if hashlib.sha256(strict_canonical_json(probe_counts)).hexdigest() != data.get(
        "probe_counts_sha256"
    ):
        raise V3DefinitionError("probe_counts.json does not match probe_counts_sha256")

    return V3Definition(
        definition_sha256=stored_hash,
        dev_view_sha256=data["dev_view_sha256"],
        calibration_view_sha256=data["calibration_view_sha256"],
        dev_seed_table_sha256=data["dev_seed_table_sha256"],
        probe_counts_sha256=data["probe_counts_sha256"],
        catalog_sha256=data["catalog_sha256"],
        support_bundle_sha256=data["support_bundle_sha256"],
        leakage_audit=audit,
        support_certificate_summary={},
        file_entries=tuple(entries),
        task_counts=data["task_counts"],
        probe_counts=probe_counts,
        root=root,
    )


def _rebuild_and_compare(root: Path, data: dict[str, Any]) -> DevTaskCatalog:
    """Rebuild the v2-lineage catalog and require it to equal the frozen files."""
    tg_config = TaskGeneratorConfig(
        root_seed=data["generator_object_root_seed"]
        if "generator_object_root_seed" in data
        else _root_seed_from_generator_json(root),
        train_tasks_per_family=data["train_tasks_per_family"],
        validation_tasks_per_family=(
            data["tuning_tasks_per_family"] + data["calibration_tasks_per_family"]
        ),
        min_events=data["event_min"],
        max_events=data["event_max"],
    )
    catalog, _bundle = build_dev_catalog_v2(tg_config)
    if catalog.catalog_sha256 != data["catalog_sha256"]:
        raise V3DefinitionError(
            f"rebuilt catalog digest {catalog.catalog_sha256} does not match the "
            f"frozen {data['catalog_sha256']}"
        )

    train_tasks = tuple(catalog.meta_train)
    tuning_tasks, calibration_tasks = _split_validation(
        catalog,
        tuning_tasks_per_family=data["tuning_tasks_per_family"],
        calibration_tasks_per_family=data["calibration_tasks_per_family"],
    )
    expected_files = {
        "public/tasks/meta_train.jsonl": ("meta_train", train_tasks),
        "public/tasks/meta_validation_tuning.jsonl": ("meta_validation_tuning", tuning_tasks),
        "public/tasks/meta_validation_calibration.jsonl": (
            "meta_validation_calibration",
            calibration_tasks,
        ),
    }
    for rel_path, (split_role, tasks) in expected_files.items():
        path = root / rel_path
        if not path.is_file():
            raise V3DefinitionError(f"missing task file {rel_path}")
        rebuilt = b"".join(
            strict_canonical_json(_task_to_jsonl_record(t, split_role, i)) + b"\n"
            for i, t in enumerate(tasks)
        )
        if rebuilt != path.read_bytes():
            raise V3DefinitionError(
                f"{rel_path} differs from a fresh rebuild of the v2 lineage"
            )
    return catalog


def _root_seed_from_generator_json(root: Path) -> int:
    generator = strict_json_loads(
        (root / "public" / "generator.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    seed = generator.get("root_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise V3DefinitionError("generator.json has no usable root_seed")
    return seed


def assert_catalog_split_firewall(catalog: DevTaskCatalog) -> Any:
    """Public helper: assert the v2 lineage's train/validation firewall."""
    audit = audit_split_firewall(tuple(catalog.meta_train), tuple(catalog.meta_validation))
    assert_split_firewall(audit)
    return audit


def probe_count_by_kind(tasks: Sequence[MetaTask]) -> dict[str, int]:
    """Count probes per kind across *tasks* (used by freeze reports)."""
    counts: dict[str, int] = {}
    for task in tasks:
        for kind in ProbeKind:
            name = kind.value
            counts[name] = counts.get(name, 0) + len(task.probes.by_kind(kind))
    return counts


# Keep the unused-import linters honest about names re-exported for callers.
_ = (DevSplit, TASK_RECORD_SCHEMA)
