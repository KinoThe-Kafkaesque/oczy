#!/usr/bin/env python3
"""Fail-closed materializer for the ``meta_cortex/v4-r20`` DEV prompt amendments.

This is the G1-lineage analogue of ``scripts/materialize_dev_prompt_repair.py``.
That script pinned the **v2** public DEV view and hard-refused the R20 (v3)
lineage with ``Not the approved v2 public DEV instrument`` (reproduced on
2026-10-05, exit 1).  This one pins the **v3** public DEV view and refuses
anything else just as loudly.

It materializes the two user-approved 2026-09-11 prompt amendments — the
bare-answer system instruction (Amendment A) and the complete transformation
oracle rule descriptions (Amendment B) — over the *frozen v3 task content*, and
records the result as a self-hashed manifest:

    <output>/MANIFEST.json   base hashes, amendment record, task digests
    <output>/TASKS.json      the amended task records (the 105 public DEV tasks)

Fail-closed checks, all performed before anything is written:

- the base directory must be the approved v3 public DEV instrument
  (``dev_view_sha256`` and ``definition_sha256`` pinned; instrument id/version,
  taskgen lineage, prompt registry and the signed ``max_new_tokens``=32 checked);
- the base ``DEFINITION.json`` must self-verify to the pinned hash, list the
  pinned task/registry bytes, and contain no ``sealed/`` directory;
- a fresh in-memory rebuild of the ``oczy/meta-cortex/taskgen/v2-dev`` lineage
  must reproduce the base task bytes exactly, so the base is proven to be a
  record of the generator rather than a drifted snapshot;
- the amended tasks must differ from the base tasks in exactly the approved
  ways: every probe gains the Amendment A system message and nothing else
  changes, and only ``rule_transformation`` ``oracle_context`` headers change
  for Amendment B.

Exit codes: ``0`` success, ``3`` wrong/refused base, ``1`` any other failure,
``2`` argument error.  Nothing is ever written to the base instrument.

DEV only.  No model run, no optimizer step, no threshold, no sealed or
meta-test access, and no edit to ``taskgen.py`` / ``taskgen_v2.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oczy.experiments.meta_cortex.contracts import (  # noqa: E402
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from oczy.experiments.meta_cortex.instrument import (  # noqa: E402
    _task_to_jsonl_record,
)
from oczy.experiments.meta_cortex.instrument_contracts import (  # noqa: E402
    strict_canonical_json,
    strict_json_loads,
)
from oczy.experiments.meta_cortex.instrument_v4_r20 import (  # noqa: E402
    AMENDMENT_RECORD,
    BARE_ANSWER,
    BASE_CATALOG_SHA256,
    BASE_DEFINITION_SHA256,
    BASE_DEV_VIEW_SHA256,
    INSTRUMENT_ID,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_RECORD,
    NAMING_RECORD,
    ORACLE_DESCRIPTIONS,
    V4R20BaseError,
    V4R20DefinitionError,
    _split_validation,
    amend_task,
    verify_base_public_view,
)
from oczy.experiments.meta_cortex.taskgen_v2 import build_dev_catalog_v2  # noqa: E402

SCHEMA = "oczy/r20-v4-r20-materializer/v1"
EXIT_WRONG_BASE = 3

#: Every field of a task except ``probes`` must be untouched by the amendments.
def _digest(obj: Any) -> str:
    return hashlib.sha256(strict_canonical_json(obj)).hexdigest()


def _task_records(catalog: Any, base: bool) -> dict[str, list[dict[str, Any]]]:
    """Serialize train + tuning task records (same shape the instrument writes)."""
    return {
        "meta_train": [
            _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(catalog.meta_train)
        ],
        "meta_validation_tuning": [
            _task_to_jsonl_record(t, "meta_validation_tuning", i)
            for i, t in enumerate(catalog.meta_validation)
        ],
    }


def _records_digest(records: dict[str, list[dict[str, Any]]]) -> str:
    return _digest(records)


def _amendment_diff(base_catalog: Any, amended_catalog: Any) -> dict[str, Any]:
    """Prove the amended tasks differ from the base in exactly the approved ways."""
    oracle_headers_changed = 0
    system_messages_added = 0
    non_probe_field_changes: list[str] = []
    probe_payload_changes: list[str] = []

    base_tasks = list(base_catalog.meta_train) + list(base_catalog.meta_validation)
    amended_tasks = list(amended_catalog.meta_train) + list(amended_catalog.meta_validation)
    if len(base_tasks) != len(amended_tasks):
        raise V4R20DefinitionError("amendment changed the task count")

    for base_task, amended_task in zip(base_tasks, amended_tasks, strict=True):
        if replace(amended_task, probes=base_task.probes) != base_task:
            non_probe_field_changes.append(base_task.family.value)
        for kind in ProbeKind:
            old_probes = base_task.probes.by_kind(kind)
            new_probes = amended_task.probes.by_kind(kind)
            if len(old_probes) != len(new_probes):
                raise V4R20DefinitionError(
                    f"amendment changed the probe count for {base_task.family.value}/{kind.value}"
                )
            for old, new in zip(old_probes, new_probes, strict=True):
                if replace(new, messages=old.messages) != old:
                    probe_payload_changes.append(f"{base_task.family.value}/{kind.value}")
                if not new.messages or new.messages[0].role != "system":
                    raise V4R20DefinitionError("amended probe is missing the system message")
                if new.messages[0].content != BARE_ANSWER:
                    raise V4R20DefinitionError("amended probe system message is not Amendment A")
                if tuple(new.messages[1:]) != tuple(old.messages):
                    if not (
                        base_task.family == TaskFamily.RULE_TRANSFORMATION
                        and kind == ProbeKind.ORACLE_CONTEXT
                    ):
                        raise V4R20DefinitionError(
                            "amendment changed probe content outside the approved scope: "
                            f"{base_task.family.value}/{kind.value}"
                        )
                    oracle_headers_changed += 1
                    if new.messages[2:] != old.messages[1:]:
                        raise V4R20DefinitionError(
                            "Amendment B changed more than the oracle header"
                        )
                system_messages_added += 1

    if non_probe_field_changes:
        raise V4R20DefinitionError(
            f"amendment changed non-probe task fields: {sorted(set(non_probe_field_changes))}"
        )
    if probe_payload_changes:
        raise V4R20DefinitionError(
            f"amendment changed probe payloads outside the approved scope: {sorted(set(probe_payload_changes))}"
        )
    return {
        "system_messages_added": system_messages_added,
        "oracle_headers_rewritten": oracle_headers_changed,
        "non_probe_task_fields_changed": 0,
        "probe_payloads_changed_outside_amendment_b": 0,
    }


def _expected_amendment_diff(records: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Count the amendment surface the base records imply, for a cross-check."""
    probes = 0
    oracle = 0
    for split in ("meta_train", "meta_validation_tuning"):
        for record in records[split]:
            for kind, entries in record["probes"].items():
                for _entry in entries:
                    probes += 1
                    if record["family"] == "rule_transformation" and kind == "oracle_context":
                        oracle += 1
    return {"system_messages_added": probes, "oracle_headers_rewritten": oracle}


def materialize(base_public_root: Path, output: Path, *, root_seed: int = 20260709) -> dict[str, Any]:
    """Materialize the approved amendments over the pinned v3 base.  Fail-closed."""
    base_public_root = Path(base_public_root)
    output = Path(output)

    # 1. Fail-closed base verification (wrong base raises loudly).
    base_catalog = verify_base_public_view(base_public_root)
    dev_view = strict_json_loads(
        (base_public_root / "DEV_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    definition = strict_json_loads(
        (base_public_root.parent / "DEFINITION.json").read_bytes().decode("utf-8").rstrip("\n")
    )

    # 2. Rebuild the v2-dev lineage and require the base task bytes to match.
    tg_config = TaskGeneratorConfig(
        root_seed=root_seed,
        train_tasks_per_family=dev_view["train_tasks_per_family"],
        validation_tasks_per_family=(
            dev_view["tuning_tasks_per_family"] + definition["calibration_tasks_per_family"]
        ),
        min_events=definition["event_min"],
        max_events=definition["event_max"],
    )
    rebuilt, _bundle = build_dev_catalog_v2(tg_config)
    if rebuilt.catalog_sha256 != BASE_CATALOG_SHA256:
        raise V4R20DefinitionError("rebuilt lineage digest is not the pinned v3 catalog")
    tuning_tasks, calibration_tasks = _split_validation(
        rebuilt,
        tuning_tasks_per_family=dev_view["tuning_tasks_per_family"],
        calibration_tasks_per_family=definition["calibration_tasks_per_family"],
    )
    base_records = {
        "meta_train": [
            _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(rebuilt.meta_train)
        ],
        "meta_validation_tuning": [
            _task_to_jsonl_record(t, "meta_validation_tuning", i) for i, t in enumerate(tuning_tasks)
        ],
    }
    for rel, key in (
        ("public/tasks/meta_train.jsonl", "meta_train"),
        ("public/tasks/meta_validation_tuning.jsonl", "meta_validation_tuning"),
    ):
        rebuilt_bytes = b"".join(
            strict_canonical_json(r) + b"\n" for r in base_records[key]
        )
        if hashlib.sha256(rebuilt_bytes).hexdigest() != _base_task_file_sha256(definition, rel):
            raise V4R20DefinitionError(
                f"in-memory rebuild of {rel} does not reproduce the pinned v3 bytes"
            )
    # The loaded base view must equal the rebuild record-for-record.
    loaded_records = _task_records(base_catalog, base=True)
    if _records_digest(loaded_records) != _records_digest(base_records):
        raise V4R20DefinitionError("loaded base view differs from a fresh lineage rebuild")
    if _expected_amendment_diff(base_records) != _expected_amendment_diff(loaded_records):
        raise V4R20DefinitionError("base view probe inventory is inconsistent")

    # 3. Apply the amendments and prove the diff is exactly the approved surface.
    amended_train = tuple(amend_task(t) for t in rebuilt.meta_train)
    amended_tuning = tuple(amend_task(t) for t in tuning_tasks)
    amended_calibration = tuple(amend_task(t) for t in calibration_tasks)
    amended_records = {
        "meta_train": [
            _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(amended_train)
        ],
        "meta_validation_tuning": [
            _task_to_jsonl_record(t, "meta_validation_tuning", i)
            for i, t in enumerate(amended_tuning)
        ],
    }
    amended_calibration_records = [
        _task_to_jsonl_record(t, "meta_validation_calibration", i)
        for i, t in enumerate(amended_calibration)
    ]

    from oczy.experiments.meta_cortex.contracts import DevTaskCatalog

    diff = _amendment_diff(
        base_catalog,
        DevTaskCatalog(
            meta_train=amended_train,
            meta_validation=amended_tuning,
            catalog_sha256=base_catalog.catalog_sha256,
            split_audit=base_catalog.split_audit,
        ),
    )
    expected = _expected_amendment_diff(base_records)
    if diff["system_messages_added"] != expected["system_messages_added"]:
        raise V4R20DefinitionError(
            f"Amendment A applied to {diff['system_messages_added']} probes, expected "
            f"{expected['system_messages_added']}"
        )
    if diff["oracle_headers_rewritten"] != expected["oracle_headers_rewritten"]:
        raise V4R20DefinitionError(
            f"Amendment B rewrote {diff['oracle_headers_rewritten']} oracle headers, expected "
            f"{expected['oracle_headers_rewritten']}"
        )

    # 4. Write the manifest + tasks (never inside the base instrument).
    if output.exists():
        raise V4R20DefinitionError(f"Output directory already exists: {output}")
    if output.resolve().is_relative_to(base_public_root.parent.resolve()):
        raise V4R20DefinitionError("Do not write inside the frozen v3 instrument")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "instrument_id": INSTRUMENT_ID,
        "scope": "DEV_DIAGNOSTIC_ONLY",
        "meta_test_authorized": False,
        "model_runs": 0,
        "thresholds_selected": False,
        "naming": NAMING_RECORD,
        "base_instrument_id": "meta_cortex/v3",
        "base_dev_view_sha256": dev_view["dev_view_sha256"],
        "base_definition_sha256": definition["definition_sha256"],
        "base_catalog_sha256": BASE_CATALOG_SHA256,
        "base_prompt_registry_sha256": dev_view["prompt_registry_sha256"],
        "base_max_new_tokens": dev_view["max_new_tokens"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_new_tokens_change": MAX_NEW_TOKENS_RECORD,
        "amendments": AMENDMENT_RECORD,
        "amendment_diff": diff,
        "base_tasks_sha256": _records_digest(base_records),
        "amended_tasks_sha256": _records_digest(amended_records),
        "amended_calibration_tasks_sha256": _digest(amended_calibration_records),
        "base_public_task_files": {
            rel: _base_task_file_sha256(definition, rel)
            for rel in (
                "public/tasks/meta_train.jsonl",
                "public/tasks/meta_validation_tuning.jsonl",
                "public/tasks/meta_validation_calibration.jsonl",
            )
        },
        "oracle_descriptions": ORACLE_DESCRIPTIONS,
        "format_instruction": BARE_ANSWER,
        "task_counts": {
            "meta_train": len(amended_train),
            "meta_validation_tuning": len(amended_tuning),
            "meta_validation_calibration": len(amended_calibration),
        },
        "transform_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest = dict(body, manifest_sha256=_digest(body))

    output.mkdir(parents=True, exist_ok=False)
    (output / "MANIFEST.json").write_bytes(strict_canonical_json(manifest) + b"\n")
    (output / "TASKS.json").write_bytes(
        strict_canonical_json(
            {
                **amended_records,
                "meta_validation_calibration": amended_calibration_records,
            }
        )
        + b"\n"
    )
    return manifest


def _base_task_file_sha256(definition: dict[str, Any], rel: str) -> str:
    for entry in definition["public_files"]:
        if entry["path"] == rel:
            value = entry["sha256"]
            if not isinstance(value, str):
                raise V4R20DefinitionError(f"{rel} hash entry is malformed")
            return value
    raise V4R20DefinitionError(f"base definition does not list {rel}")


def load_materialization(base_public_root: Path, root: Path) -> dict[str, Any]:
    """Re-verify a materialized amendment bundle against the pinned base."""
    base_public_root = Path(base_public_root)
    root = Path(root)
    manifest = strict_json_loads((root / "MANIFEST.json").read_bytes())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if _digest(body) != manifest["manifest_sha256"]:
        raise V4R20DefinitionError("Amendment manifest hash mismatch")
    verify_base_public_view(base_public_root)
    # Rebuild deterministically and compare digests, without writing anything.
    dev_view = strict_json_loads(
        (base_public_root / "DEV_VIEW.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    definition = strict_json_loads(
        (base_public_root.parent / "DEFINITION.json").read_bytes().decode("utf-8").rstrip("\n")
    )
    if body["base_dev_view_sha256"] != dev_view["dev_view_sha256"]:
        raise V4R20DefinitionError("Manifest binds a different base dev view")
    if body["base_definition_sha256"] != definition["definition_sha256"]:
        raise V4R20DefinitionError("Manifest binds a different base definition")
    if body["base_dev_view_sha256"] != BASE_DEV_VIEW_SHA256:
        raise V4R20DefinitionError("Manifest base dev view is not the pinned v3 value")
    if body["base_definition_sha256"] != BASE_DEFINITION_SHA256:
        raise V4R20DefinitionError("Manifest base definition is not the pinned v3 value")
    if body["max_new_tokens"] != MAX_NEW_TOKENS:
        raise V4R20DefinitionError("Manifest max_new_tokens is not the authorized value")
    tasks = strict_json_loads((root / "TASKS.json").read_bytes())
    if tasks["meta_train"] != json.loads(strict_canonical_json(tasks["meta_train"]).decode()):
        raise V4R20DefinitionError("TASKS.json meta_train is not canonical")
    if _digest({k: tasks[k] for k in ("meta_train", "meta_validation_tuning")}) != body[
        "amended_tasks_sha256"
    ]:
        raise V4R20DefinitionError("TASKS.json does not match amended_tasks_sha256")
    if _digest(tasks["meta_validation_calibration"]) != body["amended_calibration_tasks_sha256"]:
        raise V4R20DefinitionError("TASKS.json calibration records do not match the manifest")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True, help="the v3 public DEV view")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--root-seed", type=int, default=20260709)
    parser.add_argument("--verify-only", action="store_true", help="re-verify an existing bundle")
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            manifest = load_materialization(args.public_root, args.output)
        else:
            manifest = materialize(args.public_root, args.output, root_seed=args.root_seed)
    except V4R20BaseError as exc:
        print(f"REFUSED (wrong base): {exc}", file=sys.stderr)
        return EXIT_WRONG_BASE
    except V4R20DefinitionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"{manifest['instrument_id']}: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
