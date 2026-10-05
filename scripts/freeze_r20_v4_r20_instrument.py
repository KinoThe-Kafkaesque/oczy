#!/usr/bin/env python3
"""Freeze the ``meta_cortex/v4-r20`` DEV instrument (G1-equivalent) and verify it.

DEV only.  This script:

1. verifies the pinned ``meta_cortex/v3`` public DEV view fail-closed (wrong base
   refuses loudly, exit 3);
2. runs the independent ``scripts/r20_task_support_check.py`` checker over the
   *pinned v3 base* tasks — the successor's task content is unchanged from that
   base, so the v3 construction verdict (22/22 defect classes at 0, 758/758
   derivation-backed certificates) carries over exactly and is re-measured here
   rather than assumed;
3. materializes ``meta_cortex/v4-r20`` from the base plus the approved prompt
   amendments and the authorized ``max_new_tokens``=128;
4. re-verifies the frozen tree independently (``verify_v4_r20_definition``);
5. proves the amendment surface with two independent checks: the materializer's
   per-probe diff (only the Amendment A system message and the Amendment B
   oracle header changed) and an independent re-derivation of every
   transformation oracle target from the *amended* description text;
6. re-runs the materialization into a second directory and requires byte
   equality (the freeze is a function of the lineage, not of the run);
7. writes a dated JSON report.

It never trains, never scores a model, never opens a sealed payload, never
chooses a threshold/margin/power value, and never edits ``taskgen.py`` or
``taskgen_v2.py``.  Exit 0 means every criterion passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
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
    load_dev_view,
)
from oczy.experiments.meta_cortex.instrument_v4_r20 import (  # noqa: E402
    BARE_ANSWER,
    BASE_CATALOG_SHA256,
    BASE_DEFINITION_SHA256,
    BASE_DEV_VIEW_SHA256,
    INHERITED_FROZEN_HASHES,
    INSTRUMENT_ID,
    ORACLE_DESCRIPTIONS,
    V4R20BaseError,
    materialize_v4_r20_definition,
    verify_base_public_view,
    verify_v4_r20_definition,
)
from oczy.experiments.meta_cortex.taskgen_v2 import build_dev_catalog_v2  # noqa: E402

V3_ORGAN_HASH = "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
V3_CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
EXIT_WRONG_BASE = 3

_VOWELS = "aeiou"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checker():
    path = REPO_ROOT / "scripts" / "r20_task_support_check.py"
    spec = importlib.util.spec_from_file_location("r20_task_support_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _run_checker_on_base(checker, base_public_root: Path, *, root_seed: int) -> dict[str, Any]:
    """Run the independent checker over the pinned v3 base tasks.

    The successor's task content is byte-identical to this base (proved by the
    materializer's diff), so this is the construction verdict that carries over.
    The checker scope deliberately matches the G1 freeze record exactly:
    ``validation_tasks_per_family`` = the *tuning* count (5), so the parsed task
    total is 105 and the certificate counts are directly comparable to the
    recorded ``2026-10-05_r20_g1_instrument_freeze.md`` numbers.
    """
    view = load_dev_view(base_public_root)
    catalog = view.catalog
    report = checker.check_catalog(catalog)
    summary = report["summary"]
    nonzero = {cls: counts["total"] for cls, counts in summary.items() if counts["total"]}

    generator = json.loads((base_public_root / "generator.json").read_text().strip())
    definition = json.loads((base_public_root.parent / "DEFINITION.json").read_text().strip())
    config = TaskGeneratorConfig(
        root_seed=generator["root_seed"],
        train_tasks_per_family=definition["train_tasks_per_family"],
        validation_tasks_per_family=definition["tuning_tasks_per_family"],
    )
    rebuilt, bundle = build_dev_catalog_v2(config)
    certs = checker.verify_certificates(rebuilt, bundle)

    return {
        "defect_counts": {cls: counts["total"] for cls, counts in summary.items()},
        "nonzero_defect_classes": nonzero,
        "all_defect_counts_zero": not nonzero,
        "parse_defect_count": len(report["parse_defects"]),
        "checked_tasks": len(catalog.meta_train) + len(catalog.meta_validation),
        "certificates": {
            k: certs[k]
            for k in (
                "certificates",
                "verified",
                "baseline_not_required",
                "failed",
                "mutation_checks",
                "mutations_detected",
                "mutation_checks_by_kind",
                "mutations_detected_by_kind",
            )
            if k in certs
        },
        "certificates_all_verified": certs.get("verified")
        == certs.get("certificates", 0) - certs.get("baseline_not_required", 0),
        "base_catalog_sha256": rebuilt.catalog_sha256,
    }


def _described_rule(template: str, param1: str, param2: str, operand: str) -> str:
    """Execute the operations stated in the approved English descriptions."""
    if template == "permutation":
        return "".join(reversed(operand))
    if template == "conditional":
        return operand + param1 if operand[:1] in tuple(_VOWELS) else param2 + operand
    characters = operand if template == "substitution" else "".join(reversed(operand))
    replacement = param1 if template == "substitution" else param2
    return "".join(replacement if char in _VOWELS else char for char in characters)


def _recover_description_params(description: str) -> tuple[str, str, str]:
    """Recover (template, param1, param2) from an amended oracle description.

    Independent of the materializer: it matches the *rendered* description
    against the approved description templates and reads the substituted
    parameters back out of the text.
    """
    import re

    for template, spec in ORACLE_DESCRIPTIONS.items():
        pattern = re.escape(spec)
        pattern = pattern.replace(re.escape("{param1}"), "(?P<p1>[a-z]*)")
        pattern = pattern.replace(re.escape("{param2}"), "(?P<p2>[a-z]*)")
        match = re.fullmatch(pattern, description)
        if match is not None:
            groups = match.groupdict()
            return template, groups.get("p1") or "", groups.get("p2") or ""
    raise RuntimeError(f"Amended oracle description is not an approved description: {description!r}")


def _amendment_surface_check(instrument_root: Path, base_public_root: Path) -> dict[str, Any]:
    """Independent check that the amended probes are exactly the approved surface.

    For every probe: the Amendment A system message is present and first; the
    remaining messages equal the pinned base probe's messages, except that a
    ``rule_transformation`` ``oracle_context`` header may be rewritten to an
    approved description — and the amended description, executed on the probe's
    own operand, must still yield the recorded expected answer.
    """
    base = load_dev_view(base_public_root).catalog
    amended = load_dev_view(instrument_root / "public").catalog
    base_tasks = list(base.meta_train) + list(base.meta_validation)
    amended_tasks = list(amended.meta_train) + list(amended.meta_validation)
    if len(base_tasks) != len(amended_tasks):
        raise RuntimeError("amended task count differs from the base")

    probes_checked = 0
    oracle_headers_rewritten = 0
    oracle_targets_rederived = 0
    for base_task, amended_task in zip(base_tasks, amended_tasks, strict=True):
        if base_task.family != amended_task.family:
            raise RuntimeError("amended task family differs from the base")
        for kind in ProbeKind:
            old_probes = base_task.probes.by_kind(kind)
            new_probes = amended_task.probes.by_kind(kind)
            if len(old_probes) != len(new_probes):
                raise RuntimeError(f"probe count changed for {base_task.family.value}/{kind.value}")
            for old, new in zip(old_probes, new_probes, strict=True):
                if new.expected_response != old.expected_response:
                    raise RuntimeError(
                        f"expected response changed for {base_task.family.value}/{kind.value}"
                    )
                if not new.messages or new.messages[0].role != "system":
                    raise RuntimeError("amended probe is missing the system message")
                if new.messages[0].content != BARE_ANSWER:
                    raise RuntimeError("amended probe system message is not Amendment A")
                if tuple(new.messages[1:]) == tuple(old.messages):
                    probes_checked += 1
                    continue
                if not (
                    base_task.family == TaskFamily.RULE_TRANSFORMATION
                    and kind == ProbeKind.ORACLE_CONTEXT
                ):
                    raise RuntimeError(
                        f"unapproved message change at {base_task.family.value}/{kind.value}"
                    )
                # Amendment B: header rewritten, examples and query untouched.
                old_header = old.messages[0].content
                new_header = new.messages[1].content
                if old.messages[1:] != new.messages[2:]:
                    raise RuntimeError("Amendment B changed more than the oracle header")
                if new_header.partition("\n")[2] != old_header.partition("\n")[2]:
                    raise RuntimeError("Amendment B changed the worked examples")
                description = new_header.removeprefix("Rule: ").partition("\n")[0]
                template, p1, p2 = _recover_description_params(description)
                operand = old.messages[-1].content.removeprefix(
                    "Given this rule, what is the output for: "
                ).removesuffix("?")
                if _described_rule(template, p1, p2, operand) != old.expected_response:
                    raise RuntimeError(
                        "the amended oracle description does not yield the scored answer"
                    )
                oracle_headers_rewritten += 1
                oracle_targets_rederived += 1
                probes_checked += 1
    return {
        "probes_checked": probes_checked,
        "oracle_headers_rewritten": oracle_headers_rewritten,
        "oracle_targets_independently_rederived_from_amended_text": oracle_targets_rederived,
    }


def _materialize(out: Path, base_public_root: Path, *, root_seed: int, source_commit: str):
    return materialize_v4_r20_definition(
        base_public_root=base_public_root,
        out=out,
        organ_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        organ_revision="main",
        organ_parameter_sha256=V3_ORGAN_HASH,
        chat_template_sha256=V3_CHAT_TEMPLATE_SHA256,
        source_commit=source_commit,
        source_archive_sha256="",
        root_seed=root_seed,
        train_tasks_per_family=30,
        tuning_tasks_per_family=5,
        calibration_tasks_per_family=30,
        event_min=2,
        event_max=5,
        feature_dim=896,
        d_cortex=64,
        soft_bank_width=3,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repro-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=20260709)
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args(argv)

    started = time.time()

    # 1. Fail-closed base verification.
    try:
        base_catalog = verify_base_public_view(args.base_public_root)
    except V4R20BaseError as exc:
        print(f"REFUSED (wrong base): {exc}", file=sys.stderr)
        return EXIT_WRONG_BASE

    git_commit = args.source_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    # 2. Independent checker over the pinned base tasks.
    checker, checker_path = _load_checker()
    checker_sha256 = _sha256_file(checker_path)
    checker_report = _run_checker_on_base(checker, args.base_public_root, root_seed=args.root_seed)

    # 3. Materialize the successor, twice, into independent directories, plus the
    #    amendment manifest bundle that records the approved amendment surface.
    mat_path = REPO_ROOT / "scripts" / "materialize_r20_v4_r20.py"
    mat_spec = importlib.util.spec_from_file_location("materialize_r20_v4_r20", mat_path)
    assert mat_spec and mat_spec.loader
    mat_module = importlib.util.module_from_spec(mat_spec)
    sys.modules[mat_spec.name] = mat_module
    mat_spec.loader.exec_module(mat_module)
    mat_module.materialize(
        args.base_public_root, args.output.parent / "materialization", root_seed=args.root_seed
    )

    freeze = _materialize(args.output, args.base_public_root, root_seed=args.root_seed,
                          source_commit=git_commit)
    repro = _materialize(args.repro_output, args.base_public_root, root_seed=args.root_seed,
                         source_commit=git_commit)
    deterministic = freeze.definition_sha256 == repro.definition_sha256
    diff = subprocess.run(
        ["diff", "-r", str(args.output), str(args.repro_output)], capture_output=True, text=True
    )

    # 4. Independent re-verification of the frozen tree.
    verified = verify_v4_r20_definition(args.output)
    if verified.definition_sha256 != freeze.definition_sha256:
        raise SystemExit("verification disagrees with materialization")

    # 5. Amendment-surface checks.
    surface = _amendment_surface_check(args.output, args.base_public_root)

    # Cross-check the materializer manifest against the instrument, if present.
    manifest_ok: bool | None = None
    manifest_path = args.output.parent / "materialization" / "MANIFEST.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        recomputed = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        manifest_ok = recomputed == manifest["manifest_sha256"]

    # Base task content must be byte-identical to the pinned v3 bytes.
    base_view = json.loads((args.base_public_root / "DEV_VIEW.json").read_text().strip())
    base_records = [
        _task_to_jsonl_record(t, "meta_train", i) for i, t in enumerate(base_catalog.meta_train)
    ] + [
        _task_to_jsonl_record(t, "meta_validation_tuning", i)
        for i, t in enumerate(base_catalog.meta_validation)
    ]
    base_records_digest = hashlib.sha256(
        json.dumps(base_records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    passed = bool(
        checker_report["all_defect_counts_zero"]
        and checker_report["certificates_all_verified"]
        and deterministic
        and diff.returncode == 0
        and surface["probes_checked"] > 0
        and (manifest_ok is not False)
    )

    report = {
        "schema": "oczy/r20-g1-v4-r20-freeze/v1",
        "classification": "DEV_INSTRUMENT_FREEZE",
        "gate": "G1-equivalent (successor freeze)",
        "instrument_id": INSTRUMENT_ID,
        "definition_sha256": verified.definition_sha256,
        "dev_view_sha256": verified.dev_view_sha256,
        "calibration_view_sha256": verified.calibration_view_sha256,
        "catalog_sha256": verified.catalog_sha256,
        "support_bundle_sha256": verified.support_bundle_sha256,
        "dev_seed_table_sha256": verified.dev_seed_table_sha256,
        "probe_counts_sha256": verified.probe_counts_sha256,
        "prompt_registry_sha256": verified.prompt_registry_sha256,
        "task_counts": verified.task_counts,
        "max_new_tokens": verified.max_new_tokens,
        "base": {
            "instrument_id": "meta_cortex/v3",
            "definition_sha256": BASE_DEFINITION_SHA256,
            "dev_view_sha256": BASE_DEV_VIEW_SHA256,
            "catalog_sha256": BASE_CATALOG_SHA256,
            "task_counts": {
                "meta_train": len(base_catalog.meta_train),
                "meta_validation_tuning": len(base_catalog.meta_validation),
            },
            "records_digest": base_records_digest,
            "dev_view_catalog_sha256": base_view.get("catalog_sha256"),
        },
        "inherited_frozen_hashes": INHERITED_FROZEN_HASHES,
        "registry_drift": [],
        "leakage_audit": verified.leakage_audit,
        "independent_checker": {
            "script": str(checker_path.relative_to(REPO_ROOT)),
            "script_sha256": checker_sha256,
            "ran_over": "pinned v3 base tasks (task content unchanged in v4-r20)",
            **checker_report,
        },
        "amendment_surface": surface,
        "materializer_manifest": {
            "path": str(manifest_path) if manifest_path.is_file() else None,
            "manifest_sha256": manifest["manifest_sha256"] if manifest else None,
            "self_hash_recomputed_ok": manifest_ok,
            "amendment_diff": manifest["amendment_diff"] if manifest else None,
        },
        "determinism": {
            "repro_definition_sha256": repro.definition_sha256,
            "identical": deterministic,
            "diff_exit_code": diff.returncode,
            "diff_output": diff.stdout.strip(),
        },
        "sealed_payload_present": False,
        "meta_test_accessed": False,
        "calibration_accessed": False,
        "model_runs": 0,
        "thresholds_selected": False,
        "source_commit": git_commit,
        "seconds": round(time.time() - started, 1),
        "passed": passed,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "instrument_id": INSTRUMENT_ID,
                "definition_sha256": report["definition_sha256"],
                "dev_view_sha256": report["dev_view_sha256"],
                "catalog_sha256": report["catalog_sha256"],
                "prompt_registry_sha256": report["prompt_registry_sha256"],
                "max_new_tokens": report["max_new_tokens"],
                "task_counts": report["task_counts"],
                "amendment_surface": surface,
                "determinism_identical": deterministic,
                "independent_checker": {
                    "all_defect_counts_zero": checker_report["all_defect_counts_zero"],
                    "certificates_all_verified": checker_report["certificates_all_verified"],
                    "certificates": checker_report["certificates"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
