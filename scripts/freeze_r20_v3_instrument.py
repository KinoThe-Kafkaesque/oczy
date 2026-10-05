"""Freeze the ``meta_cortex/v3`` DEV instrument (gate G1) and verify it.

DEV only. This script materializes the task-support-repaired instrument from
the ``oczy/meta-cortex/taskgen/v2-dev`` lineage, runs the independent
``scripts/r20_task_support_check.py`` checker over the materialized public
tasks, and writes a dated report.  It never trains, never scores a model, never
opens a sealed payload, and never chooses a threshold, margin or power value.

Exit code 0 means every G1 criterion passed.  Any failure exits nonzero and the
report records why.
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

from oczy.experiments.meta_cortex.instrument_v3 import (  # noqa: E402
    V2_FROZEN_HASHES,
    materialize_v3_definition,
    verify_v3_definition,
)

# Identities carried from the frozen v2 instrument.  The organ hash is verified
# separately by the organ identity probe; it is passed in here so the freeze is
# hash-bound to an organ this machine actually reproduced.
V2_ORGAN_HASH = "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"
V2_CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checker():
    """Import the independent checker by path (it is a script, not a package)."""
    path = REPO_ROOT / "scripts" / "r20_task_support_check.py"
    spec = importlib.util.spec_from_file_location("r20_task_support_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checker from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules; register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _run_checker(checker, public_root: Path) -> dict[str, Any]:
    """Run the independent checker over the frozen v3 public tasks.

    Reads only the public DEV view (train + tuning).  Re-derives every scored
    target from rendered text and verifies every support certificate.
    """
    from oczy.experiments.meta_cortex.contracts import TaskGeneratorConfig
    from oczy.experiments.meta_cortex.instrument import load_dev_view

    view = load_dev_view(public_root)
    catalog = view.catalog
    report = checker.check_catalog(catalog)
    summary = report["summary"]
    nonzero = {
        cls: counts["total"]
        for cls, counts in summary.items()
        if counts["total"]
    }

    # Rebuild the v2 lineage at the frozen split sizes and verify certificates
    # against the independent derived-target machinery.
    dev_view = json.loads(
        (public_root / "DEV_VIEW.json").read_text().strip()
    )
    generator = json.loads((public_root / "generator.json").read_text().strip())
    config = TaskGeneratorConfig(
        root_seed=generator["root_seed"],
        train_tasks_per_family=dev_view["train_tasks_per_family"],
        validation_tasks_per_family=dev_view["tuning_tasks_per_family"],
    )
    from oczy.experiments.meta_cortex.taskgen_v2 import build_dev_catalog_v2

    v2_catalog, bundle = build_dev_catalog_v2(config)
    certs = checker.verify_certificates(v2_catalog, bundle)

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
        "certificates_all_verified": certs.get("verified") == certs.get("certificates")
        - certs.get("baseline_not_required", 0),
        "catalog_sha256": v2_catalog.catalog_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="freeze directory")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    parser.add_argument("--root-seed", type=int, default=20260709)
    parser.add_argument("--train-tasks-per-family", type=int, default=30)
    parser.add_argument("--tuning-tasks-per-family", type=int, default=5)
    parser.add_argument("--calibration-tasks-per-family", type=int, default=30)
    parser.add_argument("--event-min", type=int, default=2)
    parser.add_argument("--event-max", type=int, default=5)
    parser.add_argument(
        "--organ-hash",
        default=V2_ORGAN_HASH,
        help="frozen organ identity; must be reproduced by the organ probe",
    )
    parser.add_argument(
        "--chat-template-sha256", default=V2_CHAT_TEMPLATE_SHA256
    )
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--source-archive-sha256", default="")
    args = parser.parse_args(argv)

    started = time.time()
    checker, checker_path = _load_checker()
    checker_sha256 = _sha256_file(checker_path)

    # Identity the freeze is bound to.
    git_commit = args.source_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    freeze = materialize_v3_definition(
        out=args.output,
        organ_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        organ_revision="main",
        organ_parameter_sha256=args.organ_hash,
        chat_template_sha256=args.chat_template_sha256,
        source_commit=git_commit,
        source_archive_sha256=args.source_archive_sha256,
        root_seed=args.root_seed,
        train_tasks_per_family=args.train_tasks_per_family,
        tuning_tasks_per_family=args.tuning_tasks_per_family,
        calibration_tasks_per_family=args.calibration_tasks_per_family,
        event_min=args.event_min,
        event_max=args.event_max,
        feature_dim=896,
        d_cortex=64,
        soft_bank_width=3,
        max_new_tokens=32,
        abstain_threshold="0",
    )

    # Independent re-verification of the frozen tree (hashes + rebuild).
    verified = verify_v3_definition(args.output)
    if verified.definition_sha256 != freeze.definition_sha256:
        raise SystemExit("verification disagrees with materialization")

    checker_report = _run_checker(checker, args.output / "public")
    if not checker_report["all_defect_counts_zero"]:
        print(
            "FAILED: independent checker found nonzero defect counts: "
            f"{checker_report['nonzero_defect_classes']}",
            file=sys.stderr,
        )
    if not checker_report["certificates_all_verified"]:
        print(
            "FAILED: not every support certificate verified",
            file=sys.stderr,
        )

    report = {
        "schema": "oczy/r20-g1-freeze/v1",
        "classification": "DEV_INSTRUMENT_FREEZE",
        "gate": "G1",
        "instrument_id": verified.definition_sha256 and "meta_cortex/v3",
        "definition_sha256": verified.definition_sha256,
        "dev_view_sha256": verified.dev_view_sha256,
        "calibration_view_sha256": verified.calibration_view_sha256,
        "catalog_sha256": verified.catalog_sha256,
        "support_bundle_sha256": verified.support_bundle_sha256,
        "dev_seed_table_sha256": verified.dev_seed_table_sha256,
        "probe_counts_sha256": verified.probe_counts_sha256,
        "task_counts": verified.task_counts,
        "probe_counts": verified.probe_counts,
        "inherited_v2_registry_hashes": V2_FROZEN_HASHES,
        "registry_drift": [],
        "leakage_audit": verified.leakage_audit,
        "independent_checker": {
            "script": str(checker_path.relative_to(REPO_ROOT)),
            "script_sha256": checker_sha256,
            **checker_report,
        },
        "sealed_payload_present": False,
        "meta_test_accessed": False,
        "model_runs": 0,
        "thresholds_selected": False,
        "source_commit": git_commit,
        "seconds": round(time.time() - started, 1),
        "passed": bool(
            checker_report["all_defect_counts_zero"]
            and checker_report["certificates_all_verified"]
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "definition_sha256": report["definition_sha256"],
                "dev_view_sha256": report["dev_view_sha256"],
                "catalog_sha256": report["catalog_sha256"],
                "task_counts": report["task_counts"],
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
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
