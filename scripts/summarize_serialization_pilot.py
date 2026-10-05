"""Summarize all registered pilot conditions without selecting seeds or probes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.serialization_pilot_contract import read_manifest, recovery, write_json  # noqa: E402


def summarize(root, manifest):
    executions = json.loads((root / "executions.json").read_text())
    if [r["phase"] for r in executions] != ["reference", "restore_text", "train", "restore_soft"]:
        raise ValueError("Missing, extra or reordered execution phases")
    for execution in executions:
        if execution["exit_code"] != 0 or not execution["network_disabled"]:
            raise ValueError("Failed or online execution")
        if execution["manifest_sha256"] != manifest["manifest_sha256"] or execution["runtime_manifest_sha256"] != manifest["runtime_manifest_sha256"]:
            raise ValueError("Execution provenance mismatch")
        if execution["filesystem_firewall"]["visible_forbidden_files"]:
            raise ValueError("Filesystem isolation failed")
    phases = {phase: json.loads((root / phase / "results.json").read_text())
              for phase in ("reference", "restore_text", "train", "restore_soft")}
    for phase, result in phases.items():
        if result["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Result belongs to another pilot")
        if result["organ_hash_before"] != manifest["organ_hash"] or result["organ_hash_after"] != manifest["organ_hash"]:
            raise ValueError("Organ identity mismatch")
        if result["scorer_sha256"] != manifest["scorer_sha256"]:
            raise ValueError("Scorer mismatch")
        if phase != "train" and result["optimizer_steps"] != 0:
            raise ValueError("Optimization in evaluation phase")
        if result["meta_test_accessed"]:
            raise ValueError("Unauthorized meta-test access")
    if phases["train"]["heldout_examples_read"] != 0 or phases["restore_soft"]["training_files_read"] != 0 or phases["restore_soft"]["raw_text_artifacts_read"] != 0:
        raise ValueError("Phase data boundary failed")
    if len({r["pid"] for r in phases.values()}) != len(phases):
        raise ValueError("Pilot phases did not run in distinct processes")
    reference = phases["reference"]["rows"]
    replay = phases["restore_text"]["rows"]
    for row in (r for r in reference if r["condition"] == "with_context"):
        copied = next(r for r in replay if r["condition"] == "raw_reloaded" and r["pattern"] == row["pattern"] and r["input"] == row["input"])
        if any(row[k] != copied[k] for k in ("prompt_sha256", "target", "generated", "correct")):
            raise ValueError("Raw context roundtrip failed")
    all_rows = reference + replay + phases["restore_soft"]["rows"]
    summaries = []
    patterns = sorted({r["pattern"] for r in reference})
    for pattern in patterns:
        selected = [r for r in all_rows if r["pattern"] == pattern]
        no_context = sum(r["correct"] for r in selected if r["condition"] == "no_context") / 4
        with_context = sum(r["correct"] for r in selected if r["condition"] == "with_context") / 4
        for condition in manifest["conditions"]:
            seeds = manifest["training"]["seeds"] if condition.startswith("soft_") else [None]
            for seed in seeds:
                rows = [r for r in selected if r["condition"] == condition and r["seed"] == seed]
                if len(rows) != 4 or len({r["input"] for r in rows}) != 4:
                    raise ValueError("Missing, extra or duplicate held-out probes")
                correct = sum(r["correct"] for r in rows)
                ratio = recovery(correct / 4, no_context, with_context)
                summaries.append({"pattern": pattern, "condition": condition, "seed": seed,
                                  "correct": correct, "total": 4, "accuracy": correct / 4,
                                  "recovery": ratio, "recovery_defined": ratio is not None,
                                  "meets_reference_30_percent": None if ratio is None else ratio >= manifest["measurement"]["reference_recovery_criterion"]})
    trajectory = [json.loads(line) for line in (root / "train/trajectory.jsonl").read_text().splitlines()]
    expected_updates = len(patterns) * len(manifest["training"]["seeds"]) * manifest["training"]["steps"]
    if len(trajectory) != expected_updates or phases["train"]["optimizer_steps"] != expected_updates:
        raise ValueError("Incomplete optimization budget")
    for pattern in patterns:
        for seed in manifest["training"]["seeds"]:
            steps = [r["step"] for r in trajectory if r["pattern"] == pattern and r["seed"] == seed]
            if steps != list(range(1, manifest["training"]["steps"] + 1)):
                raise ValueError("Missing, duplicate or unordered training steps")
    gradients = [r["gradient_norm_before_clip"] for r in trajectory]
    if any(not math.isfinite(g) or g <= 0 for g in gradients):
        raise ValueError("Invalid training gradients")
    return {"schema": "oczy/r23.5-dev-pilot-report/v1", "date": "2026-09-12", "classification": "DEV_PILOT_ONLY",
            "manifest_sha256": manifest["manifest_sha256"], "phases": phases, "summary": summaries,
            "trajectory": trajectory, "gradient_distribution": {"min": min(gradients), "max": max(gradients),
                "clipped_updates": sum(g > manifest["training"]["gradient_clip_norm"] for g in gradients), "total_updates": len(gradients)},
            "raw_roundtrip_exact": True, "distinct_processes_verified": True,
            "executions": executions,
            "limitations": ["Two simple DEV patterns and four held-out inputs each", "Three seeds and one fixed optimization budget; no best-checkpoint search",
                            "No hidden-layer vector sweep or full-study hypothesis verdict", "A nonpositive context effect makes recovery undefined",
                            "Text and initial/zeroed/swapped controls retained; latent bytes can exceed raw-text bytes",
                            "No R20 instrument repair, calibration or meta-test"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, read_manifest(args.instrument))
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
