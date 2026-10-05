"""Independently audit the frozen DEV-v2 run, including a blocked correction arm."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infrastructure.kaggle.runtime_manifest import validate_runtime_manifest  # noqa: E402
from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from scripts.context_preservation_contract import (  # noqa: E402
    capacity_gate,
    manifest_at,
    messages,
    sha,
    write_json,
)
from scripts.summarize_capability_validation import grouped_counts, retention  # noqa: E402


def key(row):
    return (row["split"], row["stage"], row["condition"], row["seed"], tuple(row["clients"]), row["input"])


def validate_rows(rows, expected):
    """Reject missing, duplicated, relabelled or incorrectly scored trials."""
    assert Counter(map(key, rows)) == Counter(map(key, expected)), "Trial coverage mismatch"
    lookup = {key(r): r for r in expected}
    scorer = FrozenScorer()
    for row in rows:
        gold = lookup[key(row)]
        assert row["target"] == gold["target"] and row["category"] == gold["category"], "Target changed"
        assert row["prompt_sha256"] == gold["prompt_sha256"], "Prompt changed"
        assert bool(row["correct"]) == bool(scorer.score_response(gold["target"], row["generated"])), "Score mismatch"


def expected_rows(manifest, data, correction):
    oracle = {r["stage"]: r for r in data["oracle"]["stages"]}
    teaching = data["capacity_training"]["rows"]
    text = "Corrected examples:\n" + "\n".join(f"Client {r['clients'][0]}, input {r['input']} -> correct output {r['target']}" for r in teaching)
    groups = [("calibration", 2, data["probes"]["calibration"])]
    groups += [("confirmation", s["stage"], s["rows"]) for s in data["probes"]["confirmation"]]
    expected = []
    for split, stage, probes in groups:
        for condition in manifest["reference_conditions"]:
            if stage == 3 and condition["name"] not in ("concise_table", "resolved_oracle"):
                continue
            for row in probes:
                form = condition["oracle"]
                context = (None if form is None else text if form == "examples" else
                           oracle[stage][form][row["clients"][0]] if form == "resolved" else oracle[stage][form])
                prompt = messages(row, condition["interface"], context)
                expected.append({**row, "split": split, "stage": stage, "condition": condition["name"], "seed": None,
                                 "prompt_sha256": sha(json.dumps(prompt, sort_keys=True).encode())})
    for stage in (2, 3) if correction else (2,):
        probes = next(s["rows"] for s in data["probes"]["confirmation"] if s["stage"] == stage)
        for seed in manifest["training"]["seeds"]:
            conditions = list(manifest["training"]["correction_arms"]) if stage == 3 else ["initial", "joint_restored"]
            if stage == 2 and seed == manifest["training"]["seeds"][0]:
                conditions.append("zeroed")
            for condition in conditions:
                expected += [{**r, "split": "confirmation", "stage": stage, "condition": condition, "seed": seed,
                              "prompt_sha256": sha(json.dumps(messages(r), sort_keys=True).encode())} for r in probes]
    return expected


def check_numeric(run, phase, released, entry, config):
    original = run / phase / "states" / entry["path"]
    copied = run / released / entry["path"]
    assert sha(original.read_bytes()) == sha(copied.read_bytes()) == entry["sha256"]
    array = np.load(copied, allow_pickle=False)
    assert list(array.shape) == entry["shape"] == [1, config["bank_width"], config["feature_dim"]]
    assert str(array.dtype) == entry["dtype"] == "float32" and np.isfinite(array).all()
    assert array.nbytes == entry["payload_bytes"] and copied.stat().st_size == entry["file_bytes"]


def summarize(root, run):
    manifest = manifest_at(root)
    repo = Path(__file__).resolve().parents[1]
    assert sha((repo / "scripts/prepare_context_preservation.py").read_bytes()) == manifest["preparer_sha256"]
    data = {}
    for role, entry in manifest["files"].items():
        raw = (root / entry["path"]).read_bytes()
        assert sha(raw) == entry["sha256"]
        data[role] = json.loads(raw)
    admission = json.loads((run / "admission.json").read_text())
    assert admission["manifest_sha256"] == manifest["manifest_sha256"]
    admitted = admission["admitted"]
    names = ["reference", "train_capacity", "restore_capacity"]
    if admitted:
        names += ["train_correction", "restore_correction"]
    else:
        blocked = json.loads((run / "blocked.json").read_text())
        assert blocked["correction_optimizer_steps"] == 0 and blocked["manifest_sha256"] == manifest["manifest_sha256"]
        assert not (run / "train_correction").exists() and not (run / "restore_correction").exists()
    phases = {name: json.loads((run / name / "results.json").read_text()) for name in names}
    executions = json.loads((run / "executions.json").read_text())
    assert [e["phase"] for e in executions] == names
    assert len({p["pid"] for p in phases.values()}) == len(names)
    for execution in executions:
        assert execution["exit_code"] == 0 and execution["network_disabled"]
        assert not execution["filesystem_firewall"]["visible_forbidden_files"]
        assert execution["runtime_manifest_sha256"] == manifest["runtime_manifest_sha256"]
        assert execution["manifest_sha256"] == manifest["manifest_sha256"]
    runtime = validate_runtime_manifest(json.loads((run / "runtime_manifest.json").read_text()))
    assert runtime["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    for phase in phases.values():
        assert phase["manifest_sha256"] == manifest["manifest_sha256"]
        assert phase["organ_hash_before"] == phase["organ_hash_after"] == manifest["organ_hash"]
        assert phase["scorer_sha256"] == manifest["scorer_sha256"] and not phase["meta_test_accessed"]
        if "rows" in phase:
            assert [json.loads(line) for line in (run / phase["phase"] / "rows.jsonl").read_text().splitlines()] == phase["rows"]
    rows = [r for p in phases.values() for r in p.get("rows", [])]
    validate_rows(rows, expected_rows(manifest, data, admitted))
    assert len(phases["reference"]["rows"]) == 256 and len(phases["restore_capacity"]["rows"]) == 112
    config = manifest["training"]
    trajectories, fits = [], []
    scorer = FrozenScorer()
    for phase, role, released in [("train_capacity", "capacity_training", "released_capacity")] + (
            [("train_correction", "correction_training", "released_correction")] if admitted else []):
        result = phases[phase]
        is_capacity = phase == "train_capacity"
        conditions = ["joint"] if is_capacity else list(config["correction_arms"])
        steps = config["capacity_steps" if is_capacity else "correction_steps"]
        trajectory = [json.loads(line) for line in (run / phase / "trajectory.jsonl").read_text().splitlines()]
        assert Counter((r["seed"], r["condition"], r["step"]) for r in trajectory) == Counter(
            (s, c, step) for s in config["seeds"] for c in conditions for step in range(1, steps + 1))
        assert len(trajectory) == result["optimizer_steps"]
        assert all(r["gradient_norm_before_clip"] > 0 and all(math.isfinite(v) for v in r.values() if isinstance(v, (int, float))) for r in trajectory)
        assert Counter((f["seed"], f["condition"]) for f in result["fits"]) == Counter((s, c) for s in config["seeds"] for c in conditions)
        examples = data[role]["rows"]
        for fit in result["fits"]:
            observed = fit["training_fit"]
            assert Counter((tuple(r["clients"]), r["input"], r["target"]) for r in observed) == Counter(
                (tuple(r["clients"]), r["input"], r["target"]) for r in examples)
            assert all(bool(r["correct"]) == bool(scorer.score_response(r["target"], r["generated"])) for r in observed)
        state_manifest = json.loads((run / released / "state_manifest.json").read_text())
        assert state_manifest == {"manifest_sha256": manifest["manifest_sha256"], "states": result["states"]}
        assert Counter((s["seed"], "joint" if is_capacity else s["condition"]) for s in result["states"]) == Counter(
            (s, c) for s in config["seeds"] for c in conditions)
        for state in result["states"]:
            for label in ("initial", "trained") if is_capacity else ("trained",):
                check_numeric(run, phase, released, state[label], config)
            if not is_capacity:
                parent = next(s for s in phases["train_capacity"]["states"] if s["seed"] == state["seed"])
                assert state["parent_sha256"] == parent["trained"]["sha256"]
        trajectories += trajectory
        fits += result["fits"]
    fit_pass = all(r["correct"] for f in phases["train_capacity"]["fits"] for r in f["training_fit"])
    probe_pass = capacity_gate(phases["restore_capacity"]["rows"], config["seeds"])
    assert admitted == (fit_pass and probe_pass)
    preservation = []
    if admitted:
        for seed in config["seeds"]:
            for arm in config["correction_arms"]:
                for category in ("cobalt", "silver", "quartz"):
                    before = [r for r in rows if r["condition"] == "joint_restored" and r["seed"] == seed and r["category"] == category]
                    after = [r for r in rows if r["condition"] == arm and r["seed"] == seed and r["category"] == category]
                    preservation.append({"seed": seed, "condition": arm, "category": category, **retention(before, after)})
    prior = json.loads((repo / "experiments_logs/2026-09-13_capability_validation_v1.json").read_text())["rows"]
    old_rows = {(r["category"], r["input"]): r for r in prior if r["condition"] == "complete_oracle" and r["stage"] == 2 and r["category"] != "composition"}
    paired = [r for r in rows if r["condition"] == "v1_full" and r["split"] == "calibration" and r["category"] != "quartz"]
    assert len(paired) == 12
    reproduced = sum(r["generated"] == old_rows[(r["category"], r["input"])]["generated"] for r in paired)
    return {"instrument_id": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"],
            "counts": grouped_counts(rows, ["split", "stage", "condition", "seed", "category"]),
            "totals": grouped_counts(rows, ["split", "stage", "condition", "seed"]),
            "capacity_admission": {"admitted": admitted, "all_teaching_fits": bool(fit_pass), "all_fresh_probes": bool(probe_pass)},
            "preservation": preservation, "training_fits": fits, "rows": rows, "trajectories": trajectories,
            "verification": {"source_input_numeric_hashes_verified": True, "scores_independently_recomputed": len(rows),
                             "teaching_scores_recomputed": sum(len(f["training_fit"]) for f in fits),
                             "coverage_and_prompt_hashes_verified": True, "frozen_model_scorer_verified": True,
                             "optimizer_steps": len(trajectories), "correction_optimizer_steps": phases.get("train_correction", {}).get("optimizer_steps", 0),
                             "old_v1_calibration_outputs_reproduced": {"correct": reproduced, "total": len(paired)},
                             "fresh_process_pids": [p["pid"] for p in phases.values()], "meta_test_accessed": False,
                             "seconds": sum(e["seconds"] for e in executions)}, "executions": executions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.run)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("totals", "capacity_admission", "verification")}, indent=2))


if __name__ == "__main__":
    main()
