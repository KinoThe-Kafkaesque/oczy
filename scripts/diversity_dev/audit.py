"""Independently recompute trial coverage, scores, provenance and paired changes."""

import argparse
import json
import math
import zlib
from collections import Counter
from pathlib import Path

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from scripts.diversity_dev.contract import (
    admission,
    canonical,
    key,
    manifest_at,
    messages,
    reference_prompt,
    sha,
    write_json,
)


def read(path):
    return json.loads(path.read_text())


def tally(rows, fields):
    groups = {}
    for row in rows:
        group = tuple(row.get(field) for field in fields)
        record = groups.setdefault(group, {**dict(zip(fields, group, strict=True)), "correct": 0, "total": 0})
        record["correct"] += int(row["correct"])
        record["total"] += 1
    return list(groups.values())


def audit(root, run):
    manifest, scorer = manifest_at(root), FrozenScorer()
    for entry in manifest["files"].values():
        assert sha((root / entry["path"]).read_bytes()) == entry["sha256"]
    for control in manifest["controls"]:
        for kind in ("initial", "trained"):
            entry = control[kind]
            assert sha((root / "controls" / entry["path"]).read_bytes()) == entry["sha256"]
    runtime = read(run / "runtime_manifest.json")
    assert runtime["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    assert sha(canonical({k: v for k, v in runtime.items() if k != "manifest_sha256"})) == runtime["manifest_sha256"]
    teaching = read(root / "capacity_training/data.json")
    probes = read(root / "probes/data.json")
    oracle = read(root / "oracle/data.json")
    training, evaluated = read(run / "train/results.json"), read(run / "evaluate/results.json")
    states = read(run / "released_diversity/state_manifest.json")
    assert states["states"] == training["states"]
    assert states["manifest_sha256"] == manifest["manifest_sha256"]
    for phase in (training, evaluated):
        assert phase["manifest_sha256"] == manifest["manifest_sha256"]
        assert phase["organ_hash_before"] == phase["organ_hash_after"] == manifest["organ_hash"]
        assert phase["scorer_sha256"] == scorer.sha256 == manifest["scorer_sha256"]
        assert phase["meta_test_accessed"] is False
    assert training["optimizer_steps"] == 144 and training["heldout_examples_read"] == 0 and evaluated["optimizer_steps"] == 0
    assert [s["seed"] for s in states["states"]] == [0, 1, 2]
    for state in states["states"]:
        for role in ("initial", "trained"):
            entry = state[role]
            assert sha((run / "released_diversity" / entry["path"]).read_bytes()) == entry["sha256"]
            assert entry["file_bytes"] == 28800 and entry["shape"] == [1, 8, 896]
        assert state["initial"]["sha256"] == manifest["controls"][state["seed"]]["initial"]["sha256"]
    trajectories = [json.loads(line) for line in (run / "train/trajectory.jsonl").read_text().splitlines()]
    assert Counter((r["seed"], r["step"]) for r in trajectories) == Counter((s, t) for s in range(3) for t in range(1, 49))
    for row in trajectories:
        assert row["group"] == (row["step"] - 1) % 3
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ("gradient_norm_before_clip", "bank_norm_after_update", "seconds"))
        assert math.isfinite(row["mean_ce_before_update"])
    fits = training["fits"]
    assert Counter(f["seed"] for f in fits) == Counter(range(3))
    for fit in fits:
        assert Counter(key(r) for r in fit["training_fit"]) == Counter(key(r) for r in teaching["rows"])
        for row in fit["training_fit"]:
            target = next(r["target"] for r in teaching["rows"] if key(r) == key(row))
            assert row["target"] == target and row["correct"] == scorer.score_response(target, row["generated"])
        history = [r for r in trajectories if r["seed"] == fit["seed"]]
        assert fit["first_ce"] == history[0]["mean_ce_before_update"] and fit["last_ce"] == history[-1]["mean_ce_before_update"]
        assert math.isclose(fit["first_ce"], manifest["parent_first_ce"][str(fit["seed"])], rel_tol=1e-6, abs_tol=1e-6)
    cohorts = {"calibration": probes["calibration"], "confirmation": probes["confirmation"][0]["rows"]}
    conditions = [(name, s) for s in range(3) for name in ("initial", "control", "diversity")] + [("zeroed", 0)]
    conditions += [(name, None) for name in manifest["references"]]
    expected = Counter((split, condition, seed, *key(row)) for split, cases in cohorts.items() for condition, seed in conditions for row in cases)
    rows = evaluated["rows"]
    assert Counter((r["split"], r["condition"], r["seed"], *key(r)) for r in rows) == expected
    for row in rows:
        case = next(r for r in cohorts[row["split"]] if key(r) == key(row))
        assert all(row[k] == case[k] for k in case)
        assert row["correct"] == scorer.score_response(case["target"], row["generated"])
        prompt = reference_prompt(case, row["condition"], teaching, oracle["stages"][0]["table"]) if row["seed"] is None else messages(case)
        assert row["prompt_sha256"] == sha(json.dumps(prompt, sort_keys=True).encode())
    for arm in ("control", "diversity"):
        storage = next(r for r in evaluated["storage"] if r["arm"] == arm)
        text, compressed = (run / "evaluate" / f"{arm}.txt").read_bytes(), (run / "evaluate" / f"{arm}.zlib").read_bytes()
        assert zlib.decompress(compressed) == text and storage["roundtrip_exact"]
        assert storage["text_sha256"] == sha(text) and storage["zlib_sha256"] == sha(compressed)
        assert storage["example_utf8_bytes"] == len(text) and storage["zlib_bytes"] == len(compressed)
        a = {(r["split"], *key(r)): r["generated"] for r in rows if r["condition"] == "text_" + arm}
        b = {(r["split"], *key(r)): r["generated"] for r in rows if r["condition"] == "zlib_" + arm}
        assert a == b
    parent_path = Path(__file__).resolve().parents[2] / "experiments_logs/2026-09-13_context_preservation_dev_v2.json"
    parent = read(parent_path)
    assert parent["manifest_sha256"] == manifest["parent_manifest_sha256"]
    aliases = {"joint_restored": "control", "initial": "initial", "zeroed": "zeroed"}
    previous = {(aliases[r["condition"]], r["seed"], *key(r)): r["generated"] for r in parent["rows"]
                if r["condition"] in aliases and r["split"] == "confirmation" and r["stage"] == 2}
    current = {(r["condition"], r["seed"], *key(r)): r["generated"] for r in rows
               if r["condition"] in aliases.values() and r["split"] == "calibration"}
    assert len(previous) == len(current) == 112
    parent_parity = {"rows": 112, "exact_outputs": sum(current[k] == value for k, value in previous.items()),
                     "all_outputs_equal": current == previous, "reference_sha256": sha(parent_path.read_bytes())}
    passed = admission(rows, fits, manifest, cohorts["confirmation"])
    assert read(run / "admission.json") == {"manifest_sha256": manifest["manifest_sha256"], "admitted": passed}
    executions = read(run / "executions.json")
    assert all(r["exit_code"] == 0 and r["network_disabled"] and not r["filesystem_firewall"]["visible"] for r in executions)
    recovery = read(run / "controller_recovery.json") if (run / "controller_recovery.json").exists() else None
    if recovery:
        assert [r["phase"] for r in executions] == ["train"]
        assert recovery["parent_manifest_sha256"] == manifest["manifest_sha256"]
        assert recovery["original_controller_exit_code"] == 1
        assert recovery["evaluation_child_exit_code"] is None and recovery["evaluation_seconds"] is None
        assert recovery["optimizer_steps"] == recovery["new_model_calls"] == 0
        assert recovery["evaluated_rows_verified"] == len(rows)
        for name, expected_hash in recovery["artifacts"].items():
            assert sha((run / name).read_bytes()) == expected_hash
    else:
        assert {r["phase"] for r in executions} >= {"train", "evaluate"}
    if not passed:
        assert read(run / "blocked.json")["correction_optimizer_steps"] == 0
        assert not (run / "train_correction").exists()
    paired = []
    for split in cohorts:
        for seed in range(3):
            for category in ("amber", "cobalt", "silver", "quartz"):
                a = {key(r): r for r in rows if (r["split"], r["seed"], r["category"], r["condition"]) == (split, seed, category, "control")}
                b = {key(r): r for r in rows if (r["split"], r["seed"], r["category"], r["condition"]) == (split, seed, category, "diversity")}
                assert a.keys() == b.keys() and len(a) == 4
                paired.append({"split": split, "seed": seed, "category": category, "control_correct": sum(r["correct"] for r in a.values()),
                               "diversity_correct": sum(r["correct"] for r in b.values()), "total": 4,
                               "lost": [k[1] for k in a if a[k]["correct"] and not b[k]["correct"]],
                               "gained": [k[1] for k in a if not a[k]["correct"] and b[k]["correct"]]})
    return {"instrument_id": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"],
            "verification": {"passed": True, "probe_rows": len(rows), "teaching_fits": sum(len(f["training_fit"]) for f in fits), "optimizer_updates": len(trajectories)},
            "admission": passed, "preservation_status": "run separately; inspect correction outputs" if passed else "blocked: acquisition prerequisite failed; zero correction updates",
            "counts": tally(rows, ("split", "condition", "seed", "category")), "totals": tally(rows, ("split", "condition", "seed")),
            "paired": paired, "parent_control_reproduction": parent_parity, "training_fits": fits, "storage": evaluated["storage"], "rows": rows,
            "trajectories": trajectories, "executions": executions, "execution_recovery": recovery,
            "limitations": ["Local DEV only", "Equal updates and example losses; token counts differ", "Joint learning is not sequential accumulation", "Oracle rule selection is externally supplied", "No learned compression or meta-test claim"] + ([recovery["provenance_limit"]] if recovery else [])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.root, args.run))
