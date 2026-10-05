"""Audit and summarize the predeclared DEV battery without modifying scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from scripts.capability_validation_contract import manifest_at, sha, write_json  # noqa: E402


def count(rows):
    return {"correct": sum(bool(r["correct"]) for r in rows), "total": len(rows)}


def grouped_counts(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key, strict=True)), **count(items)} for key, items in groups.items()]


def index_rows(rows):
    return {(tuple(r["clients"]), r["input"]): r for r in rows}


def retention(before, after):
    old, new = index_rows(before), index_rows(after)
    if old.keys() != new.keys():
        raise ValueError("Retention probes do not pair")
    acquired = [key for key, row in old.items() if row["correct"]]
    if any(old[key]["target"] != new[key]["target"] for key in old):
        raise ValueError("Unrelated-rule retention changed target")
    return {"previously_correct": len(acquired), "retained": sum(bool(new[key]["correct"]) for key in acquired),
            "lost_inputs": [old[key]["input"] for key in acquired if not new[key]["correct"]],
            "all_old": count(before), "all_after": count(after)}


def summarize(root, output):
    manifest = manifest_at(root)
    phases = {name: json.loads((output / name / "results.json").read_text()) for name in ("reference", "train", "restore", "actions")}
    executions = json.loads((output / "executions.json").read_text())
    assert len(executions) == 4 and all(e["exit_code"] == 0 for e in executions)
    assert len({p["pid"] for p in phases.values()}) == 4
    for phase in phases.values():
        assert phase["manifest_sha256"] == manifest["manifest_sha256"]
        assert phase["organ_hash_before"] == phase["organ_hash_after"] == manifest["organ_hash"]
        assert phase["scorer_sha256"] == manifest["scorer_sha256"] and not phase["meta_test_accessed"]
    for entry in manifest["files"].values():
        assert sha((root / entry["path"]).read_bytes()) == entry["sha256"]
    scorer = FrozenScorer()
    rows = phases["reference"]["rows"] + phases["restore"]["rows"]
    probes = json.loads((root / "probes/data.json").read_text())["stages"]
    gold = {(stage["stage"], tuple(r["clients"]), r["input"]): r for stage in probes for r in stage["probes"]}
    expected_coverage = Counter()
    conditions = [(c, None) for c in ("no_context", "raw_history", "latest_example_retrieval", "complete_oracle", "zlib_retrieval")]
    conditions += [(c, seed) for c in ("initial", "learned_restored") for seed in manifest["training"]["seeds"]]
    conditions += [("zeroed", 0)]
    for condition, seed in conditions:
        expected_coverage.update((condition, seed, *key) for key in gold)
    assert Counter((r["condition"], r["seed"], r["stage"], tuple(r["clients"]), r["input"]) for r in rows) == expected_coverage
    for row in rows:
        target = gold[(row["stage"], tuple(row["clients"]), row["input"])]
        assert row["target"] == target["target"] and row["category"] == target["category"]
        assert bool(row["correct"]) == bool(scorer.score_response(row["target"], row["generated"]))
    assert len(phases["reference"]["rows"]) == 240 and len(phases["restore"]["rows"]) == 336
    for state in phases["train"]["states"]:
        for entry in [state["initial"], *(s["state"] for s in state["stages"])]:
            assert sha((output / "released_states" / entry["path"]).read_bytes()) == entry["sha256"]
            assert entry["shape"] == [1, 8, 896] and entry["payload_bytes"] == 28672
    trajectories = [json.loads(line) for line in (output / "train/trajectory.jsonl").read_text().splitlines()]
    assert len(trajectories) == phases["train"]["optimizer_steps"] == 216
    assert Counter((r["seed"], r["stage"], r["step"]) for r in trajectories) == Counter(
        (seed, stage, step) for seed in manifest["training"]["seeds"] for stage in (1, 2, 3) for step in range(1, 25))
    assert all(r["gradient_norm_before_clip"] > 0 and all(math.isfinite(v) for v in r.values() if isinstance(v, (int, float))) for r in trajectories)
    acquired, corrected, composed = [], [], []
    learned = [r for r in rows if r["condition"] == "learned_restored"]
    for seed in manifest["training"]["seeds"]:
        def select(stage, category, seed=seed):
            return [r for r in learned if r["seed"] == seed and r["stage"] == stage and r["category"] == category]
        acquired.append({"seed": seed, "new_cobalt": count(select(2, "cobalt")), "amber_retention": retention(select(1, "amber"), select(2, "amber"))})
        corrected.append({"seed": seed, "replacement_amber": count(select(3, "amber")), "cobalt_retention": retention(select(2, "cobalt"), select(3, "cobalt"))})
        for stage in (2, 3):
            components = index_rows(select(stage, "amber") + select(stage, "cobalt"))
            probes = select(stage, "composition")
            supported = [r for r in probes if all(components[((client,), r["input"])]["correct"] for client in r["clients"])]
            composed.append({"seed": seed, "stage": stage, **count(probes), "both_single_rule_prerequisites_correct": len(supported),
                             "correct_where_both_prerequisites_correct": count(supported)["correct"]})
    compression = []
    for entry in phases["reference"]["storage"]:
        stage = entry["stage"]
        raw = index_rows([r for r in rows if r["condition"] == "latest_example_retrieval" and r["stage"] == stage])
        zipped = index_rows([r for r in rows if r["condition"] == "zlib_retrieval" and r["stage"] == stage])
        paired_exact = raw.keys() == zipped.keys() and all(raw[k]["generated"] == zipped[k]["generated"] for k in raw)
        numeric = phases["train"]["states"][0]["stages"][stage - 1]["state"]
        compression.append({**entry, "neural_payload_bytes": numeric["payload_bytes"], "neural_file_bytes": numeric["file_bytes"],
                            "neural_storage_below_examples": numeric["file_bytes"] < entry["active_example_bytes"],
                            "zlib_generations_match_raw_exactly": paired_exact,
                            "zlib_storage_below_examples": entry["zlib_file_bytes"] < entry["active_example_bytes"], "retrieval_accuracy": count(list(raw.values()))})
    action_cases = {c["name"]: c for c in json.loads((root / "actions/data.json").read_text())["cases"]}
    action_rows = phases["actions"]["rows"]
    assert len(action_rows) == 20
    assert Counter((r["condition"], r["seed"], r["case"]) for r in action_rows) == Counter(
        (condition, seed, name) for condition, seed in [("no_context", None), ("latest_example_retrieval", None),
                                                      *(("learned_restored", s) for s in manifest["training"]["seeds"])] for name in action_cases)
    for row in action_rows:
        expected = action_cases[row["case"]]
        fields = {"tool_sequence_exact": row["calls"] == expected["expected_calls"],
                  "filesystem_exact": row["actual_files"] == expected["expected_files"],
                  "final_exact": row["final"] == expected["expected_final"], "execution_error_free": row["error"] is None}
        assert all(row[k] == v for k, v in fields.items()) and row["correct"] == all(fields.values())
    return {"instrument_id": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"],
            "counts": grouped_counts(rows, ["condition", "seed", "stage", "category"]),
            "accumulation": acquired, "correction": corrected, "composition": composed, "compression": compression,
            "actions": grouped_counts(action_rows, ["condition", "seed"]), "action_rows": action_rows,
            "training_fits": phases["train"]["fits"], "rows": rows, "trajectories": trajectories,
            "verification": {"source_input_and_state_hashes_verified": True, "model_and_scorer_unchanged": True,
                             "probe_action_and_update_coverage_exact": True,
                             "scores_independently_recomputed": len(rows), "action_verdicts_recomputed": len(action_rows),
                             "optimizer_steps": len(trajectories), "fresh_process_pids": [p["pid"] for p in phases.values()],
                             "no_forbidden_registered_files_visible": all(not e["filesystem_firewall"]["visible_forbidden_files"] for e in executions),
                             "seconds": sum(e["seconds"] for e in executions), "meta_test_accessed": False}, "executions": executions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.run)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "trajectories", "executions", "training_fits", "action_rows")}, indent=2))


if __name__ == "__main__":
    main()
