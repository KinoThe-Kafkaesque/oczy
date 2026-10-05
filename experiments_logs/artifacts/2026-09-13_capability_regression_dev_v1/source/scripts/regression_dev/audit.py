"""Audit replay equality and component-level behavioral regressions."""

import argparse
from collections import Counter
from pathlib import Path

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from scripts.diversity_dev.audit import tally
from scripts.regression_dev.contract import REPLAYS, REPO, compare_rows, read, verify, write_json


def audit(root, run):
    manifest, scorer = verify(root), FrozenScorer()
    result = read(run / "replay/results.json")
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["optimizer_steps"] == 0 and not result["meta_test_accessed"]
    assert result["scorer_sha256"] == scorer.sha256
    assert result["organ_hash_before"] == result["organ_hash_after"] == read(REPO / "experiments/scoped-diversity-dev-v3/MANIFEST.json")["organ_hash"]
    all_rows = {name: read(run / "replay" / (name + ".json"))["rows"] for name in manifest["counts"]}
    for name, expected_count in manifest["counts"].items():
        assert len(all_rows[name]) == expected_count == result["counts"][name]
    for name, (path, _) in REPLAYS.items():
        assert result["verdicts"][name]["exact_replay"] == compare_rows(all_rows[name], read(REPO / path)["rows"])
    for name in ("direct_operation", "decoder_single", "decoder_batch4"):
        for row in all_rows[name]:
            assert row["exact_replay"] == (row["generated"] == row["previous_generated"])
        assert result["verdicts"][name]["exact_replay"] == all(r["exact_replay"] for r in all_rows[name])
    assert result["historical_replay_exact"] == all(v["exact_replay"] for v in result["verdicts"].values())
    for name, rows in all_rows.items():
        if "actions" not in name:
            for row in rows:
                assert row["correct"] == scorer.score_response(row["target"], row["generated"])
    cases = read(REPO / "experiments/capability-validation-v1/actions/data.json")["cases"]
    for name in ("capability_actions", "candidate_actions"):
        for row in all_rows[name]:
            case = next(c for c in cases if c["name"] == row["case"])
            fields = {"tool_sequence_exact": row["calls"] == case["expected_calls"], "filesystem_exact": row["actual_files"] == case["expected_files"],
                      "final_exact": row["final"] == case["expected_final"], "execution_error_free": row["error"] is None}
            assert all(row[k] == v for k, v in fields.items()) and row["correct"] == all(fields.values())
    candidates = all_rows["candidate_probes"]
    probes = read(REPO / "experiments/capability-validation-v1/probes/data.json")["stages"][1]["probes"]
    conditions = [(c, s) for s in range(3) for c in ("initial", "control", "diversity")] + [("zeroed", 0)]
    def key(r):
        return tuple(r["clients"]), r["input"]
    assert Counter((r["condition"], r["seed"], key(r)) for r in candidates) == Counter((c, s, key(r)) for c, s in conditions for r in probes)
    for row in candidates:
        case = next(p for p in probes if key(p) == key(row))
        assert all(row[k] == v for k, v in case.items())
    action_conditions = conditions + [("text_control", None), ("text_diversity", None)]
    assert Counter((r["condition"], r["seed"], r["case"]) for r in all_rows["candidate_actions"]) == Counter((c, s, r["name"]) for c, s in action_conditions for r in cases)
    paired = []
    for seed in range(3):
        for category in sorted({r["category"] for r in candidates}):
            old = {key(r): r for r in candidates if (r["condition"], r["seed"], r["category"]) == ("control", seed, category)}
            new = {key(r): r for r in candidates if (r["condition"], r["seed"], r["category"]) == ("diversity", seed, category)}
            assert old.keys() == new.keys()
            paired.append({"seed": seed, "category": category, "control_correct": sum(r["correct"] for r in old.values()),
                           "diversity_correct": sum(r["correct"] for r in new.values()), "total": len(old),
                           "lost": [list(k) for k in old if old[k]["correct"] and not new[k]["correct"]],
                           "gained": [list(k) for k in old if not old[k]["correct"] and new[k]["correct"]]})
    actions = []
    for condition, seed in action_conditions:
        selected = [r for r in all_rows["candidate_actions"] if (r["condition"], r["seed"]) == (condition, seed)]
        actions.append({"condition": condition, "seed": seed, "total": len(selected), "strict_correct": sum(r["correct"] for r in selected),
                        "tool_core_correct": sum(all(r[k] for k in ("tool_sequence_exact", "filesystem_exact", "execution_error_free")) for r in selected)})
    execution = read(run / "execution.json")
    assert execution["exit_code"] == 0 and execution["network_disabled"]
    return {**result, "instrument_id": manifest["instrument_id"], "verification": {"passed": True, "total_rows": sum(map(len, all_rows.values()))},
            "paired": paired, "candidate_counts": tally(candidates, ("condition", "seed", "category")), "candidate_action_counts": actions,
            "execution": execution, "candidate_summary": "Candidate checks use the unchanged earlier prompts. Joint learning does not establish sequential accumulation or correction; composition requires correct individual rules.",
            "limitations": ["Read-only behavioral replay; historical optimizers were not rerun", "Known failed final-answer action prompt preserved", "No meta-test", "Exact replay of failure is not capability success"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.root, args.run))
