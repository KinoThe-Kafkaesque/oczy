"""Recover bookkeeping after a completed evaluation; never rerun or rescore an optimizer."""

import json
from collections import Counter
from pathlib import Path

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from scripts.diversity_dev.contract import admission, key, manifest_at, sha, write_json


def recover(root, run):
    manifest = manifest_at(root)
    def read(path):
        return json.loads(path.read_text())
    trained, evaluated = read(run / "train/results.json"), read(run / "evaluate/results.json")
    ledger = read(run / "executions.json")
    assert [entry["phase"] for entry in ledger] == ["train"] and ledger[0]["exit_code"] == 0
    assert not (run / "admission.json").exists() and not (run / "train_correction").exists()
    scorer = FrozenScorer()
    for result in (trained, evaluated):
        assert result["manifest_sha256"] == manifest["manifest_sha256"]
        assert result["organ_hash_before"] == result["organ_hash_after"] == manifest["organ_hash"]
        assert result["scorer_sha256"] == scorer.sha256
    probes = read(root / "probes/data.json")
    cohorts = {"calibration": probes["calibration"], "confirmation": probes["confirmation"][0]["rows"]}
    conditions = [(name, seed) for seed in range(3) for name in ("initial", "control", "diversity")] + [("zeroed", 0)]
    conditions += [(name, None) for name in manifest["references"]]
    expected = Counter((split, name, seed, key(row)) for split, cases in cohorts.items() for name, seed in conditions for row in cases)
    actual = Counter((row["split"], row["condition"], row["seed"], key(row)) for row in evaluated["rows"])
    assert actual == expected and len(evaluated["rows"]) == 608
    for row in evaluated["rows"]:
        case = next(case for case in cohorts[row["split"]] if key(case) == key(row))
        assert row["target"] == case["target"]
        assert row["correct"] == scorer.score_response(case["target"], row["generated"])
    passed = admission(evaluated["rows"], trained["fits"], manifest, cohorts["confirmation"])
    assert not passed, "This recovery only closes the already-failed acquisition prerequisite"
    recovery = {"parent_manifest_sha256": manifest["manifest_sha256"], "recovery_source_sha256": sha(Path(__file__).read_bytes()),
                "original_controller_exit_code": 1, "original_error": "FileExistsError: executions.json; write_json uses exclusive creation",
                "evaluation_child_exit_code": None, "evaluation_seconds": None,
                "provenance_limit": "Complete evaluation results were durably written; child exit status, duration and its pre-run firewall-check output were lost when the controller failed to append the ledger.",
                "recovery_scope": "Validate complete outputs and compute the original frozen admission function. Original source, ledger, scores and states are preserved.",
                "optimizer_steps": 0, "new_model_calls": 0, "evaluated_rows_verified": 608,
                "artifacts": {name: sha((run / name).read_bytes()) for name in ("executions.json", "train/results.json", "evaluate/results.json", "evaluate.log", "runtime_manifest.json")}}
    write_json(run / "controller_recovery.json", recovery)
    write_json(run / "admission.json", {"manifest_sha256": manifest["manifest_sha256"], "admitted": passed})
    write_json(run / "blocked.json", {"phase": "train_correction", "reason": "Acquisition prerequisite failed", "correction_optimizer_steps": 0,
                                      "manifest_sha256": manifest["manifest_sha256"]})
    print(json.dumps(recovery, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root, args.run)
