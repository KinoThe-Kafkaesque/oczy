"""Audit the complete decoder ablation and candidate batch qualification."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infrastructure.kaggle.runtime_manifest import validate_runtime_manifest  # noqa: E402
from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from scripts.context_preservation_contract import write_json  # noqa: E402
from scripts.probe_decoder_parity import verify  # noqa: E402
from scripts.summarize_capability_validation import grouped_counts  # noqa: E402


def summarize(root, run):
    manifest = verify(root)
    cases = json.loads((root / "cases.json").read_text())
    result = json.loads((run / "results.json").read_text())
    rows = result["rows"]
    assert len(rows) == len(cases) == 32
    assert rows == [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    scorer, scores, parity = FrozenScorer(), [], []
    for row, case in zip(rows, cases, strict=True):
        assert all(row[k] == v for k, v in case.items())
        expected = set(manifest["conditions"]) | ({"native_neutral"} if row["bank"] == "none" else set())
        assert set(row["outputs"]) == set(row["correct"]) == expected
        for condition, value in row["outputs"].items():
            assert row["correct"][condition] == scorer.score_response(case["target"], value)
            scores.append({"bank": row["bank"], "condition": condition, "correct": row["correct"][condition]})
            parity.append({"bank": row["bank"], "condition": condition, "correct": value == row["outputs"]["scalar"]})
    batches = result["batches"]
    assert Counter(r["index"] for r in batches) == Counter(range(32))
    batch_parity = [{"bank": rows[r["index"]]["bank"], "condition": "dynamic_neutral_batch4",
                     "correct": r["generated"] == rows[r["index"]]["outputs"]["scalar"]} for r in batches]
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["organ_hash_before"] == result["organ_hash_after"] == manifest["organ_hash"]
    assert result["scorer_sha256"] == manifest["scorer_sha256"] == scorer.sha256
    assert not result["meta_test_accessed"] and result["optimizer_steps"] == 0
    runtime = validate_runtime_manifest(json.loads((run / "runtime_manifest.json").read_text()))
    assert runtime["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    assert result["default_generation_config"]["repetition_penalty"] == 1.1
    assert runtime["greedy_generation"]["repetition_penalty"] == 1.0
    execution = json.loads((run / "execution.json").read_text())
    assert execution["exit_code"] == 0 and execution["network_disabled"]
    qualified = all(r["correct"] for r in parity if r["condition"] == "dynamic_neutral") and all(r["correct"] for r in batch_parity)
    return {"instrument_id": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"], "candidate_qualified": qualified,
            "parity": grouped_counts(parity + batch_parity, ["bank", "condition"]),
            "task_scores": grouped_counts(scores, ["bank", "condition"]), "rows": rows, "batches": batches,
            "verification": {"all_input_source_state_and_model_hashes_verified": True, "scores_independently_recomputed": len(scores),
                             "parity_verdicts_recomputed": len(parity) + len(batches), "exact_coverage": True, "optimizer_steps": 0,
                             "meta_test_accessed": False, "seconds": execution["seconds"]}, "execution": execution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.run)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "batches", "execution")}, indent=2))


if __name__ == "__main__":
    main()
