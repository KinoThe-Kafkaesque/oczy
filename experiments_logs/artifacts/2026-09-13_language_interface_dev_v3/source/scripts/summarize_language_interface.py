"""Audit all registered DEV-v3 prompts, scores, decoder checks and paired controls."""

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
from scripts.probe_language_interface import verify  # noqa: E402
from scripts.summarize_capability_validation import grouped_counts  # noqa: E402


def row_key(row):
    return row["split"], row["condition"], row["category"], row["input"]


def summarize(root, run, parent_run):
    manifest = verify(root)
    cases = json.loads((root / "cases.json").read_text())
    result = json.loads((run / "results.json").read_text())
    rows = result["rows"]
    assert len(rows) == 224 and Counter(map(row_key, rows)) == Counter(map(row_key, cases))
    assert rows == [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    gold = {row_key(r): r for r in cases}
    scorer = FrozenScorer()
    for row in rows:
        case = gold[row_key(row)]
        assert all(row[k] == v for k, v in case.items())
        assert row["correct"] == scorer.score_response(case["target"], row["generated"])
        if "decoder_parity" in row:
            parity = row["decoder_parity"]
            assert parity["all_text_equal"] == (parity["native"] == parity["batch"] == row["generated"])
    checked = [r for r in rows if "decoder_parity" in r]
    assert {row_key(r) for r in checked} == {row_key(r) for r in cases if r["condition"] == "resolved_v2" and r["split"] == "calibration"}
    assert len(checked) == 16
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["organ_hash_before"] == result["organ_hash_after"] == manifest["organ_hash"]
    assert result["scorer_sha256"] == manifest["scorer_sha256"] == scorer.sha256
    assert result["optimizer_steps"] == 0 and not result["meta_test_accessed"]
    runtime = validate_runtime_manifest(json.loads((run / "runtime_manifest.json").read_text()))
    assert runtime["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    execution = json.loads((run / "execution.json").read_text())
    assert execution["exit_code"] == 0 and execution["network_disabled"]
    old = json.loads((parent_run / "reference/results.json").read_text())
    assert old["manifest_sha256"] == manifest["parent_manifest_sha256"]
    baseline_map = {"resolved_v2": "resolved_oracle", "table_v2": "concise_table", "text_examples_v2": "crossed_example_retrieval"}
    previous = {(r["condition"], r["category"], r["input"]): r for r in old["rows"] if r["split"] == "confirmation" and r["stage"] == 2}
    paired = []
    for row in rows:
        if row["condition"] in baseline_map and row["split"] == "calibration":
            earlier = previous[(baseline_map[row["condition"]], row["category"], row["input"])]
            paired.append({"condition": row["condition"], "category": row["category"], "input": row["input"],
                           "correct": row["generated"] == earlier["generated"]})
    assert len(paired) == 48
    return {"instrument_id": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"],
            "counts": grouped_counts(rows, ["split", "condition", "category"]), "totals": grouped_counts(rows, ["split", "condition"]),
            "rows": rows, "decoder_parity": {"exact_text_matches": sum(r["decoder_parity"]["all_text_equal"] for r in checked), "total": len(checked)},
            "v2_baseline_reproduction": grouped_counts(paired, ["condition"]),
            "verification": {"all_input_source_prompt_and_model_hashes_verified": True, "scores_independently_recomputed": len(rows),
                             "exact_coverage": True, "optimizer_steps": 0, "meta_test_accessed": False, "seconds": execution["seconds"]},
            "execution": execution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "parent-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.run, args.parent_run)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "execution", "counts")}, indent=2))


if __name__ == "__main__":
    main()
