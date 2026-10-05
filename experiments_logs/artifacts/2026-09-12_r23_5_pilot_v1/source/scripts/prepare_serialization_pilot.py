"""Freeze two identifiable DEV formatting patterns before any model run."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.serialization_pilot_contract import (  # noqa: E402
    QUERY,
    REFINER,
    SYSTEM,
    canonical,
    sha,
    write_json,
)


def pattern_definitions():
    suffix_candidates = {"identity": lambda x: x, "reverse": lambda x: x[::-1], "upper": str.upper}
    for text in ("zor", "siv", "nurp"):
        suffix_candidates["append_" + text] = lambda x, text=text: x + text
        suffix_candidates["prepend_" + text] = lambda x, text=text: text + x
    date_candidates = {}
    for order in itertools.permutations(range(3)):
        for separator in ("/", "-"):
            date_candidates[str(order) + separator] = lambda x, order=order, separator=separator: separator.join(x.split("-")[i] for i in order)
    return [
        {"name": "suffix", "training": ["cat", "dog", "fox"], "probes": ["sun", "rain", "snow", "wind"],
         "candidates": suffix_candidates, "chosen": "append_zor"},
        {"name": "date_order", "training": ["2024-01-23", "2025-11-04", "2026-07-15"],
         "probes": ["2023-02-17", "2027-12-06", "2022-09-28", "2028-03-14"],
         "candidates": date_candidates, "chosen": "(2, 1, 0)/"},
    ]


def build_data():
    training, probes, audit = [], [], []
    for index, pattern in enumerate(pattern_definitions()):
        function = pattern["candidates"][pattern["chosen"]]
        examples = [{"input": x, "target": function(x)} for x in pattern["training"]]
        heldout = [{"input": x, "target": function(x)} for x in pattern["probes"]]
        compatible = [name for name, candidate in pattern["candidates"].items()
                      if all(candidate(r["input"]) == r["target"] for r in examples)]
        if compatible != [pattern["chosen"]] or set(pattern["training"]) & set(pattern["probes"]):
            raise ValueError("Teaching fails to identify the rule or inputs overlap")
        if len(examples) != 3 or len(heldout) != 4:
            raise ValueError("Pilot requires exactly three examples and four held-out probes")
        training.append({"index": index, "examples": examples})
        probes.append({"index": index, "probes": heldout})
        audit.append({"index": index, "name": pattern["name"], "candidate_count": len(pattern["candidates"]),
                      "compatible_candidates": compatible, "inputs_disjoint": True,
                      "limitation": "Identifiable within the declared finite rule family, not every possible function."})
    return {"training": {"patterns": training}, "probes": {"patterns": probes}, "audit": {"patterns": audit}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    files = {}
    for role, data in build_data().items():
        name = f"{role}/patterns.json"
        write_json(args.output / name, data)
        files[role] = {"path": name, "sha256": sha((args.output / name).read_bytes())}
    sources = ["scripts/serialization_pilot_contract.py", "scripts/serialization_pilot_worker.py",
               "src/oczy/experiments/meta_cortex/organ.py", "src/oczy/experiments/meta_cortex/calibration.py"]
    manifest = {
        "schema": "oczy/r23.5-dev-serialization/v1", "instrument_id": "r23.5-dev-serialization/v1",
        "human_authorization": {"date": "2026-09-12", "reply": "go ahead", "scope": "Small valid held-out serialization pilot; no R20 instrument repair or meta-test"},
        "files": files, "execution_sources": {name: sha((repo / name).read_bytes()) for name in sources},
        "preparer_sha256": sha(Path(__file__).read_bytes()), "system": SYSTEM, "query": QUERY, "refiner": REFINER,
        "organ_hash": "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea",
        "runtime_manifest_sha256": "a6214355c1c6b9192d435e62f3add6bef5db8c3a6c1cf3a55cb2a9dbfc91182e",
        "scorer_sha256": "e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742",
        "generation": {"mode": "greedy", "max_new_tokens": 32, "harness_max_new_tokens": 64},
        "training": {"seeds": [0, 1, 2], "steps": 24, "bank_width": 8, "feature_dim": 896,
                     "optimizer": "Adam", "learning_rate": 0.03, "initialization_std": 0.02,
                     "gradient_clip_norm": 1.0, "loss": "Mean per-example teacher-forced CE, including EOS",
                     "dtype": "float32", "early_stopping": False, "checkpoint_selection": "final step only"},
        "conditions": ["no_context", "with_context", "raw_reloaded", "text_harness", "soft_initial", "soft_restored", "soft_zeroed", "soft_swapped"],
        "measurement": {"scoring": "normalized-exact/v1", "recovery": "(accuracy_X - accuracy_no_context)/(accuracy_with_context - accuracy_no_context)",
                        "nonpositive_denominator": "undefined", "reference_recovery_criterion": 0.30,
                        "claim_scope": "Small DEV pilot; no full-study accept/refute, no layer sweep or full R20 claim"},
        "boundaries": {"train_reads": ["training"], "soft_restore_reads": ["probes", "numeric_states"],
                       "text_restore_reads": ["probes", "serialized_text"], "meta_test_authorized": False,
                       "train_all_patterns_regardless_of_context_score": True},
    }
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(args.output / "MANIFEST.json", manifest)
    print(json.dumps({"instrument": manifest["instrument_id"], "manifest_sha256": manifest["manifest_sha256"]}))


if __name__ == "__main__":
    main()
