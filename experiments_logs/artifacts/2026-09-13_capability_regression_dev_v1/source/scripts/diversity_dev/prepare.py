"""Freeze a fixed-update teaching-diversity comparison and paired controls."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from scripts.diversity_dev.contract import canonical, manifest_at, sha, write_json


def build_data(parent_root):
    groups = []
    for words in (("oak", "pine", "elm"), ("birch", "cedar", "ash"), ("beech", "maple", "willow")):
        groups.append([{"clients": [client], "input": word, "target": word + suffix, "category": client}
                       for client, suffix in (("amber", "vek"), ("cobalt", "mip"), ("silver", "")) for word in words])
    original = json.loads((parent_root / "capacity_training/data.json").read_text())["rows"]
    assert groups[0] == original
    fresh = []
    for stage in (2, 3):
        fresh.append({"stage": stage, "rows": [
            {"clients": [client], "input": word, "target": word + suffix, "category": client}
            for client, suffix in (("amber", "vek" if stage == 2 else "zul"), ("cobalt", "mip"), ("silver", ""), ("quartz", ""))
            for word in ("berry", "cherry", "papaya", "orange")]})
    old = json.loads((parent_root / "probes/data.json").read_text())["confirmation"][0]["rows"]
    return {"capacity_training": {"groups": groups, "rows": [r for group in groups for r in group]},
            "probes": {"calibration": old, "confirmation": fresh},
            "oracle": json.loads((parent_root / "oracle/data.json").read_text()),
            "correction_training": json.loads((parent_root / "correction_training/data.json").read_text())}


def prepare(root, parent_run):
    repo = Path(__file__).resolve().parents[2]
    parent_root = repo / "experiments/context-preservation-dev-v2"
    parent = manifest_at(parent_root)
    evidence = json.loads((parent_run / "train_capacity/results.json").read_text())
    assert evidence["manifest_sha256"] == parent["manifest_sha256"]
    assert evidence["organ_hash_before"] == evidence["organ_hash_after"] == parent["organ_hash"]
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, value in build_data(parent_root).items():
        relative = f"{role}/data.json"
        write_json(root / relative, value)
        files[role] = {"path": relative, "sha256": sha((root / relative).read_bytes())}
    controls = copy.deepcopy(evidence["states"])
    (root / "controls").mkdir()
    for state in controls:
        for label in ("initial", "trained"):
            entry = state[label]
            source = parent_run / "released_capacity" / entry["path"]
            assert sha(source.read_bytes()) == entry["sha256"]
            shutil.copyfile(source, root / "controls" / entry["path"])
    config = copy.deepcopy(parent["training"])
    config.update({"schedule": "groups[zero_based_step % 3]", "examples_per_update": 9,
                   "unique_words_per_context": 9, "control_unique_words_per_context": 3})
    sources = dict(parent["execution_sources"])
    for name in ("__init__", "contract", "train", "evaluate", "run"):
        relative = f"scripts/diversity_dev/{name}.py"
        sources[relative] = sha((repo / relative).read_bytes())
    manifest = {"instrument_id": "oczy/scoped-diversity/dev-v3", "parent_manifest_sha256": parent["manifest_sha256"],
                "human_authorization": "User request 2026-09-13: get on the next objective, check previous ones against regression, and package one experience",
                "scope": "Local DEV only; new version, historical instruments immutable; no meta-test or remote launch",
                "execution_sources": sources, "preparer_sha256": sha(Path(__file__).read_bytes()), "files": files, "controls": controls,
                **{k: parent[k] for k in ("organ_hash", "scorer_sha256", "runtime_manifest_sha256", "generation")},
                "training": config, "parent_first_ce": {str(f["seed"]): f["first_ce"] for f in evidence["fits"]},
                "references": ["no_context", "direct_oracle", "complete_table", "text_control", "text_diversity",
                               "chat_control", "chat_diversity", "zlib_control", "zlib_diversity"],
                "measurement": {"primary": "Paired new-word scoped application and neutral retention; all seeds, no selection",
                                "gate": "All 27 diversity teaching fits and all 16 new confirmation cases correct for every seed",
                                "variable": "Teaching-word diversity only; same initial states, optimizer, interface, 48 updates and 9 examples/update",
                                "compute_limit": "Equal update/example counts, not identical FLOPs; token-length distributions and elapsed time reported",
                                "control": "Previously audited frozen DEV-v2 final checkpoints; exact initial hash and first loss checked",
                                "correction": "If admitted, unchanged 24-step unprotected/proximal comparison using only new amber examples",
                                "fresh_words": ["berry", "cherry", "papaya", "orange"],
                                "selection": "All seeds and final states reported; no best seed, adaptive curriculum or checkpoint selection",
                                "baselines_not_run": ["logit-bias", "rerank"]}}
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(root / "MANIFEST.json", manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.parent_run.resolve())
