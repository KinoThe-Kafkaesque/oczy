"""Freeze the authorized DEV-v2 interface ladder, crossed curriculum and conditional preservation comparison."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.context_preservation_contract import SYSTEM, canonical, sha, write_json  # noqa: E402


def build_data():
    stages = {1: {"amber": "vek"}, 2: {"amber": "vek", "cobalt": "mip"}, 3: {"amber": "zul", "cobalt": "mip"}}
    def row(client, word, stage):
        return {"clients": [client], "input": word, "target": word + stages[stage].get(client, ""), "category": client}
    teaching_words = ("oak", "pine", "elm")
    training = [row(client, word, 2) for client in ("amber", "cobalt", "silver") for word in teaching_words]
    correction = [row("amber", word, 3) for word in teaching_words]
    calibration = [row(client, word, 2) for client in ("amber", "cobalt", "silver", "quartz") for word in ("pear", "plum", "kiwi", "fig")]
    confirmation = [{"stage": stage, "rows": [row(client, word, stage) for client in ("amber", "cobalt", "silver", "quartz")
                                             for word in ("lime", "melon", "grape", "peach")]} for stage in (2, 3)]
    oracle = []
    for stage in (2, 3):
        rules = stages[stage]
        original = "Current complete rules: " + "; ".join(
            f"client {c}: append the literal suffix {rules[c]} to the whole input" if c in rules else f"client {c}: copy the input unchanged"
            for c in ("amber", "cobalt", "silver")) + ". Apply clients sequentially in the requested order."
        table = "Rules (write the input first, then the suffix, with no spaces):\n" + "\n".join(
            f'{c}: input + "{rules[c]}"' for c in ("amber", "cobalt")) + "\nAny other client: input unchanged."
        resolved = {c: f'Write the input followed immediately by the literal text "{rules[c]}".' if c in rules else "Copy the input exactly."
                    for c in ("amber", "cobalt", "silver", "quartz")}
        oracle.append({"stage": stage, "original": original, "table": table, "resolved": resolved})
    assert len(training) == 9 and len(correction) == 3
    assert not set(teaching_words) & {r["input"] for s in confirmation for r in s["rows"]}
    assert not set(teaching_words) & {r["input"] for r in calibration}
    # Identical words require different outputs across clients: input-only lookup cannot fit.
    assert all(len({r["target"] for r in training if r["input"] == w}) == 3 for w in teaching_words)
    return {"capacity_training": {"rows": training}, "correction_training": {"rows": correction},
            "probes": {"calibration": calibration, "confirmation": confirmation}, "oracle": {"stages": oracle},
            "audit": {"crossed_inputs": True, "input_only_lookup_cannot_fit_training": True,
                      "identity_training_client": "silver", "untaught_confirmation_client": "quartz",
                      "teaching_confirmation_disjoint": True, "old_probes_are_dev_calibration_only": True,
                      "joint_training_is_not_continual_learning": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    files = {}
    for role, data in build_data().items():
        name = f"{role}/data.json"
        write_json(args.output / name, data)
        files[role] = {"path": name, "sha256": sha((args.output / name).read_bytes())}
    prior = json.loads((repo / "experiments/capability-validation-v1/MANIFEST.json").read_text())
    sources = ["scripts/context_preservation_contract.py", "scripts/context_preservation_worker.py", "scripts/run_context_preservation.py",
               "scripts/capability_validation_contract.py", "scripts/serialization_pilot_contract.py", "scripts/serialization_pilot_worker.py",
               "scripts/reproduce_r20_dev.py", "infrastructure/kaggle/runtime_manifest.py",
               "src/oczy/experiments/meta_cortex/organ.py", "src/oczy/experiments/meta_cortex/calibration.py", "src/oczy/experiments/meta_cortex/contracts.py"]
    manifest = {"instrument_id": "oczy/context-preservation/dev-v2", "parent_manifest_sha256": prior["manifest_sha256"],
                "human_authorization": {"date": "2026-09-13", "request": "proceed on learning within the right context while preserving existing knowledge; language interface needs further work",
                                        "scope": "New versioned local DEV interface/curriculum/learner comparisons; no historical instrument edits or meta-test"},
                "files": files, "execution_sources": {s: sha((repo / s).read_bytes()) for s in sources},
                "preparer_sha256": sha(Path(__file__).read_bytes()), "system": SYSTEM,
                **{k: prior[k] for k in ("organ_hash", "runtime_manifest_sha256", "scorer_sha256")},
                "training": {"bank_width": 8, "feature_dim": 896, "seeds": [0, 1, 2], "initialization_std": .02,
                             "learning_rate": .03, "gradient_clip_norm": 1., "capacity_steps": 48, "correction_steps": 24,
                             "optimizer": "Adam", "dtype": "float32", "interface": "concise_v2", "checkpoint_selection": "final only",
                             "loss": "Mean example CE including EOS", "correction_arms": {"unprotected": 0., "proximal": 10.},
                             "proximal_loss": "coefficient * mean((bank - pre_update_bank)^2); anchor transient, not extra persistent state",
                             "correction_replay": False, "joint_capacity_is_development_only": True},
                "generation": {"max_new_tokens": 32, "mode": "greedy"},
                "reference_conditions": [
                    {"name": "v1_full", "interface": "v1", "oracle": "original"},
                    {"name": "query_full", "interface": "query_v2", "oracle": "original"},
                    {"name": "concise_full", "interface": "concise_v2", "oracle": "original"},
                    {"name": "concise_table", "interface": "concise_v2", "oracle": "table"},
                    {"name": "resolved_oracle", "interface": "concise_v2", "oracle": "resolved"},
                    {"name": "no_context", "interface": "concise_v2", "oracle": None},
                    {"name": "crossed_example_retrieval", "interface": "concise_v2", "oracle": "examples"}],
                "measurement": {"scorer": "normalized-exact/v1 unchanged", "fresh_confirmation_words": ["lime", "melon", "grape", "peach"],
                                "capacity_gate": "All 16 fresh single-context probes correct for all three seeds after reload, plus all 9 teaching fits per seed",
                                "if_gate_fails": "Preserve all results and block correction optimizer; no claim of representational impossibility",
                                "if_gate_passes": "Compare unprotected vs proximal correction from identical joint states; unchanged target/probe scoring",
                                "selection": "concise_v2 fixed before any result; no selection of interface, seed or checkpoint from heldout performance",
                                "ladder_causality": "Only adjacent v1_full -> query_full -> concise_full -> concise_table isolate one interface change each",
                                "resolved_oracle": "Oracle-assisted rule selection only; cannot establish learned context selection",
                                "primary_scope": "DEV representational diagnostic and conditional correction/preservation; not full R20 or unbounded continual learning"},
                "boundaries": {"meta_test_authorized": False, "no_old_examples_during_correction": True}}
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(args.output / "MANIFEST.json", manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
