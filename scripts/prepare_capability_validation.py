"""Freeze a new local DEV capability suite under the user's explicit validation request."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.capability_validation_contract import (  # noqa: E402
    SYSTEM,
    canonical,
    example_text,
    latest_lessons,
    sha,
    write_json,
)


def build_data():
    specifications = [("amber", "vek", ["oak", "pine", "elm"]),
                      ("cobalt", "mip", ["birch", "cedar", "ash"]),
                      ("amber", "zul", ["oak", "pine", "elm"])]
    lessons, probes, oracle, audit, state = [], [], [], [], {}
    candidates = {"identity": lambda x: x, "upper": str.upper, "reverse": lambda x: x[::-1]}
    for affix in ("vek", "mip", "zul", "tav"):
        candidates["append_" + affix] = lambda x, a=affix: x + a
        candidates["prepend_" + affix] = lambda x, a=affix: a + x
    for stage, (client, suffix, inputs) in enumerate(specifications, 1):
        examples = [{"clients": [client], "input": word, "target": word + suffix} for word in inputs]
        compatible = [name for name, fn in candidates.items() if all(fn(r["input"]) == r["target"] for r in examples)]
        assert compatible == ["append_" + suffix]
        lessons.append({"stage": stage, "client": client, "examples": examples})
        state[client] = suffix
        rows = []
        for scope in ("amber", "cobalt", "silver"):
            for word in ("pear", "plum", "kiwi", "fig"):
                rows.append({"clients": [scope], "input": word, "target": word + state.get(scope, ""), "category": scope})
        for order in (("amber", "cobalt"), ("cobalt", "amber")):
            for word in ("pear", "plum"):
                rows.append({"clients": list(order), "input": word, "target": word + "".join(state.get(c, "") for c in order), "category": "composition"})
        assert not set(inputs) & {r["input"] for r in rows}
        probes.append({"stage": stage, "probes": rows})
        text = "Current complete rules: " + "; ".join(
            f"client {c}: append the literal suffix {state[c]} to the whole input" if c in state else f"client {c}: copy the input unchanged"
            for c in ("amber", "cobalt", "silver")) + ". Apply clients sequentially in the requested order."
        oracle.append({"stage": stage, "context": text})
        audit.append({"stage": stage, "candidate_count": len(candidates), "compatible": compatible,
                      "training_probe_inputs_disjoint": True, "composition_closed_on_all_strings": True,
                      "unassigned_scope_explicit_identity": True, "limited_to_declared_rule_family": True})
    def call(tool, **arguments):
        return {"tool": tool, "arguments": arguments}
    cases = [
        {"name": "read", "initial_files": {"source.txt": "delta\n"},
         "request": "Read source.txt with read_file, then return its exact contents as the final string.",
         "expected_calls": [call("read_file", path="source.txt")], "expected_files": {"source.txt": "delta\n"}, "expected_final": "delta\n"},
        {"name": "case_sensitive_write", "initial_files": {"report.txt": "keep"},
         "request": "Use write_file to create Report.txt containing exactly ready (no newline). Preserve report.txt. Then return final string done.",
         "expected_calls": [call("write_file", path="Report.txt", content="ready")],
         "expected_files": {"report.txt": "keep", "Report.txt": "ready"}, "expected_final": "done"},
        {"name": "read_then_edit", "initial_files": {"config.txt": "mode=slow\nowner=amber\n"},
         "request": "Read config.txt with read_file. Then use replace_text on config.txt to replace the exact text slow with fast. Preserve everything else. Return final string done.",
         "expected_calls": [call("read_file", path="config.txt"), call("replace_text", path="config.txt", old="slow", new="fast")],
         "expected_files": {"config.txt": "mode=fast\nowner=amber\n"}, "expected_final": "done"},
        {"name": "grounded_copy", "initial_files": {"code.txt": "CODE-731"},
         "request": "Read code.txt with read_file. Use write_file to create copy.txt containing exactly the contents you read. Preserve code.txt. Return final string done.",
         "expected_calls": [call("read_file", path="code.txt"), call("write_file", path="copy.txt", content="CODE-731")],
         "expected_files": {"code.txt": "CODE-731", "copy.txt": "CODE-731"}, "expected_final": "done"},
    ]
    return {"training": {"lessons": lessons}, "probes": {"stages": probes}, "oracle": {"stages": oracle},
            "audit": {"stages": audit}, "actions": {"cases": cases}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    data_bundle = build_data()
    for role, data in data_bundle.items():
        name = f"{role}/data.json"
        write_json(args.output / name, data)
        files[role] = {"path": name, "sha256": sha((args.output / name).read_bytes())}
    repo = Path(__file__).resolve().parents[1]
    sources = ["scripts/capability_validation_contract.py", "scripts/capability_validation_actions.py",
               "scripts/capability_validation_worker.py", "scripts/run_capability_validation.py",
               "scripts/serialization_pilot_contract.py", "scripts/serialization_pilot_worker.py",
               "scripts/reproduce_r20_dev.py", "infrastructure/kaggle/runtime_manifest.py",
               "src/oczy/experiments/meta_cortex/organ.py", "src/oczy/experiments/meta_cortex/calibration.py",
               "src/oczy/experiments/meta_cortex/contracts.py"]
    manifest = {
        "instrument_id": "oczy/capability-validation/dev-v1", "files": files,
        "human_authorization": {"date": "2026-09-12", "request": "validate these: selective application, accumulating knowledge, correcting knowledge, composition, useful compression, reliable action",
                                "scope": "New bounded local DEV validation; no modification of old instruments or meta-test"},
        "execution_sources": {name: sha((repo / name).read_bytes()) for name in sources},
        "preparer_sha256": sha(Path(__file__).read_bytes()), "system": SYSTEM,
        "final_active_text_sha256": sha(example_text(latest_lessons(data_bundle["training"]["lessons"])).encode()),
        "organ_hash": "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea",
        "runtime_manifest_sha256": "a6214355c1c6b9192d435e62f3add6bef5db8c3a6c1cf3a55cb2a9dbfc91182e",
        "scorer_sha256": "e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742",
        "training": {"seeds": [0, 1, 2], "steps": 24, "bank_width": 8, "feature_dim": 896,
                     "learning_rate": 0.03, "initialization_std": 0.02, "gradient_clip_norm": 1.0,
                     "dtype": "float32", "optimizer": "Adam", "reset_optimizer_each_lesson": True,
                     "loss": "Mean CE of three corrected examples plus EOS", "replay": False,
                     "shared_bank_for_all_clients": True, "checkpoint_selection": "final step only", "early_stopping": False},
        "generation": {"mode": "greedy", "max_new_tokens": 32, "action_max_new_tokens": 128, "action_max_calls": 4},
        "conditions": ["no_context", "raw_history", "latest_example_retrieval", "complete_oracle", "zlib_retrieval",
                       "initial", "zeroed", "learned_restored"],
        "measurement": {"scoring": "normalized-exact/v1", "paired_probes_repeated_across_stages_seeds": True,
                        "battery_pass": "All required cases on all three seeds; otherwise report partial counts and counterexamples, not full-study accept/refute",
                        "selective": "Stage 1 amber and two unassigned clients; stage 2/3 each scope separately",
                        "accumulating": "Stage 2 learns cobalt and retains previously correct amber; report acquisition prerequisite",
                        "correcting": "Stage 3 learns replacement amber and retains previously correct cobalt; report acquisition prerequisite",
                        "composition": "Stage 2/3, both client orders, novel inputs; prerequisite component successes reported",
                        "compression": "Exact paired behavior retained and total per-memory bytes below raw active teaching text; shared runtime excluded equally",
                        "actions": "Exact full tool sequence and args, exact final string, exact entire filesystem snapshot, no execution error",
                        "scope": "One explicit-context append-rule family, 4 held-out base words, 4 local tool tasks; no broad competence or promotion threshold"},
        "boundaries": {"train_reads": ["training"], "restore_reads": ["probes", "numeric_states"], "meta_test_authorized": False},
    }
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(args.output / "MANIFEST.json", manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
