"""Execute registered historical replays and new-state behavioral checks."""

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from oczy.experiments.meta_cortex.contracts import DialogueMessage
from oczy.experiments.meta_cortex.generation_v2 import generate_batch
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan
from scripts import capability_validation_worker as cap
from scripts import serialization_pilot_worker as pilot
from scripts.capability_validation_actions import run_case
from scripts.capability_validation_contract import manifest_at, messages, read_data
from scripts.diversity_dev.contract import example_text
from scripts.regression_dev.contract import (
    CAP,
    DIVERSITY,
    PILOT,
    REPLAYS,
    REPO,
    compare_rows,
    read,
    verify,
    write_json,
)
from scripts.serialization_pilot_contract import read_manifest


def run(root, output, candidate):
    manifest = verify(root)
    dm = manifest_at(REPO / DIVERSITY)
    state_meta = read(candidate / "state_manifest.json")
    if state_meta["manifest_sha256"] != manifest["candidate_manifest_sha256"] or [r["seed"] for r in state_meta["states"]] != [0, 1, 2]:
        raise ValueError("Candidate release differs from registered study")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
    before = organ.parameter_hash()
    assert before == dm["organ_hash"] and scorer.sha256 == dm["scorer_sha256"]
    results, verdicts = {}, {}

    def save(name, result, expected=None):
        assert len(result["rows"]) == manifest["counts"][name], name
        results[name] = result
        write_json(output / f"{name}.json", result)
        if expected is not None:
            verdicts[name] = {"rows": len(expected), "exact_replay": compare_rows(result["rows"], expected)}
        print(f"Completed {name}: {len(result['rows'])} rows", flush=True)

    try:
        cm = manifest_at(REPO / "experiments/capability-validation-v1")
        for phase in ("reference", "restore", "actions"):
            phase_output = output / f"capability_{phase}"
            phase_output.mkdir()
            args = SimpleNamespace(root=REPO / "experiments/capability-validation-v1", output=phase_output, phase=phase,
                                   artifacts=REPO / CAP / "run/released_states", text_artifact=REPO / CAP / "run/reference/stage3-active.txt")
            name = "capability_" + phase
            save(name, getattr(cap, phase)(organ, scorer, args, cm), read(REPO / REPLAYS[name][0])["rows"])
        pm = read_manifest(REPO / "experiments/r23.5-serialization-dev/pilot_v1")
        for phase in ("restore_soft", "restore_text"):
            phase_output = output / ("pilot_" + phase)
            phase_output.mkdir()
            args = SimpleNamespace(root=REPO / "experiments/r23.5-serialization-dev/pilot_v1", output=phase_output, phase=phase,
                                   artifacts=REPO / PILOT / ("released_states" if phase == "restore_soft" else "released_text"))
            name = "pilot_" + phase
            save(name, getattr(pilot, phase)(organ, scorer, args, pm), read(REPO / REPLAYS[name][0])["rows"])
        zero_width = torch.zeros((1, 0, organ.feature_dim))
        cases = [r for r in read(REPO / "experiments/language-interface-dev-v3/cases.json") if r["condition"] == "direct_operation"]
        old = [r for r in read(REPO / "experiments_logs/2026-09-13_language_interface_dev_v3.json")["rows"] if r["condition"] == "direct_operation"]
        rows = []
        for case, previous in zip(cases, old, strict=True):
            assert all(case[k] == previous[k] for k in ("input", "category", "split", "target"))
            generated = pilot.generate(organ, case["messages"], zero_width, 32)
            rows.append({**case, "generated": generated, "correct": scorer.score_response(case["target"], generated),
                         "previous_generated": previous["generated"], "exact_replay": generated == previous["generated"]})
        save("direct_operation", {"rows": rows})
        decoder_cases = read(REPO / "experiments/decoder-parity-dev-v1/cases.json")
        decoder_old = read(REPO / "experiments_logs/2026-09-13_decoder_parity_dev_v1.json")["rows"]
        banks = {"none": zero_width, "joint": torch.from_numpy(np.load(REPO / "experiments/decoder-parity-dev-v1/joint.npy", allow_pickle=False).copy())}
        for size, name in ((1, "decoder_single"), (4, "decoder_batch4")):
            rows = []
            for start in range(0, len(decoder_cases), size):
                batch = decoder_cases[start:start + size]
                prompts = [tuple(DialogueMessage(**m) for m in case["messages"]) for case in batch]
                generated = generate_batch(organ, prompts, torch.cat([banks[c["bank"]] for c in batch]), 32)
                for index, text in enumerate(generated, start):
                    expected = decoder_old[index]["outputs"]["scalar"]
                    rows.append({**decoder_cases[index], "generated": text, "previous_generated": expected,
                                 "correct": scorer.score_response(decoder_cases[index]["target"], text), "exact_replay": text == expected})
            save(name, {"rows": rows})
        conditions = []
        for state in state_meta["states"]:
            control = next(c for c in dm["controls"] if c["seed"] == state["seed"])
            for name, directory, entry in (("initial", REPO / DIVERSITY / "controls", control["initial"]),
                                          ("control", REPO / DIVERSITY / "controls", control["trained"]),
                                          ("diversity", candidate, state["trained"])):
                conditions.append((name, state["seed"], pilot.load_numeric(directory, entry, dm), None))
            if state["seed"] == 0:
                conditions.append(("zeroed", 0, torch.zeros_like(conditions[-1][2]), None))
        probes = read_data(REPO / "experiments/capability-validation-v1", cm, "restore", "probes")["stages"][1]["probes"]
        rows = []
        for condition, seed, bank, _ in conditions:
            for case in probes:
                generated = pilot.generate(organ, messages(case), bank, cm["generation"]["max_new_tokens"])
                rows.append({**case, "condition": condition, "seed": seed, "generated": generated, "correct": scorer.score_response(case["target"], generated)})
        save("candidate_probes", {"rows": rows})
        teaching = read(REPO / DIVERSITY / "capacity_training/data.json")
        for arm, examples in (("control", teaching["groups"][0]), ("diversity", teaching["rows"])):
            conditions.append(("text_" + arm, None, zero_width, example_text(examples)))
        rows = []
        cases = read_data(REPO / "experiments/capability-validation-v1", cm, "actions", "actions")["cases"]
        for condition, seed, bank, context in conditions:
            def actor(prompt, bank=bank, context=context):
                if context is not None:
                    prompt = [prompt[0], {"role": "user", "content": context}, *prompt[1:]]
                return pilot.generate(organ, prompt, bank, cm["generation"]["action_max_new_tokens"])
            for case in cases:
                rows.append({**run_case(case, actor, max_calls=cm["generation"]["action_max_calls"]), "condition": condition, "seed": seed})
        save("candidate_actions", {"rows": rows})
        for name in ("direct_operation", "decoder_single", "decoder_batch4"):
            verdicts[name] = {"rows": len(results[name]["rows"]), "exact_replay": all(r["exact_replay"] for r in results[name]["rows"])}
        organ.assert_frozen()
        after = organ.parameter_hash()
        assert before == after
        verify(root)
        write_json(output / "results.json", {"manifest_sha256": manifest["manifest_sha256"], "verdicts": verdicts,
                   "historical_replay_exact": all(v["exact_replay"] for v in verdicts.values()), "counts": {k: len(v["rows"]) for k, v in results.items()},
                   "candidate_state_manifest": state_meta, "organ_hash_before": before, "organ_hash_after": after,
                   "scorer_sha256": scorer.sha256, "optimizer_steps": 0, "meta_test_accessed": False})
    finally:
        organ.close()


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "candidate"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.candidate)
