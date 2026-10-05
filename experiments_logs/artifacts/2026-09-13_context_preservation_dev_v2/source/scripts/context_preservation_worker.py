"""Phase-isolated execution of the frozen DEV-v2 interface and preservation experiment."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from oczy.experiments.meta_cortex.contracts import DialogueMessage  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402
from scripts.context_preservation_contract import (  # noqa: E402
    manifest_at,
    messages,
    read_data,
    sha,
    write_json,
)
from scripts.serialization_pilot_worker import generate, load_numeric, save_numeric  # noqa: E402


def append(path, row):
    with path.open("a") as file:
        file.write(json.dumps(row, allow_nan=False) + "\n")


def probe_row(organ, scorer, row, bank, args, manifest, *, condition, split, stage, seed=None, interface="concise_v2", context=None):
    prompt = messages(row, interface, context)
    output = generate(organ, prompt, bank, manifest["generation"]["max_new_tokens"])
    result = {**row, "condition": condition, "split": split, "stage": stage, "seed": seed,
              "generated": output, "correct": scorer.score_response(row["target"], output),
              "prompt_sha256": sha(json.dumps(prompt, sort_keys=True).encode())}
    append(args.output / "rows.jsonl", result)
    return result


def reference(organ, scorer, args, manifest):
    data = read_data(args.root, manifest, args.phase, "probes")
    oracle = read_data(args.root, manifest, args.phase, "oracle")["stages"]
    training = read_data(args.root, manifest, args.phase, "capacity_training")["rows"]
    text = "Corrected examples:\n" + "\n".join(f"Client {r['clients'][0]}, input {r['input']} -> correct output {r['target']}" for r in training)
    bank = torch.zeros((1, 0, organ.feature_dim))
    rows = []
    groups = [("calibration", 2, data["calibration"]), *(('confirmation', s["stage"], s["rows"]) for s in data["confirmation"])]
    for split, stage, probes in groups:
        rules = next(o for o in oracle if o["stage"] == stage)
        for condition in manifest["reference_conditions"]:
            if stage == 3 and condition["name"] not in ("concise_table", "resolved_oracle"):
                continue
            for row in probes:
                form = condition["oracle"]
                if form is None:
                    context = None
                elif form == "examples":
                    context = text
                elif form == "resolved":
                    context = rules[form][row["clients"][0]]
                else:
                    context = rules[form]
                rows.append(probe_row(organ, scorer, row, bank, args, manifest, condition=condition["name"],
                                      split=split, stage=stage, interface=condition["interface"], context=context))
            print(f"reference {split} stage={stage} condition={condition['name']}", flush=True)
    return {"rows": rows, "optimizer_steps": 0, "crossed_retrieval_bytes": len(text.encode())}


def optimize(organ, scorer, bank, examples, args, manifest, *, seed, condition, steps, coefficient=0.):
    config = manifest["training"]
    anchor = bank.detach().clone()
    optimizer = torch.optim.Adam([bank], lr=config["learning_rate"])
    history = []
    for step in range(steps):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for example in examples:
            prompt = tuple(DialogueMessage(**m) for m in messages(example))
            loss = organ.teacher_forced_loss(prompt, example["target"] + organ._tokenizer.eos_token, bank)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CE")
            (loss / len(examples)).backward()
            losses.append(loss.detach().item())
        penalty = coefficient * (bank - anchor).square().mean()
        if coefficient:
            penalty.backward()
        if bank.grad is None or not torch.isfinite(bank.grad).all() or bank.grad.norm().item() == 0:
            raise ValueError("Invalid bank gradient")
        if any(p.requires_grad or p.grad is not None for p in organ._model.parameters()):
            raise ValueError("Gradient reached frozen organ")
        norm = torch.nn.utils.clip_grad_norm_([bank], config["gradient_clip_norm"]).item()
        optimizer.step()
        if not torch.isfinite(bank).all():
            raise ValueError("Nonfinite bank")
        record = {"seed": seed, "condition": condition, "step": step + 1, "mean_ce_before_update": sum(losses) / len(losses),
                  "penalty_before_update": penalty.detach().item(), "gradient_norm_before_clip": norm,
                  "bank_norm_after_update": bank.detach().norm().item(), "mean_square_delta": (bank.detach() - anchor).square().mean().item(),
                  "seconds": time.monotonic() - started}
        history.append(record)
        append(args.output / "trajectory.jsonl", record)
        print(json.dumps(record), flush=True)
    fit = []
    for example in examples:
        output = generate(organ, messages(example), bank.detach(), manifest["generation"]["max_new_tokens"])
        fit.append({**example, "generated": output, "correct": scorer.score_response(example["target"], output)})
    return {"seed": seed, "condition": condition, "training_fit": fit, "first_ce": history[0]["mean_ce_before_update"], "last_ce": history[-1]["mean_ce_before_update"]}


def load_states(args, manifest):
    meta = json.loads((args.artifacts / "state_manifest.json").read_text())
    if meta["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("State manifest belongs to another experiment")
    return meta["states"]


def train_capacity(organ, scorer, args, manifest):
    examples = read_data(args.root, manifest, args.phase, "capacity_training")["rows"]
    config = manifest["training"]
    directory = args.output / "states"
    directory.mkdir()
    states, fits = [], []
    for seed in config["seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bank = torch.nn.Parameter(torch.randn((1, config["bank_width"], organ.feature_dim), generator=generator) * config["initialization_std"])
        initial = save_numeric(directory / f"seed{seed}-initial.npy", bank)
        fits.append(optimize(organ, scorer, bank, examples, args, manifest, seed=seed, condition="joint", steps=config["capacity_steps"]))
        trained = save_numeric(directory / f"seed{seed}-joint.npy", bank)
        states.append({"seed": seed, "initial": initial, "trained": trained})
        organ.assert_frozen()
    write_json(directory / "state_manifest.json", {"manifest_sha256": manifest["manifest_sha256"], "states": states})
    return {"states": states, "fits": fits, "optimizer_steps": len(states) * config["capacity_steps"], "heldout_examples_read": 0}


def restore_capacity(organ, scorer, args, manifest):
    probes = next(s["rows"] for s in read_data(args.root, manifest, args.phase, "probes")["confirmation"] if s["stage"] == 2)
    rows = []
    for state in load_states(args, manifest):
        seed = state["seed"]
        initial = load_numeric(args.artifacts, state["initial"], manifest)
        trained = load_numeric(args.artifacts, state["trained"], manifest)
        controls = [("initial", initial), ("joint_restored", trained)]
        if seed == manifest["training"]["seeds"][0]:
            controls.append(("zeroed", torch.zeros_like(trained)))
        for condition, bank in controls:
            for row in probes:
                rows.append(probe_row(organ, scorer, row, bank, args, manifest, condition=condition, split="confirmation", stage=2, seed=seed))
            print(f"restore_capacity seed={seed} condition={condition}", flush=True)
    return {"rows": rows, "optimizer_steps": 0, "training_files_read": 0}


def train_correction(organ, scorer, args, manifest):
    gate = json.loads(args.admission.read_text())
    if gate != {"manifest_sha256": manifest["manifest_sha256"], "admitted": True}:
        raise ValueError("Correction is not admitted")
    examples = read_data(args.root, manifest, args.phase, "correction_training")["rows"]
    config = manifest["training"]
    directory = args.output / "states"
    directory.mkdir()
    states, fits = [], []
    for state in load_states(args, manifest):
        for condition, coefficient in config["correction_arms"].items():
            bank = torch.nn.Parameter(load_numeric(args.artifacts, state["trained"], manifest))
            fits.append(optimize(organ, scorer, bank, examples, args, manifest, seed=state["seed"], condition=condition,
                                 steps=config["correction_steps"], coefficient=coefficient))
            entry = save_numeric(directory / f"seed{state['seed']}-{condition}.npy", bank)
            states.append({"seed": state["seed"], "condition": condition, "trained": entry, "parent_sha256": state["trained"]["sha256"]})
    write_json(directory / "state_manifest.json", {"manifest_sha256": manifest["manifest_sha256"], "states": states})
    return {"states": states, "fits": fits, "optimizer_steps": len(states) * config["correction_steps"], "old_examples_read": 0}


def restore_correction(organ, scorer, args, manifest):
    probes = next(s["rows"] for s in read_data(args.root, manifest, args.phase, "probes")["confirmation"] if s["stage"] == 3)
    rows = []
    for state in load_states(args, manifest):
        bank = load_numeric(args.artifacts, state["trained"], manifest)
        for row in probes:
            rows.append(probe_row(organ, scorer, row, bank, args, manifest, condition=state["condition"], split="confirmation", stage=3, seed=state["seed"]))
    return {"rows": rows, "optimizer_steps": 0, "training_files_read": 0}


def main():
    phases = {"reference": reference, "train_capacity": train_capacity, "restore_capacity": restore_capacity,
              "train_correction": train_correction, "restore_correction": restore_correction}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=phases)
    for name in ("root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    manifest = manifest_at(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    scorer = FrozenScorer()
    assert scorer.sha256 == manifest["scorer_sha256"]
    organ = QwenFrozenOrgan.load()
    try:
        before = organ.parameter_hash()
        assert before == manifest["organ_hash"]
        result = phases[args.phase](organ, scorer, args, manifest)
        organ.assert_frozen()
        after = organ.parameter_hash()
        assert after == before
        result.update({"phase": args.phase, "pid": os.getpid(), "manifest_sha256": manifest["manifest_sha256"],
                       "organ_hash_before": before, "organ_hash_after": after, "scorer_sha256": scorer.sha256, "meta_test_accessed": False})
        write_json(args.output / "results.json", result)
    finally:
        organ.close()


if __name__ == "__main__":
    main()
