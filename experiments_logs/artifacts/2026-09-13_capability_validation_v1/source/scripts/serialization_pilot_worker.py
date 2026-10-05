"""Phase-isolated execution of a previously frozen DEV serialization pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from oczy.experiments.meta_cortex.contracts import DialogueMessage  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402
from scripts.serialization_pilot_contract import (  # noqa: E402
    example_context,
    query_messages,
    read_manifest,
    read_split,
    sha,
    write_json,
)


def generate(organ, messages, bank, limit):
    return organ.generate(tuple(DialogueMessage(**m) for m in messages), bank, max_new_tokens=limit)


def record_probe(organ, scorer, probe, bank, manifest, condition, pattern, seed=None, context=None):
    messages = query_messages(probe["input"], context)
    output = generate(organ, messages, bank, manifest["generation"]["max_new_tokens"])
    return {"pattern": pattern, "seed": seed, "condition": condition,
            "input": probe["input"], "target": probe["target"], "generated": output,
            "correct": scorer.score_response(probe["target"], output),
            "prompt_sha256": sha(json.dumps(messages, sort_keys=True).encode())}


def reference(organ, scorer, args, manifest):
    training = read_split(args.root, manifest, args.phase, "training")
    probes = read_split(args.root, manifest, args.phase, "probes")
    bank = torch.zeros((1, 0, organ.feature_dim))
    rows, artifacts = [], []
    for train, heldout in zip(training, probes, strict=True):
        if train["index"] != heldout["index"]:
            raise ValueError("Pattern order mismatch")
        index = train["index"]
        text = example_context(train["examples"])
        for condition, context in (("no_context", None), ("with_context", text)):
            for probe in heldout["probes"]:
                rows.append(record_probe(organ, scorer, probe, bank, manifest, condition, index, context=context))
        refiner_messages = [{"role": "system", "content": manifest["system"]},
                            {"role": "user", "content": manifest["refiner"] + "\n\n" + text}]
        harness = generate(organ, refiner_messages, bank, manifest["generation"]["harness_max_new_tokens"])
        for label, value in (("raw", text), ("harness", harness)):
            name = f"p{index}-{label}.txt"
            (args.output / name).write_text(value)
            artifacts.append({"pattern": index, "kind": label, "path": name,
                              "sha256": sha((args.output / name).read_bytes()),
                              "utf8_bytes": len(value.encode()),
                              "tokens": len(organ._tokenizer.encode(value, add_special_tokens=False))})
        print(f"Reference complete for pattern {index}", flush=True)
    write_json(args.output / "text_manifest.json", {"pilot_manifest_sha256": manifest["manifest_sha256"], "artifacts": artifacts})
    return {"rows": rows, "optimizer_steps": 0, "text_artifacts": artifacts}


def save_numeric(path, value):
    value = value.detach().cpu().numpy().astype(np.float32, copy=False)
    with path.open("xb") as file:
        np.save(file, value, allow_pickle=False)
    return {"path": path.name, "sha256": sha(path.read_bytes()),
            "payload_bytes": value.nbytes, "file_bytes": path.stat().st_size,
            "shape": list(value.shape), "dtype": str(value.dtype)}


def load_numeric(root, entry, manifest):
    path = root / entry["path"]
    if sha(path.read_bytes()) != entry["sha256"]:
        raise ValueError("Numeric state hash mismatch")
    array = np.load(path, allow_pickle=False)
    expected_shape = (1, manifest["training"]["bank_width"], manifest["training"]["feature_dim"])
    if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Invalid numeric state")
    return torch.from_numpy(array.copy())


def train(organ, scorer, args, manifest):
    training = read_split(args.root, manifest, args.phase, "training")
    config = manifest["training"]
    state_root = args.output / "states"
    state_root.mkdir()
    states, fits = [], []
    for pattern in training:
        for seed in config["seeds"]:
            index = pattern["index"]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            bank = torch.nn.Parameter(torch.randn((1, config["bank_width"], organ.feature_dim), generator=generator) * config["initialization_std"])
            initial = save_numeric(state_root / f"p{index}-s{seed}-initial.npy", bank)
            optimizer = torch.optim.Adam([bank], lr=config["learning_rate"])
            trajectory = []
            for step in range(config["steps"]):
                start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for example in pattern["examples"]:
                    messages = tuple(DialogueMessage(**m) for m in query_messages(example["input"]))
                    # EOS is part of the fixed training objective, never a
                    # replacement scorer or constrained generation schedule.
                    target = example["target"] + organ._tokenizer.eos_token
                    loss = organ.teacher_forced_loss(messages, target, bank)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss / len(pattern["examples"])).backward()
                    losses.append(loss.detach().item())
                if bank.grad is None or not torch.isfinite(bank.grad).all() or bank.grad.norm().item() == 0:
                    raise ValueError("Missing/invalid soft-bank gradient")
                if any(p.grad is not None or p.requires_grad for p in organ._model.parameters()):
                    raise ValueError("Gradient reached frozen organ parameters")
                grad_norm = torch.nn.utils.clip_grad_norm_([bank], config["gradient_clip_norm"]).item()
                optimizer.step()
                if not torch.isfinite(bank).all():
                    raise ValueError("Nonfinite learned state")
                row = {"pattern": index, "seed": seed, "step": step + 1,
                       "mean_loss_before_update": sum(losses) / len(losses), "gradient_norm_before_clip": grad_norm,
                       "bank_norm_after_update": bank.detach().norm().item(), "seconds": time.monotonic() - start}
                trajectory.append(row)
                with (args.output / "trajectory.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            final = save_numeric(state_root / f"p{index}-s{seed}-trained.npy", bank)
            states.append({"pattern": index, "seed": seed, "initial": initial, "trained": final})
            fit = []
            for example in pattern["examples"]:
                output = generate(organ, query_messages(example["input"]), bank.detach(), manifest["generation"]["max_new_tokens"])
                fit.append({"input": example["input"], "target": example["target"], "generated": output,
                            "correct": scorer.score_response(example["target"], output)})
            fits.append({"pattern": index, "seed": seed, "training_fit": fit,
                         "first_loss": trajectory[0]["mean_loss_before_update"],
                         "last_loss": trajectory[-1]["mean_loss_before_update"]})
            organ.assert_frozen()
            del optimizer, bank
    write_json(state_root / "state_manifest.json", {"pilot_manifest_sha256": manifest["manifest_sha256"], "states": states})
    return {"optimizer_steps": len(training) * len(config["seeds"]) * config["steps"],
            "heldout_examples_read": 0, "fits": fits, "states": states}


def restore_text(organ, scorer, args, manifest):
    probes = read_split(args.root, manifest, args.phase, "probes")
    meta = json.loads((args.artifacts / "text_manifest.json").read_text())
    if meta["pilot_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Text artifact belongs to another pilot")
    rows = []
    bank = torch.zeros((1, 0, organ.feature_dim))
    for pattern in probes:
        for kind, condition in (("raw", "raw_reloaded"), ("harness", "text_harness")):
            entry = next(a for a in meta["artifacts"] if a["pattern"] == pattern["index"] and a["kind"] == kind)
            path = args.artifacts / entry["path"]
            if sha(path.read_bytes()) != entry["sha256"]:
                raise ValueError("Serialized text changed")
            for probe in pattern["probes"]:
                rows.append(record_probe(organ, scorer, probe, bank, manifest, condition, pattern["index"], context=path.read_text()))
    return {"rows": rows, "optimizer_steps": 0, "training_files_read": 0}


def restore_soft(organ, scorer, args, manifest):
    probes = read_split(args.root, manifest, args.phase, "probes")
    meta = json.loads((args.artifacts / "state_manifest.json").read_text())
    if meta["pilot_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Numeric artifact belongs to another pilot")
    rows = []
    for pattern in probes:
        index = pattern["index"]
        for seed in manifest["training"]["seeds"]:
            state = next(s for s in meta["states"] if s["pattern"] == index and s["seed"] == seed)
            donor = next(s for s in meta["states"] if s["pattern"] != index and s["seed"] == seed)
            initial = load_numeric(args.artifacts, state["initial"], manifest)
            trained = load_numeric(args.artifacts, state["trained"], manifest)
            swapped = load_numeric(args.artifacts, donor["trained"], manifest)
            for condition, bank in (("soft_initial", initial), ("soft_restored", trained),
                                    ("soft_zeroed", torch.zeros_like(trained)), ("soft_swapped", swapped)):
                for probe in pattern["probes"]:
                    row = record_probe(organ, scorer, probe, bank, manifest, condition, index, seed=seed)
                    rows.append(row)
                    with (args.output / "rows.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
            print(f"Restored pattern {index}, seed {seed}", flush=True)
    return {"rows": rows, "optimizer_steps": 0, "training_files_read": 0, "raw_text_artifacts_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "train", "restore_text", "restore_soft"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    manifest = read_manifest(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    scorer = FrozenScorer()
    if scorer.sha256 != manifest["scorer_sha256"]:
        raise ValueError("Scorer mismatch")
    organ = QwenFrozenOrgan.load()
    try:
        before = organ.parameter_hash()
        if before != manifest["organ_hash"]:
            raise ValueError("Historical organ mismatch")
        result = {"reference": reference, "train": train, "restore_text": restore_text, "restore_soft": restore_soft}[args.phase](organ, scorer, args, manifest)
        organ.assert_frozen()
        after = organ.parameter_hash()
        if before != after:
            raise ValueError("Organ changed")
        result.update({"phase": args.phase, "pid": os.getpid(), "classification": "DEV_PILOT_ONLY",
                       "manifest_sha256": manifest["manifest_sha256"], "organ_hash_before": before,
                       "organ_hash_after": after, "scorer_sha256": scorer.sha256,
                       "meta_test_accessed": False})
        write_json(args.output / "results.json", result)
    finally:
        organ.close()


if __name__ == "__main__":
    main()
