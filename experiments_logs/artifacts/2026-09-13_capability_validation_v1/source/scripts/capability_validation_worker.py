"""Execute one phase of the frozen six-capability DEV battery."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from oczy.experiments.meta_cortex.contracts import DialogueMessage  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402
from scripts.capability_validation_actions import run_case  # noqa: E402
from scripts.capability_validation_contract import (  # noqa: E402
    example_text,
    latest_lessons,
    manifest_at,
    messages,
    read_data,
    sha,
    write_json,
)
from scripts.serialization_pilot_worker import generate, load_numeric, save_numeric  # noqa: E402


def append_row(path, row):
    with path.open("a") as file:
        file.write(json.dumps(row, allow_nan=False) + "\n")


def score_probe(organ, scorer, probe, bank, manifest, stage, condition, seed=None, context=None, cache=None):
    prompt = messages(probe, context)
    key = (sha(bank.detach().cpu().numpy().tobytes()), json.dumps(prompt, sort_keys=True))
    cached = cache is not None and key in cache
    output = cache[key] if cached else generate(organ, prompt, bank, manifest["generation"]["max_new_tokens"])
    if cache is not None:
        cache[key] = output
    return {"stage": stage, "condition": condition, "seed": seed, **probe, "generated": output,
            "correct": scorer.score_response(probe["target"], output), "prompt_sha256": sha(key[1].encode()),
            "deterministic_output_reused": cached}


def reference(organ, scorer, args, manifest):
    lessons = read_data(args.root, manifest, args.phase, "training")["lessons"]
    stages = read_data(args.root, manifest, args.phase, "probes")["stages"]
    oracle = read_data(args.root, manifest, args.phase, "oracle")["stages"]
    bank = torch.zeros((1, 0, organ.feature_dim))
    rows, storage, cache = [], [], {}
    for stage in stages:
        index = stage["stage"]
        history = example_text(lessons[:index])
        active = example_text(latest_lessons(lessons[:index]))
        raw_path = args.output / f"stage{index}-active.txt"
        raw_path.write_text(active)
        compressed_path = args.output / f"stage{index}-active.zlib"
        compressed_path.write_bytes(zlib.compress(active.encode(), level=9))
        restored = zlib.decompress(compressed_path.read_bytes()).decode()
        assert restored == active
        storage.append({"stage": index, "active_example_bytes": len(active.encode()),
                        "chronological_example_bytes": len(history.encode()), "zlib_file_bytes": compressed_path.stat().st_size,
                        "zlib_roundtrip_exact": True, "zlib_sha256": sha(compressed_path.read_bytes()),
                        "raw_sha256": sha(raw_path.read_bytes()), "required_decoder_metadata": "codec and UTF-8 fixed in shared contract"})
        contexts = [("no_context", None), ("raw_history", history), ("latest_example_retrieval", active),
                    ("complete_oracle", oracle[index - 1]["context"]), ("zlib_retrieval", restored)]
        for condition, context in contexts:
            for probe in stage["probes"]:
                # Zlib gets independent generation after a real decode, not a copied score.
                row = score_probe(organ, scorer, probe, bank, manifest, index, condition, context=context,
                                  cache=None if condition == "zlib_retrieval" else cache)
                rows.append(row)
                append_row(args.output / "rows.jsonl", row)
            print(f"reference stage={index} condition={condition}", flush=True)
    return {"rows": rows, "storage": storage, "optimizer_steps": 0}


def train(organ, scorer, args, manifest):
    lessons = read_data(args.root, manifest, args.phase, "training")["lessons"]
    config = manifest["training"]
    state_root = args.output / "states"
    state_root.mkdir()
    states, fits = [], []
    for seed in config["seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bank = torch.nn.Parameter(torch.randn((1, config["bank_width"], organ.feature_dim), generator=generator) * config["initialization_std"])
        initial = save_numeric(state_root / f"seed{seed}-initial.npy", bank)
        entries = []
        for lesson in lessons:
            stage = lesson["stage"]
            optimizer = torch.optim.Adam([bank], lr=config["learning_rate"])
            trajectory = []
            for step in range(config["steps"]):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for example in lesson["examples"]:
                    prompt = tuple(DialogueMessage(**m) for m in messages(example))
                    loss = organ.teacher_forced_loss(prompt, example["target"] + organ._tokenizer.eos_token, bank)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss / len(lesson["examples"])).backward()
                    losses.append(loss.detach().item())
                if bank.grad is None or not torch.isfinite(bank.grad).all() or bank.grad.norm().item() == 0:
                    raise ValueError("Invalid shared-state gradient")
                if any(p.requires_grad or p.grad is not None for p in organ._model.parameters()):
                    raise ValueError("Gradient reached frozen organ")
                norm = torch.nn.utils.clip_grad_norm_([bank], config["gradient_clip_norm"]).item()
                optimizer.step()
                if not torch.isfinite(bank).all():
                    raise ValueError("Nonfinite learned state")
                row = {"stage": stage, "seed": seed, "step": step + 1,
                       "mean_loss_before_update": sum(losses) / len(losses), "gradient_norm_before_clip": norm,
                       "bank_norm_after_update": bank.detach().norm().item(), "seconds": time.monotonic() - started}
                trajectory.append(row)
                append_row(args.output / "trajectory.jsonl", row)
                print(json.dumps(row), flush=True)
            entry = save_numeric(state_root / f"seed{seed}-stage{stage}.npy", bank)
            entries.append({"stage": stage, "state": entry})
            fit = []
            for example in lesson["examples"]:
                output = generate(organ, messages(example), bank.detach(), manifest["generation"]["max_new_tokens"])
                fit.append({**example, "generated": output, "correct": scorer.score_response(example["target"], output)})
            fits.append({"seed": seed, "stage": stage, "training_fit": fit,
                         "first_loss": trajectory[0]["mean_loss_before_update"], "last_loss": trajectory[-1]["mean_loss_before_update"]})
            organ.assert_frozen()
            del optimizer
        states.append({"seed": seed, "initial": initial, "stages": entries})
        del bank
    write_json(state_root / "state_manifest.json", {"manifest_sha256": manifest["manifest_sha256"], "states": states})
    return {"states": states, "fits": fits, "optimizer_steps": len(lessons) * len(config["seeds"]) * config["steps"],
            "heldout_examples_read": 0, "replay_examples_read": 0}


def state_metadata(args, manifest):
    metadata = json.loads((args.artifacts / "state_manifest.json").read_text())
    if metadata["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("State belongs to another instrument")
    return metadata["states"]


def restore(organ, scorer, args, manifest):
    stages = read_data(args.root, manifest, args.phase, "probes")["stages"]
    states = state_metadata(args, manifest)
    rows, cache = [], {}
    for state in states:
        seed = state["seed"]
        initial = load_numeric(args.artifacts, state["initial"], manifest)
        for stage in stages:
            index = stage["stage"]
            entry = next(s["state"] for s in state["stages"] if s["stage"] == index)
            learned = load_numeric(args.artifacts, entry, manifest)
            controls = [("initial", initial), ("learned_restored", learned)]
            if seed == manifest["training"]["seeds"][0]:
                controls.append(("zeroed", torch.zeros_like(learned)))
            for condition, bank in controls:
                for probe in stage["probes"]:
                    row = score_probe(organ, scorer, probe, bank, manifest, index, condition, seed=seed, cache=cache)
                    rows.append(row)
                    append_row(args.output / "rows.jsonl", row)
                print(f"restore stage={index} seed={seed} condition={condition}", flush=True)
    return {"rows": rows, "optimizer_steps": 0, "training_files_read": 0, "raw_text_artifacts_read": 0}


def actions(organ, scorer, args, manifest):
    cases = read_data(args.root, manifest, args.phase, "actions")["cases"]
    states = state_metadata(args, manifest)
    conditions = [("no_context", None, torch.zeros((1, 0, organ.feature_dim)), None)]
    for state in states:
        entry = max(state["stages"], key=lambda s: s["stage"])["state"]
        conditions.append(("learned_restored", state["seed"], load_numeric(args.artifacts, entry, manifest), None))
    raw = args.text_artifact.read_text()
    if sha(raw.encode()) != manifest["final_active_text_sha256"]:
        raise ValueError("Released retrieval text changed")
    conditions.append(("latest_example_retrieval", None, torch.zeros((1, 0, organ.feature_dim)), raw))
    rows = []
    for condition, seed, bank, context in conditions:
        def actor(prompt, bank=bank, context=context):
            if context is not None:
                prompt = [prompt[0], {"role": "user", "content": context}, *prompt[1:]]
            return generate(organ, prompt, bank, manifest["generation"]["action_max_new_tokens"])
        for case in cases:
            row = run_case(case, actor, max_calls=manifest["generation"]["action_max_calls"])
            row.update({"condition": condition, "seed": seed})
            rows.append(row)
            append_row(args.output / "rows.jsonl", row)
            print(f"action condition={condition} seed={seed} case={case['name']} correct={row['correct']}", flush=True)
    return {"rows": rows, "optimizer_steps": 0, "executor": "strict JSON, real temporary UTF-8 filesystem"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "train", "restore", "actions"))
    for field in ("root", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--text-artifact", type=Path)
    args = parser.parse_args()
    manifest = manifest_at(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    scorer = FrozenScorer()
    if scorer.sha256 != manifest["scorer_sha256"]:
        raise ValueError("Scorer mismatch")
    organ = QwenFrozenOrgan.load()
    try:
        before = organ.parameter_hash()
        if before != manifest["organ_hash"]:
            raise ValueError("Organ mismatch")
        result = {"reference": reference, "train": train, "restore": restore, "actions": actions}[args.phase](organ, scorer, args, manifest)
        organ.assert_frozen()
        after = organ.parameter_hash()
        if before != after:
            raise ValueError("Organ changed")
        result.update({"phase": args.phase, "pid": os.getpid(), "manifest_sha256": manifest["manifest_sha256"],
                       "organ_hash_before": before, "organ_hash_after": after, "scorer_sha256": scorer.sha256,
                       "meta_test_accessed": False, "classification": "BOUNDED_DEV_VALIDATION"})
        write_json(args.output / "results.json", result)
    finally:
        organ.close()


if __name__ == "__main__":
    main()
