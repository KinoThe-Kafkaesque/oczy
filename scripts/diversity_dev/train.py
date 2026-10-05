"""Train only on the frozen, rotating nine-example curriculum."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from oczy.experiments.meta_cortex.contracts import DialogueMessage
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan, render_chat
from scripts.diversity_dev.contract import data, manifest_at, messages, write_json
from scripts.serialization_pilot_worker import generate, save_numeric


def run(root, output):
    manifest = manifest_at(root)
    teaching = data(root, manifest, "train", "capacity_training")
    config = manifest["training"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    torch.set_num_threads(4)
    organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
    before = organ.parameter_hash()
    assert before == manifest["organ_hash"] and scorer.sha256 == manifest["scorer_sha256"]
    states, fits, token_lengths = [], [], []
    try:
        for group_index, group in enumerate(teaching["groups"]):
            for row in group:
                prompt = tuple(DialogueMessage(**m) for m in messages(row))
                token_lengths.append({"group": group_index, "input": row["input"], "client": row["category"],
                                      "prompt_tokens": len(organ._tokenizer.encode(render_chat(prompt, organ._tokenizer), add_special_tokens=True)),
                                      "target_tokens": len(organ._tokenizer.encode(" " + row["target"] + organ._tokenizer.eos_token, add_special_tokens=False))})
        for seed in config["seeds"]:
            bank = torch.nn.Parameter(torch.randn((1, 8, 896), generator=torch.Generator().manual_seed(seed)) * config["initialization_std"])
            initial = save_numeric(output / "states" / f"seed{seed}-initial.npy", bank)
            expected = next(s for s in manifest["controls"] if s["seed"] == seed)
            assert initial["sha256"] == expected["initial"]["sha256"]
            optimizer = torch.optim.Adam([bank], lr=config["learning_rate"])
            history = []
            for step in range(config["capacity_steps"]):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                group_index = step % 3
                losses = []
                for row in teaching["groups"][group_index]:
                    prompt = tuple(DialogueMessage(**m) for m in messages(row))
                    loss = organ.teacher_forced_loss(prompt, row["target"] + organ._tokenizer.eos_token, bank)
                    assert torch.isfinite(loss)
                    (loss / 9).backward()
                    losses.append(loss.detach().item())
                mean_ce = sum(losses) / 9
                if step == 0:
                    assert math.isclose(mean_ce, manifest["parent_first_ce"][str(seed)], rel_tol=1e-6, abs_tol=1e-6)
                assert bank.grad is not None and torch.isfinite(bank.grad).all() and bank.grad.norm().item() > 0
                assert all(not p.requires_grad and p.grad is None for p in organ._model.parameters())
                norm = torch.nn.utils.clip_grad_norm_([bank], config["gradient_clip_norm"]).item()
                optimizer.step()
                assert torch.isfinite(bank).all()
                record = {"seed": seed, "step": step + 1, "group": group_index, "mean_ce_before_update": mean_ce,
                          "gradient_norm_before_clip": norm, "bank_norm_after_update": bank.detach().norm().item(), "seconds": time.monotonic() - started}
                history.append(record)
                with (output / "trajectory.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            trained = save_numeric(output / "states" / f"seed{seed}-diversity.npy", bank)
            fit = []
            for row in teaching["rows"]:
                result = generate(organ, messages(row), bank.detach(), 32)
                fit.append({**row, "generated": result, "correct": scorer.score_response(row["target"], result)})
            fits.append({"seed": seed, "training_fit": fit, "first_ce": history[0]["mean_ce_before_update"], "last_ce": history[-1]["mean_ce_before_update"]})
            states.append({"seed": seed, "initial": initial, "trained": trained})
        after = organ.parameter_hash()
        organ.assert_frozen()
        assert after == before
        write_json(output / "states/state_manifest.json", {"manifest_sha256": manifest["manifest_sha256"], "states": states})
        write_json(output / "results.json", {"manifest_sha256": manifest["manifest_sha256"], "states": states, "fits": fits,
                                            "organ_hash_before": before, "organ_hash_after": after, "scorer_sha256": scorer.sha256,
                                            "optimizer_steps": 144, "token_lengths": token_lengths, "heldout_examples_read": 0, "meta_test_accessed": False})
    finally:
        organ.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
