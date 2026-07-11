"""Research 18 diagnostic: training trajectory recorder.

Mechanism-level diagnostic that records per-step training loss and DEV-only
student/teacher/vanilla accuracy at fixed checkpoints (step 0 through
``max_steps``) for seeds 0–4.  This explains the failed teacher prerequisite
and seed-2 LoRA null by showing underfit / saturation / instability and
seed divergence — without scoring holdout and without issuing an H-DISTILL
verdict.

Frozen instrument reuse
-----------------------
Imports ``LoRAAdapter``, ``_distillation_prompts``, ``_token_count``,
and ``_score_stage`` from ``consolidation_distillation``.
Does NOT edit that module.  Uses the same rank=8, alpha=16, lr=0.005,
templates, and teacher (reserved-position prefix) mode.

Holdout firewall
----------------
Holdout scoring is explicitly prohibited.  Only DEV probes are scored at
each checkpoint.  No holdout ids are ever passed to ``_score_stage``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import sys
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.consolidation_distillation import (
    LoRAAdapter,
    _distillation_prompts,
    _score_stage,
    _token_count,
)
from oczy.experiments.organism_curriculum.dataset import (
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm._types import ReservedPosition
from oczy.lm.hf_driver import HFDriver

# ---------------------------------------------------------------------------
# DEV scoring helpers (holdout-free)
# ---------------------------------------------------------------------------


def _score_dev_student(
    driver: HFDriver,
    stage: Stage,
    dev_ids: set[str],
) -> tuple[float, int]:
    """Score DEV probes with the LoRA adapter enabled (student)."""
    return _score_stage(driver, stage, dev_ids)


def _score_dev_vanilla(
    driver: HFDriver,
    adapter: LoRAAdapter,
    stage: Stage,
    dev_ids: set[str],
) -> tuple[float, int]:
    """Score DEV probes with the adapter disabled (vanilla baseline)."""
    adapter.set_enabled(False)
    try:
        return _score_stage(driver, stage, dev_ids)
    finally:
        adapter.set_enabled(True)


def _score_dev_teacher(
    driver: HFDriver,
    adapter: LoRAAdapter,
    stage: Stage,
    dev_ids: set[str],
    dev_episodes: list[Any],
) -> tuple[float, int]:
    """Score DEV probes with adapter disabled and per-episode correction prefix.

    This replicates the teacher-accuracy path from ``_run_one_seed``'s validity
    gate: for each dev episode, set the reserved position to the correction
    utterance, generate, then clear it.

    *stage* is used to validate that every dev episode belongs to the stage,
    guarding against accidental cross-stage scoring without affecting results.
    """
    # Validate dev_episodes belong to the stage (defensive; does not affect scoring).
    stage_episode_ids = {ep.id for ep in stage.episodes}
    for ep in dev_episodes:
        if ep.id not in stage_episode_ids:
            raise ValueError(f"dev episode {ep.id} not in stage {stage.name}")
    adapter.set_enabled(False)
    driver.clear_reserved_position()
    correct = 0
    total = 0
    try:
        for ep in dev_episodes:
            for probe in ep.probes:
                pid = f"{ep.id}|{probe.request}|{probe.category}"
                if pid not in dev_ids:
                    continue
                driver.set_reserved_position(
                    cast(Any, ReservedPosition(text=ep.correction_utterance))
                )
                ans = driver.generate(probe.request, max_tokens=32)
                driver.clear_reserved_position()
                correct += probe_matches(ans, probe, ep)
                total += 1
        return (correct / max(total, 1), total) if total > 0 else (0.0, 0)
    finally:
        adapter.set_enabled(True)
        driver.clear_reserved_position()


# ---------------------------------------------------------------------------
# Instrumented distillation (one outer step across all episodes)
# ---------------------------------------------------------------------------


def _precompute_episode(
    driver: HFDriver,
    adapter: LoRAAdapter,
    tokenizer: Any,
    ep: Any,
) -> dict[str, Any] | None:
    """Precompute teacher logits and student entries for one episode.

    Returns None if the episode has a zero-length answer.
    """
    prefix = ep.correction_utterance
    request = ep.initial_request
    answer = ep.corrected_response
    answer_text = " " + answer
    prompts = _distillation_prompts(request)
    prefix_count = _token_count(tokenizer, prefix)
    answer_count = len(tokenizer.encode(answer_text, add_special_tokens=False))
    if answer_count <= 0:
        return None

    entries: list[tuple[int, list[int]]] = []
    teacher_logits_list: list[torch.Tensor] = []

    adapter.set_enabled(False)
    with torch.no_grad():
        for prompt in prompts:
            prompt_count = _token_count(tokenizer, prompt)
            student_text = prompt + answer_text
            teacher_text = prefix + " " + student_text
            student_ids = tokenizer.encode(student_text, add_special_tokens=True)
            teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=True)
            entries.append((prompt_count, student_ids))

            t = torch.tensor([teacher_ids], dtype=torch.long, device="cpu")
            out = driver._model(input_ids=t, use_cache=False, output_hidden_states=False)
            start = prefix_count + prompt_count - 1
            end = start + answer_count
            teacher_logits_list.append(out.logits[:, start:end, :])

    return {
        "episode": ep,
        "entries": entries,
        "teacher_logits": teacher_logits_list,
        "answer_count": answer_count,
    }


def _instrumented_step(
    driver: HFDriver,
    adapter: LoRAAdapter,
    optimizer: torch.optim.Optimizer,
    episode_data: list[dict[str, Any]],
) -> dict[str, float]:
    """Run one distillation step across all episodes with read-only instrumentation.

    Returns mean loss, mean grad norm, and mean update norm across all
    gradient updates in this step.  Does NOT modify the training algorithm —
    only observes gradients and parameter deltas.
    """
    adapter.set_enabled(True)
    losses: list[float] = []
    grad_norms: list[float] = []
    update_norms: list[float] = []

    for ed in episode_data:
        entries = ed["entries"]
        teacher_logits_list = ed["teacher_logits"]
        answer_count = ed["answer_count"]

        for p_idx, (prompt_count, student_ids) in enumerate(entries):
            start = prompt_count - 1
            end = start + answer_count
            if start < 0 or end <= start:
                continue

            # Snapshot params before update for update-norm measurement.
            params_before = [p.detach().clone() for p in adapter.parameters()]

            s_ids = torch.tensor([student_ids], dtype=torch.long, device="cpu")
            out = driver._model(input_ids=s_ids, use_cache=False, output_hidden_states=False)
            s_logits = out.logits[:, start:end, :].reshape(-1, out.logits.size(-1))

            t_logits = teacher_logits_list[p_idx]
            t_logits = t_logits.reshape(-1, t_logits.size(-1))

            loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
                log_target=False,
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient norm (read-only, after backward, before step).
            grad_norm_sq = 0.0
            for p in adapter.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(p.grad.norm().item()) ** 2
            grad_norm = math.sqrt(grad_norm_sq)

            optimizer.step()

            # Update norm = ||theta_after - theta_before||.
            update_norm_sq = 0.0
            for i, p in enumerate(adapter.parameters()):
                delta = p.detach() - params_before[i]
                update_norm_sq += float(delta.norm().item()) ** 2
            update_norm = math.sqrt(update_norm_sq)

            losses.append(float(loss.item()))
            grad_norms.append(grad_norm)
            update_norms.append(update_norm)

    n = max(len(losses), 1)
    return {
        "loss": sum(losses) / n,
        "grad_norm": sum(grad_norms) / n,
        "update_norm": sum(update_norms) / n,
    }


# ---------------------------------------------------------------------------
# One seed trajectory
# ---------------------------------------------------------------------------


def _run_one_seed_trajectory(
    driver: HFDriver,
    seed: int,
    stage: Stage,
    dev_ids: set[str],
    max_steps: int,
    lora_rank: int,
    lora_lr: float,
    lora_alpha: float,
    calibrate: bool = False,
    svd_target: float = 0.01,
) -> dict[str, Any]:
    """Run training trajectory recording for one seed.

    Returns a dict with per-step checkpoints (step 0..max_steps), each
    containing train_loss, dev_student_acc, dev_teacher_acc, dev_vanilla_acc,
    grad_norm, update_norm.  Also records final_param_bytes.

    Holdout is never scored.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = driver._tokenizer
    model = driver._model
    model.requires_grad_(False)
    model.eval()

    adapter = LoRAAdapter(model, rank=lora_rank, alpha=lora_alpha)

    dev_episode_ids = {pid.split("|")[0] for pid in dev_ids}
    dev_episodes = [ep for ep in stage.episodes if ep.id in dev_episode_ids]

    if calibrate:
        adapter.calibrate(driver, tokenizer, dev_episodes, svd_target=svd_target)

    rng = random.Random(seed)
    teach_order = list(stage.episodes)
    rng.shuffle(teach_order)

    # Precompute teacher logits and student entries for all episodes.
    episode_data: list[dict[str, Any]] = []
    for ep in teach_order:
        ed = _precompute_episode(driver, adapter, tokenizer, ep)
        if ed is not None:
            episode_data.append(ed)

    adapter.set_enabled(True)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=lora_lr)

    checkpoints: list[dict[str, Any]] = []

    # --- Step 0: baseline (no training) ---
    student_acc, _ = _score_dev_student(driver, stage, dev_ids)
    teacher_acc, _ = _score_dev_teacher(driver, adapter, stage, dev_ids, dev_episodes)
    vanilla_acc, _ = _score_dev_vanilla(driver, adapter, stage, dev_ids)
    checkpoints.append({
        "step": 0,
        "train_loss": 0.0,
        "dev_student_acc": student_acc,
        "dev_teacher_acc": teacher_acc,
        "dev_vanilla_acc": vanilla_acc,
        "grad_norm": 0.0,
        "update_norm": 0.0,
    })
    print(f"ASI traj_seed{seed}_step0_dev_student_acc={student_acc}", file=sys.stderr)
    print(f"ASI traj_seed{seed}_step0_dev_teacher_acc={teacher_acc}", file=sys.stderr)
    print(f"ASI traj_seed{seed}_step0_dev_vanilla_acc={vanilla_acc}", file=sys.stderr)

    # --- Steps 1..max_steps ---
    for step in range(1, max_steps + 1):
        step_stats = _instrumented_step(driver, adapter, optimizer, episode_data)

        student_acc, _ = _score_dev_student(driver, stage, dev_ids)
        teacher_acc, _ = _score_dev_teacher(driver, adapter, stage, dev_ids, dev_episodes)
        vanilla_acc, _ = _score_dev_vanilla(driver, adapter, stage, dev_ids)

        ckpt = {
            "step": step,
            "train_loss": step_stats["loss"],
            "dev_student_acc": student_acc,
            "dev_teacher_acc": teacher_acc,
            "dev_vanilla_acc": vanilla_acc,
            "grad_norm": step_stats["grad_norm"],
            "update_norm": step_stats["update_norm"],
        }
        checkpoints.append(ckpt)

        # Per-step sentinel lines.
        print(f"METRIC traj_seed{seed}_step{step}_train_loss={step_stats['loss']}", file=sys.stderr)
        print(f"METRIC traj_seed{seed}_step{step}_dev_student_acc={student_acc}", file=sys.stderr)
        print(f"METRIC traj_seed{seed}_step{step}_dev_teacher_acc={teacher_acc}", file=sys.stderr)
        print(f"METRIC traj_seed{seed}_step{step}_dev_vanilla_acc={vanilla_acc}", file=sys.stderr)
        print(f"ASI traj_seed{seed}_step{step}_grad_norm={step_stats['grad_norm']}", file=sys.stderr)
        print(f"ASI traj_seed{seed}_step{step}_update_norm={step_stats['update_norm']}", file=sys.stderr)

    # Final parameter bytes.
    state = adapter.state_dict()
    final_param_bytes = len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"METRIC traj_seed{seed}_final_param_bytes={final_param_bytes}", file=sys.stderr)

    adapter.remove()
    driver.clear_reserved_position()
    del state, adapter
    gc.collect()

    return {
        "seed": seed,
        "checkpoints": checkpoints,
        "final_param_bytes": final_param_bytes,
    }


# ---------------------------------------------------------------------------
# Aggregate diagnostics
# ---------------------------------------------------------------------------


def _aggregate_diagnostics(
    trajectories: list[dict[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    """Compute aggregate trajectory diagnostics across seeds.

    Flags:
    - underfit: mean loss slope is still strongly negative at the end
    - saturation: mean loss slope is near-zero in the second half
    - instability: any seed has loss increase between consecutive steps,
      or grad_norm exceeds 10x the median, or dev accuracy swings >0.3
    - seed_divergence: max pairwise difference in final-step train_loss

    *max_steps* validates that no checkpoint step exceeds the configured
    ceiling, guarding against truncated or runaway trajectories without
    affecting the computed metrics.
    """
    n_seeds = len(trajectories)
    if n_seeds == 0:
        return {}

    # Validate checkpoint steps do not exceed max_steps (defensive).
    for traj in trajectories:
        for c in traj["checkpoints"]:
            if c["step"] > max_steps:
                raise ValueError(f"checkpoint step {c['step']} exceeds max_steps={max_steps}")

    # Collect per-seed final-step loss and loss sequences.
    final_losses: list[float] = []
    loss_sequences: list[list[float]] = []
    grad_norms_all: list[float] = []

    for traj in trajectories:
        ckpts = traj["checkpoints"]
        losses = [c["train_loss"] for c in ckpts if c["step"] > 0]
        if losses:
            loss_sequences.append(losses)
            final_losses.append(losses[-1])
        for c in ckpts:
            if c["step"] > 0 and c["grad_norm"] > 0:
                grad_norms_all.append(c["grad_norm"])

    # Loss slope: (last - first) / n_steps, averaged across seeds.
    slopes: list[float] = []
    for losses in loss_sequences:
        if len(losses) >= 2:
            slopes.append((losses[-1] - losses[0]) / max(len(losses) - 1, 1))
    loss_slope_mean = sum(slopes) / max(len(slopes), 1)

    # Second-half slope for saturation detection.
    second_half_slopes: list[float] = []
    for losses in loss_sequences:
        half = len(losses) // 2
        if len(losses) - half >= 2:
            second = losses[half:]
            second_half_slopes.append((second[-1] - second[0]) / max(len(second) - 1, 1))
    second_half_slope_mean = sum(second_half_slopes) / max(len(second_half_slopes), 1)

    # Seed divergence: max pairwise |final_loss_i - final_loss_j|.
    seed_divergence_max = 0.0
    for i in range(len(final_losses)):
        for j in range(i + 1, len(final_losses)):
            d = abs(final_losses[i] - final_losses[j])
            seed_divergence_max = max(seed_divergence_max, d)

    # Underfit: loss slope is still strongly negative (>0.01 abs decrease per step).
    underfit_flag = 1 if loss_slope_mean < -0.01 else 0

    # Saturation: second-half slope is near-zero (|slope| < 0.005).
    saturation_flag = 1 if abs(second_half_slope_mean) < 0.005 else 0

    # Instability: any consecutive loss increase, or grad_norm explosion, or
    # dev accuracy swing > 0.3 between consecutive checkpoints.
    instability_flag = 0
    for traj in trajectories:
        ckpts = traj["checkpoints"]
        for i in range(2, len(ckpts)):
            if ckpts[i]["train_loss"] > ckpts[i - 1]["train_loss"] + 1e-6:
                instability_flag = 1
                break
            acc_delta = abs(ckpts[i]["dev_student_acc"] - ckpts[i - 1]["dev_student_acc"])
            if acc_delta > 0.3:
                instability_flag = 1
                break
        if instability_flag:
            break

    if grad_norms_all:
        median_grad = sorted(grad_norms_all)[len(grad_norms_all) // 2]
        if median_grad > 0:
            for g in grad_norms_all:
                if g > 10 * median_grad:
                    instability_flag = 1
                    break

    return {
        "loss_slope_mean": loss_slope_mean,
        "second_half_slope_mean": second_half_slope_mean,
        "seed_divergence_max": seed_divergence_max,
        "underfit_flag": underfit_flag,
        "saturation_flag": saturation_flag,
        "instability_flag": instability_flag,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 diagnostic: training trajectory recorder (DEV-only, no holdout, no H-DISTILL verdict)",
    )
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    parser.add_argument("--max-steps", type=int, default=10, help="distillation steps per episode (checkpoints at 0..max_steps)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling")
    parser.add_argument("--lora-lr", type=float, default=0.005, help="LoRA learning rate")
    parser.add_argument("--calibrate", action="store_true", help="SVD-initialize LoRA from dev distillation gradient")
    parser.add_argument("--svd-target", type=float, default=0.01, help="target Frobenius norm for calibrated LoRA")
    parser.add_argument("--stage", type=str, default="stage_0_grounding", help="target stage")
    args = parser.parse_args(argv)

    # build_curriculum verifies eval manifest integrity internally.
    stage = build_curriculum(stage_names=(args.stage,))[0]
    dev_ids, _holdout_ids = split_probes(stage, fraction=0.3, salt="v2")
    # Holdout ids are intentionally discarded — never used in this diagnostic.

    # Load the real HFDriver once and reuse across seeds (avoids exit-137 under
    # memory pressure, per documented R18 lesson).
    try:
        driver = HFDriver.load()
    except Exception as exc:
        print(f"ASI traj_driver_error={type(exc).__name__}: {exc}", file=sys.stderr)
        print("ASI traj_driver_loaded=0", file=sys.stderr)
        return 1

    model_id = getattr(driver, "model_id", "unknown")
    print("ASI traj_driver_loaded=1", file=sys.stderr)

    trajectories: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"# traj seed {seed}", file=sys.stderr)
        t0 = time.monotonic()
        try:
            traj = _run_one_seed_trajectory(
                driver=driver,
                seed=seed,
                stage=stage,
                dev_ids=dev_ids,
                max_steps=args.max_steps,
                lora_rank=args.lora_rank,
                lora_lr=args.lora_lr,
                lora_alpha=args.lora_alpha,
                calibrate=args.calibrate,
                svd_target=args.svd_target,
            )
        except Exception as exc:
            print(f"ASI traj_seed{seed}_error={type(exc).__name__}: {exc}", file=sys.stderr)
            gc.collect()
            continue

        traj["wall_s"] = time.monotonic() - t0
        trajectories.append(traj)
        print(f"ASI traj_seed{seed}_wall_s={traj['wall_s']:.1f}", file=sys.stderr)
        gc.collect()

    driver.close()
    gc.collect()

    if not trajectories:
        print("ASI traj_no_completed_seeds=1", file=sys.stderr)
        return 1

    # Aggregate diagnostics.
    agg = _aggregate_diagnostics(trajectories, args.max_steps)

    # Emit aggregate METRIC / ASI sentinel lines to stdout (machine-readable).
    print(f"METRIC traj_loss_slope_mean={agg['loss_slope_mean']}")
    print(f"METRIC traj_second_half_slope_mean={agg['second_half_slope_mean']}")
    print(f"METRIC traj_seed_divergence_max={agg['seed_divergence_max']}")
    print(f"ASI traj_underfit_flag={agg['underfit_flag']}")
    print(f"ASI traj_saturation_flag={agg['saturation_flag']}")
    print(f"ASI traj_instability_flag={agg['instability_flag']}")
    print(f"ASI traj_seeds={agg['n_seeds']}")
    print(f"ASI traj_model={model_id}")
    print(f"ASI traj_lora_rank={args.lora_rank}")
    print(f"ASI traj_lora_alpha={args.lora_alpha}")
    print(f"ASI traj_lora_lr={args.lora_lr}")
    print(f"ASI traj_max_steps={args.max_steps}")
    print(f"ASI traj_stage={args.stage}")
    print(f"ASI traj_calibrate={args.calibrate}")
    print(f"ASI traj_svd_target={args.svd_target}")

    # Per-seed final-step summary for easy cross-seed comparison.
    for traj in trajectories:
        final_ckpt = traj["checkpoints"][-1]
        print(f"METRIC traj_seed{traj['seed']}_final_train_loss={final_ckpt['train_loss']}")
        print(f"METRIC traj_seed{traj['seed']}_final_dev_student_acc={final_ckpt['dev_student_acc']}")
        print(f"METRIC traj_seed{traj['seed']}_final_dev_teacher_acc={final_ckpt['dev_teacher_acc']}")
        print(f"METRIC traj_seed{traj['seed']}_final_dev_vanilla_acc={final_ckpt['dev_vanilla_acc']}")
        print(f"ASI traj_seed{traj['seed']}_final_param_bytes={traj['final_param_bytes']}")

    # Serialize full per-seed trajectory to stderr as JSON for downstream parsing.
    print("ASI traj_json=" + json.dumps(trajectories, default=str), file=sys.stderr)

    # No H-DISTILL verdict is emitted by design.
    return 0


if __name__ == "__main__":
    sys.exit(main())
