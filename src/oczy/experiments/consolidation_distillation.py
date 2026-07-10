"""Research 18: consolidation as context distillation.

Pre-registered experiment: research/18-consolidation-as-distillation.md

For each correction, the prefix-conditioned frozen teacher is distilled into a
small LoRA adapter on the same frozen model. The prefix and all raw traces are
deleted before scoring; the persistent state is the LoRA adapter only.
"""
from __future__ import annotations

import argparse
import gc
import math
import pickle
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eval.v2 import verify_manifest
from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.organism_curriculum.dataset import (
    STAGE_ORDER,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm.cvec_driver import ReservedPosition
from oczy.lm.hf_driver import HFDriver


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------


class LoRAAdapter:
    """Rank-r LoRA adapter on q_proj and v_proj of every decoder layer.

    The adapter is toggled via ``enabled`` so the same model can serve as the
    base teacher (adapter off) and the LoRA student (adapter on).
    """

    def __init__(self, model: torch.nn.Module, rank: int, alpha: float = 1.0):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True
        self.A: dict[tuple[int, str], torch.nn.Parameter] = {}
        self.B: dict[tuple[int, str], torch.nn.Parameter] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._module_map: dict[tuple[int, str], torch.nn.Linear] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        target = self.model
        for attr in ("model.layers", "model.decoder.layers", "transformer.h"):
            parts = attr.split(".")
            obj = target
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if isinstance(obj, torch.nn.ModuleList):
                    layers = list(obj)
                    break
        else:
            raise RuntimeError("Cannot find decoder layers for LoRA")

        for layer_idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                module = getattr(attn, proj_name, None)
                if not isinstance(module, torch.nn.Linear):
                    continue
                key = (layer_idx, proj_name)
                a_init = torch.randn(module.in_features, self.rank) / math.sqrt(module.in_features)
                b_init = torch.zeros(self.rank, module.out_features)
                a = torch.nn.Parameter(a_init, requires_grad=True)
                b = torch.nn.Parameter(b_init, requires_grad=True)
                self.model.register_parameter(f"lora_A_{layer_idx}_{proj_name}", a)
                self.model.register_parameter(f"lora_B_{layer_idx}_{proj_name}", b)
                self.A[key] = a
                self.B[key] = b
                self._module_map[key] = module

                def hook(
                    module: torch.nn.Module,
                    args: tuple[torch.Tensor],
                    output: torch.Tensor,
                    A: torch.nn.Parameter = a,
                    B: torch.nn.Parameter = b,
                    adapter: LoRAAdapter = self,
                ) -> torch.Tensor:
                    if not adapter.enabled:
                        return output
                    x = args[0]
                    delta = (x @ A.to(x.device) @ B.to(x.device)) * adapter.scaling
                    return output + delta

                self.hooks.append(module.register_forward_hook(hook))

        if not self.A:
            raise RuntimeError("No LoRA target modules found")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.A.values()) + list(self.B.values())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {f"A_{k}": v.detach().cpu() for k, v in self.A.items()} | {
            f"B_{k}": v.detach().cpu() for k, v in self.B.items()
        }

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def calibrate(
        self,
        driver: HFDriver,
        tokenizer: Any,
        dev_episodes: list[Any],
        svd_target: float = 0.01,
    ) -> None:
        """Initialize A and B with the rank-r SVD of the distillation gradient.

        For each dev episode, compute the gradient of the base model's weights
        with respect to the KL loss against the prefix-conditioned teacher on
        the answer positions.  The accumulated gradient is a (out, in) matrix
        dW; we take the rank-r SVD of dW.T to produce A (in, rank) and B
        (rank, out) so that A @ B approximates dW.T.  B is then scaled so the
        LoRA effective matrix has Frobenius norm ``svd_target``.
        """
        self.set_enabled(False)

        # Enable gradient for target weights only.
        for key, module in self._module_map.items():
            module.weight.requires_grad = True
            if module.weight.grad is not None:
                module.weight.grad.zero_()

        n = 0
        for ep in dev_episodes:
            prefix = ep.correction_utterance
            request = ep.initial_request
            answer = ep.corrected_response
            answer_text = " " + answer
            prefix_count = _token_count(tokenizer, prefix)
            answer_count = len(tokenizer.encode(answer_text, add_special_tokens=False))
            if answer_count <= 0:
                continue
            for prompt in _distillation_prompts(request):
                prompt_count = _token_count(tokenizer, prompt)
                student_text = prompt + answer_text
                teacher_text = prefix + " " + student_text
                student_ids = tokenizer.encode(student_text, add_special_tokens=True)
                teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=True)

                t_ids = torch.tensor([teacher_ids], dtype=torch.long, device="cpu")
                s_ids = torch.tensor([student_ids], dtype=torch.long, device="cpu")

                with torch.no_grad():
                    t_out = driver._model(
                        input_ids=t_ids, use_cache=False, output_hidden_states=False
                    )
                    t_start = prefix_count + prompt_count - 1
                    t_end = t_start + answer_count
                    t_logits = t_out.logits[:, t_start:t_end, :]

                s_out = driver._model(
                    input_ids=s_ids, use_cache=False, output_hidden_states=False
                )
                s_start = prompt_count - 1
                s_end = s_start + answer_count
                s_logits = s_out.logits[:, s_start:s_end, :]

                loss = F.kl_div(
                    F.log_softmax(s_logits, dim=-1),
                    F.softmax(t_logits, dim=-1),
                    reduction="batchmean",
                    log_target=False,
                )
                loss.backward()
                n += 1

        with torch.no_grad():
            n = max(n, 1)
            for key, module in self._module_map.items():
                a = self.A[key]
                b = self.B[key]
                dW = module.weight.grad / n
                if dW is None:
                    continue
                try:
                    U, S, Vt = torch.linalg.svd(dW.T, full_matrices=False)
                except RuntimeError:
                    # Fallback to the existing random init.
                    continue
                U_r = U[:, : self.rank]
                Vt_r = Vt[: self.rank, :]
                S_r = S[: self.rank]
                a.copy_(U_r)
                b.copy_(S_r.unsqueeze(1) * Vt_r)
                # Scale the effective matrix to the target Frobenius norm.
                eff_norm = float((a @ b).norm().item())
                if eff_norm > 0:
                    scale = svd_target / eff_norm
                    b.mul_(scale)
                if module.weight.grad is not None:
                    module.weight.grad.zero_()
                module.weight.requires_grad = False

        self.set_enabled(True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _distillation_prompts(request: str) -> list[str]:
    """Generic templates for distillation; no eval/expected text allowed."""
    return [
        request,
        f"Q: {request}\nA:",
        f"Question: {request}\nAnswer:",
    ]


# ---------------------------------------------------------------------------
# Logit helpers
# ---------------------------------------------------------------------------


def _token_count(tokenizer: Any, text: str) -> int:
    """Return the number of content token ids in *text* (excluding BOS/EOS)."""
    ids = tokenizer.encode(text, add_special_tokens=True)
    n = len(ids)
    if n >= 1 and getattr(tokenizer, "bos_token_id", None) is not None and ids[0] == tokenizer.bos_token_id:
        n -= 1
    if n >= 1 and getattr(tokenizer, "eos_token_id", None) is not None and ids[-1] == tokenizer.eos_token_id:
        n -= 1
    return n


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


def _distill_correction(
    driver: HFDriver,
    adapter: LoRAAdapter,
    episode: Any,
    max_steps: int,
    lr: float,
    tokenizer: Any,
) -> dict[str, Any]:
    """Distill one per-fact prefix into the LoRA adapter.

    Teacher = base model with the correction prefix; student = base model plus
    LoRA.  Both are shown a prompt followed by the answer, and the student is
    trained with token-level KL on the answer positions.  At test time the
    student is given only the prompt, so the LoRA must encode the prefix.
    """
    prefix = episode.correction_utterance
    request = episode.initial_request
    answer = episode.corrected_response

    prompts = _distillation_prompts(request)
    # Prepend a space so the answer token merges with the prompt naturally.
    answer_text = " " + answer

    # Token counts for slicing the answer-position logits.
    prefix_count = _token_count(tokenizer, prefix)
    answer_count = len(tokenizer.encode(answer_text, add_special_tokens=False))
    if answer_count <= 0:
        return {"loss_mean": 0.0, "n_updates": 0}

    # Build full (teacher) and short (student) token sequences once.
    entries: list[tuple[int, list[int], list[int]]] = []
    for prompt in prompts:
        prompt_count = _token_count(tokenizer, prompt)
        student_text = prompt + answer_text
        teacher_text = prefix + " " + student_text
        student_ids = tokenizer.encode(student_text, add_special_tokens=True)
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=True)
        entries.append((prompt_count, student_ids, teacher_ids))

    # Teacher logits are computed once with the base model, adapter disabled.
    adapter.set_enabled(False)
    teacher_logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for prompt_count, _student_ids, teacher_ids in entries:
            t = torch.tensor([teacher_ids], dtype=torch.long, device="cpu")
            out = driver._model(input_ids=t, use_cache=False, output_hidden_states=False)
            # Slice logits for the answer positions.
            start = prefix_count + prompt_count - 1
            end = start + answer_count
            teacher_logits_list.append(out.logits[:, start:end, :])

    # Train student with adapter enabled.
    adapter.set_enabled(True)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)
    loss_sum = 0.0
    n_updates = 0

    for _ in range(max_steps):
        for p_idx, (prompt_count, student_ids, _teacher_ids) in enumerate(entries):
            start = prompt_count - 1
            end = start + answer_count
            if start < 0 or end <= start:
                continue

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
            optimizer.step()

            loss_sum += float(loss.item())
            n_updates += 1

    return {"loss_mean": loss_sum / max(n_updates, 1), "n_updates": n_updates}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_stage(
    driver: HFDriver,
    stage: Stage,
    probe_ids: set[str],
) -> tuple[float, int]:
    """Score a subset of probes and return (accuracy, total)."""
    results: list[bool] = []
    for ep in stage.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in probe_ids:
                continue
            answer = driver.generate(probe.request, max_tokens=32)
            results.append(probe_matches(answer, probe, ep))
    if not results:
        return 0.0, 0
    return sum(results) / len(results), len(results)


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def _run_one_seed(
    seed: int,
    stage: Stage,
    dev_ids: set[str],
    holdout_ids: set[str],
    other_stages: tuple[Stage, ...],
    max_steps: int,
    lora_rank: int,
    lora_lr: float,
    lora_alpha: float,
    calibrate: bool = False,
    svd_target: float = 0.01,
) -> dict[str, Any]:
    """Run research 18 for one random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    driver = HFDriver.load()
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

    # Validity gate: teacher with per-fact prefix on dev probes.
    teacher_dev_correct = 0
    teacher_dev_total = 0
    vanilla_dev_correct = 0
    for ep in dev_episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in dev_ids:
                continue
            ans_v = driver.generate(probe.request, max_tokens=32)
            vanilla_dev_correct += probe_matches(ans_v, probe, ep)
            driver.set_reserved_position(ReservedPosition(text=ep.correction_utterance))
            ans_t = driver.generate(probe.request, max_tokens=32)
            driver.clear_reserved_position()
            teacher_dev_correct += probe_matches(ans_t, probe, ep)
            teacher_dev_total += 1
    teacher_dev_delta = (teacher_dev_correct - vanilla_dev_correct) / max(teacher_dev_total, 1)

    # Distill each correction in seed order.
    for ep in teach_order:
        driver.clear_reserved_position()
        _distill_correction(driver, adapter, ep, max_steps, lora_lr, tokenizer)

    # Clear prefix and traces; persistent state is LoRA only.
    driver.clear_reserved_position()

    # Score holdout with LoRA.
    lora_holdout_acc, holdout_total = _score_stage(driver, stage, holdout_ids)

    # Score holdout with vanilla (no LoRA).
    adapter.set_enabled(False)
    vanilla_holdout_acc, _ = _score_stage(driver, stage, holdout_ids)
    adapter.set_enabled(True)

    # Specificity on other stages' holdout probes.
    other_total = 0
    other_lora_correct = 0
    other_vanilla_correct = 0
    for other in other_stages:
        _, other_holdout = split_probes(other, fraction=0.3, salt="v2")
        for ep in other.episodes:
            for probe in ep.probes:
                pid = f"{ep.id}|{probe.request}|{probe.category}"
                if pid not in other_holdout:
                    continue
                ans_l = driver.generate(probe.request, max_tokens=32)
                other_lora_correct += probe_matches(ans_l, probe, ep)
                adapter.set_enabled(False)
                ans_v = driver.generate(probe.request, max_tokens=32)
                adapter.set_enabled(True)
                other_vanilla_correct += probe_matches(ans_v, probe, ep)
                other_total += 1
    specificity_delta = (
        (other_lora_correct - other_vanilla_correct) / max(other_total, 1)
        if other_total
        else 0.0
    )

    # Persistent bytes: serialized LoRA adapter.
    state = adapter.state_dict()
    persistent_bytes = len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    adapter.remove()
    driver.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "teacher_dev_delta": teacher_dev_delta,
        "vanilla_holdout_acc": vanilla_holdout_acc,
        "lora_holdout_acc": lora_holdout_acc,
        "distill_delta_holdout": lora_holdout_acc - vanilla_holdout_acc,
        "specificity_delta": specificity_delta,
        "persistent_bytes": persistent_bytes,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower, upper) 95% CI using normal approximation."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    n = len(arr)
    se = std / math.sqrt(max(n, 1))
    lower = mean - 1.96 * se
    upper = mean + 1.96 * se
    return mean, lower, upper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research 18: consolidation as distillation")
    parser.add_argument("--seeds", type=int, default=1, help="number of random seeds")
    parser.add_argument("--max-steps", type=int, default=10, help="distillation steps per fact")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling")
    parser.add_argument("--lora-lr", type=float, default=0.005, help="LoRA learning rate")
    parser.add_argument("--calibrate", action="store_true", help="SVD-initialize LoRA from dev distillation gradient")
    parser.add_argument("--svd-target", type=float, default=0.01, help="target Frobenius norm for calibrated LoRA effective matrix")
    parser.add_argument("--stage", type=str, default="stage_0_grounding", help="target stage")
    args = parser.parse_args(argv)

    verify_manifest()

    stage = build_curriculum(stage_names=(args.stage,))[0]
    dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2")

    other_names = tuple(n for n in STAGE_ORDER if n != args.stage)
    other_stages = build_curriculum(stage_names=other_names)

    results: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"# seed {seed}", file=sys.stderr)
        t0 = time.monotonic()
        r = _run_one_seed(
            seed=seed,
            stage=stage,
            dev_ids=dev_ids,
            holdout_ids=holdout_ids,
            other_stages=other_stages,
            max_steps=args.max_steps,
            lora_rank=args.lora_rank,
            lora_lr=args.lora_lr,
            lora_alpha=args.lora_alpha,
            calibrate=args.calibrate,
            svd_target=args.svd_target,
        )
        r["wall_s"] = time.monotonic() - t0
        results.append(r)
        for k, v in r.items():
            print(f"ASI seed_{seed}_{k}={v}", file=sys.stderr)

    deltas = [r["distill_delta_holdout"] for r in results]
    specificities = [r["specificity_delta"] for r in results]
    d_mean, d_lower, d_upper = _mean_ci(deltas)
    s_mean, s_lower, s_upper = _mean_ci(specificities)

    print(f"METRIC distill_delta_holdout={d_mean}")
    print(f"METRIC distill_specificity_delta={s_mean}")
    print(f"ASI distill_delta_holdout_ci95=[{d_lower},{d_upper}]")
    print(f"ASI distill_specificity_delta_ci95=[{s_lower},{s_upper}]")
    print(f"ASI seeds={args.seeds}")
    print(f"ASI lora_rank={args.lora_rank}")
    print(f"ASI lora_alpha={args.lora_alpha}")
    print(f"ASI max_steps={args.max_steps}")
    print(f"ASI lora_lr={args.lora_lr}")
    print(f"ASI calibrate={args.calibrate}")
    print(f"ASI svd_target={args.svd_target}")
    print(f"ASI stage={args.stage}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
