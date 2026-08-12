"""Kaggle CPU smoke workload for offline cortex development.

This is an infrastructure test, not a Research/20 experiment.  It exercises
the expected compute path without using eval/v2, meta_cortex/v1, episode IDs,
retrieval, or a real language model:

    learned writer -> fast state -> consolidation -> latent coupler
        -> frozen differentiable organ

The source runs on a CPU kernel.  Its JSON artifact proves whether gradients
remained finite, whether the held-out synthetic loss improved, and whether the
frozen organ stayed exactly unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SCHEMA_VERSION = "oczy/kaggle-cortex-smoke/v3"
CLAIM_BOUNDARY = (
    "Infrastructure smoke only. This is not meta_cortex/v1 and cannot support "
    "a cortex capability claim."
)


@dataclass(frozen=True)
class SmokeConfig:
    d_cortex: int = 64
    # Match Qwen2.5-0.5B-Instruct's hidden width without loading model weights.
    d_organ: int = 896
    soft_tokens: int = 4
    organ_outputs: int = 32
    batch_size: int = 64
    eval_batch_size: int = 256
    steps: int = 40
    learning_rate: float = 3e-3
    seed: int = 20260709
    minimum_improvement: float = 0.02


class Writer(nn.Module):
    """Learn a content-addressed outer-product update from one teaching event."""

    def __init__(self, d_cortex: int) -> None:
        super().__init__()
        self.d_cortex = d_cortex
        self.network = nn.Sequential(
            nn.Linear(2 * d_cortex, 2 * d_cortex),
            nn.GELU(),
            nn.Linear(2 * d_cortex, 2 * d_cortex + 2),
        )

    def forward(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        update = self.network(torch.cat((key, value), dim=-1))
        written_key, written_value, eta_logit, decay_logit = torch.split(
            update,
            (self.d_cortex, self.d_cortex, 1, 1),
            dim=-1,
        )
        written_key = F.normalize(written_key, dim=-1)
        written_value = F.normalize(written_value, dim=-1)
        eta = 2.0 * torch.sigmoid(eta_logit)
        decay = torch.sigmoid(decay_logit)
        return written_key, written_value, eta, decay


class FrozenOrgan(nn.Module):
    """Small differentiable stand-in for the frozen language organ."""

    def __init__(self, d_organ: int, outputs: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_organ)
        self.token_mixer = nn.Sequential(
            nn.Linear(d_organ, 2 * d_organ),
            nn.SiLU(),
            nn.Linear(2 * d_organ, d_organ),
        )
        self.head = nn.Linear(d_organ, outputs, bias=False)

    def forward(self, soft_bank: Tensor) -> Tensor:
        mixed = soft_bank + self.token_mixer(self.norm(soft_bank))
        return self.head(mixed.mean(dim=1))


class CortexPath(nn.Module):
    """Minimal learned write/consolidate/read/articulate path for hardware QA."""

    def __init__(self, config: SmokeConfig) -> None:
        super().__init__()
        self.config = config
        self.writer = Writer(config.d_cortex)
        self.consolidation_gate = nn.Sequential(
            nn.Linear(2 * config.d_cortex, config.d_cortex),
            nn.GELU(),
            nn.Linear(config.d_cortex, 1),
            nn.Sigmoid(),
        )
        self.coupler = nn.Linear(
            config.d_cortex,
            config.soft_tokens * config.d_organ,
        )

    def forward(self, key: Tensor, value: Tensor, query: Tensor, organ: FrozenOrgan) -> Tensor:
        written_key, written_value, eta, decay = self.writer(key, value)

        # The empty fast state is explicit so both update terms are exercised.
        batch = key.shape[0]
        fast = torch.zeros(
            batch,
            self.config.d_cortex,
            self.config.d_cortex,
            device=key.device,
            dtype=key.dtype,
        )
        fast = decay[:, :, None] * fast
        fast = fast + eta[:, :, None] * torch.bmm(
            written_value.unsqueeze(2),
            written_key.unsqueeze(1),
        )

        gate = self.consolidation_gate(torch.cat((written_key, written_value), dim=-1))
        slow = gate[:, :, None] * fast
        read = torch.bmm(slow, query.unsqueeze(2)).squeeze(2)
        soft_bank = self.coupler(read).view(
            batch,
            self.config.soft_tokens,
            self.config.d_organ,
        )
        return organ(soft_bank)


class OfflineCortexSystem(nn.Module):
    """Keep the trainable cortex and frozen organ together."""

    def __init__(self, config: SmokeConfig) -> None:
        super().__init__()
        self.cortex = CortexPath(config)
        self.organ = FrozenOrgan(config.d_organ, config.organ_outputs)

    def forward(self, key: Tensor, value: Tensor, query: Tensor) -> Tensor:
        return self.cortex(key, value, query, self.organ)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _state_hash(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            materialized = tensor.detach().cpu().contiguous()
            digest.update(f"{module_index}:{name}:{materialized.dtype}:".encode())
            digest.update(json.dumps(list(materialized.shape)).encode())
            digest.update(materialized.numpy().tobytes())
    return digest.hexdigest()


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested != "cpu":
        raise RuntimeError("only CPU is supported; GPU/CUDA selection has been removed")
    return torch.device("cpu")


def _make_batch(
    batch_size: int, d_cortex: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    key = F.normalize(torch.randn(batch_size, d_cortex, device=device), dim=-1)
    value = F.normalize(torch.randn(batch_size, d_cortex, device=device), dim=-1)
    query = F.normalize(key + 0.05 * torch.randn_like(key), dim=-1)
    return key, value, query


def _target_logits(
    value: Tensor,
    teacher: nn.Linear,
    organ: FrozenOrgan,
    config: SmokeConfig,
) -> Tensor:
    with torch.no_grad():
        soft_bank = teacher(value).view(
            value.shape[0],
            config.soft_tokens,
            config.d_organ,
        )
        return organ(soft_bank)


def _evaluate(
    system: nn.Module,
    base_system: OfflineCortexSystem,
    organ: FrozenOrgan,
    teacher: nn.Linear,
    batch: tuple[Tensor, Tensor, Tensor],
) -> float:
    system.eval()
    with torch.no_grad():
        key, value, query = batch
        prediction = system(key, value, query)
        target = _target_logits(value, teacher, organ, base_system.cortex.config)
        loss = F.mse_loss(prediction, target)
    system.train()
    return float(loss.item())


def _device_report(device: torch.device) -> dict[str, Any]:
    return {
        "selected": str(device),
        "cuda_available": False,
        "cuda_device_count": 0,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": None,
        "name": platform.processor() or platform.machine(),
    }


def run(config: SmokeConfig, requested_device: str) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = _resolve_device(requested_device)
    base_system = OfflineCortexSystem(config).to(device)
    organ = _freeze(base_system.organ)
    system: nn.Module = base_system
    parallel_mode = "single-device"
    devices_used = [str(device)]
    teacher = _freeze(
        nn.Linear(
            config.d_cortex,
            config.soft_tokens * config.d_organ,
            bias=False,
        ).to(device)
    )
    frozen_hash_before = _state_hash(organ, teacher)

    optimizer = torch.optim.AdamW(
        base_system.cortex.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    eval_batch = _make_batch(config.eval_batch_size, config.d_cortex, device)
    initial_eval_loss = _evaluate(system, base_system, organ, teacher, eval_batch)

    training_losses: list[float] = []
    gradients_finite = True
    started = time.perf_counter()
    for _ in range(config.steps):
        key, value, query = _make_batch(config.batch_size, config.d_cortex, device)
        target = _target_logits(value, teacher, organ, config)
        prediction = system(key, value, query)
        loss = F.mse_loss(prediction, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in base_system.cortex.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                gradients_finite = False
                break
        optimizer.step()
        training_losses.append(float(loss.detach().item()))
    elapsed_seconds = time.perf_counter() - started

    final_eval_loss = _evaluate(system, base_system, organ, teacher, eval_batch)
    frozen_hash_after = _state_hash(organ, teacher)
    improvement_fraction = (initial_eval_loss - final_eval_loss) / max(initial_eval_loss, 1e-12)
    losses_finite = all(math.isfinite(loss) for loss in training_losses)
    passed = (
        gradients_finite
        and losses_finite
        and math.isfinite(final_eval_loss)
        and frozen_hash_before == frozen_hash_after
        and improvement_fraction >= config.minimum_improvement
    )

    total_episodes = config.batch_size * config.steps
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "passed": passed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_sha256": _source_hash(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "device": _device_report(device),
        "config": asdict(config),
        "architecture": {
            "fast_state_shape": [config.d_cortex, config.d_cortex],
            "slow_state_shape": [config.d_cortex, config.d_cortex],
            "latent_bank_shape": [config.soft_tokens, config.d_organ],
            "parallel_mode": parallel_mode,
            "devices_used": devices_used,
            "trainable_parameters": sum(
                parameter.numel() for parameter in base_system.cortex.parameters()
            ),
            "frozen_parameters": sum(parameter.numel() for parameter in organ.parameters())
            + sum(parameter.numel() for parameter in teacher.parameters()),
        },
        "checks": {
            "gradients_finite": gradients_finite,
            "losses_finite": losses_finite,
            "frozen_organ_unchanged": frozen_hash_before == frozen_hash_after,
            "frozen_hash_before": frozen_hash_before,
            "frozen_hash_after": frozen_hash_after,
            "minimum_improvement_met": improvement_fraction >= config.minimum_improvement,
        },
        "measurements": {
            "initial_eval_loss": initial_eval_loss,
            "final_eval_loss": final_eval_loss,
            "improvement_fraction": improvement_fraction,
            "first_10_train_loss_mean": sum(training_losses[:10]) / min(10, len(training_losses)),
            "last_10_train_loss_mean": sum(training_losses[-10:]) / min(10, len(training_losses)),
            "elapsed_seconds": elapsed_seconds,
            "episodes": total_episodes,
            "episodes_per_second": total_episodes / max(elapsed_seconds, 1e-12),
        },
    }
    return report


def _default_output() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "cortex_smoke_report.json"
    return Path("cortex_smoke_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=SmokeConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=SmokeConfig.eval_batch_size)
    parser.add_argument("--steps", type=int, default=SmokeConfig.steps)
    parser.add_argument("--seed", type=int, default=SmokeConfig.seed)
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=SmokeConfig.minimum_improvement,
    )
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SmokeConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        steps=args.steps,
        seed=args.seed,
        minimum_improvement=args.minimum_improvement,
    )
    report = run(config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
