"""CLI for R24 tiny decoder Phase A v1 and the human-authorized v2 suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oczy.experiments.r24_tiny_decoder")
    parser.add_argument("--protocol-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--diagnostic-ladder", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--root-seed", type=int, default=123)
    parser.add_argument("--catalog-seed", type=int)
    parser.add_argument("--init-seed", type=int)
    parser.add_argument("--batch-seed", type=int)
    parser.add_argument("--dropout-seed", type=int)
    parser.add_argument("--control-seed", type=int)
    parser.add_argument("--train-per-family", type=int, default=20)
    parser.add_argument("--val-per-family", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--conditioning", choices=["film", "additive", "prefix", "none"], default="film"
    )
    parser.add_argument("--deep-film", action="store_true")
    parser.add_argument("--n-prefix-tokens", type=int, default=4)
    parser.add_argument(
        "--encoder-pooling",
        choices=["mean", "cls", "attention", "line_attention"],
        default="mean",
    )
    parser.add_argument("--oracle-mode", choices=["text", "hash", "learned"], default="text")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="outputs/r24-phase-a")
    parser.add_argument("--use-text-encoder", action="store_true", default=True)
    parser.add_argument("--no-text-encoder", dest="use_text_encoder", action="store_false")
    return parser


def _emit_metrics(artifact: dict[str, Any]) -> None:
    oracle = float(artifact["oracle_dev_accuracy"])
    query_only = float(artifact["query_only_dev_accuracy"])
    delta = float(artifact["delta"])
    print(f"METRIC oracle_dev_accuracy={oracle:.6f}", flush=True)
    print(f"METRIC query_only_accuracy={query_only:.6f}", flush=True)
    print(f"METRIC delta={delta:.6f}", flush=True)
    if "swapped_dev_accuracy" in artifact:
        print(
            f"METRIC swapped_dev_accuracy={float(artifact['swapped_dev_accuracy']):.6f}",
            flush=True,
        )
    if "swapped_delta" in artifact:
        print(
            f"METRIC swapped_delta={float(artifact['swapped_delta']):.6f}",
            flush=True,
        )
    validation = artifact.get("validation")
    if isinstance(validation, dict) and "teacher_forced_token_accuracy" in validation:
        print(
            "METRIC teacher_forced_token_accuracy="
            f"{float(validation['teacher_forced_token_accuracy']):.6f}",
            flush=True,
        )
    weight_hash = str(artifact["weight_hash"])
    print(
        f"METRIC weight_hash_prefix={int(weight_hash[:8], 16) % 1000000}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    if args.protocol_version == "v1":
        v2_only = {
            "--catalog-seed",
            "--init-seed",
            "--batch-seed",
            "--dropout-seed",
            "--control-seed",
            "--encoder-pooling",
            "--oracle-mode",
            "--scheduler",
            "--warmup-steps",
            "--counterfactual-weight",
            "--dropout",
            "--encoder-lr-multiplier",
            "--weight-decay",
            "--n-prefix-tokens",
        }
        used = sorted(
            option
            for option in v2_only
            if any(token == option or token.startswith(option + "=") for token in raw_argv)
        )
        if used:
            raise SystemExit("v2-only flags require --protocol-version v2: " + ", ".join(used))
    output = Path(args.output)
    if args.diagnostic_ladder:
        if args.protocol_version != "v2":
            raise SystemExit("--diagnostic-ladder requires --protocol-version v2")
        from .diagnostics_v2 import run_overfit_ladder

        summary = run_overfit_ladder(
            output_dir=output,
            root_seed=args.root_seed,
            quick=args.quick,
        )
        print(f"Wrote {output / 'summary.json'}", flush=True)
        print("METRIC ladder_cases=" + str(len(summary["cases"])), flush=True)
        return 0

    print(
        f"R24 Phase A {args.protocol_version}: train {args.train_per_family} "
        f"val {args.val_per_family} d{args.d_model} L{args.n_layers} "
        f"{args.conditioning} steps {args.steps} lr {args.lr}",
        flush=True,
    )
    if args.protocol_version == "v1":
        if args.conditioning in {"prefix", "none"}:
            raise SystemExit("prefix/none conditioning requires --protocol-version v2")
        from .pretrain import train_phase_A

        artifact = train_phase_A(
            root_seed=args.root_seed,
            train_per_family=args.train_per_family,
            val_per_family=args.val_per_family,
            d_model=args.d_model,
            n_layers=args.n_layers,
            conditioning=args.conditioning,
            deep_film=args.deep_film,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            use_text_encoder=args.use_text_encoder,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    else:
        from .phase_a_v2 import PhaseAV2Config, train_phase_a_v2

        config = PhaseAV2Config(
            root_seed=args.root_seed,
            catalog_seed=args.catalog_seed,
            init_seed=args.init_seed,
            batch_seed=args.batch_seed,
            dropout_seed=args.dropout_seed,
            control_seed=args.control_seed,
            train_per_family=args.train_per_family,
            val_per_family=args.val_per_family,
            d_model=args.d_model,
            n_layers=args.n_layers,
            conditioning=args.conditioning,
            deep_film=args.deep_film,
            n_prefix_tokens=args.n_prefix_tokens,
            encoder_pooling=args.encoder_pooling,
            oracle_mode=args.oracle_mode,
            steps=args.steps,
            lr=args.lr,
            encoder_lr_multiplier=args.encoder_lr_multiplier,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            counterfactual_weight=args.counterfactual_weight,
            dropout=args.dropout,
        )
        artifact = train_phase_a_v2(config, output_dir=output)
    _emit_metrics(artifact)
    print(
        f"Wrote {output / 'artifact.json'} hash {str(artifact['weight_hash'])[:8]} "
        f"delta {float(artifact['delta']):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
