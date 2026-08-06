"""CLI for R24 tiny decoder Phase A.
Invoked as: python -m oczy.experiments.r24_tiny_decoder --train-per-family 20 ...
Writes metrics via METRIC lines and artifact JSON.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

def _build_parser():
    p = argparse.ArgumentParser(prog="oczy.experiments.r24_tiny_decoder")
    p.add_argument("--root-seed", type=int, default=123)
    p.add_argument("--train-per-family", type=int, default=20)
    p.add_argument("--val-per-family", type=int, default=10)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--conditioning", choices=["film","additive"], default="film")
    p.add_argument("--deep-film", action="store_true")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--output", type=str, default="outputs/r24-phase-a")
    p.add_argument("--use-text-encoder", action="store_true", default=True)
    p.add_argument("--no-text-encoder", dest="use_text_encoder", action="store_false")
    return p

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Import here to avoid import cost during help
    from oczy.experiments.r24_tiny_decoder.pretrain import train_phase_A
    import torch
    print(f"R24 Phase A: train {args.train_per_family} val {args.val_per_family} d{args.d_model} L{args.n_layers} {args.conditioning} steps {args.steps} lr {args.lr} text_enc {args.use_text_encoder}", flush=True)
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
        use_text_encoder=args.use_text_encoder,
    )
    # Emit METRIC lines for runner
    print(f"METRIC oracle_dev_accuracy={artifact['oracle_dev_accuracy']:.6f}", flush=True)
    print(f"METRIC query_only_accuracy={artifact['query_only_dev_accuracy']:.6f}", flush=True)
    print(f"METRIC delta={artifact['delta']:.6f}", flush=True)
    print(f"METRIC weight_hash_prefix={int(artifact['weight_hash'][:8],16) % 1000000}", flush=True)
    # Write artifact
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    (out / "execution_report.json").write_text(json.dumps({"phase":"r24-phase-a","artifact":artifact}, indent=2))
    print(f"Wrote {out / 'artifact.json'} hash {artifact['weight_hash'][:8]} delta {artifact['delta']:.3f}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
