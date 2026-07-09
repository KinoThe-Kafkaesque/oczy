#!/usr/bin/env python3
"""Run the S3.M2a additive retrieval ablation and write JSON results."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from oczy.common.stats import summarize
from oczy.experiments.organ_additive_retrieval import run_additive_ablation
from oczy.experiments.organism_curriculum.dataset import (
    build_curriculum,
    split_probes,
)

OUT_DIR = Path("experiments_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    seeds = [0, 1, 2]  # 3 seeds (pre-registered fallback)

    stages = list(build_curriculum())
    print(f"Loaded {len(stages)} stages")

    holdout_splits: dict[str, set[str]] = {}
    total_holdout = 0
    for stage in stages:
        _, holdout = split_probes(stage, fraction=0.3, salt="v2")
        holdout_splits[stage.name] = holdout
        total_holdout += len(holdout)
        print(
            f"  {stage.name}: {len(stage.episodes)} eps,"
            f" {len(holdout)} holdout probes"
        )
    print(f"Total holdout probes: {total_holdout}")

    t0 = time.monotonic()
    results = run_additive_ablation(stages, holdout_splits, seeds)
    elapsed = time.monotonic() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Print summary
    print("\nCondition x Stage Accuracy Matrix:")
    for cond_name, stage_dict in results.items():
        print(f"\n  {cond_name}:")
        for stage_name, seed_results in stage_dict.items():
            accs = [r["holdout_accuracy"] for r in seed_results]
            s = summarize(accs)
            print(
                f"    {stage_name}: {s['mean']:.4f}"
                f" ± {s['ci95_half']:.4f} (n={int(s['n'])})"
            )

    # Write JSON
    out_path = OUT_DIR / "2026-07-02_s3_m2_retrieval_results.json"
    payload = {
        "spec": "research/14-s3-organ-ablation-matrix.md",
        "task": "S3.M2a",
        "seeds": seeds,
        "wall_clock_s": elapsed,
        "total_holdout_probes": total_holdout,
        "results": {},
    }
    for cond_name, stage_dict in results.items():
        payload["results"][cond_name] = {}
        for stage_name, seed_results in stage_dict.items():
            accs = [r["holdout_accuracy"] for r in seed_results]
            s = summarize(accs)
            payload["results"][cond_name][stage_name] = {
                "accuracies": accs,
                "mean": s["mean"],
                "ci95_half": s["ci95_half"],
                "n": int(s["n"]),
                "vanilla_acc": (
                    seed_results[0].get("vanilla_holdout_accuracy", 0.0)
                    if seed_results
                    else 0.0
                ),
            }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
