"""Organ ablation harness for Sprint 3.1 / S3.M1.

Runs the full 6-stage curriculum for each organ configuration (FULL, MINIMAL,
FULL-minus-X) and reports per-stage holdout accuracy across ``--seeds`` seeds.
Two modes:

- **Mock path** (default): raw backend, no GGUF, no cortex agent.  Fast,
  already run and logged.
- **Real-driver path** (``--real-driver``): full ``OrganismAgent`` with
  LlamaCVecDriver via the legacy GGUF substrate — the honest reproduction
  pass required by ``research/14`` before archive decisions.  Per-organ
  off-switch configs are threaded through to the real agent.

Usage::

    uv run python -m oczy.experiments.organ_ablation [--seeds N] [--stages S0 S1 ...] [--real-driver]

Outputs::

    experiments_logs/2026-07-02_s3_m1_subtractive_ablation.md   (real-driver)
    experiments_logs/2026-07-02_s3_m1_subtractive_ablation.json  (real-driver)
    experiments_logs/2026-07-01_s3_1_organ_ablation_mock.*       (mock, existing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oczy.common.stats import summarize, format_row
from oczy.experiments.baselines import VanillaAgent
from oczy.experiments.organism_curriculum.dataset import (
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.experiments.organism_curriculum.run_curriculum import (
    _build_agent_for_seed,
    _run_stages,
    _stage_accuracy,
    run_battery,
    StageResult,
)

# ---------------------------------------------------------------------------
# Organ definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Organ:
    """One organ that can be ablated via a config override."""
    short: str           # stable key, e.g. "hippocampus"
    display: str         # human-readable, e.g. "NeuralHippocampus"
    off_cfg: dict[str, Any]  # config dict to disable this organ


ORGANS: list[Organ] = [
    Organ("hippocampus",        "NeuralHippocampus",       {"use_neural_hippocampus": False}),
    Organ("critic",             "WorldModelCritic",         {"use_world_model_critic": False}),
    Organ("identity",           "IdentityHypernetwork",     {"use_identity_hypernetwork": False}),
    Organ("immune",             "SkillImmuneCortex",        {"use_skill_immune_cortex": False}),
    Organ("autoencoder",        "ExperienceAutoencoder",    {"use_experience_autoencoder": False}),
    Organ("dsi",                "DifferentiableFactIndex",  {"use_diff_fact_index": False}),
    Organ("scope_slot_reranker","ScopeSlotReranker",        {"scope_rerank_weight": 0.0}),
]


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def build_configs() -> dict[str, dict[str, Any]]:
    """Return ``{config_name: config_dict}`` for all ablation configs.

    Each config is a **delta** — only the keys that differ from the default
    (FULL) are included, so the remaining values fall back to OrganismAgent
    defaults.
    """
    configs: dict[str, dict[str, Any]] = {}

    # FULL — empty delta, all defaults.
    configs["FULL"] = {}

    # MINIMAL — every organ disabled.
    minimal: dict[str, Any] = {}
    for organ in ORGANS:
        minimal.update(organ.off_cfg)
    configs["MINIMAL"] = minimal

    # FULL-minus-X — disable exactly one organ.
    for organ in ORGANS:
        configs[f"FULL-{organ.short}"] = dict(organ.off_cfg)

    return configs


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------

def _build_minimal_args(seeds: int = 3, *, use_real_driver: bool = False) -> argparse.Namespace:
    """Build an argparse.Namespace suitable for programmatic curriculum runs.

    When ``use_real_driver`` is False (default), no cortex agent is attached
    (raw backend path, no GGUF).  When True, the real-driver path is wired.
    """
    return argparse.Namespace(
        agent="OrganismAgent",
        use_cortex_shim=False,
        use_cortex_agent_mock=False,
        use_real_driver=use_real_driver,
        policy_log=None,
        semantic=False,
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_ablation(
    stages: list[Stage],
    stage_splits: dict[str, set[str]],
    n_seeds: int = 3,
    *,
    use_real_driver: bool = False,
    verbose: bool = True,
) -> dict[str, list[tuple[str, dict[str, float]]]]:
    """Run the organ ablation matrix.

    When ``use_real_driver`` is True, policy-loop gates are enabled in every
    config (mirroring ``run_curriculum.py``'s main) and the real GGUF driver
    is attached via ``_build_agent_for_seed``.

    Returns ``{config_name: [(stage_name, summary_dict), ...]}`` where each
    summary dict carries ``n``, ``mean``, ``std``, ``ci95_half``, ``min``,
    ``max``.
    """
    configs = build_configs()
    args = _build_minimal_args(seeds=n_seeds, use_real_driver=use_real_driver)

    # When running the real driver, enable the policy-loop gates that
    # run_curriculum.py's main activates for --use-real-driver, so the
    # CortexAgent policy head and value baseline are wired in.
    real_driver_gates: dict[str, Any] = {}
    if use_real_driver:
        real_driver_gates = {
            "use_cortex_policy": True,
            "use_value_baseline": True,
            "use_acceptance_policy_reward": True,
            "policy_suppresses_fast_answer": True,
        }

    results: dict[str, list[tuple[str, dict[str, float]]]] = {}

    for config_name, extra_config in configs.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {config_name}")
            print(f"{'=' * 60}")

        # Merge real-driver gates into each config before agent construction.
        merged_config = dict(extra_config)
        merged_config.update(real_driver_gates)

        seed_results: list[list[StageResult]] = []
        for seed in range(n_seeds):
            if verbose:
                print(f"    seed {seed + 1}/{n_seeds} ...", end="", flush=True)
            agent = _build_agent_for_seed(args, merged_config, seed=seed)
            sr = _run_stages(agent, stages, None, args, stage_splits)
            seed_results.append(sr)
            if verbose:
                accs = ", ".join(
                    f"{_stage_accuracy(sr[i]):.4f}" for i in range(len(sr))
                )
                print(f" [{accs}]", flush=True)

        # Per-stage summaries
        config_summary: list[tuple[str, dict[str, float]]] = []
        n_stages = len(seed_results[0]) if seed_results else 0
        for i in range(n_stages):
            accuracies = [
                _stage_accuracy(seed_results[s][i]) for s in range(n_seeds)
            ]
            config_summary.append(
                (seed_results[0][i].name, summarize(accuracies))
            )

        results[config_name] = config_summary

    return results


def run_vanilla_baseline(
    stages: list[Stage],
    stage_splits: dict[str, set[str]],
    *,
    verbose: bool = True,
) -> list[tuple[str, dict[str, float]]]:
    """Run a single deterministic VanillaAgent pass for the baseline column."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("  VANILLA (baseline)")
        print(f"{'=' * 60}")

    vanilla = VanillaAgent({})
    summary: list[tuple[str, dict[str, float]]] = []
    for stage in stages:
        split_ids = stage_splits.get(stage.name)
        post_results = run_battery(
            vanilla,
            stage,
            stage.episodes,
            split_ids=split_ids,
        )
        post_ok = sum(1 for _, _, ok in post_results if ok)
        post_total = len(post_results)
        acc = post_ok / post_total if post_total else 0.0
        summary.append((stage.name, {"n": 1, "mean": acc, "std": 0.0,
                                      "ci95_half": 0.0, "min": acc, "max": acc}))
        if verbose:
            print(f"    {stage.name}: {acc:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_acc(summary: dict[str, float]) -> str:
    """Format accuracy summary as ``mean ± ci95_half (n=N)``."""
    return f"{summary['mean']:.4f} ± {summary['ci95_half']:.4f} (n={int(summary['n'])})"


def print_matrix(
    results: dict[str, list[tuple[str, dict[str, float]]]],
    vanilla: list[tuple[str, dict[str, float]]] | None = None,
) -> None:
    """Print a human-readable config × stage accuracy matrix."""
    # Collect all config names in display order
    config_names = ["FULL", "MINIMAL"] + [
        f"FULL-{o.short}" for o in ORGANS
    ]
    # Only include configs that actually ran
    present = [cn for cn in config_names if cn in results]
    if vanilla is not None:
        present.append("VANILLA")

    # Stage names from first config
    first_config = results[present[0]] if present else []
    stage_names = [sn for sn, _ in first_config]

    # Header
    col_width = 28
    header = f"{'Config':<{col_width}}"
    header += "".join(f"{sn:>22}" for sn in stage_names)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  Organ Ablation Matrix  (dev split)")
    print(f"{'=' * len(header)}")
    print()
    print(header)
    print(sep)

    for cn in present:
        row = f"{cn:<{col_width}}"
        if cn == "VANILLA" and vanilla is not None:
            for sn, summary in vanilla:
                row += f"{_format_acc(summary):>22}"
        elif cn in results:
            stage_map = dict(results[cn])
            for sn in stage_names:
                summary = stage_map.get(sn, {})
                if summary:
                    row += f"{_format_acc(summary):>22}"
                else:
                    row += " " * 22
        print(row)

    print()


def _verdict(
    full_summary: dict[str, float],
    ablated_summary: dict[str, float],
) -> str:
    """Heuristic per-organ verdict based on accuracy deltas across stages.

    Returns a short verdict string.  On the mock/raw path deltas will be
    small; the real-driver follow-up is where behavioral effects manifest.
    """
    delta = ablated_summary.get("mean", 0.0) - full_summary.get("mean", 0.0)
    if abs(delta) < 0.02:
        return "dead weight (mock path — delta indistinguishable from noise)"
    if delta < -0.02:
        return f"contributing (+{abs(delta):.4f} when enabled; mock path)"
    return f"harmful ({delta:+.4f} — ablation improves accuracy; mock path)"


def write_outputs(
    results: dict[str, list[tuple[str, dict[str, float]]]],
    vanilla: list[tuple[str, dict[str, float]]] | None,
    out_dir: Path,
    out_stem: str,
    *,
    use_real_driver: bool = False,
) -> tuple[Path, Path]:
    """Write JSON + markdown log to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{out_stem}.md"
    json_path = out_dir / f"{out_stem}.json"

    path_label = "real-driver (LlamaCVecDriver GGUF)" if use_real_driver else "mock (raw backend)"
    backend_label = "real GGUF + CortexAgent" if use_real_driver else "raw (no cortex)"

    # ----------------------------------------------------------------
    # Markdown log
    # ----------------------------------------------------------------
    md_lines: list[str] = []
    md_lines.append(f"# Organ Ablation Matrix — {out_stem}")
    md_lines.append("")
    md_lines.append(f"**Path:** {path_label}.")
    md_lines.append("")
    md_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    md_lines.append(f"**Seeds:** {int(next(iter(results.values()))[0][1]['n'])}  |  "
                    f"**Split:** dev  |  **Backend:** {backend_label}")
    md_lines.append("")

    # Matrix table
    config_names = ["FULL", "MINIMAL"] + [f"FULL-{o.short}" for o in ORGANS]
    present = [cn for cn in config_names if cn in results]
    if vanilla is not None:
        present.append("VANILLA")

    first_config = results[present[0]]
    stage_names = [sn for sn, _ in first_config]

    md_lines.append("## Accuracy Matrix")
    md_lines.append("")
    header = "| Config | " + " | ".join(stage_names) + " |"
    md_lines.append(header)
    sep = "|" + "|".join("---" for _ in range(len(stage_names) + 1)) + "|"
    md_lines.append(sep)

    for cn in present:
        row = f"| {cn} |"
        if cn == "VANILLA" and vanilla is not None:
            vmap = dict(vanilla)
            for sn in stage_names:
                s = vmap.get(sn, {})
                row += f" {s.get('mean', 0):.4f} |"
        elif cn in results:
            smap = dict(results[cn])
            for sn in stage_names:
                s = smap.get(sn, {})
                row += f" {s.get('mean', 0):.4f} ± {s.get('ci95_half', 0):.4f} |"
        md_lines.append(row)

    md_lines.append("")

    # Per-organ Δ = FULL − (FULL−organ) table — per stage and all-stage mean.
    full_map = dict(results.get("FULL", []))
    md_lines.append("## Per-Organ Δ (FULL minus FULL-organ)")
    md_lines.append("")
    md_lines.append("Positive Δ → organ contributes; negative → ablation improves accuracy.")
    md_lines.append("")
    delta_header = "| Organ | " + " | ".join(stage_names) + " | All-stage mean ± CI |"
    md_lines.append(delta_header)
    delta_sep = "|" + "|".join("---" for _ in range(len(stage_names) + 2)) + "|"
    md_lines.append(delta_sep)

    for organ in ORGANS:
        cn = f"FULL-{organ.short}"
        if cn not in results:
            continue
        ablated_map = dict(results[cn])
        deltas: list[float] = []
        row = f"| **{organ.display}** |"
        for sn in stage_names:
            f_mean = full_map.get(sn, {}).get("mean", 0.0)
            a_mean = ablated_map.get(sn, {}).get("mean", 0.0)
            d = f_mean - a_mean
            deltas.append(d)
            row += f" {d:+.4f} |"
        # All-stage mean ± CI of the per-stage deltas.
        if deltas:
            delta_summary = summarize(deltas)
            row += f" {delta_summary['mean']:+.4f} ± {delta_summary['ci95_half']:.4f} |"
        else:
            row += " N/A |"
        md_lines.append(row)

    md_lines.append("")

    # Per-organ verdicts
    md_lines.append("## Per-Organ Verdicts")
    md_lines.append("")
    md_lines.append(f"Each verdict compares FULL-*organ* against FULL ({path_label}).")
    if not use_real_driver:
        md_lines.append("")
        md_lines.append("> ⚠️ **MOCK-PATH CAVEAT:** The raw backend has no cortex agent, so "
                         "critic/DSI/scope-slot signals are minimal or absent.  "
                         "The real-driver pass is required before organ archive decisions.")
    md_lines.append("")
    for organ in ORGANS:
        cn = f"FULL-{organ.short}"
        if cn not in results:
            continue
        ablated_map = dict(results[cn])
        deltas: list[float] = []
        for sn in stage_names:
            f = full_map.get(sn, {}).get("mean", 0.0)
            a = ablated_map.get(sn, {}).get("mean", 0.0)
            deltas.append(f - a)
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        if abs(avg_delta) < 0.005:
            verdict = "**dead weight** (delta indistinguishable from noise)"
        elif avg_delta > 0:
            verdict = f"**contributing** (organ saves {avg_delta:+.4f} on average)"
        else:
            verdict = f"**harmful** (ablation improves accuracy by {abs(avg_delta):+.4f} on average)"

        md_lines.append(
            f"- **{organ.display}** (`{organ.short}`): {verdict}  "
            f"(per-stage Δ: {', '.join(f'{sn}: {d:+.4f}' for sn, d in zip(stage_names, deltas))})"
        )

    md_lines.append("")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    # ----------------------------------------------------------------
    # JSON output
    # ----------------------------------------------------------------
    json_payload: dict[str, Any] = {
        "path": "real" if use_real_driver else "mock",
        "split": "dev",
        "n_seeds": int(next(iter(results.values()))[0][1]["n"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_names": stage_names,
        "configs": {},
    }

    for cn in present:
        if cn == "VANILLA" and vanilla is not None:
            json_payload["configs"][cn] = {
                "per_stage": {sn: s for sn, s in vanilla},
            }
        elif cn in results:
            json_payload["configs"][cn] = {
                "per_stage": {sn: s for sn, s in results[cn]},
            }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2, default=str)

    return md_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S3.1 / S3.M1 Organ ablation matrix harness."
    )
    p.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds per config (default: 3).",
    )
    p.add_argument(
        "--stages", nargs="*", default=None,
        help="Limit to these stage names (default: all 6).",
    )
    p.add_argument(
        "--real-driver",
        action="store_true",
        help="Attach a CortexAgent backed by the real LFM2.5 GGUF model (legacy substrate).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "experiments_logs",
        help="Output directory for JSON + markdown.",
    )
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    stage_names = tuple(args.stages) if args.stages else None
    stages = build_curriculum(stage_names=stage_names)

    # Dev split
    stage_splits: dict[str, set[str]] = {
        stage.name: split_probes(stage)[0] for stage in stages
    }

    # Run ablation
    results = run_ablation(
        stages, stage_splits, n_seeds=args.seeds,
        use_real_driver=args.real_driver,
    )

    # Vanilla baseline
    vanilla = run_vanilla_baseline(stages, stage_splits)

    # Print and write
    print_matrix(results, vanilla)
    if args.real_driver:
        out_stem = "2026-07-02_s3_m1_subtractive_ablation"
    else:
        out_stem = "2026-07-01_s3_1_organ_ablation_mock"
    md_path, json_path = write_outputs(
        results, vanilla, args.output_dir, out_stem,
        use_real_driver=args.real_driver,
    )
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
