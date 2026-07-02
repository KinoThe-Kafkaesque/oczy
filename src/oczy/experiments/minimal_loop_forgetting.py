"""S2.5 — The forgetting test: 2×2 deletion experiment.

Builds a minimal organism (HFDriver + fast-weight KVCortex +
NeuralHippocampus as consolidation-time replay only, 48-token
consolidated prefix channel) and runs the four-arm deletion test
per research/13.

Usage::

    uv run python -m oczy.experiments.minimal_loop_forgetting \
        --seeds 5 --stage 0
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.v2 import verify_manifest
from neural_hippocampus import NeuralHippocampus
from oczy.common.bytes import mem_bytes
from oczy.common.stats import format_row, summarize
from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm.hf_driver import HFDriver
from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MAX_PREFIX_TOKENS = 48
CONSOLIDATION_OBSERVE_LAYER = 5  # mid-layer for cortex observe
DEFAULT_SEEDS = 5
FALLBACK_SEEDS = 3

# ---------------------------------------------------------------------------
# Minimal Forgetting Organism
# ---------------------------------------------------------------------------


@dataclass
class OrganismSnapshot:
    """Pickleable snapshot of full organism state for arm derivation."""

    cold_state: np.ndarray
    warm_state: np.ndarray
    proj_hidden: np.ndarray
    proj_c: np.ndarray
    proj_c_shared: np.ndarray | None
    state_bias: np.ndarray
    prefix_text: str | None
    hippo_config: dict[str, Any]
    hippo_episodes: list[dict[str, Any]]
    cortex_config: dict[str, Any]


class MinimalForgettingOrganism:
    """Minimal organism per research/11 component rules.

    Components:
    - HFDriver (Qwen2.5-0.5B-Instruct, CPU float32, greedy)
    - KVCortex (warm/cold fast-weight cortex)
    - NeuralHippocampus (consolidation-time replay ONLY)
    - Content channel: bounded 48-token consolidated articulation prefix

    Banned: any answer-time hippocampus access, any other organs.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cortex_dim: int = 128,
        seed: int = 42,
        driver: HFDriver | None = None,
    ) -> None:
        self.model_id = model_id
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Driver
        if driver is not None:
            self.driver = driver
        else:
            self.driver = HFDriver.load(model_id)

        # Cortex: match driver dimensions
        self.cortex = KVCortex(
            KVCortexConfig(
                d_cortex=cortex_dim,
                d_embd=self.driver.n_embd,
                n_layers=self.driver.n_layers,
                seed=seed,
            )
        )

        # Hippocampus — consolidation-time replay ONLY
        self.hippocampus = NeuralHippocampus()

        # Content channel: consolidated articulation prefix
        self._prefix_text: str | None = None

        # Replay bank: accumulated hidden states from teaching
        self._replay_bank: list[np.ndarray] = []

        # Counters
        self._episode_count: int = 0
        self._consolidation_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def memory_bytes(self) -> int:
        """Pickle-based total memory footprint for the organism's mutable state."""
        state = {
            "cold_state": self.cortex.cold_state,
            "warm_state": self.cortex.warm_state,
            "prefix_text": self._prefix_text,
            "hippo_status": self.hippocampus.status(include_size=True),
        }
        return mem_bytes(state)

    @property
    def prefix_token_count(self) -> int:
        """Approximate token count of the current consolidation prefix."""
        if not self._prefix_text:
            return 0
        ids = self.driver._tokenize(self._prefix_text)
        return int(ids.shape[1])

    # ------------------------------------------------------------------
    # Boot / reset
    # ------------------------------------------------------------------

    def boot(self) -> None:
        """Cold boot: warm_state := cold_state.copy()."""
        self.cortex.reset_warm_from_cold()
        self._prefix_text = None
        self._replay_bank.clear()

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach(self, episode: Episode) -> None:
        """Feed one correction episode: perceive + store + accumulate replay.

        The hippocampus stores the episode. A hidden-state vector is captured
        for later consolidation replay.
        """
        correction = episode.correction_utterance

        # Capture hidden state at the observe layer for this correction text
        hidden = self.driver.peek_layer(
            correction, CONSOLIDATION_OBSERVE_LAYER, pooling="last"
        )

        # Feed through cortex (high correction_signal = 1.0)
        self.cortex.observe(hidden, correction_signal=1.0)

        # Store in hippocampus
        self.hippocampus.store(
            query=episode.initial_request,
            answer=episode.default_response,
            correction=correction,
            prediction_error=0.5,
            corrected_answer=episode.corrected_label,
        )

        # Accumulate hidden for consolidation replay
        self._replay_bank.append(hidden.copy())
        self._episode_count += 1

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Run consolidation: replay → cold_state update → compile prefix.

        Returns dict with consolidation stats.
        """
        # 1. Replay traces through cortex consolidation path
        replays = self._replay_bank.copy()
        self.cortex.consolidate(replays=replays, strength=1.0)

        # 2. Hippocampus consolidation → summaries
        hippo_summaries = self.hippocampus.consolidate()

        # 3. Compile consolidated prefix from summaries (≤ 48 tokens)
        prefix_candidates: list[str] = []
        for summary in hippo_summaries:
            if isinstance(summary, dict):
                query = summary.get("representative_query", "")
                corrections = summary.get("summary_corrections", [])
                if corrections and query:
                    fact = f"{query} -> {corrections[0]}"
                    prefix_candidates.append(fact)
                elif query:
                    prefix_candidates.append(query)

        self._prefix_text = self._build_prefix(prefix_candidates)
        if self._prefix_text:
            self.driver.set_articulation_prefix(self._prefix_text)

        self._consolidation_count += 1

        result: dict[str, Any] = {
            "consolidation_num": self._consolidation_count,
            "hippo_summaries": len(hippo_summaries),
            "prefix_tokens": self.prefix_token_count,
            "replay_count": len(replays),
            "memory_bytes": self.memory_bytes,
        }
        return result

    def _build_prefix(self, candidates: list[str]) -> str | None:
        """Build a consolidated prefix ≤ MAX_PREFIX_TOKENS tokens.

        Joins candidates with newlines, truncates to fit budget.
        """
        if not candidates:
            return None

        # Try full join first
        full = "\n".join(candidates)
        ids = self.driver._tokenize(full)
        if ids.shape[1] <= MAX_PREFIX_TOKENS:
            return full

        # Truncate: add candidates one at a time until budget exceeded
        result_parts: list[str] = []
        for cand in candidates:
            test = "\n".join(result_parts + [cand])
            ids = self.driver._tokenize(test)
            if ids.shape[1] <= MAX_PREFIX_TOKENS:
                result_parts.append(cand)
            else:
                break

        if not result_parts:
            return None
        return "\n".join(result_parts)

    # ------------------------------------------------------------------
    # Answer (NO hippocampus access)
    # ------------------------------------------------------------------

    def answer(self, request: str, max_tokens: int = 64) -> str:
        """Generate an answer through the LM with cortex steering + prefix.

        NEVER accesses the hippocampus at answer time.
        """
        # Apply cortex steering
        if self.cortex.has_uniform_proj_c():
            vec = self.cortex.emit_uniform_cvec()
            self.driver.set_cvec_uniform(vec)
        else:
            self.driver.set_cvecs_per_layer(self.cortex.emit_all_cvecs())

        try:
            result = self.driver.generate(request, max_tokens=max_tokens)
        finally:
            self.driver.clear_cvec()

        return result

    # ------------------------------------------------------------------
    # Deletion APIs
    # ------------------------------------------------------------------

    def delete_raw_traces(self) -> tuple[int, int]:
        """Clear hippocampus + replay bank. Returns (before_bytes, after_bytes)."""
        before = self.memory_bytes

        # Clear hippocampus: create new instance (no public clear_all)
        self.hippocampus = NeuralHippocampus()
        self._replay_bank.clear()
        self._episode_count = 0

        # Verify: episode count should be 0
        status = self.hippocampus.status()
        assert status.get("total_episodes", 0) == 0, (
            "delete_raw_traces: hippocampus episode count is "
            f"{status.get('total_episodes')}, expected 0"
        )

        after = self.memory_bytes
        return before, after

    def delete_consolidated_artifact(self) -> tuple[int, int]:
        """Clear prefix + reset cold state to boot value. Returns (before, after)."""
        before = self.memory_bytes

        # Clear prefix
        self.driver.clear_articulation_prefix()
        self._prefix_text = None

        # Reset cold state to boot (zeros) and warm with it
        boot_cold = np.zeros(self.cortex.config.d_cortex, dtype=np.float32)
        self.cortex.cold_state = boot_cold.copy()
        self.cortex.warm_state = boot_cold.copy()
        self.cortex._dirty = True

        after = self.memory_bytes
        return before, after

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> OrganismSnapshot:
        """Capture full state as a pickleable snapshot."""
        hippo_status = self.hippocampus.status()
        hippo_episodes_raw = hippo_status.get("episodes", [])
        hippo_episodes = [
            copy.deepcopy(ep) if isinstance(ep, dict) else ep
            for ep in hippo_episodes_raw
        ]

        return OrganismSnapshot(
            cold_state=self.cortex.cold_state.copy(),
            warm_state=self.cortex.warm_state.copy(),
            proj_hidden=self.cortex.proj_hidden.copy(),
            proj_c=self.cortex.proj_c.copy(),
            proj_c_shared=(
                self.cortex.proj_c_shared.copy()
                if self.cortex.proj_c_shared is not None
                else None
            ),
            state_bias=self.cortex.state_bias.copy(),
            prefix_text=self._prefix_text,
            hippo_config=copy.deepcopy(self.hippocampus.config),
            hippo_episodes=hippo_episodes,
            cortex_config={
                "d_cortex": self.cortex.config.d_cortex,
                "d_embd": self.cortex.config.d_embd,
                "n_layers": self.cortex.config.n_layers,
                "alpha_warm": self.cortex.config.alpha_warm,
                "alpha_correction": self.cortex.config.alpha_correction,
                "steering_mode": self.cortex.config.steering_mode,
            },
        )

    def restore(self, snap: OrganismSnapshot) -> None:
        """Restore full organism state from a snapshot.

        Replaces cortex state, hippocampus, prefix, and replay bank.
        Does NOT recreate the driver (LM is frozen).
        """
        # Cortex state
        self.cortex.cold_state = snap.cold_state.copy()
        self.cortex.warm_state = snap.warm_state.copy()
        self.cortex.proj_hidden = snap.proj_hidden.copy()
        self.cortex.proj_c = snap.proj_c.copy()
        if snap.proj_c_shared is not None:
            self.cortex.proj_c_shared = snap.proj_c_shared.copy()
        else:
            self.cortex.proj_c_shared = None
        self.cortex.state_bias = snap.state_bias.copy()
        self.cortex._dirty = True

        # Prefix
        self._prefix_text = snap.prefix_text
        if self._prefix_text:
            self.driver.set_articulation_prefix(self._prefix_text)
        else:
            self.driver.clear_articulation_prefix()

        # Hippocampus: rebuild from config + episodes
        self.hippocampus = NeuralHippocampus(config=snap.hippo_config)
        _STORE_KEYS = {"query", "answer", "correction", "prediction_error",
                        "corrected_answer", "hidden"}
        for ep in snap.hippo_episodes:
            if isinstance(ep, dict):
                store_kwargs = {k: v for k, v in ep.items() if k in _STORE_KEYS}
                if "query" in store_kwargs:
                    self.hippocampus.store(**store_kwargs)

        # Replay bank: derived from cortex state (not stored in snapshot)

    # ------------------------------------------------------------------
    # Convenience: train-then-snapshot
    # ------------------------------------------------------------------

    def train_all(self, episodes: list[Episode], seed_order: int) -> OrganismSnapshot:
        """Teach all episodes in seed-shuffled order, consolidate, snapshot."""
        order = list(range(len(episodes)))
        rng = np.random.default_rng(seed_order)
        rng.shuffle(order)

        for idx in order:
            self.teach(episodes[idx])

        self.consolidate()
        return self.snapshot()


# ---------------------------------------------------------------------------
# CLI runner — 2×2 forgetting test
# ---------------------------------------------------------------------------


def _score_holdout(
    organism: MinimalForgettingOrganism,
    probes: list[Any],
) -> float:
    """Score holdout probe accuracy for an organism in its current state."""
    correct = 0
    total = 0
    for probe in probes:
        ans = organism.answer(probe.request, max_tokens=64)
        if probe_matches(ans, probe.expected, probe.match_mode):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _run_seed(
    seed: int,
    model_id: str,
    stage: Stage,
    dev_probes: list[Any],
    holdout_probes: list[Any],
) -> dict[str, Any]:
    """Run one seed: train → snapshot → score four arms."""
    result: dict[str, Any] = {"seed": seed}

    # Build organism
    org = MinimalForgettingOrganism(model_id=model_id, seed=seed)

    # Train all episodes
    episodes = list(stage.episodes)
    t0 = time.monotonic()
    snap = org.train_all(episodes, seed_order=seed)
    train_time = time.monotonic() - t0
    result["train_time_s"] = train_time

    # Pre-deletion memory
    pre_bytes = org.memory_bytes
    result["memory_bytes_pre"] = pre_bytes

    # Score four arms from snapshot
    arm_accuracies: dict[str, float] = {}
    arm_memory: dict[str, dict[str, int]] = {}

    for arm_name, del_traces, del_artifact in [
        ("A_full", False, False),
        ("A_forget", True, False),
        ("A_retrieval", False, True),
        ("A_none", True, True),
    ]:
        org.restore(snap)

        if del_traces:
            before_t, after_t = org.delete_raw_traces()
            arm_memory[arm_name] = {
                "trace_before_bytes": before_t,
                "trace_after_bytes": after_t,
            }
        if del_artifact:
            before_a, after_a = org.delete_consolidated_artifact()
            arm_memory[arm_name] = {
                **(arm_memory.get(arm_name, {})),
                "artifact_before_bytes": before_a,
                "artifact_after_bytes": after_a,
            }

        acc = _score_holdout(org, holdout_probes)
        arm_accuracies[arm_name] = acc

    result["arm_accuracies"] = arm_accuracies
    result["arm_memory"] = arm_memory

    # Post-deletion memory (A_none state)
    org.restore(snap)
    org.delete_raw_traces()
    org.delete_consolidated_artifact()
    result["memory_bytes_post"] = org.memory_bytes

    return result


def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute primary and secondary metrics across seeds."""
    n_seeds = len(results)
    arm_keys = ["A_full", "A_forget", "A_retrieval", "A_none"]

    # Per-arm accuracies
    per_arm: dict[str, list[float]] = {k: [] for k in arm_keys}
    for r in results:
        for k in arm_keys:
            per_arm[k].append(r["arm_accuracies"][k])

    # Primary metric: forgetting_survival_ratio
    survival_ratios: list[float] = []
    retrieval_ratios: list[float] = []
    validity_gate_passes: list[bool] = []
    blocked_count = 0

    for r in results:
        full = r["arm_accuracies"]["A_full"]
        forget = r["arm_accuracies"]["A_forget"]
        retrieval = r["arm_accuracies"]["A_retrieval"]
        none_acc = r["arm_accuracies"]["A_none"]
        delta = full - none_acc

        if delta < 0.10:
            survival_ratios.append(float("nan"))
            retrieval_ratios.append(float("nan"))
            blocked_count += 1
            validity_gate_passes.append(False)
        else:
            survival_ratios.append((forget - none_acc) / delta)
            retrieval_ratios.append((retrieval - none_acc) / delta)
            validity_gate_passes.append(True)

    arm_summaries = {k: summarize(v) for k, v in per_arm.items()}

    # Filter NaN for ratio stats
    valid_survival = [x for x in survival_ratios if not np.isnan(x)]
    valid_retrieval = [x for x in retrieval_ratios if not np.isnan(x)]

    metrics: dict[str, Any] = {
        "n_seeds": n_seeds,
        "n_blocked": blocked_count,
        "arm_summaries": arm_summaries,
        "forgetting_survival_ratio": (
            summarize(valid_survival) if valid_survival else None
        ),
        "retrieval_dependence": (
            summarize(valid_retrieval) if valid_retrieval else None
        ),
        "validity_gate_passes": validity_gate_passes,
        "raw_survival_ratios": survival_ratios,
        "raw_retrieval_ratios": retrieval_ratios,
    }

    # behavior_delta_per_byte (pre and post deletion)
    pre_bytes = [r["memory_bytes_pre"] for r in results]
    post_bytes = [r["memory_bytes_post"] for r in results]
    deltas = [r["arm_accuracies"]["A_full"] - r["arm_accuracies"]["A_none"] for r in results]

    bdpb_pre = [d / max(b, 1) for d, b in zip(deltas, pre_bytes)]
    bdpb_post = [d / max(b, 1) for d, b in zip(deltas, post_bytes)]
    # Scale to KB for readability
    bdpb_pre_kb = [d / max(b / 1024, 1) for d, b in zip(deltas, pre_bytes)]
    bdpb_post_kb = [d / max(b / 1024, 1) for d, b in zip(deltas, post_bytes)]

    metrics["behavior_delta_per_byte_pre"] = summarize(bdpb_pre)
    metrics["behavior_delta_per_byte_post"] = summarize(bdpb_post)
    metrics["behavior_delta_per_kb_pre"] = summarize(bdpb_pre_kb)
    metrics["behavior_delta_per_kb_post"] = summarize(bdpb_post_kb)

    # Verdict
    survival_mean = metrics["forgetting_survival_ratio"]["mean"] if valid_survival else float("nan")
    if blocked_count == n_seeds:
        verdict = "BLOCKED"
    elif survival_mean >= 0.8:
        verdict = "ACCEPT"
    elif survival_mean < 0.5:
        verdict = "REFUTE"
    else:
        verdict = "PARTIAL"
    metrics["verdict"] = verdict

    return metrics


def _write_log(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    model_id: str,
    seeds: int,
    stage_name: str,
    output_path: Path,
    command: str,
) -> None:
    """Write the experiment log markdown report."""
    arm_order = ["A_full", "A_forget", "A_retrieval", "A_none"]

    lines: list[str] = []
    lines.append(f"# S2.5 Forgetting Test — {metrics['verdict']}")
    lines.append("")
    lines.append(f"**Date:** 2026-07-02")
    lines.append(f"**Spec:** research/13-s2-forgetting-test.md")
    lines.append(f"**Model:** {model_id}")
    lines.append(f"**Stage:** {stage_name}")
    lines.append(f"**Seeds:** {seeds}")
    lines.append(f"**Command:** `{command}`")
    lines.append("")

    # 2×2 table
    lines.append("## 2×2 Arm Accuracies (mean ± 95% CI)")
    lines.append("")
    arm_labels = {
        "A_full": "Raw traces: kept | Artifact: kept",
        "A_forget": "Raw traces: **deleted** | Artifact: kept",
        "A_retrieval": "Raw traces: kept | Artifact: **deleted**",
        "A_none": "Raw traces: **deleted** | Artifact: **deleted**",
    }
    lines.append("| Arm | Condition | Accuracy |")
    lines.append("|---|---|---|")
    for arm in arm_order:
        s = metrics["arm_summaries"][arm]
        label = arm_labels[arm]
        lines.append(
            f"| **{arm}** | {label} | "
            f"{s['mean']:.4f} ± {s['ci95_half']:.4f} (n={s['n']}) |"
        )
    lines.append("")

    # Primary metrics
    lines.append("## Primary Metric")
    lines.append("")
    sr = metrics["forgetting_survival_ratio"]
    if sr:
        lines.append(
            f"- **forgetting_survival_ratio:** "
            f"{sr['mean']:.4f} ± {sr['ci95_half']:.4f} "
            f"(n={sr['n']})"
        )
    else:
        lines.append("- **forgetting_survival_ratio:** BLOCKED (all seeds failed validity gate)")
    lines.append("")

    lines.append("## Validity Gate")
    lines.append(f"- `A_full − A_none ≥ 0.10`: "
                 f"{metrics['n_seeds'] - metrics['n_blocked']}/{metrics['n_seeds']} seeds passed")
    lines.append(f"- BLOCKED seeds: {metrics['n_blocked']}")
    lines.append("")

    # Secondary analyses
    lines.append("## Secondary Analyses")
    lines.append("")
    rd = metrics["retrieval_dependence"]
    if rd:
        lines.append(
            f"- **retrieval_dependence:** "
            f"{rd['mean']:.4f} ± {rd['ci95_half']:.4f} "
            f"(n={rd['n']})"
        )
    else:
        lines.append("- **retrieval_dependence:** BLOCKED")
    lines.append("")

    bpb_pre = metrics["behavior_delta_per_kb_pre"]
    bpb_post = metrics["behavior_delta_per_kb_post"]
    lines.append(
        f"- **behavior_delta_per_kb (pre-deletion):** "
        f"{bpb_pre['mean']:.6f} ± {bpb_pre['ci95_half']:.6f}"
    )
    lines.append(
        f"- **behavior_delta_per_kb (post-deletion):** "
        f"{bpb_post['mean']:.6f} ± {bpb_post['ci95_half']:.6f}"
    )
    lines.append("")

    # Per-seed detail
    lines.append("## Per-Seed Detail")
    lines.append("")
    lines.append("| Seed | A_full | A_forget | A_retrieval | A_none | Survival | Retrieval | mem_pre | mem_post | Validity |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results):
        acc = r["arm_accuracies"]
        sr_val = metrics["raw_survival_ratios"][i]
        rr_val = metrics["raw_retrieval_ratios"][i]
        gated = metrics["validity_gate_passes"][i]
        sr_str = f"{sr_val:.4f}" if not np.isnan(sr_val) else "BLOCKED"
        rr_str = f"{rr_val:.4f}" if not np.isnan(rr_val) else "BLOCKED"
        lines.append(
            f"| {r['seed']} | {acc['A_full']:.4f} | {acc['A_forget']:.4f} | "
            f"{acc['A_retrieval']:.4f} | {acc['A_none']:.4f} | "
            f"{sr_str} | {rr_str} | "
            f"{r['memory_bytes_pre']} | {r['memory_bytes_post']} | "
            f"{'PASS' if gated else 'BLOCKED'} |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append(f"**{metrics['verdict']}**")
    lines.append("")
    lines.append("Per research/13 acceptance criteria:")
    lines.append("- ACCEPT: forgetting_survival_ratio ≥ 0.8 with validity gate passing")
    lines.append("- REFUTE: ratio < 0.5")
    lines.append("- PARTIAL: 0.5 ≤ ratio < 0.8")
    lines.append("- BLOCKED: validity gate failed on all seeds")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLog written to {output_path}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S2.5 — The forgetting test (2×2 deletion experiment)"
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"Number of seeds (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Curriculum stage index (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID override"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output log path override"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit"
    )
    args = parser.parse_args(argv)

    model_id = args.model or DEFAULT_MODEL_ID
    stage_idx = args.stage
    seeds = args.seeds
    output_path = Path(args.output) if args.output else Path(
        "experiments_logs/2026-07-02_s2_5_forgetting_test.md"
    )

    if args.dry_run:
        print(f"Model: {model_id}")
        print(f"Stage index: {stage_idx}")
        print(f"Seeds: {seeds}")
        print(f"Output: {output_path}")
        return 0

    # Verify manifest
    print("Verifying eval manifest...")
    verify_manifest()

    # Load curriculum
    stages = build_curriculum()
    if stage_idx >= len(stages):
        print(f"Stage {stage_idx} not found (max {len(stages) - 1})")
        return 1
    stage = stages[stage_idx]

    # Split probes
    dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2")
    holdout_probes = [p for p in stage.probes if p.id in holdout_ids]
    dev_probes = [p for p in stage.probes if p.id in dev_ids]
    print(
        f"Stage {stage_idx} ({stage.name}): "
        f"{len(stage.episodes)} episodes, "
        f"{len(holdout_probes)} holdout / {len(dev_probes)} dev probes"
    )

    # Dry-run one seed to estimate time
    print(f"\n--- Dry-run seed 0 (timing) ---")
    t_dry_start = time.monotonic()
    _run_seed(0, model_id, stage, dev_probes, holdout_probes)
    dry_time = time.monotonic() - t_dry_start
    est_total = dry_time * seeds
    print(f"Dry-run: {dry_time:.1f}s → estimated {est_total:.1f}s for {seeds} seeds")

    if est_total > seeds * 900 and seeds > FALLBACK_SEEDS:
        print(
            f"WARNING: per-seed time ~{dry_time:.0f}s exceeds 15 min/seed. "
            f"Dropping to {FALLBACK_SEEDS} seeds (pre-registered fallback)."
        )
        seeds = FALLBACK_SEEDS

    # Run all seeds
    results: list[dict[str, Any]] = []
    for seed in range(seeds):
        print(f"\n--- Seed {seed + 1}/{seeds} ---")
        t0 = time.monotonic()
        result = _run_seed(seed, model_id, stage, dev_probes, holdout_probes)
        elapsed = time.monotonic() - t0
        acc = result["arm_accuracies"]
        print(
            f"  A_full={acc['A_full']:.4f} A_forget={acc['A_forget']:.4f} "
            f"A_retrieval={acc['A_retrieval']:.4f} A_none={acc['A_none']:.4f} "
            f"({elapsed:.1f}s)"
        )
        results.append(result)

    # Compute metrics
    metrics = _compute_metrics(results)

    # Print summary
    print("\n=== Results ===")
    arm_order = ["A_full", "A_forget", "A_retrieval", "A_none"]
    for arm in arm_order:
        print(format_row(arm, metrics["arm_summaries"][arm]))

    sr = metrics["forgetting_survival_ratio"]
    if sr:
        print(format_row("forgetting_survival_ratio", sr))
    else:
        print("forgetting_survival_ratio: BLOCKED")

    rd = metrics["retrieval_dependence"]
    if rd:
        print(format_row("retrieval_dependence", rd))

    print(f"\nVerdict: {metrics['verdict']}")

    # Verify manifest again
    verify_manifest()

    # Write log
    cmd = " ".join(sys.argv)
    _write_log(results, metrics, model_id, seeds, stage.name, output_path, cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
