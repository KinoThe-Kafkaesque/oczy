"""Minimal metabolism loop on the HF substrate (Sprint 2 / S2.1).

Pre-registered experiment: research/11-s2-minimal-metabolism-loop.md

A MinimalOrganism with EXACTLY three components:
1. HFDriver (Qwen2.5-0.5B-Instruct, CPU float32)
2. KVCortex (fast-weight warm/cold state, Hebbian observe, consolidate)
3. NeuralHippocampus (consolidation-time replay buffer ONLY)

Explicitly banned: critic, identity, immune, autoencoder, DSI,
scope-slot reranker, logit bias, answer-time retrieval.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any

import numpy as np

from eval.v2 import verify_manifest
from neural_hippocampus import NeuralHippocampus
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

MAX_PREFIX_TOKENS: int = 48
CHECKPOINTS: list[int] = [0, 1, 2, 4, -1]  # -1 means "all episodes"


def _spearman_rho(x: list[int], y: list[float]) -> float:
    """Spearman's rank correlation coefficient (no scipy dependency)."""
    if len(x) < 3 or len(set(x)) < 3 or len(set(y)) < 2:
        return float("nan")
    x_rank = np.argsort(np.argsort(x)).astype(float)
    y_rank = np.argsort(np.argsort(y)).astype(float)
    x_mean = np.mean(x_rank)
    y_mean = np.mean(y_rank)
    cov = np.sum((x_rank - x_mean) * (y_rank - y_mean))
    sx = np.sqrt(np.sum((x_rank - x_mean) ** 2))
    sy = np.sqrt(np.sum((y_rank - y_mean) ** 2))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(cov / (sx * sy))


# ---------------------------------------------------------------------------
# MinimalOrganism
# ---------------------------------------------------------------------------


class _HippocampusGuard:
    """Wraps a NeuralHippocampus and raises if touched during answer()."""

    def __init__(self, hippo: NeuralHippocampus) -> None:
        self._hippo = hippo
        self._in_answer = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._in_answer and name not in ("status",):
            raise RuntimeError(
                f"Hippocampus.{name}() called during answer() — "
                "answer-time retrieval is banned by spec"
            )
        return getattr(self._hippo, name)


class MinimalOrganism:
    """Minimal organism: HFDriver + KVCortex + NeuralHippocampus only.

    Lifecycle::

        org = MinimalOrganism(driver)
        org.boot()
        for ep in curriculum:
            org.teach(ep)           # perceive + store
        org.consolidate(queries)    # replay + consolidate + build prefix
        answer = org.answer(probe)  # prefix + cvec only, no hippocampus

    Banned at answer time: any hippocampus method except ``status()``.
    """

    def __init__(
        self,
        driver: HFDriver,
        d_cortex: int = 128,
        cortex_seed: int = 0,
        hippocampus_config: dict[str, Any] | None = None,
    ) -> None:
        self.driver = driver
        self.d_cortex = d_cortex

        # Hippocampus: store everything (surprise_threshold=0) so every
        # teaching episode is available for consolidation-time replay.
        hc: dict[str, Any] = {"dim": d_cortex, "surprise_threshold": 0.0}
        if hippocampus_config:
            hc.update(hippocampus_config)
        self._hippo_raw = NeuralHippocampus(hc)
        self.hippocampus = _HippocampusGuard(self._hippo_raw)

        # Cortex: patch dimensions from the driver.
        cortex_config = KVCortexConfig(
            d_cortex=d_cortex,
            d_embd=driver.n_embd,
            n_layers=driver.n_layers,
            seed=cortex_seed,
        )
        self.cortex = KVCortex(cortex_config)

        # Content channel state.
        self._prefix_text: str = ""
        self._prefix_token_count: int = 0
        self._prefix_overflow_count: int = 0

        # Bookkeeping for consolidation replay queries.
        self._taught_queries: list[str] = []


    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self) -> None:
        """Cold boot: reset warm state from cold, clear prefix/hippocampus."""
        self.cortex.reset_warm_from_cold()
        self.driver.clear_cvec()
        self.driver.clear_articulation_prefix()
        self._prefix_text = ""
        self._prefix_token_count = 0
        self._prefix_overflow_count = 0
        self._taught_queries = []

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach(self, episode: Episode) -> None:
        """Feed one curriculum episode: perceive correction, store in hippocampus.

        Hidden state is extracted from the *correction utterance* (the
        teaching signal), not the default response.
        """
        hidden = self.driver.peek_embedding(
            episode.correction_utterance, last_token_only=True
        )
        self.cortex.observe(hidden, correction_signal=1.0)

        self._hippo_raw.store(
            query=episode.initial_request,
            answer=episode.default_response,
            correction=episode.correction_utterance,
            prediction_error=1.0,
            hidden=hidden,
        )
        self._taught_queries.append(episode.initial_request)

    # ------------------------------------------------------------------
    # Answer (NEVER touches hippocampus)
    # ------------------------------------------------------------------

    def answer(self, request: str, max_tokens: int = 32) -> str:
        """Generate an answer from driver + consolidated artifact only.

        The hippocampus is guarded: any method call (except ``status()``)
        during answer() raises ``RuntimeError``.
        """
        self.hippocampus._in_answer = True
        try:
            # Articulate: push current cortex cvecs to the driver.
            cvecs = self.cortex.emit_all_cvecs()
            self.driver.set_cvecs_per_layer(cvecs)
            result = self.driver.generate(request, max_tokens=max_tokens)
            return result
        finally:
            self.hippocampus._in_answer = False

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Consolidation-time replay: reinforce → hippocampus.consolidate →
        cortex.consolidate → build prefix → set prefix on driver.

        Returns a dict with consolidation metadata for logging.
        """
        # Snapshot cold state BEFORE consolidation (same instance).
        cold_before = self.cortex.cold_state.copy()

        # Step 1: Reinforce from hippocampus for each taught query.
        for query in self._taught_queries:
            self._hippo_raw.reinforce(query, k=3)

        # Step 2: Consolidate hippocampus → slow-update summaries.
        summaries = self._hippo_raw.consolidate()

        # Step 3: Extract representative_hidden vectors for cortex replay.
        replays: list[np.ndarray] = []
        for summary in summaries:
            h = summary.get("representative_hidden")
            if h is not None:
                replays.append(np.asarray(h, dtype=np.float32))

        # Step 4: Cortex consolidation with replays.
        self.cortex.consolidate(replays, strength=1.0)

        # Step 5: Build articulation prefix from summary corrections.
        corrections_text: list[str] = []
        for summary in summaries:
            for corr in summary.get("summary_corrections", []):
                if corr:
                    corrections_text.append(corr)
        prefix = self._build_prefix(corrections_text)
        self._prefix_text = prefix
        self.driver.set_articulation_prefix(prefix)

        # Drift metrics (same instance, captured in one shot).
        cold_after = self.cortex.cold_state
        cold_drift = float(np.linalg.norm(cold_after - cold_before))
        cvec_norm = self._cvec_combined_norm()

        # Clamped variant: recompute cvecs at budget = cvec_norm (no real
        # clamp needed since norm == budget; but we record for completeness).
        cvec_norm_clamped = min(cvec_norm, 1.0)  # budget = 1.0

        return {
            "cold_before_norm": float(np.linalg.norm(cold_before)),
            "cold_after_norm": float(np.linalg.norm(cold_after)),
            "cold_drift": cold_drift,
            "cvec_norm": cvec_norm,
            "cvec_norm_budget": 1.0,
            "cvec_norm_clamped": cvec_norm_clamped,
            "drift_target": cold_drift,
            "drift_control": 0.0,  # no control words in minimal loop
            "drift_target_clamped": (
                cold_drift * (cvec_norm_clamped / max(cvec_norm, 1e-6))
            ),
            "prefix_tokens": self._prefix_token_count,
            "prefix_overflow": self._prefix_overflow_count,
            "n_summaries": len(summaries),
            "hippo_status": self._hippo_raw.status(),
        }

    # ------------------------------------------------------------------
    # Content channel (48-token prefix budget)
    # ------------------------------------------------------------------

    def _build_prefix(self, corrections: list[str]) -> str:
        """Compile corrections into a bounded articulation prefix.

        Budget: MAX_PREFIX_TOKENS tokens total.  Oldest dropped on overflow;
        dropped count reported.
        """
        if not corrections:
            self._prefix_token_count = 0
            self._prefix_overflow_count = 0
            return ""

        token_ids = self.driver._tokenize(" ".join(corrections))
        # _tokenize returns a (1, seq_len) tensor; index the row to get a
        # 1-D sequence so len() / slicing operate on tokens, not the batch dim.
        seq = token_ids[0]
        total = len(seq)
        if total <= MAX_PREFIX_TOKENS:
            self._prefix_token_count = total
            self._prefix_overflow_count = 0
            return " ".join(corrections)

        # Drop oldest tokens to fit budget.
        kept_ids = seq[-MAX_PREFIX_TOKENS:]
        self._prefix_token_count = len(kept_ids)
        self._prefix_overflow_count = total - MAX_PREFIX_TOKENS
        return self.driver._tokenizer.decode(kept_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cvec_combined_norm(self) -> float:
        """Combined L2 norm across all per-layer cvecs."""
        cvecs = self.cortex.emit_all_cvecs()
        merged_sq = sum(float(np.sum(v * v)) for v in cvecs)
        return float(np.sqrt(merged_sq))

    def memory_bytes(self) -> int:
        """Approximate memory footprint of organism state."""
        import pickle

        state = {
            "cold_state": self.cortex.cold_state,
            "warm_state": self.cortex.warm_state,
            "proj_hidden": self.cortex.proj_hidden,
            "proj_c": self.cortex.proj_c,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _run_one_seed(
    seed: int,
    stage: Stage,
    holdout_ids: set[str],
    checkpoints: list[int],
    driver_factory: Any = HFDriver.load,
) -> dict[str, Any]:
    """Run the full experiment for one seed.

    Returns a dict with per-checkpoint metrics.
    """
    rng = np.random.RandomState(seed)

    # Build organism.
    driver = driver_factory()
    org = MinimalOrganism(driver, d_cortex=128, cortex_seed=seed)
    org.boot()

    # Build holdout probes index.
    holdout_probes: list[tuple[str, Any, Any]] = []
    for ep in stage.episodes:
        for probe in ep.probes:
            probe_id = f"{ep.id}|{probe.request}|{probe.category}"
            if probe_id in holdout_ids:
                holdout_probes.append((probe_id, probe, ep))

    # Shuffle episodes by seed.
    episodes = list(stage.episodes)
    rng.shuffle(episodes)  # type: ignore[arg-type]
    n_episodes = len(episodes)
    checkpoint_values = [c if c >= 0 else n_episodes for c in checkpoints]

    # Compute vanilla baseline (cold, no prefix, no cvecs).
    vanilla_driver = driver_factory()
    vanilla_results: list[bool] = []
    for _, probe, ep in holdout_probes:
        answer = vanilla_driver.generate(probe.request, max_tokens=32)
        vanilla_results.append(probe_matches(answer, probe, ep))
    vanilla_acc = sum(vanilla_results) / max(len(vanilla_results), 1)
    del vanilla_driver
    gc.collect()

    # Run through checkpoints cumulatively.
    taught_idx = 0
    per_checkpoint: list[dict[str, Any]] = []
    cold_norms: list[float] = []

    for cp_val in sorted(set(checkpoint_values)):
        # Teach episodes up to this checkpoint.
        while taught_idx < cp_val and taught_idx < n_episodes:
            org.teach(episodes[taught_idx])
            taught_idx += 1

        t0 = time.monotonic()

        # Consolidate.
        cold_before_norm = float(np.linalg.norm(org.cortex.cold_state))
        cons_meta = org.consolidate()
        cold_after_norm = float(np.linalg.norm(org.cortex.cold_state))
        cold_norms.append(cold_after_norm)

        # Score holdout probes.
        results: list[bool] = []
        for _, probe, ep in holdout_probes:
            ans = org.answer(probe.request, max_tokens=32)
            results.append(probe_matches(ans, probe, ep))
        holdout_acc = sum(results) / max(len(results), 1)

        elapsed = time.monotonic() - t0

        mem = org.memory_bytes()

        per_checkpoint.append(
            {
                "K": taught_idx,
                "holdout_accuracy": holdout_acc,
                "cold_before_norm": cold_before_norm,
                "cold_after_norm": cold_after_norm,
                "cold_drift": cons_meta["cold_drift"],
                "cvec_norm": cons_meta["cvec_norm"],
                "cvec_norm_clamped": cons_meta["cvec_norm_clamped"],
                "drift_target": cons_meta["drift_target"],
                "drift_control": cons_meta["drift_control"],
                "drift_target_clamped": cons_meta["drift_target_clamped"],
                "memory_bytes": mem,
                "prefix_tokens": cons_meta["prefix_tokens"],
                "prefix_overflow": cons_meta["prefix_overflow"],
                "n_summaries": cons_meta["n_summaries"],
                "wall_clock_s": elapsed,
            }
        )

    del driver
    gc.collect()

    return {
        "seed": seed,
        "vanilla_holdout_accuracy": vanilla_acc,
        "checkpoints": per_checkpoint,
    }


def _run_experiment(
    seeds: list[int],
    stage: Stage,
    holdout_ids: set[str],
    checkpoints: list[int],
) -> dict[str, Any]:
    """Run across all seeds and compute aggregate statistics."""
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"  Seed {seed} ...", end=" ", flush=True)
        t0 = time.monotonic()
        result = _run_one_seed(seed, stage, holdout_ids, checkpoints)
        elapsed = time.monotonic() - t0
        print(f"done ({elapsed:.0f}s), K=N acc={result['checkpoints'][-1]['holdout_accuracy']:.4f}")

        seed_results.append(result)

    # Compute primary metrics.
    final_accuracies = [
        r["checkpoints"][-1]["holdout_accuracy"] for r in seed_results
    ]
    vanilla_accuracies = [r["vanilla_holdout_accuracy"] for r in seed_results]
    mean_final = np.mean(final_accuracies)
    mean_vanilla = np.mean(vanilla_accuracies)

    loop_delta_holdout = mean_final - mean_vanilla

    # Compounding rho: Spearman ρ between K and mean-over-seeds holdout accuracy.
    ks: list[int] = []
    mean_accs: list[float] = []
    cp_count = len(seed_results[0]["checkpoints"])
    for ci in range(cp_count):
        ks.append(seed_results[0]["checkpoints"][ci]["K"])
        mean_accs.append(
            float(np.mean([r["checkpoints"][ci]["holdout_accuracy"] for r in seed_results]))
        )

        loop_compounding_rho = _spearman_rho(ks, mean_accs)
    else:
        loop_compounding_rho = float("nan")

    # Validity gate.
    vanilla_valid = mean_vanilla < 0.5

    # Per-checkpoint aggregate table.
    cp_aggregates: list[dict[str, Any]] = []
    for ci in range(cp_count):
        accs = [r["checkpoints"][ci]["holdout_accuracy"] for r in seed_results]
        drifts = [r["checkpoints"][ci]["cold_drift"] for r in seed_results]
        cvec_vals = [r["checkpoints"][ci]["cvec_norm"] for r in seed_results]
        mems = [r["checkpoints"][ci]["memory_bytes"] for r in seed_results]

        cp_aggregates.append(
            {
                "K": seed_results[0]["checkpoints"][ci]["K"],
                "holdout_accuracy": summarize(accs),
                "cold_drift": summarize(drifts),
                "cvec_norm": summarize(cvec_vals),
                "memory_bytes": summarize(mems),
            }
        )

    return {
        "seeds": len(seed_results),
        "loop_delta_holdout": loop_delta_holdout,
        "loop_compounding_rho": loop_compounding_rho,
        "vanilla_holdout_accuracy": mean_vanilla,
        "vanilla_holdout_valid": vanilla_valid,
        "final_holdout_accuracy": summarize(final_accuracies),
        "checkpoint_aggregates": cp_aggregates,
        "seed_results": seed_results,
        "K_trajectory": {"K": ks, "mean_holdout_accuracy": mean_accs},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S2.1: Minimal metabolism loop on the HF substrate",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of independent seeds (default: 5)",
    )
    parser.add_argument(
        "--stage", type=str, default="stage_0_grounding",
        help="Curriculum stage to run (default: stage_0_grounding)",
    )
    parser.add_argument(
        "--checkpoints", type=str, default="0,1,2,4,-1",
        help="Checkpoint K values, comma-separated, -1 = all (default: 0,1,2,4,-1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print setup info and exit without running",
    )
    args = parser.parse_args(argv)

    # Parse checkpoints.
    checkpoints_raw = [int(x.strip()) for x in args.checkpoints.split(",")]
    checkpoints = checkpoints_raw  # [0, 1, 2, 4, -1]

    # Verify manifest integrity.
    print("Verifying eval manifest ...")
    try:
        verify_manifest()
    except Exception as e:
        print(f"MANIFEST VERIFICATION FAILED: {e}")
        print("Set EVAL_CHANGE_APPROVED=1 to bypass.")
        return 1
    print("  OK\n")

    # Load stage.
    print(f"Loading stage: {args.stage}")
    (stage,) = build_curriculum(stage_names=(args.stage,))
    _, holdout_ids = split_probes(stage, fraction=0.3, salt="v2")
    n_episodes = len(stage.episodes)
    n_holdout = len(holdout_ids)
    checkpoint_values = [c if c >= 0 else n_episodes for c in checkpoints]

    print(f"  Episodes: {n_episodes}")
    print(f"  Holdout probes: {n_holdout}")
    print(f"  Checkpoints (K): {checkpoint_values}")
    print(f"  Seeds: {args.seeds}\n")

    if args.dry_run:
        print("DRY RUN — exiting.")
        return 0

    # Verify manifest again post-run (spec requirement).
    verify_manifest()

    # Run experiment.
    seeds = list(range(args.seeds))
    print(f"Running experiment with {args.seeds} seeds ...\n")

    t_start = time.monotonic()
    results = _run_experiment(seeds, stage, holdout_ids, checkpoints)
    total_elapsed = time.monotonic() - t_start

    # Verify manifest post-run.
    verify_manifest()

    # Print results.
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    verdict = "ACCEPT" if (
        results["vanilla_holdout_valid"]
        and results["loop_delta_holdout"] > 0
        and results["loop_compounding_rho"] >= 0.6
    ) else (
        "INVALID" if not results["vanilla_holdout_valid"] else "REFUTE"
    )

    print(f"Verdict: {verdict}")
    print()
    print("PRIMARY METRICS:")
    print(f"  loop_delta_holdout     = {results['loop_delta_holdout']:.4f}")
    print(f"  loop_compounding_rho   = {results['loop_compounding_rho']:.4f}")
    print(f"  vanilla_holdout_acc    = {results['vanilla_holdout_accuracy']:.4f}")
    print(f"  vanilla_holdout_valid  = {results['vanilla_holdout_valid']}")
    print(f"  final_holdout_accuracy = {format_row('holdout', results['final_holdout_accuracy'])}")
    print()

    # Per-checkpoint table.
    print("Per-checkpoint holdout accuracy (mean ± 95% CI):")
    for cp in results["checkpoint_aggregates"]:
        print(f"  K={cp['K']:3d}: {format_row('holdout', cp['holdout_accuracy'])}")
    print()

    print(f"K trajectory: {results['K_trajectory']['K']}")
    print(f"Mean accs:    {[f'{a:.4f}' for a in results['K_trajectory']['mean_holdout_accuracy']]}")
    print()
    print(f"Total wall clock: {total_elapsed:.0f}s")

    # Per-seed detail.
    print("\nPer-seed final holdout accuracy:")
    for r in results["seed_results"]:
        print(f"  seed={r['seed']}: {r['checkpoints'][-1]['holdout_accuracy']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
