"""Minimal metabolism organism with KV content channel (Sprint 2 / S2.2).

Implements the pre-registered experiment research/12: replacing the
S2.1 consolidated articulation prefix with KV entries.

Organism: HFDriver + KVCortex (warm/cold fast-weight) + NeuralHippocampus
(replay only, never queried at answer time). The content_channel parameter
controls how consolidated facts are delivered to the LM:

- "prefix": set_articulation_prefix(text) — the C1 condition.
- "kv": encode_kv(text) → generate_with_kv at answer time — the C2 condition.
- "vanilla": no organism at all — bare HFDriver — the C0 condition.

Usage:
    uv run python -m oczy.experiments.minimal_loop_kv --seeds 5 --stage 0
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.v2 import verify_manifest
from neural_hippocampus import NeuralHippocampus
from oczy.common.stats import summarize
from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    Stage,
    load_stage,
    split_probes,
)
from oczy.lm.hf_driver import HFDriver, KVHandle
from oczy.lm.hf_model_choice import HF_MODEL_ID
from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROBE_TEMPLATE = "\n\nRecall the answer in lowercase. Question: {}\nAnswer:"
_MAX_PREFIX_TOKENS = 48

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregate results for one (seed, condition, checkpoint)."""

    seed: int
    condition: str  # "C0", "C1", "C2"
    checkpoint: int  # K value
    holdout_accuracy: float
    holdout_correct: int
    holdout_total: int
    memory_bytes: int = 0
    prefix_token_count: int = 0
    latency_sec: float = 0.0
    kv_handle_bytes: int = 0


# ---------------------------------------------------------------------------
# Minimal organism
# ---------------------------------------------------------------------------


class MinimalOrganismKV:
    """Minimal organism with configurable content channel.

    content_channel:
        - "prefix": C1 — consolidated facts set as articulation prefix.
        - "kv": C2 — consolidated facts encoded as KV entries.
        - "vanilla": C0 — bare driver, no organism state (use standalone).
    """

    def __init__(
        self,
        content_channel: str,
        seed: int = 0,
        driver: HFDriver | None = None,
    ) -> None:
        if content_channel not in ("prefix", "kv"):
            raise ValueError(
                f"content_channel must be 'prefix' or 'kv', got {content_channel!r}"
            )
        self.content_channel = content_channel

        self.driver = driver or HFDriver.load(HF_MODEL_ID)

        # Fast-weight cortex: warm/cold states with Hebbian observe/consolidate.
        self.cortex = KVCortex(
            KVCortexConfig(
                d_embd=self.driver.n_embd,
                n_layers=self.driver.n_layers,
                d_cortex=64,
                seed=seed,
            )
        )

        # Neural hippocampus: consolidation-time replay ONLY, never queried
        # at answer time. replay_threshold=1 ensures all stored traces are
        # eligible for consolidation after one reinforce call.
        self.hippocampus = NeuralHippocampus(
            config={"replay_threshold": 1}
        )

        # Cached KV handle (C2 only).
        self.kv_handle: KVHandle | None = None

        # Tracked corrections for prefix building.
        self._tracked_corrections: list[dict[str, str]] = []
        self.episodes_taught: int = 0
        self._consolidated: bool = False

    def _extract_ambiguous(self, correction: str) -> str:
        """Extract the ambiguous word from a correction utterance."""
        m = re.search(r"'(\w+)'", correction)
        if m:
            return m.group(1)
        return ""

    def perceive(self, episode: dict[str, Any]) -> None:
        """Process one correction episode through the organism.

        Observes the correction through the cortex (correction_signal=1.0),
        stores the episode in the hippocampus, and tracks the corrected label
        for prefix building.
        """
        correction = str(episode.get("correction_utterance", ""))
        if not correction:
            return

        hidden = self.driver.peek_embedding(correction)

        # Cortex observation: high correction signal.
        self.cortex.observe(hidden, correction_signal=1.0)

        # Store in hippocampus with hidden vector for replay.
        self.hippocampus.store(
            query=str(episode.get("initial_request", "")),
            answer=str(episode.get("default_response", "")),
            correction=correction,
            prediction_error=0.8,
            corrected_answer=str(episode.get("corrected_label", "")),
            hidden=hidden,
        )

        # Force one replay to increment replay_count so hippocampus.consolidate()
        # picks up the trace (replay_threshold=1).
        self.hippocampus.reinforce(
            str(episode.get("initial_request", "")), k=3
        )

        # Track for prefix content building.
        ambiguous = self._extract_ambiguous(correction)
        label = str(episode.get("corrected_label", ""))
        if ambiguous and label:
            self._tracked_corrections.append(
                {"ambiguous": ambiguous, "label": label}
            )

        self.episodes_taught += 1

    def consolidate(self) -> str:
        """Consolidate hippocampus replays → cortex cold state → content channel.

        Returns the compiled content text (the prefix or KV-encoded text).
        """
        # Hippocampus consolidation: cluster traces, produce summaries.
        summaries = self.hippocampus.consolidate()

        # Build replay hidden vectors from summaries.
        replays: list[np.ndarray] = []
        for s in summaries:
            hidden = s.get("representative_hidden")
            if (
                isinstance(hidden, np.ndarray)
                and hidden.ndim == 1
                and hidden.shape[0] > 0
            ):
                replays.append(hidden.copy())

        # Cortex consolidation: warm → cold with replay absorption.
        self.cortex.consolidate(replays=replays if replays else None)

        # Build content text from tracked corrections.
        content_text = self._build_content_text()

        if self.content_channel == "prefix":
            if content_text:
                self.driver.set_articulation_prefix(content_text)
            else:
                self.driver.clear_articulation_prefix()
        elif self.content_channel == "kv":
            if content_text:
                self.kv_handle = self.driver.encode_kv(content_text)
            else:
                self.kv_handle = None

        self._consolidated = True
        return content_text

    def _build_content_text(self) -> str:
        """Compile tracked corrections into a ≤48-token text.

        Format: "log: captain's journal. file: submit officially. key: map legend."
        Oldest entries are dropped when the token budget overflows.
        """
        if not self._tracked_corrections:
            return ""

        parts = []
        for c in self._tracked_corrections:
            parts.append(f"{c['ambiguous']}: {c['label']}.")

        # Fit within token budget, dropping oldest first.
        while parts:
            text = " ".join(parts)
            tokens = self.driver._tokenize(text)  # type: ignore[attr-defined]
            if tokens.shape[1] <= _MAX_PREFIX_TOKENS:
                return text
            parts.pop(0)

        return ""

    def answer(self, query: str) -> str:
        """Generate an answer for a probe query.

        Answers are always generated with freshly-cleared cvec state, so no
        residual steering from previous answers leaks across probes.
        """
        prompt = _PROBE_TEMPLATE.format(query)

        # Apply cortex cvecs regardless of channel — the cortex state
        # (warm/cold) is identical across C1/C2.
        self.driver.clear_cvec()
        cvecs = self.cortex.emit_all_cvecs()
        self.driver.set_cvecs_per_layer(cvecs)

        if self.content_channel == "kv" and self.kv_handle is not None:
            answer = self.driver.generate_with_kv(
                prompt, self.kv_handle, max_tokens=32
            )
        else:
            answer = self.driver.generate(prompt, max_tokens=32)

        self.driver.clear_cvec()
        return answer

    def prompt_token_count(self, query: str) -> int:
        """Count visible prompt tokens for a probe query (audit)."""
        prompt = _PROBE_TEMPLATE.format(query)
        if self.content_channel == "prefix" and self.driver.articulation_prefix:
            effective = self.driver._apply_reserved_prefix(prompt)  # type: ignore[attr-defined]
        else:
            effective = prompt
        return int(self.driver._tokenize(effective).shape[1])  # type: ignore[attr-defined]

    def visible_prompt_tokens(self, query: str) -> int:
        """Alias for prompt_token_count (public API)."""
        return self.prompt_token_count(query)

    def memory_bytes(self) -> int:
        """Total memory in bytes across all organs."""
        total = 0
        # Cortex: cold_state, warm_state, proj_hidden, proj_c
        total += self.cortex.cold_state.nbytes
        total += self.cortex.warm_state.nbytes
        if hasattr(self.cortex, "proj_hidden"):
            total += self.cortex.proj_hidden.nbytes
        if hasattr(self.cortex, "proj_c") and self.cortex.proj_c is not None:
            total += self.cortex.proj_c.nbytes
        # Hippocampus size.
        status = self.hippocampus.status(include_size=True)
        total += status.get("trace_bytes", 0)
        # KV handle bytes (C2 only).
        if self.kv_handle is not None:
            for layer_kv in self.kv_handle.past_key_values:
                for t in layer_kv:
                    if t is not None:
                        total += t.element_size() * t.numel()
        return total

    def kv_handle_bytes(self) -> int:
        """Size of the KV handle in bytes (C2 only)."""
        if self.kv_handle is None:
            return 0
        total = 0
        for layer_kv in self.kv_handle.past_key_values:
            for t in layer_kv:
                if t is not None:
                    total += t.element_size() * t.numel()
        return total

    def consolidated_prefix_token_count(self) -> int:
        """Return the number of tokens in the current articulation prefix (C1 only).

        Returns 0 for C2 (kv channel never sets a prefix).
        """
        if self.content_channel == "prefix" and self.driver.articulation_prefix:
            return int(
                self.driver._tokenize(self.driver.articulation_prefix).shape[1]  # type: ignore[attr-defined]
            )
        return 0

    def close(self) -> None:
        """Release driver resources."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _load_stage_and_split(stage_name: str) -> tuple[Stage, set[str], set[str]]:
    """Load a frozen stage JSON and partition probes into dev/holdout.

    The stage is loaded from eval/v2/<stage_name>.json.
    """
    data_dir = Path(__file__).resolve().parents[3] / "eval" / "v2"
    stage_path = data_dir / f"{stage_name}.json"
    stage = load_stage(stage_path)
    _, holdout = split_probes(stage, fraction=0.3, salt="v2")
    return stage, _, holdout


def _episode_to_dict(ep: Episode) -> dict[str, Any]:
    """Convert a frozen Episode to a mutable dict for the organism API."""
    return {
        "initial_request": ep.initial_request,
        "default_response": ep.default_response,
        "correction_utterance": ep.correction_utterance,
        "corrected_label": ep.corrected_label,
        "corrected_response": ep.corrected_response,
        "domain": ep.domain,
    }


def _score_probe(
    organism: MinimalOrganismKV | None,
    probe: Probe,
    episode: Episode,
    vanilla_driver: HFDriver | None = None,
) -> bool:
    """Score one probe against the organism (or vanilla driver).

    For C0 (organism is None), a single vanilla_driver is reused across all
    probes rather than creating one per probe.
    """
    if organism is None:
        # C0: vanilla — use the provided driver.
        assert vanilla_driver is not None, "vanilla_driver required for C0"
        prompt = _PROBE_TEMPLATE.format(probe.request)
        answer = vanilla_driver.generate(prompt, max_tokens=32)
    else:
        answer = organism.answer(probe.request)

    return probe_matches(answer, probe, episode)


def _vanilla_prompt_token_count(query: str) -> int:
    """Count prompt tokens for a vanilla (C0) probe."""
    driver = HFDriver.load(HF_MODEL_ID)
    try:
        prompt = _PROBE_TEMPLATE.format(query)
        return int(driver._tokenize(prompt).shape[1])  # type: ignore[attr-defined]
    finally:
        driver.close()


def _run_organism_seed(
    seed: int,
    stage: Stage,
    holdout_ids: set[str],
    content_channel: str | None,
) -> list[RunResult]:
    """Run one seed of the organism experiment.

    If content_channel is None, runs C0 (vanilla — no organism).
    Otherwise creates a MinimalOrganismKV with the given channel.
    """
    rng = np.random.RandomState(seed)

    # Collect all episodes and shuffle by seed.
    episodes = list(stage.episodes)


    # Collect holdout probes with their episodes (needed for scoring).
    holdout_probes: list[tuple[str, Probe, Episode]] = []
    for ep in episodes:
        for probe in ep.probes:
            probe_id = f"{ep.id}|{probe.request}|{probe.category}"
            if probe_id in holdout_ids:
                holdout_probes.append((probe_id, probe, ep))

    # Determine K checkpoints: K ∈ {0, 1, 2, 4, N}
    N = len(episodes)
    checkpoint_Ks = sorted(set([0, 1, 2, 4, N]))
    checkpoint_Ks = [k for k in checkpoint_Ks if k <= N]

    # Shuffle episode order for this seed.
    indices = list(range(N))
    rng.shuffle(indices)

    # Vanilla (C0) — no organism, just score holdout probes once.
    if content_channel is None:
        results: list[RunResult] = []
        correct = 0
        total = len(holdout_probes)
        t0 = time.monotonic()
        # Create one driver for all C0 probes — not one per probe.
        vanilla = HFDriver.load(HF_MODEL_ID)
        try:
            for _, probe, ep in holdout_probes:
                ok = _score_probe(None, probe, ep, vanilla_driver=vanilla)
                if ok:
                    correct += 1
        finally:
            vanilla.close()
        elapsed = time.monotonic() - t0
        results.append(
            RunResult(
                seed=seed,
                condition="C0",
                checkpoint=N,
                holdout_accuracy=correct / total if total else 0.0,
                holdout_correct=correct,
                holdout_total=total,
                latency_sec=elapsed,
            )
        )
        return results

    # For C1/C2: create organism and evaluate at each checkpoint.
    organism = MinimalOrganismKV(content_channel=content_channel, seed=seed)

    try:
        results: list[RunResult] = []
        taught_count = 0  # episodes already taught (cumulative)

        for K in checkpoint_Ks:
            # Teach only new episodes since last checkpoint (cumulative).
            new_episodes = indices[taught_count:K]
            for idx in new_episodes:
                ep = episodes[idx]
                organism.perceive(_episode_to_dict(ep))
            taught_count = K
            organism.consolidate()

            # Score holdout probes.
            correct = 0
            total = len(holdout_probes)
            t0 = time.monotonic()
            for _, probe, ep in holdout_probes:
                ok = _score_probe(organism, probe, ep)
                if ok:
                    correct += 1
            elapsed = time.monotonic() - t0

            prefix_tokens = organism.consolidated_prefix_token_count()

            results.append(
                RunResult(
                    seed=seed,
                    condition=("C1" if content_channel == "prefix" else "C2"),
                    checkpoint=K,
                    holdout_accuracy=correct / total if total else 0.0,
                    holdout_correct=correct,
                    holdout_total=total,
                    memory_bytes=organism.memory_bytes(),
                    prefix_token_count=prefix_tokens,
                    latency_sec=elapsed,
                    kv_handle_bytes=(
                        organism.kv_handle_bytes()
                        if content_channel == "kv"
                        else 0
                    ),
                )
            )

        return results

    finally:
        organism.close()


# ---------------------------------------------------------------------------
# Prompt-token audit
# ---------------------------------------------------------------------------


def _audit_prompt_tokens(
    organism: MinimalOrganismKV,
    holdout_probes: list[tuple[str, Probe, Episode]],
) -> dict[str, Any]:
    """Verify C2 visible prompt tokens match C0 for every probe."""
    mismatches: list[dict[str, Any]] = []
    c2_counts: list[int] = []
    c0_counts: list[int] = []

    for _, probe, _ep in holdout_probes:
        c0 = _vanilla_prompt_token_count(probe.request)
        c2 = organism.prompt_token_count(probe.request)
        c0_counts.append(c0)
        c2_counts.append(c2)
        if c0 != c2:
            mismatches.append({
                "probe": probe.request,
                "c0_tokens": c0,
                "c2_tokens": c2,
            })

    return {
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "c0_mean": float(np.mean(c0_counts)) if c0_counts else 0.0,
        "c2_mean": float(np.mean(c2_counts)) if c2_counts else 0.0,
    }


# ---------------------------------------------------------------------------
# Rank-table continuity check (lane_02 style)
# ---------------------------------------------------------------------------


def _rank_table_check(
    organism_c1: MinimalOrganismKV,
    organism_c2: MinimalOrganismKV,
) -> list[dict[str, Any]]:
    """Compare C1 vs C2 target-token ranks at the probe blank.

    Uses the lane_02 facts for continuity against S1.3.
    """
    FACTS = [
        {
            "query": "What is the secret passphrase for level 7?",
            "target": "marmalade",
        },
        {
            "query": "What is the skylark protocol?",
            "target": "skylark",
        },
        {
            "query": "What chess piece is the secret key?",
            "target": "rook",
        },
    ]

    results: list[dict[str, Any]] = []
    for fact in FACTS:
        prompt = _PROBE_TEMPLATE.format(fact["query"])
        c1_ranks = organism_c1.driver.token_ranks(prompt, [fact["target"]])
        c2_ranks = organism_c2.driver.token_ranks_with_kv(
            prompt, organism_c2.kv_handle, [fact["target"]]
        ) if organism_c2.kv_handle is not None else []
        results.append({
            "query": fact["query"],
            "target": fact["target"],
            "c1_rank": c1_ranks[0]["rank"] if c1_ranks else None,
            "c2_rank": c2_ranks[0]["rank"] if c2_ranks else None,
        })
    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _run_single_condition(
    seeds: list[int],
    stage: Stage,
    holdout_ids: set[str],
    content_channel: str | None,
    condition_label: str,
) -> tuple[dict[int, list[RunResult]], dict[str, Any] | None]:
    """Run all seeds for one condition (C0/C1/C2).

    Returns (seed_results, audit) where audit is only populated for C2.
    """
    seed_results: dict[int, list[RunResult]] = {}
    audit: dict[str, Any] | None = None

    for seed in seeds:
        print(
            f"  [{condition_label}] seed={seed} ...",
            end=" ",
            flush=True,
            file=sys.stderr,
        )
        results = _run_organism_seed(seed, stage, holdout_ids, content_channel)
        seed_results[seed] = results

        # Run prompt-token audit for C2.
        if content_channel == "kv" and audit is None:
            # Run a fresh organism at K=N for the audit.
            org = MinimalOrganismKV(content_channel="kv", seed=seed)
            try:
                N = len(stage.episodes)
                rng = np.random.RandomState(seed)
                indices = list(range(N))
                rng.shuffle(indices)
                for idx in indices:
                    ep = stage.episodes[idx]
                    org.perceive(_episode_to_dict(ep))
                org.consolidate()

                holdout_probes: list[tuple[str, Probe, Episode]] = []
                for ep in stage.episodes:
                    for probe in ep.probes:
                        pid = f"{ep.id}|{probe.request}|{probe.category}"
                        if pid in holdout_ids:
                            holdout_probes.append((pid, probe, ep))
                audit = _audit_prompt_tokens(org, holdout_probes)
            finally:
                org.close()

        print("done", file=sys.stderr)

    return seed_results, audit


def _extract_at_k(
    seed_results: dict[int, list[RunResult]],
    K: int,
) -> list[float]:
    """Extract holdout accuracy at checkpoint K across seeds."""
    accs: list[float] = []
    for _seed, results in seed_results.items():
        for r in results:
            if r.checkpoint == K:
                accs.append(r.holdout_accuracy)
                break
    return accs


def run_experiment(
    seeds: list[int],
    stage_name: str = "stage_0_grounding",
) -> dict[str, Any]:
    """Run the full S2.2 experiment: C0 + C1 + C2.

    Returns a dict of results suitable for reporting.
    """
    stage, _, holdout_ids = _load_stage_and_split(stage_name)
    N = len(stage.episodes)
    K = N  # Use K=N for primary metrics.

    print(f"Stage: {stage_name}  episodes={N}  holdout_probes={len(holdout_ids)}",
          file=sys.stderr)
    print(f"Seeds: {seeds}  K={K}", file=sys.stderr)


    # --- C0: vanilla ---
    print("\nC0 (vanilla):", file=sys.stderr)
    c0_results, _ = _run_single_condition(seeds, stage, holdout_ids, None, "C0")
    c0_accs = _extract_at_k(c0_results, K)

    # --- C1: prefix ---
    print("\nC1 (prefix):", file=sys.stderr)
    c1_results, _ = _run_single_condition(seeds, stage, holdout_ids, "prefix", "C1")
    c1_accs = _extract_at_k(c1_results, K)

    # --- C2: KV ---
    print("\nC2 (kv):", file=sys.stderr)
    c2_results, c2_audit = _run_single_condition(seeds, stage, holdout_ids, "kv", "C2")
    c2_accs = _extract_at_k(c2_results, K)

    # --- Compute primary metrics ---
    kv_effect_deltas = [c2 - c0 for c2, c0 in zip(c2_accs, c0_accs, strict=True)]
    kv_parity_deltas = [c2 - c1 for c2, c1 in zip(c2_accs, c1_accs, strict=True)]

    c0_summary = summarize(c0_accs)
    c1_summary = summarize(c1_accs)
    c2_summary = summarize(c2_accs)
    kv_effect_summary = summarize(kv_effect_deltas)
    kv_parity_summary = summarize(kv_parity_deltas)

    # --- Secondaries: latency ---
    c0_latencies: list[float] = []
    c1_latencies: list[float] = []
    c2_latencies: list[float] = []
    for results in c0_results.values():
        for r in results:
            if r.checkpoint == K:
                c0_latencies.append(r.latency_sec)
    for results in c1_results.values():
        for r in results:
            if r.checkpoint == K:
                c1_latencies.append(r.latency_sec)
    for results in c2_results.values():
        for r in results:
            if r.checkpoint == K:
                c2_latencies.append(r.latency_sec)

    # --- Secondaries: KV bytes vs prefix bytes ---
    c1_memory = []
    c2_memory = []
    c2_kv_bytes = []
    for results in c1_results.values():
        for r in results:
            if r.checkpoint == K:
                c1_memory.append(r.memory_bytes)
    for results in c2_results.values():
        for r in results:
            if r.checkpoint == K:
                c2_memory.append(r.memory_bytes)
                if r.kv_handle_bytes > 0:
                    c2_kv_bytes.append(r.kv_handle_bytes)

    # --- Rank-table check ---
    rank_table: list[dict[str, Any]] = []
    try:
        # Build fresh C1 and C2 organisms for rank table.
        o_c1 = MinimalOrganismKV(content_channel="prefix", seed=0)
        o_c2 = MinimalOrganismKV(content_channel="kv", seed=0)
        try:
            rng = np.random.RandomState(0)
            indices = list(range(N))
            rng.shuffle(indices)
            for idx in indices:
                ep = stage.episodes[idx]
                d = _episode_to_dict(ep)
                o_c1.perceive(d)
                o_c2.perceive(d)
            o_c1.consolidate()
            o_c2.consolidate()
            rank_table = _rank_table_check(o_c1, o_c2)
        finally:
            o_c1.close()
            o_c2.close()
    except Exception as exc:
        rank_table = [{"error": str(exc)}]

    return {
        "stage": stage_name,
        "N": N,
        "K": K,
        "seeds": seeds,
        "holdout_probe_count": len(holdout_ids),
        "c0_accuracy": c0_summary,
        "c1_accuracy": c1_summary,
        "c2_accuracy": c2_summary,
        "c0_accs": c0_accs,
        "c1_accs": c1_accs,
        "c2_accs": c2_accs,
        "kv_effect_delta": kv_effect_summary,
        "kv_effect_deltas": kv_effect_deltas,
        "kv_parity_delta": kv_parity_summary,
        "kv_parity_deltas": kv_parity_deltas,
        "c0_latency": summarize(c0_latencies) if c0_latencies else {},
        "c1_latency": summarize(c1_latencies) if c1_latencies else {},
        "c2_latency": summarize(c2_latencies) if c2_latencies else {},
        "c1_memory_bytes": summarize(c1_memory) if c1_memory else {},
        "c2_memory_bytes": summarize(c2_memory) if c2_memory else {},
        "c2_kv_bytes": summarize(c2_kv_bytes) if c2_kv_bytes else {},
        "c2_audit": c2_audit,
        "rank_table": rank_table,
        "model_id": HF_MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(result: dict[str, Any]) -> str:
    """Produce a markdown experiment log."""
    lines: list[str] = []
    lines.append("# S2.2: KV content channel — experiment log")
    lines.append("")
    lines.append("**Date:** 2026-07-02")
    lines.append(f"**Model:** `{result['model_id']}`")
    lines.append(f"**Stage:** `{result['stage']}`")
    lines.append(f"**N episodes:** {result['N']}")
    lines.append(f"**K (primary):** {result['K']}")
    lines.append(f"**Seeds:** {result['seeds']}")
    lines.append(f"**Holdout probes:** {result['holdout_probe_count']}")
    lines.append("")

    # Primary metrics
    lines.append("## Primary metrics (K=N)")
    lines.append("")
    c0 = result["c0_accuracy"]
    c1 = result["c1_accuracy"]
    c2 = result["c2_accuracy"]
    ke = result["kv_effect_delta"]
    kp = result["kv_parity_delta"]

    lines.append("| Condition | Accuracy |")
    lines.append("|-----------|----------|")
    lines.append(
        f"| C0 (vanilla) | {c0['mean']:.4f} ± {c0['ci95_half']:.4f} (n={c0['n']}) |"
    )
    lines.append(
        f"| C1 (prefix)  | {c1['mean']:.4f} ± {c1['ci95_half']:.4f} (n={c1['n']}) |"
    )
    lines.append(
        f"| C2 (kv)      | {c2['mean']:.4f} ± {c2['ci95_half']:.4f} (n={c2['n']}) |"
    )
    lines.append("")
    lines.append(
        f"**kv_effect_delta:** {ke['mean']:.4f} ± {ke['ci95_half']:.4f} (n={ke['n']})"
    )
    lines.append(
        f"**kv_parity_delta:** {kp['mean']:.4f} ± {kp['ci95_half']:.4f} (n={kp['n']})"
    )
    lines.append("")

    # Verdict
    effect_pos = ke["mean"] > 0 and ke["mean"] - ke["ci95_half"] > 0
    parity_ok = kp["mean"] >= -0.05

    lines.append("## Verdict")
    lines.append("")
    if effect_pos and parity_ok:
        lines.append("**ACCEPT H-KVCONTENT** — KV matches or exceeds prefix at zero visible-token cost.")
    elif not effect_pos:
        lines.append("**REFUTE H-KVCONTENT** — KV slot shows no positive effect over vanilla.")
    elif not parity_ok:
        lines.append("**REFUTE H-KVCONTENT** — KV slot fails non-inferiority vs prefix.")
    lines.append("")

    # Per-seed table
    lines.append("## Per-seed holdout accuracy (K=N)")
    lines.append("")
    lines.append("| Seed | C0 | C1 | C2 | kv_effect | kv_parity |")
    lines.append("|------|----|----|----|-----------|-----------|")
    for i, s in enumerate(result["seeds"]):
        lines.append(
            f"| {s} | {result['c0_accs'][i]:.4f} | "
            f"{result['c1_accs'][i]:.4f} | "
            f"{result['c2_accs'][i]:.4f} | "
            f"{result['kv_effect_deltas'][i]:+.4f} | "
            f"{result['kv_parity_deltas'][i]:+.4f} |"
        )
    lines.append("")

    # Prompt-token audit
    audit = result.get("c2_audit")
    if audit:
        lines.append("## Prompt-token audit (C2 vs C0)")
        lines.append("")
        lines.append(f"**Passed:** {audit['passed']}")
        lines.append(f"C0 mean tokens: {audit['c0_mean']:.1f}")
        lines.append(f"C2 mean tokens: {audit['c2_mean']:.1f}")
        if audit["mismatches"]:
            lines.append("")
            lines.append("| Probe | C0 tokens | C2 tokens |")
            lines.append("|-------|-----------|-----------|")
            for m in audit["mismatches"]:
                lines.append(
                    f"| {m['probe']} | {m['c0_tokens']} | {m['c2_tokens']} |"
                )
        lines.append("")

    # Secondaries
    c0_lat = result.get("c0_latency", {})
    c1_lat = result.get("c1_latency", {})
    c2_lat = result.get("c2_latency", {})
    lines.append("## Per-answer latency (s, mean ± CI)")
    lines.append("")
    def _fmt_lat(d: dict[str, float]) -> str:
        if "mean" not in d:
            return "NA"
        mean = d["mean"]
        ci = d.get("ci95_half", float("nan"))
        return f"{mean:.3f} ± {ci}" if ci == ci else f"{mean:.3f}"
    lines.append(f"- C0: {_fmt_lat(c0_lat)}")
    lines.append(f"- C1: {_fmt_lat(c1_lat)}")
    lines.append(f"- C2: {_fmt_lat(c2_lat)}")
    lines.append("")

    # Memory
    c1_mem = result.get("c1_memory_bytes", {})
    c2_mem = result.get("c2_memory_bytes", {})
    c2_kvb = result.get("c2_kv_bytes", {})
    lines.append("## Memory usage (bytes)")
    lines.append("")
    lines.append(
        f"- C1 organ memory: {c1_mem.get('mean', 0):.0f} ± {c1_mem.get('ci95_half', 0):.0f}"
    )
    lines.append(
        f"- C2 organ memory: {c2_mem.get('mean', 0):.0f} ± {c2_mem.get('ci95_half', 0):.0f}"
    )
    if c2_kvb.get('mean', 0) > 0:
        lines.append(
            f"- C2 KV handle bytes: {c2_kvb.get('mean', 0):.0f} ± {c2_kvb.get('ci95_half', 0):.0f}"
        )
    lines.append("")

    # Rank table
    rt = result.get("rank_table", [])
    if rt:
        lines.append("## Rank table (lane_02 facts, C1 vs C2)")
        lines.append("")
        if "error" in rt[0]:
            lines.append(f"*Error: {rt[0]['error']}*")
        else:
            lines.append("| Query | Target | C1 rank | C2 rank |")
            lines.append("|-------|--------|---------|---------|")
            for row in rt:
                lines.append(
                    f"| {row['query']} | {row['target']} | "
                    f"{row['c1_rank']} | {row['c2_rank']} |"
                )
        lines.append("")

    lines.append("**Spec:** research/12-s2-kv-slot-content-path.md")
    lines.append("**Commands:**")
    lines.append("```")
    lines.append(f"uv run python -m oczy.experiments.minimal_loop_kv --seeds {len(result['seeds'])} --stage 0")
    lines.append("```")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S2.2: KV content channel experiment"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds (default 5, fallback 3 if too slow).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage_0_grounding",
        help="Stage name (default stage_0_grounding).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Single-seed dry run to estimate wall-clock time.",
    )
    args = parser.parse_args(argv)

    verify_manifest()

    if args.dry_run:
        print("Dry run: 1 seed for timing estimation...", file=sys.stderr)
        seeds = [0]
    else:
        seeds = list(range(args.seeds))

    result = run_experiment(seeds=seeds, stage_name=args.stage)
    # Verify manifest integrity again after the run (spec requirement).
    verify_manifest()

    report = _format_report(result)
    print(report)

    # Write log
    log_path = Path("experiments_logs/2026-07-02_s2_2_kv_content_path.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report)
    print(f"\nLog written to {log_path}", file=sys.stderr)

    # Terminal summary
    ke = result["kv_effect_delta"]
    kp = result["kv_parity_delta"]
    verdict = "ACCEPT" if (
        ke["mean"] > 0 and ke["mean"] - ke["ci95_half"] > 0
        and kp["mean"] >= -0.05
    ) else "REFUTE"
    print(
        f"\nS2.2 complete. {verdict} H-KVCONTENT. "
        f"kv_effect={ke['mean']:.4f}±{ke['ci95_half']:.4f} "
        f"kv_parity={kp['mean']:.4f}±{kp['ci95_half']:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
