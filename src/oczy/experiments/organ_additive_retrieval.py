"""S3.M2a — Additive retrieval ablation harness.

Subclasses :class:`MinimalOrganism` to add three retrieval components behind
independent constructor flags:

* ``use_hippocampus_at_answer`` — query the raw hippocampus at answer time
  (bypassing the guard) and prepend a ``[Recall: ...]`` hint.
* ``use_dsi_fact_index`` — maintain a :class:`DifferentiableFactIndex` and
  prepend a ``[Fact: ...]`` hint from the top retrieved label.
* ``use_scope_slot_reranker`` — store per-episode slot keys and prepend a
  ``[Scope: ...]`` hint from cosine-similar slots above threshold.

All three flags compose simultaneously.  The runner functions sweep a
condition × stage × seed matrix and return nested result dicts.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any

import numpy as np

from oczy.common.stats import format_row, summarize
from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.minimal_loop import (
    MinimalOrganism,
    _HippocampusGuard,
    CHECKPOINTS,
    _run_one_seed as _minimal_run_one_seed,
    _run_experiment as _minimal_run_experiment,
)
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm.hf_driver import HFDriver
from oczy.experiments.scope_selectivity_stressor import _cosine, _RETRIEVE_THRESHOLD
from oczy.experiments.differentiable_fact_index import DifferentiableFactIndex


# ---------------------------------------------------------------------------
# RetrievalOrganism
# ---------------------------------------------------------------------------


class RetrievalOrganism(MinimalOrganism):
    """MinimalOrganism + additive retrieval components behind constructor flags.

    Each flag independently controls one retrieval mechanism.  When all flags
    are ``False`` (BASE), behaviour is identical to :class:`MinimalOrganism`.
    """

    def __init__(
        self,
        driver,
        *,
        use_hippocampus_at_answer: bool = False,
        use_dsi_fact_index: bool = False,
        use_scope_slot_reranker: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(driver, **kwargs)
        self._use_hippocampus_at_answer = use_hippocampus_at_answer
        self._use_dsi_fact_index = use_dsi_fact_index
        self._use_scope_slot_reranker = use_scope_slot_reranker

        # Always initialise so attribute access is safe regardless of flags.
        self._dsi: DifferentiableFactIndex | None = None
        self._scope_slot_keys: list[np.ndarray] = []
        self._scope_slot_labels: list[str] = []
        self._scope_rerank_topk: int = 3

        if use_dsi_fact_index:
            self._dsi = DifferentiableFactIndex(
                n_facts=64, d_model=driver.n_embd, lora_rank=8
            )

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach(self, episode: Episode) -> None:
        """Teach one episode, then populate active retrieval components."""
        super().teach(episode)

        expected = self._expected_from_utterance(episode.correction_utterance)

        if self._use_dsi_fact_index and self._dsi is not None:
            hidden = self.driver.peek_embedding(
                episode.correction_utterance, last_token_only=True
            )
            self._dsi.store(hidden, expected, is_correction=True)

        if self._use_scope_slot_reranker:
            key = np.asarray(
                self.driver.peek_embedding(
                    episode.initial_request, last_token_only=False
                ),
                dtype=np.float32,
            )
            self._scope_slot_keys.append(key)
            self._scope_slot_labels.append(expected)

    # ------------------------------------------------------------------
    # Answer (bypasses the hippocampus guard)
    # ------------------------------------------------------------------

    def answer(self, request: str, max_tokens: int = 32) -> str:
        """Generate an answer with optional retrieval hints prepended.

        Builds a retrieval prefix from active components, then runs the same
        cvec-posture + generate logic as :meth:`MinimalOrganism.answer` but
        on the augmented request and **without** the hippocampus guard (so
        the hippocampus-at-answer flag can touch ``_hippo_raw`` directly).
        """
        hints: list[str] = []

        if self._use_hippocampus_at_answer:
            replays = self._hippo_raw.reinforce(request, k=3)
            if replays:
                hints.append(f"[Recall: {replays[0].get('correction', '')}]")

        if self._use_dsi_fact_index and self._dsi is not None:
            hidden = self.driver.peek_embedding(request, last_token_only=False)
            hits = self._dsi.retrieve(hidden, k=3, use_lora=True)
            if hits:
                hints.append(f"[Fact: {hits[0][0]}]")

        if self._use_scope_slot_reranker:
            key = np.asarray(
                self.driver.peek_embedding(request, last_token_only=False),
                dtype=np.float32,
            )
            sims: list[tuple[str, float]] = []
            for slot_key, slot_label in zip(
                self._scope_slot_keys, self._scope_slot_labels
            ):
                sim = _cosine(key, slot_key)
                if sim >= _RETRIEVE_THRESHOLD and slot_label:
                    sims.append((slot_label, sim))
            if sims:
                sims.sort(key=lambda x: x[1], reverse=True)
                top = [label for label, _ in sims[: self._scope_rerank_topk]]
                hints.append(f"[Scope: {' | '.join(top)}]")

        augmented = "\n".join(hints) + "\n" + request if hints else request

        # Apply cvec posture if enabled (same as MinimalOrganism.answer).
        if self.use_cvec_posture:
            cvecs = self.cortex.emit_all_cvecs()
            combined = float(
                np.sqrt(sum(float(np.sum(v * v)) for v in cvecs))
            )
            if combined > 1.0:
                scale = 1.0 / combined
                cvecs = [v * scale for v in cvecs]
            self.driver.set_cvecs_per_layer(cvecs)
        else:
            self.driver.clear_cvec()
        return self.driver.generate(augmented, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expected_from_utterance(self, utterance: str) -> str:
        """Extract expected answer from a correction utterance.

        Correction format is typically ``'No, <original> means <corrected>.'``
        Simple heuristic: take text after last ``' means '`` or ``' is '``.
        """
        text = utterance.strip()
        for marker in (" means ", " is "):
            if marker in text:
                return text.rsplit(marker, 1)[-1].rstrip(".").strip()
        return text


# ---------------------------------------------------------------------------
# Runner: one (seed, stage, condition) cell
# ---------------------------------------------------------------------------


def run_additive_one_seed(
    seed: int,
    stage: Stage,
    holdout_ids: set[str],
    condition_kwargs: dict[str, Any],
    checkpoints: list[int] | None = None,
) -> dict[str, Any]:
    """Run one seed of one condition.

    Returns a dict with ``holdout_accuracy``, ``vanilla_holdout_accuracy``,
    ``seed``, and ``n_probes``.
    """
    if checkpoints is None:
        checkpoints = CHECKPOINTS
    rng = np.random.RandomState(seed)
    driver = HFDriver.load()
    org = RetrievalOrganism(driver, d_cortex=128, cortex_seed=seed, **condition_kwargs)
    org.boot()

    # Vanilla baseline (cold driver).
    vanilla_driver = HFDriver.load()
    holdout_probes: list[tuple[str, Any, Episode]] = []
    for ep in stage.episodes:
        for probe in ep.probes:
            probe_id = f"{ep.id}|{probe.request}|{probe.category}"
            if probe_id in holdout_ids:
                holdout_probes.append((probe_id, probe, ep))
    vanilla_results = [
        probe_matches(vanilla_driver.generate(probe.request, max_tokens=32), probe, ep)
        for _, probe, ep in holdout_probes
    ]
    vanilla_acc = sum(vanilla_results) / max(len(vanilla_results), 1)
    del vanilla_driver
    gc.collect()

    # Teach all episodes (shuffled for ordering robustness).
    episodes = list(stage.episodes)
    rng.shuffle(episodes)
    for ep in episodes:
        org.teach(ep)

    # Consolidate.
    org.consolidate()

    # Score holdout.
    results = [
        probe_matches(org.answer(probe.request, max_tokens=32), probe, ep)
        for _, probe, ep in holdout_probes
    ]
    holdout_acc = sum(results) / max(len(results), 1)

    del driver
    gc.collect()
    return {
        "seed": seed,
        "holdout_accuracy": holdout_acc,
        "vanilla_holdout_accuracy": vanilla_acc,
        "n_probes": len(holdout_probes),
    }


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def run_additive_ablation(
    stages: list[Stage],
    holdout_splits: dict[str, set[str]],
    seeds: list[int],
    conditions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Run additive matrix: condition × stage × seed.

    Returns ``{condition: {stage_name: [seed_result, ...]}}``.
    """
    if conditions is None:
        conditions = {
            "BASE": {},
            "HIPPOCAMPUS_AT_ANSWER": {"use_hippocampus_at_answer": True},
            "DSI_FACT_INDEX": {"use_dsi_fact_index": True},
            "SCOPE_SLOT_RERANKER": {"use_scope_slot_reranker": True},
        }
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cond_name, kwargs in conditions.items():
        print(f"Condition: {cond_name}")
        results[cond_name] = {}
        for stage in stages:
            stage_results: list[dict[str, Any]] = []
            for seed in seeds:
                print(f"  Stage {stage.name} seed {seed} ... ", end="", flush=True)
                sr = run_additive_one_seed(
                    seed, stage, holdout_splits[stage.name], kwargs
                )
                stage_results.append(sr)
                print(f"acc={sr['holdout_accuracy']:.4f}")
            results[cond_name][stage.name] = stage_results
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S3.M2a: Additive retrieval ablation")
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5])
    args = parser.parse_args(argv)

    stage_names = args.stages or None
    stages = build_curriculum(stage_names=stage_names)
    print(f"Loaded {len(stages)} stages")

    # Compute holdout splits.
    holdout_splits: dict[str, set[str]] = {}
    for stage in stages:
        _, holdout = split_probes(stage, fraction=0.3, salt="v2")
        holdout_splits[stage.name] = holdout
        print(f"  {stage.name}: {len(stage.episodes)} eps, {len(holdout)} holdout probes")

    results = run_additive_ablation(stages, holdout_splits, args.seeds)

    # Print matrix.
    print("\nCondition x Stage Accuracy Matrix:")
    for cond_name, stage_dict in results.items():
        print(f"\n  {cond_name}:")
        for stage_name, seed_results in stage_dict.items():
            accs = [r["holdout_accuracy"] for r in seed_results]
            s = summarize(accs)
            print(
                f"    {stage_name}: {s['mean']:.4f} ± {s['ci95_half']:.4f} "
                f"(n={int(s['n'])})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
