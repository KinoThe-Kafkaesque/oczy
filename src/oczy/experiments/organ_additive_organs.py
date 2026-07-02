"""S3.M2b: Additive organ ablation on the MinimalOrganism substrate.

This module subclasses :class:`MinimalOrganism` (from
:mod:`oczy.experiments.minimal_loop`) and adds one organ at a time behind
four keyword-only flags.  With every flag off the organism is behaviourally
identical to ``MinimalOrganism`` — no organ is constructed, no organ method
is called, so there is zero overhead on the BASE condition.

The four organs and their *behavioural* role inside MinimalOrganism:

- ``WorldModelCritic`` — predicts answer acceptance and records the real
  outcome.  In MinimalOrganism the hippocampus stores unconditionally
  (``surprise_threshold=0``) and is banned at answer time, so the critic's
  prediction has **no behavioural output path**: it learns but never gates
  storage or steers generation.
- ``IdentityHypernetwork`` — turns replayed corrections into identity-latent
  updates.  ``generate_adapters`` produces concept scores for candidate
  ranking, but MinimalOrganism performs a direct ``generate()`` with no
  ranking step, so the hypernetwork has **no behavioural output path**.
- ``SkillImmuneCortex`` — distils corrections into trigger-gated detectors.
  ``check()`` *does* modify what the LLM sees (the matched immune responses
  are prepended to the request), so this **is** a behavioural output path.
- ``ExperienceAutoencoder`` — compresses episodes into Δz vectors.  Its
  residual would feed the identity hypernetwork, which itself has no output
  path in MinimalOrganism, so the autoencoder has **no behavioural output
  path** either.

Pre-registered experiment: research/11-s2-minimal-metabolism-loop.md (M2b
additive matrix extension).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any

import numpy as np

from eval.v2 import verify_manifest
from experience_autoencoder import ExperienceAutoencoder
from identity_hypernetwork import IdentityHypernetwork
from oczy.common import extract_expected_from_correction
from oczy.common.stats import format_row, summarize
from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.minimal_loop import (
    CHECKPOINTS,
    CVEC_NORM_BUDGET,
    MAX_PREFIX_TOKENS,
    MinimalOrganism,
)
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.lm.hf_driver import HFDriver
from skill_immune_cortex import SkillImmuneCortex
from world_model_critic import WorldModelCritic

# ---------------------------------------------------------------------------
# OrganAdditiveOrganism
# ---------------------------------------------------------------------------


class OrganAdditiveOrganism(MinimalOrganism):
    """MinimalOrganism plus one optional organ at a time.

    Each keyword-only flag enables exactly one organ.  When a flag is False
    the corresponding attribute is ``None`` and every override skips that
    organ entirely, so BASE (all flags off) is byte-for-byte identical to
    :class:`MinimalOrganism`.
    """

    def __init__(
        self,
        driver: HFDriver,
        d_cortex: int = 128,
        cortex_seed: int = 0,
        hippocampus_config: dict[str, Any] | None = None,
        use_cvec_posture: bool = False,
        *,
        use_world_model_critic: bool = False,
        use_identity_hypernetwork: bool = False,
        use_skill_immune_cortex: bool = False,
        use_experience_autoencoder: bool = False,
    ) -> None:
        super().__init__(
            driver,
            d_cortex=d_cortex,
            cortex_seed=cortex_seed,
            hippocampus_config=hippocampus_config,
            use_cvec_posture=use_cvec_posture,
        )

        # Store the flags so overrides can branch without re-deriving them.
        self.use_world_model_critic = use_world_model_critic
        self.use_identity_hypernetwork = use_identity_hypernetwork
        self.use_skill_immune_cortex = use_skill_immune_cortex
        self.use_experience_autoencoder = use_experience_autoencoder

        # Construct only the organs that are enabled.  Each accepts None /
        # defaults so no config dict is required.
        self.world_model_critic: WorldModelCritic | None = (
            WorldModelCritic() if use_world_model_critic else None
        )
        self.identity_hypernetwork: IdentityHypernetwork | None = (
            IdentityHypernetwork() if use_identity_hypernetwork else None
        )
        self.skill_immune_cortex: SkillImmuneCortex | None = (
            SkillImmuneCortex() if use_skill_immune_cortex else None
        )
        self.experience_autoencoder: ExperienceAutoencoder | None = (
            ExperienceAutoencoder() if use_experience_autoencoder else None
        )
        # The autoencoder's hidden-delta path is enabled by configuration but
        # MinimalOrganism never supplies a ``hidden_delta`` (it exposes no
        # cortex agent), so encode() falls back to bag-of-words features.
        if self.experience_autoencoder is not None:
            self.experience_autoencoder.config["use_hidden_delta"] = True

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach(self, episode: Episode) -> None:
        """Perceive the episode via MinimalOrganism, then wire organs.

        MinimalOrganism.teach extracts the correction-utterance hidden state,
        observes it into the cortex, and stores the episode in the
        hippocampus unconditionally.  The organ wiring below is purely
        observational — it records/learns but does not alter what the
        hippocampus stored.
        """
        super().teach(episode)

        # WorldModelCritic: predict acceptance of the default response, then
        # record the real outcome (the correction utterance).  No output
        # path: MinimalOrganism's hippocampus stores unconditionally
        # (surprise_threshold=0) and is banned at answer time, so the
        # critic's prediction never gates storage or steers generation.
        if self.world_model_critic is not None:
            self.world_model_critic.predict_acceptance(
                query=episode.initial_request,
                proposed_answer=episode.default_response,
            )
            self.world_model_critic.record_outcome(
                query=episode.initial_request,
                proposed_answer=episode.default_response,
                correction=episode.correction_utterance,
            )

        # SkillImmuneCortex: register a detector from the correction.  The
        # detector is consulted at answer time (see answer()), which is the
        # one behavioural output path for this organ.
        if self.skill_immune_cortex is not None:
            expected = extract_expected_from_correction(episode.correction_utterance)
            self.skill_immune_cortex.add_detector(
                correction_text=episode.correction_utterance,
                mistake_class="corrected_sense",
                response=expected,
            )

        # ExperienceAutoencoder: encode + train on the episode.  No
        # hidden_delta is supplied (MinimalOrganism exposes no cortex agent),
        # so the bag-of-words feature path is used.  No output path: the
        # residual feeds the identity hypernetwork, which itself has no
        # behavioural output path in MinimalOrganism.
        if self.experience_autoencoder is not None:
            ep_dict = {
                "situation": episode.initial_request,
                "model_answer": episode.default_response,
                "correction": episode.correction_utterance,
                "outcome": "corrected",
                "source": "user_correction",
            }
            self.experience_autoencoder.encode(ep_dict)
            self.experience_autoencoder.train_step(ep_dict)

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, request: str, max_tokens: int = 32) -> str:
        """Generate an answer, optionally prepending immune responses.

        Only the SkillImmuneCortex has a behavioural output path in
        MinimalOrganism's answer(): its matched responses are prepended to
        the request so the LLM sees them.  The other organs have no output
        path here:

        - WorldModelCritic gates hippocampus replay, which is banned at
          answer time.
        - IdentityHypernetwork generates concept_scores for candidate
          ranking, but MinimalOrganism does a direct generate() with no
          ranking step.
        - ExperienceAutoencoder feeds the identity hypernetwork, which has
          no output path.
        """
        if self.skill_immune_cortex is not None:
            immune_responses = self.skill_immune_cortex.check(request, "")
            if immune_responses:
                meta = "[immune] " + " ".join(immune_responses)
                request = f"{meta} {request}"

        return super().answer(request, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Consolidate via MinimalOrganism, replaying into the hypernetwork.

        ``update_identity`` stores learning (moves the relevant identity
        slice toward the target concept), but ``generate_adapters`` has no
        behavioural output path in MinimalOrganism — the concept scores it
        produces are never consumed by the answer path.

        The replays must be captured *before* ``super().consolidate()``:
        MinimalOrganism.consolidate calls ``reinforce`` per query (bumping
        ``replay_count`` to the threshold) and then
        ``NeuralHippocampus.consolidate`` decays/removes the raw traces
        (``decay_after_consolidation`` defaults to True).  By the time the
        super call returns, fast memory is empty, so a post-super
        ``reinforce`` would yield no replays and the identity latent would
        never move.  Pre-capturing holds our own reference to the replayed
        dicts; the extra ``reinforce`` call merely bumps ``replay_count``
        earlier, which is harmless for the super's consolidation.
        """
        replays_per_query: list[list[dict]] = []
        if self.use_identity_hypernetwork and self.identity_hypernetwork is not None:
            for query in self._taught_queries:
                replays_per_query.append(self._hippo_raw.reinforce(query, k=3))

        result = super().consolidate()

        if self.use_identity_hypernetwork and self.identity_hypernetwork is not None:
            for replays in replays_per_query:
                for ep in replays:
                    # MinimalOrganism.teach stores the correction utterance
                    # under the ``correction`` key (no ``corrected_answer``
                    # field is written), so fall back through candidate
                    # sources before giving up.  Without this fallback the
                    # identity latent would never move and
                    # +IdentityHypernetwork would be indistinguishable from
                    # BASE.
                    corrected = (
                        ep.get("corrected_answer")
                        or ep.get("correction")
                        or ""
                    )
                    if corrected:
                        self.identity_hypernetwork.update_identity({
                            "source": "user_correction",
                            "correct_label": corrected,
                            "token": corrected,
                        })

        return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_m2_organ_matrix(
    seeds: list[int],
    stages: list[Stage],
    holdout_splits: dict[str, set[str]],
    *,
    verbose: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    """Run the M2 additive organ matrix: BASE + 4 organ conditions.

    For each condition × stage × seed the runner teaches the curriculum
    cumulatively up to each checkpoint, consolidates, and scores the
    holdout probes.  Returns a nested dict keyed by condition name then
    stage name, with one result dict per seed.
    """
    conditions: dict[str, dict[str, bool]] = {
        "BASE": {},
        "+WorldModelCritic": {"use_world_model_critic": True},
        "+IdentityHypernetwork": {"use_identity_hypernetwork": True},
        "+SkillImmuneCortex": {"use_skill_immune_cortex": True},
        "+ExperienceAutoencoder": {"use_experience_autoencoder": True},
    }

    results: dict[str, dict[str, list[dict]]] = {
        cond: {stage.name: [] for stage in stages} for cond in conditions
    }

    for condition, organ_flags in conditions.items():
        for stage in stages:
            holdout_ids = holdout_splits[stage.name]

            # Build the holdout probe index once per stage (stable across
            # seeds/conditions — mirrors _run_one_seed in minimal_loop).
            holdout_probes: list[tuple[str, Any, Any]] = []
            for ep in stage.episodes:
                for probe in ep.probes:
                    probe_id = f"{ep.id}|{probe.request}|{probe.category}"
                    if probe_id in holdout_ids:
                        holdout_probes.append((probe_id, probe, ep))

            for seed in seeds:
                if verbose:
                    print(
                        f"  {condition} {stage.name} seed={seed} ...",
                        end=" ",
                        flush=True,
                    )
                t0 = time.monotonic()

                rng = np.random.RandomState(seed)

                # Fresh driver + organism per (condition, stage, seed).
                driver = HFDriver.load()
                org = OrganAdditiveOrganism(
                    driver,
                    d_cortex=128,
                    cortex_seed=seed,
                    **organ_flags,
                )
                org.boot()

                # Seed-shuffle episodes.
                episodes = list(stage.episodes)
                rng.shuffle(episodes)  # type: ignore[arg-type]
                n_episodes = len(episodes)
                checkpoint_values = [c if c >= 0 else n_episodes for c in CHECKPOINTS]

                # Vanilla baseline: fresh driver, no organism, no prefix.
                vanilla_driver = HFDriver.load()
                vanilla_results: list[bool] = []
                for _, probe, ep in holdout_probes:
                    answer = vanilla_driver.generate(probe.request, max_tokens=32)
                    vanilla_results.append(probe_matches(answer, probe, ep))
                vanilla_acc = sum(vanilla_results) / max(len(vanilla_results), 1)
                del vanilla_driver
                gc.collect()

                # Cumulative teaching across checkpoints.
                taught_idx = 0
                final_holdout_acc = vanilla_acc
                final_cold_drift = 0.0

                for cp_val in sorted(set(checkpoint_values)):
                    while taught_idx < cp_val and taught_idx < n_episodes:
                        org.teach(episodes[taught_idx])
                        taught_idx += 1

                    cons_meta = org.consolidate()

                    score_results: list[bool] = []
                    for _, probe, ep in holdout_probes:
                        ans = org.answer(probe.request, max_tokens=32)
                        score_results.append(probe_matches(ans, probe, ep))
                    final_holdout_acc = sum(score_results) / max(len(score_results), 1)
                    final_cold_drift = cons_meta["cold_drift"]

                elapsed = time.monotonic() - t0

                del driver
                gc.collect()

                results[condition][stage.name].append({
                    "seed": seed,
                    "vanilla_acc": vanilla_acc,
                    "holdout_acc": final_holdout_acc,
                    "cold_drift": final_cold_drift,
                    "wall_clock_s": elapsed,
                })

                if verbose:
                    print(f"acc={final_holdout_acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_STAGES = (
    "stage_0_grounding",
    "stage_1_transfer",
    "stage_2_scope",
    "stage_3_dialog",
    "stage_4_consolidation",
    "stage_5_cross_domain",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S3.M2b: Additive organ ablation on the MinimalOrganism substrate",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of independent seeds (default: 5)",
    )
    parser.add_argument(
        "--stages", type=str, default=",".join(_DEFAULT_STAGES),
        help="Comma-separated curriculum stage names (default: all 6 stages)",
    )
    args = parser.parse_args(argv)

    stage_names = tuple(s.strip() for s in args.stages.split(",") if s.strip())

    # Verify manifest integrity before the run.
    print("Verifying eval manifest ...")
    try:
        verify_manifest()
    except Exception as e:
        print(f"MANIFEST VERIFICATION FAILED: {e}")
        print("Set EVAL_CHANGE_APPROVED=1 to bypass.")
        return 1
    print("  OK\n")

    seeds = list(range(args.seeds))
    print(f"Loading stages: {', '.join(stage_names)}")
    stages_tuple = build_curriculum(stage_names=stage_names)
    stages = list(stages_tuple)

    holdout_splits: dict[str, set[str]] = {}
    for stage in stages:
        _, holdout_ids = split_probes(stage, fraction=0.3, salt="v2")
        holdout_splits[stage.name] = holdout_ids
        print(
            f"  {stage.name}: {len(stage.episodes)} episodes, "
            f"{len(holdout_ids)} holdout probes"
        )
    print(f"\nRunning M2 organ matrix with {args.seeds} seeds ...\n")

    t_start = time.monotonic()
    results = run_m2_organ_matrix(seeds, stages, holdout_splits, verbose=True)
    total_elapsed = time.monotonic() - t_start

    # Verify manifest integrity after the run.
    verify_manifest()

    # Print summary table.
    print("\n" + "=" * 70)
    print("M2 ORGAN MATRIX RESULTS")
    print("=" * 70)
    print()
    print(
        f"{'condition':<26} {'stage':<26} {'vanilla':>8} {'holdout':>8} "
        f"{'delta':>8} {'drift':>8}"
    )
    print("-" * 70)
    for condition in results:
        for stage_name in results[condition]:
            rows = results[condition][stage_name]
            if not rows:
                continue
            vanilla_accs = [r["vanilla_acc"] for r in rows]
            holdout_accs = [r["holdout_acc"] for r in rows]
            drifts = [r["cold_drift"] for r in rows]
            mean_vanilla = float(np.mean(vanilla_accs))
            mean_holdout = float(np.mean(holdout_accs))
            mean_drift = float(np.mean(drifts))
            delta = mean_holdout - mean_vanilla
            print(
                f"{condition:<26} {stage_name:<26} "
                f"{mean_vanilla:>8.4f} {mean_holdout:>8.4f} "
                f"{delta:>+8.4f} {mean_drift:>8.4f}"
            )
    print()
    print(f"Total wall clock: {total_elapsed:.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
