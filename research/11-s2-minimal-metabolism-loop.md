# 11 — Minimal metabolism loop on the HF substrate (Sprint 2 / S2.1)

**Pre-registered 2026-07-02** (human-approved sprint setup, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations.

## Problem

The audit's central architectural finding is inversion: in the full organism,
retrieval (scope-slot reranker, DSI, hippocampal lookup at answer time) does
the work the thesis attributes to changed dynamics. Every past "loop closed"
claim ran with five organs attached, so nothing could attribute the behavior
change to the fast-weight path. Sprint 0 froze the eval and re-measured honest
baselines; Sprint 1 delivered the HF substrate (Qwen2.5-0.5B-Instruct,
`HFDriver`). This experiment asks the thesis's minimal question with nothing
else in the room.

## Hypothesis

**H-LOOP:** a minimal organism consisting of ONLY

1. `HFDriver` (Qwen/Qwen2.5-0.5B-Instruct, greedy, CPU float32),
2. a fast-weight cortex (warm/cold state, Hebbian `observe`, `consolidate`
   merging warm→cold), and
3. `NeuralHippocampus` used strictly as a consolidation-time replay buffer
   (NEVER queried at answer time)

closes the loop: correction → fast-weight change → consolidation → measurably
changed LM behavior on held-out frozen-eval probes, compounding with the
number of corrections K.

**Explicitly banned components:** WorldModelCritic, IdentityHypernetwork,
SkillImmuneCortex, ExperienceAutoencoder, DifferentiableFactIndex, the
scope-slot reranker, logit bias, and ANY answer-time retrieval (no hippocampus
`reinforce()` during `answer()`, no per-probe lookup of stored corrections).

## Content channel (S2.1 form)

At `consolidate()`, replayed corrections are compiled into a **bounded
consolidated articulation state**: a single articulation prefix of at most
**48 tokens total** (all facts share the budget; overflow = oldest-dropped and
reported), set once via `HFDriver.set_articulation_prefix`, plus optional cvec
posture. The prefix is built from consolidated summaries at consolidation
time — NOT re-derived per probe from raw traces. (S2.2 / research/12 replaces
this prefix with written KV entries; S2.5 / research/13 deletes the raw traces
and tests survival.)

## Data & protocol

- **Stage:** `eval/v2/stage_0_grounding.json` (frozen; `verify_manifest()`
  must pass before and after the run).
- **Split:** `split_probes(stage, fraction=0.3, salt="v2")` — development on
  dev, **all primary numbers on holdout only**.
- **Teaching:** episodes taught cumulatively in a seed-shuffled order; each
  teaching event feeds `correction_utterance` through perceive/metabolize and
  stores the episode in the hippocampus; consolidation fires per the digestive
  gate (or forced at each checkpoint boundary — implementer's choice, fixed in
  code and reported).
- **K checkpoints:** K ∈ {0, 1, 2, 4, N} where N = all stage-0 episodes.
  After each checkpoint: consolidate, then score ALL holdout probes with the
  frozen scorer (`oczy.eval_v2.scoring.probe_matches`).
- **Seeds:** ≥5 (vary teaching order + cortex init; LM is deterministic
  greedy). Mean ± 95% CI via `oczy.common.stats`.
- **Vanilla column:** bare `HFDriver` on the same holdout probes, mandatory.

## Primary metrics & acceptance

1. `loop_delta_holdout` = mean over seeds of
   [holdout accuracy at K=N] − [vanilla holdout accuracy].
2. `loop_compounding_rho` = Spearman ρ between K and mean-over-seeds holdout
   accuracy across the 5 checkpoints.

- **Accept H-LOOP:** `loop_delta_holdout > 0` with 95% CI excluding 0, AND
  `loop_compounding_rho >= 0.6`.
- **Refute:** either fails. A refutation is a recorded result, not a failure.

**Validity gate (not acceptance):** vanilla holdout accuracy must be < 0.5 on
stage 0 (otherwise there is nothing to learn and the run is INVALID, not a
refutation).

## Pre-registered secondary analyses (exploratory only — cannot flip acceptance)

1. The S2.3 drift triple (Δ_target / Δ_control / Δ_target_clamped) at every
   checkpoint, using the FIXED clamp-budget capture (the cross-instance
   stochasticity artifact from `2026-07-01_s2_4_breakthrough_ablation.md`
   must be repaired before this run).
2. Dev-split trajectory (for overfitting comparison dev vs holdout).
3. `memory_bytes` at each checkpoint and prefix-token count actually used.
4. Wall-clock per checkpoint.

## Pre-registered fallbacks (fixed now, so they cannot be chosen post hoc)

- If wall-clock per seed exceeds 15 min: drop to 3 seeds, keep all checkpoints,
  and report the reduction as a deviation.
- If the digestive gate never fires: force consolidation at checkpoint
  boundaries and report it.

## Reporting

Full per-seed, per-checkpoint table (holdout accuracy, drift triple,
memory_bytes); vanilla column; model id; exact commands; log to
`experiments_logs/2026-07-02_s2_1_minimal_loop.md` quoting this spec.
