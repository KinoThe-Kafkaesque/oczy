# 18 — Consolidation as context distillation (the surviving route to changed dynamics)

**Pre-registered 2026-07-03** (human-approved, before implementation). **Status: TESTED-PARTIAL with gate resolved (human-adjudicated 2026-08-06).**

> **Original-condition outcome (2026-07-11; preserved): BLOCKED at the teacher validity gate / diagnostic only.** Campaign 0d48130 (Kaggle CPU-only, commit `537260c`):
> - **R18 teacher-gate run (1 seed, 5 steps):** gate FAILED. `distill_delta_holdout=0.3333`, `distill_specificity_delta=0.04348`, and `teacher_dev_delta=0.1765 < 0.2`; fallback chat-template prompting did not clear the registered threshold. LoRA rank=8/alpha=16/lr=0.005.
> - **R18 full run (3 seeds, 10 steps):** BLOCKED / diagnostic only. `distill_delta_holdout` mean=0.2222 — bimodal {0.3333, 0.3333, 0.0}; conditional signal present in 2/3 seeds, absent in 1/3. `teacher_dev_delta=0.1765` and `persistent_bytes=17,699,903` were identical across all seeds; `specificity_delta`: 0.0/0.0/0.04348.
> No H-DISTILL verdict was permitted for this original condition because the teacher gate failed. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md` and `../experiments_logs/2026-07-11_r18_five_seed_diagnostic.json`.


## REGISTERED AMENDMENT A1 — frontier teacher substitution (2026-08-06, human-authorized)

**Authorizing human decision (2026-08-06):** approve the one-variable teacher
substitution proposed in
`notes/2026-07-26_r18-r19_deblock_proposal.md` (P1), using the OpenRouter
frontier teacher `deepseek/deepseek-v4-flash-0731` (provider pinned to
DeepSeek, no fallback) as the R18 dev-gate teacher. Rationale and evidence:

- The registered gate (`teacher_dev_delta >= 0.2`) was **cleared by the
  frontier teacher**: measured 0.5294 on the full stage-0 dev split (17
  probes; with-correction 9/17, without 0/17) vs the failed 0.5B-prefix
  baseline of 0.1765 on every seed. Record:
  `../experiments_logs/2026-08-06_stage0_openrouter_teacher_gate.md`.
- Scope of the amendment: **dev-gate teacher source only**, enabled by
  `--teacher openrouter` on
  `src/oczy/experiments/consolidation_distillation.py`. Student, LoRA
  distillation (local prefix logits), metric, split (`salt="v2"`), and the
  `>= 0.2` gate are unchanged. eval/v2 is untouched.
- The prior BLOCKED disposition (teacher expressivity ceiling, 0.5B) is
  superseded for the amended condition by the clearing evidence above; the
  original 0.5B condition's verdict records remain intact.
- Capability-floor caveat (separate, unchanged): the 0.5B *student* may still
  sit below an expression floor; that question belongs to S4.3.

**Amendment status:** **EXECUTED (2026-08-06).** 5-seed stage-0 run
completed: `teacher_dev_delta` = {0.4706, 0.5294, 0.4706, 0.4706, 0.5294}
(mean 0.4941) — **gate cleared on all seeds** vs 0.1765 pre-amendment;
`distill_delta_holdout` mean 0.2667 [0.136, 0.397], 4/5 positive (seed 2
null). Record: `../experiments_logs/2026-08-06_r18_openrouter_5seed_stage0.{md,log}`.
**Human adjudication (2026-08-06): TESTED-PARTIAL with gate resolved.** The
amended evidence is admissible and positive on 4/5 seeds, but the seed-2 null
prevents full H-DISTILL acceptance. The original local 0.5B-teacher condition
remains BLOCKED and its historical records are unchanged.

Agents running this experiment MUST NOT edit this spec; deviations are
reported as deviations.

## Problem

Every mechanism tested for "memory becomes changed dynamics" has been refuted
under controls: cvecs cannot force tokens (06-27), the 13.5x drift was
magnitude inflation (S2.4), the Hebbian posture channel *harms* generation at
every tested amplitude (S2.1), and the mid-layer geometry assumption is dead
on two architectures (S1.4). Everything that works is content injection —
prefix (S2.1's transient K=4 bump), KV-splice (S1.3 parity), reranker (M1's
+0.205). Meanwhile the substrate migration (Sprint 1) made gradients
available for the first time — llama.cpp structurally prevented training.

This spec pre-registers the one credible untested mechanism: at consolidation,
**distill prefix-conditioned behavior into a small weight delta (LoRA), then
delete the prefix and the raw traces**. If behavior survives in the weights,
memory has literally become changed dynamics.

Why this can succeed where S2.1 refuted: S2.1's failure mode was *budget
eviction* — one persistent 48-token prefix shared by all facts. Distillation
uses the prefix only as **transient per-fact scaffolding**: each correction is
distilled with its own dedicated prefix (~13 tokens, no crowding), and the
facts accumulate in the adapter, not in a token budget.

## Hypothesis

**H-DISTILL:** for corrections taught per research/11's protocol, a
consolidation step that trains a LoRA adapter to match the model's own
prefix-conditioned behavior (context distillation, one fact at a time)
produces held-out behavior change that survives deletion of the prefix AND
all raw traces, without degrading unrelated behavior.

## Design

- **Substrate:** `HFDriver` / `Qwen/Qwen2.5-0.5B-Instruct`, CPU float32,
  greedy eval. Base organism: `MinimalOrganism` (prefix channel, posture off).
- **Teaching:** research/11 protocol, stage 0, all N episodes, seed-shuffled.
- **Consolidation-as-distillation:** for each stored correction c:
  - Teacher = frozen base model WITH prefix `c` (the correction utterance
    alone).
  - Student = base model + LoRA (rank ≤ 8, attention projections), NO prefix.
  - Loss: token-level KL(student ‖ teacher) on a distillation prompt set.
  - **Distillation prompts:** the episode's `initial_request` plus a FIXED
    generic template list defined in code (e.g. "Q: {request}\nA:",
    imperative/question paraphrase frames). Eval `expected` strings and
    holdout probe texts MUST NOT appear anywhere in training data or loss.
    The teacher's own outputs are the only target.
- **After consolidation:** prefix cleared, hippocampus traces deleted
  (verified count 0, `memory_bytes` before/after), organism = driver + LoRA
  only. Score stage-0 HOLDOUT probes (repaired split, frozen scorer).
- **Seeds:** ≥5 (LoRA init + teaching order); pre-registered fallback 3 if
  >15 min/seed. Vanilla column mandatory.
- **Hyperparameters** (lr, steps, rank, template list) may be tuned on the
  DEV split only, before any holdout measurement; the final holdout run is
  one shot.

## Validity gate — the teacher must work (measured on DEV, adjudicated first)

`teacher_dev_delta` = dev accuracy of base-model-with-per-fact-prefix on
taught episodes' dev probes − vanilla dev accuracy. Gate: ≥ 0.2.

- If the gate fails with raw prompting, the single pre-registered fallback is
  **chat-template prompting** (applied identically to teacher, student, and
  vanilla), then re-test the gate once.
- If it still fails: **BLOCKED** (substrate cannot express the behavior even
  with the fact in-context; no distillation verdict is drawn).

## Primary metrics & acceptance

1. `distill_delta_holdout` = mean over seeds of
   [holdout accuracy, LoRA-only, traces deleted] − [vanilla holdout accuracy].
2. `distill_specificity_delta` = mean over seeds of the change in accuracy on
   the OTHER five stages' holdout probes (untaught, no adapter should touch
   them) plus the S2.3 control-word logit shift, reported together.

- **Accept H-DISTILL:** `distill_delta_holdout > 0` with 95% CI excluding 0,
  AND `distill_specificity_delta ≥ −0.05`, AND trace deletion verified.
- **Refute:** delta fails with the teacher gate passed. (A refutation here
  retires gradient consolidation at this scale — the last standing mechanism —
  and the honest conclusion becomes "retrieval is the architecture.")
- **BLOCKED:** teacher gate fails under both prompting modes.

## Pre-registered secondary analyses (exploratory only — cannot flip acceptance)

1. **North-star accounting:** `behavior_delta_per_byte` with LoRA bytes as
   the denominator (serialized float32 state_dict), reported HONESTLY next to
   the raw-text-bytes alternative (~tens of bytes/fact, which naive per-byte
   always wins) and `behavior_delta_per_context_token` (LoRA: 0 tokens).
2. **Transfer:** stage-1 holdout delta (weights may generalize to paraphrases
   where a verbatim prefix cannot — the axis on which changed-dynamics can
   beat retrieval).
3. **Sequential consolidation:** adapter updated correction-by-correction
   (K-trajectory {1,2,4,N}) — compounding and interference between facts.
4. **Forgetting-harness cross-check:** research/13's 2×2 arms with
   artifact = LoRA (uses the merged `minimal_loop_forgetting.py` machinery).
5. Wall-clock and peak-RSS of the distillation step.

## Reporting

Per-seed tables (holdout, specificity, transfer), teacher-gate record with
prompting mode used, hyperparameters + template list frozen in the log,
deletion verification, model id, exact commands; log to
`experiments_logs/<date>_s18_consolidation_distillation.md` quoting this
spec. The dev/holdout firewall (tuning on dev only) must be explicitly
attested in the log.
