# 18 — Consolidation as context distillation (the surviving route to changed dynamics)

**Pre-registered 2026-07-03** (human-approved, before implementation).
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
