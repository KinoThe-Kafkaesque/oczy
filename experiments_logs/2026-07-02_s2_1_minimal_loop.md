# S2.1 Minimal Metabolism Loop — REFUTE

**Date:** 2026-07-02
**Pre-registered spec:** `research/11-s2-minimal-metabolism-loop.md` (incl. the
2026-07-02 amendment repairing the degenerate stage-0 holdout split)
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (HFDriver, CPU float32, greedy)
**Stage:** `stage_0_grounding` (8 episodes; 3 holdout probes post-repair)
**Seeds:** 5 (seed-shuffled teaching order + cortex init)
**Checkpoints:** K ∈ {0, 1, 2, 4, 8}
**Code:** `src/oczy/experiments/minimal_loop.py`
(`uv run python -m oczy.experiments.minimal_loop --seeds 5`)
**Wall clock:** 745 s. `verify_manifest()` OK before and after.

## Verdict

**REFUTE H-LOOP.** The minimal organism (HFDriver + fast-weight cortex +
hippocampus-as-replay-buffer) does not close the loop on this substrate under
the pre-registered design:

- `loop_delta_holdout` = **0.0000** (need > 0, CI excluding 0) — every seed
  scored 0/3 holdout at K=N, exactly matching vanilla (0.0).
- `loop_compounding_rho` = **nan** (endpoint series not monotone; need ≥ 0.6).
- Validity gate PASSED: vanilla holdout accuracy 0.0 < 0.5, so this is a true
  refutation, not an invalid run.

## Per-checkpoint holdout accuracy (mean ± 95% CI, n=5)

| K | Holdout acc |
|---|---|
| 0 | 0.0000 ± 0.0000 |
| 1 | 0.0667 ± 0.1851 |
| 2 | 0.0000 ± 0.0000 |
| 4 | **0.1333 ± 0.2267** |
| 8 | 0.0000 ± 0.0000 |

Vanilla column: 0.0000 (all seeds). Per-seed K=N accuracy: 0.0 × 5.

## Mechanistic reading (exploratory, does not alter the verdict)

The trajectory is not noise-flat: at K=4, 2 of 5 seed orderings answered one
holdout probe correctly; at K=8 all effect vanishes. This is the **prefix
budget eviction signature**: ~13 tokens per correction means the spec's
48-token budget holds ~3.7 corrections. At K=4 most taught corrections still
fit; at K=8 the oldest are dropped (`prefix_overflow` > 0) and the probes'
facts are usually evicted. The content channel demonstrably *can* carry a
correction into changed holdout behavior — but only while its fact survives
the budget, and it does not compound; it crowds out.

Live-debug observations (single organism, seed 0, K=8): the consolidated
prefix compiled correctly (`'model' means fashion model. … 'log' means the
captain's journal.`, 48/48 tokens, 3 corrections evicted); answers were
coherent but the 0.5B raw-completion model paraphrases or rambles instead of
applying the sense mapping ("Deploy the model." → a classification list
containing "fashion" but never "fashion model").

## Invalid first run + implementation deviations (all recorded)

1. **Run 1 (INVALID, implementation bug):** flat 0.0 at every checkpoint
   because `emit_all_cvecs()` pushed warm-state cvecs at combined norm ~140
   (~29/layer × 24 layers), collapsing every generation into a repeated
   garbage token — the S2.4 magnitude pathology as a bug, masking the content
   channel entirely. Fixed in `d9702fc`.
2. **Cvec posture disabled (spec: "optional cvec posture").** Even clamped to
   combined norm 1.0 the posture still degraded some probes into token noise;
   the working amplitude on this substrate is uncalibrated. Posture defaults
   OFF; the clamp path remains for a future calibration experiment. The
   s2-5 fanout agent hit the same failure independently and made the same
   call — convergent diagnosis.
3. Prefix gains a `\n` separator (was bare-concatenated into the probe text).
4. `HFDriver.load()` no-arg default repaired (`39454d7`) — the S1.1/S1.2
   parallel-development name mismatch (`HF_MODEL_ID` vs `DEFAULT_MODEL_ID`)
   had never been exercised before this experiment.

## Consequences for the dependent specs (adjudicated per their gates)

- **research/12 (S2.2 KV content path): BLOCKED.** The spec's validity gate —
  "if research/11 REFUTED H-LOOP (C1 itself produces no effect), this
  experiment is reported as BLOCKED, not run to a fake verdict" — now binds.
  The KV channel implementation + tests are merged and remain valid code.
- **research/13 (S2.5 forgetting test): BLOCKED.** Validity gate
  `A_full − A_none ≥ 0.10` cannot hold when the loop does not close.
  Deletion APIs + harness are merged and remain valid code.

## What would have to change before a re-attempt (future specs, not tweaks)

1. **Chat-template prompting.** Qwen2.5-Instruct is chat-trained; raw
   completion severely underuses it (research/17 already anticipates
   template use for the second model). A new spec (research/18 candidate)
   would re-pose H-LOOP under the template, identically for vanilla.
2. **Content budget scaling law.** The K=4 bump suggests measuring holdout
   effect vs budget (16/48/128 tokens) — is eviction the binding constraint?
3. **KV-channel capacity.** S1.3 showed KV-splice ≡ prefix rank-for-rank at
   zero visible tokens; whether KV entries escape the visible-token budget
   trade-off is now the interesting question, but it needs a design that
   works at C1 first.

An honest REFUTE is a successful outcome; this one comes with a mechanism.

## See also

Conceptual post-mortem of why the steering/posture channel failed here and
across S1.3/S1.4/S2.4 — the three broken assumptions and what survives:
`notes/2026-07-03_steering_vs_posture_postmortem.md`. The surviving
mechanism (consolidation as context distillation) is pre-registered as
`research/18-consolidation-as-distillation.md`.
