# S3 — Organ triage adjudication (research/14 verdicts)

**Date:** 2026-07-03
**Pre-registered spec:** `research/14-s3-organ-ablation-matrix.md`; the
KEEP / RETRIEVAL-BASELINE / ARCHIVE rule is applied mechanically below.
**Inputs:**
- **M1 subtractive** (full OrganismAgent − organ, real GGUF driver, dev
  split, 3 seeds): `2026-07-02_s3_m1_subtractive_ablation.md`
- **M2 additive** (MinimalOrganism + component, HF driver, holdout, eval v2,
  3 seeds): `2026-07-02_s3_m2_retrieval_ablation.md`
- Code audit predictions recorded in the spec before either run.

## Verdicts (spec rule: KEEP requires M2 Δ>0 with CI excluding 0 on ≥1 stage AND all-stage Δ≥0; retrieval paths that meet it are RETRIEVAL-BASELINE; everything else ARCHIVE)

| Component | M2 evidence | M1 evidence | Verdict |
|---|---|---|---|
| **Scope-slot reranker** | S0 **+0.667 ±0.00**, S4 **+0.250 ±0.00** (zero seed variance), all-stage +0.283 | S2 +0.205 (largest single-organ effect) | **RETRIEVAL-BASELINE** — kept, labeled retrieval in every future table, never counted as metabolism |
| **Hippocampus at answer time** | Δ = 0.0000 on every stage and seed (bit-identical to BASE) | all-stage +0.007 ±0.036 (noise) | **ARCHIVE** (the answer-time retrieval path only; the hippocampus as consolidation-time replay buffer is part of the minimal organism and not under test here) |
| **DSI fact index** | no stage with CI excluding 0 (S1 +0.667 ±1.43 on v2's single stage-1 probe — unsupported) | net **harmful** in full stack (−0.060; removing it improves S2/S3/S4) | **ARCHIVE**, with a named appeal: re-test S1 transfer on eval v2.1's 40-probe battery under a new pre-registered spec |
| **WorldModelCritic** | not run (M2b run stopped; harness merged) | −0.001 ±0.046 (noise) | **ARCHIVE** (cannot be KEEP without M2 positive; M1 noise + audit: untrained MLP) |
| **IdentityHypernetwork** | not run | −0.012 ±0.021 (noise/harmful) | **ARCHIVE** |
| **SkillImmuneCortex** | not run | +0.001 ±0.033 (noise) | **ARCHIVE** (audit: keyword matcher, no learned params) |
| **ExperienceAutoencoder** | not run | −0.014 ±0.023 (noise/harmful) | **ARCHIVE** (audit: no decoder exists) |

Per the spec: "No middle category. A component that 'almost' helps is
archived; it can return by winning a future pre-registered experiment."
The M2b harness (`organ_additive_organs.py`, merged, 6 tests) is the standing
appeal instrument for the four organs.

## Deviations

- M2b (additive arm for the four audited organs) was not executed: two agent
  attempts died silently and the orchestrator's run was stopped by the
  operator. Their ARCHIVE verdicts therefore rest on M1 (all CIs include 0)
  plus the spec's KEEP precondition (M2 positive required), which no
  unexecuted run can satisfy. This is the weakest link in the adjudication
  and is recorded as such; running M2b would only be needed to *promote* an
  organ, and every prediction and M1 datum says none would be.
- M1 ran on the dev split at 3 seeds; M2 on eval v2 (pre-expansion) at 3
  seeds. Verdict-relevant CIs are split-relative, so the rule applies
  unchanged.

## Consequences

- **research/15 (S3.3 tensor wiring): VACUOUS** — no component earned KEEP.
  Per that spec, this outcome "closes Goal 3's question honestly": there is
  no organ output worth wiring to tensors; the retrieval baseline stays a
  reranker by honest label.
- **S3.4 (archive under `attic/` with post-mortems):** now unblocked for
  critic, identity, immune, autoencoder, DSI, and the answer-time
  hippocampal path. Code task, pending.
- **Exploratory observation for the plasticity bets (18/19):** BASE itself
  lifts stage 4 (0.250 ±0.00 vs vanilla 0) and stage 5 (0.556 vs 0.333) —
  the content channel is not uniformly dead outside stage 0; and the
  reranker's zero-variance wins mark the exact bar research/19's *trained*
  head must clear on transfer, where exemplar rerank structurally cannot
  generalize.
