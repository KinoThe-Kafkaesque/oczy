# 14 — Organ ablation matrix & triage verdicts (Sprint 3 / S3.1 + S3.2 + S3.4)

**Pre-registered 2026-07-02** (human-approved sprint setup, before the runs). **Status: TESTED-METRICLESS-NULL (2026-07-11).**

> **Outcome (2026-07-11):** TESTED-METRICLESS-NULL. Campaign 0d48130 (kaggle CPU-only, commit `2a22049`): the M2B additive-organs run (`--seeds 3`) exited 0 after 11,786.6 s but emitted **no `METRIC` or `ASI` values**. No effect estimate or positive/negative mechanism verdict is available beyond the registered metricless null — the harness completed but produced no scored output. This is distinct from a scientific null (which measures zero effect) and from a refutation (which measures a negative effect). Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations. Depends on research/11's minimal organism being merged; the
S3.1 harness (`src/oczy/experiments/organ_ablation.py`, merged 2026-07-01)
provides the subtractive matrix.

## Problem

Five organs (NeuralHippocampus-at-answer-time, WorldModelCritic,
IdentityHypernetwork, SkillImmuneCortex, ExperienceAutoencoder) plus two
retrieval components (DSI DifferentiableFactIndex, scope-slot reranker) ride
along in every full-organism run. The code audit found most have no trained
parameters or no wired output. Nothing may be kept on vibes: every component
either moves a held-out behavioral metric on the frozen eval or is archived.

## Design — two complementary matrices, both on frozen eval v2, holdout split

**M1 (subtractive, existing harness):** full `OrganismAgent` minus one organ
at a time via the off-switch config, `organ_ablation.py`, real driver,
≥5 seeds (pre-registered fallback 3 if >15 min/seed, reported as deviation).

**M2 (additive, the decisive one):** research/11's `MinimalOrganism` plus one
component at a time. A component only *earns a place* through M2; M1 alone
cannot save it (full-stack interactions can mask dead weight).

Both matrices report per-stage holdout accuracy mean±CI (`oczy.common.stats`)
with the vanilla column, plus the S2.3 drift triple where steering is involved.

## Per-component pre-registered predictions (from the code audit — the run
confirms or refutes; predictions cannot bias scoring)

| Component | Prediction | Reason |
|---|---|---|
| SkillImmuneCortex | no effect | keyword substring matcher, no learned params |
| WorldModelCritic | no effect | 3/4 weights hardcoded, MLP never trained |
| IdentityHypernetwork | no effect | historical 0.001–0.006 on a broken probe |
| ExperienceAutoencoder | no effect | no decoder exists; not an autoencoder |
| Hippocampus (answer-time) | positive | it is retrieval — the baseline to beat |
| DSI fact index | positive | retrieval |
| Scope-slot reranker | positive on Stage 2/5 | retrieval; source of honest 0.92 |

## Verdict rule (fixed now)

For each component, with Δ = (with − without) holdout accuracy, mean over
seeds, on the stage(s) the component claims to serve and on the all-stage
mean:

- **KEEP (wire to tensors, research/15):** Δ > 0 with 95% CI excluding 0 on
  at least one stage AND all-stage Δ ≥ 0.
- **RETRIEVAL-BASELINE:** same as KEEP but the component is a retrieval path
  (hippocampus-at-answer, DSI, reranker) — kept, but labeled as the retrieval
  baseline in every future table, never counted as metabolism.
- **ARCHIVE:** CI includes 0 on every stage (dead weight), or all-stage
  Δ < 0 with CI excluding 0 (harmful). Archived under `attic/` with a
  one-page post-mortem (S3.4); autonomous sessions stop touching it.

No middle category. A component that "almost" helps is archived; it can
return by winning a future pre-registered experiment.

## Reporting

Both matrices (component × stage tables, mean±CI), verdict per component with
the rule applied mechanically, archive list, model id, commands; log to
`experiments_logs/2026-07-02_s3_organ_ablation_matrix.md` quoting this spec.
