# 19 — The LM as language organ: trained embedding-cortex head, frozen LM

**Pre-registered 2026-07-03** (human-approved, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are
reported as deviations. Designed as the paired competitor to `research/18`
(plasticity in LM weights) on the same eval, same forgetting test, same
accounting.

## Problem / reframe

Every failed mechanism tried to make experience live *inside* the LM
(steering: refuted 4×; layer-L extraction: refuted). Every working component
treats the LM as an interface (prefix/KV hand it content; the reranker — the
only organ that earns its keep in M1 — decides *outside* it). And the one
component that was supposed to be the organism, the cortex, never received a
training loop at all.

Reframe: the LM is the **language organ** — perception (`peek_embedding`) and
articulation (generation) — and *changed dynamics* lives in a small trainable
cortex head over frozen LM embeddings. Corrections train the head; the frozen
LM supplies linguistic generalization (paraphrases land near each other in
embedding space); raw texts are then deletable because the mapping lives in
head weights, not stored exemplars.

## Hypothesis

**H-ORGAN:** a small head (≤ 64k params, e.g. logistic/2-layer MLP) trained
online from correction events on frozen `peek_layer`/`peek_embedding`
features — mapping request → sense label, with articulation conditioned on
the predicted label — produces held-out behavior change that (a) survives
deletion of all raw correction texts, (b) transfers to untaught paraphrases,
and (c) conditions on context (scope), which no template/keyword engine can.

## Architecture (fixed here)

- **Perception:** frozen `HFDriver` (Qwen2.5-0.5B-Instruct) embedding of the
  request (`peek_layer`, final layer, mean pooling — the S1.4 winner).
- **Cortex head:** trainable, ≤ 64k params, gradient-trained online per
  correction event (label = the sense phrase extracted from the correction
  utterance — the `corrected_label` that the teacher SAYS; eval `expected`
  strings and the frozen scorer are never inputs to training).
- **Articulation:** the LM generates conditioned on the head's winning label
  (e.g. label text supplied as a short generation prefix). The head decides;
  the language organ phrases. An "abstain" output (below-threshold
  confidence) falls through to vanilla generation — required, so untaught
  requests are untouched (specificity by construction is a claim to TEST,
  not assume).
- **Banned:** storing correction texts past consolidation, exemplar lookup at
  answer time, episode-ID conditioning, eval-derived features.

## Protocol

- **Curriculum:** eval v2 (current frozen version at run time; if the v3
  expansion has landed, use v3 and say so), research/11 replay protocol:
  teach stage-0 episodes seed-shuffled, consolidate (= finish head training +
  DELETE all raw correction texts, verified count 0), then score.
- **Tuning firewall (Gap provision a1):** all hyperparameters (features,
  head size, lr, epochs, abstain threshold, label-prefix phrasing) tuned on
  **stage-0 dev probes only**. Attested in the log. Holdout and all other
  stages are one-shot.
- **Transfer battery (Gap provision a2):** stage 1 is **never taught**. Its
  probes (paraphrase requests over stage-0's ambiguous words) are therefore
  all untouched by both teaching and tuning, and ALL stage-1 probes — dev and
  holdout alike — form the pre-registered transfer battery.
- **Scope test:** stages 2+5 taught and scored per protocol (holdout only) —
  the same-word-two-contexts conditioning that kills keyword/template
  engines.
- **Seeds:** ≥5 (head init + teaching order); fallback 3 (>15 min/seed,
  reported). Vanilla column mandatory everywhere.

## Primary metrics & acceptance

1. `organ_delta_holdout` — stage-0 holdout accuracy (head active, raw texts
   deleted) − vanilla.
2. `organ_transfer_delta` — full stage-1 battery accuracy − vanilla.
3. `organ_scope_delta` — stage-2 holdout accuracy − vanilla.
4. `organ_specificity_delta` — accuracy change on untaught stages' holdout
   (3, 4) — the abstain path must keep this ≥ −0.05.

- **Accept H-ORGAN:** (1) > 0 with 95% CI excluding 0, AND (2) > 0 with 95%
  CI excluding 0, AND (4) ≥ −0.05. Criterion (3) is reported and interpreted
  but does not gate acceptance (scope may need more capacity than 64k).
- **Refute:** (1) or (2) fails. If (1) passes and (2) fails, the head is an
  exemplar-memorizer with extra steps — record that framing explicitly.

## Comparisons (mandatory columns, same tables)

- Vanilla; the S3.M2a retrieval conditions (exemplar lookup — the baseline a
  *learned* mapping must beat on TRANSFER, where lookup structurally fails);
  research/18's LoRA result if available (weights-in-LM vs weights-in-cortex,
  the pair this spec exists for).

## Pre-registered secondaries (exploratory only)

1. **North-star accounting:** `behavior_delta_per_byte` with head bytes
   (serialized) as denominator — on RETENTION and separately on TRANSFER
   (where stored-text baselines pay per-probe failure, the axis per-byte can
   honestly flip).
2. Forgetting 2×2 via the merged harness (artifact = head weights).
3. Abstain-rate and confidence calibration curve.
4. Head size sweep (8k/64k/256k params) on dev only.
5. External-battery spot check (research/16 composition) — a head trained on
   8 episodes is exactly the overfitting risk that battery exists to catch.

## Reporting

Per-seed tables for all four primaries with CIs; deletion verification;
tuning-firewall attestation; comparison columns; model id; commands; log to
`experiments_logs/<date>_s19_language_organ.md` quoting this spec.

## Known eval gaps this spec inherits (and their remedy)

Stage-1 holdout is 1 probe and stage-0 holdout is 3 — hence provision a2
(full-battery transfer) above, and the separately-executed **eval v3
expansion** (S0.6 growth path: more episodes, paraphrase/adversarial holdout
variants, human-approved version bump via `scripts/bump_eval_version.py`).
If v3 lands before this experiment runs, this spec runs on v3 unchanged —
thresholds and metrics are split-relative, not count-relative.
