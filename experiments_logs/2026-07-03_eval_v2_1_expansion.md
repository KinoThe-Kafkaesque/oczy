# Eval v2.1 — Curriculum expansion (S0.6 growth path)

**Date:** 2026-07-03
**Approval:** explicit human decision in-session ("address the gaps"), executed
per the standing agreement (deliberate change + version bump via
`scripts/bump_eval_version.py`). Manifest version: `v2` → `v2.1`.
**Authoring provenance:** `scripts/eval_expansion_v2_1.py` (deterministic,
re-runnable); validated by `validate_curriculum` (0 errors, 0 warnings) and
`validate_split` (0 errors).

## Why

The power gaps identified while pre-registering `research/19`
(and inherited by `research/18`): stage-0 holdout had 3 probes, stage-1
holdout had 1 — thinnest exactly on the transfer axis where learned-mapping
vs exemplar-lookup must be distinguished — and a trained head could saturate
stage 0 quickly (ceiling).

## What was added (existing episodes/probes never modified or removed)

12 new ambiguous words threaded through stages 0/1/2 in the existing
pattern (bare request → corrected non-tech sense; paraphrase transfer;
context-qualified tech sense + scope back-probe): draft, pitch, bar, note,
scale, bridge, port, crane, jam, seal, mole, organ.

- **Stage 0:** +12 episodes (8 → 20 episodes/probes).
- **Stage 1:** +12 episodes with 2 paraphrase probes each, and +1 extra
  cue-light paraphrase probe appended to each of the 8 original episodes
  (8 → 40 probes).
- **Stage 2:** +5 episodes (pitch, note, port, seal, organ: office/tech sense
  taught in context, scope probe pointing back at the stage-0 sense), and
  +4 **adversarial paraphrase scope probes** on existing episodes, where the
  surface verb cues the wrong domain ("Roll out the model." → fashion model;
  "Kick off the run." → river run) (16 → 30 probes).
- Stages 3, 4, 5 unchanged (their expansion deferred; noted as remaining gap).

## Split impact (fraction=0.3, salt="v2" — parameters unchanged)

| Stage | v2 dev/holdout | v2.1 dev/holdout |
|---|---|---|
| 0 grounding | 5 / 3 (promoted) | 17 / 3 (raw-hash) |
| 1 transfer | 7 / 1 | 31 / 9 |
| 2 scope | 13 / 3 | 26 / 4 |
| 3 dialog | 5 / 3 | 5 / 3 |
| 4 consolidation | 6 / 4 | 6 / 4 |
| 5 cross-domain | 9 / 3 | 9 / 3 |

**Disclosure — stage-0 holdout composition changed.** In v2, stage 0's raw
thresholding produced an EMPTY holdout and the repaired guarantee promoted 3
probes. In v2.1, 3 of the 12 new probes hash below 0.3 on their own, so the
guarantee no longer engages and the 3 previously-promoted v2 probes revert to
dev. Existing probes' *raw hash assignments* were never altered — the
promotion was always conditional on emptiness. Consequence: S2.1's REFUTE
was measured on the promoted trio; experiments from v2.1 onward measure on
the raw-hash holdout. Both are recorded; neither is retroactively changed.
Regression locks updated: `test_split_counts_locked_to_eval_v2_1`.

## Design notes

- New-word senses avoid all `PlasticCortex.BASELINE` tokens (validated).
- Stage-1 additions are the `research/19` transfer battery's substrate: with
  stage 1 never taught and tuning confined to stage-0 dev, all 40 probes are
  legitimately untouched test items.
- The 4 adversarial scope probes are the template/keyword-engine killers:
  correct answers require conditioning on the *absence* of domain
  qualification, against the surface verb's cue.

## Remaining gaps (deferred, tracked)

Stages 3/4/5 unexpanded; no adversarial variants outside stage 2; external
(non-repo-authored) benchmark still pending in research/16's battery.
