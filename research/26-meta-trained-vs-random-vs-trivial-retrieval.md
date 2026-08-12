# 26 — Meta-trained vs. random vs. trivial retrieval

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

The external review identified this as the single most important missing
control in the entire program. Every positive result in Oczy's history could
be an artifact of the developmental loop structure or a trivial retrieval
strategy, and the project has never run the control that would distinguish
"the meta-learner acquired a useful prior" from "the architecture happens to
be a decent interpolator."

Under the "single-patterned users" insight, a random cortex with a simple
"store the last 3 examples and match" strategy might work almost as well as a
meta-trained cortex. The meta-training needs to beat not just a random cortex
but a **trivial retrieval baseline.**

## Problem

R20 meta-trains a writer, reader, consolidator, and coupler. But the
meta-training objective itself has never been ablated against the critical
question: does a randomly initialized (non-meta-trained) cortex of the same
architecture perform significantly worse? If a random cortex + the same
developmental loop produces similar holdout deltas, then the meta-learning is
not contributing anything — the result is an artifact of the developmental
task distribution leaking through the architecture.

## Hypothesis

**H-META-CONTRIBUTES:** the meta-trained cortex produces a holdout delta that
exceeds both the random-init cortex and the trivial retrieval baseline by a
meaningful margin (≥ 0.01 vs random, ≥ 0.01 vs trivial retrieval).

## Method

Full-scale system (or the toy, if R24/R25 have not been run yet). Five
conditions, same developmental tasks, same held-out probes:

1. **Meta-trained cortex** (R20 output).
2. **Randomly initialized cortex**, same architecture, same developmental
   loop, no meta-training.
3. **Frozen random cortex** — same as (2) but the writer/consolidator weights
   are not updated during the developmental loop either.
4. **Explicit retrieval baseline** — store the 3 examples verbatim; at test
   time, find the nearest stored example and apply the same transformation. No
   cortex, no meta-learning, no coupling.
5. **No cortex** — organ alone, no developmental loop.

## Measure

Holdout delta for each. The critical comparisons:
- (1) vs (2): does meta-learning contribute beyond architecture?
- (1) vs (4): does the cortex apparatus contribute beyond trivial retrieval?
- (2) vs (3): does the developmental loop itself contribute beyond random
  initialization?

## Success criterion

- (1) − (2) ≥ 0.01 with non-overlapping 95% CIs across seeds.
- (1) − (4) ≥ 0.01 with non-overlapping 95% CIs across seeds.

## Kill criterion

- If (1) − (2) < 0.01, the meta-learning objective is not producing a useful
  prior. Redesign the outer-loop objective, not add more shards.
- If (1) − (4) < 0.01, the entire cortex apparatus is unnecessary for the
  product. The system reduces to nearest-neighbor retrieval. Decide whether
  that is acceptable or whether the scientific claim is the point.

## Cost

Same as one R20 calibration run. Can be done in parallel with R24.

## Why this is urgent

This control has never been run. Every positive result in the program's
history could be an artifact of the developmental loop structure or a trivial
retrieval strategy. Without this control, every positive result is ambiguous.
You cannot distinguish "the meta-learner acquired a useful prior" from "the
architecture happens to be a decent interpolator." Run this even if R24
succeeds — the toy model and the full system may behave differently.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
