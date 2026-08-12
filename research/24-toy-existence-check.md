# 24 — Toy existence check

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized. See `notes/2026-07-26_in-context_serialization_thesis_reframe.md`
for the conceptual basis.

## Source and evidence boundary

This entry is the toy-model existence proof proposed in the external review.
It tests whether the meta-learning mechanism exists at all, in the smallest
possible setting, under the actual UX constraint (single-patterned users, few
interactions). It should run after or alongside R23.5 (the serialization
baseline). If R23.5 shows no signal, this entry is unnecessary.

The fast weight programmer literature (Schmidhuber 1992; Ba et al. 2016)
suggests that external state modulates a base model most effectively through
multiplicative interaction, not additive injection. R02, R09, and R19 all
used additive or concatenative coupling and failed. This entry tests the
mechanism at toy scale with the coupling geometry left as a variable (see
R25 for the full coupling ablation).

## Problem

R20 meta-trains a cortex to control a frozen organ, but after 21 research
entries the mechanism has never been shown to exist even in a toy setting.
Every null result in the program is uninterpretable without a toy existence
proof: you cannot distinguish "the mechanism does not work at full scale"
from "the mechanism does not exist at all." This entry provides the cheapest
possible existence check.

## Hypothesis

**H-TOY-EXISTENCE:** a tiny meta-trained cortex can causally alter a tiny
frozen network's behavior on a held-out user pattern from a different family,
using 3 inner-loop interactions, in a way that is not reducible to retrieval
and not explained by architecture alone.

## Method

### Setup

- **Frozen "organ":** a tiny transformer (2 heads, 4 layers, vocab ~200)
  trained on a base distribution of simple string transformations.
- **"Cortex":** a small learned module (writer + reader + consolidator,
  ~10K params total) that reads interaction traces and produces a state
  vector.
- **"Coupler":** a single linear projection from cortex state into the
  organ's hidden layer at layer 2.
- **Outer loop (meta-training):** 50 "users," each with a single consistent
  pattern (e.g., one deterministic string-transformation rule per user). The
  meta-learner sees many different single-patterned users.
- **Inner loop (user-facing):** 3 examples from one user's pattern. No
  variety. The same rule applied to 3 different inputs.
- **Held-out test:** a new user with a new pattern drawn from the same
  distribution. 3 examples. Evaluate on novel inputs from that pattern.

### Conditions

1. **Meta-trained cortex** after 3 interactions.
2. **Randomly initialized cortex** after 3 interactions (same architecture,
   same developmental loop, no meta-training).
3. **No cortex** (organ alone).

### Coupling variants

Run all three conditions with at least two coupling geometries:
- Additive: `h' = h + W · s`
- Multiplicative / FiLM: `h' = γ(s) ⊙ h + β(s)`

(See R25 for the full coupling geometry ablation.)

## Measure

Holdout accuracy delta between conditions 1, 2, and 3.

## Success criterion

(1) − (2) ≥ 0.02 across all held-out users, AND (1) − (3) ≥ 0.02. The
meta-learned update rule contributes something beyond architecture, and
something beyond no cortex at all.

## Kill criterion

If (1) − (2) < 0.02 across all held-out users and coupling variants, the
meta-learned update rule contributes nothing beyond architecture. The
full-scale R20 calibration will not save you. Redesign the architecture or
the coupling mechanism before spending 90 shards.

## Cost

One afternoon. No Kaggle queue. CPU or single small GPU.

## Why this is first (after R23.5)

R23.5 tests whether in-context adaptation can be serialized at all. This
entry tests whether a meta-trained cortex can do the serialization better
than a random one. Together they cost less than one 90-shard calibration run
and answer the question of whether the next 90-shard run is worth doing.

If R23.5 is null, this entry is unnecessary (the serialization approach is
dead). If R23.5 is positive but this entry is null, the serialization works
but meta-learning does not help — the product may still be viable via
per-pattern soft prompt training (R26 condition 4), but the scientific claim
(meta-learning contributes) is false.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
