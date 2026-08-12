# 25 — Coupling geometry ablation

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

The fast weight programmer literature (Schmidhuber 1992; Ba et al. 2016)
suggests that external state modulates a base model most effectively through
**multiplicative** interaction (gating, scaling, rank-1 updates to weight
matrices) rather than additive injection into hidden states. Oczy's R02, R09,
and R19 all used additive or concatenative coupling and failed. This entry
tests whether the *type* of coupling is the structural reason for those
failures — a different and more fundamental diagnosis than "the coupler wasn't
good enough."

Under the reframed thesis (see
`notes/2026-07-26_in-context_serialization_thesis_reframe.md`), the coupling
question becomes: **what representation format preserves the most in-context
adaptation information per dimension?** Additive, multiplicative,
attention-gated — these are different compression formats. The one that
preserves the most behavioral information at the smallest dimensionality wins.

## Problem

Three coupling attempts (R02, R09, R19) have been attributed to implementation
issues. But the pattern of failure is consistent: additive injection into
hidden layers does not robustly steer a frozen transformer. This entry tests
whether the coupling geometry itself is the bottleneck, which is a different
diagnosis than "the coupler needs more engineering."

## Hypothesis

**H-COUPLING-GEOMETRY:** at least one non-additive coupling variant
(multiplicative/FiLM or attention-gated) produces a measurably higher holdout
delta than additive coupling, under identical cortex architecture and
developmental loop.

## Method

Same toy setting as R24. Test four coupling mechanisms:

1. **Additive:** `h' = h + W · s` (what R02/R09/R19 effectively tried).
2. **Multiplicative / FiLM:** `h' = γ(s) ⊙ h + β(s)` (feature-wise linear
   modulation conditioned on cortex state).
3. **Attention-gated:** cortex state produces a query that selects which organ
   activations to modulate.
4. **Prefix-equivalent control:** cortex state is decoded to tokens and
   prepended (the known-working baseline that is not the thesis).

## Measure

Holdout delta for each coupling. Critically, measure whether (1) is
significantly worse than (2) or (3).

## Success criterion

At least one non-additive variant produces holdout delta ≥ 0.05 where the
additive variant does not. If all three fail equally, coupling is not the
bottleneck.

## Kill criterion

- If all four couplings produce equivalent deltas, the coupling geometry does
  not matter and the problem is elsewhere (capacity, task distribution,
  meta-objective).
- If only (4) works, the thesis is empirically false for this architecture
  family. The frozen organ can only be controlled through its token interface.
- If (2) or (3) significantly outperforms (1), adopt the winning geometry for
  the full system.

## Cost

Moderate. Three variants × seeds, but each variant is a small code change.
Can be done on the toy model first (cheap), then on the full system if the
toy result is informative.

## Why this matters

You have had three failed coupling attempts (R02, R09, R19) and attributed
each to implementation issues. This tests whether the *type* of coupling is
the problem, which is a different and more fundamental diagnosis. If
multiplicative coupling works but additive does not, you have found an
architectural constraint and all downstream work should use the multiplicative
path.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
