# 27 — Retrieval exclusion via trace corruption

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

This is the unblocked version of R13's core question (trace deletion
exclusion), redesigned to avoid R13's pipeline dependency. Under the reframed
thesis (see
`notes/2026-07-26_in-context_serialization_thesis_reframe.md`), the
trace-deletion test simplifies to: remove the original context, inject the
compressed state, does the behavior persist? The philosophical
metabolism/retrieval distinction is replaced by a compression ratio curve.

## Problem

The Oczy thesis requires that the cortex causally controls behavior, not that
behavior is mediated by retrieval of stored traces. R13 was designed to test
this via trace deletion but has been blocked on pipeline dependencies. This
entry decomposes the test into conditions that can be run without the full
trace-deletion infrastructure.

## Hypothesis

**H-EXCLUSION:** the cortex state is necessary and sufficient for the
behavioral adaptation. Traces alone (without cortex state) do not produce the
same behavior.

## Method

Take the best-performing system that survives R24–R26. Four conditions on
held-out probes:

1. **Intact:** cortex state + intact traces → probe.
2. **Trace-corrupted:** cortex state + shuffled/corrupted traces → probe.
3. **Cortex-corrupted:** random/zeroed cortex state + intact traces → probe.
4. **Both corrupted:** random cortex state + corrupted traces → probe.

### Logic

| Condition | Result | Interpretation |
|---|---|---|
| (1) works, (2) fails | Traces are necessary | This is retrieval, not metabolism |
| (1) works, (3) fails | Cortex state is necessary | The cortex is doing something |
| (1) works, (2) works | Traces are sufficient without cortex | The cortex is a redundant intermediary |
| (1) works, (3) works | Cortex state is sufficient without traces | **This is the metabolism claim** |

Condition (3) is the trace-deletion test (R13) approximated without the full
pipeline. It is the one that has been blocked. This decomposition allows
partial evidence without waiting for the full deletion infrastructure.

## Measure

Holdout delta for each condition. Apply the logic table above.

## Success criterion

Condition (3) (cortex-corrupted) fails AND condition (1) (intact) succeeds.
The cortex state is necessary. Combined with R26 showing meta-trained > random,
this is evidence that the cortex is doing something beyond retrieval.

The strongest success: condition (1) works and condition (3) works (cortex
state is sufficient without traces). This is the metabolism claim.

## Kill criterion

If condition (2) performs as well as (1), the system is doing retrieval. Full
stop. The cortex is a fancy lookup table. Fundamentally change what the cortex
stores and how it transforms traces, or accept that the product works via
retrieval and reframe the scientific claim.

## Product vs. science split

If the system fails trace deletion (condition 3 works as well as condition 1)
but still produces correct, consistent, durable behavior via retrieval, that
may be acceptable for the product. The metabolism claim is scientifically
interesting but may not be product-necessary.

- **Scientific:** Does the cortex internalize patterns into autonomous
  dynamics? (This entry as designed.)
- **Product:** Does the user get correct, consistent, durable behavior? (A
  simpler test that does not require trace deletion.)

Decide in advance whether the scientific claim is required or whether the
product claim is sufficient. R30 (decision gate) makes this explicit.

## Cost

Moderate. Requires a clean separation of cortex state from traces, which may
require some refactoring. But no meta-training, no 90-shard runs.

## Why this matters

This is the experiment the entire thesis is actually about. Everything else is
engineering. If you can only run one more experiment, run this one.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
