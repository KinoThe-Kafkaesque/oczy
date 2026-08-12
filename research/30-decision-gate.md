# 30 — Decision gate

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

This is not an experiment. It is a structured evaluation of everything
R23.5–R29 produced, with explicit continue/pivot/stop criteria. The external
review emphasized that every entry in R24–R30 produces a useful result
regardless of outcome, and that the program should not gamble on a single
binary outcome. This entry is where the honest decision is made.

## Problem

The Oczy program has spent months on engineering around a hypothesis that may
be architecturally impossible. R23.5–R29 are designed to produce cheap,
decisive signals before any more 90-shard calibration runs. This entry
consolidates those signals into a decision: continue, pivot, or stop.

## Method

No new experiments. Evaluate the results of R23.5–R29 against the criteria
below.

### Continue if

- R23.5 showed that in-context adaptation can be serialized (≥ 30% recovery).
- R24 showed a nonzero toy signal (meta-trained > random by ≥ 0.02).
- R25 identified a coupling geometry that carries the signal.
- R26 showed meta-trained > random AND meta-trained > trivial retrieval.
- R27 condition (3) showed cortex-state-sufficiency (behavior persists without
  traces).
- R28 showed generalization through at least level 3.
- R29 showed the cortex state contains more rule information than raw traces.

### Pivot if

- The mechanism exists in the toy (R24) but not at full scale → the scaling
  path is broken; redesign the full system around what worked in the toy.
- Additive coupling is dead but multiplicative works (R25) → rebuild the
  coupler.
- The cortex stores information (R29) but the coupler does not transmit it →
  the interface is the bottleneck, not the learning.
- Meta-trained ≈ random (R26) but the architecture + developmental loop still
  works → the product is viable without meta-learning; reframe the scientific
  claim.
- The system works via retrieval (R27) but behavior is correct and durable →
  the product ships; the scientific thesis is false; decide whether that
  matters.

### Stop if

- R23.5 shows no serialization signal → the in-context adaptation cannot be
  compressed. The entire approach is dead.
- R24 shows no toy signal → the mechanism does not exist in this architecture
  family.
- R25 shows only prefix-equivalent control works → the frozen organ can only
  be controlled through its token interface; the latent-interface thesis is
  false.
- R26 shows meta-trained ≈ random ≈ trivial retrieval → the entire apparatus
  is unnecessary.
- R27 shows all positive results reduce to retrieval AND retrieval behavior is
  not durable or consistent enough for the product → neither the science nor
  the product works.

### Write the negative result up

A well-documented "we tested whether a meta-trained external cortex can
causally control a frozen transformer through latent interfaces under
few-shot user constraints, and it cannot, and here is exactly where the
information is lost" is a legitimate and useful contribution. The 73-record
evidence ledger makes this credible in a way that most negative results are
not.

## Sequencing summary

```
R23.5 (serialization baseline)  ← cheapest, do first
  │
R24   (toy existence check)     ← do in parallel
R24.5 (prior coverage)          ← do in parallel, no GPU
R25   (coupling geometry)       ← informed by R24
R26   (meta vs random vs retrieval)  ← the missing control
  │
  ├── if R23.5–R26 all fail, stop
  │
R27   (retrieval exclusion)     ← the thesis experiment
R28   (generalization boundary) ← the product constraint
R29   (information bottleneck)  ← the per-stage diagnostic
  │
R30   (decision gate)           ← continue, pivot, or stop
```

R23.5 through R26 are cheap and should be completed before any further
full-scale calibration. R27 through R29 are the scientific payload. R30 is
the honest decision point.

Every entry produces a useful result regardless of outcome. A negative at
R23.5 or R24 saves months. A positive at R27 is the thesis. A gradient at R28
is a contribution even if the top level fails. The program is no longer
gambling on a single binary outcome.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
