# 29 — Information bottleneck measurement

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

The external review flagged that the entire Oczy program is empirical — run
seeds, measure deltas, classify outcomes — with no formal analysis of where
information is lost in the pipeline. This entry provides a per-stage
diagnostic: writer, consolidator, coupler, organ. Instead of asking "can the
coupler learn to bridge the spaces?" in the abstract, this entry measures
whether the spaces are bridgeable in principle and where information is lost.

## Problem

Every prior entry has debugged the whole pipeline blindly. When a null result
appears, you cannot tell whether the writer failed to encode the rule, the
consolidator destroyed the information, the coupler failed to transmit it, or
the organ could not read it. This entry gives a per-stage diagnostic that
targets repairs at the specific stage that loses information.

## Hypothesis

**H-BOTTLENEND-LOCALIZED:** the information about the learned user pattern is
lost at a identifiable stage of the pipeline (writer, consolidator, coupler,
or organ), and the loss point is consistent across seeds.

## Method

Train a simple probe (linear classifier or small MLP) to predict the user's
rule identity from four representations:

1. **Cortex state vector** (after consolidation).
2. **Raw developmental traces** (the 3 stored examples).
3. **Organ hidden activations** with cortex coupled.
4. **Organ hidden activations** without cortex (prefix-only baseline).

## Measure

Probe accuracy or mutual information for each representation.

### Logic

- If (1) < (2): the writer/consolidator is *destroying* information, not
  transforming it usefully.
- If (1) > (2): the cortex is integrating across traces and adding structure.
  This is evidence for metabolism.
- If (3) ≈ (4): the coupling is not transmitting the cortex's information into
  the organ. The interface is the bottleneck.
- If (3) > (4): the cortex is successfully modulating the organ's computation.

## Success criterion

The information loss point is identified and is consistent across seeds. If
(1) > (2) and (3) > (4), the cortex is both storing and transmitting
information — the strongest positive signal.

## Kill criterion

- If (1) < (2) across all seeds, the writer/consolidator is destroying
  information. Redesign the cortex's internal representation before anything
  else.
- If (3) ≈ (4) across all seeds, the coupler is not transmitting information.
  Redesign the coupling (see R25) or the interface dimensionality.
- If (1) ≈ (2) and (3) ≈ (4), the cortex is a pass-through that adds nothing.
  The architecture is not contributing.

## Cost

Cheap. No training, just forward passes and linear algebra. Can be done on
existing checkpoints if you have them.

## Why this matters

This tells you *where the information is lost* in the pipeline. Every prior
entry has debugged the whole pipeline blindly. This gives a per-stage
diagnostic: writer, consolidator, coupler, organ. You can target repairs at
the specific stage that loses information.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
