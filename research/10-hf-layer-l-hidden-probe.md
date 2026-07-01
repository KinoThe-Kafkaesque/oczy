# 10 — HF-substrate layer-L hidden extraction probe (Sprint 1 / S1.4)

**Pre-registered 2026-07-01** (human-approved sprint setup, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations. This re-adjudicates lane_03's refuted H1 on a substrate that can
actually see every layer (`output_hidden_states=True`), with the fallback
analyses fixed in advance — lane_03's post-hoc pooling exploration is exactly
what this pre-registration prevents.

## Hypothesis

H-L: at some mid-depth layer L (25–75% of depth), sense-corrected phrase
hiddens cluster by concept better than at the final layer, by silhouette
score, gap >= +0.10.

## Corpus

The lane_03 phrase corpus (same concepts × paraphrases; reuse its definition
verbatim from `lanes/lane_03.py`). No new phrases may be added after seeing
results.

## Primary analysis (the ONLY acceptance surface)

- Embedding per phrase per layer: **mean-pool over content tokens**
  (stopword-token positions excluded, matching lane_03's mean-pool variant).
- `silhouette(L)` per layer; `gap = max over mid layers silhouette(L) −
  silhouette(final layer)`.
- **Accept H-L:** gap >= +0.10.
- **Refute:** gap < +0.10. If the HF substrate confirms llama.cpp's
  refutation, that is a strong, clean negative: the cortex should consume
  final-layer hiddens and Goal 2's "mid-layer semantic intent" assumption is
  retired.

## Pre-registered secondary analyses (exploratory only — cannot flip acceptance)

1. Last-token pooling per layer (lane_03's post-hoc variant, now registered).
2. Max-pooling per layer.
3. Per-layer table for the chosen model AND (if cached weights permit) the
   LFM2.5-1.2B HF checkpoint, to separate "substrate keyhole" from "model
   property".

## Reporting

Per-layer silhouette table (all layers, all three poolings, clearly marking
the primary), model id(s), corpus hash, and the accept/refute verdict against
THIS spec. Log to `experiments_logs/`.
