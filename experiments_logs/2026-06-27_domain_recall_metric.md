# Domain-Level Recall Metric — 2026-06-27

## Question

When cvec-only consolidation cannot force exact target tokens, does it at least
steer answers into the correct semantic domain? And can a domain-level metric
discriminate architecture S from architecture H?

## Method

Added `--domain-recall` to `multi_fact_stressor.py`. Domain targets are:

- Fact A: `["alpha", "skylark", "project alpha"]`
- Fact B: `["beta", "rook", "project beta"]`

An answer scores `domain_recall=1` if any of the listed keywords appears
case-insensitively. `domain_co_recall=1` when both facts' domains are present.

Ran real-driver S vs H with `--auto-consolidate --hybrid-cap 0 --domain-recall
--length 512` (cvec-only, no prefix).

## Results

| mode | co_recall | domain_recall_a | domain_recall_b | domain_co_recall | memory_bytes |
|---|---|---|---|---|---|
| scalar | 0/0 | 1 | 1 | 1/1 | 29,211 |
| hybrid (uncapped) | 0/0 | 1 | 1 | 1/1 | 29,214 |

## Interpretation

- Cvec-only consolidation reliably shifts the LM's answer into the correct
  project-name domain (both facts), even though exact target tokens are not
  emitted.
- Architecture H's ~3.6x higher consolidation strength does not improve
  domain-level recall beyond scalar.
- This confirms an earlier finding: cvec steering is a posture/domain-level
  surface; exact recall requires a reserved-position/prefix surface.

## Implication for architecture

A domain-level metric is useful, but it does not separate S from H in this
probe. The remaining differentiator is likely hippocampus-derived prefixes or
identity-hypernetwork adapters. The ingestion scaffold itself is now a
well-instrumented, validated platform for those experiments.

## Open questions

1. Would a stricter domain metric (e.g. both correct project names in the same
   answer) discriminate S vs H?
2. Does hybrid mode improve domain recall on noisier/ambiguous prompts where
   scalar fails?
3. How do hippocampus-derived prefixes perform on this exact probe?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `ae64cef` — Add --domain-recall metric.
- `1194e77` — Update SUMMARY.md with run #95 result.

## Run

Run #95: benchmark `code_qa_accuracy=1.0`, fast suite `305 passed`.
