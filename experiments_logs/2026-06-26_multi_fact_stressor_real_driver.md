# Multi-Fact Turn Stressor — Real Driver — 2026-06-26 (corrected)

## Question

Does the multi-fact stressor run end-to-end against the real LFM2.5 GGUF
and can it discriminate architecture S vs H on co-retention?

## Method

Extended `src/oczy/experiments/multi_fact_stressor.py` with an optional
`--use-real-driver` flag that loads `LlamaCVecDriver` and passes it to
`CortexAgent`. Ran `--length 256` in both scalar and hybrid modes.

## Results

| mode | recall_a | recall_b | co_recall | traces | embedding_calls | consolidation_strength | cold_drift |
|---|---|---|---|---|---|---|---|
| scalar | 0 | 0 | 0 | 3 | 0 | 1.00 | 0.086 |
| hybrid | 0 | 0 | 0 | 3 | 0 | 3.56 | 0.305 |

Both modes produced valid `METRIC` and `ASI` lines against the real driver.
Hybrid shows the expected mechanical scaling (strength 1.0 → 3.56, cold_drift
0.086 → 0.305). `embedding_calls` reports `0` because only `_MockDriver` counts
embedder invocations, not `LlamaCVecDriver`.

## Interpretation

- The integration works: real driver loads, pipeline stores 3 traces, and the
  probe emits metrics.
- Co-recall is 0/0 not primarily because of instruction formatting but because
  cvec-only consolidation cannot reliably force exact target tokens. This
  matches the prior consolidation-uptake finding: residual steering shifts the
  semantic domain but does not bind exact vocabulary.
- Hybrid modulation is mechanically real: it produces stronger persistent traces
  from high-drift chunks. But stronger cvecs still do not guarantee exact-token
  recall.

## Implication for architecture

To make the multi-fact stressor behaviorally discriminating, the retrieval path
needs a reserved-position/prefix surface (e.g. `LlamaCVecDriver` articulation
prefix) or a knowledge-store fact-injection path, not just stronger cvecs. This
is the next frontier.

## Open questions

1. Does adding `ReservedPosition` prefix support to the stressor yield non-zero
   co_recall?
2. Once exact-token recall is possible, does hybrid mode improve co_recall or
   reduce memory cost?
3. Should the stressor also test mock-foreign embedder recall to compare cost?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`
- `experiments_logs/2026-06-26_multi_fact_stressor_real_driver.md`

## Commits

- `b171ec0` — Make multi_fact_stressor runnable against real LlamaCVecDriver.
- `2a5b786` — Update experiments/logs/SUMMARY.md with run #86 result.
