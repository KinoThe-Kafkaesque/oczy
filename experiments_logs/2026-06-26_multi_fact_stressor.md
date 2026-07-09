# Multi-Fact Turn Stressor — 2026-06-26

## Question

Can a single long turn store both a novel fact and a correction-styled fact,
and can architecture H (hybrid consolidation-strength modulation) improve
"co-retention" over architecture S (scalar stats gate)?

## Method

Created `src/oczy/experiments/multi_fact_stressor.py`:

- Builds a 512-token turn with filler and two facts:
  - Fact A (novel): "The codeword for project alpha is skylark."
  - Fact B (correction-style): "Correction: the codeword for project beta is not raven, it is rook."
- Processes the turn through a CortexAgent with the ingestion pipeline active:
  - chunker: `fixed-window`, 64 tokens
  - salience: `lexical-novelty`
  - embedder: `same-lm` (mock driver)
  - aggregator: `stats`
- Forces consolidation via `agent.consolidate()`.
- Queries each fact and checks target substrings:
  - Query A: expect "skylark"
  - Query B: expect "rook"
- Supports `--mode scalar` and `--mode hybrid` to toggle
  `DigestiveGateConfig.use_hybrid_consolidation`.

## Results (mock driver)

| mode | recall_a | recall_b | co_recall | traces | embedding_calls | consolidation_strength | cold_drift |
|---|---|---|---|---|---|---|---|
| scalar | 0 | 0 | 0 | 3 | 4 | 1.00 | 0.10 |
| hybrid | 0 | 0 | 0 | 3 | 4 | 6.39 | 0.64 |

Both modes retain 0/0 facts with the mock driver, but the mechanical effect of
hybrid mode is visible: consolidation strength is scaled by `(1 + drift_max)`
capped at 10.0, and cold drift is correspondingly larger.

## Interpretation

- The mock driver's embeddings are deterministic hash-based vectors; they
  distinguish chunks but carry no semantic content, so retrieval cannot work
  even when the correct chunk is stored. This is the same limitation as the
  mock needle sweep.
- The stressor *does* exercise the architecture S vs H difference on the
  consolidation-strength axis, which is meaningful for real-driver evaluation.
- Lexical-novelty keeps only 3 traces for the entire 512-token turn — the
  embedding cost is already very low.

## Implication for architecture

The multi-fact probe is now ready to run against the real LFM2.5 driver. If
the real driver shows `co_recall > 0`, architecture H and S can be compared on
memory cost vs retention. If the mock behavior is representative, H should
produce stronger persistent traces for high-drift (correction) chunks.

## Open questions

1. Does the real-driver version of this probe show non-zero co_recall?
2. Does hybrid mode improve co_recall at the same memory delta?
3. Would boosting the novel fact's chunk persistence (not just strength) help?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `0f566c4` — Add multi_fact_stressor.py probe and tests.
- `b3e1488` — Update experiments/logs/SUMMARY.md with run #85 result.
