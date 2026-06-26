# Configurable Hybrid Consolidation Cap — 2026-06-27

## Question

Does the 10.0 consolidation-strength cap mask architecture H's effect in the
multi-fact stressor? If the cap is removed, does H produce stronger persistent
traces and better recall?

## Method

Added `--hybrid-cap` to `src/oczy/experiments/multi_fact_stressor.py`.
`--hybrid-cap 0` disables the cap. Ran real-driver S vs H auto-consolidate
probes at length 512, with and without reserved-position prefix.

## Results

### Without prefix (cvec-only)

| mode | consolidation_strength | cold_drift | co_recall |
|---|---|---|---|
| scalar | 10.00 | 0.867 | 0 |
| hybrid (uncapped) | 35.99 | 0.867 | 0 |

### With prefix

| mode | consolidation_strength | co_recall |
|---|---|---|
| scalar | 10.00 | 1 |
| hybrid (uncapped) | 35.99 | 1 |

## Interpretation

- Removing the cap exposes a real architecture difference: hybrid produces
  ~3.6x stronger consolidation strength than scalar on the same turn.
- However, the extra strength does **not** increase `cold_drift` or exact-token
  recall. The cvec-only path remains blocked at 0/0; the prefix path remains at
  1/1 regardless of mode.
- This suggests the extra consolidation strength is either:
  1. Saturated inside `agent.consolidate()` before affecting trace strength,
  2. Detrimental or neutral rather than helpful for exact recall, or
  3. Manifesting on a metric we are not yet measuring (e.g. memory-per-byte).

## Implication for architecture

Hybrid consolidation modulation is mechanically effective, but the cap of 10.0
was hiding that effect. Whether the effect is *useful* requires a non-exact
recall metric or a memory-cost metric. The current probe design does not
reward higher strength; it only checks exact target tokens, which cvecs cannot
force.

## Open questions

1. Does higher consolidation strength reduce the number of traces needed for
   equivalent recall, improving memory-per-byte?
2. Does it improve domain-level or paraphrase recall?
3. Is there an optimal cap, or is unbounded H destabilizing at high drift?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `48e85f8` — Add configurable --hybrid-cap.
- `ae14feb` — Update SUMMARY.md with run #93 result.

## Run

Run #93: benchmark `code_qa_accuracy=1.0`, fast suite `302 passed`.
