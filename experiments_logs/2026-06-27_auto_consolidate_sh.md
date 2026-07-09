# Auto-Consolidate S vs H Probe — 2026-06-27

## Question

Can an auto-consolidate path in the multi-fact stressor behaviorally
discriminate architecture S (scalar) from architecture H (hybrid consolidation
strength modulation)?

## Method

Added `--auto-consolidate` to `src/oczy/experiments/multi_fact_stressor.py`.
When active:

- The agent's `auto_consolidate` config is set to True.
- `consolidation_pressure_threshold` is lowered to 0.05 so a single 512-token
  high-drift turn can trigger consolidation.
- After `metabolize()`, if `agent.should_consolidate()` is True, the probe runs
  `agent.consolidate(strength=...)`. For H mode, strength scales by
  `(1.0 + digest.drift_max)` capped at 10.0.
- The result includes `auto_consolidated` boolean.

Real-driver runs at length 512, no prefix, mode scalar and hybrid.

## Results

| mode | auto_consolidated | cold_drift | consolidation_strength | recall_a | recall_b | co_recall |
|---|---|---|---|---|---|---|
| scalar | 1 | 0.867 | 10.0 | 0 | 0 | 0 |
| hybrid | 1 | 0.867 | 10.0 | 0 | 0 | 0 |

## Interpretation

- Both modes auto-consolidate because the lowered threshold is easily crossed.
- Both hit the 10.0 strength cap, so hybrid's additional scaling has no effect.
- Exact-token recall remains 0/0 because cvec-only consolidation cannot force
  target tokens.

The S vs H difference is masked by the cap and by the exact-token limitation.

## Implication for architecture

The multi-fact stressor in its current form cannot discriminate S vs H. To do
so, one of these changes is needed:

1. Remove or raise the 10.0 consolidation-strength cap so hybrid can scale
   beyond scalar.
2. Use multi-turn pressure accumulation so scalar and hybrid diverge in
   *when* consolidation fires.
3. Measure domain-level recall (e.g. whether the answer mentions the right
   project) rather than exact target tokens, since cvec steering shifts domain.
4. Switch the exact-recall path to hippocampus-derived prefixes and measure
   whether hybrid produces better/compressed prefixes.

## Open questions

1. Does removing the strength cap destabilize generation or improve recall?
2. Does hybrid mode save memory (fewer traces for equivalent recall) when the
   cap is removed?
3. Is domain-level recall in the multi-fact probe a more sensitive metric?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `a2814e7` — Add --auto-consolidate mode.
- `fcbb9cd` — Update SUMMARY.md with run #92 result.

## Run

Run #92: benchmark `code_qa_accuracy=1.0`, fast suite `301 passed`.
