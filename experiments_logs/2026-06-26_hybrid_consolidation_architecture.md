# Architecture S vs H: Hybrid Consolidation Boost — 2026-06-26

## Question

Can we implement architecture H (hybrid) from the design matrix: route organ
weights through the scalar digestive gate, but let `TurnDigest` statistics
modulate consolidation strength? And does it differ from scalar stats-only
(architecture S)?

## Method

- Added `DigestiveGateConfig.use_hybrid_consolidation` (default `False`).
- Modified `CortexAgent.metabolize()` to capture the `TurnDigest` returned by
  `IngestionPipeline.process()`.
- Modified `CortexAgent.turn()`: when the pipeline is active and
  `use_hybrid_consolidation` is on, scale the consolidation `strength` by
  `(1.0 + digest.drift_max)`, capped at `10.0`.
- Updated `run_curriculum.py` to propagate `use_ingestion_pipeline`,
  `ingestion`, and `use_hybrid_consolidation` from `--config` JSON into the
  mock/real CortexAgent.
- Added a unit test verifying that hybrid mode produces a higher consolidation
  strength than scalar mode.

Ran mock-driver organism curriculum stages 0-1 with:

1. Pipeline enabled, `use_hybrid_consolidation=False` (architecture S).
2. Pipeline enabled, `use_hybrid_consolidation=True` (architecture H).

## Results

| mode | retention | transfer | memory delta |
|---|---|---|---|
| S | 0.88 | 1.00 | +15.7KB / +3.3KB |
| H | 0.88 | 1.00 | +15.7KB / +3.3KB |

The results are identical because the default `run_curriculum.py` command has
`auto_consolidate=False` and the curriculum episodes are single sentences, so
chunking and consolidation never fire.

## Interpretation

- The hybrid hook is in place and mechanically correct (unit test confirms
  strength is scaled).
- A meaningful S vs H comparison needs multi-chunk turns where consolidation
  actually runs; the current organism curriculum is too coarse-grained.
- Architecture H's value proposition appears only when long turns produce
  within-turn novelty that should strengthen cold persistence.

## Implication for architecture

We now have all four scaffold axes (chunking, salience, embedder choice, gate
routing) plus the hybrid consolidation knob. The next discriminating benchmark
is a **multi-fact turn** stressor where two salient facts are buried in filler
and the agent must retain both after a single auto-consolidation event.

## Open questions

1. Does the multi-fact stressor show H > S when consolidation is forced?
2. Is `drift_max` the right statistic to scale, or should it be
   `novelty_spread` or a combination?
3. Should hybrid modulation also affect hippocampus write priority (e.g.
   persist high-drift chunks longer) in addition to consolidation strength?

## Artifacts

- `src/oczy/experiments/digestive_gate.py`
- `src/oczy/experiments/cortex_agent.py`
- `src/oczy/experiments/organism_curriculum/run_curriculum.py`
- `src/oczy/experiments/tests/test_ingestion.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `f78f43d` — Add hybrid consolidation-strength modulation.
- `002c45e` — Update experiments/logs/SUMMARY.md with run #84 result.
