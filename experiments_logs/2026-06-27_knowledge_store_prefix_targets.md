# KnowledgeStore-Guided Hippocampus Prefix Extraction — 2026-06-27

## Question

Can the hippocampus-derived ReservedPosition path benefit from the
KnowledgeStore when a recalled fact does not have a hand-seeded `reserved_token`?

## Method

Added `KnowledgeStore.get_prefix_targets(query)` which returns target strings
from the top recalled facts. Preference order:

1. `metadata["prefix_target"]` if present.
2. `metadata["reserved_token"]` if present.
3. The fact `"value"` otherwise.

Facts must meet the same `min_score` threshold used by `format_context()` and
`get_reserved_position()`.

Added `CortexAgentConfig.knowledge_store_supplies_prefix_targets` (default
False). When enabled, `CortexAgent.articulate()` collects targets from the
KnowledgeStore and passes them as `prefix_targets` to
`_derive_reserved_position_from_hippocampus()` whenever no explicit
`ReservedPosition` was already set by `get_reserved_position()`.

Added tests:
- `test_knowledge_store_get_prefix_targets`: verifies values and explicit
  `prefix_target` metadata are returned.
- `test_hippocampus_prefix_uses_knowledge_store_targets`: mock driver shows a
  non-reserved-token fact still yields a `ReservedPosition` via the hippocampus
  helper when the flag is enabled.

## Results

- KnowledgeStore unit tests: 16 passed.
- CortexAgent tests: 15 passed.
- CortexAgent reserved-position tests: 4 passed.
- Fast suite: 315 passed, 26 deselected.
- `ruff check` clean.
- Benchmark `code_qa_accuracy=1.0` (run #102).

## Interpretation

The KnowledgeStore is no longer limited to exact-token steering only for facts
with `reserved_token`. Any recalled fact can now guide hippocampal
snippet-extraction via `prefix_targets`, extending exact-token recall potential
to the broader codebase-QA corpus.

The flag is default-off, so existing behavior is unchanged. When enabled, the
precedence remains: explicit `ReservedPosition` > hippocampus-derived prefix >
none.

## Limitations

- The benchmark was already at `code_qa_accuracy=1.0`; no improvement was
  observed, because the existing reserved-token facts cover the questions.
- A real workload showing improved recall on facts without `reserved_token` has
  not been run.
- `get_prefix_targets` currently returns the top fact(s') values; a long fact
  value could produce a very long prefix if the truncation window also captures
  filler text.

## Next steps

1. Run codebase-QA with `knowledge_store_supplies_prefix_targets=True` and
   compare recall_lift.
2. Measure IdentityHypernetwork adapter effects on the multi-fact probe.
3. Close the benchmark gap on exact-token consolidation uptake.

## Artifacts

- `src/oczy/experiments/codebase_qa/knowledge_store.py`
- `src/oczy/experiments/cortex_agent.py`
- `src/oczy/experiments/codebase_qa/test_knowledge_store.py`
- `src/oczy/experiments/tests/test_cortex_agent.py`

## Commits

- `311cf19` — Integrate KnowledgeStore prefix_targets.
- `b1311d6` — Update SUMMARY.md with run #102 result.

## Run

Run #102: benchmark `code_qa_accuracy=1.0`, fast suite `315 passed`.
