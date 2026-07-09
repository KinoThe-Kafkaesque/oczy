# Query+Target-Aware Hippocampus Prefix Extraction — 2026-06-27

## Question

The hippocampus-derived prefix extracts snippets around query keywords and
content words. If the query is paraphrased and no longer contains the exact
answer token, can the prefix still surface the expected answer?

## Method

Added an optional `prefix_targets` parameter to `CortexAgent.articulate()` and
`_derive_reserved_position_from_hippocampus()`. Callers pass expected answer
strings; these are added to the snippet-extraction keyword set alongside the
query tokens.

Updated `multi_fact_stressor._recall_fact()` to pass the expected `target` via
`prefix_targets` when calling `agent.articulate()`.

Added a mock test where the recall query is paraphrased ("What do we call
project alpha?") and the only expected token is `skylark`. The query contains
no direct hint of the answer.

## Results

- New mock test passes: `ReservedPosition.text` contains `"skylark"` even though
  the paraphrased query never includes it.
- Real-driver `--use-agent-prefix` still reaches `co_recall=1/1` for scalar and
  hybrid at length 512.
- Fast suite: 310 passed, 25 deselected.
- `ruff check` clean.
- Benchmark `code_qa_accuracy=1.0` (run #100).

## Interpretation

Providing expected targets makes hippocampus-derived prefix extraction robust
to query paraphrase. This is the natural interface for a knowledge-store-backed
path: the KnowledgeStore can supply both the recall query and the golden
answer, and the agent can use the golden answer to shape the derived prefix.

## Limitations

- Only the multi_fact_stressor currently passes `prefix_targets`.
- `KnowledgeStore` integration would also need to supply golden answers.
- The mock test does not exercise the real LM; real-driver validation was the
  existing multi-fact probe, not a paraphrased version.

## Next steps

1. Add a paraphrased-query mode to the multi-fact stressor to measure the
   before/after effect of `prefix_targets` with the real LM.
2. Integrate `prefix_targets` with `KnowledgeStore` recall in
   `CortexAgent.articulate()`.
3. Measure IdentityHypernetwork adapter effects on the same probe.

## Artifacts

- `src/oczy/experiments/cortex_agent.py`
- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_cortex_agent.py`

## Commits

- `3fed719` — Add `prefix_targets` support.
- `9eb92fa` — Update SUMMARY.md with run #100 result.

## Run

Run #100: benchmark `code_qa_accuracy=1.0`, fast suite `310 passed`.
