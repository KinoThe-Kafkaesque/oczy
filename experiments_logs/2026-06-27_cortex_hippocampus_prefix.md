# CortexAgent Hippocampus-Derived ReservedPosition Prefix — 2026-06-27

## Question

Can the hippocampus-derived prefix mechanism proven in the stressor wrapper be
moved into the live `CortexAgent` so any articulation with a recall query can
optionally receive a_RESERVEDPosition generated from stored memory?

## Method

Added `use_hippocampus_prefix: bool = False` to `CortexAgentConfig` in
`src/oczy/experiments/cortex_agent.py`.

Added `_derive_reserved_position_from_hippocampus(query)` which:
1. Replays top-k episodes from `self.neural_hippocampus.reinforce(query, k=3)`.
2. Extracts salient snippets around query tokens and non-stop content words.
3. Concatenates, truncates to 128 tokens, and returns a `ReservedPosition` with
   `source="hippocampus"`.

Wired the helper into `articulate()` after the knowledge_store reserved-position
block and before the empty-prompt guard. Explicit knowledge_store reserved
tokens still take precedence.

Added tests:
- `test_hippocampus_prefix_derives_from_stored_episode`: verifies a stored
  episode yields a `ReservedPosition` with the expected text and source.
- `test_knowledge_store_prefix_takes_precedence_over_hippocampus`: verifies the
  knowledge_store reserved_token wins when both sources are available.

## Results

- `src/oczy/experiments/tests/test_cortex_agent.py`: 13 passed
- `src/oczy/experiments/tests/test_cortex_agent_reserved.py`: 4 passed
- Fast suite: 308 passed, 24 deselected
- `ruff check` clean on changed files
- Benchmark `code_qa_accuracy=1.0` (run #97)

## Interpretation

The live `CortexAgent` now has two reserved-position sources:
1. **KnowledgeStore** — hand-seeded exact tokens via `metadata["reserved_token"]`.
2. **Hippocampus** — memory-derived snippets extracted from replayed episodes.

This is the next step toward closing the exact-recall loop in the live agent,
not just the stressor wrapper. The default path is unchanged, so existing
benchmarks are unaffected.

## Open questions

1. Does the live `use_hippocampus_prefix` path improve multi-fact co_recall in a
   real-driver probe that does not use the stressor's hand-built prefix logic?
2. Should snippet extraction be query-target aware (e.g. accept expected answers
   from a KnowledgeStore match) instead of keyword-based?
3. How does this interact with IdentityHypernetwork adapters at articulation time?

## Artifacts

- `src/oczy/experiments/cortex_agent.py`
- `src/oczy/experiments/tests/test_cortex_agent.py`

## Commits

- `9b7f454` — Add `use_hippocampus_prefix` and helper.
- `f9b3fd8` — Update SUMMARY.md with run #97 result.

## Run

Run #97: benchmark `code_qa_accuracy=1.0`, fast suite `308 passed`.
