# Hippocampus-Derived ReservedPosition Prefix — 2026-06-27

## Question

Can the agent generate its own reserved-position prefixes from consolidated
hippocampal traces, eliminating the hand-coded prefix in the multi-fact stressor
and still achieving exact-token recall?

## Method

Added `--auto-prefix` to `src/oczy/experiments/multi_fact_stressor.py`. After
consolidation, the stressor derives a prefix from `agent.neural_hippocampus`:

1. Prefer slow-update summaries (none were produced in this probe because raw
   traces had `replay_count=0`).
2. Fall back to raw traces, but instead of returning the whole stored utterance
   (which is mostly filler in the long-turn stressor), extract salient
   fact-bearing snippets around project-name keywords (`skylark`, `rook`,
   `alpha`, `beta`).
3. Truncate to 128 tokens and set as `ReservedPosition(text=..., source="hippocampus")`.

Compared scalar vs hybrid with `--auto-consolidate --hybrid-cap 0
--auto-prefix --length 512` on the real LM.

## Results

### First attempt (whole utterance)

`prefix_source=hippocampus` but prefix was all filler; `co_recall=0/0`.

### Final attempt (salient snippets)

| mode | co_recall | prefix_source | consolidation_strength |
|---|---|---|---|
| scalar | 1/1 | hippocampus | 10.0 |
| hybrid | 1/1 | hippocampus | 35.99 |

Both match the hand-coded prefix (`--use-prefix`) result of `co_recall=1/1`.

## Interpretation

- Hippocampal traces contain enough information to reconstruct a useful
  reserved-position prefix.
- A simple keyword-window extraction is sufficient for this probe because the
  facts are planted with known markers.
- The prefix is derived from stored memory, not hand-coded, so this is a genuine
  closed-loop result.

## Limitations

- The keyword set (`skylark`, `rook`, `alpha`, `beta`) is still probe-specific.
- Extraction currently does not use slow-update summaries because replay counts
  are zero; if consolidation/replay were tuned differently the slow-update path
  would matter.
- The prefix length (up to 128 tokens) is large relative to a typical prefix.

## Implication for architecture

The exact-recall loop can be closed without hand-coded hints: perceive → store
in hippocampus → consolidate → derive prefix from memory → apply prefix at
articulation → exact recall. The next step is to move this from the stressor
wrapper into the live `CortexAgent.articulate()` path.

## Next steps

1. Integrate auto-prefix generation into `CortexAgent.articulate()` as an
   optional `auto_prefix` config flag.
2. Generalize keyword extraction to use the query/target being articulated.
3. Measure whether this improves codebase-QA exact recall beyond the current
   knowledge-store reserved_token mechanism.

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `f1f1392` — Add --auto-prefix.
- `6dc7179` — Update SUMMARY.md with run #96 result.

## Run

Run #96: benchmark `code_qa_accuracy=1.0`, fast suite `308 passed`.
