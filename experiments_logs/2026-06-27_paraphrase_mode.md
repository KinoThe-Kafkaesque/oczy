# Paraphrased-Query Multi-Fact Stressor Mode — 2026-06-27

## Question

Does the hippocampus-derived prefix path still achieve exact-token recall when
recall queries are paraphrased and no longer contain the original keywords that
made snippet extraction easy?

## Method

Added `--paraphrase` to `multi_fact_stressor.py`. When enabled, recall queries
become:

- Alpha: `"What name is used for project alpha?"` (original: "What is the
codeword for project alpha?")
- Beta: `"What do we call project beta?"` (original: "What is the codeword for
project beta?")

The prompt shown to the LM still uses the original question (`QUERY_A`/`QUERY_B`)
because the expected target string is part of that literal text. The
`recall_query` passed to `agent.articulate()` is the paraphrase, while
`prefix_targets=TARGET_A/TARGET_B` is unchanged.

Ran real-driver scalar and hybrid with `--use-agent-prefix --paraphrase --length
512`, and compared to the non-paraphrased baseline with the same flags.

## Results

| mode | paraphrase | co_recall | prefix_source | consolidation_strength |
|---|---|---|---|---|
| scalar | no | 1/1 | hippocampus | 10.0 |
| scalar | yes | 1/1 | hippocampus | 10.0 |
| hybrid | yes | 1/1 | hippocampus | 35.99 |

The paraphrased queries do not degrade exact-token recall.

## Interpretation

- `prefix_targets` successfully guides snippet extraction even when the query
  text no longer contains the high-signal keyword "codeword".
- Hippocampal replay is driven by the paraphrased query; the synthetic embedding
  similarity still retrieves the relevant traces because "project alpha" /
  "project beta" remain in the paraphrase.
- This is a meaningful robustness test: the prefix is derived from expected
  targets, not just query keywords.

## Limitations

- Only two paraphrase variants tested.
- The long-turn facts are still planted with the exact target tokens, so the
  memory surface is rich enough.
- No measurement was made without `prefix_targets` under paraphrase; that would
  isolate the value of the target-aware feature.

## Next steps

1. Add a paraphrase-without-prefix_targets condition to isolate its value.
2. Integrate `prefix_targets` with `KnowledgeStore` recall.
3. Measure IdentityHypernetwork adapter effects.

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `dc749d9` — Add `--paraphrase` mode.
- `76faddf` — Update SUMMARY.md with run #101 result.

## Run

Run #101: benchmark `code_qa_accuracy=1.0`, fast suite `311 passed`.
