# Foreign vs Same-LM Embedder Fork — 2026-06-26

## Question

Should the ingestion pipeline use expensive same-LM embeddings or a cheaper
foreign sentence embedder projected into `n_embd`? The cost difference matters
for long-turn chunking, but the projection layer may degrade recall.

## Method

The real venv has no `sentence-transformers`, `torch`, or `sentencepiece`, so
we cannot use MiniLM/BGE out of the box. Instead, we added a `mock-foreign`
embedder to `src/oczy/experiments/ingestion.py`:

- Builds a deterministic character-trigram histogram.
- Projects it to `n_embd` through a lazy-learned seeded linear layer.
- L2-normalizes the result.

This exercises the architecture (foreign feature space → projection → cortex)
without new dependencies.

We ran the 512-token needle-position sweep for:

1. `same-lm` + `lexical-novelty` salience
2. `mock-foreign` + `lexical-novelty` salience

## Results

| embedder | mean_recall | total_embedding_calls |
|---|---|---:|
| same-lm | 1.00 | 14 |
| mock-foreign | 1.00 | 5 |

Both configurations achieve perfect needle recall across positions
`[0.0, 0.25, 0.5, 0.75, 1.0]`. The mock-foreign embedder has no driver
embedding cost because it bypasses the LM entirely.

## Interpretation

- The foreign + projection *architecture* can work: a learned projection from
  a non-LM feature space into `n_embd` does not inherently destroy retrieval
  fidelity in this synthetic test.
- The `mock-foreign` embedder is too good a surrogate (character trigrams
  perfectly distinguish chunks here). A real MiniLM/BGE embedder would need an
  actual trained projection and evaluation on semantic paraphrase.
- Given that `lexical-novelty` salience already reduces same-LM calls to ~14 at
  512 tokens (and only 8 at 4096 tokens), same-LM may be practical for the
  foreseeable turn lengths in this curriculum.

## Implication for architecture

We can defer the real-model foreign-embedder fork. The priority now is:

1. Ensure the ingestion pipeline does not regress Stage 0/1 when enabled.
2. Compare architecture S (scalar stats gate) vs H (hybrid: stats gate for
   routing, richer signals for consolidation strength / chunk persistence).

If same-LM + lexical-novelty stays cheap enough, the foreign fork is
optimization rather than architecture.

## Caveats

- Mock-foreign is not MiniLM/BGE. Its perfect recall does not prove a real
  foreign embedder will match same-LM fidelity.
- The projection layer is randomly initialized and only normalizes; it is not
  trained on downstream tasks.

## Artifacts

- `src/oczy/experiments/ingestion.py` (`MockForeignEmbedder`)
- `src/oczy/experiments/tests/test_ingestion.py`
- `src/oczy/experiments/needle_sweep.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `87ff263` — Add mock-foreign embedder option.
- `69a3790` — Update experiments/logs/SUMMARY.md with run #83 result.
