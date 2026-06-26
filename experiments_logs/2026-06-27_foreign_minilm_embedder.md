# Foreign MiniLM Embedder Integration — 2026-06-27

## Question

Can the IngestionPipeline use a real CPU sentence embedder (MiniLM) as a
cheaper alternative to same-LM embeddings, while preserving recall through a
learned projection into `n_embd`?

## Method

Added `MiniLMEmbedder` to `src/oczy/experiments/ingestion.py`:

- Lazy import of `sentence_transformers.SentenceTransformer` so the rest of the
  code still works when the package is absent.
- Default model `all-MiniLM-L6-v2` (384-dim output), configurable via
  `foreign_model_name`.
- Projects foreign vectors into `n_embd` via a lazy-learned random normal
  projection matrix, reusing the same pattern as `MockForeignEmbedder`.
- Added `embedder: "foreign-minilm"` to the IngestionPipeline factory.
- Added `sentence-transformers>=3` to the `lm` optional dependency group in
  `pyproject.toml`.
- Added a unit test using `pytest.importorskip("sentence_transformers")`.

Ran a synthetic 512-token needle sweep with:
1. `foreign-minilm` + `lexical-novelty`
2. `same-lm` + `lexical-novelty`

## Results

| embedder | mean_recall | embedding_calls_total | notes |
|---|---|---|---|
| foreign-minilm | 1.00 | 5 | loads real MiniLM once |
| same-lm | 1.00 | 14 | mock driver, artificially cheap |

## Interpretation

- The foreign-MiniLM embedder integrates cleanly and reaches perfect recall on
  the synthetic needle sweep. The learned projection from 384-dim MiniLM space
  into the mock `n_embd` does not destroy the retrieval signal for this task.
- The embedding-call count (5 vs 14) is misleading here because the mock driver
  makes `same-lm` embedding artificially fast; on a real LFM2.5 driver the cost
  ratio would favor foreign-MiniLM much more strongly.
- Adding `sentence-transformers` pulls in `torch` and CUDA wheels; it is an
  optional dependency group, not a hard requirement.

## Implication for architecture

The embedder fork is now instrumented. A real-driver comparison on the same
needle sweep is the decisive experiment: it will measure whether foreign-MiniLM
saves wall-clock relative to LFM2.5 `peek_embedding` forwards without dropping
recall.

## Open questions

1. Does foreign-MiniLM match same-LM recall on the real LFM2.5 needle sweep?
2. What is the wall-clock ratio between the two embedders on real hardware?
3. Does the learned projection need training/update during agent lifetime, or
   is a fixed random projection sufficient?
4. How does foreign-MiniLM behave on the multi-fact turn stressor?

## Artifacts

- `src/oczy/experiments/ingestion.py`
- `src/oczy/experiments/tests/test_ingestion.py`
- `pyproject.toml`

## Commits

- `849c1fb` — Integrate optional foreign-minilm sentence embedder.
- `ca8c142` — Update experiments/logs/SUMMARY.md with run #88 result.

## Run

Run #88: benchmark `code_qa_accuracy=1.0`, fast suite `300 passed`.
