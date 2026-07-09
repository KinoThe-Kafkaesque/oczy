# Real-Driver Needle Sweep — 2026-06-27

## Question

Can the `IngestionPipeline` use a real foreign CPU sentence embedder to reduce
wall-clock time versus same-LM embeddings on the LFM2.5 GGUF driver for the
needle-recall stressor?

## Method

Extended `src/oczy/experiments/needle_sweep.py` to optionally load the real
LFM2.5 GGUF driver (`--use-real-driver`, `--n-ctx`). Added per-position and
total wall-clock timing.

Real-driver runs at length 512, positions `[0.0, 0.25, 0.5, 0.75, 1.0]`,
64-token chunks, `lexical-novelty` salience.

Also cached the MiniLM sentence-transformer model across embedder instances
(after noticing the model was re-initializing for every position).

## Results

### Length 512, cached MiniLM

| embedder | mean_recall | traces | total wall seconds |
|---|---|---|---|
| same-lm | 1.00 | 1-2 | 29.6 |
| foreign-minilm | 1.00 | 1-2 | 42.8 |

Both achieve perfect recall. Foreign-MiniLM is slower at this scale.

## Interpretation

- `lexical-novelty` salience keeps only 1-2 chunks per position at length 512,
  so the total number of embeddings is small.
- `LlamaCVecDriver.peek_embedding` caches embeddings by prompt, and the filler
  chunks are mostly identical across positions; after the first chunk, same-LM
  embedding calls are essentially free.
- Foreign-MiniLM encodes each surviving chunk via torch on CPU. Even with a
  cached model, per-call overhead exceeds cached same-LM lookups in this regime.

This means the naive expectation ("foreign embedder is always cheaper") is
wrong for short/medium turns with high lexical repetition. The cost advantage
appears only when:
1. The turn contains many distinct chunks (cache misses dominate),
2. The foreign embedder is fast (e.g. ONNX/Core ML), or
3. Same-LM embedding is deliberately uncached or batched per-position.

## Implication for architecture

The embedder fork decision depends strongly on input statistics. The current
`foreign-minilm` integration is correct but not a universal win. To see savings,
we must run length 4096 or higher where same-LM does many distinct forward
passes.

## Open questions

1. Does foreign-minilm beat same-lm at length 4096 or 8192?
2. Would `pass-through` salience (embed all chunks) change the ratio?
3. Is there a smaller/faster sentence model (ONNX) that reduces per-chunk
   overhead enough to win at length 512?

## Artifacts

- `src/oczy/experiments/needle_sweep.py`
- `src/oczy/experiments/tests/test_ingestion_needle.py`
- `src/oczy/experiments/ingestion.py` (MiniLM cache)

## Commits

- `fb954fe` — Make needle_sweep runnable against real LlamaCVecDriver; cache MiniLM model.
- `dc408b5` — Update SUMMARY with run #89.

## Run

Run #89: benchmark `code_qa_accuracy=1.0`, fast suite `300 passed`.
