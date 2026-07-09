# Needle-Per-Turn Stressor — 2026-06-26

## Question

Does the new `IngestionPipeline` retrieve a fact buried deep in a long turn
when the status-quo single-embed metabolism would truncate or lose it?

## Method

Added `src/oczy/experiments/tests/test_ingestion_needle.py` with a mock-driver
`CortexAgent` and three slow tests:

1. **Baseline**: `use_ingestion_pipeline=False`, 512-token turn, needle at
   position 0.8. Expect ≤1 hippocampal trace and recall 0.
2. **Pipeline**: `use_ingestion_pipeline=True`, fixed-window chunker
   (64 tokens), pass-through salience, same-LM embedder, stats aggregator.
   Expect multiple traces and recall 1.
3. **Cost scaling**: pipeline active, assert embedding calls match the
   fixed-window chunk count.

Mock driver returns deterministic but distinct hidden vectors correlated with
text so the needle has its own retrievable representation.

Needle used: `"The secret codeword is octarine."`

## Results

- All three slow tests pass in ~0.24s.
- Baseline: stores ≤1 trace, fails to retrieve the needle.
- Pipeline: stores multiple chunk traces, successfully retrieves the needle.
- Embedding calls scale with chunk count (perceive + one per surviving chunk).
- `bash autoresearch.sh` run #80: `code_qa_accuracy=1.0` with the pipeline
  gated off by default.

## Interpretation

Chunking + per-chunk storage is necessary for deep-needle recall when the
overall turn length exceeds the useful single-embed context window. The
salience filter will become the cost gate for expensive same-LM embedding.

## Open questions

1. What does the position sweep look like? Needle at 0.0, 0.25, 0.5, 0.75, 1.0.
2. How does performance change with `salience=correction-marker` culling non-needle chunks?
3. What is the cost-vs-recall curve as the salience top-K is reduced?

## Artifacts

- `src/oczy/experiments/tests/test_ingestion_needle.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `bbd3474` — Add needle-per-turn stressor tests.
- `7597e36` — Update experiments/logs/SUMMARY.md with run #80 result.
