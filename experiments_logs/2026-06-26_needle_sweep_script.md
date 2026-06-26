# Needle Position/Length Sweep Script — 2026-06-26

## Question

Can we turn the needle-recall stressor into a runnable benchmark that emits
`METRIC` / `ASI` lines for the autoresearch harness, and what does the baseline
vs. pipeline recall curve look like?

## Method

Added `src/oczy/experiments/needle_sweep.py`.

CLI:

```bash
uv run python -m src.oczy.experiments.needle_sweep \
  --length 512 \
  --positions 0.0,0.25,0.5,0.75,1.0 \
  --config '{"use_ingestion_pipeline": true, "ingestion": {"chunker": "fixed-window", "chunker_window_tokens": 64, "salience": "pass-through", "embedder": "same-lm", "aggregator": "stats"}}'
```

Default needle: `"The secret codeword is octarine."`

## Results

### Baseline (single-embed, no pipeline)

| position | recall | traces | embedding_calls |
|---|---|---|---|
| 0.00 | 0 | 0 | 1 |
| 0.25 | 0 | 0 | 1 |
| 0.50 | 0 | 0 | 1 |
| 0.75 | 0 | 0 | 1 |
| 1.00 | 0 | 0 | 1 |

**mean_recall = 0.00**, total_embedding_calls = 5

### Pipeline (64-token fixed-window, pass-through salience, same-LM embedder)

| position | recall | traces | embedding_calls |
|---|---|---|---|
| 0.00 | 1 | 9 | 10 |
| 0.25 | 1 | 9 | 10 |
| 0.50 | 1 | 1 | 10 |
| 0.75 | 1 | 9 | 10 |
| 1.00 | 1 | 9 | 10 |

**mean_recall = 1.00**, total_embedding_calls = 50

## Interpretation

- The baseline single-embed metabolism stores **zero** traces in the mock setup,
  so it cannot retrieve the needle at any position. (This reflects that the
  digestive gate is conservative with low-drift mock signals.)
- The chunked pipeline stores per-chunk traces and retrieves the needle at
  every tested position.
- The cost is currently linear in chunk count (9 chunks + 1 perceive call per
  position). The next variable to test is whether a salience filter can reduce
  embedding calls while preserving recall.

## Implication for architecture

The sweep script gives us a reproducible way to measure the ablation ladder.
Row 2 (chunking) works for retrieval; row 3 (salience filtering) is the
throughput gate that decides whether same-LM embedding is practical.

## Trade-offs

- `pass-through` salience is honest about the cost of chunking but does not
  reduce embedding calls.
- `correction-marker` salience would keep only the explicit correction chunk
  (cheap) but requires the needle to be phrased as a correction.
- `lexical-novelty` salience is the likely sweet spot for general long turns.

## Open questions

1. What is the recall vs. embedding-calls curve for `lexical-novelty` and
   `centroid-cosine` filters?
2. Does `correction-marker` salience suffice for the organism curriculum
   (where corrections are explicit)?
3. How does the curve change with turn length (1k, 4k, 32k tokens)?

## Artifacts

- `src/oczy/experiments/needle_sweep.py`
- `src/oczy/experiments/tests/test_ingestion_needle.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `a1e9313` — Add needle_sweep.py benchmark script.
- `49469f0` — Update experiments/logs/SUMMARY.md with run #81 result.
