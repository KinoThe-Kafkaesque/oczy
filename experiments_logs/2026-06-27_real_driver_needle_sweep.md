# Real-driver needle sweep verification

**Branch:** `autoresearch/session-20260625`  
**Date:** 2026-06-26

Goal: confirm `needle_sweep.py` can load the real LFM2.5 GGUF driver and emit
per-position wall-clock timing.

## Commands run

```bash
uv run python -m src.oczy.experiments.needle_sweep --length 512
```

```bash
uv run python -m src.oczy.experiments.needle_sweep \
  --use-real-driver --length 128 --positions 0.0,0.5
```

```bash
uv run pytest src/oczy/experiments/tests/test_ingestion_needle.py -q
```

```bash
uv run pytest -m "not slow and not requires_model" -q
```

```bash
uv run ruff check src/oczy/experiments/needle_sweep.py \
  src/oczy/experiments/tests/test_ingestion_needle.py
```

## Observed results

Mock sweep:

```
METRIC length=512 position=0.00 recall=0 traces=0 embedding_calls=1 wall_seconds=0.002755
METRIC length=512 position=1.00 recall=0 traces=0 embedding_calls=1 wall_seconds=0.002054
METRIC length=512 mean_recall=0.00 max_recall=0 embedding_calls_total=5 total_wall_seconds=0.011664
ASI config={"use_ingestion_pipeline": false, "ingestion": {}}
```

Real-driver sweep (LFM2.5 GGUF present locally):

```
METRIC length=128 position=0.00 recall=0 traces=0 embedding_calls=0 wall_seconds=2.370353
METRIC length=128 position=0.50 recall=0 traces=0 embedding_calls=0 wall_seconds=1.708054
METRIC length=128 mean_recall=0.00 max_recall=0 embedding_calls_total=0 total_wall_seconds=4.078407
ASI config={"use_ingestion_pipeline": false, "ingestion": {}}
```

Tests:

- `src/oczy/experiments/tests/test_ingestion_needle.py` — 4 passed
- Fast suite (`not slow and not requires_model`) — 300 passed, 22 deselected
- `ruff check` — clean

## Conclusion

`needle_sweep.py` now supports `--use-real-driver` / `--n-ctx`, reuses a single
`LlamaCVecDriver` across positions, and reports `wall_seconds` per position plus
`total_wall_seconds`. The slow/requires_model test skips cleanly when the GGUF is
missing and passes against the local model.
