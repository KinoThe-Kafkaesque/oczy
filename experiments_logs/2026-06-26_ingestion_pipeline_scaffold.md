# Ingestion Pipeline Scaffold — 2026-06-26

## Question

Can we build a single ingestion scaffold that turns a long utterance into
chunk-level traces and a single within-turn digest, so that future ablations
(chunking strategy, salience filter, embedder choice, gate architecture) are
config flags rather than separate implementations?

## Method

Add a new `IngestionPipeline` module (`src/oczy/experiments/ingestion.py`)
upstream of `CortexAgent.metabolize()`.  The pipeline is selected by string
flags in a single config dict and emits:

- `ChunkSignal` objects → stored directly in `neural_hippocampus` (bypassing
  the digestive gate, per the spec)
- `TurnDigest` object → consumed by `DigestiveGate.ingest_digest()`

Implemented stages:

| stage | flag options |
|---|---|
| Chunker | `fixed-window`, `sentence`, `paragraph`, `recursive` |
| Salience | `pass-through`, `correction-marker`, `lexical-novelty`, `centroid-cosine` |
| Embedder | `none`, `same-lm`, `identity` (foreign placeholder) |
| Aggregator | `stats` |
| Observation mode | `parallel`, `sequential` |

`DigestiveGate.ingest_digest(digest)` maps the digest back onto the existing
scalar `ingest()` surface:

- `drift` ← `digest.drift_max`
- `correction_signal` ← `digest.correction_fraction`
- `novelty` ← `density + novelty_spread`
- `critic_correction_prob` ← `digest.critic_prob_max`

The entire pipeline is gated by `CortexAgentConfig.use_ingestion_pipeline` and
defaults to `False`, preserving the existing single-trace metabolism path.

## Results

- `src/oczy/experiments/ingestion.py`: 660 lines, pure NumPy, no new
  dependencies.
- `src/oczy/experiments/tests/test_ingestion.py`: 19 tests covering
  chunking, salience, embedding drift, observation modes, top-K pruning, gate
  digest mapping, and CortexAgent end-to-end wiring.
- Related verification:
  - `test_ingestion.py`: 19 passed
  - `test_digestive_gate.py`: 13 passed
  - `test_cortex_agent.py`: 11 passed
  - Full fast suite: `290 passed, 17 deselected`
  - `ruff check` clean on modified files
- `bash autoresearch.sh` run #79: `code_qa_accuracy = 1.0` with the pipeline
  gated off (default).

### Smoke test: correction-marker salience on a long utterance

```text
"This is a long sentence that should be split into chunks. "
"No, this part is a correction and should score high."
```

With a 10-token fixed window:

| chunk | salience | is_correction |
|---|---|---|
| `This is a long sentence that should be split into` | 0.00 | False |
| `split into chunks. No, this part is a correction a` | 1.00 | True |
| `correction and should score high.` | 0.00 | False |

`TurnDigest` reports `correction_fraction=0.33` and
`novelty_spread=0.47`.

## Interpretation

- A single scaffold can support the full design matrix.
- Within-turn resolution is now available to the digestive gate without
  changing the gate's scalar surface.
- Chunking and salience filtering are decoupled from embedding cost: the
  salience stage runs before embedding and can cull chunks before expensive
  same-LM forwards.

## Implication for architecture

This is row 4 of the ablation ladder (chunking + salience + stats gate).
Next steps:

1. Build needle-per-turn stressors with position and length sweeps.
2. Run ladder rows 0–4 to isolate truncation vs chunking vs salience vs gate
   resolution.
3. Decide the embedder fork: same-LM forwards vs foreign CPU embedder +
   learned projection.

## Trade-offs

- `same-lm` embedder is faithful but expensive per chunk; the salience filter
  must be good enough to keep embedding counts sublinear in turn length.
- Foreign embedder + projection is cheaper but adds a trainable projection
  layer and potential recall-fidelity loss.
- Sequential observation mode is more faithful to reading order but creates
  order-dependence; parallel mode is simpler and cheaper.

## Open questions

1. What tokenizer should the fixed-window chunker use for token spans?
   Current implementation uses a whitespace estimate; real token spans would
   improve throughput accounting.
2. Should the salience filter share a running centroid across turns or reset
   per turn?
3. How does `TurnDigest` interact with `auto_consolidate` pressure
   calculation?

## Artifacts

- `src/oczy/experiments/ingestion.py`
- `src/oczy/experiments/digestive_gate.py` (added `ingest_digest`)
- `src/oczy/experiments/cortex_agent.py` (added `use_ingestion_pipeline`)
- `src/oczy/experiments/tests/test_ingestion.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `ffbe2b9` — Add configurable IngestionPipeline scaffold.
- `928eb3c` — Update experiments/logs/SUMMARY.md with run #79 result.
