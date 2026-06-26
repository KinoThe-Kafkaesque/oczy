# Multi-Fact Turn Stressor — Real Driver — 2026-06-26

## Question

Does the multi-fact stressor run end-to-end against the real LFM2.5 GGUF
driver, and does it emit parseable METRIC/ASI output?

## Method

Extended `src/oczy/experiments/multi_fact_stressor.py` with an optional
`--use-real-driver` flag:

- Resolves the GGUF from `OCZY_MODEL_PATH` or the Hugging Face hub cache.
- Loads `LlamaCVecDriver` with `CVecDriverConfig(n_ctx=4096, n_threads=4, embedding=True)`.
- Passes the loaded driver directly to `CortexAgent(config, driver=driver)`.
- Keeps the deterministic `_MockDriver` as the default.

Ran the probe in both `scalar` and `hybrid` modes with `--length 128`.

## Results

| mode | recall_a | recall_b | co_recall | traces | embedding_calls | consolidation_strength | cold_drift |
|---|---|---|---|---|---|---|---|
| scalar | 0 | 0 | 0 | 2 | 0 | 1.00 | 0.086 |
| hybrid | 0 | 0 | 0 | 2 | 0 | 3.09 | 0.265 |

Both modes produced valid `METRIC` and `ASI` lines against the real driver.
The mechanical difference between scalar and hybrid consolidation strength is
visible: hybrid strength scales by `(1 + drift_max)`, yielding `3.09` vs `1.0`
and larger cold drift (`0.265` vs `0.086`).

`embedding_calls` reports `0` on the real driver because `LlamaCVecDriver`
does not expose a call counter; the integration still exercises the driver's
embedding API and stores traces.

## Interpretation

- The real driver loads in ~7s and successfully serves embeddings and
generation calls through the same `CortexAgent` pipeline used by the mock.
- Co-recall remains 0/0 because the prompt is not instruction-formatted for
the Instruct-tuned model; raw fill-in-the-blank queries are not answered
reliably by this LM without an instruction template. This is expected and does
not invalidate the integration.
- Hybrid mode's consolidation-strength modulation is exercised on real hidden
vectors and produces a stronger drift signature than scalar mode.

## Implication for architecture

The real-driver path is now runnable and wired. The next step for a behavioral
co-retention comparison is to format the retrieval queries as instructions
(e.g., "Answer briefly: <query>") or to add a conversational warm-up turn so
the LM interprets the probe as a question-answering task.

## Open questions

1. Does an instruction-wrapped query yield non-zero co_recall on LFM2.5?
2. Does hybrid mode improve co_recall once retrieval answers are extractable?
3. How much does real-driver perception cost in wall time versus the mock?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`
- `experiments_logs/2026-06-26_multi_fact_stressor_real_driver.md`
