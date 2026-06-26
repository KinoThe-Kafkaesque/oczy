# Multi-Fact Stressor: Same-LM vs Foreign-MiniLM — 2026-06-27

## Question

Does the foreign-MiniLM embedder preserve multi-fact recall quality on the real
LFM2.5 driver when a reserved-position prefix is used?

## Method

Ran `src/oczy/experiments/multi_fact_stressor.py` with `--use-real-driver
--use-prefix --length 128`, comparing `embedder: "same-lm"` and
`embedder: "foreign-minilm"`.

## Results

| embedder | co_recall | wall seconds |
|---|---|---|
| same-lm | 1/1 | 8.8 |
| foreign-minilm | 1/1 | 25.0 |

## Interpretation

- Foreign-MiniLM preserves recall quality in this prefix-dominated task.
- It is slower than same-lm even with MiniLM model caching, because the
  per-chunk torch CPU encode overhead exceeds cached llama.cpp embedding lookups
  for the small number of surviving chunks.
- This confirms the embedder-fork conclusion: in this environment, same-lm is
  both faster and sufficient.

## Dependency note

An attempt to add an ONNX Runtime foreign embedder (`optimum[onnxruntime]`)
failed because `optimum` requires `huggingface-hub<1.0`, while the repo's `lm`
dependency group pins `huggingface-hub>=1.0`. The conflicting packages were
uninstalled and the environment restored; the fast suite and benchmark still
pass.

## Open questions

1. Is there a foreign embedder backend compatible with the repo's dependency
   pins (e.g. a custom ONNX model loaded directly via `onnxruntime`)?
2. Does foreign-MiniLM show an advantage at much higher chunk counts or on
   different hardware?

## Run

Run #91: benchmark `code_qa_accuracy=1.0`, fast suite `300 passed`.

## Commits

- `3f7a239` — Update SUMMARY.md with run #91 results.
