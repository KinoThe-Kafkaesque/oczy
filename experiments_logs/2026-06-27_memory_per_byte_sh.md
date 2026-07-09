# Memory-per-Byte Probe: S vs H under Trace Caps — 2026-06-27

## Question

Does architecture H save memory or improve recall under a trace cap, compared
to scalar (S)?

## Method

Added `memory_bytes` (pickle size of hippocampus) and `--max-traces` pruning
to `multi_fact_stressor.py`. Ran real-driver S vs H auto-consolidate with
`--hybrid-cap 0` and `max-traces=1` or `2`, with and without reserved-position
prefix.

## Results

| max_traces | prefix | mode | co_recall | memory_bytes | consolidation_strength |
|---|---|---|---|---|---|
| 1 | no | scalar | 0/0 | 10,443 | 10.0 |
| 1 | no | hybrid | 0/0 | 10,444 | 36.0 |
| 1 | yes | scalar | 1/1 | 10,443 | 10.0 |
| 1 | yes | hybrid | 1/1 | 10,444 | 36.0 |
| 2 | no | scalar | 0/0 | 19,822 | 10.0 |
| 2 | no | hybrid | 0/0 | 19,822 | 36.0 |
| 2 | yes | scalar | not run | — | — |
| 2 | yes | hybrid | not run | — | — |

## Interpretation

- Hybrid's ~3.6x higher consolidation strength does not reduce `memory_bytes`
  or improve `co_recall` under trace caps.
- Prefix still determines exact-token recall.
- The extra consolidation strength appears to be a latent diagnostic, not a
  lever that improves measured behavior in these probes.

## Implication for architecture

The ingestion scaffold and hybrid modulation are instrumented, validated, and
produce the expected mechanical signal. But the S vs H comparison does not
favor H on memory-per-byte or exact recall. The architecture H "win" would
require either:

1. A metric that benefits from stronger consolidated traces (e.g. domain-level
   recall, paraphrase recall), or
2. A memory model where stronger traces can replace multiple weaker traces
   (compression), which the current hippocampus/pruning does not implement.

## Open questions

1. What metric does uncapped hybrid consolidation improve?
2. Can the hippocampus consolidate high-drift chunks more aggressively under
   hybrid mode, reducing raw trace count?
3. Should the consolidation-strength cap remain 10.0 as a guardrail, or is it
   an artificial ceiling that hides useful behavior at higher values?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `1c6572a` — Add memory_bytes and max_traces.
- `e056161` — Update SUMMARY.md with run #94 result.

## Run

Run #94: benchmark `code_qa_accuracy=1.0`, fast suite `304 passed`.
