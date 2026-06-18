# Correction-to-Competence Benchmark — v1 Notes

## What works
- A hand-curated dataset of 12 text/semantic ambiguity episodes (`profile`, `model`, `batch`, `branch`, `table`, `run`, `cell`, `record`, `module`, `key`, `service`, `file`).
- Each episode has an ambiguous request, a plausible wrong answer, a user correction, the corrected answer, and one probe each for transfer, scope, and forgetting.
- The `Scorer` class computes the requested metrics:
  - `correction_uptake_latency` — 0 if the agent fixed the request immediately, 1 if not.
  - `transfer_score` — accuracy on probes where the lesson should transfer.
  - `scope_score` — accuracy on probes where the lesson should *not* apply.
  - `forgetting_score` — accuracy on unrelated probes.
  - `memory_bytes_per_delta` — persistent memory bytes divided by distinct successful lessons.
- `run_benchmark(agent)` walks every episode through one answer/correct/answer cycle and returns a score card.
- Two baseline agents: `AlwaysWrongAgent` and `OracleAgent`.
- Tests cover the dataset structure, the scorer, and both baselines.

## Current limitations
1. **String equality scoring.** Answers are compared with simple normalization (lowercase, collapse whitespace). There is no semantic similarity or LLM-as-judge fallback, so agents that rephrase the expected answer slightly still fail.
2. **One-shot corrections only.** The benchmark delivers exactly one correction per episode. It does not measure how many turns of correction are required, only whether zero or more turns were needed.
3. **Single-session evaluation.** All episodes run against the same agent instance. Interference and forgetting are measured with deliberately unrelated probes, not with a pre-test/post-test split on a learned skill.
4. **Tiny, frozen dataset.** The episodes are hard-coded. There is no curriculum builder, ambiguity generator, or adversarial probing.
5. **Memory accounting is honor-system.** `memory_bytes_per_delta` relies on the agent exposing `persistent_memory()` or `memory_size()`. It does not inspect model weights, context caches, or external vector stores.
6. **No consolidation metric.** The current protocol does not test whether raw correction traces can be deleted after consolidation while preserving behavior.
7. **No identity drift metric.** There is no long-horizon stress test that accumulates many corrections and checks whether the agent's general style or capabilities degrade.

## What v2 should add
- **Semantically aware judging**, e.g. an optional LLM critic or sentence-embedding cosine similarity so minor rephrasings are not punished.
- **Multi-turn latency** allowing repeated correction attempts and recording how many turns were actually needed.
- **Replay / pre-test baseline** so forgetting can be measured against a known pre-correction competence, not just unrelated trivia probes.
- **Programmatic episode generation** from a lexical ambiguity taxonomy so the dataset can scale.
- **Compression/consolidation test** that snapshots behavior before and after deleting raw correction traces.
- **Identity drift suite** that runs a long correction session and then re-evaluates an independent held-out skillset.
- **Optional model-based agent harnesses** that wrap an API model with a tiny memory layer (while still keeping the benchmark itself dependency-light).
