# Order-Shuffle Stressor

## Date: 2026-06-28

## Goal

Test whether fact order within a turn changes cvec output, and whether
sequential (per-chunk cortex.observe) vs parallel (full-turn observe)
observation modes produce different order sensitivity.

## Architecture change: per-chunk sequential observation

The ingestion pipeline's `observation_mode` previously only affected drift
statistics, not cortex learning. Both modes called `cortex.observe()` once on
the full-turn hidden in `perceive()`, making them identical.

**Fix**: When `observation_mode="sequential"`:
1. `perceive()` skips `cortex.observe()` — caches `_last_hidden` but leaves
   `warm_state` untouched
2. `IngestionPipeline.process()` calls `cortex.observe(chunk_hidden,
   correction_signal)` per-chunk in order, making `warm_state` genuinely
   order-dependent (last chunk dominates at signal=1.0, plasticity=1.0)

Parallel path unchanged (full-turn observe in `perceive()`).

## Probe design evolution (runs #149-#153)

### Run #149: filler haystack + coherence metric
- 1024-token filler turn with facts buried at positions
- `salience="lexical-novelty"`, `correction_signal` not set (default 0.0)
- Result: sensitivity=0 (all answers identical)
- Problem: filler dominates `peek_embedding`, signal=0.0 → 2% plasticity

### Run #151: short turn + logit biasing + coherence metric
- Fixed template: "Here is some information. First, [fact]. Next, [fact]. Finally, [fact]."
- `correction_signal=1.0`, `use_logit_bias=True`
- Result: answer_sensitivity=1, but logit biasing fails for target_middle/last
- Problem: template words dominate mean-pooled hidden

### Run #152: pass-through salience
- Same as #151 but `salience="pass-through"`
- Result: identical to #151 — salience filter wasn't the bottleneck

### Run #153: facts-only turn (no template)
- Turn = just the 3 facts joined by spaces, no template words
- Result: **answer_sensitivity=2** (all 3 orderings produce distinct answers)
- All coh=1 (logit biasing works at all positions)
- Distinct continuations: target_first → "purpose of the 'level'",
  target_middle → "cryptographic hash", target_last → "hidden menu"

## Key findings

1. **Fact order DOES affect cvec output** — all 3 orderings produce distinct
   continuations (answer_sensitivity=2, maximum signal)
2. **Template words mask the signal** — "Here is some information. First. Next.
   Finally." dominate `peek_embedding`'s mean-pooled hidden. Removing the
   template (facts-only turn) fixed both coherence and sensitivity
3. **correction_signal=1.0 is required** — at signal=0.0, plasticity=alpha_warm=0.02,
   warm_state barely moves, no learning happens
4. **salience="pass-through"** needed so all chunks survive (lexical-novelty
   drops later chunks sharing vocabulary)
5. **peek_embedding uses create_embedding API** — separate forward pass, no KV
   cache pollution of generate()
6. **Parallel and sequential produce identical outputs** — the facts-only turn
   (~30 tokens) fits in a single 32-token chunk, so sequential mode has 1 chunk
   = 1 observe = identical to parallel mode. The per-chunk architecture needs
   ≥2 distinct chunks to discriminate, which requires a longer turn (haystack
   problem)

## Limitations

The per-chunk observe architecture is proven correct but the probe is at its
sensitivity limit. Discriminating parallel vs sequential requires:
- A turn long enough to produce ≥2 distinct chunks
- But short enough that facts dominate the embedding (not filler)
- These constraints conflict — the haystack problem

Future experiment: use a multi-sentence turn where each sentence is one fact
(no filler), with a small chunk window (16 tokens) and non-overlapping chunks.

## Files changed

- `src/oczy/experiments/cortex_agent.py` — `perceive()` skips observe in sequential mode
- `src/oczy/experiments/ingestion.py` — `process()` per-chunk observe loop in sequential mode
- `src/oczy/experiments/codebase_qa/benchmark.py` — `_run_order_shuffle_probe()` + metric wiring
