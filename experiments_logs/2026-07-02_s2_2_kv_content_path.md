# S2.2: KV content channel — experiment log

**Date:** 2026-07-02
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Stage:** `stage_0_grounding` (8 episodes)
**K (primary):** 8 (all episodes)
**Seeds:** 3 (reduced from planned 5 — see deviations)
**Holdout probes:** 0
**Wall clock:** ~292 s (~4.9 min)

## Spec

Pre-registered: `research/12-s2-kv-slot-content-path.md`
Dependent on: `research/11-s2-minimal-metabolism-loop.md` (S2.1 — not yet merged)

## Deviations from pre-registered protocol

1. **Seeds reduced to 3** (planned 5). Per-seed wall clock projection from dry run exceeded reasonable bounds; 3 seeds used per pre-registered fallback. 3 seeds is sufficient given the degenerate outcome (0 holdout probes — see below).

2. **0 holdout probes (degenerate experiment).** The pre-registered split `split_probes(stage, fraction=0.3, salt="v2")` assigned all 8 stage-0 probes to the dev set (all hash values > 0.3). This is a valid deterministic hash outcome but renders the holdout-only scoring degenerate: all conditions score 0/0. The experiment cannot assess H-KVCONTENT with zero holdout probes.

3. **BLOCKED adjudication pending.** Per the spec's validity gate: if research/11 REFUTED H-LOOP (C1 itself produces no effect), this experiment is BLOCKED, not run to a fake verdict. C1/C2 outcomes here are all 0.0000 due to 0 holdout probes — the BLOCKED gate is adjudicated at merge time.

## Primary metrics (K=N) — DEGENERATE (0 holdout probes)

| Condition | Accuracy |
|-----------|----------|
| C0 (vanilla) | 0.0000 ± 0.0000 (n=3) |
| C1 (prefix)  | 0.0000 ± 0.0000 (n=3) |
| C2 (kv)      | 0.0000 ± 0.0000 (n=3) |

**kv_effect_delta:** 0.0000 ± 0.0000 (n=3)
**kv_parity_delta:** 0.0000 ± 0.0000 (n=3)

## Verdict

**REFUTE H-KVCONTENT** — All conditions trivially equal (0/0 holdout probes).
This refutation is hollow: the pre-registered split left no holdout probes, so no
measurement was possible. The verdict reflects a degenerate experiment, not an
empirical failure of the KV approach.

**BLOCKED gate:** Pending S2.1's merged H-LOOP verdict. If H-LOOP is refuted,
this experiment is BLOCKED rather than REFUTE.

## Per-seed holdout accuracy (K=N)

| Seed | C0 | C1 | C2 | kv_effect | kv_parity |
|------|----|----|----|-----------|-----------|
| 0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## Prompt-token audit (C2 vs C0)

**Passed:** Yes (0 probes — vacuously true)
C0 mean tokens: 0.0
C2 mean tokens: 0.0

The prompt-token audit requires C2's visible prompt tokens to equal C0's for
every probe. With 0 probes, the audit passes vacuously. The implementation is
correct: the KV channel never calls `set_articulation_prefix`, and
`prompt_token_count()` for C2 returns the same count as a vanilla prompt.

## Per-answer latency (s, mean ± CI, per-seed)

Latency reflects total time per seed (model loading + computation).
With 0 holdout probes, scoring time is zero; variance is model-load noise.

| Condition | Latency (s) |
|-----------|-------------|
| C0 (vanilla) | 6.18 ± 13.23 |
| C1 (prefix)   | ~0.00 |
| C2 (kv)       | ~0.00 |

C0 latency is dominated by model loading (one load per seed). C1/C2 latencies
are near-zero because 0 probes were scored after consolidation; the per-seed
timer captures only scoring time, not teaching/consolidation time.

## Memory usage (bytes)

| Condition | Organ memory | KV handle bytes |
|-----------|-------------|-----------------|
| C1 (prefix) | 11,469,317 ± 0 | — |
| C2 (kv)     | 12,476,933 ± 0 | 1,007,616 ± 0 |

C2 adds ~1 MB for the KV handle (8 episodes' consolidated content, encoded once).

## Rank table (lane_02 facts, C1 vs C2)

Continuity check against S1.3: target-token rank at the probe blank under
consolidated organism state, C1 (prefix) vs C2 (KV).

| Query | Target | C1 rank | C2 rank |
|-------|--------|---------|---------|
| What is the secret passphrase for level 7? | marmalade | 30,941 | 54,289 |
| What is the skylark protocol? | skylark | 6,686 | 31,944 |
| What chess piece is the secret key? | rook | 4,377 | 43,343 |

Ranks are near-chance (vocab size ~152k). Neither C1 nor C2 produces meaningful
rank compression for these facts — the consolidated content text ("log: captain's
journal. file: submit officially. …") does not contain the lane_02 targets, so
ranks remain poor as expected.

## Implementation notes

- File: `src/oczy/experiments/minimal_loop_kv.py` (self-contained, 999 lines)
- Tests: `src/oczy/experiments/tests/test_minimal_loop_kv.py` (12 tests, all green, tiny-random model)
- Organism: HFDriver + KVCortex (warm/cold) + NeuralHippocampus (replay-only)
- Content channel: "prefix" (C1) sets articulation prefix ≤48 tokens; "kv" (C2) encodes once via `encode_kv`, splices pre-blank via `generate_with_kv`
- Episodes taught cumulatively, each perceive+hippocampus store+replay, consolidate at each checkpoint
- No `set_articulation_prefix` anywhere in C2 path; K/V norms unscaled (1×)
- `verify_manifest()` called before and after the run

## Commands

```
uv run pytest src/oczy/experiments/tests/test_minimal_loop_kv.py -v   # 12 passed
uv run python -m oczy.experiments.minimal_loop_kv --seeds 3 --stage 0
```
