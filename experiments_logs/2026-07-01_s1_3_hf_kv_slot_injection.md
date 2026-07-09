# S1.3 — HF-substrate KV-slot fact injection

## Date: 2026-07-02 (run: 2026-07-02; spec date: 2026-07-01)

## Model: Qwen/Qwen2.5-0.5B-Instruct

## Spec

Pre-registered: `research/09-hf-kv-slot-fact-injection.md`

## Rank table (fact x condition)

| Fact | Target | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|
| alpha/skylark | skylark | 8605 | 1 | 1 | 8723 |
| beta/rook | rook | 5048 | 4 | 4 | 5174 |
| level7/marmalade | marmalade | 158 | 0 | 0 | 162 |

## Top-1 token table

| Fact | Target | C0 top1 | C1 top1 | C2 top1 | C3 top1 |
|---|---|---|---|---|---|
| alpha/skylark | skylark | ` The` | ` Sk` | ` Sk` | ` The` |
| beta/rook | rook | ` The` | ` The` | ` The` | ` The` |
| level7/marmalade | marmalade | ` The` | ` m` | ` m` | ` The` |

## Primary metric

`hf_kv_slot_rank1_count` = **1** / 3

**Verdict: REFUTE H-KV** — the hypothesis does not survive.
  Reason: C2 rank-1 count (1) < 2 (threshold).

## Sanity guard: C0 baseline

C0 ranks: [8605, 5048, 158]
C0 far-from-1 check: PASS (all ranks > 10)

## C2 injection latency (ms)

Mean: 183.89 ms
Median: 185.64 ms
Range: 159.76 – 206.26 ms

## Secondary: splice position (exploratory)

| Fact | Front (C2) | Pre-blank |
|---|---:|--:|
| alpha/skylark | 1 | 2 |
| beta/rook | 4 | 0 |
| level7/marmalade | 0 | 0 |

## Secondary: K/V norm scaling (exploratory)

| Fact | 0.5x | 1.0x | 2.0x |
|---|---:|---:|---:|
| alpha/skylark | 4 | 1 | 2106 |
| beta/rook | 11 | 4 | 99 |
| level7/marmalade | 2 | 0 | 181 |
