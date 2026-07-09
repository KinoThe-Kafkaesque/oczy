# 09 — HF-substrate KV-slot fact injection (Sprint 1 / S1.3)

**Pre-registered 2026-07-01** (human-approved sprint setup, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations.

## Problem

The 2026-06-27 finding: residual cvecs cannot force exact tokens (target rank
stuck ~47k); post-forward logit bias reaches rank 1 but is a decoding trick,
not memory; the text prefix works but burns tokens every forward pass.
llama.cpp blocked true KV-slot writes (Goal 1). The HF substrate removes the
blocker: `past_key_values` is an ordinary tensor structure.

## Hypothesis

H-KV: a fact encoded once into KV entries and spliced into the cache of a
later, unrelated probe forward pass forces exact-token recall (rank-1 at the
blank position) **without** logit bias and **without** any prompt-visible
prefix text.

## Conditions (all greedy, deterministic; the 3 facts from lane_02)

- **C0** baseline: probe alone, no injection. (Expected: rank far from 1.)
- **C1** text-prefix reference: fact text prepended to the probe prompt.
  (Upper anchor; expected rank 1 from prior work.)
- **C2** KV-slot injection: encode fact text alone → capture its
  `past_key_values` → splice those entries ahead of the probe's cache →
  decode probe. No fact tokens in the probe's visible prompt.
- **C3** cvec-only reference at the working amplitude. (Known-fail anchor.)

## Primary metric & acceptance

`hf_kv_slot_rank1_count` = number of facts (of 3) whose target token reaches
rank 1 at the blank position under C2.

- **Accept H-KV:** `hf_kv_slot_rank1_count >= 2` AND C0 rank » 1 for those
  facts (sanity that the probe alone doesn't already know).
- **Refute:** count <= 1. A refutation is a recorded result, not a failure.

## Pre-registered secondary analyses (exploratory only — cannot flip acceptance)

1. Rank as a function of splice position (front vs. immediately pre-blank).
2. Rank as a function of K/V norm scaling (0.5×, 1×, 2×).
3. Injection latency (ms) — reported, no threshold (CPU torch; the <5 ms
   GOALS.md figure was set for llama.cpp and does not gate this experiment).

## Reporting

Full rank table per fact × condition; model id; exact code path; log to
`experiments_logs/` with the spec version quoted. Vanilla/no-injection column
mandatory (C0 is that column).
