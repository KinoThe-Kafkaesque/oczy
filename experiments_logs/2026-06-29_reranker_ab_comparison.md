> **INVALIDATION NOTICE — 2026-07-01**  
> **Classification: PARTIAL**  
> **Reason:** The original A/B comparison body (lines 1–117) was run on code with three compounding bugs in the scope-slot reranker (all slots collapsed into one, retrieval threshold too high, topk degenerate). The conclusions drawn ("topk=3 is neutral," "scope=0.0") are INVALIDATED artifacts of the broken reranker. The "Update: bug fix changes conclusions (2026-06-30)" section (line ~118) correctly identifies and corrects these errors and is VALID.  
> **See:** `2026-06-30_scope_slot_reranker_fix.md` for the full bug diagnosis; `2026-07-01_honest_post_leakage_baseline.md` for the current reference point.

# Scope-Slot Reranker A/B Comparison

**Date**: 2026-06-29
**Branch**: `autoresearch/session-20260625`
**Driver**: LFM2.5-1.2B-Instruct Q4 GGUF (real)
**Scoring**: semantic neighbour fallback

## Configurations Tested

| Config | `scope_rerank_weight` | `scope_rerank_topk` | `scope_rerank_sense_split` | `scope_rerank_multi_label` |
|--------|----------------------|---------------------|---------------------------|---------------------------|
| baseline | 2.0 (default) | 1 (default) | False (default) | False (default) |
| topk3 | 2.0 | 3 | False | False |
| sense_split | 2.0 | 1 | True | False |
| multi_label | 2.0 | 1 | False | True |
| combined | 2.0 | 3 | True | True |

All configs share the similarity-weighted boost (`weight * scope_sim * overlap`),
which was introduced as the new default in this session. The previous flat boost
(`weight * overlap`) is no longer tested.

## Results

### Stage-by-stage fixed counts

| Config | Stage 0 (8) | Stage 1 (8) | Stage 2 (8) | Stage 3 (4) | Stage 4 (10) | Stage 5 (6) |
|--------|-------------|-------------|-------------|-------------|--------------|-------------|
| **baseline** | **8/8** | **8/8** | **8/8** | **4/4** | **10/10** | **6/6** |
| topk3 | 8/8 | 8/8 | 8/8 | 4/4 | 10/10 | 6/6 |
| sense_split | 8/8 | 8/8 | 8/8 | 4/4 | 10/10 | 5/6 |
| multi_label | 1/8 | 1/8 | 0/8 | 1/4 | 1/10 | 1/6 |
| combined | 3/8 | 3/8 | 0/8 | 3/4 | 3/10 | 1/6 |

### Stage 5 (Cross-domain disambiguation) detail

| Config | Fixed | Scope (pre) | Retention (pre) | Memory (bytes) |
|--------|-------|-------------|-----------------|----------------|
| **baseline** | **6/6** | 0.0 | 0.0 | 286,920 |
| topk3 | 6/6 | 0.0 | 0.0 | 286,921 |
| sense_split | 5/6 | 0.0 | 0.0 | 286,921 |
| multi_label | 1/6 | 0.0 | 0.167 | 286,922 |
| combined | 1/6 | 0.0 | 0.333 | 286,921 |

## Analysis

### Similarity-weighted boost is the critical improvement

The baseline config achieves **6/6 Stage 5** — up from **1/6** in the previous
session (commit `0aba650`). The only change is the `scope_sim` multiplier in the
boost formula: `weight * scope_sim * overlap` instead of `weight * overlap`.

**Mechanism**: When a new request is dissimilar from a stored slot key (low
cosine similarity), the boost is weakened proportionally. This prevents the
reranker from boosting a wrong-sense label when the request's context doesn't
match the slot's context. The previous flat boost applied the same boost
regardless of request-slot similarity, causing cross-contamination between
different senses of the same ambiguous word.

### topk=3 is neutral

Topk=3 produces identical results to the baseline. When each request maps to
exactly one slot (which is the case in this curriculum), only one slot passes
the `_ALLOC_THRESHOLD`, so topk=3 degenerates to topk=1.

### sense_split slightly hurts (5/6)

Excluding the ambiguous word from the overlap computation removes useful
disambiguating signal in one Stage 5 episode. The ambiguous word itself can be
part of the correct label (e.g., "file" in "disk file" is both the ambiguous
word and part of the correct sense). Removing it makes the overlap computation
noisier.

### multi_label is catastrophic (1/6 across all stages)

Storing multiple labels per slot (joined by `" | "`) causes the reranker to
match against all stored label parts, creating a "super-slot" that matches too
broadly. The agent gets stuck on one answer ("the captain's journal" for
everything in Stage 5) because the multi-label slot boosts the most-recently
stored label regardless of context.

This degradation is not limited to Stage 5 — it affects ALL stages:
- Stage 0: 1/8 (vs 8/8 baseline)
- Stage 2: 0/8 (vs 8/8 baseline)
- Stage 4: 1/10 (vs 10/10 baseline)

The multi-label slot store fundamentally breaks the context-addressed retrieval
that makes the reranker work.

### combined inherits multi_label's toxicity

The combined config (topk3 + sense_split + multi_label) performs identically to
multi_label on Stage 5 (1/6). The multi_label damage dominates any benefit from
the other strategies. The slightly better Stage 0 (3/8 vs 1/8) is likely noise
from the interaction of topk3's broader retrieval with multi_label's diluted
labels.

## Conclusion

The **similarity-weighted boost** (already the default) is the optimal
configuration. It improves Stage 5 from 1/6 to 6/6 by preventing cross-sense
contamination. The other strategies either don't help (topk3), slightly hurt
(sense_split), or are catastrophic (multi_label).

**Recommended config**: defaults (`scope_rerank_weight=2.0`,
`scope_rerank_topk=1`, `scope_rerank_sense_split=False`,
`scope_rerank_multi_label=False`).

### Remaining gap: scope=0.0

All configs show `scope=0.0` on Stage 5 pre-accuracy, meaning no config achieves
proactive cross-domain disambiguation (correctly answering before any
correction). The 6/6 fixed count means the agent learns from corrections within
Stage 5, but doesn't retain cross-domain knowledge from earlier stages. This is
the next improvement target: enabling the scope-slot reranker to proactively
select the correct sense based on context alone, without needing a correction
first.

## Update: bug fix changes conclusions (2026-06-30)

The A/B test above was run on code with **three compounding bugs** in the
scope-slot reranker. The conclusions drawn from it are now known to be wrong
in several respects. The historical results are preserved above as a session
log; this section documents the corrected picture.

### The three bugs

1. **`_scope_key` used `last_token_only=True`** (commit `43cfc9f`):
   `peek_embedding()` with the default `last_token_only=True` embeds only the
   last token. Every curriculum request ends with `.`, so every request got an
   identical embedding (cosine sim = 1.0). All 44+ episodes collapsed into a
   single slot. Fix: `last_token_only=False` for mean-pooled whole-request
   embeddings.

2. **`_MAX_SLOTS=16` too small** (commit `091046c`): The slot store filled
   after Stage 1 (8 + 8 = 16), so Stage 2 corrections overwrote the Stage 0/1
   labels. Fix: `_MAX_SLOTS=64`.

3. **`_ALLOC_THRESHOLD=0.85` used for label retrieval** (commit `091046c`):
   Mean-pooled embeddings of related-but-different requests have cosine sim
   ~0.3–0.65, well below 0.85. No labels were ever returned, so the reranker
   never fired. Fix: a separate `_RETRIEVE_THRESHOLD=0.3` for label retrieval.

A fourth change (commit `e316cb1`) set `scope_rerank_topk=3` as the new
default, and commit `9e8eef4` updated the test defaults to match.

### Why "topk=3 is neutral" was wrong

The original analysis observed that topk=3 produced identical results to the
baseline (topk=1) and concluded topk=3 was neutral. This was an **artifact of
the bugs**: with all slots collapsed into one (bug #1), topk=3 degenerated to
topk=1 because there was only ever one slot to retrieve from. The original
report even noted this degeneration ("only one slot passes the
`_ALLOC_THRESHOLD`") but misattributed it to the curriculum having one slot
per request rather than to the embedding collapse.

With the slots correctly separated (`last_token_only=False`,
`_MAX_SLOTS=64`), topk=3 returns the correct technical-sense label that
topk=1 misses. topk=1 only returns the single most-similar label, which is
often the wrong sense. topk=3 gives the correct sense a chance to be
retrieved and boosted. This is why `scope_rerank_topk=3` is now the default.

### Why "scope=0.0" was wrong

The `scope=0.0` result across all configs was also a **bug artifact**: the
reranker never fired because no labels passed the 0.85 retrieval threshold
(bug #3). The "Remaining gap: scope=0.0" section above is therefore
**partially resolved** — the gap was not fundamental, it was a retrieval
threshold bug. With `_RETRIEVE_THRESHOLD=0.3`, the reranker fires and
proactive cross-domain disambiguation works.

### New recommended config

```text
scope_rerank_weight=2.0
scope_rerank_topk=3
scope_rerank_sense_split=False
scope_rerank_multi_label=False
```

The only change from the old recommendation is `scope_rerank_topk`: `1 → 3`.
`scope_rerank_weight=2.0`, `sense_split=False`, and `multi_label=False` are
unchanged.

### New curriculum results (with the fix)

Real LFM2.5-1.2B Q4 driver, semantic scoring, topk=3, sense_split=False:

| Stage | Fixed | Scope | Retention / Transfer |
|-------|-------|-------|----------------------|
| Stage 0 (8) | 8/8 | — | retention=0.88 |
| Stage 1 (8) | 7/8 | — | transfer=0.75 |
| Stage 2 (8) | 8/8 | scope=1.00 | retention=0.88 |
| Stage 3 (4) | 4/4 | scope=1.00 | transfer=0.25 |
| Stage 4 (10) | 10/10 | — | retention=1.00 |
| Stage 5 (6) | 6/6 | scope=0.50 | retention=1.00 |

Stage 5 scope improved **0.00 → 0.50** (3/6 proactive correct), Stage 2
scope improved **0.12 → 1.00**, and Stage 4 retention improved
**0.10 → 1.00**.

### What still holds from the original report

The **multi_label catastrophe** conclusion still holds and was **not** caused
by the three bugs. multi_label stores multiple labels per slot (joined by
`" | "`), creating a "super-slot" that matches too broadly and boosts the
most-recently stored label regardless of context. This breaks the
context-addressed retrieval that makes the reranker work, and it degrades
every stage, not just Stage 5. `scope_rerank_multi_label=False` remains
correct.

The **sense_split slightly hurts** observation is also unaffected by the bug
fixes in its reasoning (excluding the ambiguous word from overlap removes
useful disambiguating signal), though its magnitude should be re-measured on
the fixed code.

### Reference

The detailed fix report, including the per-commit diagnosis and the full
post-fix curriculum numbers, is in
`2026-06-30_scope_slot_reranker_fix.md`.
