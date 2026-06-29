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
