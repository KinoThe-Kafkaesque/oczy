# Cortex Dimension Benchmark: Low-Dim vs High-Dim

**Date**: 2026-06-30
**Driver**: LFM2.5-1.2B-Instruct Q4 GGUF (real)
**Scoring**: semantic
**Script**: `src/oczy/experiments/organism_curriculum/bench_cortex_dim.py`

## Protocol

Swept `d_cortex` ∈ {2, 4, 8, 16, 32, 64, 128} on the full 6-stage organism
curriculum. Each config creates a fresh `OrganismAgent` with a `CortexAgent`
using `KVCortexConfig(d_cortex=N)`. The driver is loaded once and shared
across configs (embedding cache persists).

Two fronts measured:
- **Performance**: post-test accuracy per stage (retention, scope, transfer)
- **Improvement speed**: uptake_latency (fraction of episodes NOT fixed on
  first try — lower = faster) and pre→post delta

## Results

### Performance (post-test accuracy)

| d_cortex | S0 post | S1 post | S2 post | S3 post | S4 post | S5 post | **avg post** |
|----------|---------|---------|---------|---------|---------|---------|-------------|
| 2        | 0.88    | 0.88    | 1.00    | 0.62    | 1.00    | 0.67    | 0.84        |
| 4        | 0.88    | 0.75    | 1.00    | 0.62    | 0.90    | 0.75    | 0.82        |
| 8        | 0.75    | 0.75    | 1.00    | 0.62    | 1.00    | 0.75    | 0.81        |
| **16**   | 0.88    | **1.00**| 1.00    | 0.62    | 1.00    | 0.75    | **0.88**    |
| 32       | 0.88    | 0.75    | 1.00    | 0.62    | 1.00    | 0.67    | 0.82        |
| 64       | 0.88    | 0.88    | 0.94    | 0.62    | 1.00    | 0.75    | 0.84        |
| 128      | 0.88    | 0.75    | 1.00    | 0.62    | 1.00    | **0.83**| 0.85        |

### Improvement speed (uptake + pre→post delta)

| d_cortex | S0 upt | S1 upt | S2 upt | S3 upt | S4 upt | S5 upt | avg upt | avg Δ  |
|----------|--------|--------|--------|--------|--------|--------|---------|--------|
| 2        | 0.00   | 0.12   | 0.00   | 0.00   | 0.00   | 0.00   | 0.02    | +0.38  |
| 4        | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00    | +0.42  |
| 8        | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00    | +0.43  |
| 16       | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00    | +0.43  |
| 32       | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00    | +0.39  |
| 64       | 0.00   | 0.12   | 0.00   | 0.00   | 0.00   | 0.00   | 0.02    | +0.40  |
| 128      | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00    | +0.44  |

### Scope (post-test scope accuracy)

| d_cortex | S0 scope | S1 scope | S2 scope | S3 scope | S4 scope | S5 scope | avg scope |
|----------|----------|----------|----------|----------|----------|----------|-----------|
| 2        | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| 4        | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| 8        | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| 16       | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| 32       | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| 64       | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | 0.50     | 0.42      |
| **128**  | 0.00     | 0.00     | 1.00     | 1.00     | 0.00     | **0.67** | **0.44**  |

### Memory footprint

All configs produce identical memory growth: **+2,206,417 bytes** total across
the curriculum. The cortex's warm_state (d_cortex × 4 bytes) is negligible
(8 B for d_cortex=2, 512 B for d_cortex=128) compared to the 1M-param
autoencoder/identity core (~2.16 MB).

## Analysis

### 1. d_cortex has minimal effect on curriculum performance

Avg post-test accuracy ranges from **0.81** (d_cortex=8) to **0.88**
(d_cortex=16) — a 7-point spread. Most of the variation is in Stage 1
(transfer: 0.75–1.00) and Stage 5 (cross-domain: 0.67–0.83), which are
the noisiest stages with the fewest episodes.

### 2. d_cortex=16 is the sweet spot

d_cortex=16 achieves the best avg post (0.88) and the only perfect Stage 1
transfer (1.00). This is consistent with the memory finding that the
optimal CortexAgent config uses `d_cortex=2-3` with SVD-init — but here,
without SVD-init, a slightly higher dimension helps the policy head
discriminate candidates.

### 3. d_cortex=128 has the best Stage 5 cross-domain

d_cortex=128 is the only config achieving Stage 5 scope=0.67 (4/6 instead
of 3/6) and Stage 5 post=0.83. The higher-dimensional warm_state gives the
policy head more capacity to discriminate cross-domain senses.

### 4. Improvement speed is dimension-invariant

Uptake is ~0.00 for all dims (nearly all episodes fixed on first correction).
The pre→post delta ranges +0.38 to +0.44 — d_cortex=128 has the highest
delta (+0.44) but the difference is within noise.

### 5. The cortex is NOT the bottleneck

The scope-slot reranker (which uses LFM2.5 embeddings, not the cortex's
warm_state) and the label-based answer path dominate curriculum performance.
The cortex's warm_state only affects:
- The policy head's candidate scores (via `policy_scores` in `_rank_answer`)
- The cvec steering during LM generation (not exercised in label-based path)

The architectural bottleneck for Stage 5 scope (0.50–0.67) is the
concept vocabulary mismatch in the identity hypernetwork, not the cortex
dimension.

## Conclusion

**The architectural problem is partially resolved.** Stage 5 scope improved
from 0.00 to 0.50–0.67 across all dims. The remaining gap is not a capacity
problem — d_cortex=2 and d_cortex=128 produce nearly identical scope scores
(0.50 vs 0.67). The bottleneck is the concept-scoring path's inability to
distinguish word senses (e.g. "file" as nautical vs office).

**d_cortex=16 is recommended** for the label-based curriculum: best avg post
(0.88), perfect Stage 1 transfer, and minimal memory overhead. d_cortex=128
is better for cross-domain disambiguation (Stage 5 scope=0.67) but worse on
Stage 1 transfer (0.75 vs 1.00).

**The cortex dimension is not the lever for improving Stage 5 scope.** The
next improvement should target the concept vocabulary (sense-specific
concepts) or semantic concept matching, not the cortex dimension.
