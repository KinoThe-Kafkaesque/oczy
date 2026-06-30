# Knowledge Core Expansion: 280KB → 1M Parameters

**Date**: 2026-06-29
**Branch**: `autoresearch/session-20260625`

## What Changed

Expanded the organism's knowledge core from ~29K parameters (~280KB) to ~1.05M parameters (~8.4MB), a 36× growth.

### Dimension changes

| Component | Dimension | Old | New | Params |
|-----------|-----------|-----|-----|--------|
| Experience Autoencoder `_A` | (RESIDUAL_DIM, NUM_SOURCES×MAX_VOCAB) | (28, 1024) | (128, 8192) | 1,048,576 |
| Identity Hypernetwork `W` | (concepts, 4×latent_dim) | (14, 32) | (14, 128) | 1,792 |
| Plastic Cortex `hidden_dim` | RNN hidden state | 8 | 64 | ~5,000 |
| **Total unique params** | | ~29,000 | **~1,055,000** | |

### Constants changed

| Constant | Old | New | File |
|----------|-----|-----|------|
| `RESIDUAL_DIM` | 28 | 128 | `autoencoder.py` |
| `MAX_VOCAB` | 256 | 2048 | `autoencoder.py` |
| `LATENT_DIM` | 32 | 132 | `autoencoder.py` (auto-computed) |
| `DECODE_SPARSITY` | 10 | 30 | `autoencoder.py` |
| `HEBBIAN_LR` | 0.01 | 0.05 | `autoencoder.py` |
| `latent_dim` (identity) | 8 | 32 | `hypernet.py` |
| `hidden_dim` (plastic cortex) | 8 | 64 | `cortex.py` |

## Curriculum Results

### Stage-by-stage comparison

| Stage | 280KB core | 1M param core | Change |
|-------|-----------|---------------|--------|
| Stage 0 (8 episodes) | 8/8 | 8/8 | — |
| Stage 1 (8 episodes) | 8/8 | 8/8 | — |
| Stage 2 (8 episodes) | 8/8 | 8/8 | — |
| Stage 3 (4 episodes) | 4/4 | 4/4 | — |
| Stage 4 (10 episodes) | 10/10 | 10/10 | — |
| Stage 5 (6 episodes) | 6/6 | 6/6 | — |
| Stage 5 scope (pre) | 0.0 | 0.0 | — |
| Stage 5 retention (pre) | 0.0 | 0.0 | — |

### Memory footprint

| Stage | 280KB core | 1M param core |
|-------|-----------|---------------|
| After Stage 0 | 254,943 B | 8,478,377 B |
| After Stage 5 | 286,920 B | 8,517,457 B |
| Growth during curriculum | +47,726 B | +39,080 B |

## Analysis

### The 36× expansion had zero effect on curriculum performance

All stage scores, scope, and retention metrics are identical between the 280KB and 1M param cores. The 1M param core's 8.4MB sensing matrix provides no behavioral advantage over the 280KB core's 229KB matrix.

### Why: the bottleneck is architectural, not capacity

The experience autoencoder's sensing matrix `_A` is a random projection that compresses bag-of-words episode features into a residual vector. The expanded matrix gives 128 residual dimensions (vs 28) and 2048 vocab slots (vs 256), but:

1. **The downstream organs don't use the residual.** The identity hypernetwork learns through text-based concept extraction (`_extract_first_concept`), not through the autoencoder's Δz vector. The plastic cortex learns through token-label associations, not through the residual. The autoencoder's output flows nowhere behaviorally — it's a side channel for memory tracking.

2. **The scope-slot reranker uses LFM2.5 embeddings.** The reranker retrieves corrected labels via `driver.peek_embedding(request)` — the LFM2.5's own 2048-dim hidden states, not the autoencoder's 128-dim residual. Expanding the autoencoder doesn't affect the reranker.

3. **The Hebbian learning doesn't improve reconstruction.** With the larger matrix, the OMP decoder can't benefit from Hebbian updates because the 128-dim residual is underdetermined with 30-sparse recovery. The reconstruction error stays constant or increases, meaning the autoencoder's self-training loop is ineffective at this scale.

4. **The vocab cap (2048) is still not the bottleneck.** The curriculum uses ~30 unique words. Even the old 256-vocab cap was sufficient. The 2048 cap helps only if the curriculum vocabulary grows 8×, which it doesn't.

### What would actually help

The `scope=0.0` gap (no proactive cross-domain disambiguation) is caused by:

1. **The scope-slot reranker only fires after corrections.** It stores corrected labels keyed by request embedding, but can only retrieve them when the same (or similar) request is seen again. For novel cross-domain requests, there's no stored slot.

2. **The identity hypernetwork globally boosts concept tokens.** It doesn't distinguish between senses of the same word — "file" in "clerk" context vs "file" in "tool" context both map to the same concept token.

3. **The plastic cortex has only 2 labels.** It can't represent the 10+ senses needed for cross-domain disambiguation.

To improve `scope`, the architecture needs to change, not just the capacity:
- Store scope slots from earlier stages that generalize to cross-domain requests
- Add sense-specific concept tokens to the identity hypernetwork
- Expand the plastic cortex's label vocabulary
- Connect the autoencoder's residual to the identity hypernetwork's update path

## Verification

- 261 tests passed (6 warnings)
- `autoresearch.sh`: 7/7 experiments accepted
- `bounded_growth_m1_ratio`: 0.073 → 0.002 (still accepted; larger baseline makes relative growth smaller)
- Curriculum: all stages pass, Stage 5 = 6/6, scope = 0.0
