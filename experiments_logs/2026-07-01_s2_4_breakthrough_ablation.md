# S2.4 Breakthrough Ablation — Magnitude-Controlled Drift

**Date:** 2026-07-01
**Driver:** mock
**Seeds:** 1

## Condition × K Trajectory

| Cond | K | Δ Target | Δ Control | Δ Target (clamped) |
|------|---|------------|------------|----------------------|
| OLD | 0 | 0.000000 | 0.000000 | 0.000000 |
| OLD | 5 | 0.000000 | 0.000000 | 0.000000 |
| OLD | 10 | 0.000000 | 0.000000 | 0.000000 |
| OLD | 15 | 0.000000 | 0.000000 | 0.000000 |
| OLD | 20 | 0.000000 | 0.000000 | 0.000000 |
| | | | | |
| NEW (breakthrough) | 0 | 0.000000 | 0.000000 | 0.000000 |
| NEW (breakthrough) | 5 | 0.000000 | 0.000000 | 0.000000 |
| NEW (breakthrough) | 10 | 0.000000 | 0.000000 | 0.000000 |
| NEW (breakthrough) | 15 | 0.000000 | 0.000000 | 0.000000 |
| NEW (breakthrough) | 20 | 0.000000 | 0.000000 | 0.000000 |
| | | | | |
| OLD + alpha only | 0 | 0.000000 | 0.000000 | 0.000000 |
| OLD + alpha only | 5 | 0.000000 | 0.000000 | 0.000000 |
| OLD + alpha only | 10 | 0.000000 | 0.000000 | 0.000000 |
| OLD + alpha only | 15 | 0.000000 | 0.000000 | 0.000000 |
| OLD + alpha only | 20 | 0.000000 | 0.000000 | 0.000000 |
| | | | | |
| OLD + threshold only | 0 | 0.000000 | 0.000000 | 0.000000 |
| OLD + threshold only | 5 | 0.000000 | 0.000000 | 0.000000 |
| OLD + threshold only | 10 | 0.000000 | 0.000000 | 0.000000 |
| OLD + threshold only | 15 | 0.000000 | 0.000000 | 0.000000 |
| OLD + threshold only | 20 | 0.000000 | 0.000000 | 0.000000 |
| | | | | |
| OLD + diverse only | 0 | 0.000000 | 0.000000 | 0.000000 |
| OLD + diverse only | 5 | 0.000000 | 0.000000 | 0.000000 |
| OLD + diverse only | 10 | 0.000000 | 0.000000 | 0.000000 |
| OLD + diverse only | 15 | 0.000000 | 0.000000 | 0.000000 |
| OLD + diverse only | 20 | 0.000000 | 0.000000 | 0.000000 |

## Per-Condition Summary

| Cond | Max K Δ Target | Max K Δ Control | Max K Δ Clamped | Wall Time (s) |
|------|------------------|-------------------|-------------------|---------------|
| OLD | 0.000000 | 0.000000 | 0.000000 | 0.15 |
| NEW (breakthrough) | 0.000000 | 0.000000 | 0.000000 | 0.16 |
| OLD + alpha only | 0.000000 | 0.000000 | 0.000000 | 0.15 |
| OLD + threshold only | 0.000000 | 0.000000 | 0.000000 | 0.15 |
| OLD + diverse only | 0.000000 | 0.000000 | 0.000000 | 0.16 |

## Norm Control Survival

| Cond | Δ Target | Δ Clamped | Survival Ratio |
|------|-----------|------------|---------------|
| OLD | 0.000000 | 0.000000 | NaN |
| NEW (breakthrough) | 0.000000 | 0.000000 | NaN |
| OLD + alpha only | 0.000000 | 0.000000 | NaN |
| OLD + threshold only | 0.000000 | 0.000000 | NaN |
| OLD + diverse only | 0.000000 | 0.000000 | NaN |

## VERDICT

1. **No single variable improves upon OLD.**
   - alpha: Δ_target = +0.000000 vs OLD
   - threshold: Δ_target = +0.000000 vs OLD
   - diversity: Δ_target = +0.000000 vs OLD
2. NEW survival ratio is NaN (delta_target = 0).
3. Cannot evaluate norm-control retraction (NaN).

## Raw Data (CSV for reproducibility)
```csv
cond,seed,K,delta_target,delta_control,delta_target_clamped
1,0,0,0.0,0.0,0.0
1,0,5,0.0,0.0,0.0
1,0,10,0.0,0.0,0.0
1,0,15,0.0,0.0,0.0
1,0,20,0.0,0.0,0.0
2,0,0,0.0,0.0,0.0
2,0,5,0.0,0.0,0.0
2,0,10,0.0,0.0,0.0
2,0,15,0.0,0.0,0.0
2,0,20,0.0,0.0,0.0
3,0,0,0.0,0.0,0.0
3,0,5,0.0,0.0,0.0
3,0,10,0.0,0.0,0.0
3,0,15,0.0,0.0,0.0
3,0,20,0.0,0.0,0.0
4,0,0,0.0,0.0,0.0
4,0,5,0.0,0.0,0.0
4,0,10,0.0,0.0,0.0
4,0,15,0.0,0.0,0.0
4,0,20,0.0,0.0,0.0
5,0,0,0.0,0.0,0.0
5,0,5,0.0,0.0,0.0
5,0,10,0.0,0.0,0.0
5,0,15,0.0,0.0,0.0
5,0,20,0.0,0.0,0.0
```
