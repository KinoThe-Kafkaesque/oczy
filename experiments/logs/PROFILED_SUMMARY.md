# Profiled Curriculum Evaluation Summary

**Run timestamp:** 2026-06-19T08:42:20Z

## Score & Resource Table

| Agent | Uptake | Transfer | Scope | TotalTime(ms) | PeakMem(B) | TopComponent |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| ZeroMemoryAgent | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | none |
| ContextOnlyAgent | 0.6667 | 0.0000 | 0.0000 | 0 | 0 | none |
| FastOnlyAgent | 0.6667 | 0.1667 | 0.1667 | 61 | 27807 | plastic_cortex |
| HippocampusOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 33 | 35278 | neural_hippocampus |
| IdentityOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 2 | 1256 | identity_hypernetwork |
| OrganismAgent | 0.6667 | 0.2500 | 0.1667 | 120 | 84927 | plastic_cortex |

## Per-Agent Resource Breakdown

### ZeroMemoryAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |

**Totals:** 0 ms wall time, 0 bytes peak memory. Largest consumer: `none`.

### ContextOnlyAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |

**Totals:** 0 ms wall time, 0 bytes peak memory. Largest consumer: `none`.

### FastOnlyAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |
| plastic_cortex           |      180 |       61.444 |          27807 |

**Totals:** 61 ms wall time, 27807 bytes peak memory. Largest consumer: `plastic_cortex`.

### HippocampusOnlyAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |
| neural_hippocampus       |      193 |       33.810 |          35278 |

**Totals:** 33 ms wall time, 35278 bytes peak memory. Largest consumer: `neural_hippocampus`.

### IdentityOnlyAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |
| identity_hypernetwork    |      180 |        2.511 |           1256 |

**Totals:** 2 ms wall time, 1256 bytes peak memory. Largest consumer: `identity_hypernetwork`.

### OrganismAgent

| Component | Calls | Time (ms) | Peak Mem (B) |
| :--- | ---: | ---: | ---: |
| experience_autoencoder   |       12 |        1.647 |          14468 |
| identity_hypernetwork    |      180 |        3.561 |           1464 |
| neural_hippocampus       |      181 |       22.430 |          34966 |
| plastic_cortex           |      192 |       83.099 |          27591 |
| skill_immune_cortex      |      180 |        4.105 |           2641 |
| world_model_critic       |      192 |        7.097 |           3797 |

**Totals:** 120 ms wall time, 84927 bytes peak memory. Largest consumer: `plastic_cortex`.

## Interpretation

- ``FastOnlyAgent`` is typically the cheapest baseline because it only updates
  a small fast-weight scratchpad in ``PlasticCortex``.
- ``OrganismAgent`` spends most wall time in slow-path components such as the
  hippocampal replay buffer, identity hypernetwork, and immune regression
  checks, and it allocates substantially more peak memory than the baselines.
- The per-component breakdown above points to the dominant subsystem for each
  agent and helps decide where optimization effort should be focused.
