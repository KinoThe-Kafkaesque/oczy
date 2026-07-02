# Organ Ablation Matrix — 2026-07-02_s3_m1_subtractive_ablation

**Path:** real-driver (LlamaCVecDriver GGUF).

**Generated:** 2026-07-02 22:22:17 UTC
**Seeds:** 3  |  **Split:** dev  |  **Backend:** real GGUF + CortexAgent

## Accuracy Matrix

| Config | Stage 0: Sense grounding | Stage 1: Transfer within domain | Stage 2: Scope control | Stage 3: Dialog | Stage 4: Consolidation stress | Stage 5: Cross-domain disambiguation |
|---|---|---|---|---|---|---|
| FULL | 0.8000 ± 0.0000 | 0.8095 ± 0.2049 | 0.7179 ± 0.1103 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8519 ± 0.1594 |
| MINIMAL | 0.4000 ± 0.0000 | 0.9048 ± 0.2049 | 0.6410 ± 0.1103 | 0.2000 ± 0.0000 | 0.5556 ± 0.2391 | 0.5926 ± 0.4217 |
| FULL-hippocampus | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.6667 ± 0.1103 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8148 ± 0.1594 |
| FULL-critic | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.6410 ± 0.1103 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8889 ± 0.0000 |
| FULL-identity | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.7436 ± 0.2207 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8519 ± 0.1594 |
| FULL-immune | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.6667 ± 0.1103 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8519 ± 0.1594 |
| FULL-autoencoder | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.7179 ± 0.1103 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.8889 ± 0.0000 |
| FULL-dsi | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.8462 ± 0.0000 | 0.4000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6667 ± 0.0000 |
| FULL-scope_slot_reranker | 0.8000 ± 0.0000 | 0.8095 ± 0.2049 | 0.5128 ± 0.2207 | 0.2000 ± 0.0000 | 0.8333 ± 0.0000 | 0.7778 ± 0.0000 |

## Per-Organ Δ (FULL minus FULL-organ)

Positive Δ → organ contributes; negative → ablation improves accuracy.

| Organ | Stage 0: Sense grounding | Stage 1: Transfer within domain | Stage 2: Scope control | Stage 3: Dialog | Stage 4: Consolidation stress | Stage 5: Cross-domain disambiguation | All-stage mean ± CI |
|---|---|---|---|---|---|---|---|
| **NeuralHippocampus** | +0.0000 | -0.0476 | +0.0513 | +0.0000 | +0.0000 | +0.0370 | +0.0068 ± 0.0363 |
| **WorldModelCritic** | +0.0000 | -0.0476 | +0.0769 | +0.0000 | +0.0000 | -0.0370 | -0.0013 ± 0.0459 |
| **IdentityHypernetwork** | +0.0000 | -0.0476 | -0.0256 | +0.0000 | +0.0000 | +0.0000 | -0.0122 ± 0.0212 |
| **SkillImmuneCortex** | +0.0000 | -0.0476 | +0.0513 | +0.0000 | +0.0000 | +0.0000 | +0.0006 ± 0.0328 |
| **ExperienceAutoencoder** | +0.0000 | -0.0476 | +0.0000 | +0.0000 | +0.0000 | -0.0370 | -0.0141 ± 0.0232 |
| **DifferentiableFactIndex** | +0.0000 | -0.0476 | -0.1282 | -0.2000 | -0.1667 | +0.1852 | -0.0596 ± 0.1481 |
| **ScopeSlotReranker** | +0.0000 | +0.0000 | +0.2051 | +0.0000 | +0.0000 | +0.0741 | +0.0465 ± 0.0873 |

## Per-Organ Verdicts

Each verdict compares FULL-*organ* against FULL (real-driver (LlamaCVecDriver GGUF)).

- **NeuralHippocampus** (`hippocampus`): **contributing** (organ saves +0.0068 on average)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: +0.0513, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: +0.0370)
- **WorldModelCritic** (`critic`): **dead weight** (delta indistinguishable from noise)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: +0.0769, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: -0.0370)
- **IdentityHypernetwork** (`identity`): **harmful** (ablation improves accuracy by +0.0122 on average)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: -0.0256, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: +0.0000)
- **SkillImmuneCortex** (`immune`): **dead weight** (delta indistinguishable from noise)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: +0.0513, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: +0.0000)
- **ExperienceAutoencoder** (`autoencoder`): **harmful** (ablation improves accuracy by +0.0141 on average)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: +0.0000, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: -0.0370)
- **DifferentiableFactIndex** (`dsi`): **harmful** (ablation improves accuracy by +0.0596 on average)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: -0.0476, Stage 2: Scope control: -0.1282, Stage 3: Dialog: -0.2000, Stage 4: Consolidation stress: -0.1667, Stage 5: Cross-domain disambiguation: +0.1852)
- **ScopeSlotReranker** (`scope_slot_reranker`): **contributing** (organ saves +0.0465 on average)  (per-stage Δ: Stage 0: Sense grounding: +0.0000, Stage 1: Transfer within domain: +0.0000, Stage 2: Scope control: +0.2051, Stage 3: Dialog: +0.0000, Stage 4: Consolidation stress: +0.0000, Stage 5: Cross-domain disambiguation: +0.0741)

