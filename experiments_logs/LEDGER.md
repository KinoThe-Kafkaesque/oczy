# Experiments Logs Ledger — Authoritative Index

**Date:** 2026-07-01  
**Purpose:** This ledger classifies every experiment log from 2026-06-26 through 2026-07-01 against two invalidation events:

1. **Scope-slot reranker bugs** (fixed 2026-06-30): Three compounding bugs silently prevented the reranker from functioning correctly between its introduction on 2026-06-29 and the fix on 2026-06-30. See `2026-06-30_scope_slot_reranker_fix.md`.
2. **Test-set leakage removal** (2026-07-01): Two leakage paths (`_SCOPE_TEACHING` per-episode-ID entries, `prefix_targets=[probe.expected]` for scope probes) were removed, superseding all prior Stage-2/Stage-5 scope claims. See `2026-07-01_honest_post_leakage_baseline.md`.

**Current reference point:** `2026-07-01_honest_post_leakage_baseline.md` — these are the numbers all future work must beat.

## Classification Key

| Class | Meaning |
|-------|---------|
| **VALID** | Unaffected by either event — e.g., pure prefix/needle/ingestion stressors that never touched the reranker or leakage paths |
| **INVALIDATED** | Depends on the broken reranker window (2026-06-29 to 2026-06-30 pre-fix) |
| **SUPERSEDED** | Leakage-era Stage-2/5 claims (pre-2026-07-01, superseded by honest baseline) |
| **PARTIAL** | Mixed — some sections valid, some invalidated or superseded; see individual file banner |

## Ledger

| Date | File | Classification | Reason | Superseded By |
|------|------|----------------|--------|---------------|
| 2026-06-26 | `2026-06-26_embedder_fork_mock_foreign.md` | VALID | Ingestion embedder architecture; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_hybrid_consolidation_architecture.md` | VALID | Architecture S vs H consolidation; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_ingestion_pipeline_scaffold.md` | VALID | Ingestion pipeline scaffold; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor.md` | VALID | Multi-fact stressor (mock driver); no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor_prefix.md` | VALID | ReservedPosition prefix stressor; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor_real_driver.md` | VALID | Real-driver multi-fact stressor; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_needle_per_turn_stressor.md` | VALID | Needle-per-turn stressor tests; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_needle_sweep_script.md` | VALID | Needle sweep benchmark script; no reranker or leakage | — |
| 2026-06-26 | `2026-06-26_policy_head_ranking_loop.md` | VALID | Policy head ranking loop (pre-reranker era); curriculum numbers from policy head path, not reranker | — |
| 2026-06-26 | `2026-06-26_salience_threshold_ablation.md` | VALID | Salience filter ablation for ingestion; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_auto_consolidate_sh.md` | VALID | Auto-consolidate S vs H probe; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_contrastive_cvec_discovery.md` | VALID | Contrastive cvec discovery and logit biasing; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_cortex_hippocampus_prefix.md` | VALID | Hippocampus-derived prefix in CortexAgent; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_cvec_prefix_composition_tradeoffs.md` | VALID | Cvec+prefix composition tradeoffs; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_deprecate_auto_prefix.md` | VALID | Deprecation of stressor auto-prefix; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_domain_recall_metric.md` | VALID | Domain-level recall metric; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_foreign_minilm_embedder.md` | VALID | Foreign MiniLM embedder integration; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_hippocampus_auto_prefix.md` | VALID | Hippocampus-derived auto-prefix (stressor wrapper); no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_hybrid_cap_sh.md` | VALID | Configurable hybrid consolidation cap; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_knowledge_store_prefix_targets.md` | VALID | KnowledgeStore prefix_targets integration; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_length_4096_needle_sweep.md` | VALID | Real-driver needle sweep at length 4096; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_memory_per_byte_sh.md` | VALID | Memory-per-byte S vs H probe; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_multi_fact_embedder_comparison.md` | VALID | Same-LM vs foreign-MiniLM embedder comparison; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_paraphrase_mode.md` | VALID | Paraphrased-query multi-fact stressor; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_prefix_targets_paraphrase.md` | VALID | Query+target-aware hippocampus prefix; no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_real_driver_needle_sweep.md` | VALID | Real-driver needle sweep (length 512); no reranker or leakage | — |
| 2026-06-27 | `2026-06-27_use_agent_prefix_validation.md` | VALID | Live CortexAgent hippocampus prefix validation; no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_01_desaturation.md` | VALID | Lane 01 desaturation (isolated eval suite metrics); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_02_kv_slot_injection.md` | VALID | Lane 02 KV-slot fact injection (isolated lane experiment); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_03_layer_L_extraction.md` | VALID | Lane 03 layer-L extraction (HF diagnostics, spec refutation); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_04_context_attractors.md` | VALID | Lane 04 context-addressed slot store (isolated lane experiment, separate slot store); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_05_metabolism_status.md` | VALID | Lane 05 metabolism status (prior session carry-forward); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_06_bounded_growth.md` | VALID | Lane 06 bounded growth (isolated A0b/A1 autoencoder); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_07_world_model_critic.md` | VALID | Lane 07 world-model critic (isolated lane experiment); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_08_cross_lane_synthesis.md` | VALID | Lane 08 cross-lane synthesis (composed isolated mechanisms); no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_lane_orchestration_session_summary.md` | VALID | Session summary for lane orchestration; no reranker or leakage | — |
| 2026-06-28 | `2026-06-28_order_shuffle_stressor.md` | VALID | Order-shuffle stressor (ingestion pipeline); no reranker or leakage | — |
| 2026-06-29 | `2026-06-29_curriculum_experiments_aggregate.md` | **PARTIAL** | "Scope-slot reranker (post-aggregate)" and "Verification rerun" sections are INVALIDATED (broken reranker). Pre-reranker aggregate (7/7) and post-fix update sections are VALID. | `2026-06-30_scope_slot_reranker_fix.md` |
| 2026-06-29 | `2026-06-29_knowledge_core_expansion_1m.md` | VALID | Pre-reranker honest scope=0.0 measurement; post-fix update sections are post-reranker-fix and valid. No leakage-era Stage-2/5 claims. | — |
| 2026-06-29 | `2026-06-29_reranker_ab_comparison.md` | **PARTIAL** | Original A/B comparison body (lines 1–117) is INVALIDATED (run on broken reranker). "Update: bug fix changes conclusions" section is VALID. | `2026-06-30_scope_slot_reranker_fix.md` |
| 2026-06-30 | `2026-06-30_cortex_dim_benchmark.md` | VALID | Cortex dimension benchmark run after reranker fix; no leakage-era claims. Bilinear policy head analysis is valid. | — |
| 2026-06-30 | `2026-06-30_residual_to_identity_wiring.md` | VALID | Residual-to-identity wiring report documents both pre- and post-reranker-fix; update sections correctly identify the fix. No leakage-era claims. | — |
| 2026-06-30 | `2026-06-30_scope_slot_reranker_fix.md` | **PARTIAL** | Bug diagnosis and fix description are VALID. "Curriculum Impact" table is SUPERSEDED (leakage-era 1.00 claims). | `2026-07-01_honest_post_leakage_baseline.md` |
| 2026-07-01 | `2026-07-01_honest_post_leakage_baseline.md` | VALID | **Current reference point.** Post-leakage-removal honest baseline. These are the numbers all future work must beat. | — |
| 2026-07-01 | `2026-07-01_stage5_scope_dsi_benchmarks.md` | **PARTIAL** | Stage 5 scope=1.00 and retention=1.00 claims are SUPERSEDED (leakage-era `_SCOPE_TEACHING`). DSI Fact Index implementation, external benchmark integration, and papers analysis are VALID. | `2026-07-01_honest_post_leakage_baseline.md` |

## Summary

| Classification | Count |
|----------------|-------|
| VALID | 36 |
| PARTIAL | 4 |
| INVALIDATED (pure) | 0 |
| SUPERSEDED (pure) | 0 |

All files with INVALIDATED or SUPERSEDED content are classified PARTIAL because they also contain VALID content (either pre-reranker measurements, post-fix updates, or non-curriculum architecture/implementation documentation).

## Reference Point

The **honest post-leakage baseline** (`2026-07-01_honest_post_leakage_baseline.md`) is the current reference point. Key honest numbers (real driver, semantic scoring):

| Stage | Post Accuracy |
|-------|---------------|
| Stage 0: Sense grounding | 0.88 |
| Stage 1: Transfer | 0.75 |
| Stage 2: Scope control | 0.69 |
| Stage 3: Dialog | 0.38 |
| Stage 4: Consolidation | 0.90 |
| Stage 5: Cross-domain | 0.92 |

These supersede all prior Stage-2 and Stage-5 claims in the ledger.
