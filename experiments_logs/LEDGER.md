# Experiments Logs Ledger — Authoritative Index

**Date:** 2026-07-11 (updated through campaign `0d48130` adjudication, Exp03 `ad77e93` real-driver closure, and R18 5-seed diagnostic clarification)
**Purpose:** This ledger classifies every experiment log against three
invalidation events:

1. **Scope-slot reranker bugs** (fixed 2026-06-30): Three compounding bugs
   silently prevented the reranker from functioning correctly between its
   introduction on 2026-06-29 and the fix on 2026-06-30. See
   `2026-06-30_scope_slot_reranker_fix.md`.
2. **Test-set leakage removal** (2026-07-01): Two leakage paths
   (`_SCOPE_TEACHING` per-episode-ID entries, `prefix_targets=[probe.expected]`
   for scope probes) were removed, superseding all prior Stage-2/Stage-5 scope
   claims. See `2026-07-01_honest_post_leakage_baseline.md`.
3. **Gameable-metrics retirement** (2026-07-01, Sprint 0.4): lane_07's
   "0-by-construction" lexical baseline was replaced with a competitive
   token-overlap baseline (headline gap collapsed 1.0 → 0.0); lane_05's
   coverage-as-score was split from the honest result (= 0.0); lane_01's
   sub-metric set was frozen. See the retirement addenda in
   `2026-06-28_lane_07_world_model_critic.md` and
   `2026-06-28_lane_05_metabolism_status.md`, and the audit in
   `2026-07-01_remediation_audit.md`.

**Current reference point:** `2026-07-01_honest_post_leakage_baseline.md` —
these are the numbers all future work must beat. The 13.5x drift claim
(`044cb51`) is retracted (`2026-07-01_s2_4_breakthrough_ablation.md`).

## Classification Key

| Class | Meaning |
|-------|---------|
| **VALID** | Unaffected by any invalidation event |
| **INVALIDATED** | Depends on the broken reranker window (2026-06-29 to 2026-06-30 pre-fix) |
| **SUPERSEDED** | Leakage-era Stage-2/5 claims, or gameable-metric headlines, replaced by honest re-runs |
| **PARTIAL** | Mixed — some sections valid, some invalidated/superseded; see individual file banner |

## Ledger

| Date | File | Classification | Reason | Superseded By |
|------|------|----------------|--------|---------------|
| 2026-06-19 | `2026-06-19_extended_learning_evaluation.md` | **PARTIAL** | Aggregate ranking put NeuralHippocampus at 1.000 (above the Oracle 0.844) — a measurement artifact: it scored internal bookkeeping, not learned behavior. Caught by the 2026-06-21 review (see `NOTES.md`). Ranking table is SUPERSEDED; the artifact diagnosis is VALID. | `NOTES.md` (2026-06-21 review) |
| 2026-06-22 | `2026-06-22_organism_curriculum_and_lm_perception.md` | VALID | Organism curriculum + LM perception design; predates the reranker, leakage, and gameable metrics | — |
| 2026-06-23 | `2026-06-23_cortex_kv_contract.md` | VALID | Cortex KV contract design; predates all three invalidation events | — |
| 2026-06-24 | `2026-06-24_cortexagent_raw_hidden_steering.md` | VALID | CortexAgent raw-hidden steering probe; predates all three invalidation events | — |
| 2026-06-25 | `2026-06-25_prefix_steering_poc.md` | VALID | Prefix steering PoC; predates all three invalidation events | — |
| 2026-06-25 | `2026-06-25_svd_init_proj_c_persistence.md` | VALID | SVD-init proj_c persistence; predates all three invalidation events | — |
| 2026-06-26 | `2026-06-26_embedder_fork_mock_foreign.md` | VALID | Ingestion embedder architecture; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_hybrid_consolidation_architecture.md` | VALID | Architecture S vs H consolidation; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_ingestion_pipeline_scaffold.md` | VALID | Ingestion pipeline scaffold; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor.md` | VALID | Multi-fact stressor (mock driver); no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor_prefix.md` | VALID | ReservedPosition prefix stressor; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_multi_fact_stressor_real_driver.md` | VALID | Real-driver multi-fact stressor; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_needle_per_turn_stressor.md` | VALID | Needle-per-turn stressor tests; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_needle_sweep_script.md` | VALID | Needle sweep benchmark script; no reranker, leakage, or gameable metric | — |
| 2026-06-26 | `2026-06-26_policy_head_ranking_loop.md` | VALID | Policy head ranking loop (pre-reranker era); curriculum numbers from policy head path, not reranker | — |
| 2026-06-26 | `2026-06-26_salience_threshold_ablation.md` | VALID | Salience filter ablation for ingestion; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_auto_consolidate_sh.md` | VALID | Auto-consolidate S vs H probe; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_contrastive_cvec_discovery.md` | VALID | Contrastive cvec discovery and logit biasing; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_cortex_hippocampus_prefix.md` | VALID | Hippocampus-derived prefix in CortexAgent; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_cvec_prefix_composition_tradeoffs.md` | VALID | Cvec+prefix composition tradeoffs; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_deprecate_auto_prefix.md` | VALID | Deprecation of stressor auto-prefix; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_domain_recall_metric.md` | VALID | Domain-level recall metric; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_foreign_minilm_embedder.md` | VALID | Foreign MiniLM embedder integration; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_hippocampus_auto_prefix.md` | VALID | Hippocampus-derived auto-prefix (stressor wrapper); no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_hybrid_cap_sh.md` | VALID | Configurable hybrid consolidation cap; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_knowledge_store_prefix_targets.md` | VALID | KnowledgeStore prefix_targets integration; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_length_4096_needle_sweep.md` | VALID | Real-driver needle sweep at length 4096; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_memory_per_byte_sh.md` | VALID | Memory-per-byte S vs H probe; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_multi_fact_embedder_comparison.md` | VALID | Same-LM vs foreign-MiniLM embedder comparison; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_paraphrase_mode.md` | VALID | Paraphrased-query multi-fact stressor; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_prefix_targets_paraphrase.md` | VALID | Query+target-aware hippocampus prefix; no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_real_driver_needle_sweep.md` | VALID | Real-driver needle sweep (length 512); no reranker, leakage, or gameable metric | — |
| 2026-06-27 | `2026-06-27_use_agent_prefix_validation.md` | VALID | Live CortexAgent hippocampus prefix validation; no reranker, leakage, or gameable metric | — |
| 2026-06-28 | `2026-06-28_lane_01_desaturation.md` | **PARTIAL** | Desaturation-count acceptance criterion (a metric about metrics) retired by S0.4; sub-metric set now frozen. Lane mechanism itself valid. | `2026-07-01_remediation_audit.md` (Finding 3) |
| 2026-06-28 | `2026-06-28_lane_02_kv_slot_injection.md` | VALID | Lane 02 KV-slot fact injection (isolated lane experiment); no reranker, leakage, or gameable metric | — |
| 2026-06-28 | `2026-06-28_lane_03_layer_L_extraction.md` | VALID | Lane 03 layer-L extraction refutation. **Confirmed** by S1.4 (`2026-07-01_s1_4_hf_layer_probe.md`) on two architectures via HF substrate — the refutation is a model property, not a llama.cpp keyhole. | — |
| 2026-06-28 | `2026-06-28_lane_04_context_attractors.md` | VALID | Lane 04 context-addressed slot store (isolated lane experiment, separate slot store); no reranker, leakage, or gameable metric | — |
| 2026-06-28 | `2026-06-28_lane_05_metabolism_status.md` | **PARTIAL** | Coverage-as-score (1.0) retired by S0.4 — split from the honest `lane_05_result` (= 0.0). Coverage and result are now separate outputs. See retirement addendum in-file. | `2026-07-01_remediation_audit.md` (Finding 3) |
| 2026-06-28 | `2026-06-28_lane_06_bounded_growth.md` | **PARTIAL** | A0b "seed-regenerable" autoencoder met the byte target by regenerating a random matrix from a seed — by construction learned updates can't persist. Compression metric passed; thesis (compress *learned* experience) abandoned in that move. Flagged by the audit. | `2026-07-01_remediation_audit.md` (Finding 3) |
| 2026-06-28 | `2026-06-28_lane_07_world_model_critic.md` | **PARTIAL** | Headline `marker_free_uptake_gap = 1.0` SUPERSEDED — lexical baseline was "0 by construction"; replaced with competitive token-overlap baseline, gap collapsed to 0.0 (S0.4). Mechanism analysis and TD(0) notes remain VALID. See in-file retirement addendum. | `2026-07-01_remediation_audit.md` (Finding 3) |
| 2026-06-28 | `2026-06-28_lane_08_cross_lane_synthesis.md` | VALID | Lane 08 cross-lane synthesis (composed isolated mechanisms); no reranker, leakage, or gameable metric | — |
| 2026-06-28 | `2026-06-28_lane_orchestration_session_summary.md` | VALID | Session summary for lane orchestration; no reranker, leakage, or gameable metric | — |
| 2026-06-28 | `2026-06-28_order_shuffle_stressor.md` | VALID | Order-shuffle stressor (ingestion pipeline); no reranker, leakage, or gameable metric | — |
| 2026-06-29 | `2026-06-29_curriculum_experiments_aggregate.md` | **PARTIAL** | "Scope-slot reranker (post-aggregate)" and "Verification rerun" sections are INVALIDATED (broken reranker). Pre-reranker aggregate (7/7) and post-fix update sections are VALID. | `2026-06-30_scope_slot_reranker_fix.md` |
| 2026-06-29 | `2026-06-29_knowledge_core_expansion_1m.md` | VALID | Pre-reranker honest scope=0.0 measurement; post-fix update sections are post-reranker-fix and valid. No leakage-era Stage-2/5 claims. | — |
| 2026-06-29 | `2026-06-29_reranker_ab_comparison.md` | **PARTIAL** | Original A/B comparison body (lines 1–117) is INVALIDATED (run on broken reranker). "Update: bug fix changes conclusions" section is VALID. | `2026-06-30_scope_slot_reranker_fix.md` |
| 2026-06-30 | `2026-06-30_cortex_dim_benchmark.md` | VALID | Cortex dimension benchmark run after reranker fix; no leakage-era claims. Bilinear policy head analysis is valid. | — |
| 2026-06-30 | `2026-06-30_residual_to_identity_wiring.md` | VALID | Residual-to-identity wiring report documents both pre- and post-reranker-fix; update sections correctly identify the fix. No leakage-era claims. | — |
| 2026-06-30 | `2026-06-30_scope_slot_reranker_fix.md` | **PARTIAL** | Bug diagnosis and fix description are VALID. "Curriculum Impact" table is SUPERSEDED (leakage-era 1.00 claims). | `2026-07-01_honest_post_leakage_baseline.md` |
| 2026-07-01 | `2026-07-01_honest_post_leakage_baseline.md` | VALID | **Current reference point.** Post-leakage-removal honest baseline. These are the numbers all future work must beat. | — |
| 2026-07-01 | `2026-07-01_remediation_audit.md` | VALID | Full-repo experiment audit driving `SPRINT.md`. Meta-document; classifies the three invalidation events and the five strategic findings. | — |
| 2026-07-01 | `2026-07-01_s1_3_hf_kv_slot_injection.md` | VALID | S1.3 HF-substrate KV-slot fact injection — REFUTE on absolute recall (rank-1 on 1/3 facts); KV-splice ≡ text-prefix parity found. New experiment, leak-free. | — |
| 2026-07-01 | `2026-07-01_s1_4_hf_layer_probe.md` | VALID | S1.4 HF layer-L probe — REFUTE on Qwen-0.5B (gap −0.083) and LFM2.5 (+0.058 < +0.10). Confirms lane_03; retires Goal 2's mid-layer assumption. New experiment, pre-registered. | — |
| 2026-07-01 | `2026-07-01_s2_4_breakthrough_ablation.md` | VALID | S2.4 single-variable ablation of the "13.5x breakthrough" (`044cb51`) — **RETRACTED as magnitude inflation**. Survival ratio 0.354 < 0.5; control logits rose more than target. New experiment, pre-registered. | — |
| 2026-07-01 | `2026-07-01_stage5_scope_dsi_benchmarks.md` | **PARTIAL** | Stage 5 scope=1.00 and retention=1.00 claims are SUPERSEDED (leakage-era `_SCOPE_TEACHING`). DSI Fact Index implementation, external benchmark integration, and papers analysis are VALID. | `2026-07-01_honest_post_leakage_baseline.md` |
| 2026-07-02 | `2026-07-02_s1_1_model_selection.md` | VALID | S1.1 HF substrate model selection — Qwen2.5-0.5B-Instruct (82.8 ms/tok). Decision record, not a claim under any invalidation event. | — |
| 2026-07-02 | `2026-07-02_s2_1_minimal_loop.md` | VALID | S2.1 minimal metabolism loop — **REFUTE H-LOOP**. `loop_delta_holdout=0.0000` (5 seeds, 3 holdout probes post-repair), `loop_compounding_rho=nan`. Validity gate passed (vanilla 0.0 < 0.5). Post-reranker-fix, post-leakage-removal, pre-registered. Mechanism: prefix budget eviction at K=8. | — |
| 2026-07-02 | `2026-07-02_s2_2_kv_content_path.md` | VALID | S2.2 KV content channel — **BLOCKED** (degenerate 0-probe holdout + S2.1 REFUTE gate binds). C1/C2 all 0.0000 due to 0 holdout probes. Addendum adjudicates as BLOCKED, not REFUTE. Implementation (`minimal_loop_kv.py`) merged and valid. | — |
| 2026-07-02 | `2026-07-02_s2_5_forgetting_test.md` | VALID | S2.5 forgetting test — **BLOCKED** (0 holdout probes, validity gate failed all 5 seeds). Not a refutation of H-FORGET. Addendum confirms BLOCKED via S2.1 REFUTE gate. Deletion APIs + 2×2 harness merged. | — |
| 2026-07-02 | `2026-07-02_s3_m1_subtractive_ablation.md` | VALID | S3.M1 subtractive organ ablation (real GGUF driver, dev split, 3 seeds). ScopeSlotReranker +0.0465 all-stage (largest single-organ effect); DSI net-harmful (−0.060). No reranker, leakage, or gameable metric. | — |
| 2026-07-02 | `2026-07-02_s3_m2_retrieval_ablation.md` | VALID | S3.M2 additive retrieval ablation (HF driver, eval v2 holdout, 3 seeds). Scope-slot reranker zero-variance positive (S0 +0.667, S4 +0.250); hippocampus-at-answer Δ=0.000 exactly; DSI unsupported. Post-reranker-fix, post-leakage-removal. | — |
| 2026-07-03 | `2026-07-03_s3_organ_triage_adjudication.md` | VALID | S3 organ triage adjudication — combines M1+M2 into KEEP/RETRIEVAL-BASELINE/ARCHIVE verdicts. ScopeSlotReranker=RETRIEVAL-BASELINE; all other organs=ARCHIVE. research/15 declared VACUOUS. No reranker, leakage, or gameable metric. | — |
| 2026-07-03 | `2026-07-03_eval_v2_1_expansion.md` | VALID | Eval v2→v2.1 curriculum expansion (S0.6 growth path). +12 new ambiguous words across stages 0/1/2; stage-1 holdout 1→9 probes. Existing episodes/probes never modified. Regression locks updated. No reranker, leakage, or gameable metric. | — |
| 2026-07-11 | `2026-07-11_campaign_0d48130.md` | VALID | **Campaign 0d48130 curated evidence log.** 10 experiment outcomes across 3 commits and 2 providers (kaggle CPU-only, colab). Scientific outcomes: 3 POSITIVE (Exp04, Exp06, R18 gate), 2 POSITIVE+NULL (Exp07, R18 full PARTIAL), 3 NULL (Exp01, Exp05, R14 M2B metricless), 1 REFUTATION (Exp02), 1 INFRASTRUCTURE BLOCKED (Exp03, original campaign). Exp03 reproducibility closure appended 2026-07-11 (commit `ad77e93`): real-driver rerun exit 0, `layer_l_silhouette_gap=0.10925446726657728` (> +0.10, threshold unchanged) → positive/accept for this single closure; S1.4 not reopened. See the log and `2026-07-11_exp03_real_driver_closure.json`. | — |
| 2026-07-11 | `2026-07-11_exp03_real_driver_closure.json` | VALID | **Exp03 real-driver reproducibility closure.** Durable execution report object: commit `ad77e93`, `--driver real`, Colab, exit 0, `layer_l_silhouette_gap=0.10925446726657728` (> +0.10 registered threshold, unchanged), all ASI scores, model provenance (`LiquidAI/LFM2.5-1.2B-Instruct` rev `868df74d…`, manifest `infrastructure/kaggle/model_manifests/lfm2_5-1_2b-instruct.json`), infrastructure fix description. Single run on one architecture; does not reopen S1.4. | — |

## Summary

| Classification | Count |
|----------------|-------|
| VALID | 56 |
| PARTIAL | 9 |
| INVALIDATED (pure) | 0 |
| SUPERSEDED (pure) | 0 |

All files with INVALIDATED or SUPERSEDED content are classified PARTIAL because
they also contain VALID content (either pre-reranker measurements, post-fix
updates, mechanism analysis, or non-curriculum architecture/implementation
documentation).

## Reference Point

The **honest post-leakage baseline** (`2026-07-01_honest_post_leakage_baseline.md`)
is the current reference point. Key honest numbers (real driver, semantic scoring):

| Stage | Post Accuracy |
|-------|---------------|
| Stage 0: Sense grounding | 0.88 |
| Stage 1: Transfer | 0.75 |
| Stage 2: Scope control | 0.69 |
| Stage 3: Dialog | 0.38 |
| Stage 4: Consolidation | 0.90 |
| Stage 5: Cross-domain | 0.92 |

These supersede all prior Stage-2 and Stage-5 claims in the ledger.

## Retracted headline claims (session `f645e4af`)

Three of the four headline claims the repo carried into July were re-adjudicated
under pre-registered, leak-free conditions and fell:

| Claim | Source | Verdict | Evidence |
|-------|--------|---------|----------|
| Stage-2 scope = 1.00 | leakage-era curriculum | SUPERSEDED → 0.69 | `2026-07-01_honest_post_leakage_baseline.md` |
| Stage-5 cross-domain = 1.00 | leakage-era curriculum | SUPERSEDED → 0.92 | `2026-07-01_honest_post_leakage_baseline.md` |
| lane_07 marker-free gap = 1.0 | "0-by-construction" baseline | SUPERSEDED → 0.0 | lane_07 in-file addendum |
| Mid-layer hiddens beat final layer | lane_03 (llama.cpp keyhole) | REFUTED on 2 architectures | `2026-07-01_s1_4_hf_layer_probe.md` |
| 13.5x metabolism drift (`044cb51`) | 4-variable bundle, no ablation | RETRACTED as magnitude inflation | `2026-07-01_s2_4_breakthrough_ablation.md` |

What survives: the KV-splice mechanism (≡ text prefix at zero token cost, see
S1.3), the scope-slot reranker's legitimate 0.92, and a frozen eval that can no
longer be quietly bent. See `SPRINT.md` for the remediation plan and current
sprint status.

## Campaign 0d48130 Adjudication (2026-07-11)

**Full curated evidence log:** `2026-07-11_campaign_0d48130.md` — per-run
metrics, seed distributions, non-runnable inventory, artifact provenance paths,
infrastructure fixes, and next steps. The summary below is a quick reference;
the curated log is the durable record.

Five execution summaries adjudicated from three source commits: `0d48130`
(Exp06 batch, kaggle), `537260c` (colab-importfix + R18 gate + R18 full),
and `2a22049` (R14 M2B fixed re-run, kaggle). All completed jobs ran under
CPU-only contract (cuda_available=false, torch 2.10.0+cpu).

**Scientific outcomes (complete):**

| Experiment | Outcome | Primary metric |
|------------|---------|----------------|
| Exp01 | NULL (behavior-delta transfer) | `v2_behavior_delta_mock=0.0` |
| Exp02 | REFUTATION (KV-slot injection) | `kv_slot_rank1_count=0.0` |
| Exp04 | POSITIVE (scope selectivity) | `scope_selectivity_index=1.0` |
| Exp05 | NULL (metabolism drift) | `metabolism_drift_delta=0.0` |
| Exp06 | POSITIVE (bounded growth) | `bounded_growth_m1_ratio=0.002079` (5 seeds, zero variance) |
| Exp07 | POSITIVE (marker-free uptake) + NULL (critic AUC) | `marker_free_uptake_gap=1.0`, `critic_auc_delta=0.0` |
| R18 gate | POSITIVE (gate passed) | `distill_delta_holdout=0.3333` (1 seed) |
| R18 full | PARTIAL (2/3 seeds positive, 1/3 null); 5-seed `stage_0` rerun diagnostic unless `teacher_dev_delta` ≥ 0.2 gate passes (currently 0.1765) | `distill_delta_holdout` mean=0.2222 (3 seeds) |
| R14 M2B | NULL (metricless completed run) | 3 seeds, exit 0, no `METRIC`/`ASI` values |

**Non-scientific outcomes:**

| Experiment | Status | Reason |
|------------|--------|--------|
| Exp03 | INFRASTRUCTURE BLOCKED (original campaign) → **REPRODUCIBILITY CLOSURE** (2026-07-11, commit `ad77e93`) | Original: Colab job failed (HF snapshot transfer failures); no metrics emitted. Not a scientific null or refutation. Closure: real-driver rerun (`--driver real`, Colab, exit 0) produced `layer_l_silhouette_gap=0.10925446726657728` (> +0.10, threshold unchanged) → positive/accept for this single reproducibility closure. Does not reopen or overturn the pre-registered S1.4 refutation (two architectures). Durable record: `2026-07-11_exp03_real_driver_closure.json`. |

**Seed distributions:** Exp06 — 5 seeds (0–4), `m1_ratio` zero variance,
bytes_per_delta spread ≤20 B across all agents. R18 full — 3 seeds:
`distill_delta_holdout` bimodal {0.3333, 0.3333, 0.0}; `teacher_dev_delta` and
`persistent_bytes` identical across seeds. Colab experiments (01/02/04/05/07)
are single-run with no cross-seed variance data.

**R18 5-seed diagnostic note:** The 3-seed `stage_0` run is PARTIAL.
`teacher_dev_delta=0.1765` is below the ≥ 0.2 validity gate, so a 5-seed
`stage_0` rerun is **diagnostic** unless that unchanged gate passes. No
threshold changes.

Source: `2026-07-11_campaign_0d48130.md` (adjudicated from
`/tmp/oczy-campaign-0d48130/` execution summaries) and
`2026-07-11_exp03_real_driver_closure.json` (ad77e93 real-driver closure,
from `/tmp/oczy-exp03-real-run-v2/`). No threshold changes or
causal claims beyond measured metrics.

## Notes (conceptual, non-log)

Analysis documents live in `notes/` (created 2026-07-03) — they interpret
logged evidence but are not themselves experiment logs:

- `notes/2026-07-03_steering_vs_posture_postmortem.md` — why the
  steering/posture intuition failed (three broken assumptions: common-mode
  accumulation has magnitude not direction; constant vectors cannot condition;
  mention-space ≠ use-space), synthesizing S1.3, S1.4, S2.1, S2.4. Successor
  mechanism pre-registered in `research/18-consolidation-as-distillation.md`.
