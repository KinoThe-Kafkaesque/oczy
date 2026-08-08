# Experiments Logs Ledger — Authoritative Index

**Date:** 2026-08-08 (updated through campaign `d756ff4 R24 Phase A v1 invalidation` and v2 protocol authorization, Exp03 `ad77e93` real-driver closure, R18 5-seed diagnostic clarification, R18 5-seed diagnostic adjudication, R18 mechanism diagnostics adjudication, R19 DEV calibration adjudication, R20 DEV implementation/smoke adjudication, R20 INT8 transport recovery, and the corrected R20 meta_cortex/v2 DEV calibration closure with local diagnostics)
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

**Legacy v2 reference point:** `2026-07-01_honest_post_leakage_baseline.md`.
Eval v2.2 repairs the runner protocol and split policy, so a new real-driver
multi-seed baseline is pending. The legacy numbers must not be presented as a
v2.2 difficulty curve. The 13.5x drift claim (`044cb51`) remains retracted
(`2026-07-01_s2_4_breakthrough_ablation.md`).

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
| 2026-07-11 | `2026-07-11_eval_v2_2_protocol_repair.md` | VALID | Human-approved protocol repair: Stage 1 probe-only, Stage 3 episode-interleaved, Stage 4 consolidate-before-post-test, consistent semantic scoring, category-stratified v2.2 split; legacy `salt="v2"` preserved. New baseline pending. | — |
| 2026-07-11 | `2026-07-11_campaign_0d48130.md` | VALID | **Campaign 0d48130 curated evidence log.** 10 experiment outcomes across 3 commits and 2 providers (kaggle CPU-only, colab). Scientific outcomes: 2 POSITIVE (Exp04, Exp06), 1 POSITIVE+NULL (Exp07), 3 NULL (Exp01, Exp05, R14 M2B metricless), 1 REFUTATION (Exp02), 2 BLOCKED at teacher validity gate / diagnostic only (R18 gate, R18 full), 1 INFRASTRUCTURE BLOCKED (Exp03, original campaign). Exp03 reproducibility closure appended 2026-07-11 (commit `ad77e93`): real-driver rerun exit 0, `layer_l_silhouette_gap=0.10925446726657728` (> +0.10, threshold unchanged) → positive/accept for this single closure; S1.4 not reopened. See the log and `2026-07-11_exp03_real_driver_closure.json`. | — |
| 2026-07-11 | `2026-07-11_exp03_real_driver_closure.json` | VALID | **Exp03 real-driver reproducibility closure.** Durable execution report object: commit `ad77e93`, `--driver real`, Colab, exit 0, `layer_l_silhouette_gap=0.10925446726657728` (> +0.10 registered threshold, unchanged), all ASI scores, model provenance (`LiquidAI/LFM2.5-1.2B-Instruct` rev `868df74d…`, manifest `infrastructure/kaggle/model_manifests/lfm2_5-1_2b-instruct.json`), infrastructure fix description. Single run on one architecture; does not reopen S1.4. | — |
| 2026-07-11 | `2026-07-11_live_runner_queue.json` | VALID | **Live runner queue launch provenance and completion record.** Durable record of the live experiment queue at implementation commit `5b5e93c63d769fea7854073a4e6c359e5d36606f`. Records UTC launch date, live local state paths under `/tmp/oczy-live-queue/` (batch, state, campaign, campaign_manifest — explicitly labeled as non-tracked live local state), scheduler flags (`--watch-batch --watch-interval 30`), additive provider capacity contract (10 Kaggle hard-cap + AIMD-learned Colab X, no global cap), source dataset/archive provenance (`abdellahkadem/oczy-source-5b5e93c63d76`, sha256 `bc1ff926…`), and the first job `r18-distillation-5seed-diagnostic` (Kaggle, kernel `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, module `oczy.experiments.consolidation_distillation`, args `--seeds 5 --max-steps 10 --stage stage_0_grounding`). **Job completed** (exit 0, state=succeeded, completed 2026-07-11T15:04:52Z, collected 2026-07-11T15:04:54Z). Scientific classification: **BLOCKED** at teacher validity gate (`teacher_dev_delta=0.17647058823529413` < 0.2, identical across all 5 seeds). No positive scientific verdict claimed. Full adjudication in `2026-07-11_r18_five_seed_diagnostic.json`. | — |
| 2026-07-11 | `2026-07-11_r18_five_seed_diagnostic.json` | VALID | **R18 5-seed diagnostic adjudication.** Durable execution/adjudication JSON: commit `5b5e93c63d769fea7854073a4e6c359e5d36606f`, Kaggle CPU, kernel `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, exit 0, 5 seeds. Per-seed `distill_delta_holdout` {0.3333, 0.3333, 0.0, 0.3333, 0.3333} (4/5 positive, seed 2 null); `teacher_dev_delta=0.17647058823529413` identical across all seeds. Mean `distill_delta_holdout=0.2667`, mean `specificity_delta=0.0261`. Gate comparison: `0.1765 < 0.2` → FAILED. Scientific classification: **BLOCKED** at teacher validity gate / diagnostic only. No H-DISTILL verdict permitted (teacher gate failed after registered fallback). 4/5 conditional signal and seed-2 null both visible. No threshold, metric, or research spec changed. | — |
| 2026-07-12 | `2026-07-11_r18_mechanism_diagnostics.json` | VALID | **R18 mechanism diagnostics adjudication.** Durable execution/adjudication JSON: commit `33169cc0340bf752a67adf63721ec64cb5f3c9f8`, Kaggle CPU. Three diagnostic jobs (teacher ceiling, prompt-contract, training trajectory), all exit 0. Teacher ceiling (n=17): vanilla=0, raw_prefix=0.17647058823529413, chat_template=0; neither reaches 0.2 gate; registered chat fallback (0) worse than raw_prefix (0.1765). Prompt-contract audit: all six defect counts (issue/malformed/missing/truncated/answer-leak/mismatch) = 0; teacher_correct_rate=0.17647058823529413; raw/chat prompt accuracies 0; no structural prompt defect. Training trajectory: first submission failed HTTP 400 (long slug, preserved); short-slug retry exit 0 after ~12798s (run of record, preserved). Train loss 0.70→0.16, mean slope -0.0615, second-half -0.0190; underfit=1, instability=1, saturation=0, max final-loss divergence 0.01259. Final DEV student accuracies seeds 0–4: {0.117647, 0, 0, 0, 0.117647}; teacher 0.17647; seed 2 not uniquely divergent (seeds 1, 3 also 0). Adjudication decomposes failed gate into three axes: (1) prompt integrity — NO DEFECT; (2) capability ceiling — teacher expressivity/prompt-task ceiling IS THE BLOCKER; (3) optimization dynamics — token loss fits but DEV behavior unstable/weak, not saturated. Classification: **BLOCKED** at teacher validity gate / diagnostic only. No H-DISTILL verdict permitted. No threshold/spec/eval changes. All nulls visible. | — |
| 2026-07-12 | `2026-07-12_r19_dev_calibration.json` | VALID | **R19 DEV calibration adjudication.** Durable execution/adjudication JSON: commit `bd1ead9a8358b675af5e929c53a01eb505839639`, Kaggle CPU. calibrate-dev v4 exit 0, all metrics collected. Manifest SHA-256 `77ef4607…`, parameter_total 60,388/64,000. DEV articulation gate **FAILED** (Arm B latent-control DEV accuracy ≤ C1 random-cortex DEV accuracy); oracle ceiling 0.357143 > 0 (PASSED independently). No signoff requested; no holdout accessed. Three prior infrastructure-failed attempts (v1 offline model resolution, v2 source-path/provenance + feature explosion, v3 artifacts not rooted in `/kaggle/working`); v4 infrastructure-successful but scientifically BLOCKED. C7 adapter discrepancy: manifest `c7_available=true` but `_try_s3m2a_retrieval_adapter()` returns None. R20 remains separately blocked on human signoff. No H-LATENT or H-LABEL verdict permitted. | — |
| 2026-07-12 | `2026-07-12_r20_dev_smoke.json` | VALID | **R20 DEV implementation/smoke adjudication.** Durable execution/adjudication JSON: commit `e26d8291879d078b701f19802f72041e08cfd6a6`, Kaggle CPU, kernel `abdellahkadem/oczy-r20-dev-v3-e26d8291879d`, exit 0, audit_status ok. Infrastructure/mechanism smoke only — no scientific verdict. Three attempts: v1 failed (offline loader failure), v2 failed (inference-tensor/autograd failure), v3 succeeded after fixes. Audit invariants: frozen organ hash identical before/after `d8a3a3b…`, checkpoint theta hash `8d6c41c5…`, trace count 0 after deletion, online optimizer counts unchanged. 207,364 theta params / 829,456 bytes, F/S 64×64, bank 3×896, optimizer steps 1, best DEV validation score 0.0. Causal DEV deltas: trained-vs-update 0, untrained 0, shuffled 0, zeroed 0, swapped 0.0666667 — recorded as observed mechanism smoke. Test suites: focused 262 passed/2 skipped, organ 54 passed/2 skipped. **Meta-test remains BLOCKED**: no frozen `meta_cortex/v1` instrument, distribution checks, power analysis, manifest, or human signoff exists. No ACCEPT/REFUTE verdict permitted. No threshold, metric, baseline, episode, scoring, eval manifest, or research spec changed. No holdout accessed; no signoff requested. | — |
| 2026-07-09 | `2026-07-09_r18_implementation.md` | VALID | **R18 consolidation-as-distillation implementation and first runs.** Initial implementation of `consolidation_distillation.py`, autoresearch segment 10 wiring (runs #200–#202). Run #202: Qwen2.5-0.5B + LoRA rank 2, ~220s, `teacher_dev_delta ~0.176` below 0.2 validity gate — teacher gate FAILED, no H-DISTILL verdict. Concurrent: Numba CPU kernel acceleration (`62ab18e`), Kaggle research compute workflow (`6dee16b`), INT8 rescheduling planning. Gate failure confirmed by later mechanism/five-seed diagnostics (2026-07-11). | — |
| 2026-07-22 | `2026-07-16_campaign_959e114.md` | PARTIAL | **R20 INT8 meta_cortex/v2 DEV training and calibration campaign.** Training/checkpoint, transport, runtime, and failure-timing evidence remains valid. Scientific aggregation of the v5 shards remains invalid: source `949871b…` chose C6 donors from shard-local membership, making `state_addressing_delta` partition-dependent, and width-1 shards omitted C6. The prior control-plane-only width claim remains withdrawn; v5 is infrastructure-valid but scientifically invalid. Corrected source `a8c98d638209a8425b14a0f853e9fc46ae7da581` uses canonical next-within-family donors. The completed v6 closure is recorded in `2026-08-05_r20_corrected_dev_calibration.json`: complete corrected coverage and trace audits are valid, but all endpoint effects and condition scores are zero and power feasibility is blocked. Thus the DEV decision is **BLOCKED/no-go**, not REFUTED; no meta-test or holdout was accessed and no H-META-CORTEX verdict is permitted. | `2026-08-05_r20_corrected_dev_calibration.json` |
| 2026-07-26 | `2026-07-26_local_t550_gpu_throughput_probe.md` | VALID | **Local T550 GPU throughput probe — infrastructure, not a cortex experiment.** Benchmarked the local NVIDIA T550 Laptop GPU (4 GB, Turing compute 7.5, torch 2.6.0+cu124, fp16) against the local i7-1260P CPU (fp32) on the five small causal LMs in the local HF cache, including the pinned `Qwen/Qwen2.5-0.5B-Instruct` organ and the `Qwen/Qwen2.5-1.5B-Instruct` fallback. Workload: ~512-token prompt, 128 new tokens, greedy, KV cache on, batch 1 and 4. Result: at batch=1 the T550 is 1.1–2.0× faster than CPU on decode (Qwen-0.5B 27.4 vs 18.9 t/s; Qwen-1.5B 9.6 vs 4.9 t/s); at batch=4 the CPU beats the GPU on aggregate throughput for every model that fits (Qwen-0.5B 30.2 vs 47.8 t/s; LFM-1.2B 15.8 vs 28.1 t/s), and Qwen-1.5B OOMs on GPU at batch=4. **Decision: the T550 is not added as a verified compute path.** The CPU-only contract stands. The T550 is weaker than the archived T4, the frozen LM is not the research bottleneck, and wiring in a GPU code path would break the CPU-only contract in `infrastructure/kaggle/RESEARCH_GUIDE.md` and `AGENTS.md` rule 7. The T550 is acceptable for local dev iteration and benchmark scripts only. No `eval/v2`, `research/`, `lanes/`, or `experiments/organism_curriculum/` paths were modified; no remote compute was used; no scientific claim was made. | — |
| 2026-08-05 | `2026-08-05_r20_corrected_dev_calibration.json` | VALID | **R20 corrected meta_cortex/v2 DEV calibration closure.** VALID classifies execution, provenance, coverage, aggregation, and calibration-evidence integrity—not hypothesis acceptance. Corrected source `a8c98d6…`; 150 shards, 9,000 no-update records, 2,250 seed cells, five theta hashes; trace audits passed. All nine endpoints have mean/CI/SD exactly 0 in all three families, and all six conditions score 0 over every denominator. Equivalence margin 0; power feasibility `blocked`; no endpoint/family has a finite required N. Exact local reproduction stopped before generation on organ-hash mismatch; different-organ output and forcing runs are diagnostic only. **DEV decision: BLOCKED/no-go, not REFUTED.** Oracle gates, a signed candidate, and meta-test are absent; no holdout/meta-test was accessed and no ACCEPT/REFUTE verdict is claimed for H-META-CORTEX. | — |

| 2026-08-06 | `2026-08-06_campaign_r24_tiny_decoder.md` | **INVALIDATED** | Four v3 kernels completed with valid remote execution/provenance, but the scientific measurements are invalid: model initialization was unseeded, variable-length queries were right-padded without length-aware decoding, oracle attention ignored padding, and corpus rows included conflicting/undefined labels. The Deep-FiLM `7/264 vs 0/264` delta is not evidence. | `experiments/r24-tiny-decoder/v2_screen_plan.json` |

## Summary

| Classification | Count |
|----------------|-------|
| VALID | 64 |
| PARTIAL | 10 |
| INVALIDATED (pure) | 1 |
| SUPERSEDED (pure) | 0 |

All files with INVALIDATED or SUPERSEDED content are classified PARTIAL because
they also contain VALID content (either pre-reranker measurements, post-fix
updates, mechanism analysis, or non-curriculum architecture/implementation
documentation).

## Reference Point

The **honest post-leakage v2 baseline**
(`2026-07-01_honest_post_leakage_baseline.md`) is retained for historical
comparison only. It is not the current v2.2 reference. Key legacy numbers:

| Stage | Post Accuracy |
|-------|---------------|
| Stage 0: Sense grounding | 0.88 |
| Stage 1: Transfer | 0.75 |
| Stage 2: Scope control | 0.69 |
| Stage 3: Dialog | 0.38 |
| Stage 4: Consolidation | 0.90 |
| Stage 5: Cross-domain | 0.92 |

These supersede earlier leakage-era Stage-2 and Stage-5 claims, but a new v2.2
baseline is required before current stage-to-stage comparisons are made.

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
| R18 gate | BLOCKED at teacher validity gate / diagnostic only (`teacher_dev_delta=0.1765` < 0.2) | `distill_delta_holdout=0.3333` (1 seed) |
| R18 full | BLOCKED at teacher validity gate / diagnostic only (3-seed: 2/3 positive, 1/3 null; 5-seed `stage_0` rerun `teacher_dev_delta=0.1765` < 0.2, all 5 seeds identical; 4/5 positive holdout deltas, seed 2 null; no H-DISTILL verdict) | `distill_delta_holdout` mean=0.2222 (3 seeds), 0.2667 (5 seeds) |
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

**R18 5-seed diagnostic adjudication:** The 5-seed `stage_0` rerun completed
(exit 0, Kaggle CPU, kernel `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`).
`teacher_dev_delta=0.17647058823529413` is identical across all 5 seeds and
remains below the ≥ 0.2 validity gate. Scientific classification: **BLOCKED**
at teacher validity gate / diagnostic only. 4/5 seeds show positive
`distill_delta_holdout=0.3333`; seed 2 is null (0.0). Mean
`distill_delta_holdout=0.2667`, mean `specificity_delta=0.0261`. No H-DISTILL
verdict is permitted because the teacher gate failed after registered fallback.
No threshold changes. Durable record:
`2026-07-11_r18_five_seed_diagnostic.json`.

Source: `2026-07-11_campaign_0d48130.md` (adjudicated from
`/tmp/oczy-campaign-0d48130/` execution summaries),
`2026-07-11_exp03_real_driver_closure.json` (ad77e93 real-driver closure,
from `/tmp/oczy-exp03-real-run-v2/`), and
`2026-07-11_r18_five_seed_diagnostic.json` (5-seed diagnostic adjudication,
from `/tmp/oczy-live-queue/` live state). No threshold changes or
causal claims beyond measured metrics.

## R19 DEV Calibration Adjudication (2026-07-12)

**Full curated evidence log:**
`2026-07-11_campaign_0d48130.md` § R19 DEV calibration. **Durable
execution/adjudication JSON:**
`2026-07-12_r19_dev_calibration.json`.

Research/19 calibrate-dev phase ran from source commit
`bd1ead9a8358b675af5e929c53a01eb505839639` on Kaggle CPU. **Infrastructure:
COMPLETE** (exit 0, all metrics collected, manifest hash verified). **Scientific
verdict: BLOCKED at the pre-registered DEV articulation gate.**

### Attempt history

| Attempt | Outcome | Root cause |
|---------|---------|------------|
| v1 | INFRASTRUCTURE FAILURE | `LocalEntryNotFoundError`: hub ID used instead of local path under `HF_HUB_OFFLINE=1`. Fixed by `_resolve_load_target` resolver. |
| v2 | INFRASTRUCTURE FAILURE | Source archive mount path unavailable + feature explosion (`label_loss_mean=5.5358e21`, confidence saturated at 1.0). Fixed by SHA precedence and L2 normalization. |
| v3 | INFRASTRUCTURE FAILURE (artifact collection) | Artifacts not rooted in `/kaggle/working`; sentinel could not collect them. Fixed by rooting output paths. |
| v4 | INFRASTRUCTURE SUCCESS | All metrics collected, manifest hash `77ef4607…`. |

Attempts v1–v3 were infrastructure failures with no valid scientific evidence.
The v4 run was infrastructure-successful but scientifically BLOCKED.

### v4 calibration metrics

| Field | Value |
|-------|-------|
| Manifest SHA-256 | `77ef4607ff95c116b5b7b088a7f5cfa811b855d76feed9c329eb551ac586a1e2` |
| Parameter total | 60,388 / 64,000 (within budget) |
| DEV repeatability std | 0.0 |
| DEV confidence mean / std | 0.0525482 / 0.0002893 |
| DEV confidence range | 0.0520694 – 0.0528929 |
| DEV specificity acc | 0.134328 |
| Oracle ceiling (DEV) | 0.357143 (> 0 → PASSED) |
| DEV articulation gate | **FAILED** |
| Raw traces deleted / count | true / 0 |
| Holdout accessed | false |
| Signoff requested | false |

### Gate analysis

The oracle ceiling (0.357143 > 0) passes: the frozen LM can express the
taught behavior with a direct text prefix. The blocker is the DEV
articulation gate: the learned coupler (Arm B latent control) does not
produce a measurable improvement over the no-update baseline (C1 random
cortex) on DEV. No H-LATENT or H-LABEL verdict is permitted. No signoff
was requested; no holdout was accessed.

### C7 adapter discrepancy

The manifest carries `c7_available=true` (hardcoded in calibrate-dev),
but `_try_s3m2a_retrieval_adapter()` returns `None` — no real S3.M2a
adapter exists. The evaluate phase would block on C7 independently of
the articulation gate. This must be resolved before any new claim run.

### R19 vs R20 signoff separation

R19 signed evaluation is BLOCKED at the DEV articulation gate. No
signoff was requested and no holdout was accessed. R20 (meta-trained
cortex) remains separately blocked for lack of explicit human signoff.
R19 signoff and R20 signoff are distinct: neither has been requested or
granted. The R19 articulation gate failure does not change R20's
blocked status.

### Direction reassessment

Do not spend signed-eval or R20 budget. Before any new claim run,
diagnose at DEV level: (1) why the learned coupler does not improve
over the no-update baseline — coupler learning signal, latent interface,
or articulation path; (2) resolve the C7 adapter discrepancy. No
threshold, metric, baseline, episode, scoring, eval manifest, or
research spec was changed.

Source: `2026-07-11_campaign_0d48130.md` § R19 DEV calibration and
`2026-07-12_r19_dev_calibration.json`. No threshold changes or causal
claims beyond measured metrics.

## R20 DEV Implementation/Smoke Adjudication (2026-07-12)

**Full curated evidence log:**
`2026-07-11_campaign_0d48130.md` § R20 DEV implementation/smoke. **Durable
execution/adjudication JSON:**
`2026-07-12_r20_dev_smoke.json`.

Research/20 (`research/20-meta-trained-cortex-frozen-language-organ.md`)
DEV-only implementation/smoke ran from source commit
`e26d8291879d078b701f19802f72041e08cfd6a6` on Kaggle CPU
(Qwen/Qwen2.5-0.5B-Instruct, frozen). **Infrastructure: COMPLETE** (exit 0,
audit_status ok, all invariants verified). **Scientific verdict: none —
meta-test remains BLOCKED.** This is infrastructure/mechanism smoke only.

### Attempt history

| Attempt | Outcome | Root cause |
|---------|---------|------------|
| v1 | INFRASTRUCTURE FAILURE | Offline loader failure — frozen organ could not be loaded under `HF_HUB_OFFLINE=1`. |
| v2 | INFRASTRUCTURE FAILURE | Inference-tensor/autograd failure — tensor dtype or autograd graph mismatch during outer-loop forward/backward. |
| v3 | INFRASTRUCTURE SUCCESS | All invariants verified, exit 0, audit_status ok. |

Attempts v1 and v2 were infrastructure failures with no valid evidence
collected. They are not scientific nulls or refutations. The v3 run was
infrastructure-successful; the meta-test remains BLOCKED.

### v3 smoke results

| Field | Value |
|-------|-------|
| Source commit | `e26d8291879d078b701f19802f72041e08cfd6a6` |
| Source archive SHA-256 | `686c3b6a3de6e093f3646a3cdea6d0097d5de49cc6ef7231e262cf08643d99d5` |
| Kernel | `abdellahkadem/oczy-r20-dev-v3-e26d8291879d` |
| Exit code | 0 |
| Audit status | ok |
| Theta parameter count | 207,364 (829,456 bytes) |
| Fast/slow state dim | 64 × 64 |
| Bank width × feature dim | 3 × 896 |
| Optimizer steps | 1 |
| Best DEV validation score | 0.0 (after one outer step — observed smoke, not a passed threshold) |
| Trace count after deletion | 0 (deletion verified) |
| Online optimizer counts | unchanged |

### Audit invariants

| Invariant | Value |
|-----------|-------|
| Frozen organ hash before | `d8a3a3b262b3397f8948f13da10d3394e1a36b98a2ea374dc8711333d8d2b278` |
| Frozen organ hash after | `d8a3a3b262b3397f8948f13da10d3394e1a36b98a2ea374dc8711333d8d2b278` |
| Frozen organ hash identical | true |
| Checkpoint theta hash | `8d6c41c5dacbf31394e381dbdb5d6b8e496565bf14c2dedbbaa36f4987301d17` |
| Trace count after deletion | 0 |
| Online optimizer counts unchanged | true |

### Causal DEV deltas (observed mechanism smoke, not scientific results)

| Intervention | Delta |
|--------------|-------|
| Trained vs update | 0.0 |
| Untrained | 0.0 |
| Shuffled | 0.0 |
| Zeroed | 0.0 |
| Swapped | 0.0666667 |

These DEV-level causal intervention deltas are from the validate-dev phase.
They are recorded as observed mechanism smoke confirming that the causal
intervention pipeline runs and produces output. They are not scientific
results and cannot be used for an ACCEPT or REFUTE verdict.

### Test suite results (engineering quality checks, not scientific evidence)

| Suite | Passed | Skipped | Note |
|-------|--------|---------|------|
| Focused | 262 | 2 | before extra regression tests |
| Organ | 54 | 2 | after extra regression tests |

### Meta-test block status

The R20 meta-test remains **BLOCKED**. The pre-registered protocol
(§ Instrument freeze and threshold distribution check) requires all of the
following before any meta-test run:

1. a frozen `meta_cortex/v1` instrument (generators, seeds, family split,
   scorers, probe counts);
2. distribution checks (no-update and repeated-run distributions on
   meta-validation);
3. a power analysis freezing sample size from meta-validation effect sizes;
4. a manifest with SHA-256 hashes; and
5. human sign-off on the manifest, margin, and sample size.

None of these exist. The DEV-only smoke (train-dev, validate-dev, audit-dev)
does not constitute a meta-test run and cannot produce a scientific verdict.
No holdout or meta-test data was accessed. No signoff was requested or
granted.

### R19 vs R20 signoff separation

R19 signed evaluation is BLOCKED at the DEV articulation gate. R20
(meta-trained cortex) remains separately blocked for lack of a frozen
instrument, manifest, and human signoff. R19 signoff and R20 signoff are
distinct: neither has been requested or granted. The R20 DEV smoke does not
change R20's blocked status.

### Explicit non-claim

No ACCEPT or REFUTE verdict is claimed for H-META-CORTEX. The meta-test
remains BLOCKED. The DEV smoke is infrastructure/mechanism verification
only. The best DEV validation score (0.0), causal DEV deltas, frozen organ
hash, trace count, and test suite results are recorded as observed
infrastructure/mechanism smoke, not as scientific results. No threshold,
metric, baseline, episode, scoring, eval manifest, or research spec was
changed.

Source: `2026-07-11_campaign_0d48130.md` § R20 DEV implementation/smoke and
`2026-07-12_r20_dev_smoke.json`. No threshold changes or causal
claims beyond measured metrics.

## R20 Corrected DEV Calibration Adjudication (2026-08-05)

**Full curated campaign log:** `2026-07-16_campaign_959e114.md`.
**Durable execution/adjudication JSON:**
`2026-08-05_r20_corrected_dev_calibration.json`.

The corrected v6 DEV execution and calibration evidence is **VALID**: source
provenance, coverage, aggregation, and trace audits close cleanly. That validity
classification applies to evidence integrity only. The scientific DEV decision
is **BLOCKED/no-go**, not REFUTED: every measured effect is zero and the
registered power analysis reports `feasibility_status=blocked`. No formal
ACCEPT or REFUTE verdict is available for H-META-CORTEX.

### Corrected instrument and runtime identity

| Field | SHA-256 / value |
|-------|-----------------|
| Source commit | `a8c98d638209a8425b14a0f853e9fc46ae7da581` |
| Source archive | `40708cb9e195cba45302d64d11a9110cc4f91b74ce0325b5a4856908b1e941ca` |
| Runtime manifest | `a6214355c1c6b9192d435e62f3add6bef5db8c3a6c1cf3a55cb2a9dbfc91182e` |
| Frozen remote organ | `a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea` |
| Definition | `f62f18ef3dd4eb7cf62d82e24b7c0fea5011516dbaf16d3ea50cc7890c9db14f` |
| Calibration view | `639725f44aa7e46f84691987f9dd9454ac70902bf5f20742505a733219166dc2` |
| Scorer | `e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742` |
| DEV distributions artifact | `af644e7573bb073b7b3b75cbe4f40f468a8a39eeb934bb08915dee0e27361771` |
| Power analysis artifact | `f1c02a764d93c831057a35853c5f1a98fccb3f734630f4e3b75f8929e63b5118` |

This corrected source chooses the C6 donor from the full frozen validation
family in canonical order, so shard membership controls only packaging. The
v5 shard-local donor implementation remains scientifically invalid even though
its infrastructure and runtime observations remain valid.

### Coverage and audit closure

| Field | Observed |
|-------|----------|
| Corrected shards | 150 |
| Calibration tasks | 90 total: 30 each for `contextual_remap`, `finite_state`, and `rule_transformation` |
| No-update repeat records | 9,000 |
| Developmental × evaluation seed cells | 2,250 |
| Distinct theta hashes | 5 |
| Missing/duplicate coverage | none |
| Trace audits | passed |
| Holdout/meta-test accessed | false |

The merged artifacts cover every intended corrected DEV cell exactly once.
There were no aggregation failures, and the frozen-organ, theta, trace
deletion, optimizer-step, seed, definition, view, and scorer checks passed.

### All-zero endpoint and condition evidence

For each of the three families, all nine registered endpoints—
`adaptation_delta`, `causal_state_delta`, `composition_delta`,
`feedback_semantics_delta`, `meta_training_delta`, `specificity_delta`,
`state_addressing_delta`, `trace_free_survival`, and `transfer_delta`—have
mean `0`, confidence interval `[0, 0]`, and standard deviation `0`.

All six collected conditions—C1 `update_disabled`, C2 `untrained_rule`, C3
`trained`, C4 `feedback_shuffled`, C5 `state_zeroed`, and C6
`state_swapped`—aggregate zero correct answers over every recorded scoring
denominator. These are valid corrected DEV observations, but zero effects under
a blocked feasibility analysis do not by themselves constitute a registered
hypothesis refutation.

### Power feasibility

The registered equivalence margin is exactly `0`. Every endpoint/family pair
has `status=no_finite_n`; no finite sample size can be estimated from the
all-zero mean and variance. Consequently power feasibility is **blocked**.
The value 30 tasks per family is only the minimum fallback, not a powered
sample-size result and not authorization to proceed to meta-test.

### Exact local reproduction blocker

An exact-runtime local confirmation used the manifested Python/package
versions and matched all ten pinned model artifact hashes, but failed closed
before generation. The expected remote organ hash was
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`;
the locally constructed exact-runtime organ hash was
`2621e258b2fe8d37b2b5743f5e1d3d04037d3d0d409bbad71dd2ba18240498ce`.
Therefore the remote all-zero outputs were **not** exactly reproduced locally.

A separate, explicitly non-comparable diagnostic using organ
`2621e258b2fe8d37b2b5743f5e1d3d04037d3d0d409bbad71dd2ba18240498ce`
scored C1 `0/10` and C3 `0/10`; 9 of 10 paired raw generations were identical,
and no generated output contained a target substring. This supports
investigation of the output path only. It is not a replacement shard, cannot
be merged, and cannot validate or invalidate the remote organ's scientific
result.

### Word-forcing controls

The local controls establish that target tokens can be made visible under
deliberate interventions, not that the learned remote meta-cortex produced
them:

- A hard inference-time token schedule made both `left` and `rise` start all
  four generations (`4/4` each). This standard local FP32 path is not the
  remote INT8 organ and is a positive control only.
- On local QwenFrozenOrgan
  `60de9f75e8ae1d2507429877b4b2da48ec64c3e28eaad03db23cd3de43a1b4da`,
  an optimized soft bank raised first-position `P(left)` from
  `3.507e-10` to `0.936`; `left` began `4/4` generations, with repetition,
  at bank norm `11.265`.
- The analogous soft-bank control raised first-position `P(rise)` from
  `2.084e-9` to `0.926`; `rise` began `1/4` generations at bank norm `6.937`.
- The frozen model hash was unchanged before and after both soft-bank
  optimizations.

These controls are diagnostic only because both local organ identity and/or
inference intervention differ from the frozen remote organ. They show that the
vocabulary/output path can express the requested words under forcing; they do
not demonstrate learned causal control, recover a nonzero endpoint, or change
the power decision.

### Action and explicit non-claim

The action is **no-go**: do not freeze a signed candidate or spend meta-test
budget from this calibration. Continue DEV-only diagnosis of why trained,
untrained, shuffled, zeroed, swapped, and no-update conditions all produce
zero scoring signal, including the organ-construction mismatch and the path
between soft-bank activations and decoded outputs.

No H-META-CORTEX ACCEPT or REFUTE verdict is claimed. Required oracle gates,
a signed candidate, and the meta-test are absent; no meta-test or holdout was
accessed. The corrected v6 closure is VALID calibration evidence supporting a
BLOCKED/no-go decision, not hypothesis acceptance and not a formal
refutation.

Source: `2026-07-16_campaign_959e114.md` corrected-v6 closure and
`2026-08-05_r20_corrected_dev_calibration.json`.

## Notes (conceptual, non-log)

Analysis documents live in `notes/` (created 2026-07-03) — they interpret
logged evidence but are not themselves experiment logs:

- `notes/2026-07-03_steering_vs_posture_postmortem.md` — why the
  steering/posture intuition failed (three broken assumptions: common-mode
  accumulation has magnitude not direction; constant vectors cannot condition;
  mention-space ≠ use-space), synthesizing S1.3, S1.4, S2.1, S2.4. Successor
  mechanism pre-registered in `research/18-consolidation-as-distillation.md`.