# Experiments Logs Ledger — Authoritative Index

**Date:** 2026-09-12 (includes the v5 teaching-format result and public DEV task-coverage audit; earlier adjudications are preserved)
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
| 2026-08-06 | `2026-08-06_stage0_openrouter_teacher_gate.md` | VALID | **R18 Amendment A1 OpenRouter-teacher admission check.** On the unchanged `salt="v2"` stage-0 DEV split (17 probes), the provider-pinned `deepseek/deepseek-v4-flash-0731` teacher scored 9/17 with the correction and 0/17 without it: `teacher_dev_delta=0.5294`, clearing the unchanged `>= 0.2` gate. This is diagnostic gate evidence, not an H-DISTILL verdict; no holdout was accessed. The original local 0.5B-teacher result (`0.1765`, gate failed) remains BLOCKED and intact. | — |
| 2026-08-06 | `2026-08-06_r18_openrouter_5seed_stage0.md` | VALID | **R18 Amendment A1 5-seed execution evidence.** Under the human-authorized one-variable dev-gate teacher substitution, `teacher_dev_delta` was {0.4706, 0.5294, 0.4706, 0.4706, 0.5294} (mean 0.4941), clearing the unchanged `>= 0.2` gate on all five seeds. The unchanged local LoRA distillation path reproduced `distill_delta_holdout` {0.3333, 0.3333, 0.0, 0.3333, 0.3333}: mean 0.2667, CI95 [0.136, 0.397], 4/5 positive with seed 2 null; specificity mean 0.0261, CI95 [-0.008, 0.060]. The amended condition is human-adjudicated **TESTED-PARTIAL with gate resolved**: its evidence is admissible and positive on 4/5 seeds, but the seed-2 null prevents full acceptance. Original BLOCKED records are not superseded for their original condition. Raw execution log: `2026-08-06_r18_openrouter_5seed_stage0.log`. | — |

| 2026-08-06 | `2026-08-06_campaign_r24_tiny_decoder.md` | **INVALIDATED** | Four v3 kernels completed with valid remote execution/provenance, but the scientific measurements are invalid: model initialization was unseeded, variable-length queries were right-padded without length-aware decoding, oracle attention ignored padding, and corpus rows included conflicting/undefined labels. The Deep-FiLM `7/264 vs 0/264` delta is not evidence. | `experiments/r24-tiny-decoder/v2_screen_plan.json` |

## Summary

| Classification | Count |
|----------------|-------|
| VALID | 67 |
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

## R18 Amendment A1 Execution Evidence (2026-08-06)

The original R18 condition is unchanged: its local 0.5B prefix teacher scored
`teacher_dev_delta=0.17647058823529413 < 0.2`, so those runs remain
**BLOCKED at the teacher validity gate / diagnostic only**. Amendment A1 is a
separate, human-authorized condition that substitutes only the DEV-gate
teacher with OpenRouter `deepseek/deepseek-v4-flash-0731`, provider-pinned to
DeepSeek with no fallback. Requests used temperature 0, seed 0, reasoning
disabled, and no `max_tokens` field. The student, local prefix-logit LoRA
distillation target, eval/v2 metric, `salt="v2"` split, and `>= 0.2` threshold
are unchanged.

The standalone full-DEV check scored 9/17 with the correction and 0/17
without it (`teacher_dev_delta=0.5294`), clearing the registered gate. The
subsequent 5-seed stage-0 execution also cleared it on every seed:
{0.4706, 0.5294, 0.4706, 0.4706, 0.5294}, mean 0.4941. Per-seed
`distill_delta_holdout` was {0.3333, 0.3333, 0.0, 0.3333, 0.3333}; mean
0.2667, CI95 [0.136, 0.397], with 4/5 positive and seed 2 null. Mean
specificity delta was 0.0261, CI95 [-0.008, 0.060]. Wall time was about
27.7 minutes and estimated API spend was about $0.0005, well below the approved
$5 ceiling.

Thus the original teacher blocker is resolved **for Amendment A1**, and the
holdout evidence is admissible under that amended condition. It does not erase
or relabel the original BLOCKED condition. **Human adjudication (2026-08-06):
TESTED-PARTIAL with gate resolved.** The amended evidence is admissible and
positive on 4/5 seeds, but the seed-2 null makes the mechanism partial rather
than a full H-DISTILL acceptance. Evidence:
`2026-08-06_stage0_openrouter_teacher_gate.md`,
`2026-08-06_r18_openrouter_5seed_stage0.md`, and raw log
`2026-08-06_r18_openrouter_5seed_stage0.log`.

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

## R24 Phase-A v2 screen and confirmation closure (2026-08-09)

**Full log:** `2026-08-09_campaign_r24_phase_a_v2.md`.  
**Durable JSON:** `2026-08-09_r24_phase_a_v2_confirmation.json`.

The old R24 v1 measurements are invalidated (late construction seeding,
padded-query alignment, missing oracle padding mask, and conflicting/undefined
corpus rows). The clean v2 campaign ran all 22 one-factor cases, four closed
factorial cells, and ten fresh base-vs-finalist confirmation jobs: **36/36
COMPLETE, 0 invalid** across the three scientific batches (plus one bit-exact
canary repeat).

The tuning rule selected deep additive conditioning. Fresh confirmation across
five paired seeds produced base oracle/swapped `120/1115` / `108/1115`
(delta `0.010762`) and finalist oracle/swapped `108/1115` / `86/1115`
(delta `0.019731`). Although the finalist delta was positive in 5/5 seeds and
exceeded the base delta by `0.008969`, it failed the registered mean-delta
threshold `>=0.02` and reduced raw oracle exact accuracy by `0.010762` versus
base. The frozen decision is **DO NOT PROMOTE**.

Phase C was not run: the confirmed-organ gate failed and Phase C is outside the
Phase-A authorization. This is valid Phase-A engineering/tuning/confirmation
evidence, not an H-TOY-EXISTENCE accept/refute result and not a meta-cortex
result. No holdout beyond the registered Phase-A validation catalogs was
accessed.

## R24-v3 fixed-three-event toy existence closure (2026-08-09)

**Classification: VALID; ACCEPT H-TOY-EXISTENCE, narrow registered protocol.**
[Full log](2026-08-09_campaign_r24_v3_toy_existence.md) and
[durable JSON](2026-08-09_r24_v3_toy_existence.json).

FiLM qualified at 100% oracle accuracy; all five DEV and five registered TEST
seeds passed their frozen gates. TEST C3 is 2413/15250 (15.82%), versus
no-update 195/15250 (1.28%), and exceeds each registered causal control on
5/5 seeds. This is a toy causal existence result, not full-rule recovery,
minimality of three independent corrections, or control of Qwen. Additive
failed organ qualification at 28.375% and is articulation-BLOCKED, not a
cortex null. All 12 registered jobs completed; the v2 additive Phase-A
non-promotion remains a separate historical result.

## R20 identity/output-path diagnosis and DEV v3/v4 amendments (2026-09-11)

**Classification: DEV DIAGNOSTIC ONLY; no hypothesis verdict or promotion.**
[Full log](2026-09-11_campaign_r20_dev_output_path.md) and
[raw outputs/provenance](2026-09-11_r20_dev_output_path.json).

The old local organ-hash blocker is isolated to `_name_or_path`. Loading the
verified model at its historical path restores the required hash without
changing weights or weakening the gate. One six-condition cell reproduces all
scores and non-state-hash audit fields, but four state hashes differ; exact
reproduction therefore remains unresolved. The stopped full-shard attempt is
preserved and no replacement shard exists.

The decoder's special-token spelling leak is repaired. Its paired 17-probe
check shows a null score effect: no-context 0/7, teaching 0/7, oracle 0/3 both
before and after. Human-approved response-format Amendment A (v3) raises the
sampled oracle to 1/3. Separately approved complete-oracle Amendment B (v4)
raises it to 2/3. Teaching and no-context remain 0/7 in both versions. All
original task/answer/scorer/threshold boundaries are preserved; v3/v4 are new
DEV-only hash-bound artifacts and do not overwrite v2.

**Next blocker: teaching-context articulation.** R23.5's recovery denominator
is still zero, so serialization optimization has not run. R20's zero-effect
power no-go and unsigned meta-test gate remain in force. No remote submission
or meta-test access occurred in this diagnosis.

## DEV teaching-format v5 and public task-coverage audit (2026-09-12)

**Classification: VALID DEV DIAGNOSTIC; taught-answer presentation unblocked.**
[Full log](2026-09-12_campaign_dev_teaching_format.md),
[raw generations and provenance](2026-09-12_dev_corrective_facts.json), and
[public coverage audit](2026-09-12_dev_teaching_coverage.json).

Under the user's instruction to proceed with the next teaching-format
comparison, v5 adds only a corrections-as-facts condition to the same v4
questions and controls. It scores **4/7**, versus original teaching dialogue
**0/7** and no context **0/7**. Oracle remains **2/3**. All 17 original baseline
prompts, targets, outputs and scores reproduce exactly. All four directly
taught questions pass; three untaught contextual questions remain wrong and
remain included. This is a text/retrieval presentation effect, not learned
state, held-out generalization or serialization recovery.

The read-only public DEV audit finds 44/87 contextual same-rule probes and
30/70 contextual transfer probes query untaught independent mappings; all
59/59 finite-state transfer probes query untaught random transitions. All
35 contextual composition queries use an undefined second input, and all
35 finite-state composition targets require an untaught action. These defects
limit interpretation of previous zero scores as a language-organ capability
ceiling. Historical execution and measured score records are preserved; no
formal R20 accept/refute is issued and no calibration/meta-test is rerun.

Next gate: a frozen serialization pilot with identifiable held-out targets,
and versioned task-semantics repair before full R20 reuse. The original task
generator, v2/v3/v4 instruments and scoring thresholds remain unchanged.

## R23.5 DEV serialization pilot v1 (2026-09-12)

**Classification: VALID DEV PILOT; learned numeric-state persistence observed.**
[Full log](2026-09-12_campaign_r23_5_pilot_v1.md),
[all outputs and trajectories](2026-09-12_r23_5_pilot_v1.json), and
[state/source/provenance archive](artifacts/2026-09-12_r23_5_pilot_v1/README.md).

Under the user's “go ahead,” a new hash-frozen two-pattern pilot ran all
144 updates (three examples, 24 updates, three seeds per pattern). Numeric
state reload in a fresh process scores suffix **4/4, 4/4, 3/4** and dates
**2/4, 2/4, 4/4**. Initial, zeroed and other-pattern-swapped states all score
zero. Raw context scores **3/4** suffix and **0/4** dates and reproduces
exactly after reload. Both model-written text summaries score **0/4**.

Suffix recovery is **133.3%, 133.3%, 100%**, exceeding the unchanged 30%
reference on all seeds. Date recovery is undefined because contextual uplift
is zero; its positive counts do not become a recovery claim. Date seeds 0/1
also miss one teaching example each despite low teacher-forced losses.
Every seed, control and failed output is retained. These repeat the same four
held-out inputs per pattern and are not twelve independent test inputs.

Model/scorer hashes remain unchanged, all 144 state gradients are finite and
nonzero, and no organ parameter receives gradients. Filesystem and loading
boundaries separate teaching and probes; no optimizer or original teaching
text enters numeric reload. The learned payload is 28,672 bytes per seed,
versus 139/172 bytes raw context: **no compression claim**.

Next gate is fresh-pattern confirmation at the same settings. No full R23.5
accept/refute, hidden-layer sweep, meta-trained cortex writer, R20 repair,
remote execution or meta-test is established or authorized by this result.
The local pilot is complete; legacy autonomous research services remain idle.

## Curriculum and evaluation audit (2026-09-12)

**Classification: READ-ONLY INSTRUMENT AUDIT plus local guard repair; no model result.**
[Full audit](2026-09-12_curriculum_eval_audit.md),
[public data/witnesses](2026-09-12_curriculum_eval_audit.json),
[versioned repair plan and candidate](../experiments/eval-audit-repair-v1/REPAIR_PLAN.md).

All 120 eval-v2 probes accept an empty answer; the existing validators and
28 focused tests still pass. Semantic fallback accepts wrong senses. R20's
previous task-support defects are confirmed, plus five transformation probes
with multiple teaching-consistent targets and 23 specificity questions whose
answers contradict teaching (9 transformation, 14 FSM). Tool scoring accepts
45/45 extra-call foils and 21/21 wrong-parameter foils. These are deterministic
instrument counterexamples, not newly measured model performance.

R24's public 192-rule algebra passes three-event identifiability, but A+C
alone also identify every rule; the registered fixed-three-event result
remains narrow and is not invalidated. The existing R23.5 numeric-persistence
pilot remains unchanged, with no compression or broad confirmation claim.

The eval guard was repaired to cover active instrument paths and working-tree
changes. Thirteen new cases fail against the old guard; all 21 guard tests
pass after repair. The unapplied eval-v2.3 candidate changes blank acceptance
120/120 to 0/120, preserves 120/120 expected-answer passes and all 960 checked
nonempty comparisons, and binds runtime scoring/protocol sources. It does not
repair nonempty sense-matching errors. Human sign-off is pending on the
versioned repair plan. No original scoring, targets, thresholds or manifests
were changed; no model run, remote job or sealed/meta-test access occurred.

The initial audit report missed specificity conflicts because of the shared
v3/v4 system-format message; its `_initial.json` artifact is preserved and
superseded by the corrected user-question comparison. Historical scores are
not retrospectively recomputed or declared inflated without raw outputs.

## Six-capability DEV validation v1 (2026-09-13)

**Classification: VALID BOUNDED DEV RESULT; scoped retention and reliable coupled
action fail, with prerequisite/interface limits explicitly recorded.**
[Report](2026-09-13_capability_validation_v1.md),
[all scores, outputs and trajectories](2026-09-13_capability_validation_v1.json),
[source/state/provenance archive](artifacts/2026-09-13_capability_validation_v1/README.md).

The user's explicit six-capability request authorized a new separately frozen
DEV instrument. One shared bank learns amber, adds cobalt, then corrects amber:
216 fixed updates across three seeds, no old-example replay, 27/27 teaching fits.
All four offline local phases finish, model/scorer identities remain unchanged,
and numeric checkpoints are evaluated in a fresh process without training files.

Unassigned-context behavior leaks. Adding cobalt retains 0/1, 0/1 and 0/2
previously correct amber responses. Correcting amber gives 4/4, 1/4 and 0/4 new
amber answers while retaining 0/3, 0/2 and 0/2 previously correct cobalt answers.
Two-rule composition is 0/24, but no trial has both corresponding individual
rules correct; a distinct composition failure is therefore not isolated.
Complete-oracle scores are only 6/16, 2/16 and 3/16 under this interface.

Neural files cost 28,800 bytes versus 143–295 bytes of active examples: no
learned compression. Zlib costs 91–129 bytes, with exact byte round trips and
all replay outputs matching raw retrieval, whose scores remain 0/16, 0/16,
3/16. This is a retrieval/text-codec result, not neural compression.

Real filesystem action execution with exact arguments/outcomes is no-bank 3/4,
learned banks 0/4, 0/4 and 1/4, and latest-example retrieval 1/4. Every strict
end-to-end trial fails. No-bank final answers echo a prompt placeholder despite
three correct executions; this is separated from actual action failure.
Missing initial/zeroed action controls prevent isolating bank insertion from
learned-value effects on tool behavior. All actions use disposable directories.

Final audit recomputes 576 probe scores and 20 action verdicts, checks exact
coverage and all numeric/source/input hashes, and verifies 216 finite nonzero
state gradients. Four phases take 1,875.14 seconds. Forty-seven focused tests
pass; targeted retention checks pass again after the final audit update. The
eval guard now protects this new instrument as well. Prior eval/pilot files
remain unchanged; no remote launch, meta-test, loop restart, production action
or full-study acceptance/refutation is claimed.

Next blocker: scoped acquisition and preservation of unrelated behavior,
with separate versioned oracle/tool-prompt repairs and action controls.

## Language-interface DEV v3 (2026-09-13)

**Classification: VALID BOUNDED DEV INTERFACE IMPROVEMENT; oracle-selected
primitive execution improves, reliable context selection remains unproved.**
[Report](2026-09-13_language_interface_dev_v3.md),
[all prompts/results](2026-09-13_language_interface_dev_v3.json),
[archive](artifacts/2026-09-13_language_interface_dev_v3/README.md).

Seven frozen prompt conditions run on 16 calibration and 16 new confirmation
cases each, using the same scalar decoder and no learned bank. Minimal direct
operations score 15/16 in each set versus previous oracle-selected instructions
9/16 and 8/16. Both residual failures insert a dot before mip. This condition
supplies the operation externally and cannot establish learned client selection.
Conversation-form retrieval improves over plain-text retrieval (10/16 versus
2/16 calibration; 6/16 versus 0/16 confirmation), but remains unreliable.
Moving complete rules to the system message gives only 4/16 in both sets.

All 224 scores, prompts and exact coverage are independently audited. All 48
paired scalar baselines reproduce DEV-v2 outputs, despite two versus four CPU
threads. Frozen model/runtime/source identities pass. Zero optimizer updates,
595.98 seconds, no meta-test or changes to old scores.

Live scalar/native/batch parity is only 8/16. Native/batch APIs inherit a 1.1
repetition penalty while scalar uses raw argmax; source inspection also finds
fixed continuation positions in the original batch path under Transformers
5.0.0. These findings trigger a separate versioned decoder ablation. They do
not explain away this scalar-only interface table or the concurrent scalar-only
learning run. No historical R20 result is retrospectively recomputed here.

## Decoder parity DEV v1 (2026-09-13)

**Classification: VALID BOUNDED EXECUTION REPAIR; versioned candidate qualifies,
with no historical score rewrite or learned-capability promotion.**
[Report](2026-09-13_decoder_parity_dev_v1.md),
[full ablation](2026-09-13_decoder_parity_dev_v1.json).

The original batch path agrees with scalar strings on 21/32 cases across
zero-width and learned banks. Neutralizing its inherited repetition penalty
alone leaves 21/32. Letting continuation positions advance restores 32/32 at
either penalty, and the explicit neutral candidate additionally matches all
32 responses in four-item padded batches. Neutral native input-ID generation
matches scalar on all 16 no-bank cases. The registered qualification passes.

Transformers 5.0.0 does not extend the original caller-supplied position-ID
tensor during generation. The new generation_v2 helper lets positions follow
the extending attention mask and pins neutral generation settings. It is
versioned separately; old organ source and experiment manifests stay intact.
The old runtime declaration of repetition_penalty=1.0 was not enforced by
native/batch calls inheriting 1.1 from the model artifact.

All 176 task scores and 208 parity verdicts are independently audited, with
source/model/runtime/state identities unchanged. Zero optimizer updates,
293.54 seconds, no meta-test. The associated regression suite passes 73 tests.
Scalar/candidate task accuracy remains only 9/16 no-bank and 8/16 learned:
execution consistency is repaired, but task errors remain. A new versioned
adoption is needed before future batched R20 calibration; its old scores and
the current scalar-only learning results are not rewritten.

## Context and preservation DEV v2 (2026-09-13)

**Classification: VALID BOUNDED ACQUISITION FAILURE; preservation comparison
blocked at its frozen prerequisite, not tested or refuted.**
[Report](2026-09-13_context_preservation_dev_v2.md),
[scores/trajectories](2026-09-13_context_preservation_dev_v2.json),
[archive](artifacts/2026-09-13_context_preservation_dev_v2/README.md).

The curriculum audit identifies word/client correlation in the earlier teaching
set. A new crossed curriculum teaches the same three inputs under amber, cobalt
and neutral silver. Three seeds each receive 48 fixed nine-example Adam updates
to one shared 8 x 896 bank. All 144 updates and 27/27 teaching fits pass, with
the frozen model/scorer unchanged. This is joint DEV gradient training, not
sequential accumulation or R20's learned no-backprop writer.

Fresh-process learned-bank scores are 8/16, 11/16 and 4/16. New affix applications
are 6/24; neutral copying is 17/24 versus 24/24 initially. Untaught quartz is
7/12 versus 12/12 initially. Reliable scoped acquisition fails. The controller
records admission=false and blocks both correction phases: zero correction
updates, no unprotected/proximal comparison result. No budget extension or
checkpoint selection follows the failed gate.

The fixed full-rule interface ladder remains weak; subsequent separately
versioned interface and decoder studies are logged above. The qualified batch
candidate reproduces scalar, including its task errors; it cannot explain away
this scalar-only acquisition failure. All 16 seed-0 scalar outputs reproduce
across the decoder diagnostic and final fresh-process evaluation.

All 368 probe scores and 27 teaching scores are independently recomputed;
source/input/numeric hashes, prompt and update coverage, three process boundaries
and finite nonzero gradients pass. The original oracle prompt reproduces all
12 corresponding DEV-v1 outputs. Phase durations total 2,545.74 seconds.
The combined regression suite passes 73 checks. No meta-test, old-loop restart,
remote launch, commit, push or production operation occurs.

Next blocker: transfer a learned rule to new inputs in its context while
retaining neutral behavior. A fixed-compute teaching-diversity comparison is
proposed; preservation remains conditional on reliable prior acquisition.

## Scoped diversity DEV v3 (completed 2026-09-14)

**Classification: BOUNDED PARTIAL GAIN WITH EXECUTION PROVENANCE LIMIT;
acquisition fails and correction remains unrun.**
[Report](2026-09-13_scoped_diversity_dev_v3.md),
[audited results](2026-09-13_scoped_diversity_dev_v3.json),
[archive](artifacts/2026-09-13_scoped_diversity_dev_v3/README.md).

Only teaching-word diversity changes. Three crossed nine-example groups replace
one group, with the same 144 updates and 1,296 example losses across three seeds.
This matches update/example counts, not token counts or FLOPs. All seeds start
from the prior initial hashes and match their first losses. Final controls are
reused without selection. Teaching fits are 67/81. On identical fresh words,
affix scores improve from matched control 5/24 to diversity 12/24. Neutral copying
is 19/24 for both versus initial 24/24; seed 2 loses one total correct response.
The frozen acquisition gate fails. Zero correction updates follow.

The external direct-operation reference reaches 16/16 on these fresh words;
the complete table gets 5/16. The operation is externally selected, so this is
language articulation evidence, not learned scope selection. Raw/zlib retrieval
outputs agree independently. Numeric state is 28,800 bytes (28,672 payload)
versus 1,405 raw example bytes or 214 zlib bytes: no learned compression.
Logit-bias and rerank are explicitly not implemented/not run.

All 608 evaluation rows, 81 teaching fits and 144 gradient updates are audited.
All 112 repeated prior calibration control outputs match. Training records
1,943.92 seconds. After completed evaluation, the controller fails with
FileExistsError while rewriting its write-once execution ledger. Original
source, ledger, scores and states remain intact. A separate source-hashed
recovery checks coverage/scores/identities and invokes the original admission
function with zero model calls. Evaluation child exit status, duration and
pre-run firewall output were lost and remain unknown. This is not a clean
controller completion or complete provenance claim. The archive's 72 checksum
entries verify. No meta-test or remote execution occurs.

## Capability regression DEV v1 (completed 2026-09-14)

**Classification: EXACT HISTORICAL REPRODUCTION; NEW CANDIDATE DOES NOT PASS
CAPABILITY OR ACTION REGRESSION.**
[Report](2026-09-13_capability_regression_dev_v1.md),
[audited results](2026-09-13_capability_regression_dev_v1.json),
[archive](artifacts/2026-09-13_capability_regression_dev_v1/README.md).

All 804 historical comparisons reproduce complete rows exactly, including old
failures: six-capability probes/actions, numeric/text persistence, direct-operation
language probes and scalar/batch decoder parity. Historical optimizers are not
rerun. Another 160 probes compare all three new states with initial/control
banks and zeroed state under the unchanged v1 wording. Diversity composition is
0/12 with failed component prerequisites. Forty-eight real temporary-file action
trials include the previously missing initial/zeroed controls. Trained control
and diversity tool core are each 0/12; initial is 5/12 and zeroed prefix 0/4.
Prefix insertion itself interferes with actions, with further degradation after
training relative to initial banks. Strict success is 0/48; the frozen final-answer
placeholder remains intact and is reported separately from actual tools/files.

All 1,012 scores and coverage are independently checked. The offline replay
exits 0 in 2,161.67 seconds with zero optimizer steps and unchanged model/scorer
identities. No meta-test is accessed. The archive's 63 checksum entries verify.
The combined focused suite passes 122 tests. The local workbench exposes all
seeds, saved/live comparisons, failures, regression evidence and real transcripts;
its live browser and CLI seven-condition outputs agree without changing research
state. Packaging and reproduction instructions live in [the workbench](../workbench/README.md).

The final [3,446,209-byte bundle](../exports/oczy-workbench-dev-v1.tar.gz) was
unpacked outside the repository. Standard-library Python verifies all 607 sealed
files, source reports and released states; both embedded research archives pass
their checksums. An actual offline model trial from the unpacked source matches
the complete seven-condition repository CLI object exactly in 13.60 seconds,
with zero optimizer steps. Weights/runtime remain external. Archive SHA-256:
`dbf40741db3a27d73f8b1f82fd4b365fb955e8d732883d71d3ef63c4c24436e2`.
[Verification record](../exports/oczy-workbench-dev-v1.verification.json).
The finish review's sole mobile-reading fix is scored resolved. Its scope and
capture/detector limitations are recorded in [workbench QA](../workbench/verification/QA.md).

## Eval-v2.3 nonempty-score repair applied (2026-10-04)

**Classification: VALID INSTRUMENT REPAIR (eval data version v2.2 → v2.3),
human-authorized; no model result and no score change.**
[Full record](2026-10-04_eval_v2_3_applied.md),
[validation counts](2026-10-04_eval_v2_3_applied.json),
[repair plan](../experiments/eval-audit-repair-v1/REPAIR_PLAN.md).

Kanban task `t_b2e72297` (created by the user) authorizes and applies A1 of the
September 12 repair plan: the prepared `eval_v2_3.patch` is now applied, so the
eval data version is **v2.3** — blank and whitespace-only answers are rejected
on all 120 shipped probes (previously 120/120 passed), expected answers still
pass 120/120, and all 960 nonempty comparisons against the pre-v2.3 scorer are
identical, so historical scores and failures reproduce unchanged. The manifest
was recomputed with `scripts/bump_eval_version.py` (11 files: 7 data assets plus
the four bound runtime sources) and verifies without `EVAL_CHANGE_APPROVED`;
sandboxed runtime-source drift is rejected. The guard refused the change without
approval and proceeded with `EVAL_CHANGE_APPROVED=1 --allow` as AGENTS.md
prescribes. Tests: 21/21 eval-guard, 41/41 organism-curriculum including three
new regression tests (blank rejection per match mode/semantic setting, blank
rejection plus expected acceptance over all 120 probes, per-file runtime-source
tamper detection); the manifest-integrity version pins moved v2.2 → v2.3 with
the bump. No model ran, no historical score was recomputed, no optimizer was
rerun, and no meta-test or sealed asset was accessed.

This supersedes the "unapplied candidate / sign-off pending" status of the
2026-09-12 curriculum audit entry for A1 only. The audit itself remains open:
nonempty wrong-sense and substring false positives (A2 sense contract), R20
task support (B) and tool-output scoring (C) are still unfixed, and no
capability claim may use eval-v2 until those separately versioned repairs land.

## R20 sign-off recorded and meta_cortex/v3 frozen (2026-10-05)

**Classification: VALID INSTRUMENT CONSTRUCTION AND ADOPTION
(human-authorized; no scientific result, no threshold, no meta-test access).**

Kanban task `t_5a7b48de`, on the user authorization relayed by the manager
2026-10-04. Two dated records:

| Record | Class | Result |
|---|---|---|
| [`2026-10-05_r20_signoff_s3_comparability.md`](2026-10-05_r20_signoff_s3_comparability.md) | VALID COMPARABILITY AUDIT | S3's condition discharged: the `skip_special_tokens=True` decode change alters the answer **text** of 11 of 68 recorded probe rows and moves **0 recorded scores**. Quarantined to v2-lineage runs. |
| [`2026-10-05_r20_g1_instrument_freeze.md`](2026-10-05_r20_g1_instrument_freeze.md) | VALID INSTRUMENT FREEZE (gate G1) | **G1 PASSED.** `meta_cortex/v3` frozen, `definition_sha256` `ab99c173…`. |
| [`2026-10-05_r20_g2_oracle_screen.md`](2026-10-05_r20_g2_oracle_screen.md) | **DEV GATE — FAILED (articulation block)** | **G2 FAILED.** Oracle context 0/15, retrieval bar 4/32, no-context floor 0/32. Sequence stops. |

**Sign-off recorded verbatim** in
[`SIGNOFF_CHAIN.md`](../experiments/r20-task-support-repair-v1/SIGNOFF_CHAIN.md)
§S: `S1 approve (user via manager, 2026-10-04)`, `S2 approve (user via
manager, 2026-10-04)`, `S3 adopt-with-version-bump (user via manager,
2026-10-04)` — **instrument construction and adoption only**. No scientific
verdict, no threshold selection from results, and no meta-test authorization is
implied by any of them.

**G1 freeze.** `meta_cortex/v3` is the `oczy/meta-cortex/taskgen/v2-dev`
task-support-repaired lineage frozen as a DEV instrument, differing from v2 only
in task semantics and the decoder (S3). The prompt, scorer and endpoint
registries are inherited **byte-for-byte** from v2 and the freeze fails closed if
any of them drifts. 90 train / 15 tuning / 90 calibration tasks; leakage/support
audit passed with zero cross-domain fingerprint overlap; independent checker
found all 22 defect classes at 0 with 758/758 derivation-backed support
certificates verified (175 pre-learning baselines exempt by design, 0 failed) and
1446/1446 mutations detected. The frozen organ identity
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea` was reproduced
locally under the recorded historical runtime, not assumed. 18 guard tests prove
a single-byte change anywhere in the frozen tree rejects loading; 228 regression
tests pass; the frozen tree and its lineage are now eval-guard protected.

**What G1 does NOT establish.** No capability claim (G2 not run), no
learnability claim (G3 not run), no calibration/margin/power (G4 not run), and
no meta-test access — v3 carries no sealed payload at all and its verifier
rejects one. **v1 and v3 scores are not comparable** by construction: the task
semantics changed by design.

**G2 subsequently FAILED — articulation block, sequence stopped.** The oracle
capability screen on the frozen v3 instrument, 79 rows, organ hash matching the
frozen binding before and after, zero optimizer steps: **oracle context 0/15**,
retrieval comparator (teaching transcript) 4/32, no-context floor 0/32. The
frozen organ produced no exact answer under oracle control on any of the 15 v3
oracle probes. Per the sign-off chain this is a **mouth–cortex articulation
block, not a cortex refutation**: no cortex state existed and the learner was
never involved. The v3 tasks themselves are sound by construction (22/22 defect
classes 0, 758/758 certificates verified, leakage audit passed). G3, G4 and G5
were not reached; the meta-test stays blocked. The likely mechanism — v3 inherits
v2's frozen prompt registry, which lacks the bare-answer instruction and the
complete oracle descriptions that the 2026-09-11 v3/v4 amendments introduced and
that lifted the *v2* oracle sample from 0/3 to 2/3 — is recorded as a diagnosis,
not an established cause, and acting on it needs a new instrument version and a
new human sign-off.

**Limitations carried forward, unresolved by this entry:** R20 local
reproduction still has four differing nonzero state hashes (no replacement
shard); eval-v2 A2 and C repairs remain unsigned and unapplied; 758 flip-target
mutation checks remain **vacuous by construction** (split from the 688
discriminating delete-a-fact checks and labelled as such, not folded into a
933/933 headline); the 175 baseline certificates carry no derivation claim.

## R20 successor instrument `meta_cortex/v4-r20` frozen; G2 PASSED (2026-10-05)

**Classification: VALID INSTRUMENT CONSTRUCTION AND ADOPTION
(human-authorized; no scientific result, no threshold, no meta-test access).**

Kanban task `t_37e96ee1`, on the user decision relayed by the manager 2026-10-05
("Option A funded, successor named `meta_cortex/v4-r20`, max_new_tokens move
approved, evidence push to remote approved"), recorded in
[`SIGNOFF_CHAIN.md`](../experiments/r20-task-support-repair-v1/SIGNOFF_CHAIN.md)
§S4. Two dated records:

| Record | Class | Result |
|---|---|---|
| [`2026-10-05_r20_g1_v4_r20_instrument_freeze.md`](2026-10-05_r20_g1_v4_r20_instrument_freeze.md) | VALID INSTRUMENT FREEZE (G1-equivalent) | **PASSED.** `meta_cortex/v4-r20` frozen, `definition_sha256` `fd2d8db9…`; amended prompt registry `d8bd0b26…`; `max_new_tokens` 128. |
| [`2026-10-05_r20_g2_v4_r20_oracle_screen.md`](2026-10-05_r20_g2_v4_r20_oracle_screen.md) | **DEV GATE — PASSED** | **G2 PASSED.** Oracle context **7/15** (v3: 0/15); retrieval bar 9/32; no-context floor 2/32. Per family: finite_state 5/5, contextual_remap 2/5, rule_transformation 0/5. |

**Naming.** The user asked for "v4"; plain `meta_cortex/v4` is taken by the
approved `experiments/r23.5-serialization-dev/instruments/v4` (manifest
`d58adb749f4112f2…`) and `meta_cortex/v3` is doubly claimed, so the successor id
is the scope-qualified **`meta_cortex/v4-r20`**, with both collisions recorded in
the successor's `DEFINITION.json` under `naming`.

**The definition diff vs v3 is exactly three changes**, all explicitly
authorized: (1) Amendment A — the identical bare-answer system instruction on
every public DEV probe (933/933); (2) Amendment B — complete oracle rule
descriptions on the 35 `rule_transformation` `oracle_context` headers; (3)
`max_new_tokens` 32 → 128, a **signed field relative to v3**, justified by the
observed truncation and recorded under `max_new_tokens_change`. The task
*content* is byte-identical to the pinned v3 public DEV view (catalog digest and
support-bundle digest unchanged); the materializer's per-probe diff proves
`non_probe_task_fields_changed = 0` and `probe_payloads_changed_outside_
amendment_b = 0`, and all 35 amended oracle targets were independently
re-derived from the amended description text. The scorer and endpoint registries
are inherited byte-for-byte and the freeze fails closed on drift.

**Fail-closed lineage.** A new materializer (`scripts/materialize_r20_v4_r20.py`)
pins the v3 public view by `dev_view_sha256`, `definition_sha256`,
`catalog_sha256` and the three public task-file hashes, and refuses anything else
with exit **3** (verified on a v1 instrument, a byte-tampered v3 copy, and a
copy with a changed `instrument_id`). The original v2-lineage materializer still
refuses the v3 base with `Not the approved v2 public DEV instrument` — the
refusal behaviour is regression-tested in both directions.

**G2 PASSED but partial, and this is the honesty point.** The registered
criterion was `oracle_context correct > 0`; it is met at 7/15. The pass is not
clean: **the transformation family is still 0/5 under oracle control**, and 8/15
oracle probes still fail as a mixture of operand-selection errors, mapping-member
selection and prose-wrapping under the exact scorer. The gate did not register a
per-family criterion. **v3 and v4-r20 scores are NOT comparable as a causal
improvement** — the prompt registry, the `max_new_tokens` field and the
versioned seed table all changed at once, so the v2→v3 rule applies and no
single-variable causal claim is made.

**Sequence stopped before G3 by this session, deliberately.** The card authorizes
proceeding to G3 "only if G2 passes cleanly" and to stop before G4 "if anything
is ambiguous". The pass is real but the transformation family's 0/5 is an
ambiguity a human should rule on; G4/G5 are untouched and the meta-test remains
blocked.

**Limitations carried forward, unresolved by this entry:** R20 local
reproduction still has four differing nonzero state hashes (no replacement
shard); eval-v2 A2 and C repairs remain unsigned and unapplied; 758 flip-target
mutation checks remain **vacuous by construction** (split from the 688
discriminating delete-a-fact checks); the 175 baseline certificates carry no
derivation claim; the transformation-family oracle is 0/5; and the successor's
G2 was a local run on an uncommitted working tree, not a clean-source remote
campaign.
