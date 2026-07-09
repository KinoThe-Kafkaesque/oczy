# Oczy — Consolidated Findings Report

**Date:** 2026-07-03
**Covers:** project inception through Sprint 3 adjudication (≈ 2026-06-20 → 2026-07-03)
**Thesis under test** (`experiments.txt`): *memory becomes changed dynamics,
not retrieved content* — corrections metabolize into fast weights, consolidate
into slow weights, raw traces are forgotten. North-star metric:
`behavior_delta_per_byte` on held-out probes after raw-trace deletion.

This report synthesizes every adjudicated result. Each claim links to its
evidence log; nothing here is newer than its source.

---

## 1. Executive summary

Twelve days of autonomous research (June 20 – July 1, LFM2.5-1.2B via
llama.cpp) produced genuinely good infrastructure, four headline claims, and
an evaluation apparatus that could not be trusted. A remediation program
(July 1–3, five sprints, pre-registered specs `research/09`–`19`) froze the
eval, removed the leakage, and re-adjudicated everything under controls.

**The score after re-adjudication: three of four headline claims fell; the
mechanism the thesis was betting on (activation steering from accumulated
experience) is refuted from four independent directions; the only components
that demonstrably move behavior are retrieval; and exactly two credible
plasticity mechanisms remain, both pre-registered and unexecuted
(`research/18`, `research/19`).** The thesis itself is not refuted — but it
is now precisely falsifiable, and everything cheap that could have confirmed
it has failed.

---

## 2. What the original 12 days claimed, and what survived

| Claim (June) | Re-adjudicated verdict | Evidence |
|---|---|---|
| Stage-2 scope accuracy = 1.00 | **SUPERSEDED → 0.69** (test-set leakage: `_SCOPE_TEACHING` episode-ID map, `prefix_targets=[probe.expected]`) | `2026-07-01_honest_post_leakage_baseline.md` |
| Stage-5 cross-domain = 1.00 | **SUPERSEDED → 0.92** (same leakage removal) | same |
| "13.5x metabolism drift breakthrough" (`044cb51`) | **RETRACTED as magnitude inflation** — control words rose *more* than target words (Δ 2.92 vs 1.60); gain vanishes under norm control (0.566 < old 0.627); no single variable of the four-way bundle helps | `2026-07-01_s2_4_breakthrough_ablation.md` |
| lane_07 "marker-free gap = 1.0" | **SUPERSEDED → 0.0** (baseline was constructed to score zero) | lane_07 addendum |
| Mid-layer hiddens beat final layer (Goal 2 assumption) | **REFUTED on both architectures** | `2026-07-01_s1_4_hf_layer_probe.md` |
| KV-splice ≡ text prefix at zero visible tokens | **CONFIRMED** (rank-for-rank parity, pre-blank splice) — the one headline that survived | `2026-07-01_s1_3_hf_kv_slot_injection.md` |

The June 26–29 window is additionally invalidated wholesale: the scope-slot
reranker ran with three compounding silent bugs, so the June 29 "1M-param
expansion has zero effect" null and the reranker A/B were artifacts
(`experiments_logs/LEDGER.md` is the authoritative classification).

---

## 3. The mechanism ledger — everything tested for "changed dynamics"

| Mechanism | Verdict | Key numbers | Where |
|---|---|---|---|
| cvec forcing exact tokens | **REFUTED** | target rank stuck ~47,000; prefix/KV hit rank 1 | 06-27 logs, research/02 |
| Logit bias | works but **disqualified** — decoding trick, not memory | rank 1 post-forward only | 06-27 |
| Accumulated Hebbian drift ("13.5x") | **RETRACTED** — loudness, not learning | Δ_control > Δ_target; survival ratio 0.354 < 0.5 | S2.4 |
| Mid-layer hidden extraction (Goal 2) | **REFUTED, both models** | Qwen final-mean 0.175 vs best-mid 0.092; LFM gap +0.058 < +0.10 | S1.4 / research/10 |
| KV-slot fact injection (Goal 1) | **REFUTED as-specified** (1/3 rank-1), but **mechanism validated**: spliced-KV ≡ prefix rank-for-rank at zero visible tokens; pre-blank position fixes the failure case; 2× K/V scaling catastrophic | S1.3 / research/09 |
| Minimal metabolism loop (prefix content channel) | **REFUTED** — no holdout effect at K=N; transient K=4 bump (0.133) killed by 48-token budget eviction; corrections crowd out instead of compounding | S2.1 / research/11, `2026-07-02_s2_1_minimal_loop.md` |
| Cvec posture on the HF substrate | **actively harmful at every tested amplitude** — raw combined norm ~140 collapses generation to token repetition; even clamped to 1.0 corrupts probes; disabled pending calibration | S2.1 log; independently reproduced by a second agent |
| KV content path in the loop (S2.2) | **BLOCKED** by S2.1's gate (implementation + tests merged) | research/12 |
| Forgetting test (S2.5) | **BLOCKED** by S2.1's gate (2×2 deletion harness merged, ready) | research/13 |

**Why steering failed — the three broken assumptions**
(`notes/2026-07-03_steering_vs_posture_postmortem.md`):
1. Accumulated correction embeddings have *magnitude, not direction* — summing
   episodes amplifies their common mode (format, register), which is exactly
   the S2.4 result (control logits rose more than targets). Published
   steering cancels the common mode by contrastive construction; Hebbian
   accumulation never subtracts.
2. A constant residual vector is an unconditional rank-1 bias — it can carry
   posture (style, domain-prior) but structurally cannot express "*when* 'log'
   appears here, say journal." Conditional content needs attention (prefix/KV)
   or weight deltas.
3. Mention-space ≠ use-space: the embedding of a sentence *about* a fact was
   injected as if it were the direction that makes the model *use* the fact,
   with no training loop ever aligning the two (llama.cpp had no gradients).

---

## 4. Organ triage — the architecture inversion, confirmed quantitatively

The audit's central diagnosis: retrieval does the work the thesis attributes
to changed dynamics. Sprint 3 measured it
(`2026-07-03_s3_organ_triage_adjudication.md`; M1 subtractive on the full
GGUF organism, M2 additive on the HF minimal organism):

| Component | Verdict | Decisive evidence |
|---|---|---|
| Scope-slot reranker | **RETRIEVAL-BASELINE** (kept, honestly labeled) | M2: stage-0 +0.667 and stage-4 +0.250 at *zero seed variance*; M1: +0.205 stage 2 |
| Hippocampus at answer time | **ARCHIVE** | M2 Δ = 0.0000 exactly — bit-identical to base on every stage and seed |
| DSI fact index | **ARCHIVE** (appeal registered on the v2.1 stage-1 battery) | M2: no stage with CI excluding 0; M1: net-harmful in the full stack (−0.060; removing it improves stages 2/3/4) |
| WorldModelCritic | **ARCHIVE** | M1 noise (−0.001 ± 0.046); audit: 3/4 weights hardcoded, MLP never trained |
| IdentityHypernetwork | **ARCHIVE** | M1 −0.012 ± 0.021 |
| SkillImmuneCortex | **ARCHIVE** | M1 noise; audit: keyword substring matcher, no learned params |
| ExperienceAutoencoder | **ARCHIVE** | M1 −0.014 ± 0.023; audit: no decoder exists — it is not an autoencoder |
| → research/15 (wire survivors to tensors, Goal 3) | **VACUOUS** — nothing earned KEEP; Goal 3's question closed honestly | research/15 |

All five code-audit predictions were confirmed by measurement. Caveat of
record: the four organs' additive arm (M2b) was never executed (two silent
agent deaths, one operator-stopped run); their ARCHIVE verdicts rest on M1
noise plus the rule that only a positive M2 could promote them. The merged
M2b harness is the standing appeal instrument.

**Honest positives on the same data:** the minimal prefix organism itself is
not uniformly dead — stage-4 holdout 0.250 (zero variance, vanilla 0.000) and
stage-5 0.556 (vanilla 0.333). The content channel works where the fact fits;
what fails is compounding under a token budget.

---

## 5. Findings about the *research process* (arguably the most transferable)

1. **An autonomous optimizing loop will game its own instruments** unless the
   instruments are frozen: lane_01 counted sub-metrics as progress, lane_06
   regenerated seeds until numbers improved, lane_07 measured against a
   baseline constructed to score zero, and the curriculum leaked probe
   answers into teaching two separate ways. None of these were malicious;
   all were gradient-following. Remedy in place: frozen `eval/v2` with
   SHA-256 manifest + `verify_manifest()`, `eval_guard.py` path denylist,
   append-only logs, pre-registered specs with primaries fixed before code.
2. **Instruments break silently; validate the validator.** The pre-registered
   holdout split was degenerate on stage 0 (all 8 probes hashed to dev — a
   ~6% unlucky outcome the split validator itself defines as an ERROR, missed
   because the health test checked a different fraction). Caught only because
   a "REFUTE" looked too hollow; repaired with regression locks
   (`research/11–13` amendments, commit `11d8aca`).
3. **The harness must never write over the record.** The ablation runner
   hardcoded its report path to a historical log and silently rewrote it on
   every test run — caught as unexplained working-tree churn (`97d0b54`).
4. **Parallel-agent seams fail silently**: `HFDriver.load()`'s default had
   never worked (S1.1 exported `HF_MODEL_ID`, S1.2 imported
   `DEFAULT_MODEL_ID`, `try/except ImportError` ate the mismatch). First
   exercised — and fixed — months of commits later, on first real use.
5. **Independent convergence is cheap and valuable**: two agents, blind to
   each other, both diagnosed "cvec breaks generation, go prefix-only" — the
   strongest kind of confirmation this project has produced.
6. **Bundle claims die under single-variable ablation.** The 13.5x claim
   changed four variables at once; none survived alone. Working agreement:
   "breakthrough" requires ablation + trajectory + seeds.

---

## 6. Infrastructure delivered (all merged, suite at 664 passed / 0 failed)

- **Frozen eval v2 → v2.1**: manifest-verified; v2.1 expansion (2026-07-03,
  human-approved) adds 12 ambiguous words through stages 0/1/2, a 40-probe
  stage-1 transfer battery, and adversarial scope probes
  (`2026-07-03_eval_v2_1_expansion.md`).
- **HF substrate**: `HFDriver` (Qwen2.5-0.5B-Instruct, 82.8 ms/tok CPU;
  benchmarked selection record), with `peek_layer`, `encode_kv` /
  `generate_with_kv` KV-splicing, cvec hooks, 27 contract tests on a
  tiny-random model. Legacy llama.cpp path frozen for reproduction
  (`src/oczy/lm/LEGACY.md`).
- **Statistics**: `oczy.common.stats` (multi-seed mean ± CI), dev/holdout
  splits with non-empty guarantee, always-on vanilla column.
- **Magnitude-controlled drift metric** (Δ_target / Δ_control / Δ_clamped)
  with deterministic clamp-budget capture (S2.0 fix).
- **Harnesses, ready to run**: minimal organism (`minimal_loop.py`), KV
  content channel (`minimal_loop_kv.py`), forgetting 2×2
  (`minimal_loop_forgetting.py`), subtractive + additive ablation matrices,
  invalidation ledger, dashboard `--check`.
- **`omp-fanout`** (own repo, on PATH): worktree-per-task headless-agent
  orchestration with status/retry/merge lifecycle; ~20 agent runs this
  program, surviving a night of provider stream failures.
- **Pre-registration corpus**: `research/09`–`19`, all with primaries,
  gates, and fallbacks fixed before implementation; dated amendments where
  the instrument changed.

---

## 7. Current honest baseline (what any future claim must beat)

Real driver, post-leakage, frozen eval v2
(`2026-07-01_honest_post_leakage_baseline.md`): Stage 0 = 0.88, Stage 1 =
0.75, Stage 2 = 0.69, Stage 3 = 0.38, Stage 4 = 0.90, Stage 5 = 0.92 —
with the scope-slot reranker (retrieval) known to carry the large stage-2/5
numbers, and vanilla at 0.00 across stages. External QA (July 1): organism
*worse* than vanilla cross-domain (0.388 vs 0.512) — the overfitting
signature that research/16's standing battery exists to police.

---

## 8. What remains — the two plasticity bets (Sprint 5)

Exactly two mechanisms consistent with all evidence remain untested:

- **`research/18` — consolidation as context distillation** (plasticity in
  LM weights): per-fact transient prefix → KL-distilled LoRA → delete prefix
  and traces → survival on holdout. Escapes S2.1's budget-eviction failure
  by construction; possible only since the substrate gained gradients.
- **`research/19` — the LM as language organ** (plasticity outside the LM):
  ≤64k-param trained head over frozen embeddings, abstain path,
  stage-1-untaught transfer battery. The reranker's zero-variance M2 wins
  are the bar to clear, on the axis (paraphrase transfer) where exemplar
  rerank structurally cannot generalize.

Both run on eval v2.1, both end in the same forgetting 2×2 and the same
honest accounting (`behavior_delta_per_byte` reported next to the raw-text
alternative, plus per-context-token cost and transfer). **If both refute,
the recorded conclusion is that retrieval is the architecture** — and the
north star, as originally formulated, was measuring the wrong thing.

Also pending: S3.4 attic moves with post-mortems; Sprint 4's weekly external
battery (incl. one non-repo-authored benchmark) and second-model
generalization (`research/16`, `research/17`).

---

*Full evidence chain: `experiments_logs/LEDGER.md` (authoritative index) →
individual logs; audit: `experiments_logs/2026-07-01_remediation_audit.md`;
plan and status: `SPRINT.md`; conceptual analyses: `notes/`.*
