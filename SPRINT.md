# Oczy — Remediation Sprint Plan

Drafted 2026-07-01 from a full-repo experiment audit (logs, harnesses, organs,
git history). Objective unchanged: the `experiments.txt` thesis — *memory
becomes changed dynamics, not retrieved content*; the key headline metric is
`behavior_delta_per_byte` on **held-out** probes.

The audit found the objective unmet for three structural reasons, which this
plan attacks in order:

1. **Substrate**: `llama-cpp-python` blocks Goal 1 (KV-slot writes) and
   Goal 2 (mid-layer hiddens); everything downstream was engineered around it.
2. **Eval/search entanglement**: the autoresearch loop was allowed to modify
   metrics, thresholds, baselines, and per-episode data paths — so it
   optimized the measurement, not the capability.
3. **Breadth before closure**: five organs + DSI + ingestion were built before
   any single metabolism loop closed; the mechanisms that actually work
   (text prefix, logit bias, scope-slot retrieval) are retrieval-shaped.

Sprints are one week each. Sprint 0 is a gate: nothing else counts until it
lands, because every number produced before it is untrustworthy.

---

## Sprint 0 — Eval integrity freeze (GATE)

**Goal:** a frozen, versioned, leak-free evaluation that the optimizing loop
cannot touch. All later sprints are scored against this eval and only this
eval.

> **Status 2026-07-01: Sprint 0 COMPLETE.** S0.1–S0.8 implemented via
> parallel headless agents (branches `fix/s0-*`, merged). The eval is frozen
> in `eval/v2/` (hash-checked MANIFEST.json, `verify_manifest()` on load,
> `bump_eval_version.py` for approved changes). Honest post-removal baseline
> recorded in `experiments_logs/2026-07-01_honest_post_leakage_baseline.md`:
> real-driver Stage 2 dropped 1.00 → 0.69, Stage 5 1.00 → 0.92; vanilla
> baseline = 0.00 on all stages. lane_07 gap 1.0 → 0.0 against a
> competitive baseline; lane_05 honest result = 0.0 (coverage was 1.0).
> Full test suite green at the time (500+ passed, 0 failures, 0 collection errors). As of commit `4f1a022` (2026-07-11) the collection surface is 800 tests collected, 48 collection errors from optional-dependency packages — see CURRENT_STATE.md §3.

### Tasks

- [x] **S0.1 — Freeze the eval into `eval/` with a version stamp.**
  Move curriculum episodes, probes, scoring (`scoring.py`), and thresholds
  into a single `eval/v2/` package. Hash the episode files; the harness
  refuses to run if the hash changed without a version bump.
> **Status 2026-07-01:** Implemented: eval/v2/ package with hash-checked MANIFEST.json, verify_manifest() loader, bump_eval_version.py, backward-compatible re-export shims.
- [x] **S0.2 — Remove test-set leakage.**
  Delete or quarantine every episode-ID-conditioned code path:
  - `_SCOPE_TEACHING` entries keyed to specific failing Stage-5/Stage-2
    episodes (commit `6de232d` and the lane_04 teaching dict).
  - `prefix_targets=[probe.expected]` on failing scope probes — the expected
    answer must never enter the answer path (logit bias, prefix, or
    otherwise) at probe time.
  Re-run the curriculum after removal and record the honest post-removal
  scores as the new baseline, whatever they are.
- [x] **S0.3 — Separate eval from search (working agreement + enforcement).**
  The autoresearch loop may modify `src/`, `plastic-cortex/`, organ packages.
  It may NOT modify `eval/`, lane thresholds, baselines, or scoring. Enforce
  with a CI check (path denylist on autoresearch branches) and state it in
  `AGENTS.md` so autonomous sessions inherit the rule.
- [x] **S0.4 — Kill gameable metrics.**
  - Retire "count of sub-metrics with spread" acceptance criteria
    (lane_01 `desaturation_count`) — a metric about metrics.
  - Retire "status reflects testing coverage" scores (lane_05) — a score
    that rises by running more tests.
  - Baselines designed to lose (lane_07 lexical gate "0 by construction")
    must be replaced with a competitive baseline or the lane's claim dropped.
- [x] **S0.5 — Statistical floor.**
  Every reported metric: ≥5 seeds, mean ± std, and n per stage. Grow stages
  to ≥20 episodes each (current 6–10 episodes means one flipped episode
  moves the metric ±10–17%). Add a tiny `stats.py` helper so no experiment
  can report a bare point estimate.
- [x] **S0.6 — Held-out split from day one.**
  Split every stage 70/30 into dev/held-out. Dev may drive development;
  held-out is run only at sprint end. Add paraphrase and adversarial
  variants of each held-out probe.
- [x] **S0.7 — Vanilla-LM baseline in every table.**
  The unmodified LM (no cortex, no organs, no prefix) runs the same probes.
  Any Oczy number reported without the vanilla column is invalid.
- [x] **S0.8 — Integration smoke tests for silent-failure surfaces.**
  One end-to-end behavioral assertion per mechanism ("teach X, probe X,
  assert the component actually fired"), plus an empirical distribution
  check for every hardcoded threshold (the 0.85 cosine retrieval threshold
  sat unreachable for four days because real embeddings land at 0.3–0.65).

### Definition of done
Frozen `eval/v2` exists; leakage paths deleted; honest post-removal baseline
recorded in `experiments_logs/`; CI blocks autoresearch edits to `eval/`;
all metrics reported with seeds/CI and a vanilla-LM column.

---

## Sprint 1 — Substrate migration (unblocks Goals 1 & 2)

**Goal:** replace `llama-cpp-python` with a HuggingFace/PyTorch substrate so
KV-cache writes and mid-layer hidden reads are ordinary tensor operations,
not research blockers.

> **Status 2026-07-02: COMPLETE.** Substrate: `Qwen/Qwen2.5-0.5B-Instruct`
> (82.8 ms/tok CPU; plain-transformer KV verified; 1.5B fallback recorded).
> `HFDriver` merged (peek_layer, cvec hooks, KV-splice API; 27+ contract
> tests). Pre-registered experiment verdicts:
> - **S1.4 REFUTE** (`2026-07-01_s1_4_hf_layer_probe.md`): no mid-layer
>   beats final-layer mean-pool on Qwen-0.5B (gap −0.083) OR LFM2.5 (+0.058
>   < +0.10). lane_03 confirmed as a model property, not a llama.cpp
>   keyhole. **Goal 2's mid-layer assumption is retired — the cortex
>   consumes final-layer hiddens.**
> - **S1.3 REFUTE on absolute recall** (`2026-07-01_s1_3_hf_kv_slot_injection.md`):
>   KV-splice rank-1 on 1/3 facts. But C2 (KV splice) matched C1 (text
>   prefix) rank-for-rank on every fact — **the KV mechanism is
>   behaviorally equivalent to the prefix at zero visible-token cost**;
>   the absolute failure is the 0.5B model's recall ceiling (the prefix
>   fails the same facts). Pre-blank splice position fixed the hardest
>   fact (rank 4→0). Sprint 2 implications: KV slots replace the prefix,
>   spliced pre-blank; consider the 1.5B fallback for recall-critical runs.
> - **Campaign Exp03 block → closure** (2026-07-11): the original campaign
>   re-run of the layer-L probe under the remote scheduler was
>   **infrastructure-blocked** — repeated HF snapshot transfers failed
>   before execution; no metrics or ASI scores emitted. That history is
>   preserved and not rewritten. A follow-up real-driver rerun (commit
>   `ad77e93`, `--driver real`, Colab, 2026-07-11) closed the
>   reproducibility gap: exit 0, `layer_l_silhouette_gap=0.10925446726657728`
>   exceeds the registered +0.10 threshold (unchanged), so this single run
>   is **positive/accept** for the reproducibility closure. This is not a
>   scientific null or refutation and does not reopen or overturn the S1.4
>   verdict, which stands on the pre-registered HF probe on two
>   architectures (`2026-07-01_s1_4_hf_layer_probe.md`). The infrastructure
>   fix used a seven-file exact-revision manifest with direct atomic HTTP
>   streaming and per-file size/SHA-256 verification, a fail-closed real
>   driver, and an HF final mean-pool baseline. Evidence:
>   `experiments_logs/2026-07-11_exp03_real_driver_closure.json`;
>   original campaign record: `experiments_logs/2026-07-11_campaign_0d48130.md`.

### Tasks

- [x] **S1.1 — Pick the model.**
  A plain small *transformer* (e.g. Qwen-0.5B/1.5B or Pythia-1B class), not
  a hybrid conv+attention model — LFM2.5's recurrence state made even KV
  snapshots ambiguous (348 KB blobs including conv1d state). Use
  `bench_hf_cpu.py` as the starting harness; confirm CPU latency is
  tolerable for the curriculum (seconds/probe is fine).
- [x] **S1.2 — `HFDriver` with the same surface as `LlamaCVecDriver`.**
  `generate()`, `peek_embedding()`, `set_cvecs_per_layer()` (forward
  pre-hooks adding residual bias), plus the two capabilities llama.cpp
  never gave us:
  - `peek_layer(prompt, layer_idx)` → real Goal 2, via
    `output_hidden_states=True`.
  - `write_kv_slot(layer, position, k, v)` → real Goal 1, by editing
    `past_key_values` before decode.
- [x] **S1.3 — Re-run the Goal-1 discriminating test on the new substrate.**
  The 2026-06-27 finding (cvec cannot force exact tokens; logit bias can)
  must be re-tested with true KV-slot injection: does a written KV slot
  achieve rank-1 recall *without* logit bias and *without* a text prefix?
  This is the single most thesis-relevant experiment in the whole plan.
- [x] **S1.4 — Re-run the Goal-2 layer-L probe honestly.**
  Lane 03's refutation (no mid-layer beats final-layer pooling) was run
  through llama.cpp's keyhole. Re-run the silhouette test across all layers
  with real hiddens. Pre-register the fallback analysis (pooling variants)
  this time instead of shopping post-hoc.
- [x] **S1.5 — Keep llama.cpp as a frozen legacy path.**
  Don't port organs yet; just keep old results reproducible.

### Definition of done
`HFDriver` passes the driver contract tests; KV-slot write demonstrably
shifts next-token logits (Goal 1 "done when" from `GOALS.md`); `peek_layer`
returns per-layer hiddens; S1.3/S1.4 results logged with seeds and the
vanilla column.

---

## Sprint 2 — Close ONE metabolism loop

> **Status 2026-07-02 (end of day): SPRINT 2 COMPLETE — all five tasks
> adjudicated.** S2.1 ran on the repaired stage-0 holdout split (see the
> research/11-13 amendment; the pre-registered split was degenerate, 0
> holdout probes) and is **REFUTE**
> (`experiments_logs/2026-07-02_s2_1_minimal_loop.md`): loop_delta_holdout
> 0.0000, rho nan, vanilla gate valid. Mechanism: the 48-token prefix budget
> evicts corrections (transient K=4 bump 0.13 collapses to 0 at K=8), and
> cvec posture harms generation at every tested amplitude (disabled;
> uncalibrated on Qwen). Per their pre-registered validity gates, **S2.2 and
> S2.5 are BLOCKED** (implementations + tests merged and ready). The clamp
> metric fix (S2.0, `f761cc0`) landed.
>
> Earlier same day: **S2.3+S2.4 pulled forward and DONE**
> (`experiments_logs/2026-07-01_s2_4_breakthrough_ablation.md`).
> **The 13.5x claim (044cb51) is RETRACTED as magnitude inflation**: control
> words rose MORE than target words (Δ_control 2.92 > Δ_target 1.60); under
> norm control the NEW config (0.566) falls below OLD unclamped (0.627);
> survival ratio 0.354 < 0.5; no single variable improves on OLD. Caveats:
> single seed; clamp-budget capture has a cross-instance stochasticity
> artifact (cond 1) — fixed by S2.0 (`f761cc0`). **Evidence-integrity
> caveat:** the committed log artifact contains a mock-driver table of
> zeros and `NaN` (the old harness silently wrote test/mock runs to the
> fixed experiment-log path); the real-run summary survives only outside
> the repo in a cache log. The qualitative retraction is the working
> conclusion, but exact values are provisional until a corrected
> multi-seed rerun is written to a new dated log. See CURRENT_STATE.md
> §4 "Critical evidence-integrity caveat: S2.4" for full detail.

**Goal:** the minimal thesis loop, end to end, with nothing else attached:
correction → cortex fast-weight change → consolidation → **changed LM
behavior on held-out probes** → raw trace deleted → behavior survives.

### Tasks

- [x] **S2.1 — Minimal organism: KVCortex + hippocampus + HFDriver only.** — REFUTE
  No critic, no identity, no immune, no autoencoder, no DSI, no scope-slot
  reranker. If the loop can't close with two components, five won't help.
- [x] **S2.2 — Content path through KV slots, not prefix text.** — BLOCKED (gate: S2.1 REFUTE); code merged
  Consolidated facts are injected as written KV entries (S1.2), retiring
  the token-burning articulation prefix. Cvec stays as the posture surface
  only, per the honest 06-25/06-27 findings.
- [x] **S2.3 — Magnitude-controlled drift metric.**
  Replace `metabolism_drift_delta` reporting with a three-part report:
  (a) target-domain logit delta, (b) non-target logit delta (specificity —
  steering vs shouting), (c) the same measurement with steering-vector
  norm normalized to a fixed budget. A "drift" gain that vanishes under
  norm control is loudness, not learning.
- [x] **S2.4 — Re-adjudicate the "13.5x breakthrough" (commit `044cb51`).**
  Single-variable ablation of the four things that changed at once
  (alpha_correction 1.0→0.3, replay threshold 3→2, batch 3→2, single→8
  diverse corrections), each on identical data, 5 seeds, full K-trajectory
  (K=0,1,5,10,15,20) with Spearman ρ against the spec's C2 criterion.
  Keep whichever mechanism survives; retract the claim if none does.
- [x] **S2.5 — The forgetting test (the thesis's signature move).** — BLOCKED (gate: S2.1 REFUTE); harness merged
  After consolidation, delete the hippocampus raw traces and re-run
  held-out probes. `behavior_delta_per_byte` is only meaningful if the
  bytes counted are the ones that *remain*. This is the first experiment
  that can actually distinguish "changed dynamics" from "retrieval with
  extra steps."

### Definition of done
On frozen eval v2, held-out split, ≥5 seeds: one correction produces
measurable behavior change; repeated corrections compound (monotone
K-trajectory); behavior survives raw-trace deletion; specificity and
norm-controlled variants reported alongside every drift number.

---

## Sprint 3 — Organ triage

> **Status 2026-07-03: COMPLETE — every organ has its verdict**
> (`experiments_logs/2026-07-03_s3_organ_triage_adjudication.md`).
> M1 subtractive (real GGUF, 8 configs) + M2 additive (HF minimal organism,
> holdout) ran per research/14. **Scope-slot reranker: RETRIEVAL-BASELINE**
> (M2 S0 +0.667 and S4 +0.250 at zero seed variance; M1 S2 +0.205) — kept,
> honestly labeled. **Everything else: ARCHIVE** — answer-time hippocampal
> retrieval added exactly 0.0000; DSI unsupported at v2 power and net-harmful
> in the full stack (appeal: v2.1 stage-1 battery); critic/identity/immune/
> autoencoder all noise in M1 (M2b harness merged as the appeal instrument;
> its run completed as a **metricless NULL** — `--seeds 3` exited 0 after
> 11,786.6 s but emitted no `METRIC` or `ASI` values; no effect estimate
> is available beyond the registered metricless null).
> **research/15 tensor wiring: VACUOUS** (nothing earned KEEP) — Goal 3's
> question closed honestly. S3.4 attic moves remain a pending code task.

**Goal:** stop carrying dead weight. Every organ must either move a
behavioral metric on the frozen eval or be archived.

### Tasks

- [x] **S3.1 — Ablation matrix.** — DONE (M1 + M2a; M2b harness merged, run completed as metricless NULL)
  Minimal organism ± each organ, one at a time, on frozen eval v2.
  An organ earns its place only if it moves a held-out behavioral metric
  beyond noise (per S0.5 statistics).
- [x] **S3.2 — Expected outcomes to confirm or refute (from the code audit):** — all five audit predictions CONFIRMED
  - **SkillImmuneCortex** — pure keyword substring matcher, no learned
    parameters, `Skill` compilation never invoked → archive or rebuild as
    the thesis's detector-merging design.
  - **WorldModelCritic** — 3/4 weights hardcoded, MLP head never trained →
    either train it on real accept/correct outcomes or archive.
  - **IdentityHypernetwork** — spent a week scoring 0.001–0.006 partly on a
    broken eval probe → decide from the fixed ablation, not history.
  - **ExperienceAutoencoder** — no decoder exists; it is not an
    autoencoder. Either implement the reconstruction objective from thesis
    §9 or rename/absorb it into the cortex's `train_step`.
  - **DSI/scope-slot reranker** — honest framing: these are retrieval.
    Keep them if they win, but label them as the retrieval baseline the
    metabolism must beat, not as metabolism.
- [x] **S3.3 — Wire survivors to tensors (Goal 3, for real).** — VACUOUS per research/15 (no KEEP verdicts)
  Organ outputs must update cortex state/projectors, not rerank label
  strings in `organism.py:_rank_answer`. One organ done properly beats
  five wired to a string ranker.
- [ ] **S3.4 — Archive the rest** (unblocked 2026-07-03: critic, identity, immune, autoencoder, DSI, answer-time hippocampal path) under `attic/` with a one-page post-mortem
  each, so autonomous sessions stop "improving" them.

### Definition of done
Every organ has an ablation verdict on frozen eval; survivors consume
tensors and demonstrably move held-out behavior; the rest are archived.

---

## Sprint 4 — Honest re-baseline and external validation

**Goal:** rebuild the headline claims on trustworthy ground and test outside
the home curriculum.

### Tasks

- [ ] **S4.1 — Invalidate-and-rerun ledger.**
  Every result produced June 26–29 ran on top of the silently-broken
  scope-slot reranker (three compounding bugs); the June 29 "1M-param
  expansion has zero effect" null and the reranker A/B were artifacts.
  Mark each affected log entry, re-run what still matters on eval v2.
- [ ] **S4.2 — External benchmark battery, weekly.**
  The July 1 external QA run showed cross-domain *worse* than vanilla
  (0.388 vs 0.512) — the overfitting signature. Promote the external QA +
  Pi tool-use benchmarks to a standing weekly job, with the vanilla column,
  and add at least one benchmark not authored by this repo.
- [ ] **S4.3 — Second model.**
  Run the frozen eval on a second small model. Single-model results
  (LFM2.5 only, 12 days) say nothing about the architecture generalizing.
- [ ] **S4.4 — Headline dashboard.**
  One table, auto-generated per run: `behavior_delta_per_byte` (post
  raw-trace deletion), uptake, transfer, scope, forgetting, identity —
  held-out, multi-seed, vanilla column — appended to
  `experiments_logs/` by the harness itself so no human (or agent) curates
  the numbers.

### Definition of done
All surviving claims reproduced on eval v2 with statistics; external + second
-model results logged; the dashboard is the only source of headline numbers.

---

## Standing working agreements (all sprints)

1. **The optimizing loop never touches the measuring instrument.** Metrics,
   thresholds, baselines, episodes, scoring: frozen per version, changed
   only by explicit human decision with a version bump.
2. **No episode-ID-conditioned code, ever.** Fixes must be mechanism-level.
3. **One variable at a time.** Multi-parameter commits cannot claim causal
   improvements; "breakthrough" requires ablation + trajectory + seeds.
4. **Nulls and refutations are results.** Log them as prominently as wins
   (the repo already did this well — keep it).
5. **Retrieval is the baseline, not the enemy.** Prefix/logit-bias/rerank
   paths stay in every table as the bar that changed-dynamics must clear;
   claiming their wins as metabolism is the failure mode to avoid.
6. **Every threshold gets a distribution check** against real data before it
   ships (lesson of the 0.85 cosine bug).

## Sequencing

```
Sprint 0 (gate: eval integrity)                    ✅ complete
   └─► Sprint 1 (substrate: HF driver, Goals 1+2)  ✅ complete
          └─► Sprint 2 (one closed loop + forgetting + 13.5x)  ✅ complete
                 └─► Sprint 3 (organ triage, Goal 3)           ✅ verdicts in
                        └─► Sprint 4 (re-baseline, external, 2nd model)
                        └─► Sprint 5 (the two plasticity bets)
```

## Sprint 5 — Plasticity bets and the learned-cortex successor

The 2026-07-03 plan identified two immediate mechanisms after the Sprint 1–3
refutation arc. **Human-authorized amendment 2026-07-09:** those remain useful
comparators, but they are not the final judgement on the cortex premise.
Research/19's label-prefix path can be parametric retrieval, and neither 18 nor
19 learns the online update rule across tasks. Research/20 now carries the core
test: a meta-trained cortex controlling a frozen language organ with retrieval
disabled in the primary condition. Research/21 is the dependent multi-organ
extension.

- [ ] **S5.1 — research/18: consolidation as context distillation.** — **PARTIAL (Campaign 0d48130, 2026-07-11)**
  Per-fact transient prefix → KL-distilled LoRA → delete prefix + traces →
  survival on holdout. Plasticity in LM weights; retained as the mouth-weight
  comparator, not the frozen-organ cortex condition.
  > **Campaign 0d48130 adjudication:** the R18 teacher gate passed (1 seed,
  > `distill_delta_holdout=0.3333`, `distill_specificity_delta=0.04348`).
  > The R18 full 3-seed run is **PARTIAL**: distillation signal in 2/3 seeds,
  > absent in 1/3. `distill_delta_holdout` is bimodal {0.3333, 0.3333, 0.0}
  > (mean=0.2222); `teacher_dev_delta=0.1765` and `persistent_bytes=17,699,903`
  > are identical across seeds; `specificity_delta` is {0.0, 0.0, 0.04348}.
  > Single-seed gate does not constitute a cross-seed claim. A 5-seed
  > `stage_0` rerun is **diagnostic** unless the unchanged
  > `teacher_dev_delta` ≥ 0.2 validity gate passes (currently 0.1765 <
  > 0.2, so the gate is not met). No threshold changes. Evidence:
  > `experiments_logs/2026-07-11_campaign_0d48130.md`.
- [ ] **S5.2 — research/19: direct cortex learning, two articulation arms.** — **unimplemented**
  The same ≤64k-param online-trained cortex is evaluated through (A) a
  label-prefix parametric-retrieval readout and (B) a fixed-width latent-control
  readout into a frozen LM. Only B can support the cortex premise; zero/swap/
  shuffled-feedback interventions must prove causal dependence on cortex state.
- [ ] **S5.3 — Diagnostic head-to-head table:** 18 vs both 19 arms vs
  retrieval-baseline vs vanilla, with deletion audits, CIs, per-byte accounting,
  and explicit classification of every answer path. This table adjudicates the
  direct mechanisms; it does not close Research/20 before the learned update
  rule has been tested.
- [ ] **S5.4 — research/20 / experiment/09: meta-trained cortex over a frozen
  language organ.** — **unimplemented**; meta-test blocked on human sign-off.
  Developmentally learn write, read, consolidation, and
  latent articulation rules across task families; freeze them; then learn an
  unseen rule online with no backprop, retrieval, trace, label text, or LM
  update in the primary condition. Require transfer, composition, deletion,
  and state-causal controls. **The meta-test MUST NOT run without explicit
  human sign-off.**
- [ ] **S5.5 — research/21: cortex-routed frozen specialist organs.** Begin
  only if S5.4 accepts. Add a separately frozen action/tool organ, opaque tool
  families, learned routing, and recurrent goal state. Existing Pi tasks become
  an external battery, not the primary measuring instrument.

Background/conceptual grounding:
`notes/2026-07-03_steering_vs_posture_postmortem.md`; successor rationale and
frozen-organ boundary are fixed in `research/20` and `research/21`.

## Next actionable todos (Campaign 0d48130 → forward)

Dependency-ordered. Each item has an observable acceptance criterion. No
threshold, metric, baseline, or episode change is implied; frozen eval
remains frozen unless an item explicitly calls for the governance path.

1. **Exp03 real-driver infrastructure correction (reproducibility closure) — COMPLETE (2026-07-11).**
   The original campaign Exp03 run was infrastructure-blocked (HF snapshot
   transfer failures) and produced no scientific verdict; that history is
   preserved. A follow-up real-driver rerun (commit `ad77e93`,
   `--driver real`, Colab, 2026-07-11) closed the gap: exit 0,
   `layer_l_silhouette_gap=0.10925446726657728` exceeds the registered
   +0.10 threshold (unchanged) → positive/accept for this single
   reproducibility closure. The pre-registered S1.4 refutation (two
   architectures) is not reopened or overturned. **Acceptance met:** durable
   execution report at `experiments_logs/2026-07-11_exp03_real_driver_closure.json`;
   S1.4 is not reopened.

2. **R18: extend to ≥5 seeds and diagnose LoRA uptake variance.**
   The 3-seed run is **PARTIAL**: bimodal {0.3333, 0.3333, 0.0}; 1/3 seeds
   show no distillation signal. A 5-seed `stage_0` rerun is
   **diagnostic** unless the unchanged `teacher_dev_delta` ≥ 0.2
   validity gate passes — currently `teacher_dev_delta=0.1765` < 0.2, so
   the gate is not met and the rerun remains diagnostic, not
   confirmatory. Extend to ≥5 seeds without changing thresholds.
   Diagnose why seed 2 produced `distill_delta_holdout=0.0` while seeds 0–1
   produced 0.3333; check LoRA initialization, gradient flow, and data order.
   **Acceptance:** ≥5-seed `distill_delta_holdout` with mean ± std in a
   dated log; a written mechanism hypothesis for the null seed; no
   threshold change.

3. **S4.1: complete honest reruns.**
   Re-run every June 26–29 result that depended on the broken scope-slot
   reranker or leakage-era paths on eval v2, mark each affected ledger
   entry, and record the honest replacement. **Acceptance:** every
   INVALIDATED/SUPERSEDED ledger row points to a dated honest rerun log
   or is explicitly labeled "no longer relevant."

4. **Research/19: implement matched label-prefix vs latent-control arms.**
   Build the two-arm diagnostic under a dedicated module in
   `src/oczy/experiments/`. Arm A: online-trained cortex decoded to a label
   prefix (parametric retrieval). Arm B: same cortex state through a
   fixed-width learned latent coupler into the frozen LM, no label text.
   Include zero-state, swapped-state, shuffled-feedback, vanilla, retrieval,
   and oracle conditions. **Acceptance:** a dated log with both arms scored
   on eval v2 held-out, multi-seed, vanilla column; Arm B causal controls
   pass or fail explicitly.

5. **S5.3: build the diagnostic head-to-head comparison table.**
   One table: R18 (consolidation-as-distillation) vs both R19 arms vs
   retrieval-baseline vs vanilla, with deletion audits, CIs, per-byte
   accounting, and explicit classification of every answer path.
   **Acceptance:** `experiments_logs/DASHBOARD.md` or a dated log contains
   the table with all columns filled and every path classified as
   retrieval, metabolism, or vanilla.

6. **Research/20: materialize `meta_cortex/v1` Phase 0 instrument and
   obtain human sign-off before meta-test.**
   Build the separate task generator, task-level train/dev/test split,
   manifest, leakage audit, threshold distributions, and power analysis.
   **The R20 meta-test requires explicit human sign-off and MUST NOT run
   without it.** **Acceptance:** the Phase 0 instrument exists under
   `src/oczy/experiments/meta_cortex/`, the manifest verifies, and a
   human sign-off is recorded before any meta-test run begins.

7. **External battery / second model.**
   After honest baselines (items 3–5) are in place, run the frozen eval on
   a second small model and promote the external QA + Pi tool-use
   benchmarks to a standing weekly job with the vanilla column. Add at
   least one benchmark not authored by this repo. **Acceptance:** a dated
   log with a second-model column and at least one external benchmark
   result, both with vanilla comparison.

8. **Research/21: remains blocked on Research/20 acceptance.**
   Do not start the multi-organ router until S5.4 accepts. **Acceptance:**
   no work begins until the Research/20 decision gate is passed.

9. **Durable live watch queue — ACTIVE (2026-07-11).**
   Watch mode for `parallel_scheduler.py` is **implemented and tested**:
   atomically reload a changed batch, merge only unseen job names as
   pending, never mutate existing job definitions or states, retry
   malformed reloads without killing the daemon, and stay alive waiting
   for future jobs. Existing non-watch behavior remains terminating and
   backward compatible. The queue setup action is **complete**; the
   experiment result is **pending**. Live queue paths: batch
   `/tmp/oczy-live-queue/batch.json`, state
   `/tmp/oczy-live-queue/state.json`, campaign
   `/tmp/oczy-live-queue-campaign.json`. Source commit:
   `5b5e93c63d769fea7854073a4e6c359e5d36606f`. Capacity is **additive:
   10 Kaggle + learned Colab X**. The background scheduler runs with
   `--watch-batch --watch-interval 30`. **First running job:**
   `r18-distillation-5seed-diagnostic` (Kaggle, kernel
   `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, pinned source dataset
   `abdellahkadem/oczy-source-5b5e93c63d76`, source archive sha256
   `bc1ff926bc679fc26e5f20cfcb0756339b002ff3c1027eb3c24251fe2f6d7f72`,
   module `oczy.experiments.consolidation_distillation`, args
   `--seeds 5 --max-steps 10 --stage stage_0_grounding`). State is
   `running` — the job is merely running, **not** completed or
   successful. The R18 job is **diagnostic** unless the unchanged
   `teacher_dev_delta` ≥ 0.2 validity gate passes (currently 0.1765 <
   0.2, so the gate is not met). No threshold, metric, or eval change
   is implied. The Research/20 meta-test sign-off prohibition is
   unchanged. **Acceptance:** watch mode implemented and verified by
   tests (met); the live queue is active with the first job running;
   the experiment result remains pending until the job completes and is
   adjudicated.

Sprint 0 and Sprint 1 can overlap after S0.1–S0.4 land; nothing in Sprints
2–4 may start before Sprint 0 is fully done.
