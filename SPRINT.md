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
> Full test suite green (500+ passed, 0 failures, 0 collection errors).

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

> **Status 2026-07-01: IN PROGRESS.** Kickoff: S1.3/S1.4 experiment specs
> pre-registered as `research/09-hf-kv-slot-fact-injection.md` and
> `research/10-hf-layer-l-hidden-probe.md` (fallback analyses fixed in
> advance; acceptance judged only on primaries). Phase A (parallel agents):
> S1.1 model selection, S1.2 HFDriver + contract tests on a tiny random
> model, S1.5 legacy freeze. Phase B (after A merges): S1.3 + S1.4 per specs.

### Tasks

- [ ] **S1.1 — Pick the model.**
  A plain small *transformer* (e.g. Qwen-0.5B/1.5B or Pythia-1B class), not
  a hybrid conv+attention model — LFM2.5's recurrence state made even KV
  snapshots ambiguous (348 KB blobs including conv1d state). Use
  `bench_hf_cpu.py` as the starting harness; confirm CPU latency is
  tolerable for the curriculum (seconds/probe is fine).
- [ ] **S1.2 — `HFDriver` with the same surface as `LlamaCVecDriver`.**
  `generate()`, `peek_embedding()`, `set_cvecs_per_layer()` (forward
  pre-hooks adding residual bias), plus the two capabilities llama.cpp
  never gave us:
  - `peek_layer(prompt, layer_idx)` → real Goal 2, via
    `output_hidden_states=True`.
  - `write_kv_slot(layer, position, k, v)` → real Goal 1, by editing
    `past_key_values` before decode.
- [ ] **S1.3 — Re-run the Goal-1 discriminating test on the new substrate.**
  The 2026-06-27 finding (cvec cannot force exact tokens; logit bias can)
  must be re-tested with true KV-slot injection: does a written KV slot
  achieve rank-1 recall *without* logit bias and *without* a text prefix?
  This is the single most thesis-relevant experiment in the whole plan.
- [ ] **S1.4 — Re-run the Goal-2 layer-L probe honestly.**
  Lane 03's refutation (no mid-layer beats final-layer pooling) was run
  through llama.cpp's keyhole. Re-run the silhouette test across all layers
  with real hiddens. Pre-register the fallback analysis (pooling variants)
  this time instead of shopping post-hoc.
- [ ] **S1.5 — Keep llama.cpp as a frozen legacy path.**
  Don't port organs yet; just keep old results reproducible.

### Definition of done
`HFDriver` passes the driver contract tests; KV-slot write demonstrably
shifts next-token logits (Goal 1 "done when" from `GOALS.md`); `peek_layer`
returns per-layer hiddens; S1.3/S1.4 results logged with seeds and the
vanilla column.

---

## Sprint 2 — Close ONE metabolism loop

**Goal:** the minimal thesis loop, end to end, with nothing else attached:
correction → cortex fast-weight change → consolidation → **changed LM
behavior on held-out probes** → raw trace deleted → behavior survives.

### Tasks

- [ ] **S2.1 — Minimal organism: KVCortex + hippocampus + HFDriver only.**
  No critic, no identity, no immune, no autoencoder, no DSI, no scope-slot
  reranker. If the loop can't close with two components, five won't help.
- [ ] **S2.2 — Content path through KV slots, not prefix text.**
  Consolidated facts are injected as written KV entries (S1.2), retiring
  the token-burning articulation prefix. Cvec stays as the posture surface
  only, per the honest 06-25/06-27 findings.
- [ ] **S2.3 — Magnitude-controlled drift metric.**
  Replace `metabolism_drift_delta` reporting with a three-part report:
  (a) target-domain logit delta, (b) non-target logit delta (specificity —
  steering vs shouting), (c) the same measurement with steering-vector
  norm normalized to a fixed budget. A "drift" gain that vanishes under
  norm control is loudness, not learning.
- [ ] **S2.4 — Re-adjudicate the "13.5x breakthrough" (commit `044cb51`).**
  Single-variable ablation of the four things that changed at once
  (alpha_correction 1.0→0.3, replay threshold 3→2, batch 3→2, single→8
  diverse corrections), each on identical data, 5 seeds, full K-trajectory
  (K=0,1,5,10,15,20) with Spearman ρ against the spec's C2 criterion.
  Keep whichever mechanism survives; retract the claim if none does.
- [ ] **S2.5 — The forgetting test (the thesis's signature move).**
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

**Goal:** stop carrying dead weight. Every organ must either move a
behavioral metric on the frozen eval or be archived.

### Tasks

- [ ] **S3.1 — Ablation matrix.**
  Minimal organism ± each organ, one at a time, on frozen eval v2.
  An organ earns its place only if it moves a held-out behavioral metric
  beyond noise (per S0.5 statistics).
- [ ] **S3.2 — Expected outcomes to confirm or refute (from the code audit):**
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
- [ ] **S3.3 — Wire survivors to tensors (Goal 3, for real).**
  Organ outputs must update cortex state/projectors, not rerank label
  strings in `organism.py:_rank_answer`. One organ done properly beats
  five wired to a string ranker.
- [ ] **S3.4 — Archive the rest** under `attic/` with a one-page post-mortem
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
Sprint 0 (gate: eval integrity)
   └─► Sprint 1 (substrate: HF driver, Goals 1+2)
          └─► Sprint 2 (one closed loop + forgetting test + 13.5x ablation)
                 └─► Sprint 3 (organ triage, Goal 3)
                        └─► Sprint 4 (re-baseline, external, 2nd model)
```

Sprint 0 and Sprint 1 can overlap after S0.1–S0.4 land; nothing in Sprints
2–4 may start before Sprint 0 is fully done.
