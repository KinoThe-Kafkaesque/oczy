# Oczy — Current Project State

**Last updated:** 2026-07-16

**Evidence cutoff:** repository and local experiment artifacts inspected through
2026-07-16

**Purpose:** canonical living handoff for what Oczy is, what has actually been
demonstrated, what changed in the research direction, and where the remaining
work lives.

This is the current-state document. It does not rewrite historical results.
Use the sources in this order:

1. [`experiments_logs/LEDGER.md`](experiments_logs/LEDGER.md) classifies the
   validity of individual experiment logs.
2. [`FINDINGS.md`](FINDINGS.md) is the consolidated historical report through
   2026-07-03. Its evidence remains useful, but its old statement that only
   Research/18 and /19 remain is superseded by the human-approved July 9
   direction recorded here and in Research/20–21.
3. [`SPRINT.md`](SPRINT.md) is the active execution plan.
4. [`research/README.md`](research/README.md) and [`experiments/README.md`](experiments/README.md)
   index hypotheses and experiment specifications. A specification is not an
   implementation or a result.

## 1. What the project is

Oczy is a Plastic World Model Agent research project. Its core thesis is:

> experience → fast neural change → replay → compression → slow neural change
> → deletion of the raw trace

The intended agent does not answer by looking up a stored episode. It should
change because of experience, retain useful knowledge and behavior in bounded
neural state, consolidate it, and still act correctly after the original trace
is removed.

The current architectural framing is functional, not a claim that the design
must imitate biology:

- The **cortex is the learner**. It owns fast state, slow state, online updates,
  consolidation, addressing, goal state, and eventually routing.
- A frozen language model is the **language organ or mouth**. It contributes
  pretrained perception and language, but its weights remain frozen while the
  cortex is being tested.
- A separate small frozen action/tool model may later be the **hands**. The
  cortex, not either specialist model, should decide what state matters and
  which organ to use.
- Retrieval remains a required product capability and a required external
  baseline. During the present mechanism-isolation phase it is disabled in the
  primary cortex condition, because retrieved text would hide whether the
  cortex itself learned.

That distinction is now explicit:

| Context | Retrieval status | Reason |
|---|---|---|
| Primary cortex experiment | Disabled; no answer/label/correction text or raw episode may be reintroduced at probe time | Measures raw learned-state capability |
| Experiment tables | Always include matched retrieval, vanilla, and oracle comparators | Establishes the bar and prevents relabeling retrieval as metabolism |
| Eventual product | Expected to be available and useful | Product performance is broader than mechanism identification |

## 2. Honest current verdict

The original changed-dynamics thesis has **not yet been demonstrated**. The
repository has strong experimental infrastructure and several useful
refutations, but the mechanisms that reliably changed answers so far were
retrieval-like paths: visible prefix content, KV content equivalent to a
prefix, logit manipulation, and the scope-slot reranker. None establishes a
cortex that learns an unseen rule from experience and later expresses it
through a frozen model.

The July 9 direction is still scientifically plausible and worth testing, but
it is high risk. The project now has a sharper central experiment instead of a
collection of loosely connected organs. It will bear fruit if a meta-trained
cortex can acquire a new held-out rule online, survive deletion of the teaching
trace, and lose or exchange that behavior under cortex-state interventions.
It should be refuted cleanly if it cannot.

### The main blind spot found

Earlier work assumed that a hand-authored Hebbian update and an untrained or
random projection could turn an experience embedding into a state that a
frozen language model would know how to use. There were two missing learned
protocols:

1. **Learning how to learn:** the write, addressing, retention, forgetting,
   and consolidation rules were not meta-trained across task families.
2. **Learning how to communicate with the frozen organ:** storing information
   in a neural state is not sufficient. A learned read/coupling protocol must
   translate query-conditioned cortex state into a latent control signal the
   frozen language or action organ can interpret.

The barrier is therefore not evidence that neural networks cannot store
memories or experience. It is the credit-assignment and interface problem:
learning what to write, where to write it, when to consolidate it, how to read
it conditionally, and how to make a frozen specialist act on that readout
without smuggling the answer through text.

## 3. What was done previously

### June 20–30: broad architecture and mechanism exploration

The project built the initial plastic-cortex organism, ingestion and replay
paths, fast/slow state, consolidation experiments, a hippocampus, DSI fact
index, critic, identity hypernetwork, skill-immune component, experience
autoencoder, several steering/readout paths, curriculum stages, and many
stressors. The initial substrate was LFM2.5-1.2B through llama.cpp.

This period produced useful code and probes, but it also mixed too many
variables, built many organs before closing one causal learning loop, used
several saturated or gameable metrics, and allowed the evaluation to co-evolve
with the system. Some early headline wins did not survive audit.

Primary historical locations:

- original thesis: [`experiments.txt`](experiments.txt)
- historical findings: [`FINDINGS.md`](FINDINGS.md)
- experiment log classifications: [`experiments_logs/LEDGER.md`](experiments_logs/LEDGER.md)
- process audit: [`experiments_logs/2026-07-01_remediation_audit.md`](experiments_logs/2026-07-01_remediation_audit.md)
- steering post-mortem: [`notes/2026-07-03_steering_vs_posture_postmortem.md`](notes/2026-07-03_steering_vs_posture_postmortem.md)

### July 1–3: remediation and re-adjudication

The remediation work delivered:

- a hash-checked, frozen `eval/v2` instrument, currently manifest version
  **v2.2**;
- removal of episode-ID and expected-answer leakage;
- held-out splits, vanilla and retrieval baselines, confidence intervals,
  trajectories, seed requirements, and pre-registered acceptance/kill rules;
- a Hugging Face driver and selection of frozen
  `Qwen2.5-0.5B-Instruct` for the fast experimental substrate;
- HF KV-splice and layer-hidden probes;
- minimal-loop, forgetting, organ-ablation, and additive-control harnesses;
- a validity ledger that labels nulls, refutations, invalidations, and
  superseded claims rather than hiding them.

The July 9 audit recorded **680 passed, 14 warnings**; that number is
historical, not current. As of commit `4f1a022` (2026-07-11) the test
collection surface is **800 tests collected, 48 collection errors** — the
errors are import-time failures in optional-dependency packages
(plastic-cortex, neural-hippocampus, world-model-critic,
identity-hypernetwork, skill-immune-cortex, experience-autoencoder) and
`src/oczy/lm` driver tests, not test failures. The `eval/v2` manifest
verified. These are code-integrity checks, not evidence that the
scientific thesis passed.

## 4. Verified experimental picture

The table below separates useful mechanisms from evidence for the thesis.

| Test or claim | Current verdict | Important evidence |
|---|---|---|
| Legacy v2 curriculum baseline | Historical reference, not a v2.2 curve or cortex win | Real-driver v2 stages 0–5: **0.88, 0.75, 0.69, 0.38, 0.90, 0.92**; v2.2 baseline pending |
| Stage-2 scope = 1.00 | Superseded by leakage removal | **1.00 → 0.69** |
| Stage-5 cross-domain = 1.00 | Superseded by leakage removal | **1.00 → 0.92** |
| Exact-token residual cvec | Refuted | Target remained around rank 47k; best contrastive result lifted rank but did not reach rank 1 |
| Post-forward logit bias | Works, but disqualified as memory | Can force rank 1 by changing decoding logits |
| Mid-layer hidden premise | Refuted on both tested architectures | Qwen best-mid underperformed final mean; LFM gap was below the pre-registered margin |
| KV-slot injection | Absolute Goal-1 criterion refuted; interface mechanism validated | Only **1/3** facts at rank 1, but pre-blank KV splice matched a visible text prefix rank-for-rank with zero visible tokens |
| Minimal prefix metabolism loop | Refuted at its primary endpoint | Five seeds gave 0 holdout effect at K=N; transient **0.1333** at K=4 disappeared by K=8 as the 48-token prefix budget evicted corrections |
| HF cvec posture path | Harmful at tested amplitudes | Large norm caused repetition; even clamped steering corrupted probes |
| Scope-slot reranker | Retrieval baseline; the one organ with a useful evaluated contribution | Subtractive M1 average contribution +0.0465, concentrated in scope and cross-domain stages |
| Other answer-time organs | Archived or retained only for a narrow appeal | Hippocampus answer path bit-identical to base; DSI unsupported/net harmful on this battery; critic, identity, immune, and autoencoder were noise or harmful |
| External QA | Overfitting warning | Organism **0.388** versus vanilla **0.512** |
| Pi tool-use battery | Current external result: **0/3** | Current proxy prepends stored facts; it is not yet evidence of cortex learning or routing |

Detailed evidence:

- honest baseline: [`experiments_logs/2026-07-01_honest_post_leakage_baseline.md`](experiments_logs/2026-07-01_honest_post_leakage_baseline.md)
- KV result: [`experiments_logs/2026-07-01_s1_3_hf_kv_slot_injection.md`](experiments_logs/2026-07-01_s1_3_hf_kv_slot_injection.md)
- layer result: [`experiments_logs/2026-07-01_s1_4_hf_layer_probe.md`](experiments_logs/2026-07-01_s1_4_hf_layer_probe.md)
- minimal loop: [`experiments_logs/2026-07-02_s2_1_minimal_loop.md`](experiments_logs/2026-07-02_s2_1_minimal_loop.md)
- organ adjudication: [`experiments_logs/2026-07-03_s3_organ_triage_adjudication.md`](experiments_logs/2026-07-03_s3_organ_triage_adjudication.md)
- subtractive matrix: [`experiments_logs/2026-07-02_s3_m1_subtractive_ablation.md`](experiments_logs/2026-07-02_s3_m1_subtractive_ablation.md)
- Pi work: [`experiments/08-oczy-pi-tool-calling-curriculum/README.md`](experiments/08-oczy-pi-tool-calling-curriculum/README.md) and [`benchmarks/pi/`](benchmarks/pi/)

### Critical evidence-integrity caveat: S2.4

The broad conclusion that the old “13.5x metabolism drift” was magnitude
inflation is consistent with the audit, but the committed evidence artifact is
not trustworthy enough for precise reuse:

- [`experiments_logs/2026-07-01_s2_4_breakthrough_ablation.md`](experiments_logs/2026-07-01_s2_4_breakthrough_ablation.md)
  currently contains a mock-driver table of zeros and `NaN`, because the old
  harness silently wrote tests/mock runs to the fixed experiment-log path.
- The real-run summary survives only outside the repository in
  `~/.cache/omp-fanout/home-nyanpasu-Desktop-code-kinoSoft-oczy/logs/s2-drift-ablation.log`.
  It reports old target Δ 0.627; new target Δ 1.600; new control Δ 2.916;
  clamped new target Δ 0.566; survival 0.354; and no positive single-variable
  ablation.
- That run used one seed and also identified a stochastic clamp-budget flaw.
  The harness has since stopped writing unless given an explicit output path
  and now seeds/captures the budget consistently, but the corrected real run
  has not been recorded.

Therefore the qualitative retraction remains the working conclusion, while
the exact values must be treated as provisional until a corrected, multi-seed
rerun is written to a new dated log and the ledger is amended.

### Campaign 0d48130 (2026-07-11)

10 run groups adjudicated from three source commits (`0d48130`, `537260c`,
`2a22049`) under CPU-only contract via the remote scheduler (Kaggle +
Colab). 9 completed (including one metricless R14 result); 1 (Exp03) was
infrastructure-blocked in the original campaign and subsequently closed by a
real-driver rerun (commit `ad77e93`, 2026-07-11). Nulls and refutations are
recorded as prominently as positives. Not every catalogued project ran.

| Experiment | Verdict | Primary metric | Seeds |
|---|---|---|---|
| Exp01 correction-competence | **NULL** (behavior-delta transfer) | `v2_behavior_delta_mock=0.0`, `v2_discrimination=0.0` | 1 |
| Exp02 KV-slot injection | **REFUTATION** | `kv_slot_rank1_count=0.0` (logit bias confirmed: rank1_count=3.0) | 1 |
| Exp03 layer-L probe | **INFRASTRUCTURE BLOCKED** (original campaign); **POSITIVE/ACCEPT** (ad77e93 real-driver closure) | Original: no metrics. Closure: `layer_l_silhouette_gap=0.10925446726657728` (> +0.10) | — |
| Exp04 scope-selectivity | **POSITIVE** | `scope_selectivity_index=1.0` | 1 |
| Exp05 metabolism-loop | **NULL** (metabolism drift) | `metabolism_drift_delta=0.0`, `drift_uptake=0.0` | 1 |
| Exp06 bounded-growth | **POSITIVE** | `bounded_growth_m1_ratio=0.002079`, zero variance | 5 |
| Exp07 conversation-world-model | **POSITIVE** (marker-free) + **NULL** (critic AUC) | `marker_free_uptake_gap=1.0`, `critic_auc_delta=0.0` | 1 |
| R18 teacher gate | **BLOCKED** (teacher validity gate failed) | `teacher_dev_delta=0.1765` < 0.2 gate; `distill_delta_holdout=0.3333` | 1 |
| R18 distillation 3-seed | **BLOCKED** (teacher gate failed; diagnostic only) | `distill_delta_holdout` mean=0.2222, bimodal {0.3333, 0.3333, 0.0}; `teacher_dev_delta=0.1765` all seeds | 3 |
| R14 M2b additive-organs | **NULL (metricless)** | 3 seeds, exit 0, no `METRIC`/`ASI` values | 3 |

Exp03's original campaign run was infrastructure-blocked (HF snapshot
transfer failures) and produced no scientific verdict — that history is
preserved and not rewritten. A follow-up real-driver rerun (commit
`ad77e93`, `--driver real`, Colab, 2026-07-11) closed the reproducibility
gap: exit 0, `layer_l_silhouette_gap=0.10925446726657728` exceeds the
registered +0.10 threshold, so this single run is **positive/accept** for
the reproducibility closure. The +0.10 threshold was not changed. This
does not reopen or overturn the pre-registered S1.4 refutation
(`2026-07-01_s1_4_hf_layer_probe.md`), which was adjudicated on two
architectures (Qwen gap −0.083; LFM2.5 gap +0.058); it is a single
real-driver run on one architecture (LFM2.5-1.2B-Instruct). The
infrastructure fix used a seven-file exact-revision manifest with direct
atomic HTTP streaming and per-file size/SHA-256 verification, a fail-closed
real driver, and an HF final mean-pool baseline. R14 M2b's metricless null
means no effect estimate is available beyond the registered null — it is
not a positive or negative mechanism verdict.

Evidence: [`experiments_logs/2026-07-11_exp03_real_driver_closure.json`](experiments_logs/2026-07-11_exp03_real_driver_closure.json)
(durable execution report); [`experiments_logs/2026-07-11_campaign_0d48130.md`](experiments_logs/2026-07-11_campaign_0d48130.md)
(original campaign adjudication); adjudication in [`experiments_logs/LEDGER.md`](experiments_logs/LEDGER.md).

### R18 five-seed diagnostic (2026-07-11, commit `5b5e93c`)

A follow-up 5-seed `stage_0` rerun was submitted via the durable live
watch queue (Kaggle CPU, kernel
`abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, source commit
`5b5e93c63d769fea7854073a4e6c359e5d36606f`). **Infrastructure:
COMPLETE** (exit 0, all metrics collected). **Scientific verdict:
BLOCKED at the teacher validity gate — diagnostic only.**

The unchanged teacher gate is `teacher_dev_delta ≥ 0.2`. Every seed
observed `teacher_dev_delta=0.17647058823529413`, which is below the
0.2 gate. Because the teacher gate failed after registered fallback,
no H-DISTILL verdict is permitted regardless of the holdout deltas.

Per-seed `distill_delta_holdout`: {0.3333, 0.3333, 0.0, 0.3333,
0.3333} — 4/5 seeds positive, seed 2 null (preserved). Mean
`distill_delta_holdout=0.26666666666666666`. Mean
`specificity_delta=0.02608695652173913`. The 4/5 positive holdout
deltas are infrastructure-confirmed but scientifically inadmissible
because the teacher gate failed.

**Mechanism diagnosis COMPLETE (2026-07-12, commit `33169cc`):**
teacher ceiling, prompt-contract, and trajectory diagnostics ran on
Kaggle CPU. The teacher gate remains FAILED; no H-DISTILL verdict is
permitted.

- **Teacher ceiling** (n=17): vanilla=0, raw_prefix=0.17647058823529413,
  chat_template=0. Neither reaches the 0.2 gate. The registered chat
  fallback (0) is worse than raw_prefix (0.1765).
- **Prompt-contract audit:** issue/malformed/missing/truncated/
  answer-leak/mismatch counts all 0; `teacher_correct_rate=0.17647058823529413`;
  raw and chat-template prompt accuracies 0. No structural prompt defect.
- **Training trajectory:** the first submission failed with HTTP 400
  (long kernel slug); the short-slug retry succeeded (exit 0 after
  ~12798 s) and is the run of record — both preserved. Train loss falls
  ~0.70 → ~0.16; mean slope -0.0615; second-half slope -0.0190;
  underfit=1, instability=1, saturation=0; max final-loss divergence
  0.01259. Optimization fits token loss, but DEV behavior is
  unstable/weak and not saturated.
- **Final DEV student accuracies** (seeds 0–4) =
  {0.117647, 0, 0, 0, 0.117647}. Teacher remains 0.17647. Seed 2 is
  not uniquely divergent — seeds 1 and 3 also score 0.

**Conclusion:** no structural prompt defect; registered chat fallback
is worse than raw_prefix; the teacher expressivity/prompt-task ceiling
is the blocker. Optimization fits token loss but DEV behavior is
unstable/weak and not saturated. Further identical R18 reruns are
retired — they will not clear the unchanged teacher gate. Next work
points to R19 DEV calibration while signed evaluation (Research/20
meta-test) remains gated. No threshold, metric, or eval changes.
### R19 DEV calibration (2026-07-12, source `bd1ead9a`)

Research/19's DEV-only calibration ran on Kaggle CPU (Qwen/Qwen2.5-0.5B-Instruct,
frozen). Four submission attempts were required; the history is preserved:

| Attempt | Source commit | Outcome | Classification |
|---|---|---|---|
| v1 | `0d628118` | Failed: offline model resolution failure | Infrastructure |
| v2 | `4b737809` | Failed: source-path/provenance failure plus feature explosion | Infrastructure |
| v3 | `bd1ead9a` | Execution succeeded (exit 0) but artifacts written to a relative path, not rooted in `/kaggle/working` | Infrastructure (collection) |
| v4 | `bd1ead9a` | Succeeded and collected (exit 0, artifacts in `/kaggle/working/`) | — |

v1 and v2 are infrastructure failures (no manifest produced). v3 is an
infrastructure collection failure (execution succeeded but artifacts not
rooted in `/kaggle/working`, superseded by v4). v4 succeeded at both
execution and collection. The scientific DEV articulation gate is independent
of infrastructure success.

**v4 gate metrics** (DEV-only, `holdout_accessed=false`):

- `parameter_total=60388` / 64000 budget (W_perceive 14336, W_coupler 43008,
  b_coupler 2688, W_label 320, b_label 20, warm_state 16)
- `dev_repeatability_std=0.0`
- `dev_confidence_mean=0.0525482`, std `0.0002893`, range
  `0.0520694`–`0.0528929`
- `dev_specificity_acc=0.134328`
- `oracle_ceiling=0.357143` (upper bound on DEV with frozen organ)
- `raw_traces_deleted=true`, `raw_trace_count=0`
- `signoff_dev_articulation_gate=false`

**Scientific verdict: BLOCKED at the pre-registered DEV articulation gate.**
The v4 infrastructure is fully successful (exit 0, artifacts collected,
manifest hash `77ef4607…`, source archive SHA
`1afe7573…`), but `signoff_dev_articulation_gate=false` means the
pre-registered DEV gate is not passed. No signoff request was made. No
holdout access was attempted (`holdout_accessed=false`). No scientific
verdict beyond BLOCKED is permitted.

The `dev_confidence_mean` (0.0525482) and `dev_specificity_acc` (0.134328)
are recorded as observed DEV metrics, not as passed thresholds. The
`oracle_ceiling` (0.357143) is an upper bound on DEV performance with the
frozen language organ, not a claimed result. The proposed confidence
threshold and specificity margin are calibration proposals, not accepted
thresholds.

**Next mechanism-level direction:** the DEV confidence and specificity
distributions are now measured; the articulation coupler and label phrasing
are frozen on DEV. The gate failure points to the latent-control interface
(Arm B) not yet producing DEV articulation that clears the pre-registered
gate. The oracle ceiling (0.357143) bounds what the frozen organ can express
on these DEV tasks. R19 signed evaluation remains gated on the DEV
articulation gate passing; no signoff request is appropriate until it does.
R20 remains separately blocked for lack of explicit human signoff. No
threshold, metric, baseline, episode, scoring, eval manifest, or research
spec was changed.

Evidence: [`experiments_logs/2026-07-12_r19_dev_calibration.json`](experiments_logs/2026-07-12_r19_dev_calibration.json)

### R20 DEV-only smoke (2026-07-12, source `e26d8291879d`)

Research/20's DEV-only implementation is complete under
[`src/oczy/experiments/meta_cortex/`](src/oczy/experiments/meta_cortex/).
The package exposes exactly three CLI commands — `train-dev`,
`validate-dev`, `audit-dev` — and no `evaluate`, `meta-test`,
`run-meta-test`, `materialize`, `freeze`, `signoff`, `manifest`, `C7`,
or `C8` command. The `DevSplit` enum has `meta_train` and
`meta_validation` only; there is intentionally no test member. Parser
help labels every command "DEV only / not a scientific meta-test."

Three remote submission attempts were required; the history is preserved:

| Attempt | Source commit | Outcome | Classification |
|---|---|---|---|
| v1 | `e77314b` | Failed: offline model loader could not resolve Qwen | Infrastructure |
| v2 | `e38f91d` | Failed: inference-tensor/autograd mismatch in feature path | Infrastructure |
| v3 | `e26d8291879d` | Succeeded (exit 0, audit_status ok) | — |

v1 and v2 are infrastructure failures (no DEV smoke produced). v3
succeeded after fixing the offline loader and the feature-tensor
autograd path. The v3 DEV smoke ran on Kaggle CPU (kernel
`abdellahkadem/oczy-r20-dev-v3-e26d8291879d`, source archive SHA
`686c3b6a3de6e093f3646a3cdea6d0097d5de49cc6ef7231e262cf08643d99d5`).

**v3 audit results** (infrastructure/mechanism smoke only — not a
scientific result):

- exit 0, `audit_status=ok`
- frozen organ hash identical before and after training
  (`d8a3a3b262b3397f8948f13da10d3394e1a36b98a2ea374dc8711333d8d2b278`)
- 207,364 theta parameters / 829,456 bytes
- F/S matrices 64×64; latent bank 3×896
- optimizer steps: 1
- checkpoint theta hash
  `8d6c41c5dacbf31394e381dbdb5d6b8e496565bf14c2dedbbaa36f4987301d17`
- best DEV validation score: 0.0 after one step
- trace count: 0 after deletion
- online optimizer counts unchanged
- causal DEV deltas: trained-vs-update 0, untrained 0, shuffled 0,
  zeroed 0, swapped 0.0666667

**This is infrastructure/mechanism smoke only.** A best DEV validation
score of 0.0 after one optimizer step is not evidence of learning; the
causal deltas are mechanism checks, not scientific results. No ACCEPT
or REFUTE verdict is issued. The meta-test remains **BLOCKED**.

**Meta-test prerequisites reset by the INT8 v2 cutover.** The following must
be regenerated and signed off before any meta-test run:

1. a frozen `meta_cortex/v2` instrument bound to the INT8 organ hash
2. five fresh developmental checkpoints trained through that organ
3. fresh DEV repeatability distributions and power analysis
4. a hash-checked v2 candidate manifest with leakage audit
5. explicit human signoff on the exact manifest hash, margin, and task count

No meta-test signoff has been requested or granted. FP32 v1 checkpoints and
calibration shards are incompatible with v2 by organ identity and hash and
must not be merged into the new campaign.

### R20 INT8 organ cutover (2026-07-15)

The production `QwenFrozenOrgan` now loads the same
`Qwen/Qwen2.5-0.5B-Instruct` source artifact and applies TorchAO v2 per-row
INT8 weight-only quantization (`W8A32`) before any feature, teacher-forced, or
generation call. Dynamic-activation A8W8 was rejected because it detached the
forward result; W8A32 preserved finite, nonzero gradients to the soft bank.

The differentiable teacher-forced path uses deterministic activation
checkpointing. Without recomputation, one outer episode retains multiple full
Qwen graphs until a single backward and exits 137 under workstation memory
pressure. Evaluation and inference forwards remain direct.

The cutover creates `meta_cortex/v2` and `oczy/runtime-manifest/v2`. Runtime
identity now pins `torchao`, the exact quantization block, and canonical bytes
for TorchAO tensor subclasses. Local cached-model checks passed for load/hash,
independent-load hash determinism, final-layer feature extraction,
teacher-forced soft-bank backpropagation, and scalar/batched greedy-generation
parity.

A complete one-step DEV `train-dev` smoke then passed in 1379.14 seconds:
optimizer steps 1, `audit_status=ok`, peak observed RSS about 4.2 GB, checkpoint
and result written, and frozen organ hash identical before/after at
`60de9f75e8ae1d2507429877b4b2da48ec64c3e28eaad03db23cd3de43a1b4da`.
Best DEV validation score was 0.0. This is a mechanism check only; no
ACCEPT/REFUTE claim is made.


## 5. Current research direction

The July 9 conversation and repository audit produced a dependency-ordered
sequence. Research/18 and /19 remain useful diagnostics; Research/20 is now
the core premise test; Research/21 is conditional on it. Research/22 adds a
pending standalone LoRA-EPM addressability comparator in Stage A and keeps
Stage B cortex integration conditional on Research/20 acceptance.

| Order | Work | Role | Current state |
|---|---|---|---|
| 1 | [`research/18-consolidation-as-distillation.md`](research/18-consolidation-as-distillation.md) | Plastic-LM-weight comparator: distill transient context into LoRA, delete traces, test survival | **BLOCKED** (teacher gate failed): 5-seed run complete (exit 0), `teacher_dev_delta=0.1765` < 0.2 gate all seeds; `distill_delta_holdout` mean=0.2667, 4/5 positive (seed 2 null); no H-DISTILL verdict permitted. Mechanism diagnosis complete (2026-07-12, commit `33169cc`): teacher ceiling vanilla=0/raw_prefix=0.1765/chat_template=0 (none reach gate), prompt-contract audit all-zero (no structural defect), trajectory underfit+unstable (loss falls but DEV behavior weak); seed 2 not uniquely divergent. Further identical R18 reruns retired; next work is R19 DEV calibration; no threshold changes |
| 2 | [`research/19-lm-as-language-organ.md`](research/19-lm-as-language-organ.md) | Direct diagnostic with matched label-prefix and latent-control articulation arms | Implemented; DEV calibration **BLOCKED** at pre-registered DEV articulation gate (2026-07-12, source `bd1ead9a`): v4 infrastructure succeeded (exit 0, artifacts collected), but `signoff_dev_articulation_gate=false`; `holdout_accessed=false`, no signoff requested. `parameter_total=60388/64000`, `dev_confidence_mean=0.0525482`, `dev_specificity_acc=0.134328`, `oracle_ceiling=0.357143`, `raw_trace_count=0`. No scientific verdict beyond BLOCKED. R20 remains separately blocked. |
| 3 | [`research/20-meta-trained-cortex-frozen-language-organ.md`](research/20-meta-trained-cortex-frozen-language-organ.md) | Core test: meta-learn write/read/consolidate/articulate, then learn an unseen rule online through state only | **INT8 v2 DEV implementation complete and one-step smoke-verified** (2026-07-15); meta-test remains **BLOCKED** — v2 checkpoints, calibration distributions, power analysis, candidate manifest, and human signoff must be regenerated |
| 4 | [`research/21-cortex-routed-frozen-specialist-organs.md`](research/21-cortex-routed-frozen-specialist-organs.md) | Conditional extension: cortex routes between frozen language and action/tool organs using recurrent goal state | New specification; do not start before Research/20 accepts |
| 5 | [`research/22-parametric-memory-decoding-zero-shot-lora-routing.md`](research/22-parametric-memory-decoding-zero-shot-lora-routing.md) | Stage A: test zero-shot PMD addressability over a shared frozen backbone and LoRA EPM bank; Stage B: later cortex integration | **Specification only / PENDING**; Stage A is independent of Research/20, Stage B remains **BLOCKED** on Research/20 acceptance, retrieval is mandatory, no implementation exists, and no scientific claim is made |

[`experiments/09-meta-trained-cortex-frozen-language-organ/README.md`](experiments/09-meta-trained-cortex-frozen-language-organ/README.md)
operationalizes Research/20. Its v1 design fixes a 64-dimensional cortex,
fast and slow 64×64 matrices, a learned outer-product writer, a learned
consolidation gate, query-conditioned reads, and a fixed-width soft latent bank
into the frozen Qwen language organ. Outer-loop development spans contextual
remapping, rule transformation, and finite-state behavior. Meta-test freezes
all parameters and permits only cortex fast/slow state to change.

The DEV-only implementation is complete under
[`src/oczy/experiments/meta_cortex/`](src/oczy/experiments/meta_cortex/) with
three CLI commands — `train-dev`, `validate-dev`, `audit-dev` — and no
`evaluate`, `meta-test`, `run-meta-test`, `materialize`, `freeze`, `signoff`,
`manifest`, `C7`, or `C8` command. The `DevSplit` enum has `meta_train` and
`meta_validation` only; there is intentionally no test member. Parser help
labels every command "DEV only / not a scientific meta-test."

**The R20 meta-test requires explicit human sign-off before any meta-test
run begins and MUST NOT run without it.** No signoff has been requested or
granted. The next legitimate step is to propose and freeze the
`meta_cortex/v1` measuring instrument (task generator, train/dev/test split,
manifest, leakage audit, threshold distributions, power analysis) and obtain
human signoff — not to run the meta-test.

The primary meta-test must exclude retrieval, online backpropagation, raw
trace replay, answer/label text, correction text, and language-model weight
updates. Zeroing, swapping, shuffling, and feedback-semantic controls must show
that behavior is caused by the learned cortex state rather than an unnoticed
side channel.

## 6. What exists in code right now

| Surface | Implementation state | Location |
|---|---|---|
| Frozen curriculum eval v2.2 | Implemented; manifest verified after human-approved protocol repair | [`eval/v2/`](eval/v2/) |
| Eval guard/version tooling | Implemented | [`scripts/eval_guard.py`](scripts/eval_guard.py), [`scripts/bump_eval_version.py`](scripts/bump_eval_version.py) |
| Original organism and organs | Implemented; most answer-time organs archived by evidence, not deleted | [`src/oczy/`](src/oczy/) |
| HF driver and S1 probes | Implemented and run | [`src/oczy/lm/hf_driver.py`](src/oczy/lm/hf_driver.py), [`src/oczy/experiments/hf_kv_slot_experiment.py`](src/oczy/experiments/hf_kv_slot_experiment.py), [`src/oczy/experiments/hf_layer_probe.py`](src/oczy/experiments/hf_layer_probe.py) |
| Minimal-loop/forgetting harnesses | Implemented; primary loop refuted, gated successors blocked | [`src/oczy/experiments/minimal_loop.py`](src/oczy/experiments/minimal_loop.py), [`src/oczy/experiments/minimal_loop_forgetting.py`](src/oczy/experiments/minimal_loop_forgetting.py) |
| Organ ablations | Implemented and adjudicated | [`src/oczy/experiments/organ_ablation.py`](src/oczy/experiments/organ_ablation.py), [`src/oczy/experiments/organ_additive_retrieval.py`](src/oczy/experiments/organ_additive_retrieval.py) |
| Research/19 direct cortex | Implemented; DEV calibration complete and BLOCKED at DEV articulation gate | [`src/oczy/experiments/s19_language_organ.py`](src/oczy/experiments/s19_language_organ.py); evidence [`experiments_logs/2026-07-12_r19_dev_calibration.json`](experiments_logs/2026-07-12_r19_dev_calibration.json) |
| Research/20 / Experiment 09 | DEV-only implementation complete; smoke-verified on Kaggle CPU (2026-07-12); meta-test blocked (no frozen instrument/signoff) | [`src/oczy/experiments/meta_cortex/`](src/oczy/experiments/meta_cortex/) — `model.py`, `organ.py`, `training.py`, `contracts.py`, `taskgen.py`, `artifacts.py`, `cli.py`, `__main__.py`; tests in `src/oczy/experiments/tests/test_meta_cortex_*.py` |
| Research/21 multi-organ router | Specification only | No implementation module yet |
| Research/22 zero-shot LoRA EPM routing | Specification only / PENDING; no implementation and no scientific claim | [`research/22-parametric-memory-decoding-zero-shot-lora-routing.md`](research/22-parametric-memory-decoding-zero-shot-lora-routing.md) — Stage A independently tests PMD addressability with mandatory retrieval; Stage B cortex integration remains blocked on Research/20 acceptance |
| Remote compute pool | Mixed Kaggle/Colab CPU; Kaggle verified v4 smoke/probe/bootstrap; Colab CLI 0.6.0 verified v2 queue-starvation fix; GPU (T4/P100/L4) archived; TPU not wired | [`infrastructure/kaggle/`](infrastructure/kaggle/) |
| Pi tool-use work / Experiment 08 | Code-backed 6-stage dataset/scorer/validator implemented; live augmented run still pending; external result remains 0/3 | [`src/oczy/experiments/tool_calling_curriculum/`](src/oczy/experiments/tool_calling_curriculum/) plus [`benchmarks/pi/`](benchmarks/pi/) |
| Dashboard | Generator exists; canonical output absent | [`scripts/dashboard.py`](scripts/dashboard.py); planned `experiments_logs/DASHBOARD.md` |
| Weekly external battery | Research spec exists; runner absent | [`research/16-s4-external-benchmark-battery.md`](research/16-s4-external-benchmark-battery.md); planned `scripts/weekly_battery.sh` |
| Archived-code move | Not done | Planned `attic/` directory is absent |

### Remote compute pool — mixed Kaggle/Colab CPU 2026-07-10/11

The remote compute pool now supports two CPU-only providers:

| Provider | Status | Since | Key detail |
|---|---|---|---|
| Kaggle CLI 2.2.3 | Active, verified | 2026-07-10 | `cpu-smoke` v4, `qwen-cpu-probe` v1, `cpu-bootstrap-probe` v4 |
| Colab CLI 0.6.0 | Active, verified | 2026-07-11 | CPU sessions, dynamic AIMD admission, queue-starvation fix |
| GPU (T4/P100/L4) | Archived --- do not submit | 2026-07-09 | Historical evidence under `infrastructure/kaggle/archive/gpu/` |

**Kaggle verification (2026-07-10):** The `cpu-smoke` kernel
(`abdellahkadem/oczy-cortex-cpu-smoke`) was re-verified remotely (v4): it ran
the 64x64 cortex / width-896 frozen-organ interface workload on a Kaggle x86_64
CPU, passed finite-gradient, held-out-improvement, and frozen-parameter hash
checks, and reported `cuda_available: false`. The `qwen-cpu-probe` kernel
(`abdellahkadem/oczy-qwen-cpu-probe`) completed remotely with `passed: true`
(v1). The `cpu-bootstrap-probe` kernel
(`abdellahkadem/oczy-cpu-bootstrap-probe`) completed remotely (v4), verifying
the full generated-job pipeline.

**Colab verification (2026-07-11):** The Colab CLI 0.6.0 was installed and
authenticated. Safe allocation probes confirmed CPU session creation, execution,
and cleanup. A final live batch of four 30-second scripts ran through the
scheduler: first three concurrent on first attempt, fourth capacity-rejected
(412) then retried after slot freed, all four succeeded (attempts 1/1/1/2).
Each VM reported `cpu_count=2`. The `learned_limit` ended at 4 due to additive
probing and will self-correct on the next 412; do not claim capacity is
hardcoded at 3.

**Scheduler upgrade (2026-07-11):** The parallel scheduler was upgraded to
support mixed providers. v2 batch schema (`oczy/remote-parallel-batch/v2`) with
per-job ``provider`` field (``kaggle``/``colab``), v2 state schema
(``oczy/remote-parallel-state/v2``) with ``colab_learned_limit`` and
provider-neutral job records. New CLI flags: ``--kaggle-max``, ``--colab-max``,
``--colab-cooldown``. A queue-starvation bug (admission gate incrementing
``capacity_rejections`` on capacity blocks, causing false failure after 10
blocks) was fixed during the same session.

**Default scheduler capacity (2026-07-11):** When ``--max-parallel`` is
omitted (the default), the scheduler imposes no global concurrency cap —
capacity is **additive: 5 Kaggle + learned Colab X**. Kaggle jobs fill up
to ``--kaggle-max`` (default 5, hard-capped at 5); Colab jobs fill up to
an AIMD-learned limit that starts at 1 and probes upward, capped by
``--colab-max`` (default 10). The learned Colab limit is not a hardcoded
quota — it adapts to account-level session availability. Explicit
``--max-parallel N`` caps total concurrency globally for backward
compatibility.

**Campaign 0d48130 (2026-07-11):** The mixed-provider scheduler completed its
first full research campaign. 10 run groups were submitted across Kaggle
(CPU-only) and Colab (CPU-only) from three source commits: 9 completed
with scientific verdicts (including nulls, refutations, and one metricless
R14 result) and 1 (Exp03) was infrastructure-blocked by HF snapshot
transfer failures. See section 4 above and
[`experiments_logs/2026-07-11_campaign_0d48130.md`](experiments_logs/2026-07-11_campaign_0d48130.md)
for the full adjudication.

The exact official Qwen source
`qwen-lm/qwen2.5/transformers/0.5b-instruct/1` remains version-pinned for all
model-bearing CPU jobs. See
[`infrastructure/kaggle/RESULTS.md`](infrastructure/kaggle/RESULTS.md) for the
full evidence and acceptance contract, and
[`infrastructure/kaggle/RESEARCH_GUIDE.md`](infrastructure/kaggle/RESEARCH_GUIDE.md)
for the required workflow.

The `meta_cortex` DEV-only module is implemented and the DEV smoke has
run on Kaggle CPU (see the R20 DEV smoke subsection in §4). No meta-test
job can be generated or run until the `meta_cortex/v1` measuring instrument
is frozen and human signoff is recorded.

- Branch: `autoresearch/session-20260625`.
- The historical local branch was fast-forward published to
  `origin/autoresearch/session-20260625` without force on 2026-07-09.
- Research/19–21, Experiment 09, and their roadmap/index updates were published
  in commit `f48dccc` (`research: define meta-trained cortex program`).
- The verified Kaggle CPU/T4/Qwen workflow, standing guidance, generators, and
  tests were published in commit `6dee16b`
  (`infra: add guarded Kaggle research compute workflow`).
- The working tree is **clean** at commit `4f1a022` on
  `autoresearch/session-20260625`. The previously preserved Pi work
  (`GOALS.md`, Pi model/proxy changes, the Pi runner and logs, Experiment
  08, and its experiment-index additions) is tracked in the repository and
  no longer pending as dirty working-tree state.
- No force push, production deployment, eval change, or Pi publication was
  performed. The private Kaggle verification kernels remain the only external
  compute mutations.

Because the branch contains a large protected research history, a default
guard comparison against the remote base sees historical approved changes.
For commit `f48dccc`, the explicitly authorized scoped guard passed with
`EVAL_CHANGE_APPROVED=1`, and no `eval/v2` file changed.

## 8. Pending work, in priority order

### P0 — Repair evidence and preserve the current work

1. **Correct S2.4 provenance.** Run the fixed real-driver ablation with multiple
   seeds and an explicit new output path. Never overwrite the 2026-07-01 file;
   add a dated correction under [`experiments_logs/`](experiments_logs/) and
   update [`experiments_logs/LEDGER.md`](experiments_logs/LEDGER.md).
2. **Adjudicate the preserved Pi work separately.** Review the local proxy,
   benchmark runner/logs, Experiment 08 spec, `GOALS.md`, and index changes as
   one independent scope. Validate them before a separate commit; do not fold
   them retroactively into the published cortex/compute commits.

3. **Exp03 reproducibility closure — COMPLETE (2026-07-11).** The original
   campaign Exp03 run was infrastructure-blocked (HF snapshot transfer
   failures) and produced no scientific verdict; that history is preserved.
   A follow-up real-driver rerun (commit `ad77e93`, `--driver real`, Colab)
   closed the reproducibility gap: exit 0,
   `layer_l_silhouette_gap=0.10925446726657728` exceeds the registered
   +0.10 threshold (unchanged), so this single run is positive/accept for
   the closure. The pre-registered S1.4 refutation (two architectures) is
   not reopened or overturned. **Acceptance met:** durable execution report
   at [`experiments_logs/2026-07-11_exp03_real_driver_closure.json`](experiments_logs/2026-07-11_exp03_real_driver_closure.json);
   S1.4 is not reopened.
4. **R18 five-seed diagnostic — COMPLETE (2026-07-11); scientifically
   BLOCKED at teacher gate.** The 5-seed `stage_0` rerun (Kaggle CPU,
   kernel `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, source commit
   `5b5e93c63d769fea7854073a4e6c359e5d36606f`) completed with exit 0.
   All metrics collected. **Scientific verdict: BLOCKED — diagnostic
   only.** The unchanged teacher gate (`teacher_dev_delta ≥ 0.2`)
   failed: every seed observed `teacher_dev_delta=0.17647058823529413`
   < 0.2. No H-DISTILL verdict is permitted because the teacher gate
   failed after registered fallback. Per-seed
   `distill_delta_holdout`: {0.3333, 0.3333, 0.0, 0.3333, 0.3333} —
   4/5 positive, seed 2 null (preserved). Mean
   `distill_delta_holdout=0.26666666666666666`; mean
   `specificity_delta=0.02608695652173913`. The 4/5 positive holdout
   deltas are infrastructure-confirmed but scientifically inadmissible.
   **Mechanism diagnosis COMPLETE (2026-07-12, commit `33169cc`):**
   teacher ceiling (n=17) vanilla=0, raw_prefix=0.17647058823529413,
   chat_template=0 — none reach the 0.2 gate; registered chat fallback
   is worse than raw_prefix. Prompt-contract audit: all
   issue/malformed/missing/truncated/answer-leak/mismatch counts 0;
   no structural prompt defect. Training trajectory: loss falls
   ~0.70→~0.16, mean slope -0.0615, second-half -0.0190, underfit=1,
   instability=1, saturation=0, max final-loss divergence 0.01259;
   optimization fits token loss but DEV behavior is unstable/weak and
   not saturated. Final DEV student accuracies (seeds 0–4) =
   {0.117647, 0, 0, 0, 0.117647}; seed 2 is not uniquely divergent.
   Conclusion: the blocker is teacher expressivity/prompt-task ceiling,
   not a prompt bug. Further identical R18 reruns are retired — they
   will not clear the unchanged teacher gate. Next work is R19 DEV
   calibration; R19 signed evaluation remains gated on human approval,
   and the Research/20 meta-test remains separately blocked. No
   threshold, metric, or eval changes. **Acceptance met:** 5-seed run
   complete with per-seed values; mechanism diagnosis
   complete; null seed 2 preserved; no threshold change.
5. **S4.1: complete honest reruns.** Re-run every June 26–29 result that
   depended on the broken scope-slot reranker or leakage-era paths on eval
   v2. **Acceptance:** every INVALIDATED/SUPERSEDED ledger row points to a
   dated honest rerun log or is labeled "no longer relevant."

### P1 — Run the cheap interface diagnostic — DEV calibration COMPLETE; BLOCKED at DEV articulation gate

Research/19 is implemented as a matched two-arm test under
[`src/oczy/experiments/s19_language_organ.py`](src/oczy/experiments/s19_language_organ.py).
Arm A: online-trained cortex decoded to a label prefix (parametric retrieval).
Arm B: the same cortex state read through a fixed-width learned latent coupler
into the frozen LM, with no label text. Zero-state, swapped-state,
shuffled/permuted-feedback, vanilla, retrieval, and oracle conditions are
specified.

The DEV-only calibration ran on Kaggle CPU (2026-07-12, source `bd1ead9a`).
Four submission attempts: v1 failed (offline model resolution), v2 failed
(source-path/provenance failure plus feature explosion), v3 succeeded but
artifacts not rooted in `/kaggle/working`, v4 succeeded and collected. The
v4 infrastructure is fully successful (exit 0, artifacts collected, manifest
hash `77ef4607…`, source archive SHA `1afe7573…`), but the pre-registered
DEV articulation gate is not passed
(`signoff_dev_articulation_gate=false`). No signoff request was made; no
holdout access was attempted (`holdout_accessed=false`). `parameter_total=60388/64000`,
`dev_confidence_mean=0.0525482`, `dev_specificity_acc=0.134328`,
`oracle_ceiling=0.357143`, `raw_trace_count=0`. No scientific verdict beyond
BLOCKED is permitted. The DEV confidence and specificity distributions are
measured; the coupler and label phrasing are frozen on DEV. Next
mechanism-level direction: the gate failure points to the Arm B latent-control
interface not yet producing DEV articulation that clears the pre-registered
gate; the oracle ceiling (0.357143) bounds what the frozen organ can express
on these DEV tasks. R19 signed evaluation remains gated on the DEV
articulation gate passing. R20 remains separately blocked for lack of
explicit human signoff. No threshold, metric, baseline, episode, scoring, eval
manifest, or research spec was changed. Evidence:
[`experiments_logs/2026-07-12_r19_dev_calibration.json`](experiments_logs/2026-07-12_r19_dev_calibration.json).

### P1b — Build the S5.3 diagnostic head-to-head table

One table: R18 (consolidation-as-distillation) vs both R19 arms vs
retrieval-baseline vs vanilla, with deletion audits, CIs, per-byte
accounting, and explicit classification of every answer path.
**Acceptance:** `experiments_logs/DASHBOARD.md` or a dated log contains the
table with all columns filled and every path classified as retrieval,
metabolism, or vanilla.

### P2 — Build and adjudicate the core cortex experiment

The DEV-only implementation of
[`experiments/09-meta-trained-cortex-frozen-language-organ/`](experiments/09-meta-trained-cortex-frozen-language-organ/)
is complete under [`src/oczy/experiments/meta_cortex/`](src/oczy/experiments/meta_cortex/).
The package exposes `train-dev`, `validate-dev`, and `audit-dev` only —
no meta-test command. The DEV smoke ran on Kaggle CPU (2026-07-12,
source `e26d8291879d`, exit 0, audit_status ok; see §4 R20 DEV-only
smoke). This is infrastructure/mechanism smoke only; no ACCEPT or
REFUTE verdict is issued.

The meta-test remains **BLOCKED**. The following do not exist and must
be built and signed off before any meta-test run:

1. a frozen `meta_cortex/v1` measuring instrument (separate task
   generator, task-level train/dev/test split)
2. a hash-checked manifest with leakage audit
3. threshold distributions measured against real data
4. power analysis
5. explicit human sign-off

No signoff has been requested or granted. The next legitimate step is
to propose and freeze the instrument and obtain human signoff — not to
run the meta-test.

After signoff, the remaining adjudication order is:

1. Pass the oracle latent-articulation gate. If the frozen mouth cannot express
   the task under oracle control, repair or kill the interface before training
   the cortex.
2. Meta-train write, read, consolidation, and latent articulation across the
   development task families. Use the verified Kaggle CPU path for instrument,
   scoring, and frozen-organ/outer-loop batches; pin
   a clean source artifact with
   [`prepare_source_bundle.py`](infrastructure/kaggle/prepare_source_bundle.py)
   and generate the job with
   [`prepare_research_kernel.py`](infrastructure/kaggle/prepare_research_kernel.py)
   using `--profile cpu`. The pinned Qwen model source has passed its local
   CPU frozen-gradient probe and was verified remotely (v1, 2026-07-10).
3. Freeze all learned parameters and run one-shot held-out meta-test with only
   fast/slow cortex state mutable.
4. Run all causal state and deletion controls, multiple seeds, trajectories,
   confidence intervals, and per-byte accounting.

Research/20 must stand or fall without retrieval rescuing its primary arm.

### P3 — Add specialist action organs only after P2 succeeds

If and only if Research/20 accepts:

1. Select and hash-freeze a small structured-action/tool organ.
2. Implement the Research/21 router, organ-specific latent couplers, and
   recurrent goal state.
3. Train on opaque tool names and task families, then test held-out tool
   semantics and multi-turn chains.
4. Use the existing Pi work as an external battery after the controlled
   instrument passes; do not turn Pi into the primary measuring instrument.

### P4 — Complete external validation and research hygiene

- Implement `scripts/weekly_battery.sh` from Research/16.
- Generate and maintain `experiments_logs/DASHBOARD.md` using
  [`scripts/dashboard.py`](scripts/dashboard.py).
- Move evidence-archived answer-time organs to a documented `attic/` only after
  imports and historical reproduction paths are mapped.
- Run the registered DSI stage-1 appeal or label it permanently closed.
- M2b ran as a metricless NULL (Campaign 0d48130: 3 seeds, exit 0, no
  `METRIC`/`ASI` values). It cannot retroactively turn an archived organ
  into evidence for the learned-cortex premise; re-run only if it answers a
  still-relevant question.
- **Durable live watch queue — ACTIVE (2026-07-11); first job
  COMPLETE.** Watch mode for `parallel_scheduler.py` is
  **implemented and tested**: atomically reload a changed batch,
  merge only unseen job names as pending, never mutate existing job
  definitions or states, retry malformed reloads without killing the
  daemon, and stay alive waiting for future jobs. Existing non-watch
  behavior remains terminating and backward compatible. The queue
  setup action is **complete**; the first experiment result is
  **complete and adjudicated**. Live queue paths: batch
  `/tmp/oczy-live-queue/batch.json`, state
  `/tmp/oczy-live-queue/state.json`, campaign
  `/tmp/oczy-live-queue-campaign.json`. Source commit:
  `5b5e93c63d769fea7854073a4e6c359e5d36606f`. Capacity is
  **additive: 5 Kaggle + learned Colab X** (same defaults as the
  scheduler upgrade above). The background scheduler runs with
  `--watch-batch --watch-interval 30`. **First job:**
  `r18-distillation-5seed-diagnostic` (Kaggle, kernel
  `abdellahkadem/oczy-r18-5seed-5b5e93c63d76`, pinned source dataset
  `abdellahkadem/oczy-source-5b5e93c63d76`, source archive sha256
  `bc1ff926bc679fc26e5f20cfcb0756339b002ff3c1027eb3c24251fe2f6d7f72`,
  module `oczy.experiments.consolidation_distillation`, args
  `--seeds 5 --max-steps 10 --stage stage_0_grounding`).
  **Infrastructure: COMPLETE** (exit 0, all metrics collected).
  **Scientific verdict: BLOCKED at the teacher validity gate —
  diagnostic only.** The unchanged teacher gate
  (`teacher_dev_delta ≥ 0.2`) failed: every seed observed
  `teacher_dev_delta=0.17647058823529413` < 0.2. No H-DISTILL
  verdict is permitted because the teacher gate failed after
  registered fallback. Per-seed `distill_delta_holdout`: {0.3333,
  0.3333, 0.0, 0.3333, 0.3333} — 4/5 positive, seed 2 null
  (preserved). Mean `distill_delta_holdout=0.26666666666666666`;
  mean `specificity_delta=0.02608695652173913`. No threshold,
  metric, or eval change is implied. The Research/20 meta-test
  sign-off prohibition is unchanged.

### P5 — Expand an eval only by the governance path

Some current stages remain small or weak, especially dialog and the external
behavior surfaces. Any change to the frozen eval requires a version bump,
human sign-off, a recomputed manifest, and a distribution check. The optimizing
loop must never edit its measuring instrument.

## 9. Decision gates

The next honest project-level verdict belongs to Research/20 / Experiment 09:

- **Accept the direction** only if held-out online learning beats vanilla and
  frozen/random/no-consolidation controls, survives trace deletion, transfers
  across task families, and is causally moved by cortex-state zero/swap/shuffle
  interventions.
- **Diagnose the interface** if an oracle latent bank fails. That is a mouth–
  cortex protocol failure, not evidence about online learning.
- **Diagnose the learner** if the oracle works but the meta-trained cortex does
  not acquire held-out rules.
- **Refute v1 cleanly** if thresholds fail across the pre-registered seeds and
  task families. Do not add retrieval, labels, or episode-specific fixes to
  make the primary condition pass.
- **Start Research/21** only after Research/20 passes. More organs before a
  causal core loop would repeat the earlier breadth-first failure.

## 10. Updating this document

When the state changes:

1. update the timestamp and repository-state section;
2. distinguish specification, implementation, run, and adjudicated result;
3. add new evidence to `experiments_logs/` and classify it in the ledger;
4. preserve nulls and refutations as prominently as wins;
5. never silently edit a frozen evaluation or overwrite a historical log.
