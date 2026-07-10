# Oczy — Current Project State

**Last updated:** 2026-07-09 22:03:41 +01:00 (Africa/Casablanca)

**Evidence cutoff:** repository and local experiment artifacts inspected through
2026-07-09

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
  **v2.1**;
- removal of episode-ID and expected-answer leakage;
- held-out splits, vanilla and retrieval baselines, confidence intervals,
  trajectories, seed requirements, and pre-registered acceptance/kill rules;
- a Hugging Face driver and selection of frozen
  `Qwen2.5-0.5B-Instruct` for the fast experimental substrate;
- HF KV-splice and layer-hidden probes;
- minimal-loop, forgetting, organ-ablation, and additive-control harnesses;
- a validity ledger that labels nulls, refutations, invalidations, and
  superseded claims rather than hiding them.

The current full test verification performed during the July 9 audit was
**680 passed, 14 warnings**. The `eval/v2` manifest also verified. These are
code-integrity checks, not evidence that the scientific thesis passed.

## 4. Verified experimental picture

The table below separates useful mechanisms from evidence for the thesis.

| Test or claim | Current verdict | Important evidence |
|---|---|---|
| Honest curriculum baseline | Reference point, not a cortex win | Real-driver stages 0–5: **0.88, 0.75, 0.69, 0.38, 0.90, 0.92** |
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

## 5. Current research direction

The July 9 conversation and repository audit produced a dependency-ordered
sequence. Research/18 and /19 remain useful diagnostics; Research/20 is now
the core premise test; Research/21 is conditional on it.

| Order | Work | Role | Current state |
|---|---|---|---|
| 1 | [`research/18-consolidation-as-distillation.md`](research/18-consolidation-as-distillation.md) | Plastic-LM-weight comparator: distill transient context into LoRA, delete traces, test survival | Pre-registered; not run |
| 2 | [`research/19-lm-as-language-organ.md`](research/19-lm-as-language-organ.md) | Direct diagnostic with matched label-prefix and latent-control articulation arms | Amended 2026-07-09; not implemented |
| 3 | [`research/20-meta-trained-cortex-frozen-language-organ.md`](research/20-meta-trained-cortex-frozen-language-organ.md) | Core test: meta-learn write/read/consolidate/articulate, then learn an unseen rule online through state only | New specification; not implemented |
| 4 | [`research/21-cortex-routed-frozen-specialist-organs.md`](research/21-cortex-routed-frozen-specialist-organs.md) | Conditional extension: cortex routes between frozen language and action/tool organs using recurrent goal state | New specification; do not start before Research/20 accepts |

[`experiments/09-meta-trained-cortex-frozen-language-organ/README.md`](experiments/09-meta-trained-cortex-frozen-language-organ/README.md)
operationalizes Research/20. Its v1 design fixes a 64-dimensional cortex,
fast and slow 64×64 matrices, a learned outer-product writer, a learned
consolidation gate, query-conditioned reads, and a fixed-width soft latent bank
into the frozen Qwen language organ. Outer-loop development spans contextual
remapping, rule transformation, and finite-state behavior. Meta-test freezes
all parameters and permits only cortex fast/slow state to change.

The primary meta-test must exclude retrieval, online backpropagation, raw
trace replay, answer/label text, correction text, and language-model weight
updates. Zeroing, swapping, shuffling, and feedback-semantic controls must show
that behavior is caused by the learned cortex state rather than an unnoticed
side channel.

## 6. What exists in code right now

| Surface | Implementation state | Location |
|---|---|---|
| Frozen curriculum eval v2.1 | Implemented; manifest verified | [`eval/v2/`](eval/v2/) |
| Eval guard/version tooling | Implemented | [`scripts/eval_guard.py`](scripts/eval_guard.py), [`scripts/bump_eval_version.py`](scripts/bump_eval_version.py) |
| Original organism and organs | Implemented; most answer-time organs archived by evidence, not deleted | [`src/oczy/`](src/oczy/) |
| HF driver and S1 probes | Implemented and run | [`src/oczy/lm/hf_driver.py`](src/oczy/lm/hf_driver.py), [`src/oczy/experiments/hf_kv_slot_experiment.py`](src/oczy/experiments/hf_kv_slot_experiment.py), [`src/oczy/experiments/hf_layer_probe.py`](src/oczy/experiments/hf_layer_probe.py) |
| Minimal-loop/forgetting harnesses | Implemented; primary loop refuted, gated successors blocked | [`src/oczy/experiments/minimal_loop.py`](src/oczy/experiments/minimal_loop.py), [`src/oczy/experiments/minimal_loop_forgetting.py`](src/oczy/experiments/minimal_loop_forgetting.py) |
| Organ ablations | Implemented and adjudicated | [`src/oczy/experiments/organ_ablation.py`](src/oczy/experiments/organ_ablation.py), [`src/oczy/experiments/organ_additive_retrieval.py`](src/oczy/experiments/organ_additive_retrieval.py) |
| Research/19 direct cortex | Specification only | No dedicated implementation module yet |
| Research/20 / Experiment 09 | Specification only | Planned module: `src/oczy/experiments/meta_cortex/` — currently absent |
| Research/21 multi-organ router | Specification only | No implementation module yet |
| Kaggle offline compute | CPU-only profile active (`cpu-smoke` verified, `qwen-cpu-probe` local pass / remote pending); GPU (T4/P100/L4) archived under `infrastructure/kaggle/archive/gpu/`; TPU not wired | [`infrastructure/kaggle/`](infrastructure/kaggle/) |
| Pi tool-use work / Experiment 08 | Unpublished local spec, proxy, runner, and two JSON logs; core curriculum package absent; result 0/3 | Existing [`benchmarks/pi/`](benchmarks/pi/) plus local `experiments/08-oczy-pi-tool-calling-curriculum/` |
| Dashboard | Generator exists; canonical output absent | [`scripts/dashboard.py`](scripts/dashboard.py); planned `experiments_logs/DASHBOARD.md` |
| Weekly external battery | Research spec exists; runner absent | [`research/16-s4-external-benchmark-battery.md`](research/16-s4-external-benchmark-battery.md); planned `scripts/weekly_battery.sh` |
| Archived-code move | Not done | Planned `attic/` directory is absent |

### Remote offline compute — CPU-only cutover 2026-07-10

The Kaggle CLI is authenticated (version 2.2.3). The active remote profile is
**CPU only**. The `cpu-smoke` kernel
(`abdellahkadem/oczy-cortex-cpu-smoke`) was verified remotely on 2026-07-09:
it ran the 64×64 cortex / width-896 frozen-organ interface workload on a
Kaggle x86_64 CPU, passed finite-gradient, held-out-improvement, and
frozen-parameter hash checks, and reported `cuda_available: false`. The
`qwen-cpu-probe` kernel (`abdellahkadem/oczy-qwen-cpu-probe`) passes locally;
its remote acceptance is pending evidence from Main.

GPU verification (T4, P100, L4, and the T4-based Qwen model probe) from
2026-07-09 is preserved as historical evidence under
[`infrastructure/kaggle/archive/gpu/`](infrastructure/kaggle/archive/gpu/).
That material — including the 2×T4 throughput comparison and P100/L4
compatibility nulls — is not active and must not be resubmitted. See
[`infrastructure/kaggle/archive/gpu/RESULTS.md`](infrastructure/kaggle/archive/gpu/RESULTS.md)
for the full historical record.

The exact official Qwen source
`qwen-lm/qwen2.5/transformers/0.5b-instruct/1` remains version-pinned for all
model-bearing CPU jobs. The active `qwen-cpu-probe` task re-verifies the same
model hashes on CPU. See
[`infrastructure/kaggle/RESULTS.md`](infrastructure/kaggle/RESULTS.md) for the
current acceptance contract and
[`infrastructure/kaggle/RESEARCH_GUIDE.md`](infrastructure/kaggle/RESEARCH_GUIDE.md)
for the required CPU-only workflow. No real Research/20 job can be generated
from current work until the `meta_cortex` module is implemented and the
intended source is committed cleanly.

## 7. Repository state at this snapshot

- Branch: `autoresearch/session-20260625`.
- The historical local branch was fast-forward published to
  `origin/autoresearch/session-20260625` without force on 2026-07-09.
- Research/19–21, Experiment 09, and their roadmap/index updates were published
  in commit `f48dccc` (`research: define meta-trained cortex program`).
- The verified Kaggle CPU/T4/Qwen workflow, standing guidance, generators, and
  tests were published in commit `6dee16b`
  (`infra: add guarded Kaggle research compute workflow`).
- The working tree remains intentionally dirty only for separately scoped Pi
  work: `GOALS.md`, Pi model/proxy changes, the Pi runner and logs, Experiment
  08, and its unstaged experiment-index additions. Those files were preserved
  locally and excluded from the research and infrastructure commits.
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

### P1 — Run the cheap interface diagnostic

Implement Research/19 as a matched two-arm test:

- Arm A: online-trained cortex decoded to a label prefix. Keep it, but classify
  it as parametric retrieval.
- Arm B: the same cortex state read through a fixed-width learned latent
  coupler into the frozen LM, with no label text.
- Include zero-state, swapped-state, shuffled/permuted-feedback, vanilla,
  retrieval, and oracle conditions.
- Put code under a dedicated module in [`src/oczy/experiments/`](src/oczy/experiments/)
  and write results to a new dated file in [`experiments_logs/`](experiments_logs/).

### P2 — Build and adjudicate the core cortex experiment

Implement [`experiments/09-meta-trained-cortex-frozen-language-organ/`](experiments/09-meta-trained-cortex-frozen-language-organ/)
under the planned `src/oczy/experiments/meta_cortex/` package.

Required order:

1. Build the separate `meta_cortex/v1` task generator, task-level train/dev/test
   split, manifest, leakage audit, threshold distributions, and power analysis.
2. Obtain explicit human sign-off before freezing or changing that measuring
   instrument.
3. Pass the oracle latent-articulation gate. If the frozen mouth cannot express
   the task under oracle control, repair or kill the interface before training
   the cortex.
4. Meta-train write, read, consolidation, and latent articulation across the
   development task families. Use the verified Kaggle CPU path for instrument,
   scoring, and frozen-organ/outer-loop batches; pin
   a clean source artifact with
   [`prepare_source_bundle.py`](infrastructure/kaggle/prepare_source_bundle.py)
   and generate the job with
   [`prepare_research_kernel.py`](infrastructure/kaggle/prepare_research_kernel.py)
   using `--profile cpu`. The pinned Qwen model source has passed its local
   CPU frozen-gradient probe; remote acceptance is pending.
5. Freeze all learned parameters and run one-shot held-out meta-test with only
   fast/slow cortex state mutable.
6. Run all causal state and deletion controls, multiple seeds, trajectories,
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
- Run M2b only if it answers a still-relevant question; it cannot retroactively
  turn an archived organ into evidence for the learned-cortex premise.

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
