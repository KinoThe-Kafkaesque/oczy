# Oczy Research Agenda

A prioritized set of research projects for the **Plastic World Model Agent**
(`../experiments.txt`). Each project is a problem, falsifiable hypothesis,
approach, and discriminating success criterion. The original projects 01–07
have matching designs under [`../experiments/`](../experiments/); later numbered
specs are pre-registered remediation and successor experiments, with concrete
experiment directories linked where they exist.

> The thesis in one line: *experience → fast change → replay → compression →
> slow change → forgetting raw trace.* "An agent should not store experiences;
> it should metabolize them." The missing art is the metabolism.

## The organizing problem: the eval went blind

The single fact that shapes this whole agenda is that **the headline benchmarks
are saturated**. Across recent runs the git log repeatedly reports *"All
benchmark metrics at 1.0"*; `autoresearch.sh` prints `code_qa_accuracy=1.0`;
and architecture variants that are *mechanically* different (e.g. scalar vs
hybrid consolidation — `consolidation_strength` 10.0 vs 36.0, `cold_drift` 0.10
vs 0.64) tie on **every** behavioral metric. When the measuring instrument can
no longer separate two architectures, no architecture work can be honestly
claimed to win.

So the agenda is built as a spine and ribs:

- **The spine** is project **01** — rebuild the eval so a behavior difference
  is the *only* thing it can register (`behavior_delta_per_byte`, thesis §14).
  Everything else is masked until this lands.
- **The ribs** (02–07) are the architecture frontiers the un-blinded eval is
  meant to adjudicate: the steering ceiling, the perception depth, the
  scope-control failure, the unclosed metabolism loop, the memory-bloat
  bottleneck, and the predictive (world-model) foundation.

## The projects

| # | Project | Thesis anchor | Goal anchor | The gap it attacks |
|---|---------|---------------|-------------|--------------------|
| [01](01-correction-to-competence-benchmark.md) | **Correction-to-Competence Benchmark v2** | §14, §9/§13, §10 | Goal 1 + 3 | The eval is saturated at 1.0 and cannot discriminate architectures. |
| [02](02-kv-slot-fact-injection.md) | **Beyond the cvec ceiling: KV-slot fact injection** | §1, §7 | Goal 1 | Residual cvecs shift *posture/domain* but provably can't force an exact token; direct KV-slot write is blocked on the binding. |
| [03](03-layer-l-hidden-extraction.md) | **Real hidden-state extraction at layer L** | §1, §3 | Goal 2 | The cortex metabolizes a final-layer mean-pool, not the residual where semantic intent forms. |
| [04](04-context-scoped-attractors.md) | **Context-scoped semantic attractors** | §7 | Goal 3 | Teaching a second sense of one token overwrites the first (Stage 2 fails 100%). |
| [05](05-metabolism-loop-closure.md) | **Closing the metabolism loop** | §2, §3, §4 | Goal 3 | Tensor critic / value head / replay-SGD are wired but unvalidated; corrections don't yet *compound* into cold drift. |
| [06](06-bounded-growth-consolidation.md) | **Bounded-growth consolidation** | §5, §9 | Goal 3 | The autoencoder/hypernetwork hoard serialized objects (~68 KB/Δ) instead of compact adapters. |
| [07](07-conversation-world-model-rl.md) | **RL Phase 0: conversation world model** | §6 | — | A lexical `_looks_like_correction` stop-gap stands in for a model that should *predict* the correction. |

## Current verdicts (Campaign 0d48130, 2026-07-11)

All reported completed runs used CPU-only; Campaign Exp03 was infrastructure-blocked before execution. Curated
campaign log: `../experiments_logs/2026-07-11_campaign_0d48130.md`. The S1.4
layer-L probe log is `../experiments_logs/2026-07-01_s1_4_hf_layer_probe.md`.
Nulls and refutations are reported alongside wins.

| # | Verdict | Primary metric | Seeds | Notes |
|---|---------|----------------|-------|-------|
| 01 | **TESTED-NULL** | `v2_behavior_delta_mock=0.0`, `v2_discrimination=0.0` | 1 (colab) | 5 de-saturation events but zero behavior delta; exact_recall=0.0, domain_recall=1.0 |
| 02 | **REFUTED** | `kv_slot_rank1_count=0.0` | 1 (colab) | Logit biasing confirmed (rank1=3.0); KV-slot cannot recall |
| 03 | **REFUTED** (S1.4) | Qwen gap −0.083, LFM2.5 gap +0.058 (threshold +0.10) | 2 architectures | Mid-layer hiddens do not beat final layer; campaign re-run infrastructure-blocked (no verdict from campaign) |
| 04 | **ACCEPTED** | `scope_selectivity_index=1.0` | 1 (colab) | Single-run, no cross-seed variance |
| 05 | **TESTED-NULL** | `metabolism_drift_delta=0.0`, `drift_uptake=0.0` | 1 (colab) | Loop runs (4 consolidations, slope=0.1755) but no captured behavioral delta |
| 06 | **ACCEPTED** | `bounded_growth_m1_ratio=0.002079` | 5 (kaggle) | Zero variance; bit-identical footprints; bytes_per_delta spread ≤20 B |
| 07 | **ACCEPTED-PARTIAL** | `marker_free_uptake_gap=1.0`; `critic_auc_delta=0.0` | 1 (colab) | Predictive AUC positive (0.8125/1.0); critic AUC improvement null |
| 14 | **TESTED-METRICLESS-NULL** | no METRIC/ASI emitted | 3 (kaggle) | Exit 0 after 11,787 s; harness completed but produced no scored output |
| 18 | **TESTED-PARTIAL** | `distill_delta_holdout` mean=0.2222 | 3 (kaggle) | Gate passed (1 seed, 0.3333); full run bimodal {0.3333, 0.3333, 0.0} |

## Post-remediation frontier

Sprints 0–3 invalidated the cheap activation-steering route, established
retrieval as the honest baseline, and left a narrower question: can learned
neural state outside a frozen LM acquire and express experience without an
answer-time content store?

| # | Project | Role in the new sequence | Primary distinction | Status |
|---|---|---|---|---|
| [18](18-consolidation-as-distillation.md) | Weight-editing comparator | Experience is consolidated into LoRA weights inside the LM. | **TESTED-PARTIAL** (2026-07-11) |
| [19](19-lm-as-language-organ.md) | Direct-learning diagnostic | A label-prefix parametric-retrieval arm is separated from latent control of a frozen language organ. | PENDING |
| [20](20-meta-trained-cortex-frozen-language-organ.md) | Core cortex hypothesis; [Experiment 09](../experiments/09-meta-trained-cortex-frozen-language-organ/) | The write, read, consolidation, and articulation rules are meta-trained; only cortex state changes on an unseen task. | PENDING |
| [21](21-cortex-routed-frozen-specialist-organs.md) | Multi-organ extension | A learned cortex routes shared state into independently frozen language and action organs. | BLOCKED (depends on 20) |
| [22](22-parametric-memory-decoding-zero-shot-lora-routing.md) | LoRA-bank addressability comparator and conditional cortex integration | Stage A tests strict versus calibrated PMD on a shared frozen backbone/LoRA bank; Stage B requires both a valid Stage A route and Research/20 acceptance. | PENDING (Stage A); BLOCKED (Stage B depends on 20 + valid Stage A) |

Dependency and verdict order:

```text
18  LM-weight comparator ───────────────┐  TESTED-PARTIAL (gate passed,
                                        ├─► same reporting table         bimodal full run)
19  direct cortex: label vs latent ─────┘  PENDING
                    │
                    ▼
20  meta-trained cortex over frozen language organ  PENDING
                    │ ACCEPT + causal-state audit
                    ├──────────────────────► 21  cortex-routed frozen language +
                    │                           action organs  BLOCKED (needs 20)
                    │
                    └───────────────────────────────┐
                                                    ▼
22A standalone frozen-backbone LoRA-bank ────────► 22B cortex + PMD integration
    PMD comparator  PENDING (independent of 20)     BLOCKED (needs 20 + valid 22A)
```

Retrieval is mandatory in every comparison table but disabled in the primary
conditions for 19B, 20, and 21. A retrieval win is a result; it cannot be
relabelled as cortex metabolism.

## Dependency graph

```
                01  Benchmark v2  (de-saturate — unblocks honest claims for all)
                 │                   TESTED-NULL: 5 de-saturation events,
                 │                   zero behavior delta
   ┌─────────────┼───────────────────────────┬───────────────┐
   │             │                            │               │
  02 KV-slot    03 Layer-L hiddens           06 Bounded      07 World model
  fact inject    (perception depth)           growth          (predict the
  REFUTED        REFUTED (S1.4)               ACCEPTED        correction)
  kv_slot=0      gap < +0.10                  m1=0.002        ACCEPTED-PARTIAL
   │             │                            │   5 seeds      uptake=1.0
   │             ├──────────────┐             │               critic_auc=0.0
   │             ▼              ▼             │
   │            04 Context-     05 Metabolism │
   │            scoped          loop closure ─┘
   │            attractors      (compounding drift)
   │            ACCEPTED        TESTED-NULL
   │            SSI=1.0         drift=0.0
   └────────────┴──────────────┘
        02/03/04 all chip at the same wall: a residual control vector
        carries posture; facts need a reserved slot (02) keyed by real
        context (03/04). 05 needs 03's real hiddens; 06 needs 01's metric.
        Soft-dep: 07 → 01 — the world-model can run independently, but its
        `accept_pred_auc ≥ 0.70` acceptance threshold only becomes
        meaningful once 01's de-saturated scorecard exists.
```

Reading order if you only do one thing: **01**, then whichever rib has live
momentum. **03** is refuted (S1.4: mid-layer hiddens do not beat final layer
on two architectures); **04** and **05** both consumed 03's input and have
their own verdicts (04 ACCEPTED, 05 TESTED-NULL). The highest-leverage open
question is now **18** (TESTED-PARTIAL) → **19** (PENDING) → **20** (PENDING).

## Conventions these proposals follow

- **Matched-pair, single-variable controls** — the repo standard (see the
  SVD-vs-non-SVD and S-vs-H runs). Every experiment isolates one axis.
- **Mock vs real driver** — the `_MockDriver` (deterministic hash embeddings,
  *zero semantics*) is a structural null; the real `LFM2.5-1.2B-Instruct
  Q4_K_M` driver is the semantic test. Results are always reported separately.
- **Non-saturating metrics only** — success criteria deliberately avoid the
  metrics already pinned at 1.0 (`code_qa_accuracy`, real-driver `co_recall`,
  capped forgetting/identity ratios). Where a 1.0 metric is kept, it is used
  only as a regression *floor*, labeled as such.
- **Honest nulls are a result** — several proposals (notably 01 and 05)
  pre-register the possibility that the answer is "no measurable difference,"
  which is itself a de-saturation win over a saturated tie.

## Provenance

Drafted 2026-06-28 by a multi-agent pass: six readers mapped the subsystems,
one author drafted each project, and an adversarial grounding checker verified
every claim about the existing code against real files (correcting line
numbers, distinguishing semantics-free mock `0/0` from real-driver `1/1`, and
softening overstated capabilities). Two of the six map briefs (organs,
benchmarks) degraded to placeholders, but the per-project grounding pass read
those subsystems directly, so the cited `eval_suite.py` / `autoencoder.py` /
`hypernet.py` / `organism.py` facts are first-hand. Treat any remaining
unverified forward-looking claim as a hypothesis, not a finding.

**2026-07-09 human-authorized extension:** Research/19 was amended before
implementation to close the label-prefix interpretation loophole; Research/20
and /21 plus Experiment/09 were added to test a meta-trained cortex controlling
frozen language and action organs without retrieval in the primary path.

**2026-07-11 status update:** Campaign 0d48130 adjudicated projects 01–07, 14,
and 18. Verdicts and outcome annotations added to each project spec; this README
updated with a current-verdicts table, R18 result in the frontier table, and
dependency-graph status labels. Projects 19 and 20 remain PENDING; 21 remains
BLOCKED (depends on 20). No hypotheses, success criteria, or kill criteria
were modified. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

**2026-07-16 human-authorized extension:** Research/22 was added as a
standalone strict-versus-calibrated PMD addressability comparator over a shared
frozen backbone and LoRA memory bank. Its Stage A is independent of
Research/20, while Stage B cortex integration requires both a valid Stage A
route and Research/20 acceptance. Research/21 was not changed.
