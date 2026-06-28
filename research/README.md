# Oczy Research Agenda

A prioritized set of research projects for the **Plastic World Model Agent**
(`../experiments.txt`). Each project here is a *proposal* — problem, falsifiable
hypothesis, approach, and discriminating success criteria. Each has a matching
runnable experiment design under [`../experiments/`](../experiments/).

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

## Dependency graph

```
                01  Benchmark v2  (de-saturate — unblocks honest claims for all)
                 │
   ┌─────────────┼───────────────────────────┬───────────────┐
   │             │                            │               │
  02 KV-slot    03 Layer-L hiddens           06 Bounded      07 World model
  fact inject    (perception depth)           growth          (predict the
   │             │                            │   correction)
   │             ├──────────────┐             │
   │             ▼              ▼             │
   │            04 Context-     05 Metabolism │
   │            scoped          loop closure ─┘
   │            attractors      (compounding drift)
   └────────────┴──────────────┘
        02/03/04 all chip at the same wall: a residual control vector
        carries posture; facts need a reserved slot (02) keyed by real
        context (03/04). 05 needs 03's real hiddens; 06 needs 01's metric.
        Soft-dep: 07 → 01 — the world-model can run independently, but its
        `accept_pred_auc ≥ 0.70` acceptance threshold only becomes
        meaningful once 01's de-saturated scorecard exists.
```

Reading order if you only do one thing: **01**, then whichever rib has live
momentum. **03** is the highest-leverage rib because **04** and **05** both
depend on it.

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
