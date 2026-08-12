# 22 — Parametric memory decoding for zero-shot LoRA-bank routing

**Pre-registered 2026-07-16** (human-authorized before implementation).
Agents running this experiment MUST NOT edit this spec. Deviations are reported
as deviations. Stage A is independent of Research/20. Stage B is **BLOCKED**
unless Research/20 accepts H-META-CORTEX and passes its causal-state audit.

## Source and evidence boundary

This project is prompted by Fengxian Ji, Zhuohan Xie, Jingpu Yang, Fan Zhang,
Zirui Song, and Xiuying Chen, “Parametric Memory Decoding for Zero-Shot Routing
in LoRA-Based External Parametric Memory,” arXiv:2607.04118v1, 2026
([paper](https://arxiv.org/abs/2607.04118)). The citation motivates a comparator;
it is not evidence that Oczy already has addressable or metabolized memory.

The paper studies one specific setting: a shared frozen transformer backbone and
a bank of LoRA modules attached to that same backbone. Its reported mechanism is
therefore directly applicable only to a **LoRA external-parametric-memory bank**.
It does not establish that the same score routes Oczy's non-LoRA fast/slow cortex
state, independent models, or the multi-model organs in Research/21.

The following paper facts constrain this pre-registration:

- For an adapted layer `l` and memory unit `k`, the LoRA update is
  $\Delta W_{l,k}=B_{l,k}A_{l,k}$. In the paper's PMDRouter notation, one
  adapter-free backbone prefill produces the fixed pooled activation
  $u(x)=\operatorname{SemPool}(h^{\mathrm{base}}_{l_*}(x))$.
- The response and scale-normalized energy are

  $$
  \rho_{l,k}(x)=B_{l,k}A_{l,k}u(x),
  \qquad
  E_{l,k}(x)=
  \frac{\lVert B_{l,k}A_{l,k}u(x)\rVert_2^2}
       {\lVert u(x)\rVert_2^2\lVert B_{l,k}A_{l,k}\rVert_F^2+\epsilon}.
  $$

  The paper averages this energy over the adapter's layers, applies a
  per-adapter calibration operator, and selects

  $$
  s_k(x)=\operatorname{Calib}_k\!\left(
    |\mathcal S_k|^{-1}\sum_{l\in\mathcal S_k}E_{l,k}(x)
  \right),
  \qquad \hat k(x)=\arg\max_k s_k(x).
  $$

- “Parameter-free” means no separately trained router. The reported clean
  implementation nevertheless estimates calibration statistics from the
  training split, and the appendix says the main LAG and PMDRouter tables use
  matched train-split calibration. That is a data-dependent router even though
  it has no learned gate parameters. Research/22 therefore gives strict
  uncalibrated PMD and train-split-calibrated PMD separate conditions and
  separate verdicts.
- In the reported main table, BM25 beats PMDRouter on every PaperQA backbone
  (`0.820` versus `0.600/0.613/0.658`) and every NQ-DomainLoRA backbone (`0.897`
  versus `0.769/0.781/0.790`). PMDRouter is strongest among internal-signal
  routers in eight of nine backbone/benchmark columns; LAG is higher on
  Qwen3-4B-Instruct NQ-DomainLoRA (`0.793` versus `0.781`). Retrieval is thus a
  mandatory baseline, not an inconvenient result to omit.
- The PaperQA ablation reports Projection-only, based on the low-rank addressing
  coordinate $A_{l,k}u(x)$, as stronger than the full $B_{l,k}A_{l,k}u(x)$
  decoder on all three backbones. Full write-back is not presumed necessary.
- The paper's approximately `40x` reduction is an estimate of **routing-design
  search-space complexity** (`1.00` versus `0.025` normalized), not measured
  wall-clock speed, throughput, or inference cost. Research/22 will not report
  it as runtime acceleration.
- The paper explicitly leaves larger banks, multi-hop access, continually
  updated banks, and more diverse memory modules for future work. PMD-Bench has
  35 memory-unit partitions in total, not evidence of unbounded scale or
  continual insertion.
- Signal-writing evidence is secondary: the v1 text says the no-auxiliary and
  raw-energy controls are pending, and its displayed table and prose are not
  internally sufficient to isolate the auxiliary loss. No metabolism or
  training-side addressability claim may rest on that result.

## Problem and Oczy applicability boundary

Research/18 tested task-local LoRA weight editing, while Research/20 tests a
learned cortex controlling a frozen language organ. Neither asks the narrower
systems question posed here: given a **fixed bank of already trained LoRAs on
one shared backbone**, can the query and the bank's own parameters identify
which LoRA to activate without a learned gate or retrieval index?

Research/22 has two stages so this useful comparator cannot be mistaken for an
Oczy result:

1. **Stage A — standalone addressability comparator.** Reproduce and stress the
   PMD mechanism on a shared frozen backbone and frozen LoRA EPM bank. This can
   start regardless of Research/20 and does not use cortex state. It tests
   routing and routed-task utility, not learning, consolidation, trace deletion,
   or metabolism.
2. **Stage B — cortex integration.** Only after Research/20 accepts, test whether
   its learned query-conditioned cortex state adds useful routing information
   to the already validated LoRA-bank interface. This is an Oczy integration
   experiment, not a claim that PMD directly decodes cortex parameters.

Neither stage is Research/21. Selecting LoRAs that share a backbone does not
validate routing among independently frozen language, action, vision, or code
models. A Stage A win cannot unblock or accept Research/21.

## Falsifiable hypotheses

### Stage A

**H-PMD-STRICT:** on sealed queries from held-out memory units, the exact
uncalibrated response-energy rule contains query-conditioned addressing signal:
it beats the strongest frozen internal-signal zero-shot comparator by the
DEV-frozen meaningful margin, improves downstream execution over a random or
wrong route, and loses that advantage under matched response interventions.

**H-PMD-CAL:** train-split calibration improves or stabilizes that signal on
sealed queries and, after charging its examples, statistics, and build cost,
beats the strongest equally calibrated internal-signal comparator by the
DEV-frozen meaningful margin.

These hypotheses receive separate verdicts. A calibrated-only win is reported
as **calibration-dependent PMD**, not “strict zero-shot” routing and not a rescue
of H-PMD-STRICT. Retrieval or a learned router may remain the better product
choice even when either mechanistic hypothesis accepts.

### Stage B

**H-CORTEX-PMD:** conditional on an accepted Research/20 cortex and a valid
Stage A LoRA interface, query-conditioned learned cortex state improves routing
and routed behavior on unseen tasks beyond the frozen PMD route, and the gain
causally disappears when the cortex state is zeroed, swapped, or feedback is
shuffled while the backbone and LoRA bank remain bit-identical.

## Staged architecture

### Stage A — shared-backbone LoRA EPM

The only mutable work before the manifest freeze is instrument construction and
DEV selection. The scored system contains:

- one tokenizer and one causal language-model backbone, shared by every arm;
- a bank $\mathcal M=\{L_k\}_{k=1}^{K}$ of single-unit LoRA memories, all
  trained from the same backbone checkpoint;
- one adapter-free prefill path exposing the pre-registered layer/module inputs;
- fixed strict and calibrated PMD scorers;
- a hard top-1 route followed by one generation/execution pass with only the
  selected LoRA active; and
- immutable scorer, dataset, bank, and model manifests.

No query is run through every active LoRA to choose the best generation. PMD may
apply each low-rank matrix to the shared prefill activation, but per-candidate
full-model generation is an oracle/probing condition and is charged separately.
No router, LoRA, backbone weight, pooling rule, layer set, or calibration
statistic changes after the sealed split is exposed.

### Stage B — learned-cortex read into the validated interface

Stage B reuses the accepted Research/20 developmental parameters, fast/slow
state lifecycle, and causal audits. A fixed-width, developmentally trained
coupler maps the Research/20 query-conditioned read state into either (a) an
additive query feature presented to a frozen routing decoder or (b) logits over
the same frozen LoRA bank. The choice, capacity, training distribution, and
state width are selected on developmental train/DEV and frozen before sealed
evaluation. Only Research/20-permitted fast/slow state changes during a new
episode; backbone, LoRAs, router/coupler developmental parameters, and PMD
calibration remain frozen.

The PMD formula remains a LoRA-bank comparator. Any Stage B gain is attributed
to the learned cortex path only after matched zero/swap/shuffle interventions.
It is not described as “PMD decoding cortex weights.”

## Frozen LoRA-bank instrument

The instrument is versioned independently of `eval/v2` and all existing
Research/20 measuring instruments. Research/22 MUST NOT edit their episodes,
scorers, thresholds, or evaluation files.

### Memory families and split

The generator contains document/fact, domain/rule, and task/skill families with
surface-overlapping distractors. Each unit has disjoint source material,
training queries, DEV queries, and sealed evaluation queries. Complete memory
units—not paraphrases, templates, or random seeds—are split across development
and sealed evaluation wherever a generalization claim is made. Exact duplicate,
near-duplicate, answer-string, source-title, and template leakage checks run
before freeze.

Bank-size tiers, number of units per family, query counts, LoRA rank, target
modules, pooling span, routing layer set, and seeds are selected only through
the DEV distribution and power procedure below. At least two DEV bank sizes
must be examined so a small-bank ceiling is detectable; the largest supported
sealed tier is whatever the frozen resource and power checks justify. No result
is extrapolated beyond that tier, and a fixed bank is not called continual.

### LoRA construction and validity gates

All primary-bank LoRAs use matched rank, target modules, scaling convention,
optimizer family, token budget, and stopping rule. Training uses only the unit's
training partition. The manifest records source hashes, training-example hashes,
code revision, tokenizer/backbone hashes, all adapter tensors, optimizer and
seed, rank/targets/scaling, and training cost.

Before a unit is eligible:

1. **Oracle-attachment gate:** the correctly attached LoRA must improve its
   unit's held-out DEV task score over the bare backbone under a DEV-frozen
   confidence rule.
2. **Specificity gate:** it must not produce an indiscriminate equal gain on
   matched non-target DEV queries; the equivalence/non-inferiority margin comes
   from the DEV distribution check.
3. **Route relevance gate:** oracle attachment must beat random/wrong attachment
   on routed downstream utility. Otherwise selecting the unit is not a
   behaviorally meaningful route.
4. **Hash gate:** backbone and all adapters must be bit-identical before and
   after every scored run.

A failed unit is reported and removed only before manifest sign-off by the
pre-registered eligibility rule. Failure after freeze blocks that tier; it does
not permit replacement with an easier unit.

## Routing conditions and controls

Every method receives the identical query text and candidate bank. Any method
using source text, labels, training examples, fitted statistics, or trained
parameters declares and accounts for them.

| ID | Condition | Role |
|---|---|---|
| A0 | Uniform/random route, repeated with frozen seeds | Chance and routed-utility floor |
| A1 | **Strict PMD-full:** raw layer-mean $E_{l,k}$, `Calib_k` = identity | Primary H-PMD-STRICT condition |
| A2 | **Calibrated PMD-full:** fixed per-LoRA transform fitted on training queries only | Primary H-PMD-CAL condition |
| A3 | PMD Projection-only using $A_{l,k}u(x)$ with matched scale normalization, strict and calibrated | Tests the paper's strongest ablation and whether $B$ hurts addressing |
| A4 | Frozen internal-signal zero-shot routers (Arrow, SpectR, and the strongest reproducible LAG/SEQR-style arm) | No-trained-gate comparators |
| A5 | BM25 and dense retrieval over the permitted memory source/training text | Mandatory external retrieval baselines |
| A6 | Capacity-matched supervised router trained on training units/queries and frozen before evaluation | Learned-router comparator |
| A7 | Gold unit route and, separately, per-candidate generation oracle | Capability ceilings, never deployable routing |
| A8 | A1/A2 with response scores zeroed, candidate-permuted, or swapped after the same prefill | Causal response-path interventions |
| A9 | A2 with calibration statistics identity-set or permuted across adapters | Calibration-dependence intervention |
| A10 | Correct PMD route but wrong LoRA attached, and wrong route but correct LoRA forcibly attached | Separates selection from downstream adapter utility |

For A2, Oczy's operational calibration is specified before implementation as a
per-adapter affine transform of a stabilized scalar energy:

$$
r_k(x)=\log\!\left(\frac{1}{|\mathcal S_k|}
  \sum_{l\in\mathcal S_k}E_{l,k}(x)+\epsilon_{\log}\right),\qquad
s_k^{\mathrm{cal}}(x)=\frac{r_k(x)-\mu_k}{\max(\sigma_k,\sigma_{\min})}.
$$

Here $\mu_k$ and $\sigma_k$ are computed once from that adapter's permitted
training-query distribution. The energy-denominator $\epsilon$,
$\epsilon_{\log}$, and $\sigma_{\min}$ are selected from the pooled
training/DEV numerical-stability distribution, recorded separately, and frozen
in the manifest. No sealed query or label contributes. This is an explicit
Oczy operationalization
of the paper's `Calib_k`, whose main formula leaves the operator abstract; it
must not be presented as a verbatim paper equation. A1 is the clean test that
requires no such statistics.

A4's strongest comparator is chosen on DEV under a frozen selection rule and is
then carried unchanged to every sealed bank tier. A5 is always shown in the main
comparison table. Retrieval is not attached to PMD and is not called parametric
memory. A6's parameter count, training examples, and optimization cost are
reported rather than hidden behind routing accuracy.

## Protocol

### Stage A

1. Materialize train/DEV partitions, train matched LoRAs, run the validity and
   leakage gates, and select all method details on DEV only.
2. Run the distribution/power procedure; freeze the signed manifest, hypotheses,
   estimands, confidence procedures, meaningful/equivalence margins, sample
   sizes, bank tiers, and method-selection rule.
3. Hash the backbone, tokenizer, adapters, calibration statistics, retrieval
   corpora/indexes, learned router, scorer, and sealed query list.
4. For every sealed query, execute the adapter-free prefill once and cache the
   same allowed activation for all internal-signal arms. Cache use is logged and
   never crosses split or query boundaries.
5. Compute routes for A0–A9 without observing downstream answers. Attach only
   the selected adapter and score routed behavior. Execute A7/A10 separately.
6. Repeat A8/A9 interventions from the identical prefill and bank without
   retraining or changing the query.
7. Audit hashes, information sources, cache deletion, failures, latency, and
   bytes. Unblind only after the complete signed result bundle exists.

### Stage B

1. Proceed only after the Stage B blocking gates below pass.
2. Developmentally train/select the cortex-to-route coupler on Research/20
   training tasks and Research/22 development banks; freeze it before sealed
   evaluation.
3. On unseen tasks, record the no-experience route, present the permitted
   experience/feedback events, update only Research/20 cortex state, consolidate
   as Research/20 permits, and delete raw traces under its audit.
4. Score the frozen PMD-only route, cortex-augmented route, retrieval, learned
   router, and oracle on identical queries and banks.
5. Replay routes with cortex state zeroed, swapped between tasks, feedback
   shuffled, and routing logits permuted. Do not rerun learning for a favorable
   intervention outcome.

## Primary metrics

Stage A reports, per family and bank tier:

1. `route_top1_accuracy`, macro-averaged over memory units;
2. `route_mrr` and frozen `route_recall_at_k` as secondary ranking diagnostics;
3. `response_margin = s_gold - max(s_non_gold)` and its calibrated/strict
   distribution, without selecting a cutoff on sealed data;
4. `routed_task_score`, using the family's pre-frozen exact or semantic scorer;
5. `oracle_route_gap = oracle_task_score - routed_task_score`;
6. `strict_pmd_delta` — A1 minus the DEV-selected strongest A4 comparator;
7. `calibrated_pmd_delta` — A2 minus the equally calibrated strongest A4
   comparator;
8. `calibration_delta` — A2 minus A1, with A9 intervention results;
9. `projection_delta` — A3 minus the matched A1/A2 full-response arm;
10. `retrieval_delta` and `learned_router_delta` — PMD minus A5/A6, never
    omitted from the main table;
11. `causal_response_delta` — intact PMD minus each matched A8 intervention; and
12. specificity, failures/abstentions, persistent bytes, build cost, route and
    end-to-end latency, throughput, peak memory, and energy if observable.

Stage B adds pre/post-experience route and task deltas, PMD-only versus
cortex-augmented delta, feedback-semantics delta, zeroed-state delta,
swapped-state addressing delta, trace-free survival, specificity, cortex bytes,
and update/consolidation cost. Research/20 definitions are reused rather than
silently redefined.

The independent unit for generalization is the memory unit/task, not repeated
seeds or queries from the same unit. Estimates are macro and micro reported,
with hierarchical or cluster bootstrap CIs over units and queries. Family,
bank-size, backbone, strict/calibrated, and per-unit results are shown; pooling
may not conceal a failing family.

## DEV distribution check and power freeze

There is no fixed accuracy threshold in this spec. Before sealed evaluation:

1. Run repeated A0, A1–A6, and oracle conditions on development banks spanning
   the proposed families and bank sizes. Measure chance, ceiling, saturation,
   unit heterogeneity, paired-difference variance, scorer repeatability,
   numerical ties, and run-to-run systems noise.
2. Reject or redesign the unsealed instrument if oracle utility is absent,
   chance/ceiling leaves no discriminating range, eligible units collapse to one
   surface cue, or repeated scoring is not stable enough to resolve a useful
   effect. Every redesign increments the instrument version.
3. Define the smallest meaningful routing and downstream effect from the larger
   of observed repeatability/noise and the minimum gain that changes the
   pre-declared storage/latency tradeoff. Define specificity and task-quality
   equivalence/non-inferiority margins from no-update and repeated-run DEV
   distributions. Do not use a paper result or desired verdict as the margin.
4. Choose the confidence level, multiplicity correction, bootstrap scheme, and
   target power before unblinding. Simulate clustered resampling from DEV units
   to select unit/query counts for every primary contrast. Seeds estimate
   execution variance but do not inflate the independent sample count.
5. Freeze a fallback rule for underpowered bank tiers: report estimation and
   label the tier **INCONCLUSIVE**; never pool it post hoc with an easier tier.
6. Obtain human sign-off on the versioned manifest, signed margins, CI rules,
   sample sizes, exclusions, resource ceiling, and sealed-data hash. Any change
   requires a new version and cannot alter the current verdict.

## Acceptance, refutation, and blocking gates

### Stage A verdicts

**Accept H-PMD-STRICT** only if all of the following hold under the frozen CI and
multiplicity procedure:

- A1's `strict_pmd_delta` lower confidence bound exceeds the DEV-frozen
  meaningful routing margin on the pre-declared primary families/tiers;
- A1 improves routed downstream task score over A0/wrong-route by the frozen
  meaningful margin and closes a pre-declared portion of the oracle route gap;
- intact response scoring beats the A8 zero/swap/permutation interventions;
- specificity/task-quality remains inside its DEV-frozen margin; and
- all validity, leakage, source-boundary, and hash audits pass.

**Accept H-PMD-CAL** only by the analogous A2 comparisons against an equally
calibrated A4 comparator, plus a positive calibration contribution consistent
with A9 and full accounting of calibration data/cost. H-PMD-CAL cannot accept
H-PMD-STRICT.

**Refute a hypothesis** when its oracle and instrument gates pass but any
required primary condition fails. If the CI cannot distinguish the frozen
margin at the frozen sample size, report **INCONCLUSIVE**, not accepted or
refuted. If Projection-only wins, report it as the preferred addressing signal;
do not imply that full $BAu$ write-back was validated. If BM25/dense retrieval
or A6 wins, preserve an accepted mechanistic verdict if earned but state that
PMD lost the corresponding product comparison.

Stage A is **BLOCKED**, not refuted, if the shared-backbone LoRA bank cannot pass
oracle/specificity/route-relevance gates, the frozen backbone cannot expose the
specified activations, the instrument lacks signed DEV distribution and power
checks, or a hash/leakage audit fails. Research/20's status is not a Stage A
blocking gate.

### Stage B gates and verdict

Stage B remains **BLOCKED** until all are true:

- Research/20 accepts H-META-CORTEX on at least one relevant task family and its
  causal-state, trace-deletion, specificity, and frozen-organ audits pass;
- Stage A identifies at least one non-refuted, manifest-frozen LoRA routing path
  with meaningful routed utility; and
- a new Stage B execution manifest freezes the coupler, development split,
  interventions, margins, power, and source boundaries before sealed use.

Accept H-CORTEX-PMD only if the cortex-augmented arm beats frozen PMD-only by its
DEV-frozen routing and downstream margins, depends on correct experience and
feedback, falls under zero/swap interventions, preserves specificity and trace
freedom, and leaves all frozen-backbone/LoRA hashes unchanged. Refute if those
validity gates pass but any causal primary condition fails. If Research/20
refutes, Stage B stays blocked; a successful Stage A comparator cannot hide or
reverse that result.

## Pre-registered interventions and follow-ups

The only primary interventions are A8–A10 and the Stage B zero/swap/shuffle
controls above. Diagnostic analyses after a null may stratify already frozen
results by response margin, overlap, unit, family, or bank size, but cannot
change the verdict.

Training-side signal writing is a **secondary follow-up** on a separately hashed
bank. It uses

$$
\mathcal L=\mathcal L_{\mathrm{task}}+\lambda\mathcal L_{\mathrm{write}}
$$

and must include task-loss-only/no-aux, compute-matched extra-training,
random-code or label-permuted, raw-energy versus transformed-energy, and
projection-only controls; it must also pass task-quality equivalence and report
all training costs. `lambda`, codes, transforms, and margins are DEV-frozen.
Without those controls, any gain is labeled exploratory signal shaping. It does
not establish continual memory, Oczy metabolism, or a better Stage A router.

Larger or continually inserted banks, top-k composition, multi-hop access,
heterogeneous ranks/target modules, and another backbone require a new signed
instrument version or research spec. They are not post-null rescue knobs.

## Efficiency and accounting

For $K$ LoRAs of rank $r$ over selected layers, PMD's route consists of one
adapter-free backbone prefill plus low-rank matrix-vector work for every
candidate, approximately $O(K\sum_l d_lr_l)$, followed by one selected-adapter
inference pass. The experiment measures rather than assumes whether that is
faster than retrieval, a learned gate, adapter swapping, or batching on the
actual runtime.

Report separately:

- cold/warm prefill, score, adapter-load, selected-generation, and total latency;
- p50/p95 latency, throughput, peak host/device memory, and bytes read;
- LoRA-bank bytes, calibration-statistic bytes/examples, retrieval corpus/index
  bytes, and learned-router parameters;
- LoRA training, calibration, retrieval-index build, and learned-router training
  FLOPs/time/energy where observable; and
- amortized and non-amortized cost at each frozen bank tier.

Cached activations and resident adapters are declared in each timing regime.
The paper's `~40x` design-search number appears only in provenance/discussion and
is never entered in these runtime tables.

## Interpretation constraints

- Routing a fixed LoRA bank is addressability, not memory formation. Stage A has
  no experience-to-state update, consolidation, forgetting, or raw-trace
  deletion and cannot validate metabolism.
- “Zero-shot” must always be qualified: A1 is strict uncalibrated internal-signal
  routing; A2 uses train-split calibration; A4 may use fixed proxy statistics;
  A5 is retrieval; A6 is supervised routing.
- A route-label hit without downstream oracle-relevant gain is not useful
  memory access. Conversely, generation quality cannot choose the route in a
  deployable condition.
- Calibration fitted on training queries is routing information. It is not
  erased from accounting because it has no gradient-trained parameters.
- Projection-only beating full response is a mechanistic result, not an
  ablation inconvenience. Do not claim that $B$ or full write-back earned its
  role when $Au$ suffices.
- Retrieval wins are reported in the abstract and main table. In particular,
  the source paper's BM25 wins on PaperQA and NQ-DomainLoRA are part of the
  motivation for the mandatory baseline.
- No evidence from this fixed, bounded bank supports larger, continual,
  multi-hop, compositional, or heterogeneous banks unless separately tested.
- Signal writing remains secondary until all listed controls pass; even then it
  is LoRA training, not automatically Oczy cortex learning.
- Stage B may establish a cortex contribution to routing, but only Research/20's
  own gates can establish learned cortex memory. Research/21 remains unchanged
  and independently blocked on Research/20.
- No threshold, pooling rule, layer, calibration, comparator, family exclusion,
  or bank tier may be selected from sealed outcomes.

## Reporting

The report begins with the source-evidence audit and explicitly states the
applicability boundary. It then includes the signed manifest and deviation log;
unit/split/leakage and oracle-gate records; strict and calibrated tables side by
side; Projection-only/full-response ablation; mandatory BM25/dense retrieval,
learned-router, and oracle comparisons; routed-task and causal-intervention
results; family/unit/bank-size CIs; hash and information-source audits; and the
full efficiency/accounting table. All nulls, blocked tiers, failures, and
excluded units appear before pooled conclusions.

Stage A and Stage B receive separate verdict headings. The abstract must say
whether any win was strict or calibration-dependent and whether retrieval or a
learned router still won. Exploratory signal-writing results are in a secondary
section and cannot alter either primary verdict. Log Stage A to
`experiments_logs/<date>_s22_pmd_lora_routing.md`; if Stage B becomes eligible,
log it separately to `experiments_logs/<date>_s22_cortex_pmd_integration.md`.
