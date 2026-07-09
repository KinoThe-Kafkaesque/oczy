# Experiment 09: Meta-trained cortex over a frozen language organ

## Objective

Can a cortex learn **how to learn** across developmental episodes, then acquire
a previously unseen rule from correction, consolidate it into persistent neural
state, delete the experience, and make a frozen LM execute the rule through a
fixed-width latent interface?

This operationalizes
[`research/20-meta-trained-cortex-frozen-language-organ.md`](../../research/20-meta-trained-cortex-frozen-language-organ.md).
Research/21's multi-organ tool router is deliberately excluded until this core
single-organ experiment accepts.

## Claim boundary

The primary condition tests cortex capability with retrieval disabled.

After consolidation, the only experience-dependent persistent object allowed
on the answer path is the fixed-shape fast/slow cortex state. The runner must
prove that it does **not** retain or consult:

- experience or correction text;
- tokenized traces;
- labels, target strings, or exemplars;
- nearest-neighbor indexes or per-episode embeddings;
- episode/task identifiers;
- a correction-produced LM KV cache; or
- any variable-length structure that grows with experience count.

Retrieval is run in a separate condition as the bar to report. It is never
composed with the primary cortex condition and cannot accept the hypothesis.

## Frozen language organ

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` through `HFDriver`.
- **Mode:** deterministic greedy evaluation; chat template fixed in the
  instrument manifest.
- **Frozen surface:** tokenizer, embeddings, transformer blocks, normalization,
  and LM head. Parameters are hashed before/after developmental training and
  every evaluation seed.
- **Perception:** final-layer mean-pooled request features.
- **Articulation:** a cortex-owned coupler emits a fixed number of continuous
  soft input embeddings or equivalent KV entries. It never emits text for the
  prompt. Gradients may flow through the frozen organ during developmental
  training, but its parameters never receive updates.

## Cortex architecture (fixed for v1)

The v1 cortex is a small fast-weight programmer rather than a list of memory
slots.

- Project frozen LM features into `d_cortex = 64`.
- Maintain two fixed-shape matrices:
  - `F ∈ R^(64×64)` — fast state, reset at task start;
  - `S ∈ R^(64×64)` — slow state, persistent after consolidation.
- A learned writer receives observation, attempted behavior, correction, and
  outcome features and produces key `k`, value `v`, learning rate `eta`, and
  decay `lambda`.
- The fast update has the fixed outer form
  `F_next = lambda * F + eta * outer(v, k)`. The writer learns every term; no
  task-specific constant or episode ID enters the update.
- A learned consolidator emits gate `g` and applies
  `S_next = (1 - g) * S + g * F`, after which `F` is cleared for the
  post-consolidation primary probe.
- A query-conditioned read computes from `S`, the current request feature, and
  a learned developmental state; it does not enumerate stored experiences.
- The articulation coupler maps that readout to a DEV-selected, fixed-width
  soft bank. Width is frozen in the v1 manifest and cannot change on meta-test.

Developmental parameters (writer, consolidator, read projection, articulation
coupler) are optimized in the outer loop and frozen before meta-test. During a
meta-test episode there is no backpropagation or optimizer step: only the
learned fast update and consolidation equations may change `F` and `S`.

## Instrument: `meta_cortex/v1`

This experiment does not modify or optimize against `eval/v2`. It has a
separate frozen instrument generated deterministically from a versioned seed
table.

### Task family A — contextual remapping

An opaque symbol has different required outputs in different contexts.

Example shape, not a literal scored item:

```text
Request: In the amber room, respond to dax.
Initial behavior: dax
Correction: In the amber room, dax requires the token north.

Held-out probe: The room is amber. What response follows dax?
Scope probe: The room is blue. What response follows dax?
```

Mappings, contexts, symbols, outputs, and surface forms are regenerated per
task. Meta-test assignments never appear in meta-training.

### Task family B — rule transformation

A correction teaches an operation that must be applied to unseen operands.
Operations include permutation, substitution, composition, and conditional
transformations over opaque token strings. A probe is correct only if it applies
the rule to an operand never present in teaching; repeating the corrected answer
cannot score.

### Task family C — finite-state behavior

The cortex learns a transition/action rule from feedback and must maintain the
resulting goal/state across two or more turns. The final user turn does not
repeat the original goal. This family tests behavior rather than label recall.

### Task-level split

- Meta-train, meta-validation, and meta-test split by complete rule, mapping,
  transition graph, and output assignment.
- Paraphrases of one rule never cross splits.
- Every family must have at least 30 independent meta-test rules; exact counts
  are increased, if required, by a meta-validation-only power analysis and then
  frozen before meta-test.
- Every task contains pre-learning, same-rule holdout, transfer, composition,
  specificity, and oracle-context probes.

## Phase 0: freeze and validate the measuring instrument

Before outer-loop training:

1. Materialize all task generators, split seeds, scorers, prompt templates,
   and probe counts under `meta_cortex/instrument/v1/`.
2. Generate `MANIFEST.json` with per-file hashes.
3. Run scorer unit tests and leakage checks proving no meta-test rule is
   reachable from meta-training seeds.
4. Measure frozen-organ no-update repeatability and unrelated-task variation on
   meta-validation.
5. Define the specificity and trace-survival equivalence margin as the 95th
   percentile of absolute no-update run-to-run variation.
6. Run a meta-validation power analysis and freeze task counts.
7. Record the soft-bank width, abstain threshold, equivalence margin, task
   counts, and hashes in the manifest.
8. Obtain human sign-off before any meta-test command is enabled.

Any later change requires `meta_cortex/v2`; the optimizing loop may not modify
v1.

## Phase 1: oracle articulation gate

For each task family, give the frozen language organ the complete rule and
worked examples directly as text. Compare it with the same organ without the
rule.

The family passes if the informed condition improves held-out behavior with a
95% CI excluding zero. If it fails, that family is **BLOCKED**: the mouth cannot
express the behavior even when fully informed, so cortex memory is not under
test.

Separately train the soft articulation coupler on meta-training and tune it on
meta-validation. Freeze it before meta-test. It must pass the same oracle
articulation gate when driven by an oracle cortex code; otherwise the latent
interface is blocked.

## Phase 2: developmental outer-loop training

Each training episode follows:

1. initialize `F = 0`, `S = 0`;
2. score pre-learning probes;
3. present one to five experience/correction events;
4. apply the learned writer after each event;
5. consolidate and clear `F`;
6. delete transient event objects;
7. score retention, transfer, composition, and specificity; and
8. backpropagate the post-learning behavioral loss through the unrolled cortex
   and frozen organ into developmental cortex parameters only.

The outer objective combines:

- teacher-forced token loss on correct post-learning behavior;
- full-generation behavioral score on a fixed meta-validation cadence;
- specificity loss on unrelated probes;
- consolidation-survival loss after `F` is cleared; and
- a state-norm/byte regularizer reported separately from behavior.

Architecture, optimizer, learning schedule, and stopping point are selected on
meta-validation. Meta-test is never read during development.

## Phase 3: one-shot meta-test

For every frozen developmental checkpoint, evaluation seed, and unseen rule:

1. start from identical developmental parameters and empty cortex state;
2. record C0/C1 pre-learning behavior;
3. present the task's teaching events;
4. mutate `F` only through the frozen learned writer;
5. consolidate into `S`, clear `F`, delete all event objects, and audit;
6. run the full probe battery once; and
7. replay probes with state zeroed, swapped, and feedback-shuffled without
   additional learning.

Use at least 5 developmental seeds and 5 evaluation seeds. Report task rules as
the independent sample unit; seeds quantify optimizer/order variation.

## Conditions / ablation matrix

| ID | Condition | Mutable during meta-test | Purpose |
|---|---|---|---|
| C0 | Frozen language organ only | nothing | Vanilla mouth baseline |
| C1 | Meta-trained cortex, update disabled | nothing | Architecture/no-update control |
| C2 | Same cortex capacity, random untrained writer | `F`, `S` | Meta-training ablation |
| C3 | Meta-trained cortex, correct feedback, traces deleted | `F`, `S` | Primary condition |
| C4 | C3 trajectory with feedback assignments shuffled | `F`, `S` | Feedback-semantic control |
| C5 | C3 learned trajectory, then state zeroed | nothing after intervention | Causal state control |
| C6 | C3 learned trajectory with another task's state swapped in | nothing after intervention | Conditional addressing control |
| C7 | Research/19 label-prefix head | head weights | Parametric-retrieval comparator |
| C8 | Byte-matched compressed exemplar retrieval | external store | Retrieval bar; never composed with C3 |
| C9 | Full rule and examples in context | context only | Oracle upper bound |

Matched claims:

- C1 versus C3: effect of online cortex update.
- C2 versus C3: effect of developmental meta-training.
- C3 versus C4: semantic feedback versus generic state change.
- C3 versus C5: causal dependence on cortex state.
- C3 versus C6: correct conditional addressing.
- C3 versus C7/C8: learned dynamics versus parametric/exemplar retrieval,
  reported without composing them.

## Metrics

All behavioral metrics are computed per task rule, then aggregated by family
and overall with 95% CIs.

- `adaptation_delta` = post-learning minus pre-learning accuracy.
- `transfer_delta` = C3 minus C1 on unseen operands/requests governed by the
  learned rule.
- `composition_delta` = C3 minus C1 on novel rule compositions.
- `feedback_semantics_delta` = C3 minus C4.
- `causal_state_delta` = C3 minus C5.
- `state_addressing_delta` = correct C3 state minus C6 swapped state.
- `trace_free_survival` = post-deletion minus immediate pre-deletion behavior.
- `specificity_delta` = C3 minus C1 on unrelated tasks.
- `continual_retention_curve` = accuracy on prior rules after each subsequent
  learned rule in a 20-rule sequence.
- `persistent_bytes` = serialized `S` plus any experience-dependent cortex
  state remaining after deletion.
- `update_latency_ms`, `consolidation_latency_ms`, and
  `articulation_latency_ms`.
- `behavior_delta_per_byte`, displayed alongside raw behavior, storage, and
  latency rather than used as a lone verdict.

## Acceptance and kill criteria

**ACCEPT H-META-CORTEX** only if all conditions from Research/20 hold:

1. C3 adaptation, transfer, and composition deltas are positive with 95% CIs
   excluding zero.
2. C3 beats C2 and C4 on their matched comparisons with 95% CIs excluding
   zero.
3. Active correct cortex state beats zeroed and swapped state.
4. Trace survival and specificity remain inside the Phase-0-frozen equivalence
   margins.
5. Fixed-width, trace-deletion, no-text-injection, and frozen-organ hash audits
   pass.

**REFUTE** if both oracle gates pass but any acceptance item fails.

**BLOCKED** if an oracle gate fails, the v1 manifest/distribution audit lacks
human sign-off, or fewer than 30 independent meta-test rules survive validation
for any claimed family.

**KILL / interpretation rules:**

- C1 ≈ C3: online learning did not matter.
- C2 ≈ C3: meta-training did not matter.
- C4 ≈ C3: update magnitude, not feedback meaning, drives behavior.
- C5 ≈ C3: behavior is leaking around cortex state.
- Same-rule retention without positive transfer and composition is reported as
  neural association storage, not behavioral learning.
- C7/C8 wins cannot be claimed as C3 metabolism.
- No meta-test-specific tuning, extra cortex capacity, soft-bank widening, or
  new task prompt is allowed. Such work requires a new spec/version.

## Expected failure modes

1. **Latent interface cannot articulate.** The soft bank changes logits but not
   coherent behavior. Result: BLOCKED at Phase 1; improve the developmental
   coupler in a new spec, not on meta-test.
2. **Writer learns generic update loudness.** C4 matches C3. Result: refute the
   feedback-learning claim.
3. **Association without rule transfer.** Retention passes while transfer or
   composition fails. Result: cortex stores mappings but has not learned a
   reusable behavioral rule.
4. **Fast state works but consolidation fails.** Pre-clear behavior improves,
   post-clear behavior disappears. Result: learned working memory, failed
   slow-memory consolidation.
5. **State interference.** Sequential rules overwrite prior ones. Report the
   full retention curve; do not hide endpoint failures behind mean accuracy.
6. **Frozen organ too weak.** Oracle text or oracle latent code fails. Result:
   BLOCKED for that task family.

## Artifacts to add

- `src/oczy/experiments/meta_cortex/`
  - `config.py` — v1 architecture and frozen run configuration.
  - `model.py` — writer, fast/slow matrices, consolidator, read rule, coupler.
  - `taskgen.py` — deterministic A/B/C task-family generators.
  - `instrument.py` — manifest creation/verification and split firewall.
  - `oracle_gate.py` — text and latent controllability gates.
  - `train_outer.py` — developmental unrolled meta-training.
  - `run_meta_test.py` — C0–C9 evaluation and causal interventions.
  - `scoring.py` — behavior metrics with task-level CIs.
  - `audit.py` — trace, prompt, latent-width, state-size, and model-hash audits.
  - `instrument/v1/` — frozen configs, seeds, prompts, scorers, manifest.
- `src/oczy/experiments/tests/test_meta_cortex_*.py`
  - writer/consolidator math and deterministic state transitions;
  - no online optimizer/backprop in meta-test;
  - meta split non-overlap and manifest integrity;
  - trace deletion and fixed-width audits;
  - zero/swap/shuffle causal controls;
  - frozen-organ hash invariance.
- `experiments_logs/<date>_s20_meta_trained_cortex.md` — final run record.
- `reports/meta_cortex/` — developmental and immutable meta-test JSON reports.

## Reproduce (once implemented)

```bash
# Build and audit the separate measuring instrument. This does not touch eval/v2.
uv run python -m oczy.experiments.meta_cortex.instrument \
  --version v1 --materialize --audit-distributions

# Human signs off the generated v1 manifest, margins, and task counts here.

# Developmental outer-loop training; meta-test remains inaccessible.
uv run python -m oczy.experiments.meta_cortex.train_outer \
  --instrument v1 --developmental-seeds 5

# One-shot frozen meta-test after sign-off.
uv run python -m oczy.experiments.meta_cortex.run_meta_test \
  --instrument v1 --conditions C0,C1,C2,C3,C4,C5,C6,C7,C8,C9 \
  --evaluation-seeds 5 --delete-traces --audit
```
