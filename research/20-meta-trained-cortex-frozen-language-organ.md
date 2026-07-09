# 20 — Meta-trained plastic cortex over a frozen language organ

**Pre-registered 2026-07-09** (human-authorized before implementation).
Agents running this experiment MUST NOT edit this spec. Deviations are reported
as deviations. The concrete build-and-run specification is
[`experiments/09-meta-trained-cortex-frozen-language-organ/`](../experiments/09-meta-trained-cortex-frozen-language-organ/).

## Problem

Oczy's cortex had mutable state but no learned learning algorithm. A correction
embedding was accumulated through a hand-authored Hebbian rule and projected
back into the LM as if "a representation of the experience" were automatically
"the direction that makes the model use the experience." S1.3, S1.4, S2.1,
and S2.4 jointly refuted that shortcut.

Research/19 adds direct gradient training and distinguishes a label-text store
from latent control, but it still optimizes a fresh cortex directly on the task
being evaluated. It does not test whether the cortex has learned a reusable
algorithm for acquiring new memories and behavior from experience.

The missing developmental layer is an **outer loop**. Across many task
episodes, it must train:

- how experience and feedback write fast cortex state;
- how current context addresses that state;
- how fast state consolidates into slow persistent cortex weights;
- how a fixed-width latent control signal drives a frozen language organ; and
- how learning remains specific instead of overwriting unrelated behavior.

Retrieval remains a mandatory external baseline but is disabled in the primary
cortex condition. The experiment isolates the raw capability of cortex state
to carry post-experience information.

## Hypothesis

**H-META-CORTEX:** after developmental meta-training across a distribution of
learning episodes, a cortex with learned write, read, and consolidation rules
can adapt to a previously unseen task from correction, consolidate the change,
delete all raw traces, and cause a frozen language organ to execute the learned
rule on held-out and compositional probes through a fixed-width latent channel.

The hypothesis requires all of the following:

1. the task family and rule are unseen during developmental training;
2. the language-organ parameters remain bit-identical;
3. no correction, label, exemplar, or retrieved content enters the answer path;
4. no optimizer/backpropagation runs during the evaluation episode;
5. the learned cortex state causally controls the behavior; and
6. semantically wrong or shuffled feedback does not produce the same gain.

## Architecture

### Frozen language organ

- `HFDriver` with `Qwen/Qwen2.5-0.5B-Instruct` is the initial substrate.
- Its tokenizer, embedding, transformer, and LM-head parameters are frozen.
- Hashes are recorded before developmental training, after developmental
  training, and after every evaluation seed.
- It provides perception features and articulation only. A stronger frozen
  model may be introduced only in a new pre-registered spec.

### Cortex

The cortex owns every trainable or mutable component outside the frozen organ:

- developmental parameters `theta`, learned in the outer loop;
- fast state `F_t`, updated after each experience;
- slow state `S_t`, changed only by consolidation;
- learned write rule `U_theta(F_t, observation, feedback, outcome)`;
- learned consolidation rule `G_theta(S_t, F_t)`;
- query-conditioned read rule `R_theta(S_t, F_t, query_feature)`; and
- latent articulation coupler `P_theta`, which emits a fixed-width soft
  embedding/KV bank for the frozen language organ.

At evaluation time `theta` and `P_theta` are frozen. The learned rules may
change `F_t` and `S_t`; standard gradient descent, target-token optimization,
and LM updates are forbidden.

### Information boundary

The primary condition may persist only serialized cortex state. It may not
persist or consult:

- raw experience or correction text;
- tokenized traces;
- target labels or expected answers;
- exemplar embeddings or nearest-neighbor indexes;
- episode IDs or task IDs;
- variable-length memory proportional to episode count; or
- an LM KV cache produced directly from the original correction.

The latent bank is recomputed from the current query and persistent cortex
state. Fixed-width latent control is a readout of neural state, not a stored
trace.

## Developmental task distribution

The instrument is separate from `eval/v2` and must be frozen as
`meta_cortex/v1` before the main run. It contains deterministic generators for
three learning families:

1. **Contextual remapping:** an arbitrary symbol or word maps to different
   outputs under different contexts.
2. **Rule transformation:** a correction defines a small input-output rule
   that must be applied to unseen operands, not repeated verbatim.
3. **Finite-state behavior:** feedback changes a latent policy that must retain
   a goal across multiple turns.

Meta-train, meta-validation, and meta-test split by complete rules and task
instances, not surface paraphrases. Meta-test rules, output assignments, and
compositions must be unreachable from meta-training seeds.

Each task family supplies:

- a pre-learning probe set;
- one to five experience/feedback events;
- held-out same-rule probes;
- compositional probes combining learned operations;
- unrelated specificity probes; and
- a deterministic oracle-context form used only for the validity gate.

## Instrument freeze and threshold distribution check

Before any meta-test run:

1. materialize generator version, seeds, family split, scorers, and probe counts;
2. compute SHA-256 hashes into a `MANIFEST.json`;
3. run no-update and repeated-run distributions on meta-validation;
4. derive the specificity equivalence margin from the observed no-update
   repeatability distribution;
5. freeze sample size using a power analysis on meta-validation effect sizes;
6. obtain human sign-off on the manifest, margin, and sample size; and
7. never change them without a version bump.

No threshold is selected from meta-test data.

## Protocol

### Phase A — oracle controllability gate

Give the frozen language organ the complete rule and worked demonstrations in
text on meta-validation. It must beat organ-only zero-information performance
with a 95% CI excluding zero. If it cannot express the rule when fully informed,
the corresponding task family is **BLOCKED**, not a cortex refutation.

### Phase B — developmental meta-training

Unroll full learn/consolidate/probe episodes on meta-training tasks. Optimize
`theta` and the latent coupler through the post-learning behavioral loss plus
specificity and state-size regularizers. Meta-validation selects optimizer,
architecture, and stopping point. Meta-test is never observed.

### Phase C — frozen-rule evaluation

Freeze developmental parameters. For each meta-test task and seed:

1. initialize empty `F_0` and `S_0`;
2. record pre-learning behavior;
3. present experience and correction events;
4. update fast state through `U_theta` only;
5. consolidate once through `G_theta`;
6. delete all raw traces and verify count zero;
7. score same-rule, transfer, composition, and specificity probes; and
8. repeat causal interventions with cortex state zeroed, swapped, and feedback
   shuffled without rerunning the learning episode.

Use at least 5 developmental seeds and at least 5 evaluation seeds. Exact task
counts are frozen by the pre-run power analysis.

## Matched conditions

| ID | Condition | Question isolated |
|---|---|---|
| C0 | Frozen language organ only | What can the mouth do without cortex? |
| C1 | Meta-trained architecture, update disabled | Does architecture alone help? |
| C2 | Untrained/random update rule, same capacity | Does meta-training matter? |
| C3 | Meta-trained cortex, correct feedback, trace deleted | Primary condition |
| C4 | C3 with feedback shuffled during experience | Does feedback semantics matter? |
| C5 | C3 with cortex state zeroed after consolidation | Is learned state causal? |
| C6 | C3 with another task's cortex state swapped in | Is addressing task-specific? |
| C7 | Research/19 label-prefix head | Parametric-retrieval comparator |
| C8 | Byte-matched compressed retrieval | External retrieval bar, never attached to C3 |

C1/C3 isolates online state change. C2/C3 isolates developmental meta-training.
C3/C4 isolates feedback. C3/C5 and C3/C6 isolate the causal state path.

## Primary metrics

1. `adaptation_delta` — post-learning minus pre-learning accuracy on unseen
   meta-test tasks.
2. `transfer_delta` — C3 minus C1 on held-out inputs governed by the learned
   rule.
3. `composition_delta` — C3 minus C1 on novel compositions of learned rules.
4. `feedback_semantics_delta` — C3 minus C4.
5. `causal_state_delta` — C3 minus C5.
6. `state_addressing_delta` — correct C3 state minus C6 swapped state.
7. `trace_free_survival` — post-deletion score minus the score immediately
   before deletion, evaluated against the frozen equivalence margin.
8. `specificity_delta` — change on unrelated tasks relative to C1.
9. `persistent_bytes`, update latency, articulation latency, and
   `behavior_delta_per_byte` reported as a resource table, not a lone score.

All behavioral metrics report per-family estimates, pooled estimates, seed and
task counts, and 95% CIs. Seeds do not substitute for independent task rules.

## Acceptance

**Accept H-META-CORTEX** only if:

- `adaptation_delta`, `transfer_delta`, and `composition_delta` are positive
  with 95% CIs excluding zero;
- C3 beats C2, C4, and C5 on their matched primary comparisons with 95% CIs
  excluding zero;
- the correct state beats the swapped state;
- trace-free survival and specificity remain inside their DEV-frozen
  equivalence margins;
- fixed-width and deletion audits pass; and
- all frozen-language-organ hashes match.

**Refute** if validity gates pass but any acceptance condition fails. A C7 or C8
win does not rescue C3 and cannot be reported as cortex metabolism.

**Blocked** if the oracle controllability gate fails or if the instrument lacks
human-approved distribution checks and manifest freeze.

## Kill and interpretation rules

- If C1 matches C3, the online update is unnecessary.
- If C2 matches C3, the learned update rule did not earn its role.
- If C4 matches C3, the system is reacting to generic update magnitude rather
  than learning from feedback content.
- If C5 matches C3, the behavior does not causally depend on persistent cortex
  state; inspect leakage before any rerun.
- If only same-rule retention passes but composition fails, record the result as
  neural association storage, not learned behavioral dynamics.
- No task-specific hyperparameter rescue is allowed on meta-test. It belongs in
  a new spec.

## Reporting

Log developmental curves separately from the one-shot meta-test report. The
final report includes the frozen manifest, task-family split, oracle gate,
condition tables, causal interventions, trace and hash audits, resource table,
exact commands, and all nulls. Write to
`experiments_logs/<date>_s20_meta_trained_cortex.md`.

## Dependency

Research/20 is the core cortex experiment. Research/21 may begin only after
H-META-CORTEX accepts on at least one rule family and the latent interface
passes its causal-state audit.
