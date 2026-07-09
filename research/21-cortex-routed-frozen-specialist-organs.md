# 21 — Cortex-routed frozen specialist organs: mouth, hands, and shared state

**Pre-registered 2026-07-09** (human-authorized before implementation).
Agents running this experiment MUST NOT edit this spec. Deviations are reported
as deviations. Depends on Research/20 accepting H-META-CORTEX on at least one
task family. If Research/20 refutes, this project is **BLOCKED** rather than a
license to hide the failure behind specialist models.

## Problem

Experiment/08 asks one small general LM to learn tool-call syntax, tool
selection, parameter extraction, goal retention, and result integration at the
same time. That confounds cortex learning with whether the language model is a
competent action decoder.

The intended architecture does not require one model to be the whole organism.
The cortex is the persistent learner and controller. Independently frozen
models can be organs:

- a language organ articulates natural-language answers;
- an action organ emits structured tool calls;
- later organs may handle code, vision, or other modalities.

Research/21 isolates whether learned cortex state can select and condition
these organs while retaining a task goal across turns. Retrieval remains an
external baseline and is disabled in the primary condition.

## Hypothesis

**H-MULTI-ORGAN:** a meta-trained cortex can learn the semantics of previously
unseen, opaquely named tools from experience and feedback, preserve the task
goal across tool-result turns, route each step to the correct frozen specialist
organ, and complete held-out tool chains after raw-trace deletion.

Success must depend causally on cortex state, not tool-name priors, prompt
examples, an answer-time exemplar store, or online updates to either specialist
model.

## Architecture

### Frozen organs

1. **Language organ:** the frozen HF language model used by Research/20.
2. **Action organ:** a small structured-action model selected before main runs
   by a capability and latency record. It receives a fixed-width cortex latent
   plus the current observation and emits a schema-valid action. Its weights
   remain frozen during developmental and evaluation episodes unless a new
   human-approved spec explicitly permits joint developmental training.

Both organs expose model hashes and a deterministic greedy mode. If no action
model passes the oracle gate, the project is BLOCKED.

### Cortex-owned components

- the Research/20 fast and slow state plus learned update/consolidation rules;
- an organ router producing `language`, `action`, or `abstain`;
- one fixed-width latent coupler per organ;
- a recurrent task-goal state that is part of cortex fast state; and
- confidence/uncertainty outputs used for abstention.

The router and couplers are developmentally trained across training tool
families, then frozen for meta-test. During a new task only cortex fast/slow
state may change.

## Tool-family instrument

The instrument is versioned separately from both `eval/v2` and Experiment/08.
It uses synthetic tools with opaque names such as `op_k7` and `op_r2`; names are
randomized independently of semantics so pretrained token priors cannot solve
routing.

Each generated tool family specifies:

- JSON schema and parameter types;
- deterministic tool implementation in a sandbox;
- a natural-language teaching interaction with success/correction feedback;
- held-out requests using unseen parameter values;
- ambiguous requests where multiple tools share surface vocabulary;
- two- and three-step chains with tool results; and
- unrelated language-only turns testing abstention and specificity.

Meta-train/validation/test split by complete tool semantics and chains, not
names or prompt paraphrases. Tool names are freshly permuted for every task.
Generator, scorer, sandbox, seeds, and thresholds receive a versioned manifest
and the same DEV distribution-check procedure as Research/20.

## Oracle gates

Before cortex evaluation:

1. **Action-organ gate:** with the correct tool schema, correct route, complete
   task goal, and one worked example supplied directly, the frozen action organ
   must produce a schema-valid correct call and parameterization significantly
   above its no-information baseline.
2. **Language-organ gate:** with the original goal and complete tool results
   supplied directly, the frozen language organ must produce the correct final
   answer significantly above baseline.

A failed gate blocks that family. It is not evidence against cortex memory.

## Conditions

| ID | Condition | Purpose |
|---|---|---|
| C0 | Frozen language organ only | No action specialist, no cortex |
| C1 | Frozen action organ with full oracle routing/context | Capability upper bound |
| C2 | Meta-trained router/couplers, online cortex update disabled | No-learning control |
| C3 | Meta-trained cortex, correct feedback, traces deleted | Primary multi-organ condition |
| C4 | C3 with feedback shuffled | Feedback-semantic control |
| C5 | C3 with recurrent task-goal state zeroed after first tool result | Goal-retention causality |
| C6 | C3 with router logits permuted | Routing causality |
| C7 | C3 with another task's cortex state swapped in | State-addressing control |
| C8 | Byte-matched retrieval/tool-example baseline | External product baseline only |

C2/C3 isolates online cortex learning. C3/C5 isolates goal state. C3/C6
isolates routing. C1 identifies whether failures belong to an organ or cortex.

## Protocol

1. Developmentally meta-train the cortex router, couplers, write rule, and goal
   state on training tool families; keep all organ weights frozen.
2. Freeze developmental parameters before meta-test.
3. Present one to five teaching interactions for a completely unseen tool
   family. Apply only the learned cortex update rule.
4. Consolidate and delete teaching text, tool examples, results, and transient
   traces; verify count zero.
5. Score unseen single-tool requests, ambiguous requests, multi-step chains,
   post-tool answer integration, and language-only abstention.
6. Replay the scored trajectories under C4–C7 interventions without relearning.
7. Run the existing Pi benchmark only as an external post-verdict battery. It
   cannot define or reverse the primary verdict.

## Primary metrics

1. `organ_route_accuracy` — correct specialist selected per step.
2. `tool_selection_accuracy` — correct opaque tool selected.
3. `parameter_accuracy` — schema-valid parameters with correct values.
4. `goal_retention_accuracy` — correct next action/final answer after a tool
   result when the original request is no longer repeated.
5. `chain_completion_rate` — all required actions and final response correct.
6. `feedback_semantics_delta` — C3 minus C4.
7. `goal_state_delta` — C3 minus C5 on post-result turns.
8. `routing_delta` — C3 minus C6.
9. `state_addressing_delta` — C3 minus C7.
10. `language_specificity_delta` — change on language-only turns relative to C2.
11. Persistent bytes, update latency, and per-organ inference latency.

All estimates are reported per tool family and pooled, with independent task
counts, seeds, and 95% CIs.

## Acceptance

**Accept H-MULTI-ORGAN** only if:

- C3 beats C2 on tool selection, goal retention, and chain completion with 95%
  CIs excluding zero;
- C3 beats C4, C5, and C6 on their matched primary metrics;
- correct state beats swapped state;
- specificity remains within the DEV-frozen equivalence margin;
- trace deletion and both organ-hash audits pass; and
- at least one held-out tool family requires a two-step chain.

**Refute** if both oracle gates pass but any primary condition fails.

**Blocked** if Research/20 does not accept, an oracle gate fails, or the
instrument is not manifest-frozen with human-approved distribution checks.

## Interpretation constraints

- Correct tool JSON from the action organ is not by itself cortex learning; C3
  must beat C2 and depend on correct feedback.
- Correct routing without post-result goal retention is a router result, not a
  multi-organ organism result.
- A retrieval baseline win is recorded honestly but cannot accept
  H-MULTI-ORGAN.
- Tool names, schemas, or demonstrations may not remain in the primary answer
  path after consolidation.
- Adding another specialist model after seeing meta-test failures requires a
  new pre-registered spec.

## Reporting

Report model-selection and oracle-gate records first, then developmental curves,
one-shot meta-test tables, intervention results, deletion/hash audits, resource
costs, and the external Pi battery. Log to
`experiments_logs/<date>_s21_multi_organ_cortex.md`.
