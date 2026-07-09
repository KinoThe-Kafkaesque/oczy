# 19 — The LM as language organ: direct cortex learning, two articulation paths

**Originally pre-registered 2026-07-03** (human-approved, before
implementation). **Human-approved amendment 2026-07-09, still before
implementation:** the original label-prefix articulation path is preserved as
Arm A and explicitly classified as **parametric retrieval**. A matched
latent-control path, Arm B, is added as the primary test of whether learned
cortex state can control a frozen language organ without retrieving answer
content. This amendment does not modify `eval/v2`; it closes an interpretation
loophole in the architecture under test.

Agents running this experiment MUST NOT edit this spec. Deviations are reported
as deviations. Research/19 is a direct-online-learning diagnostic. Research/20
is the successor that meta-trains the cortex's update and consolidation rules.

## Problem / reframe

Every failed mechanism tried to derive a useful LM control direction directly
from an embedding of the correction. Cvec steering was refuted, the layer-L
assumption failed on two architectures, and the cortex never received a
behavior-aligned training loop. The one component that worked reliably was a
retrieval reranker.

The intended architecture is different: the LM is a frozen **language organ**
for perception and articulation, while experience changes a trainable cortex
outside it. The first version of this spec conditioned articulation by supplying
the cortex's winning label as text. That is a useful comparator, but it can be
understood as a classifier retrieving content from its parameters and handing
that content to the LM. Deleting raw correction text does not by itself make
that path changed dynamics.

Research/19 therefore asks two separate questions on the same learned cortex:

1. Can a small cortex learn a request-to-label mapping that survives raw-trace
   deletion? This is **parametric retrieval** (Arm A).
2. Can the same cortex alter a frozen LM through a fixed-width latent interface,
   with no label or answer text injected at probe time? This is the primary
   **latent-control** test (Arm B).

## Hypotheses

**H-LABEL (comparator):** a small head trained online from correction events on
frozen LM embeddings maps requests to sense labels, transfers to paraphrases,
and retains the mapping after raw-text deletion when its predicted label is
supplied to the LM as text.

**H-LATENT (primary):** using the same perception features, correction events,
parameter budget, and teaching order, a cortex with a learned latent
articulation coupler produces held-out retention and transfer after raw-trace
deletion while:

- the language-organ weights remain bit-identical;
- no predicted label, expected answer, correction text, exemplar, or retrieved
  content enters the LM prompt at probe time; and
- zeroing or swapping the learned cortex state causally removes or redirects the
  learned behavior.

A positive H-LABEL result cannot accept H-LATENT.

## Architecture (fixed here)

- **Frozen language organ:** `HFDriver` with
  `Qwen/Qwen2.5-0.5B-Instruct`; model parameters and tokenizer are hashed before
  and after every run.
- **Perception:** final-layer, mean-pooled frozen LM features, the S1.4 winner.
- **Shared cortex:** a trainable head of at most 64k persistent parameters. It
  receives the request feature and is updated online from the correction the
  teacher actually supplied. Eval `expected` strings and scorer outputs are
  never training inputs.
- **Arm A — `label_prefix`:** the head predicts a sense label and that label is
  supplied as a short text prefix. This preserves the original Research/19 path
  and is reported as parametric retrieval.
- **Arm B — `latent_control`:** a learned cortex-owned coupler maps the
  query-conditioned cortex state to a fixed-width bank of soft embeddings or KV
  entries consumed by the frozen LM. The bank contains no decoded text and has
  a fixed shape independent of episode count. The coupler is trained on DEV,
  then frozen before any holdout run; its parameters count toward the 64k
  budget.
- **Abstain path:** below a DEV-calibrated confidence threshold, both arms fall
  through to the unmodified language organ.
- **No meta-learned update yet:** Research/19 uses one fixed optimizer and
  direct online gradients on the cortex head. Learning the update rule itself
  is Research/20.

## Hard boundary: what Arm B may and may not transmit

Allowed at probe time:

- the current request;
- frozen LM features of that request;
- query-conditioned activations computed from persistent cortex parameters;
- the fixed-width latent control bank.

Banned at probe time:

- correction text or its tokens;
- predicted label text;
- expected-answer text;
- stored exemplars, nearest-neighbor matches, episode IDs, or raw traces;
- a variable-length latent bank that grows with experiences;
- any LM-parameter update.

The runner must emit a machine-checkable articulation audit containing prompt
text, latent-bank shape, raw-trace count, language-organ hash, and persistent
cortex bytes for every scored condition.

## Protocol

1. **Instrument:** current human-approved frozen eval version at run time
   (v2.1 as of this amendment), with its manifest verified before and after.
2. **Phase 0 distribution check on DEV only:** measure no-update repeatability,
   confidence, and specificity distributions. Freeze the abstain threshold and
   the specificity equivalence margin in the run manifest before holdout. They
   require human sign-off and may not be changed after seeing holdout.
3. **Coupler development:** train Arm B's articulation coupler on stage-0 DEV
   tasks only. Freeze it before the final teaching/evaluation runs. Arm A's
   label phrasing is frozen at the same point.
4. **Online teaching:** initialize a fresh cortex head for each seed; teach
   stage-0 corrections in seed-shuffled order using the same examples and
   optimizer for Arms A and B.
5. **Consolidation:** finish head training, serialize cortex parameters, delete
   all correction texts and transient optimizer examples, and verify raw-trace
   count zero.
6. **One-shot evaluation:** stage-0 holdout retention; the complete, untaught
   stage-1 transfer battery; stage-2 holdout scope; untaught stages for
   specificity.
7. **Causal intervention:** rerun Arm B with learned state active, zeroed, and
   swapped with a different seed/task state. No relearning occurs between arms.
8. **Seeds:** at least 5. A fallback to 3 is allowed only if a measured seed
   exceeds 15 minutes and is reported as a deviation. Vanilla is mandatory.

## Conditions / matched comparisons

| ID | Condition | Purpose | Matched variable |
|---|---|---|---|
| C0 | Frozen language organ only | Vanilla baseline | no cortex |
| C1 | Cortex architecture, online update disabled | Architectural/no-update control | update off |
| C2 | Arm A: trained head + label prefix | Parametric-retrieval comparator | text readout |
| C3 | Arm B: trained head + latent control | Primary cortex condition | latent readout |
| C4 | C3 with cortex state zeroed after learning | Causal state test | state active/zero |
| C5 | C3 with cortex state swapped | Addressing test | correct/wrong state |
| C6 | C3 with correction labels permuted during teaching | Feedback-semantic control | correct/permuted feedback |
| C7 | S3.M2a retrieval baseline | External bar only; never attached to C3 | retrieval |

C2 versus C3 isolates articulation. C3 versus C4 isolates learned cortex state.
C3 versus C6 isolates whether semantically correct feedback drives the change.

## Primary metrics

All metrics are computed separately for C2 and C3 and reported with mean and
95% CI over seeds.

1. `retention_delta` — stage-0 holdout accuracy minus C1.
2. `transfer_delta` — complete untaught stage-1 battery accuracy minus C1.
3. `scope_delta` — stage-2 holdout accuracy minus C1; reported but non-gating.
4. `specificity_delta` — change on untaught stages relative to C1.
5. `causal_state_delta` — C3 active minus C4 zeroed.
6. `state_addressing_delta` — C3 correct state minus C5 swapped state.
7. `feedback_semantics_delta` — C3 minus C6.
8. `persistent_bytes` and `behavior_delta_per_byte`, with coupler and head bytes
   both counted.

## Acceptance and verdicts

**Accept H-LATENT** only if all of the following hold:

- C3 `retention_delta > 0` with 95% CI excluding zero;
- C3 `transfer_delta > 0` with 95% CI excluding zero;
- `causal_state_delta > 0` and `feedback_semantics_delta > 0`, each with 95%
  CI excluding zero;
- specificity remains within the DEV-frozen equivalence margin;
- raw-trace deletion, fixed latent width, no-text-injection audit, and frozen-LM
  hash all pass.

**Accept H-LABEL** separately if C2 passes the original retention, transfer,
and specificity criteria. It remains a parametric-retrieval result even if it
outperforms C3.

**Refute H-LATENT** if its validity gates pass but any primary condition fails.
If C2 accepts and C3 refutes, record: "a parametric label store works; latent
control of the frozen language organ does not yet work."

**Blocked** if the frozen LM cannot perform the task with an oracle text
demonstration on DEV, or if the latent coupler cannot produce coherent DEV
articulation before holdout. No cortex-learning verdict is drawn in either
case.

## Reporting

The report must contain per-seed C0–C7 tables, the Phase 0 distribution record,
frozen thresholds, deletion and articulation audits, model hashes, exact
commands, and separate verdicts for H-LABEL and H-LATENT. Log to
`experiments_logs/<date>_s19_language_organ_two_readouts.md`.

## Relationship to the next projects

- Research/19 asks whether direct gradient training can make a small cortex
  control one frozen language organ.
- Research/20 learns the cortex update and consolidation rules across task
  families, then tests adaptation to unseen tasks without online backprop.
- Research/21 adds independently frozen specialist organs and asks whether the
  learned cortex can route and condition them while retaining task state.
