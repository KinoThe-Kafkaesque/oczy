# Contextual learning and preservation DEV v2 — September 13, 2026

**COMPLETE: all 27 teaching pairs fit, but only 6/24 new-word affix applications
succeed. Reliable scoped acquisition is not established, so the preservation
comparison is blocked at its frozen prerequisite with zero correction updates.**

The associated language-interface work improves direct operation prompts to
15/16 on both calibration and new words. A separate decoder repair qualifies
on all 64 comparisons with scalar output. These are distinct results:

- [Language interface: prompts, new-word results and limits](2026-09-13_language_interface_dev_v3.md).
- [Decoder: isolated defect, implemented candidate and qualification](2026-09-13_decoder_parity_dev_v1.md).
- [This experiment's complete scores, outputs and trajectories](2026-09-13_context_preservation_dev_v2.json).
- [Frozen protocol](../experiments/context-preservation-dev-v2/README.md).
- [Reproducibility archive](artifacts/2026-09-13_context_preservation_dev_v2/README.md).

Manifest `c68b31b35fb9c5f12dd9e68174ee26313145a4c2a3e262255bc2e23bb2454390`.
The user's September 13 request authorized new local DEV work on scoped learning,
preservation and language interfaces. Source base is
`e71268bc9cb01839995e5b3db69d0ad43a9ef04d` with hash-frozen local uncommitted
overlays. No earlier instrument, scorer, threshold or result was changed.

## Curriculum audit and fixed learning experiment

The earlier curriculum correlated input identity with client: amber and cobalt
had different teaching words. The new curriculum crosses the same oak/pine/elm
inputs with amber (append vek), cobalt (append mip) and silver (copy unchanged).
Each input has three distinct targets, so perfect teaching fit requires context
to affect the output. Quartz is never taught and tests the default copy rule.
Silver is a taught neutral context here, unlike unassigned silver in DEV v1.

One shared 8 x 896 FP32 bank is trained jointly on all nine examples. Seeds
0/1/2 receive exactly 48 full-batch Adam updates each, learning rate .03,
initialization standard deviation .02, clip norm 1, mean example CE including
EOS. The concise system/query is chosen before any prompt results. No checkpoint
selection, held-out optimization or adaptive changes occur. The final numeric
bank alone crosses into a new evaluation process.

This is a joint representational diagnostic. Its gradient updater is a DEV
serializer, not R20's primary learned writer without online backpropagation.
It does not demonstrate sequential accumulation. Curriculum, query interface,
training schedule and fresh words differ from DEV v1, so cross-version results
do not isolate the causal contribution of any one change.

## Teaching fit and fresh-process result

All 144 updates complete, with 1,296 teaching-example loss evaluations. All
gradients are finite and nonzero; no organ parameter receives gradients.
Every seed fits all nine teaching pairs exactly. Initial/final pre-update mean
CE values are 7.513/.05475, 7.419/.03987 and 7.251/.00379. These are teaching
losses, not evidence of a general rule or preservation.

The confirmation words are lime/melon/grape/peach, frozen before any model call.
Each accuracy cell below is out of four. Every initial bank and the zeroed
control score amber 0/4, cobalt 0/4, silver 4/4 and quartz 4/4.

| Restored learned bank | Amber | Cobalt | Taught neutral silver | Untaught quartz | Total |
|---|---:|---:|---:|---:|---:|
| Seed 0 | 1/4 | 0/4 | 4/4 | 3/4 | 8/16 |
| Seed 1 | 2/4 | 1/4 | 4/4 | 4/4 | 11/16 |
| Seed 2 | 1/4 | 1/4 | 2/4 | 0/4 | 4/16 |

Across seeds, new affix applications are 6/24; neutral copying is 17/24 versus
24/24 initially. Untaught quartz falls to 7/12 versus 12/12 initially. The
learned state sometimes applies a new rule, but cannot reliably generalize
within the correct context or preserve default behavior. These are repeated
four-word trials across three initializations, not 48 independent tasks.

Examples from restored seed 0: amber/lime correctly returns ` limevek`, but
amber/melon returns ` melon`, amber/grape returns ` greek`, and cobalt/melon
returns ` melip`. Quartz/grape returns ` grapes` instead of copying `grape`.
The exact scorer strips only its existing whitespace/Unicode normalization;
wrong words, suffixes and punctuation remain failures.

The bank is restored from its verified 28,800-byte .npy file (28,672 payload
bytes). The separately run decoder diagnostic reproduces all 16 seed-0 scalar
outputs, including errors, and shows that the qualified batch candidate agrees
with them. A batch-decoder defect therefore does not explain these scalar-path
learning failures. No compression claim is made in this experiment.

## Why preservation did not run

Admission requires all nine teaching fits and all 16 fresh responses correct
for every seed. Teaching fit passes; fresh behavior fails. The controller writes
`admission.json` with `admitted: false` and an explicit `blocked.json`; both
correction workers remain unstarted. The planned unprotected-versus-proximal
comparison therefore has **zero optimizer updates and no result**.

This distinguishes failure to acquire a dependable rule from forgetting a
previously mastered rule. It does not refute the proposed preservation penalty,
prove that an eight-token bank is mathematically incapable, or establish an
optimization optimum. More updates, different curricula and mechanisms were
not searched after observing this failure.

## Fixed language-interface ladder in this version

Calibration uses the earlier pear/plum/kiwi/fig words. Stage-2 confirmation
uses the new words above, with both suffixes active. Each cell is out of 16.

| Condition | Calibration | Confirmation |
|---|---:|---:|
| Original full-rule prompt | 2/16 | 3/16 |
| Query reformatted into separate lines | 3/16 | 5/16 |
| Concise system instruction, same full rules | 2/16 | 6/16 |
| Explicit input-plus-suffix rule table | 2/16 | 3/16 |
| Oracle selects applicable operation | 8/16 | 9/16 |
| No context | 8/16 | 8/16 |
| Crossed-example prefix retrieval | 1/16 | 2/16 |
| Logit-bias / rerank | Not run | Not run |

Only adjacent changes through the table condition isolate one interface element.
The oracle-selected condition removes the selection problem and cannot prove
learning it. Its confirmation score comprises one correct suffix and all eight
identity responses. Stage-3 confirmation, with amber changed to zul, gives the
table 0/16 and the selected-operation oracle 9/16. There is no stage-3 learned
bank result because correction is not admitted.

The subsequent separately frozen DEV-v3 comparison finds a clearer operation
interface: minimal explicit append/copy commands score 15/16 in both word sets,
including 7/8 affix applications. It remains an oracle-selected primitive control;
complete scoped rules and example retrieval still fail substantially. No
post-result prompt is substituted into this learning run.

## Audit, disposition and next experiment

Independent audit verifies all registered source/input and numeric-state hashes,
exact trial/update coverage, every prompt hash, unchanged pre/post model and
scorer identities, 368 probe scores and 27 teaching scores. The original full-
oracle prompt reproduces all 12 corresponding DEV-v1 calibration outputs.
All three phases are separate offline processes; forbidden registered paths
are checked absent. These are audited worker boundaries, not a guarantee against
arbitrary malicious host code. No meta-test is accessed.

Recorded phase durations sum to 2,545.74 seconds (42.43 minutes): reference
289.22, training 2,165.88, restore 90.64. Independent interface and decoder
processes ran alongside portions of training. The combined focused regression
suite passes 73 checks; all three experiments' independent audits pass.

The next scientific blocker is **generalizing a learned rule to new inputs in
the correct context while retaining neutral behavior**. Joint teaching fit is
now demonstrated; reliable acquisition is not. A suitable next isolated
curriculum comparison increases teaching-word diversity while keeping the bank,
per-update example count, total updates, optimizer and scoring fixed, with a new
frozen confirmation set. Retain retrieval and minimal direct-operation oracle
controls, and keep preservation behind the same acquisition prerequisite.
That further optimizer comparison was not run in this version.

The decoder repair is separately qualified and available as a versioned helper.
Future batched research needs a versioned adoption; historical R20 and legacy
task-support repairs and meta-test gates remain separate. No old autonomous
loop, remote campaign, commit, push or production action was started.
