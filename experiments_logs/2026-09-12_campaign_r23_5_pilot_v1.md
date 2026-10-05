# R23.5 DEV serialization pilot v1 — September 12, 2026

**Classification: VALID DEV PILOT; learned numeric-state persistence observed.**
The frozen Qwen organ expresses a learned suffix rule after a fresh process
reloads only numeric state, with 4/4, 4/4 and 3/4 unseen inputs correct across
three seeds. The date pattern scores 2/4, 2/4 and 4/4, but its contextual
recovery ratio is undefined. This is a small supervised soft-prompt result,
not compression, a learned cortex writer, or a full R23.5/R20 verdict.

## Authorization and frozen design

The user replied **“go ahead”** to the proposed valid held-out serialization
pilot. Authorization and the complete configuration were recorded before
model execution in the [pilot manifest](../experiments/r23.5-serialization-dev/pilot_v1/MANIFEST.json),
hash `c323a00b9bd8740c6574918ddb0c29205ba3e4ed044a805638fd5898b1f4e1ec`.

There are two DEV patterns, three teaching examples each, and four disjoint
held-out inputs each. Append `zor` uses cat/dog/fox for teaching and
sun/rain/snow/wind for probing. Date conversion uses three ISO dates for
teaching and four different dates for probing, with day/month/year targets.
The examples identify one candidate within each declared finite rule family;
this is not an assertion of uniqueness over all possible functions.

All patterns and seeds ran regardless of reference scores. Eight input soft
vectors of width 896 were trained independently for seeds 0, 1 and 2, with
24 Adam updates at learning rate 0.03, initial standard deviation 0.02 and
gradient clipping at 1.0. Each update averages teacher-forced CE across all
three teaching examples, including EOS. Only the final checkpoint is used.
Greedy generation allows 32 answer tokens; model-written summaries allow 64.
The existing normalized exact scorer and 30% recovery reference are unchanged.

## Every registered comparison

Counts are over four unseen inputs. The three seeds repeat those same four
inputs; pooled totals must not be treated as twelve independent test inputs.

| Condition | Suffix | Dates |
|---|---:|---:|
| No context | 0/4 | 0/4 |
| Original examples in context | 3/4 | 0/4 |
| Raw context saved/reloaded | 3/4 | 0/4 |
| Model-written text summary saved/reloaded | 0/4 | 0/4 |
| Learned state reloaded, seed 0 | 4/4 | 2/4 |
| Learned state reloaded, seed 1 | 4/4 | 2/4 |
| Learned state reloaded, seed 2 | 3/4 | 4/4 |
| Initial state, seeds 0 / 1 / 2 | 0/4 / 0/4 / 0/4 | 0/4 / 0/4 / 0/4 |
| Zeroed state, seeds 0 / 1 / 2 | 0/4 / 0/4 / 0/4 | 0/4 / 0/4 / 0/4 |
| Other-pattern state, seeds 0 / 1 / 2 | 0/4 / 0/4 / 0/4 | 0/4 / 0/4 / 0/4 |

Suffix recovery is **133.3%, 133.3%, 100%** across seeds, or **122.2% mean**,
using `(accuracy_state - accuracy_no_context) /
(accuracy_context - accuracy_no_context)`. All three clear the unchanged 30%
reference. Values above 100% follow directly from beating the 3/4 contextual
score; the ratio is not clamped. Date recovery is **undefined for every
condition**, because its contextual improvement is zero. Its positive learned
scores establish only this pilot's supervised generalization observation.

Raw replay reproduces every prompt hash, target, generated string and score
exactly. The suffix summary incorrectly reduces `zor` to adding `z` and
mentions animals. The date summary repeats examples and reaches its 64-token
budget instead of stating the rule. Both unedited summaries and every failed
answer are retained. The suffix seed-2 failure emits `rains` for `rain`.
Date failures include wrong ordering, repeated date components and wrong
components; there is no output repair or post-hoc probe filtering.

## Optimization and controls

| Pattern | Seed | First loss | Last logged loss | Final teaching fit |
|---|---:|---:|---:|---:|
| Suffix | 0 | 10.620738 | 0.005982 | 3/3 |
| Suffix | 1 | 10.795440 | 0.017062 | 3/3 |
| Suffix | 2 | 10.799067 | 0.005344 | 3/3 |
| Dates | 0 | 1.706246 | 0.015999 | 2/3 |
| Dates | 1 | 1.836724 | 0.056001 | 2/3 |
| Dates | 2 | 1.693119 | 0.001790 | 3/3 |

Losses are logged **before** their respective updates; the final teaching
generations use the final state after update 24. Lower teacher-forced loss
does not guarantee correct free generation: date seeds 0 and 1 still miss a
teaching answer. All 144 updates are present, with finite, nonzero state
gradients and no model-parameter gradients. Gradient norms before clipping
range from 0.024087 to 913.267761, median 2.025277; 93/144 updates exceed the
preselected clipping value. No setting was changed in response to this
distribution, and this pilot does not establish an optimal clipping threshold.

Initial and zeroed banks preserve the eight-vector input shape. Swapped
banks use the other pattern's trained state at the same seed. Their zero
scores support dependence on the matching learned state; this does not show
semantically correct transfer of the donor rule to the other input domain.

## Representation size

| Representation | Suffix | Dates |
|---|---:|---:|
| Raw teaching text | 139 bytes / 37 tokens | 172 bytes / 91 tokens |
| Model-written summary | 84 bytes / 20 tokens | 89 bytes / 64 tokens |
| Learned FP32 payload per seed | 28,672 bytes | 28,672 bytes |
| Actual NumPy file per seed | 28,800 bytes | 28,800 bytes |

The learned payload is about **206 times** the suffix raw text and **167 times**
the date raw text. Serialization is demonstrated here; storage compression
is not. These sizes exclude shared model/runtime, fixed prompts and metadata.

## Execution and evidence integrity

All four local offline phases completed successfully in **667.9 seconds**
combined: reference 36.5 s, text reload 26.8 s, training 478.1 s, numeric reload
126.5 s. They used distinct processes and the historical CPU runtime:
Python 3.12.13, Torch 2.10.0+cpu, TorchAO 0.17.0, Transformers 5.0.0,
Tokenizers 0.22.2, Safetensors 0.7.0 and Accelerate 1.14.0.

The runtime manifest is
`a6214355c1c6b9192d435e62f3add6bef5db8c3a6c1cf3a55cb2a9dbfc91182e`.
Before/after organ identity is
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`
in all four phases. The normalized exact scorer identity is
`e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742`.
All ten model artifacts and runtime identity were checked before launch.

The worker's loading API rejects forbidden splits. Offline Bubblewrap masks
probe files during training and masks original teaching files, training logs,
text artifacts and the pattern audit during numeric reload. Filesystem
preflights found zero visible forbidden files (1 / 8 / 14 / 29 checked for
reference / text reload / training / numeric reload). Numeric reload consumes
only released arrays/metadata plus probe inputs and targets; targets enter
the scorer after generation, never model messages. This is an execution-path
data-separation check, not a hostile-code sandbox claim about every possible
copy elsewhere on the workstation.

The original eval manifest, pilot source/input hashes, twelve numeric files
and four text artifacts verify. Nine focused pilot tests passed before
execution. Reporting guards checked every registered condition/seed, all
144 ordered steps, phase identities, zero evaluation updates and exact raw
replay. Ruff and diff whitespace checks passed.

- [Complete machine-readable report](2026-09-12_r23_5_pilot_v1.json): all 128
  held-out condition outputs, trajectories, fits, scores and execution records.
- [Durable artifact archive](artifacts/2026-09-12_r23_5_pilot_v1/README.md):
  initial/trained numeric states, raw/summarized text, execution logs,
  frozen instrument and executable source snapshots.
- [Frozen pilot design](../experiments/r23.5-serialization-dev/pilot_v1/README.md).

## Remaining blocker

The input soft-prompt path now has narrow evidence of learning, persistence
and expression through this frozen language model. The next scientific gate
is confirmation on fresh, separately frozen patterns with a positive context
effect, keeping the same optimization settings and matched controls. These
already-inspected probes must not become a fresh confirmation set. State-size
reduction or an alternative coupling mechanism must be a separate comparison.

A per-pattern gradient optimizer is not a trained online cortex writer.
No hidden-layer-vector sweep, population-level confidence claim, full R23.5
accept/refute, R20 task repair, recalibration, remote job or meta-test ran.
R20 remains blocked by its task-support defects and power/candidate gates;
R21/R22 Stage B remain dependent on R20. This local pilot is complete and
does not start or resume the older autonomous research services.
