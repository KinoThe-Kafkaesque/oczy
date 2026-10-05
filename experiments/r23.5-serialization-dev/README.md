# R23.5 DEV serialization experiments

This is the concrete next DEV workflow requested on September 11, 2026.
It does not change the frozen R20 instrument or authorize an R20 meta-test.
The full R23.5 proposal remains distinct from this preliminary interface check.

## September 12 pilot result

The user-approved [frozen pilot v1](pilot_v1/README.md) is complete. After
training on three examples and reloading only numeric state in a fresh
process, suffix accuracy is **4/4, 4/4, 3/4** and date accuracy is
**2/4, 2/4, 4/4** across three seeds. Initial, zeroed and swapped states all
score zero. Raw context/reload score 3/4 suffix and 0/4 dates; both model-written
summaries fail. Date recovery is undefined. The 28,672-byte learned payload
is much larger than either raw context, so this is persistence, not storage
compression. See [the full result and archived evidence](../../experiments_logs/2026-09-12_campaign_r23_5_pilot_v1.md).

The next gate is confirmation on fresh, separately frozen patterns, preserving
the pilot's fixed optimizer and controls. The completed pilot does not include
the full draft's hidden-layer sweep, establish a learned cortex writer, or
repair/authorize R20. Dated prerequisite results below remain historical.

## September 11 outcome

**September 12 follow-up:** [V5 corrective-facts presentation](V5_CORRECTIVE_FACTS.md)
now scores **4/7**, including all four directly taught queries, versus **0/7**
for both original dialogue and no context. Original v4 baselines reproduce
exactly. This resolves the sampled taught-answer presentation blocker; it is
not yet a held-out serialization result. Three contextual queries were never
taught. The [coverage audit and result](../../experiments_logs/2026-09-12_campaign_dev_teaching_format.md)
also identify task-support/composition defects that must be repaired before
full R20 conclusions. The dated September 11 outcome below remains historical.

The user approved [Amendments A then B](APPROVAL.md). Both were materialized
and run separately under the verified historical runtime/model identity.
The corrected v2 baseline scored no-context/teaching/oracle `0/7, 0/7, 0/3`;
v3 scored `0/7, 0/7, 1/3`; v4 scored `0/7, 0/7, 2/3`.
See the [complete diagnostic log](../../experiments_logs/2026-09-11_campaign_r20_dev_output_path.md).

The oracle prompt defects are repaired within this DEV scope. At that stage,
teaching-context articulation remained the next blocker; its zero improvement left the
serialization recovery ratio undefined. No serialization optimization ran.

The versioned inputs live in `instruments/v3` and `instruments/v4`.
`scripts/materialize_dev_prompt_repair.py` verifies the original public task
hashes, limits mutations to the approved prompts, requires the v3 parent for
v4, and rejects changed tasks/manifests or overwriting an existing artifact.
These are DEV diagnostic instruments, not calibration/candidate artifacts.

To verify an artifact against the original public root:

```python
from pathlib import Path
from scripts.materialize_dev_prompt_repair import load_repair
catalog, manifest = load_repair(
    Path("/absolute/path/to/original/instrument/public"),
    Path("experiments/r23.5-serialization-dev/instruments/v4"),
)
```

To repeat the same diagnostic in the offline historical model namespace,
pass `--repair-instrument /absolute/path/to/instruments/v3` (then v4) to
`scripts/probe_r20_articulation.py`. Use new output directories and preserve
every attempt. The original v2 run omits this option.

## Prerequisite: verify that the frozen mouth can express the answer

Run `scripts/probe_r20_articulation.py` under the verified historical model
path and exact runtime described in
`infrastructure/kaggle/LOCAL_REPRODUCTION.md`.

The input is the existing public R20 `DEV_VIEW.json`, hash
`bfb1a4003c421df3a14345bdcc6ddabad2d6ae252a5aadd6414d0cf0ba07fea6`.
It contains 90 training and 15 tuning tasks. The diagnostic selects the first
tuning task in each of the three frozen families in catalog order, without
using any observed score to select tasks. It evaluates every same-rule probe
under no context and the complete teaching transcript, plus every existing
oracle-context probe. Generation is greedy, capped at 32 new tokens, with an
empty soft bank and the frozen `normalized-exact/v1` scorer. No optimizer runs.

Record expected/raw generated text, prompt hashes, model hash before/after,
and special-token handling. Special-token removal is a counterfactual output
inspection only; it does not replace the frozen scorer's actual result.
Keep all families and every negative result in the report.

If no-context and teaching-context accuracy are equal, their recovery-ratio
denominator is zero. Report it as undefined; do not invent an epsilon, lower
a threshold, or report successful compression. Diagnose the output contract
before spending compute on a soft-prompt or latent sweep.

## Proposed subsequent serialization comparison

Once the interface check supplies an interpretable contextual effect, prepare
a separate versioned pilot manifest before optimization:

- Exactly three teaching examples per pattern; disjoint probes, with all
  examples, probe definitions, prompt formatting and scorer bytes hashed.
- Conditions: no context, original context, raw serialized/reloaded context,
  frozen-model-generated text harness, learned soft prompt, and learned
  single-layer latent vector. The text harness sees teaching examples only.
- The optimizer receives teaching inputs/targets only. Probe expected answers
  are passed solely to the scorer after generation. No task identifiers enter
  the learned mapping.
- Report the existing R23.5 recovery ratio and actual representation bytes,
  retaining the raw-context and text-harness retrieval baselines.
- Fix seeds, representation widths/layers, optimizer settings, update count,
  generation limits and stopping rule in that manifest. Do not tune them
  against the subsequent probe results.
- Persist only numerical learned state for the latent arms; reload it in a
  fresh evaluation process without the teaching examples, targets or optimizer.
- Check gradients reach learned state, never the language-model parameters;
  require identical model hashes before and after every training run.
- Preserve per-pattern/per-seed outputs, loss trajectories, nulls and errors.

The R23.5 draft's 30% recovery criterion is unchanged. This small DEV workflow
cannot yield an R20 acceptance, a full R23.5 study verdict, or permission to
change the instrument. R25's FiLM comparison becomes a separate one-variable
follow-up; the toy R24 success does not establish that FiLM controls Qwen.
