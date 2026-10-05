# Oczy learning workbench

An inspectable local DEV experience: compare learned states, retrieved examples,
an untrained state and an explicitly supplied rule; inspect frozen scores and
real temporary-filesystem tool transcripts. Live trials never update training.

From the repository or an unpacked bundle:

```bash
python -m workbench verify
python -m workbench serve
```

Open `http://127.0.0.1:8765`. Saved evidence requires only Python 3.12's standard
library. No install or model download occurs. The server binds only to loopback.

Live trials additionally require Linux with `bwrap`, the exact previously pinned
CPU Python runtime and Qwen model. On the research workstation they are detected
under `~/.local/state/oczy/diagnostics/2026-09-11-organ-identity/`. Elsewhere set:

```bash
export OCZY_WORKBENCH_PYTHON=/absolute/path/to/pinned/runtime/bin/python
export OCZY_WORKBENCH_MODEL=/absolute/path/to/pinned/model
python -m workbench trial amber berry --seed 0
```

The worker verifies model, runtime, source and state hashes, runs with networking
disabled and reads model/source files read-only. It performs seven scalar
comparisons, with no optimizer. The first request loads the model; allow up to
three minutes. Only one live request executes at a time. Exploratory inputs are
bounded lowercase words and four registered contexts. No user files are tools.

The bundle includes all three final states and matching initial/control states,
audited results, registered instruments and their source. Weights, runtime,
credentials, caches and unrelated working-tree changes are excluded.

```bash
python -m workbench package /absolute/path/to/oczy-workbench-dev-v1.tar.gz
```

Research results remain bounded DEV evidence. An exact replay of a failure is
reproducibility, not a capability pass. Joint training does not prove sequential
accumulation. An externally selected rule is not learned context selection.

## What completed

The diversity study finishes 144 updates and all 608 evaluation responses.
New-word affix answers improve from 5/24 to 12/24 against the matched control;
neutral copying remains 19/24 versus initial 24/24. Its acquisition gate fails,
so correction does not run. The separate regression run reproduces all 804
historical comparisons exactly and audits 208 additional candidate probes and
action trials. Reproduced failures remain failures. Both trained conditions
achieve tool core 0/12 versus initial states 5/12; zeroed-prefix control is 0/4.

See the [study report](../experiments_logs/2026-09-13_scoped_diversity_dev_v3.md)
and [regression report](../experiments_logs/2026-09-13_capability_regression_dev_v1.md).
Reports and complete checksummed run archives are included in the bundle.

## Research controller recovery

The original diversity controller exits 1 after evaluation because its execution
ledger uses exclusive file creation for a second write. That frozen source is
preserved. A separate recovery verifies complete results and applies the original
admission function without model calls or optimizer updates. The evaluation
child's exit status, duration and pre-run firewall record were not retained;
the recovered run does not claim complete execution provenance.

The archived run already includes recovery and must not be recovered again.
For a reproduction in a **new** output directory that reaches the same failure,
the pinned research runtime can invoke:

```bash
python -m scripts.diversity_dev.recover \
  --root experiments/scoped-diversity-dev-v3 \
  --run /absolute/path/to/new/completed-run
```

This narrowly scoped recovery refuses an incomplete evaluation or a passing
acquisition gate. Running training again is unnecessary for using the workbench.
