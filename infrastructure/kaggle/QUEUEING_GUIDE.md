# Remote queueing and experimentation guide

This is the operational guide for preparing, queueing, running, recovering,
and adjudicating Oczy experiments on Kaggle CPU and Colab CPU. Read
[`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) first for the full remote-compute
contract. Remote runners execute an already approved experiment; they do not
choose the experiment, edit the measuring instrument, select thresholds, or
inspect meta-test data.

Run all commands from the repository root in one shell unless a step says
otherwise.

The queue is a durable, single-scheduler, file-backed watch queue. It is not a
general-purpose message broker. Its safe operating model is:

- one reviewed campaign manifest per scientific question;
- one writer updating a batch manifest at a time;
- one scheduler process owning a state file;
- unique, immutable job names; and
- persistent queue files outside `/tmp` for unattended operation.

## What is queued

Keep these three artifacts conceptually separate:

| Artifact | Authority | Purpose |
|---|---|---|
| Research specification and frozen eval | Human-approved local source | Defines the question, variable, metrics, thresholds, baselines, seeds, and gates |
| Campaign manifest | Reviewed, commit-addressed experiment contract | Records source, provider, phase, module, arguments, output path, and claim class |
| Scheduler batch and state | Execution control plane | Places immutable jobs on remote providers and records lifecycle progress |

The scheduler batch is not an experiment registry. Editing it must not change
the scientific contract. If a scientific parameter changes, create and review
a new campaign/job rather than silently modifying an existing queued job.

There are also two meanings of "queued":

- **Local `pending`**: accepted by the scheduler but not yet submitted because
  a provider slot is unavailable.
- **Provider `queued`**: Kaggle accepted the kernel but has not started it. The
  scheduler records this as an active `running` job and it occupies a Kaggle
  slot until Kaggle completes or errors it.

## Queue lifecycle and capacity

Jobs move through:

```text
pending -> submitting -> running -> collecting -> succeeded
                                      \----------> failed
```

The state file is atomically replaced after transitions. With no explicit
`--max-parallel`, capacities are additive:

- Kaggle: up to `--kaggle-max`, default and hard maximum 5;
- Colab: starts at one admission and increases through the AIMD controller up
  to `--colab-max`, default ceiling 10; and
- optional `--max-parallel N`: caps the combined total across providers.

The scheduler scans jobs in manifest insertion order. This is not strict
global FIFO: a blocked Colab job does not prevent a later Kaggle job from using
an available Kaggle slot. There are no priorities or dependencies.

## Experimentation best practices

### Before generating remote work

1. State the hypothesis and the single variable changed from the matched
   control.
2. Freeze metrics, thresholds, baselines, episodes, scoring, seeds, stopping
   rule, output schema, and kill criteria before submission.
3. Check every threshold against the real-data distribution. Eval changes
   require a version bump and explicit human sign-off under `AGENTS.md`.
4. Keep retrieval, prefix, logit-bias, and rerank baselines in the comparison
   table. Changed dynamics must clear that bar.
5. Use mechanism-level changes only. Never branch on episode IDs.
6. Commit the exact source and require a clean worktree. Do not use a moving
   branch or update an existing source dataset in place.
7. Pin the model, tokenizer, chat template, instrument manifest, source
   archive, environment, and provider profile.
8. Label the claim class correctly: `scientific` for hypothesis-bearing runs,
   `infrastructure` for runner and provisioning checks.

### Canary before fan-out

Run one seed first. Fan out the pre-registered seed set only after the canary
passes all of these checks:

- source commit and source archive SHA-256 match;
- private kernel, internet disabled, and CPU hardware verified;
- model and frozen-parameter hashes match;
- module, arguments, phase, and instrument hash match the campaign;
- the expected report schema and all required artifacts are present; and
- the canary did not access meta-test or leak raw traces.

A successful canary validates execution, not the hypothesis. Do not tune the
instrument or threshold from its result and then present the expanded run as
pre-registered.

### Design seed and ablation jobs

- Encode experiment version, condition, and seed in every job name.
- Give reruns a new name, for example `r18-trajectory-v2-seed-0`; the queue
  deliberately ignores changed definitions under an existing name.
- Change one scientific variable per comparison. Infrastructure-only changes
  must be identified separately.
- A breakthrough claim requires the registered ablation, full trajectory, and
  independent seeds. A single successful seed is diagnostic only.
- Preserve null seeds, refutations, failed gates, and infrastructure failures.
  Never average them away or relabel them as wins.

## Preferred campaign workflow

For research, use `prepare_experiment_campaign.py` rather than hand-authoring a
scheduler batch. The campaign is reviewed first; the preparer then emits the
provider-specific artifacts and a scheduler-compatible `batch.json`.

### 1. Create and review the campaign

Minimal Kaggle example (replace every placeholder with reviewed values):

```json
{
  "schema_version": "oczy/remote-experiment-campaign/v1",
  "source_commit": "<40-character-lowercase-git-sha>",
  "source_repo": "https://github.com/KinoThe-Kafkaesque/oczy.git",
  "jobs": [
    {
      "name": "r20-ablation-a-seed-0",
      "provider": "kaggle",
      "phase": "development",
      "module": "oczy.experiments.example",
      "arguments": ["--condition", "ablation-a", "--seed", "0"],
      "output_path": "outputs/r20-ablation-a-seed-0",
      "claim_class": "scientific",
      "kernel_id": "abdellahkadem/oczy-r20-ablation-a-seed-0",
      "title": "Oczy R20 Ablation A Seed 0",
      "profile": "cpu",
      "source_dataset": "<owner/commit-addressed-private-dataset>",
      "source_archive_sha256": "<64-character-lowercase-sha256>"
    }
  ]
}
```

Meta-test jobs additionally require the frozen instrument manifest hash and a
recorded human sign-off identifier. Their presence records approval; it never
allows an autonomous runner to approve itself.

### 2. Prepare immutable runner artifacts in durable storage

```bash
QUEUE="$HOME/.local/state/oczy/remote-queue"
CAMPAIGN_ID="r20-ablation-a-v1"
CAMPAIGN_DIR="$QUEUE/campaigns/$CAMPAIGN_ID"
mkdir -p "$CAMPAIGN_DIR"
cp campaign.json "$CAMPAIGN_DIR/campaign.json"

uv run python infrastructure/kaggle/prepare_experiment_campaign.py \
  "$CAMPAIGN_DIR/campaign.json" \
  --output "$CAMPAIGN_DIR"
```

Review `$CAMPAIGN_DIR/campaign_manifest.json`, every generated
`job_spec.json`, kernel metadata, module argument, source reference, and output
path before queueing. Keep the original `campaign.json`: the generated
`campaign_manifest.json` is a provenance wrapper and is not the collector's
campaign input. Preparation is not authorization to submit.

Generated batch paths are relative to the generated batch's directory. Do not
copy `batch.json` somewhere else without rebasing its `kernel_dir`, `script`,
and `output_dir` fields.

### 3. Create or update the rolling queue batch

For a long-lived queue, keep the scheduler batch and state outside `/tmp`:

```bash
mkdir -p "$QUEUE"
```

Do not use `/tmp` for a queue that must survive reboot or host cleanup.
Keep generated campaign artifacts and reports in their durable campaign
directory; the scheduler state does not replace scientific provenance.

Before appending, rebase the generated campaign jobs to absolute artifact
paths. This lets one rolling batch safely reference multiple campaign
directories:

```bash
jq --arg root "$CAMPAIGN_DIR" '
  del(.campaign_source_commit)
  | .jobs |= map(
    .output_dir = ($root + "/" + .output_dir)
    | if .provider == "kaggle" then
        .kernel_dir = ($root + "/" + .kernel_dir)
      elif .provider == "colab" then
        .script = ($root + "/" + .script)
      else . end
  )
' "$CAMPAIGN_DIR/batch.json" > "$CAMPAIGN_DIR/queue-jobs.json"
```

Install the first batch or append later work under the same producer lock. The
temporary file stays in the queue directory so the final rename is atomic:

```bash
NEW_JOBS="$CAMPAIGN_DIR/queue-jobs.json"

(
  set -euo pipefail
  flock 9
  tmp=$(mktemp "$QUEUE/batch.json.XXXXXX")
  trap 'rm -f "$tmp"' EXIT
  if [ -f "$QUEUE/batch.json" ]; then
    jq --slurpfile new "$NEW_JOBS" \
      '.jobs += $new[0].jobs' "$QUEUE/batch.json" > "$tmp"
  else
    cp "$NEW_JOBS" "$tmp"
  fi
  python -m json.tool "$tmp" >/dev/null
  uv run python - "$tmp" <<'PY'
import sys
sys.path.insert(0, "infrastructure/kaggle")
from parallel_scheduler import load_batch
load_batch(sys.argv[1])
PY
  mv "$tmp" "$QUEUE/batch.json"
  trap - EXIT
) 9>"$QUEUE/batch.lock"
```

The validation step checks the complete merged batch, including artifact
paths, unique job names, unique Kaggle kernel IDs, privacy, and CPU-only
metadata. Preserve the original generated `batch.json`; it records the
campaign-local artifact layout.

### 4. Start exactly one scheduler owner

```bash
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json" \
  --watch-batch \
  --watch-interval 30 \
  --poll-interval 30
```

Use a process supervisor for unattended execution. The scheduler takes a
non-blocking OS lock beside the state file and refuses a second owner. It does
not create a PID service or restart itself after host/process failure.

### 5. Inspect without submitting

```bash
uv run python infrastructure/kaggle/parallel_scheduler.py status \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json"
```

The status command may exit non-zero when any job has failed. For a quick
state-only count:

```bash
jq '.jobs | to_entries | group_by(.value.state) |
    map({state: .[0].value.state, count: length})' "$QUEUE/state.json"
```

### 6. Inspect and plan across every runner account

The scheduler state above describes one queue, not the whole remote account
pool. Configure account-scoped credential locations in
`~/.config/oczy/runner-pool.json` using
[`runner_pool.example.json`](runner_pool.example.json), then merge the queue
state into the provider inventory:

```bash
uv run python infrastructure/kaggle/runner_pool.py status \
  --state "$QUEUE/state.json"

# Monitoring/automation form:
uv run python infrastructure/kaggle/runner_pool.py status \
  --state "$QUEUE/state.json" \
  --active-only \
  --json
```

The view normalizes Kaggle kernels and Colab sessions, correlates remote IDs
with scheduler jobs, and keeps account failures isolated. Kaggle accounts use
separate `KAGGLE_CONFIG_DIR` values. Colab OAuth accounts require separate
`home_dir` values because Colab CLI 0.6.0 stores `token.json` under `HOME`; the
inventory refuses to start an interactive login when a token is absent.

`status` is read-only. For an already-approved batch, generate a deterministic
dispatch plan without submitting anything:

```bash
POOL="$HOME/.config/oczy/runner-pool.json"
LEASES="$HOME/.local/state/oczy/runner-pool-leases.json"

uv run python infrastructure/kaggle/runner_pool.py plan \
  "$QUEUE/batch.json" \
  --config "$POOL" \
  --state "$QUEUE/state.json" \
  --output "$QUEUE/dispatch-plan.json"

uv run python infrastructure/kaggle/parallel_scheduler.py run \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json" \
  --pool-config "$POOL" \
  --dispatch-plan "$QUEUE/dispatch-plan.json" \
  --lease-state "$LEASES"
```

The plan uses healthy accounts with explicit capacities, preserves existing
account correlations, and assigns fresh jobs by deterministic projected load.
It is SHA-256-bound to the batch and pool config. The scheduler persists the
chosen `account_id`, isolates provider credentials, and uses shared expiring
leases to prevent account oversubscription across different queues.
Missing state files and degraded relevant accounts make the generated plan
diagnostic-only; the scheduler refuses to execute it.

Planning and routing do not authorize experiments. Pool-aware dispatch rejects
`--watch-batch`; regenerate and review a plan after any batch change. All
schedulers using the same account pool must share one lease-state path.

## Updating a watched batch

The scheduler has no enqueue API. It discovers unseen job names when the batch
file changes. To add work while it runs, repeat campaign preparation, path
rebasing, and the locked validate-and-rename procedure above. `NEW_JOBS` must
contain the same batch schema and only new, already reviewed job definitions.
All producers must use the same lock; otherwise concurrent read-modify-write
operations can lose jobs.

The scheduler tolerates malformed partial writes and retries them, but that is
recovery behavior, not a safe producer protocol. Existing job definitions and
states are never changed by a watched reload.

Never remove completed jobs merely to shrink the file. The state is bound to
the batch path and existing names preserve deduplication and provenance.

## Recovery and retry policy

| Situation | Scheduler behavior | Operator action |
|---|---|---|
| Malformed watched batch | Logs the validation error and retries after the next change/check | Repair or atomically replace the batch |
| Kaggle job is provider-queued | Keeps it active and polls | Wait; it occupies a Kaggle slot |
| Kaggle scheduler process stops while job is running | Reloads state and resumes polling without resubmission | Restart the same command with the same batch and state |
| Kaggle push, remote execution, collection, or timeout failure | Marks the job `failed`; no general automatic retry | Diagnose, preserve the failed artifact, and enqueue a corrected job with a new name |
| Colab capacity rejection | Returns to `pending`, cools down, and retries; fails after 10 consecutive rejections | Wait or reduce competing Colab sessions/capacity |
| Scheduler stops while a Colab job is active | Marks the job failed because the local process handle is gone; best-effort stops the session | Preserve diagnostics and enqueue a new uniquely named job if justified |
| State file is lost | Deduplication and lifecycle history are lost | Reconstruct deliberately from provider/provenance records; do not blindly resubmit |

Do not edit `failed` to `pending` in `state.json`. A rerun is a new execution
and needs a new job identity so the original failure remains visible.

## Collect and adjudicate campaign results

After jobs are terminal, run the campaign collector:

```bash
uv run python infrastructure/kaggle/collect_experiment_campaign.py \
  "$CAMPAIGN_DIR/campaign.json" \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json" \
  --output "$CAMPAIGN_DIR"
```

The collector takes the original `oczy/remote-experiment-campaign/v1`
campaign, not the generated provenance wrapper. Adjust `--output` or use
`--report-dir` only if the report layout differs from the campaign's recorded
`output_path` values. The collector verifies identity and provenance before
classifying each job:

- **COMPLETE**: execution completed and provenance is valid;
- **NULL**: valid scientific execution produced no registered metric/signal;
- **INVALID**: provenance is missing, corrupt, or mismatched; and
- **BLOCKED**: execution, infrastructure, timeout, or artifact production
  failed.

`COMPLETE` is an execution classification, not automatically a positive
scientific result. Apply the frozen experiment gates only after provenance is
valid. A valid negative result is a null/refutation and must be logged as
prominently as a win. An infrastructure failure is `BLOCKED`, never evidence
against the hypothesis.

Record the dated result summary, exact commands, source and model hashes,
provider/kernel identifiers, all seeds, matched baselines, trajectories,
confidence intervals where registered, and the final gate decision under
`experiments_logs/`. Keep large pulled artifacts in the configured report
directory.

## Operator checklist

Before queueing:

- [ ] One hypothesis and one changed scientific variable are named.
- [ ] Eval version, metrics, thresholds, baselines, episodes, seeds, and gates are frozen.
- [ ] Thresholds have real-data distribution checks.
- [ ] Source is clean, committed, bundled, and hash-verified.
- [ ] Model, tokenizer, instrument, provider, phase, and output schema are pinned.
- [ ] Kernel is private, internet-off, and CPU-only.
- [ ] Meta-test has explicit human sign-off where applicable.
- [ ] Job names are unique and immutable.
- [ ] Canary acceptance criteria are written before submission.

Before claiming a result:

- [ ] Every expected job is terminal and every artifact is pulled.
- [ ] Source, archive, model, instrument, hardware, module, and argument provenance match.
- [ ] Missing seeds and infrastructure failures are classified as `BLOCKED`/`INVALID`, not nulls.
- [ ] Nulls and refutations are retained.
- [ ] Retrieval and other matched baselines remain in the result table.
- [ ] Ablation, trajectory, and seed requirements match the registered claim.
- [ ] No threshold, scoring, or eval change was inferred from the measured result.

## Current limitations

The current scheduler intentionally does not provide:

- a network enqueue API or database-backed queue;
- multi-writer transactions;
- priorities, dependencies, cancellation, or per-job pause;
- a general Kaggle retry policy; or
- transparent resume of active Colab subprocesses.

Pool routing currently uses fixed account assignments: it does not
automatically fail over a job to another account after planning. Hash-bound
pool plans are incompatible with watch mode, and stale leases expire after the
configured TTL rather than reconciling themselves against provider history.

If those are required, add them as execution-control features without changing
the frozen evaluation instrument. Any new retry policy must preserve attempts
and failure records rather than erasing nulls or infrastructure failures.
