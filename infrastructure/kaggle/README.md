# Remote CPU compute pool for offline cortex work

This directory provides a private, reproducible **CPU-only** compute pool
using **Kaggle Kernels** and **Google Colab CLI** for offline developmental
work around Research/20 and Experiment 09.

Use [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for every real research run. It
defines source/model pinning, CPU profile generation, submission/retrieval,
provenance, meta-test gates, and the CPU-only contract.

Use [`QUEUEING_GUIDE.md`](QUEUEING_GUIDE.md) for the persistent watch queue,
safe batch updates, recovery, campaign collection, and the operator checklists
that connect queue mechanics to experimentation best practices.

The checked-in kernels are deliberately **infrastructure smoke tests**, not
cortex experiments. They run a synthetic learned-writer -> fixed fast/slow state
-> latent coupler -> frozen differentiable organ path. The state remains 64x64
and the organ interface uses Qwen2.5-0.5B-Instruct's real hidden width (896),
but no Qwen weights are loaded for the smoke task. The jobs do not use `eval/v2`,
`meta_cortex/v1`, retrieval, episode-conditioned code, or a real language
model, and their losses cannot be cited as evidence for H-META-CORTEX.

## Provider availability

| Provider | Profile | Status | Since |
|---|---|---|---|
| Kaggle CPU | `cpu` | Active | 2026-07-10 |
| Colab (CLI 0.6.0) | `cpu` | Active | 2026-07-11 |
| GPU (T4/P100/L4) | --- | Archived -- do not submit | 2026-07-09 |
| TPU | --- | Not wired | --- |

## What is wired

| Profile | Kaggle kernel | Requested machine | Intended use |
|---|---|---|---|
| CPU | `abdellahkadem/oczy-cortex-cpu-smoke` | Kaggle CPU | instrument materialization, leakage/distribution audits, scoring, aggregation |
| CPU | `abdellahkadem/oczy-qwen-cpu-probe` | Kaggle CPU | frozen real-model and input-gradient verification on CPU |

All kernels are private, have internet disabled, and set `enable_gpu: false`.
The smoke kernel writes `cortex_smoke_report.json` to `/kaggle/working`. The
report records the hardware actually allocated, source hash, throughput,
held-out loss, finite-gradient checks, and before/after hashes of every frozen
parameter. See [`RESULTS.md`](RESULTS.md) for the verified CPU smoke result and
the acceptance contract.

### CPU-only contract

Active runners and generated bootstrap code enforce a strict CPU-only
contract:

- Kernel metadata sets `enable_gpu: false` and `enable_tpu: false`.
- The generated bootstrap sets `CUDA_VISIBLE_DEVICES=""` before importing
  torch and verifies the env var is still empty at startup, with no CUDA
  or NVML runtime query performed.
- No active code calls `torch.cuda.*` for device selection, model placement, or
  profiling. Legacy report fields (`cuda_available`, `cuda_device_count`,
  `torch_cuda_version`) remain as constant `false`/`0`/`null` values for schema
  compatibility.
- Colab CLI jobs never pass `--gpu` or `--tpu`; the scheduler argv omits them.

## Local preflight

```bash
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle quota --csv
colab --help
colab sessions

uv run python infrastructure/kaggle/run_cortex_smoke.py \
  --device cpu \
  --output /tmp/oczy-cortex-local-smoke.json
```

The `--device` argument accepts only `cpu`. Authentication uses Kaggle's OAuth
credentials at `~/.kaggle/credentials.json` or the Colab CLI OAuth flow
(`colab login`). Do not copy credentials into the repository or a kernel source
directory.

## Submit and retrieve (Kaggle)

CPU kernels do not use `--accelerator`. The metadata already specifies
`enable_gpu: false`.

```bash
# Cortex CPU smoke
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels logs abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force
```

```bash
# Qwen CPU probe
kaggle kernels push \
  -p infrastructure/kaggle/qwen-cpu-probe \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels logs abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-cpu-probe \
  --path reports/kaggle/qwen-cpu-probe \
  --force
```

A run counts as wired only when its JSON artifact has `passed: true`,
`cuda_available: false`, and `cuda_device_count: 0`. See
[`RESULTS.md`](RESULTS.md) for the full acceptance contract.

## Parallel batch scheduling

For many-jobs fan-out, the **parallel scheduler** (`parallel_scheduler.py`)
manages a batch of remote CPU jobs (Kaggle kernels and/or Colab sessions) with
additive provider capacity, crash-safe durable state, and automatic resume.

### Unified runner-pool inventory

`runner_pool.py` provides one read-only account/job view across every configured
Kaggle and Colab account. It also merges durable scheduler state, so local
`pending`, `failed`, and collected jobs appear beside jobs discovered directly
from the provider. One broken or unauthenticated account is reported without
hiding healthy accounts.

Start from the credential-free example:

```bash
mkdir -p ~/.config/oczy
cp infrastructure/kaggle/runner_pool.example.json \
  ~/.config/oczy/runner-pool.json

uv run python infrastructure/kaggle/runner_pool.py validate

uv run python infrastructure/kaggle/runner_pool.py status \
  --state ~/.local/state/oczy/remote-queue/state.json

uv run python infrastructure/kaggle/runner_pool.py status \
  --active-only --json
```

The v1 config stores credential **locations**, never secrets:

- each Kaggle account has its own `config_dir`, passed as
  `KAGGLE_CONFIG_DIR`;
- each Colab OAuth account has its own `home_dir`, which isolates
  `.config/colab-cli/token.json`, plus explicit `session_config`,
  `client_oauth_config`, and `auth`; and
- `state_files` lists scheduler state files included in every view. Additional
  files can be supplied with repeated `--state` flags.

The JSON output schema is `oczy/remote-runner-pool-snapshot/v1`. Provider and
scheduler records are correlated by `(provider, remote_id)`, with both
`remote_state` and `scheduler_state` retained for diagnosis. The process exits
non-zero if an enabled account is degraded/unavailable or a state file cannot
be read, while still printing the partial inventory.

This interface does not dispatch jobs or choose experiments. The current
scheduler submission path remains unchanged; the account registry and
normalized snapshot are the control-plane boundary for future pool-aware
routing.

### Batch schemas

#### v1 (``oczy/kaggle-parallel-batch/v1``) — Kaggle only

```json
{
  "schema_version": "oczy/kaggle-parallel-batch/v1",
  "jobs": [
    {
      "name": "smoke-seed-0",
      "kernel_dir": "build/smoke-seed-0",
      "output_dir": "reports/kaggle/smoke-seed-0"
    },
    {
      "name": "smoke-seed-1",
      "kernel_dir": "build/smoke-seed-1",
      "output_dir": "reports/kaggle/smoke-seed-1"
    }
  ]
}
```

Each ``kernel_dir`` must be produced by ``prepare_research_kernel.py`` with
``kernel-metadata.json`` and ``job_spec.json``. Validation rejects any kernel
that is not private, CPU-only, or whose title would create a different Kaggle
slug than its ``id`` field.

#### v2 (``oczy/remote-parallel-batch/v2``) — mixed providers

Each job carries a ``provider`` field (``"kaggle"`` or ``"colab"``):

```json
{
  "schema_version": "oczy/remote-parallel-batch/v2",
  "jobs": [
    {
      "name": "kaggle-seed-0",
      "provider": "kaggle",
      "kernel_dir": "build/kaggle-seed-0",
      "output_dir": "reports/kaggle-seed-0"
    },
    {
      "name": "colab-run-0",
      "provider": "colab",
      "script": "scripts/train.py",
      "arguments": ["--seed", "42"],
      "output_dir": "reports/colab/run-0",
      "timeout": 3600
    }
  ]
}
```

Colab job fields:
- **script** (required) — path to the Python script, resolved relative to the
  manifest file's directory.
- **arguments** (optional) — list of string CLI arguments.
- **output_dir** (required) — local directory for pulled stdout/stderr/result.
- **timeout** (optional) — per-job wall-clock timeout in seconds.

### Provider-agnostic run and status

```bash
# Run a batch (v1 or v2)
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json

# Check status without submitting
uv run python infrastructure/kaggle/parallel_scheduler.py status \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| ``--max-parallel N`` | omitted | Omit for additive provider capacity (kaggle_max + learned Colab). Explicit N caps total concurrency for backward compatibility (must be >= 1). |
| ``--kaggle-max N`` | 10 | Max concurrent Kaggle kernels (hard max 10) |
| ``--colab-max N`` | 10 | Colab AIMD capacity ceiling (additive probe, not a hard quota) |
| ``--colab-cooldown SEC`` | 60 | Seconds to wait after a Colab 412 rejection before retrying |
| ``--push-timeout SEC`` | 21600 | Kaggle kernel run-time limit |
| ``--job-timeout SEC`` | 21600 | Max wall-clock wait per job |
| ``--poll-interval SEC`` | 30 | Seconds between status polls |
| ``--watch-batch`` | off | Live watch mode: keep the scheduler alive after all jobs finish, re-read the batch file when it changes, and merge unseen job names as new pending jobs. Exit with Ctrl-C. |
| ``--watch-interval SEC`` | 30 | Seconds between batch-file change checks when idle in watch mode (must be > 0 when ``--watch-batch`` is set) |

**Omitting ``--max-parallel`` means additive provider capacity.** The scheduler
imposes no global concurrency cap — Kaggle jobs fill up to ``--kaggle-max``
(hard-capped at 10) and Colab jobs fill up to the AIMD-learned limit
(``--colab-max`` ceiling). For example, ``--kaggle-max 10`` plus a learned
Colab limit of X allows 10 Kaggle + X Colab jobs to run concurrently.

**Explicit ``--max-parallel N`` caps total concurrency** across all providers
for backward compatibility. When provided, N must be >= 1.

**``--colab-max`` sets a ceiling, not a guaranteed capacity.** The Colab
AIMD controller starts admissions at 1 and probes upward — actual available
slots depend on account-level session limits and external sessions. See
[RESULTS.md](RESULTS.md) for the discovered `learned_limit` and probe
behavior.

### Lifecycle

The ``--state`` file is written atomically after every transition:

1. **pending** — ready to submit
2. **submitting** — push/launch in progress
3. **running** — submitted, waiting for remote completion
4. **collecting** — complete, downloading output
5. **succeeded** — output collected

A job enters **failed** on push error, kernel error status, output collection
failure, or timeout.

**Resume**: kill the process (or reboot) and re-run the same command.
Interrupted ``submitting``/``collecting`` jobs are converted back to
``pending``/``running``, already-``running`` kernels are never resubmitted,
and Colab running jobs on restart (no local process handle) are failed as
interrupted with best-effort session stop. Only new jobs from an updated
batch manifest are added.

The JSON summary is printed on completion (exit code 0 if all succeeded,
1 if any failed). The final state file is preserved for post-hoc inspection.

### Live watch mode

``--watch-batch`` keeps the scheduler alive after all jobs reach a terminal
state. The batch file is periodically re-read (every ``--watch-interval``
seconds when idle); any unseen job names are merged as new ``pending`` jobs.
Existing jobs are never modified -- their definitions and lifecycle states are
preserved exactly. Malformed or partially-written reloads are logged to stderr
and retried on the next watch cycle without terminating the daemon.

```bash
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json \
  --watch-batch \
  --watch-interval 15
```

The scheduler runs until interrupted with Ctrl-C. On interruption, state is
persisted atomically and a JSON summary is printed (exit code 0 if all jobs
succeeded, 1 if any failed). Without ``--watch-batch``, the scheduler
terminates when all jobs finish -- unchanged from previous behavior.

### Colab session lifecycle differences

Colab jobs have a different lifecycle from Kaggle kernels:

- **argv**: ``colab run --keep --session <name> --timeout <sec> -- SCRIPT ARGS...``
  The ``--`` separator ensures script paths/arguments starting with ``--`` are
  not consumed as CLI flags. The ``--keep`` flag means the session daemon
  persists on the backend; the scheduler's ``stop()`` call explicitly
  unassigns the VM (``colab stop --session <name>``).
- **Output**: ``colab`` writes ``stdout.log``, ``stderr.log``, and
  ``result.json`` (containing ``ok``, ``error``, ``exit_code``, ``status``,
  ``session``) into ``output_dir``.
- **Session cleanup**: On success, error, timeout, and capacity rejection,
  ``stop()`` is called to free the backend VM slot. The local ``Popen`` is
  killed and reaped.
- **External session accounting**: The scheduler queries ``colab sessions``
  (cached once per loop iteration) to count account-wide active sessions.
  External sessions count toward the admission gate — they reduce slots
  available to scheduler jobs even if the scheduler did not start them.
- **Restart recovery**: On restart, ``detect_orphaned_sessions`` intersects
  known scheduler session names with the backend session list and stops any
  leaked sessions. A ``sessions()`` probe runs once at restart so external
  sessions can drain without false failure.

### Colab AIMD admission control

Colab capacity is probed dynamically, not hardcoded:

- Starts admission at 1 (``COLAB_AIMD_START``).
- **On success** (first confirmed running poll): learned_limit += 1 (capped
  at ``--colab-max``).
- **On HTTP 412 TooManyAssignmentsError**: learned_limit reduced to the
  account-wide active Colab session count (including external sessions,
  min 1). A cooldown period (``--colab-cooldown``) prevents immediate
  retry.
- **On admission gate block** (external sessions fill capacity): job stays
  pending — it does not count as a capacity rejection or increment the
  rejection counter. Only an actual 412 process exit increments the
  consecutive-rejection counter.
- After 10 consecutive 412 rejections the job fails (prevents infinite
  retry when external sessions permanently occupy all capacity).

The learned_limit is persisted in the v2 state file as
``colab_learned_limit`` and restored on restart. **It is not a hardcoded
maximum** — it self-corrects on the next 412 and continues probing.

### Verified concurrent execution (Kaggle)

Two CPU smoke kernels
(``abdellahkadem/oczy-scheduler-cpu-smoke-1`` and
``abdellahkadem/oczy-scheduler-cpu-smoke-2``) were submitted with
``max_parallel=2`` on 2026-07-10. Both succeeded in one attempt and
reported ``passed: true``, ``cuda_available: false``. The scheduler
recorded submission timestamps differing by approximately 1.85 s while
completion times overlapped, confirming genuine concurrent remote
execution on Kaggle CPU. See [`RESULTS.md`](RESULTS.md) for the
complete evidence.

### Verified queue-starvation fix (Colab, 2026-07-11)

The first live four-job Colab run exposed a bug that caused the admission gate
to increment ``capacity_rejections`` on every poll cycle when a pending job was
blocked by full capacity. After ``COLAB_MAX_CAPACITY_REJECTIONS`` (10) blocks,
the job was failed even though no ``TooManyAssignmentsError`` had occurred —
it was normal queueing. The fix: admission gate blocks (``cached_external_active
>= effective_limit``) do not touch ``capacity_rejections``. See
[`RESULTS.md`](RESULTS.md) for the full evidence.

## Source bundling and kernel generation (Kaggle)

The source and kernel preparation commands used by the full workflow are:

```bash
uv run python infrastructure/kaggle/prepare_source_bundle.py --help
uv run python infrastructure/kaggle/prepare_research_kernel.py --help
```

The kernel generator accepts `--profile cpu` (the only active profile) and
produces a private, internet-off kernel with a CPU-only bootstrap. The
bootstrap verifies a commit-addressed source archive, discovers the attached
version-pinned model, disables network-backed model resolution, records runtime
provenance, and then executes one Python module. See
[`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for the end-to-end workflow.

## R19 calibration manifest

The JSON Schema at
[`model_manifests/r19_calibration_manifest.schema.json`](model_manifests/r19_calibration_manifest.schema.json)
defines and validates the output of the `calibrate-dev` phase of
`oczy.experiments.s19_language_organ` (Research/19). The manifest is a **flat
dict** — every field is a top-level key produced by
`CalibrationManifest.to_dict()`, with no nested objects except
`parameter_breakdown`. It carries every value that the `evaluate` phase needs to
reproduce the calibrated cortex, every value a human reviewer needs to approve
before holdout evaluation, and every audit artifact required to verify
calibration integrity — split salt/fraction, Phase-0 distributions, C7 retrieval
baseline reference, trace/cache deletion audit, and a full articulation audit
with banned-content verification.

### Two-phase CLI

The experiment module exposes two phases:

- **`calibrate-dev`** — runs the Phase 0 DEV distribution check, trains the
  Arm B articulation coupler on DEV, freezes the proposed confidence threshold,
  specificity margin, coupler/head state, and label phrasing, then writes the
  calibration manifest. This phase can run now.
- **`evaluate`** — runs online teaching, consolidation, one-shot holdout
  evaluation, and causal intervention using a signed-off manifest. This phase
  remains blocked pending human sign-off.

### Evaluation gate (fail-closed)

The `evaluate` phase MUST fail closed unless **all** of the following are
supplied and pass:

1. `--manifest-hash` — a 64-hex SHA-256 that must equal the manifest's
   `manifest_sha256` field.
2. `--signoff-id` — a nonempty human approval identifier that must match the
   manifest's `signoff_human_signoff_id` field.
3. `signoff_thresholds_signed_off` — must be `true` in the manifest (set by
   human review, never by calibrate-dev).
4. `signoff_oracle_ceiling` — must be positive (the DEV oracle ceiling was
   measured and is nonzero).
5. `signoff_dev_articulation_gate` — must be `true` (Arm B articulation on DEV
   probes exceeds the no-update C1 baseline).
6. `c7_available` — must be `true` (the S3.M2a retrieval baseline is available
   for the C7 external bar). If `false`, evaluation is explicitly blocked and
   `c7_blocked_reason` must state why.

The schema and calibration code never auto-sign. A missing, empty, or
mismatched sign-off ID causes the evaluate phase to refuse to run. No agent
may approve its own instrument.

#### Exact sign-off workflow

1. **calibrate-dev** writes the manifest with
   `signoff_thresholds_signed_off` set to `false` and
   `signoff_human_signoff_id` set to an empty string. All audit fields (split,
   C7 reference, trace deletion, articulation audit) are populated; the
   manifest is structurally complete but unsigned.
2. **Human reviewer** inspects the manifest: verifies the DEV articulation gate
   passed (`signoff_dev_articulation_gate` is true), verifies the oracle ceiling
   is positive (`signoff_oracle_ceiling` > 0), verifies the C7 retrieval
   baseline is available (`c7_available` is true), verifies trace/cache deletion
   is confirmed (all `trace_*` fields true/zero), and verifies the articulation
   audit shows no banned content (all `articulation_banned_*` fields true).
3. **Human reviewer** sets `signoff_thresholds_signed_off` to `true` and fills
   `signoff_human_signoff_id` with a nonempty identifier. The schema enforces
   that `signoff_thresholds_signed_off=true` requires a nonempty
   `signoff_human_signoff_id`.
4. **evaluate** is invoked with `--manifest` and `--signoff-id` matching the
   signed-off manifest. The evaluate phase verifies the manifest hash, the
   sign-off ID, `signoff_thresholds_signed_off`, `signoff_oracle_ceiling`, and
   `signoff_dev_articulation_gate` before proceeding.

### Manifest fields

All fields are top-level keys in a flat dict. `additionalProperties` is `false`
— any key not listed below fails validation.

| Field | Type | Constraint | Purpose |
|---|---|---|---|
| `schema_version` | string | const `"oczy/r19-calibration-manifest/v1"` | Schema version |
| `created_at` | string | ISO 8601 timestamp | Creation timestamp; **excluded from the canonical hashed payload** so identical calibration values produce an identical hash regardless of when the manifest was written |
| `source_commit` | string | 40 hex or empty | Git commit SHA of the source archive; empty explicitly indicates unavailable |
| `source_archive_sha256` | string | 64 hex or empty | SHA-256 of the source tar archive; empty explicitly indicates unavailable |
| `eval_version` | string | nonempty | Frozen eval instrument version (e.g. v2.1) |
| `eval_manifest_sha256` | string | 64 hex or empty | SHA-256 of the frozen eval instrument manifest; empty explicitly indicates unavailable |
| `model_repo_id` | string | nonempty | Hugging Face repo id of the frozen language organ |
| `model_revision` | string | nonempty | Pinned model revision (snapshot hash or version tag) |
| `model_config_sha256` | string | 64 hex or empty | SHA-256 of config.json; empty explicitly indicates unavailable |
| `model_safetensors_sha256` | string | 64 hex or empty | SHA-256 of model.safetensors; empty explicitly indicates unavailable |
| `model_params_requires_grad` | boolean | **const false** | All LM parameters have `requires_grad=False`; Arm B steers via `inputs_embeds` only |
| `d_embd` | integer | **const 896** | Frozen LM hidden width |
| `d_cortex` | integer | ≥1 | Cortex state dimension (contract: 16) |
| `latent_tokens` | integer | ≥1 | Fixed-width latent bank tokens (contract: 3) |
| `max_labels` | integer | 1–20 | Maximum sense labels for Arm A (hard cap: 20) |
| `arm_b_input_mode` | string | **const `"inputs_embeds"`** | Arm B injects via `inputs_embeds`; no LM parameter update |
| `parameter_total` | integer | 0–64000 | Total persistent cortex parameters (contract: 60388) |
| `parameter_budget` | integer | **const 64000** | Hard parameter budget |
| `parameter_breakdown` | object | `W_perceive`, `W_label`, `b_label`, `W_coupler`, `b_coupler`, `warm_state` | Per-component parameter counts; must sum to `parameter_total` |
| `fixed_latent_shape` | array of ints | all ≥1 | Fixed-width latent control bank shape; independent of episode count (contract: `[3, 896]`) |
| `proposed_confidence_threshold` | number | 0–1 | Proposed abstain threshold from DEV Phase 0; requires human sign-off, cannot change after seeing holdout |
| `proposed_specificity_margin` | number | ≥0 | Proposed specificity equivalence margin from DEV; requires human sign-off, cannot change after seeing holdout |
| `cortex_artifact_sha256` | string | 64 hex or empty | SHA-256 of the full serialized cortex artifact (covers head + coupler) |
| `cortex_artifact_bytes` | integer | ≥0 | Size of the full cortex artifact in bytes |
| `cortex_artifact_path` | string | optional | Relative path to the cortex artifact; empty if not set |
| `coupler_sha256` | string | 64 hex or empty | SHA-256 of the Arm B coupler state artifact |
| `coupler_bytes` | integer | ≥0 | Size of the coupler state artifact in bytes |
| `head_sha256` | string | 64 hex or empty | SHA-256 of the shared cortex head state artifact |
| `head_bytes` | integer | ≥0 | Size of the head state artifact in bytes |
| `label_phrasing_frozen` | boolean | **const true** | Label phrasing frozen at calibrate-dev time |
| `labels` | array of strings | 1–20 items, each ≥1 char | Frozen sense label strings for Arm A |
| `dev_split` | string | **const `"dev"`** | Confirms the distribution check used DEV only |
| `dev_repeatability_std` | number | ≥0 | No-update repeatability std on DEV |
| `dev_confidence_mean` | number | — | Mean head confidence on DEV |
| `dev_confidence_std` | number | ≥0 | Std of head confidence on DEV |
| `dev_confidence_min` | number | — | Min head confidence on DEV |
| `dev_confidence_max` | number | — | Max head confidence on DEV |
| `dev_specificity_acc` | number | — | Specificity accuracy on DEV |
| `dev_holdout_ids_discarded` | boolean | **const true** | Holdout IDs discarded; no holdout identifiers/scores appear in the manifest |
| `split_salt` | string | contract `"v2.2"` | Split salt; must match calibrate-dev and evaluate |
| `split_fraction` | number | contract `0.3` | Holdout fraction; must match calibrate-dev and evaluate |
| `c7_reference` | string | nonempty | Reference to the S3.M2a retrieval baseline implementation |
| `c7_available` | boolean | — | Whether the S3.M2a retrieval baseline is available for C7 |
| `c7_blocked_reason` | string or null | required if `c7_available` is false | Why C7 is unavailable; may be null/absent when `c7_available` is true |
| `trace_raw_traces_deleted` | boolean | **const true** | All raw correction traces deleted via `trace_store.delete_all()` |
| `trace_raw_trace_count` | integer | **const 0** | Raw trace count after deletion, verified by `trace_store.verify_zero()` |
| `trace_embedding_cache_cleared` | boolean | **const true** | Embedding cache cleared so no probe-level state leaks into holdout |
| `trace_optimizer_state_deleted` | boolean | **const true** | Transient optimizer examples/gradients deleted after consolidation |
| `articulation_prompt_text` | string | — | Prompt text supplied to the LM (request text only for Arm B) |
| `articulation_latent_bank_shape` | array of 2 ints | each ≥1 | Latent control bank shape `[latent_tokens, d_embd]` (contract: `[3, 896]`) |
| `articulation_raw_trace_count` | integer | **const 0** | Raw traces deleted before evaluation; audit emitted after deletion |
| `articulation_language_organ_hash` | string | 64 hex or empty | SHA-256 of the frozen language organ; identical before and after run |
| `articulation_persistent_cortex_bytes` | integer | ≥0 | Persistent cortex bytes after consolidation |
| `articulation_banned_label_text_absent` | boolean | **const true** | No label text in the Arm B prompt |
| `articulation_banned_corrected_response_absent` | boolean | **const true** | No corrected_response text in the Arm B prompt |
| `articulation_banned_correction_utterance_absent` | boolean | **const true** | No correction_utterance text in the Arm B prompt |
| `articulation_banned_expected_answer_absent` | boolean | **const true** | No expected answer text in the Arm B prompt |
| `signoff_thresholds_signed_off` | boolean | calibrate-dev writes `false` | Must be set `true` by human review; evaluate rejects if `false` |
| `signoff_human_signoff_id` | string | empty from calibrate-dev | Nonempty human approval ID; evaluate requires `--signoff-id` to match |
| `signoff_oracle_ceiling` | number | evaluate requires >0 | Oracle ceiling on DEV (corrected_response prefix accuracy) |
| `signoff_dev_articulation_gate` | boolean | evaluate requires `true` | Arm B articulation on DEV exceeds no-update C1 baseline |
| `signoff_meta_test_conflation_ok` | boolean | — | C3/C2 non-conflation pre-check; fully verified in evaluate |
| `holdout_accessed` | boolean | **const false** | Schema rejects any manifest claiming holdout access; calibrate-dev operates on DEV only |
| `manifest_sha256` | string | 64 hex | SHA-256 of the canonical JSON payload (all fields except `created_at` and `manifest_sha256`, keys sorted, compact separators, UTF-8) |

### Parameter budget

The shared cortex has at most 64k persistent parameters. The contract budget
is:

| Component | Parameters |
|---|---|
| `W_perceive` | 14336 |
| `W_label` | 320 |
| `b_label` | 20 |
| `W_coupler` | 43008 |
| `b_coupler` | 2688 |
| `warm_state` | 16 |
| **Total** | **60388** (≤ 64000) |

The schema enforces `parameter_total ≤ 64000` and `max_labels ≤ 20`.

### Split salt and fraction

The `split_salt` and `split_fraction` fields record the `split_probes`
parameters used to partition probes into DEV and holdout sets. Both are frozen
at calibration time and must match the split used by the evaluate phase.
Recording these in the manifest makes the DEV/holdout partition reproducible and
auditable — a different salt or fraction would produce a different holdout set
and invalidate any holdout-derived metric.

### C7 retrieval baseline

The `c7_reference`, `c7_available`, and `c7_blocked_reason` fields record the
validity and reference for the C7 condition — the S3.M2a nearest-neighbor
retrieval baseline that serves as the external bar. C7 retrieves the most
similar episode's `corrected_response` by cosine similarity of mean-pooled
embeddings and supplies it as a text prefix. The `c7_reference` field
identifies the frozen retrieval baseline implementation so the evaluate phase
can reproduce C7. If the S3.M2a baseline is unavailable (`c7_available: false`),
the schema requires a nonempty `c7_blocked_reason` string, and the evaluate
phase must block evaluation rather than silently skip C7.

### Trace and cache deletion audit

The `trace_*` fields confirm that all raw correction traces, embedding cache
state, and transient optimizer examples are deleted before holdout evaluation:

- `trace_raw_traces_deleted` (const true) — all raw correction traces (request,
  correction, response text) are deleted via `trace_store.delete_all()`.
- `trace_raw_trace_count` (const 0) — verified by `trace_store.verify_zero()`.
- `trace_embedding_cache_cleared` (const true) — the embedding cache
  (`peek_embedding` results from the driver) is cleared so no probe-level
  embedding state leaks into holdout.
- `trace_optimizer_state_deleted` (const true) — transient optimizer examples
  and gradients are deleted after consolidation, before the head state is
  serialized.

A manifest with any of these fields set to false or a nonzero
`trace_raw_trace_count` fails validation.

### Articulation audit

The `articulation_*` fields are the machine-checkable articulation audit record
per spec. The runner must emit them for every scored condition. They contain:

- `articulation_prompt_text` — the prompt text supplied to the LM. For Arm B,
  this is the request text only — no label, corrected response, correction
  utterance, or episode-ID text.
- `articulation_latent_bank_shape` — the fixed-width latent control bank shape
  `[latent_tokens, d_embd]`. Must be independent of episode count. Contract
  shape: `[3, 896]`.
- `articulation_raw_trace_count` (const 0) — raw traces are deleted before
  evaluation; the audit is emitted after deletion.
- `articulation_language_organ_hash` (64 hex) — SHA-256 of the frozen language
  organ. Must be identical before and after the run, verifying no LM parameter
  was updated.
- `articulation_persistent_cortex_bytes` — persistent cortex bytes after
  consolidation, counting toward the 64k parameter budget.
- `articulation_banned_*` — four boolean fields verifying that no banned text
  appears in the Arm B prompt: `articulation_banned_label_text_absent`,
  `articulation_banned_corrected_response_absent`,
  `articulation_banned_correction_utterance_absent`, and
  `articulation_banned_expected_answer_absent`. Each must be true.

### Canonicalization and deterministic hashing

The `manifest_sha256` field is the SHA-256 of the canonical JSON payload:

1. Remove `created_at` and `manifest_sha256` from the manifest.
2. Sort all keys lexicographically at every nesting level.
3. Serialize with compact separators (no insignificant whitespace), UTF-8
   encoded.
4. Compute SHA-256 of the resulting bytes.

This makes the manifest hash deterministic: re-running `calibrate-dev` with
identical calibration values at a different time produces an identical
`manifest_sha256`. The `created_at` timestamp is preserved for human-readable
provenance but excluded from the hash so it cannot perturb the evaluation gate.

### What the schema does not do

- It does not choose or approve the proposed threshold, margin, coupler weights,
  or label phrasing. Those are proposed values for human review.
- It does not auto-sign. The evaluate phase requires an external human sign-off
  ID that the schema cannot supply.
- It does not contain holdout IDs, holdout scores, or any holdout-derived value.
  The `holdout_accessed` field is `const: false` and
  `dev_holdout_ids_discarded` is `const: true`; any manifest claiming holdout
  access fails validation.
- It does not modify Research/19 or the eval assets, which are immutable.

## Archived GPU material

Historical GPU verification (T4, P100, L4, and the T4-based Qwen model probe)
is preserved under [`archive/gpu/`](archive/gpu/). That material — including
kernel metadata, the 2×T4 throughput comparison, P100/L4 compatibility nulls,
and the T4 model probe results — is historical evidence only. GPU kernels and
metadata under the archive must not be resubmitted. See
[`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md) for the full historical
record.
