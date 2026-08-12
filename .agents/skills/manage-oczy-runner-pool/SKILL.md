---
name: manage-oczy-runner-pool
description: Operate Oczy's project-specific Kaggle and Colab runner pool through its reviewed control plane. Use when Codex needs to inspect remote accounts or jobs, validate runner configuration, check queue capacity or health, create a dry-run dispatch plan, run an already-approved batch across accounts, monitor or recover scheduler state, diagnose account leases or owner locks, or explain the queueing workflow in the Oczy repository.
---

# Manage Oczy Runner Pool

Operate from the Oczy repository root. Treat remote compute as an executor,
never as experiment authority.

## Establish the contract

1. Read `AGENTS.md`.
2. Read `infrastructure/kaggle/RESEARCH_GUIDE.md` before any remote research
   action.
3. Read the relevant sections of
   `infrastructure/kaggle/QUEUEING_GUIDE.md` for queueing, recovery, and
   campaign adjudication.
4. Preserve the frozen evaluation instrument. Never let routing, capacity,
   retries, or remote results choose metrics, thresholds, episodes, seeds, or
   scientific parameters.
5. Preserve unrelated dirty-worktree changes. Never reset, stash, commit, or
   rewrite user work merely to make a remote run possible.

## Choose the operating mode

- For inspection, validation, diagnosis, or status requests, remain read-only.
- For planning requests, create a dispatch plan but do not submit remote work.
- For execution requests, submit only an already-reviewed, commit-addressed
  batch through a valid dispatch plan.
- For experiment design, eval changes, threshold selection, or meta-test
  access, stop and require the appropriate human decision. Do not encode that
  authority in scheduler artifacts.

## Inspect the pool

Use the configured default unless the user supplies another path:

```bash
POOL="${OCZY_RUNNER_POOL_CONFIG:-$HOME/.config/oczy/runner-pool.json}"

uv run python infrastructure/kaggle/runner_pool.py validate \
  --config "$POOL"

uv run python infrastructure/kaggle/runner_pool.py status \
  --config "$POOL" \
  --state "$QUEUE/state.json" \
  --active-only \
  --json
```

Report healthy, degraded, disabled, and unavailable accounts separately.
Distinguish local `pending` from provider `queued`. Treat a nonzero status exit
as a health signal while retaining the partial inventory it printed.

## Plan approved work

Confirm these conditions before planning:

- The batch comes from a reviewed campaign and contains immutable job names.
- Source, model, instrument, provider profile, arguments, seeds, and output
  schema are already registered.
- Source is clean and commit-addressed. If the worktree is dirty, do not hide
  or rewrite it; identify the exact blocker or use an already-clean committed
  source artifact.
- Relevant accounts have explicit capacities and verified credentials.

Create the non-submitting plan:

```bash
uv run python infrastructure/kaggle/runner_pool.py plan \
  "$QUEUE/batch.json" \
  --config "$POOL" \
  --state "$QUEUE/state.json" \
  --output "$QUEUE/dispatch-plan.json"
```

Before execution, inspect the plan and require:

- `all_assigned: true`;
- `ready_for_dispatch: true`;
- no assignment or inventory errors;
- expected `provider` and `account_id` for every job; and
- batch and pool-config SHA-256 bindings matching the current files.

Do not dispatch a diagnostic-only plan produced from missing state or degraded
relevant accounts.

## Dispatch through the reviewed plan

Use one shared lease path for every scheduler that can consume the same
accounts:

```bash
LEASES="${OCZY_RUNNER_LEASE_STATE:-$HOME/.local/state/oczy/runner-pool-leases.json}"

uv run python infrastructure/kaggle/parallel_scheduler.py run \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json" \
  --pool-config "$POOL" \
  --dispatch-plan "$QUEUE/dispatch-plan.json" \
  --lease-state "$LEASES"
```

Never combine pool-aware dispatch with `--watch-batch`. Regenerate and review
the plan after any batch or pool-config change. Do not manually change a
persisted `account_id` to force failover.

## Monitor and recover

Use both views:

```bash
uv run python infrastructure/kaggle/parallel_scheduler.py status \
  "$QUEUE/batch.json" \
  --state "$QUEUE/state.json"

uv run python infrastructure/kaggle/runner_pool.py status \
  --config "$POOL" \
  --state "$QUEUE/state.json" \
  --json
```

- Respect the per-state owner lock. Do not remove its file to bypass a live
  owner; verify the process first.
- Treat leases as capacity claims, not job authority. Do not delete live
  leases merely to increase concurrency.
- Restart the same scheduler command for Kaggle resume. Expect an interrupted
  Colab process without a local handle to fail explicitly and preserve its
  diagnostics.
- Never edit `failed` back to `pending`. Diagnose it, retain the failed
  attempt, and create a new uniquely named execution only when justified.
- Preserve nulls, refutations, missing artifacts, and infrastructure failures
  in collection and adjudication.

## Validate control-plane changes

When modifying the runner pool or scheduler, run:

```bash
uv run ruff check \
  infrastructure/kaggle/runner_pool.py \
  infrastructure/kaggle/parallel_scheduler.py \
  infrastructure/kaggle/colab_provider.py \
  scripts/tests/test_remote_runner_pool.py \
  scripts/tests/test_pool_aware_scheduler.py

uv run pytest -q \
  scripts/tests/test_remote_runner_pool.py \
  scripts/tests/test_pool_aware_scheduler.py \
  scripts/tests/test_kaggle_parallel_scheduler.py \
  scripts/tests/test_colab_parallel_provider.py \
  scripts/tests/test_remote_experiment_campaign.py

git diff --check
```

Do not modify protected eval or research-contract paths as an incidental
scheduler change. If an explicit eval change is requested, follow the version
bump and human-signoff procedure in `AGENTS.md`.

## Report the outcome

State separately:

- what was confirmed from live providers;
- what was only planned;
- whether any remote submission actually occurred;
- account health, active jobs, and available capacity;
- state, plan, lease, and collected-artifact paths;
- degraded or blocked conditions; and
- validation commands and results.
