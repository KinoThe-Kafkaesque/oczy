# AGENTS.md — Standing Working Agreements

These rules apply to every agent (human or autonomous) working in this repo.
They are inherited automatically by autonomous sessions. Violating them
invalidates the results of the run.

## Standing Working Agreements

1. **The optimizing loop never touches the measuring instrument.** Metrics,
   thresholds, baselines, episodes, scoring: frozen per version, changed only
   by explicit human decision with a version bump.
2. **No episode-ID-conditioned code, ever.** Fixes must be mechanism-level.
3. **One variable at a time.** Multi-parameter commits cannot claim causal
   improvements; "breakthrough" requires ablation + trajectory + seeds.
4. **Nulls and refutations are results.** Log them as prominently as wins.
5. **Retrieval is the baseline, not the enemy.** Prefix/logit-bias/rerank
   paths stay in every table as the bar that changed-dynamics must clear;
   claiming their wins as metabolism is the failure mode to avoid.
6. **Every threshold gets a distribution check** against real data before it
   ships.

## How to Change the Eval

The eval (metrics, thresholds, baselines, episodes, scoring) is frozen per
version. To change it:

1. **Bump the eval version number.**
2. **Get human sign-off.** No autonomous session may self-approve an eval
   change.
3. **Recompute the manifest.** After making deliberate changes to `eval/v2/`
   files, run:
   ```bash
   python scripts/bump_eval_version.py
   ```
   This recomputes the per-file hashes in `eval/v2/MANIFEST.json` so the
   change is recorded as intentional. The `verify_manifest()` function in
   `eval/v2/__init__.py` enforces hash integrity at curriculum load time —
   the harness refuses to run if a tracked file's hash changed without a
   manifest update. Set `EVAL_CHANGE_APPROVED=1` to bypass the hash check
   during development.
4. **Run the guard with the override flag:**
   ```bash
   EVAL_CHANGE_APPROVED=1 python scripts/eval_guard.py --allow
   ```
   This confirms the change is intentional and approved.
5. **Commit with a clear justification** referencing the version bump and the
   human decision that authorized it.

`autoresearch.sh` runs `eval_guard.py` before committing, so any unapproved
edit to a protected path (`eval/`, `research/`, `lanes/`,
`experiments/organism_curriculum/`) will block the commit.
