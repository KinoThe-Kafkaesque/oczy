# 16 — Standing external benchmark battery (Sprint 4 / S4.2)

**Pre-registered 2026-07-02** (human-approved sprint setup).
Agents MUST NOT edit this spec. Depends on research/11 (the organism under
test is the minimal loop, plus whatever research/14–15 keep).

## Problem

The 2026-07-01 external QA run showed the organism *worse* than vanilla
cross-domain (0.388 vs 0.512) — the overfitting signature: home-curriculum
gains that cost general capability. One-off external checks rot; the battery
must be standing, uneditable by the optimizing loop, and include at least one
benchmark this repo did not author.

## Battery composition (fixed here)

1. **External QA** — the existing cross-domain QA set (repo-authored,
   out-of-curriculum). Frozen copy + SHA-256 manifest, same regime as eval/v2.
2. **Pi tool-use benchmark** — `benchmarks/pi` harness (repo-authored).
   Runs the version on main at battery time; its own manifest.
3. **Non-authored benchmark** — a fixed 200-item slice of an established
   public benchmark, feasible for a 0.5B CPU model, data vendored locally
   (offline, no network at run time). Selection criteria fixed now: multiple
   choice or short-answer, published before 2026, unmodified items, slice
   chosen by seeded hash of item ids (seed=42), scored by the benchmark's own
   published metric. Concrete choice (e.g. ARC-Easy slice) is recorded at
   implementation time in the battery manifest — items may not be selected by
   looking at model performance.

## Protocol

- Runs weekly (standing job: `scripts/weekly_battery.sh`, cron-able; also
  runnable on demand) and appends — never edits — one row per run to the S4.4
  dashboard in `experiments_logs/`.
- Conditions per benchmark: **vanilla HFDriver** and **organism** (minimal
  loop + kept components, trained on its home curriculum first), same
  decoding config, ≥3 seeds where the organism is stochastic.
- The battery script verifies all manifests before running and refuses to run
  on a dirty working tree for protected paths.

## Primary metric & standing acceptance bar

`external_delta(b)` = organism − vanilla accuracy on benchmark b.

- **Green:** `external_delta(b) ≥ −0.02` on ALL benchmarks (the organism may
  not buy home-curriculum gains with general capability).
- **Red (overfitting signature):** any benchmark with `external_delta < −0.05`
  with 95% CI excluding −0.02. A red battery BLOCKS promotion of any
  concurrent home-eval claim to a headline until resolved.
- Between: YELLOW, reported, not blocking.

This is a standing gate, not a one-shot hypothesis: every weekly row gets a
Green/Yellow/Red stamp by the fixed rule above.

## Reporting

One dashboard row per run: date, git SHA, per-benchmark vanilla/organism
accuracy ± CI, delta, stamp. First run logged to
`experiments_logs/<date>_s4_2_external_battery_baseline.md` quoting this spec.
