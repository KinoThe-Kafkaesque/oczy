# Eval v2.2 — curriculum protocol repair

**Date:** 2026-07-11  
**Approval:** explicit human instruction in-session to fix the reviewed
curriculum and difficulty-scaling defects.  
**Scope:** execution order and split policy only. Existing episode text,
expected answers, probe categories, match modes, and scoring thresholds were
not edited.

## Repairs

1. Stage 1 is now a probe-only transfer battery. Its own correction exemplars
   are not delivered by the default runner.
2. Stage 3 probes run immediately after their corresponding correction rather
   than after every episode has been acquired.
3. Stage 4 consolidation runs before its post-test and before the
   `memory_bytes_after` snapshot.
4. The semantic-scoring flag is passed to post-tests as well as pre-tests and
   immediate uptake checks.
5. The default v2.2 split is deterministic and stratified by probe category.
   Every category with at least two probes has both dev and holdout coverage.
6. Explicit `salt="v2"` calls retain the exact legacy v2/v2.1 hash-threshold
   behavior so registered historical runs remain reproducible.

## v2.2 split distribution

At `fraction=0.3`, `salt="v2.2"`:

| Stage | Dev | Holdout | Holdout categories |
|---|---:|---:|---|
| 0 grounding | 14 | 6 | retention=6 |
| 1 transfer | 28 | 12 | transfer=12 |
| 2 scope | 20 | 10 | retention=4, scope=6 |
| 3 dialog | 4 | 4 | transfer=2, scope=2 |
| 4 consolidation | 7 | 3 | retention=3 |
| 5 cross-domain | 8 | 4 | retention=2, scope=2 |

These are exact allocations, not observed model-performance thresholds.

## Evidence status

The July 1 real-driver 0.88/0.75/0.69/0.38/0.90/0.92 table remains a legacy
v2 reference only. It is not a v2.2 baseline and must not be used as the
current difficulty curve. A new real-driver, multi-seed v2.2 baseline is
required before empirical stage-difficulty claims resume.
