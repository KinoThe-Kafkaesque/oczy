# Scoped diversity DEV v3 — partial transfer improvement, acquisition gate failed

Completed September 14, 2026, Africa/Casablanca; protocol and training started September 13.
Manifest: `e339fd54cedba4a039af0e98c2008b5934b781aff720f5da36ab2242c72c70b6`.

**Result:** varied teaching improves new-word suffix application to 12/24 from
5/24 for the matched earlier controls. Neutral copying remains 19/24 overall.
This does not validate reliable scoped learning: some context/seed combinations
are 0/4, teaching fit is only 67/81, and the frozen correction gate fails.
No correction updates or preservation-penalty comparison ran.

The single learner treatment is teaching-word diversity: rotate three crossed
nine-example groups instead of repeating one. Same initial numeric hashes,
three seeds, 48 updates/seed, nine example losses/update, Adam .03, clipping at 1,
8 × 896 FP32 state and scalar decoder. The earlier final controls are reused
directly. This matches 144 optimizer updates and 1,296 example losses, not
token counts or FLOPs; prompt/target token lengths are recorded in the run.

## New confirmation cohort

The words berry/cherry/papaya/orange were frozen before training. Each category
has four cases per seed. The paired comparison uses exactly the same inputs;
the earlier 6/24 result used a different cohort and is not this comparison's baseline.

| Seed | Condition | Amber +vek | Cobalt +mip | Silver copy | Quartz copy |
|---|---|---|---|---|---|
| 0 | initial | 0/4 | 0/4 | 4/4 | 4/4 |
| 0 | control | 2/4 | 0/4 | 4/4 | 3/4 |
| 0 | diversity | 0/4 | 4/4 | 4/4 | 4/4 |
| 1 | initial | 0/4 | 0/4 | 4/4 | 4/4 |
| 1 | control | 0/4 | 0/4 | 4/4 | 4/4 |
| 1 | diversity | 4/4 | 2/4 | 4/4 | 3/4 |
| 2 | initial | 0/4 | 0/4 | 4/4 | 4/4 |
| 2 | control | 0/4 | 3/4 | 3/4 | 1/4 |
| 2 | diversity | 2/4 | 0/4 | 4/4 | 0/4 |

Across all new-word contexts, varied teaching scores 12/16, 13/16 and 6/16;
the matched earlier controls score 9/16, 8/16 and 7/16. Aggregate gain hides a
seed-2 regression. Untaught quartz is 7/12 versus 8/12 in the earlier controls
and 12/12 initially. Taught silver improves to 12/12 from 11/12. Changed neutral
errors cancel in the total; they are not evidence of perfect preservation.
Per-input lost/gained pairs are recorded in the JSON report.

## Teaching fit and trajectory

| Seed | Correct / 27 | First update CE | Final update CE |
|---|---|---|---|
| 0 | 26/27 | 7.513015588124593 | 0.04881391374187337 |
| 1 | 24/27 | 7.418931749131945 | 0.2758840603960885 |
| 2 | 17/27 | 7.250552336374919 | 0.15730165710879696 |

The last update sees the last curriculum group, so its CE is not a whole-training-set
loss. Teaching fits are evaluated on all 27 examples per seed. All 144 gradients
are finite and nonzero: range 0.131341–126.357864, with
110/144 above the existing clip threshold. The initial
state and first pre-update CE reproduce the old control for every seed.

## Retrieval and language references

| Condition | Prior-word calibration / 16 | New confirmation / 16 |
|---|---|---|
| no_context | 8/16 | 8/16 |
| direct_oracle | 15/16 | 16/16 |
| complete_table | 3/16 | 5/16 |
| text_control | 2/16 | 1/16 |
| text_diversity | 7/16 | 6/16 |
| chat_control | 10/16 | 9/16 |
| chat_diversity | 14/16 | 10/16 |
| zlib_control | 2/16 | 1/16 |
| zlib_diversity | 7/16 | 6/16 |

Text and zlib conditions are independently regenerated after real decompression;
their raw outputs match. Conversation retrieval uses the same teaching examples
as separate user/assistant turns. The direct oracle selects the rule externally;
its scores do not establish learned context selection. Logit-bias and rerank
baselines are not implemented and were not run.

| Teaching set | UTF-8 example bytes | zlib bytes |
|---|---|---|
| control | 463 | 123 |
| diversity | 1405 | 214 |

Each neural state still occupies 28,800 file bytes (28,672 numeric bytes), larger
than either example payload. No learned compression claim is supported.

## Verification and boundary

All 608 probe rows, 81 teaching fits, 144 updates, prompt/source/data/state hashes,
runtime identity, unchanged model/scorer, exact coverage and paired score calculations
are independently audited. Train and evaluation use separate offline processes;
registered training-inaccessible files are confirmed absent in the training audit.
The controller failed after evaluation while appending its second ledger record:
its JSON writer only permits creating a new file. All 608 results were already
written, but evaluation exit status, duration and the pre-run evaluation firewall
check output were not durably retained. A separate, hashed recovery validates
the complete results and invokes the unchanged admission function, with zero
new model calls or optimizer updates. The original failure and ledger remain
intact. Recorded training time: 1943.92
seconds; total execution duration is unavailable. This provenance limit must
travel with the result. The optimizer is not
given probes, oracle rules, old control files or the data-builder source.

This is joint local DEV gradient learning, not sequential accumulation, a learned
writer, R20 acceptance or meta-test evidence. Nulls and regressions remain in the
record. The next blocker is stable acquisition across contexts and seeds before
testing correction while preserving unrelated knowledge. More words alone under
this budget provide a partial benefit, not a solution.

See [full scores and trajectory](2026-09-13_scoped_diversity_dev_v3.json),
[registered protocol](../experiments/scoped-diversity-dev-v3/README.md), and
[regression replay](2026-09-13_capability_regression_dev_v1.md).
