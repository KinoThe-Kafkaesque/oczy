"""Write readable reports from completed, independently audited DEV outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "experiments_logs"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def study_report():
    path = LOGS / "2026-09-13_scoped_diversity_dev_v3.json"
    data = json.loads(path.read_text())
    assert data["verification"]["passed"] and not data["admission"]
    rows = data["rows"]

    def score(condition, seed, split, categories):
        selected = [r for r in rows if r["condition"] == condition and r["seed"] == seed and r["split"] == split and r["category"] in categories]
        return f"{sum(r['correct'] for r in selected)}/{len(selected)}"

    numeric = []
    for seed in range(3):
        for condition in ("initial", "control", "diversity"):
            numeric.append([seed, condition, *[score(condition, seed, "confirmation", (category,)) for category in ("amber", "cobalt", "silver", "quartz")]])
    references = [[condition, score(condition, None, "calibration", ("amber", "cobalt", "silver", "quartz")),
                   score(condition, None, "confirmation", ("amber", "cobalt", "silver", "quartz"))]
                  for condition in ("no_context", "direct_oracle", "complete_table", "text_control", "text_diversity", "chat_control", "chat_diversity", "zlib_control", "zlib_diversity")]
    fits = [[f["seed"], f"{sum(r['correct'] for r in f['training_fit'])}/27", f["first_ce"], f["last_ce"]] for f in data["training_fits"]]
    gradient = [r["gradient_norm_before_clip"] for r in data["trajectories"]]
    text = f"""# Scoped diversity DEV v3 — partial transfer improvement, acquisition gate failed

Completed September 14, 2026, Africa/Casablanca; protocol and training started September 13.
Manifest: `{data['manifest_sha256']}`.

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

{table(['Seed', 'Condition', 'Amber +vek', 'Cobalt +mip', 'Silver copy', 'Quartz copy'], numeric)}

Across all new-word contexts, varied teaching scores 12/16, 13/16 and 6/16;
the matched earlier controls score 9/16, 8/16 and 7/16. Aggregate gain hides a
seed-2 regression. Untaught quartz is 7/12 versus 8/12 in the earlier controls
and 12/12 initially. Taught silver improves to 12/12 from 11/12. Changed neutral
errors cancel in the total; they are not evidence of perfect preservation.
Per-input lost/gained pairs are recorded in the JSON report.

## Teaching fit and trajectory

{table(['Seed', 'Correct / 27', 'First update CE', 'Final update CE'], fits)}

The last update sees the last curriculum group, so its CE is not a whole-training-set
loss. Teaching fits are evaluated on all 27 examples per seed. All 144 gradients
are finite and nonzero: range {min(gradient):.6f}–{max(gradient):.6f}, with
{sum(g > 1 for g in gradient)}/144 above the existing clip threshold. The initial
state and first pre-update CE reproduce the old control for every seed.

## Retrieval and language references

{table(['Condition', 'Prior-word calibration / 16', 'New confirmation / 16'], references)}

Text and zlib conditions are independently regenerated after real decompression;
their raw outputs match. Conversation retrieval uses the same teaching examples
as separate user/assistant turns. The direct oracle selects the rule externally;
its scores do not establish learned context selection. Logit-bias and rerank
baselines are not implemented and were not run.

{table(['Teaching set', 'UTF-8 example bytes', 'zlib bytes'], [[s['arm'], s['example_utf8_bytes'], s['zlib_bytes']] for s in data['storage']])}

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
intact. Recorded training time: {sum(r['seconds'] for r in data['executions']):.2f}
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
"""
    path.with_suffix(".md").write_text(text)


def regression_report():
    path = LOGS / "2026-09-13_capability_regression_dev_v1.json"
    data = json.loads(path.read_text())
    assert data["verification"]["passed"]
    verdict = "All historical comparisons reproduce exactly" if data["historical_replay_exact"] else "Historical reproduction has discrepancies"
    text = f"""# Capability regression DEV v1

Completed September 14, 2026, Africa/Casablanca. Manifest: `{data['manifest_sha256']}`.

**{verdict}.** The 804 historical comparisons cover the six-capability battery,
serialized-state/text persistence, the direct-operation language improvement,
and the qualified scalar-compatible decoder. Reproducing an old failure is an
integrity pass, not a capability pass. Historical optimizers were not rerun.

{table(['Replay', 'Rows', 'Exact reproduction'], [[name, row['rows'], row['exact_replay']] for name, row in data['verdicts'].items()])}

## Candidate regressions on the unchanged earlier prompts

All three final diversity states receive the original v1 stage-2 probes, with
matched initial/control states and a zeroed state. These 160 probes include
individual rules, neutral copying and composition. The prompt remains the old
v1 prompt, so this is a robustness check beyond the new concise training interface.

{table(['Seed', 'Context', 'Earlier control', 'Varied teaching', 'Lost', 'Gained'], [[r['seed'], r['category'], f"{r['control_correct']}/{r['total']}", f"{r['diversity_correct']}/{r['total']}", len(r['lost']), len(r['gained'])] for r in data['paired']])}

Joint learning is not sequential accumulation or correction. Failed composition
does not isolate a composition mechanism when individual rules also fail.
Per-input changes and every raw response remain inspectable in the archived run.

## Actual tools and files

Forty-eight new action trials exercise the same four strict-JSON temporary-file
tasks. Initial/zeroed action controls fill the previous study's missing control
coverage. Tool core requires the exact call sequence, all actual files correct,
and no execution error. Strict success also requires the exact final answer.
The frozen action prompt's known final-answer placeholder is preserved.

{table(['Condition', 'Seed', 'Tool core / 4', 'Strict / 4'], [[r['condition'], r['seed'], f"{r['tool_core_correct']}/{r['total']}", f"{r['strict_correct']}/{r['total']}"] for r in data['candidate_action_counts']])}

## Verification and packaged experience

All 1,012 rows have exact coverage and independently checked scores. Historical
comparisons include complete row objects, prompt hashes, raw strings and tool
outcomes. Candidate actions independently check expected calls, actual files,
final answers and execution errors. The model/scorer/runtime/source identities
remain pinned; zero optimizer steps and no meta-test access. Replay runs in one
fresh offline process, which is not the original phase-isolated training design.
Elapsed replay time: {data['execution']['seconds']:.2f} seconds.

The [local workbench](../workbench/README.md) combines this evidence with saved
trials and real exploratory inference. All seeds and errors are visible. Its
seven-condition live comparison does not alter frozen scores or learned state.
Model weights/runtime remain external to the source-and-evidence archive.
"""
    path.with_suffix(".md").write_text(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=("study", "regression"))
    {"study": study_report, "regression": regression_report}[parser.parse_args().report]()
