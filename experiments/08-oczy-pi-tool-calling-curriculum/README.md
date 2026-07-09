# Experiment 08: Oczy Pi Tool-Calling Curriculum

## Objective

Can the Oczy plastic cortex (KVCortex + logit biasing + correction loop)
teach a frozen LFM2.5-1.2B Q4 model to reliably use Pi's tools
(read, bash, write, edit) across multi-turn agentic tasks — without
fine-tuning the LM?

The benchmark (`benchmarks/pi/run_tool_use_benchmark.py`) scores 0/3
because the 1.2B model:
1. Sometimes outputs prose instead of tool-call format
2. Loses the original question after receiving a tool result
3. Doesn't know which tool to use for which task

This curriculum stages those three skills separately, using the
CortexAgent correction loop to steer the frozen LM toward tool-use
behavior via cvec + logit bias, then measures whether the learned
warm_state generalizes to unseen tasks.

## Setup

- **Driver:** real `LlamaCVecDriver` on
  `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`, `embedding=True`,
  `n_ctx=4096`, `n_threads=8`.
  The proxy server (`benchmarks/pi/proxy_server.py`) is the inference
  surface; the curriculum runs against the proxy's `LFMEngine` directly
  (not through Pi's full system prompt) for stages 0–4, then through
  `pi --model lfm-oczy --print` for stage 5 (Pi integration).
- **Agent:** `CortexAgent` with:
  - `cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random")`
  - `articulate_scale=0.001` (known-safe band for proj_random)
  - `use_logit_bias=True, logit_bias_strength=20.0` (exact-token recall)
  - `auto_consolidate=False` (explicit `consolidate()` between stages)
  - `correction_signal=1.0` on failed tool calls (required for plasticity)
- **Proxy:** `benchmarks/pi/proxy_server.py` with tool-call parsing
  (bracket + JSON format). The proxy's `_build_prompt` injects tool
  definitions and a concrete-example reminder before the assistant turn.
- **Reuse, do not reinvent:**
  - `benchmarks/pi/proxy_server.py` — `LFMEngine`, `_parse_tool_calls`,
    `_build_prompt`, `_format_tools` (already implemented)
  - `src/oczy/experiments/cortex_agent.py` — `CortexAgent` correction
    loop, `perceive()`, `articulate()`, `consolidate()`
  - `src/oczy/experiments/organism_curriculum/dataset.py` — `Episode`,
    `Probe`, `Stage` dataclasses (extend, don't modify)
  - `src/oczy/experiments/organism_curriculum/scoring.py` —
    `probe_matches` (extend with tool-call matchers)
  - `benchmarks/pi/run_tool_use_benchmark.py` — the 3-task benchmark
    is the Stage 5 acceptance gate

## Failure mode → mechanism mapping

| Failure | Observed behavior | Oczy mechanism | Curriculum stage |
|---------|-------------------|----------------|-----------------|
| F1: No tool call | Model outputs prose ("I don't have the capability...") instead of `[write(path="...")]` | Logit bias toward bracket tokens `[read(`, `[bash(`, `[write(`, `[edit(`; cvec steering toward tool-use subspace | Stage 0, 1 |
| F2: Lost context | Model calls tool, gets result, then says "could you clarify?" — forgets the original question | Correction loop: `correction_signal=1.0` on failed post-tool answers; warm_state retains task-goal signal across turns | Stage 2, 3 |
| F3: Wrong tool | Model says "there are no files" without searching, or "I can't create files" instead of using `write` | Scope-slot reranker: request embedding → tool-name slot; logit bias on the correct tool name | Stage 1, 4 |

## Stages

### Stage 0: Tool-format grounding

**Skill:** When asked to perform an action, output a tool call in
bracket format `[tool(param="value")]` instead of prose.

**Setup:** Minimal system prompt (no Pi harness):
```
You are a coding assistant. Use tools to act.
Tools: read(path="..."), bash(command="..."), write(path="...", content="...")
```

**Episodes (8):** Each episode is a single-turn request with a single
correct tool call.

| ID | Request | Expected tool call |
|----|---------|-------------------|
| s0_read | "Read the file config.toml" | `[read(path="config.toml")]` |
| s0_bash | "List files in src/" | `[bash(command="ls src/")]` |
| s0_write | "Create a file hello.py with print('hi')" | `[write(path="hello.py", content="print('hi')")]` |
| s0_read2 | "Show me main.py" | `[read(path="main.py")]` |
| s0_bash2 | "Run the tests" | `[bash(command="pytest")]` |
| s0_write2 | "Save 'TODO' to notes.txt" | `[write(path="notes.txt", content="TODO")]` |
| s0_read3 | "Open utils.py" | `[read(path="utils.py")]` |
| s0_bash3 | "Check git status" | `[bash(command="git status")]` |

**Correction:** If the model outputs prose instead of a tool call,
send a correction: `"No, you must call a tool. Output [read(path="config.toml")]"`
with `correction_signal=1.0`. The cortex observes the correction hidden
and updates warm_state.

**Probe:** Repeat the same request. Score `tool_format_correct` if
`_parse_tool_calls(output)` returns a tool call with the right name.

**Acceptance:** ≥7/8 episodes produce correct tool-call format on the
post-correction probe.

---

### Stage 1: Tool selection

**Skill:** Given a request, select the correct tool and fill in
parameters from the request text.

**Setup:** Same minimal system prompt, now with all 4 tools.

**Episodes (12):** Each episode tests tool selection — the model must
pick the right tool AND extract the correct parameter from the request.

| ID | Request | Expected tool | Expected param |
|----|---------|--------------|----------------|
| s1_read_1 | "Read pyproject.toml" | read | path="pyproject.toml" |
| s1_read_2 | "Show the file src/main.py" | read | path="src/main.py" |
| s1_bash_1 | "List all .py files" | bash | command contains "ls" or "find" |
| s1_bash_2 | "Run pytest" | bash | command contains "pytest" |
| s1_write_1 | "Create test.py with print('ok')" | write | path="test.py" |
| s1_write_2 | "Save 'hello' to out.txt" | write | path="out.txt" |
| s1_edit_1 | "Change 'foo' to 'bar' in config.py" | edit | path="config.py" |
| s1_edit_2 | "Replace 'old' with 'new' in app.py" | edit | path="app.py" |
| s1_read_3 | "What's in README.md?" | read | path="README.md" |
| s1_bash_3 | "Show git log" | bash | command contains "git log" |
| s1_write_3 | "Write 'done' to complete.txt" | write | path="complete.txt" |
| s1_read_4 | "Open the Dockerfile" | read | path="Dockerfile" |

**Correction:** If the model picks the wrong tool or gets the parameter
wrong, send a correction with the correct tool call and
`correction_signal=1.0`.

**Logit bias:** Bias toward the correct tool-name token
(`read`, `bash`, `write`, `edit`) on the first generated token after
`[`. This is the F3 fix — the model needs a nudge toward the right
tool name.

**Probe:** New requests (not seen in correction) that require the same
tool-selection skill. Score `tool_selected_correct` if the tool name
matches AND the parameter contains the expected value.

**Acceptance:** ≥9/12 episodes select the correct tool with correct
parameters on the post-correction probe.

---

### Stage 2: Tool-result integration

**Skill:** After receiving a tool result, answer the original question
using the result — don't lose context.

**Setup:** Two-turn episodes. Turn 1: model calls a tool. Turn 2: tool
result is injected, model must answer.

**Episodes (8):**

| ID | Request | Tool result (simulated) | Expected answer contains |
|----|---------|------------------------|-------------------------|
| s2_name | "Read pyproject.toml and tell me the project name" | `[project]\nname = "oczy"` | "oczy" |
| s2_ver | "Read pyproject.toml and tell me the version" | `[project]\nversion = "0.1.0"` | "0.1.0" |
| s2_count | "Count .py files in src/" | `42` | "42" |
| s2_branch | "Show current git branch" | `main` | "main" |
| s2_deps | "List dependencies from pyproject.toml" | `numpy, pytest` | "numpy" and "pytest" |
| s2_test | "Run pytest and tell me if tests pass" | `5 passed` | "pass" |
| s2_file | "Read utils.py and tell me what function is defined" | `def helper():` | "helper" |
| s2_error | "Run the script and tell me the error" | `SyntaxError: invalid syntax` | "SyntaxError" |

**Correction:** If the model's post-tool-result answer doesn't contain
the expected value, send a correction: `"No, the answer is 'oczy'.
The tool result says name = 'oczy'."` with `correction_signal=1.0`.

**Cvec role:** The cortex's warm_state should retain a "task goal"
signal from the original request across the tool-result turn. The cvec
emitted on turn 2 should steer the model toward answering, not asking
for clarification.

**Probe:** New two-turn episodes with unseen file contents. Score
`answer_uses_tool_result` if the expected value appears in the answer.

**Acceptance:** ≥6/8 episodes produce answers containing the expected
value from the tool result.

---

### Stage 3: Multi-turn tool chains

**Skill:** Call a tool, get a result, call a second tool if needed,
then answer — maintaining context across multiple tool turns.

**Setup:** Up to 3-turn episodes.

**Episodes (6):**

| ID | Request | Turn 1 expected | Turn 2 expected | Final answer contains |
|----|---------|-----------------|-----------------|----------------------|
| s3_read_edit | "Read config.py, then change DEBUG to True" | `[read(path="config.py")]` | `[edit(path="config.py", ...)]` | "changed" or "done" |
| s3_bash_read | "Find the file containing 'main', then read it" | `[bash(command="grep -rl 'main'")]` | `[read(path="<found>")]` | file content reference |
| s3_read_bash | "Read the test file, then run it" | `[read(path="test.py")]` | `[bash(command="pytest test.py")]` | "pass" or test result |
| s3_write_bash | "Create a script that prints hello, then run it" | `[write(path="...", content="print('hello')")]` | `[bash(command="python ...")]` | "hello" |
| s3_bash_bash | "Check git status, then commit if clean" | `[bash(command="git status")]` | `[bash(command="git commit")]` (conditional) | "commit" or "clean" |
| s3_read_read | "Read pyproject.toml and README.md, compare them" | `[read(path="pyproject.toml")]` | `[read(path="README.md")]` | comparison or summary |

**Correction:** If the model loses context between turns, send a
correction reminding it of the original request and the tool results
so far, with `correction_signal=1.0`.

**Probe:** New multi-turn episodes. Score `chain_completed` if the
model makes all expected tool calls AND the final answer references
the tool results.

**Acceptance:** ≥4/6 episodes complete the full tool chain with a
correct final answer.

---

### Stage 4: Tool selection under ambiguity

**Skill:** When multiple tools could apply, pick the right one based on
the request's intent — not just keyword matching.

**Setup:** Ambiguous requests where the wrong tool is tempting.

**Episodes (8):**

| ID | Request | Wrong tool | Correct tool | Why |
|----|---------|-----------|-------------|-----|
| s4_grep | "Find all files containing 'import numpy'" | read | bash (grep) | Searching content, not reading one file |
| s4_find | "Find files named config.py" | read | bash (find) | Searching by name, not reading |
| s4_edit_vs_write | "Fix the typo in readme.md" | write | edit | Modifying existing file, not creating |
| s4_read_vs_bash | "What does the Makefile do?" | bash (cat) | read | Reading a file, not executing |
| s4_bash_vs_read | "Show me the directory structure" | read | bash (ls/find) | Listing, not reading a file |
| s4_write_vs_edit | "Add a new test function to test_api.py" | write | edit | Adding to existing file |
| s4_bash_vs_edit | "Rename all .txt files to .md" | edit | bash (mv) | Bulk operation, not single-file edit |
| s4_read_vs_grep | "Where is 'def main' defined?" | read | bash (grep) | Searching, not reading |

**Correction:** If the model picks the wrong tool, send a correction
explaining why the other tool is correct, with `correction_signal=1.0`.

**Scope-slot reranker:** The request embedding should map to a
tool-name slot. If the model's tool call doesn't match the slot's
label, the reranker boosts the correct tool name.

**Probe:** New ambiguous requests. Score `tool_selection_correct`
if the correct tool is chosen.

**Acceptance:** ≥5/8 episodes select the correct tool on the
post-correction probe.

---

### Stage 5: Pi integration (the benchmark)

**Skill:** Full Pi CLI tool use — Pi's ~10K token system prompt, all
tools, multi-turn, real file I/O.

**Setup:** Run `benchmarks/pi/run_tool_use_benchmark.py` through
`pi --model lfm-oczy --print --no-session`.

**Pre-condition:** Stages 0–4 passed acceptance. Cortex warm_state is
consolidated (`consolidate()` called after Stage 4).

**Tasks (same as benchmark):**
1. read-file: "Read pyproject.toml and tell me the project name"
2. find-file: "Find all Python files containing 'CortexAgent'"
3. edit-file: "Create a file at /tmp/oczy_bench_marker.py"

**Cvec injection:** The proxy server injects the cortex's cvec into
the LM's forward pass during generation. The warm_state from Stages
0–4 should steer the model toward tool-use behavior even under Pi's
large system prompt.

**Logit bias:** The proxy biases toward tool-name tokens when the
model's output starts with `[`.

**Acceptance:** ≥2/3 benchmark tasks pass.

**Kill:** 0/3 pass after Stage 4 acceptance → the warm_state doesn't
generalize to Pi's prompt format. Document and hand off to
experiment 07 (conversation world model RL) for learned tool-use
policy.

## Conditions / ablation matrix

| # | Condition | What's tested | Held fixed |
|---|-----------|--------------|-----------|
| C1 | cvec only (no logit bias, no correction) | Does warm_state alone steer tool use? | stages, seed |
| C2 | logit bias only (no cvec, no correction) | Does token bias alone fix format? | stages, seed |
| C3 | cvec + logit bias (no correction) | Do steering surfaces compose for tool use? | stages, seed |
| C4 | correction only (no cvec, no logit bias) | Does the correction loop alone teach tool use? | stages, seed |
| C5 | cvec + logit bias + correction (full) | Does the full stack generalize? | stages, seed |
| C6 | no augmentation (vanilla baseline) | What does the raw 1.2B model do? | stages, seed |

C1–C5 isolate which mechanism drives each stage's improvement.
C6 is the control (must match benchmark 0/3).

## Metrics

- `tool_format_rate` = fraction of episodes where model outputs
  parseable tool-call format (Stages 0, 1, 4)
- `tool_selection_accuracy` = fraction where correct tool name is
  chosen (Stages 1, 4)
- `param_extraction_accuracy` = fraction where tool parameters contain
  the expected values (Stages 1, 4)
- `tool_result_integration` = fraction where post-tool answer contains
  expected value from tool result (Stage 2)
- `chain_completion_rate` = fraction where all expected tool calls are
  made in sequence (Stage 3)
- `benchmark_score` = 0–3, the Pi benchmark score (Stage 5)
- `tool_use_delta_per_byte` = `(benchmark_score_after − benchmark_score_before) / max(1, delta_persistent_bytes)`
  — the north-star metric: how much tool-use behavior changed per byte
  of persistent cortex state

## Acceptance & kill criteria

- **Stage 0 ACCEPT:** `tool_format_rate ≥ 0.875` (7/8) post-correction.
- **Stage 1 ACCEPT:** `tool_selection_accuracy ≥ 0.75` (9/12) AND
  `param_extraction_accuracy ≥ 0.75` post-correction.
- **Stage 2 ACCEPT:** `tool_result_integration ≥ 0.75` (6/8).
- **Stage 3 ACCEPT:** `chain_completion_rate ≥ 0.67` (4/6).
- **Stage 4 ACCEPT:** `tool_selection_accuracy ≥ 0.625` (5/8) on
  ambiguous requests.
- **Stage 5 ACCEPT:** `benchmark_score ≥ 2/3`.
- **KILL** if C6 (vanilla) passes any stage → the stage doesn't
  require Oczy augmentation; redesign to be harder.
- **KILL** if C5 (full) doesn't beat C4 (correction only) on Stage 2
  → cvec steering doesn't help context retention; the cortex isn't
  learning a task-goal signal.
- **KILL** if `tool_use_delta_per_byte ≈ 0` → the persistent state
  isn't encoding tool-use behavior; hand off to experiment 06
  (bounded-growth consolidation) for better compression.

## Expected failure modes

- **Stage 2 context overload:** The 1.2B model may not be able to
  maintain the original question across a tool-result turn even with
  cvec steering. The warm_state has d_cortex=8 — may be too small to
  encode both "tool-use mode" and "task goal". Mitigation: increase
  `d_cortex=16` for Stage 2 specifically.
- **Stage 5 Pi prompt size:** Pi's ~10K token system prompt may drown
  the cvec signal. The cvec at `articulate_scale=0.001` perturbs the
  residual stream by ~0.1% — this may be invisible under 10K tokens of
  system prompt. Mitigation: increase `articulate_scale=0.003` for
  Stage 5, or use logit bias as the primary steering surface (logit
  bias operates in logit space, not residual space, so it's immune to
  prompt-length dilution).
- **Bracket format degradation:** The model may produce malformed
  brackets under long prompts. Mitigation: the proxy's `_parse_tool_calls`
  already handles both JSON and bracket formats; add fuzzy matching.
- **Consolidation interference:** Consolidating tool-use patterns from
  Stage 0 (format) may interfere with Stage 1 (selection) if the
  warm_state conflates them. Mitigation: use separate cortex slots
  per stage, or consolidate only after Stage 4.

## Artifacts to add

- `src/oczy/experiments/tool_calling_curriculum/` — new package:
  - `dataset.py` — `ToolEpisode`, `ToolProbe`, `ToolStage` dataclasses
  - `stages/stage_0_format.json` through `stage_5_pi_integration.json`
  - `scoring.py` — `tool_call_matches`, `param_matches`,
    `answer_contains_result`
  - `run_curriculum.py` — driver that runs episodes through
    `LFMEngine` + `CortexAgent`, applies corrections, scores probes
- `src/oczy/experiments/tests/test_tool_calling_curriculum.py` —
  regression tests for scoring + data model
- `benchmarks/pi/run_tool_use_benchmark.py` — already exists; Stage 5
  reuses it directly
- `experiments_logs/YYYY-MM-DD_tool_calling_curriculum.md` — run notes

## Reproduce

```bash
# Stages 0–4 (direct, no Pi):
uv run python -m oczy.experiments.tool_calling_curriculum \
  --driver real --stages stage_0,stage_1,stage_2,stage_3,stage_4

# Stage 5 (through Pi):
# 1. Start proxy with cortex-augmented model:
uv run python benchmarks/pi/proxy_server.py --port 8080 \
  --model-path ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct-GGUF/snapshots/047e06635fbe71469926b35ea414537245218200/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
# 2. Run benchmark:
uv run python benchmarks/pi/run_tool_use_benchmark.py
```
