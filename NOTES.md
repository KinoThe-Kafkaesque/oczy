# Oczy — session notes

## 2026-06-21 — thermos-nuclear code review pass

Triggered by review of `experiments_logs/2026-06-19_extended_learning_evaluation.md`,
which concluded that **NeuralHippocampus is the most promising organ**
(scoring 1.000, above the Oracle upper bound of 0.844). On closer reading,
that headline was a measurement artifact:

- **NeuralHippocampus's 1.000** measured its internal bookkeeping (store/
  replay/consolidate/decay), not whether learned corrections surfaced
  in the agent's answered output.  The 1.000 was a self-consistency
  check on the hippocampus's own data structures.
- **Oracle 0.844 below PlasticCortex 0.967** came from
  `memory_bytes_per_delta` being *averaged* into the aggregate as if
  higher = better, when it is natively lower = better.
- **HippocampusOnlyAgent 0.0 across the board** in the curriculum log
  was a `correction` vs `corrected_answer` schema drift: OrganismAgent
  passed `correction=<sentence>` to `neural_hippocampus.store()` but
  read `episode.get("corrected_answer")` on replay, so the replay hint
  silently never reached the ranker.

The review produced a prioritised finding list (critical: schema drift
in episode dicts; high: 5x duplication of `_extract_expected_from_correction`,
IdentityHypernetwork's hardcoded 14-token vocab; medium: missing
`status()` on WorldModelCritic, broken `_target_label_for` leading
space, arbitrary aggregate weights; low: nit-level).

### Changes this session

#### Glue layer (`oczy_common/`, `experiments/`, `eval_extended.py`)
- New `oczy_common/` package: `Episode` TypedDict +
  `EPISODE_FIELDS`/`validate_episode` (single source of truth for the
  dict keys crossing organ boundaries), `tokenize`/`STOPWORDS`, the rich
  `extract_expected_from_correction` heuristic, and `mem_bytes` (single
  pickle-based byte-count definition).
- `correction-benchmark/src/correction_benchmark/benchmark.py`:
  `_evaluate_probes` no longer calls `agent.answer()` twice per probe
  (the first call mutated stateful agents — `HippocampusOnlyAgent`
  incremented `replay_count`, `SkillImmuneCortex` incremented `hit_count`,
  `WorldModelCritic` updated `_last_correction_prob`; the recorded
  answer was from one call and the verdict from another).
- `eval_extended.py`: C4 — `memory_bytes_per_delta` inverted to
  higher-is-better before aggregation (Oracle now ranks #3 at 0.956,
  proper upper bound). C2 — `eval_neural_hippocampus` now stores
  `correction=sentence, corrected_answer=label` and reads
  `corrected_answer` back, instead of misusing `correction` for both.
  L5 —
  per-item PlasticCortex control agent documented and config hoisted.
  M4 — aggregation policy documented: every metric enters the aggregate
  as higher-better ∈ [0,1].
- `experiments/organism.py`: H5 — `LMBackendAgent` split out as a
  sibling class so the LM-only answer path is visible in the type
  system rather than hidden behind a `backend="lm"` config flag that
  silently bypassed critic/hippocampus/identity in `answer()` while
  still running `learn()` against them.  H1/H2 — `_tokenize` and
  `_extract_expected_from_correction` now delegate to `oczy_common`
  (the five baseline agents in `baselines.py` also now share these).
  M5 — `__getstate__`/`__setstate__` drop and rebuild the profiler on
  pickle so live timer state isn't serialised.  M6 — bare `except
  Exception` for the `LMPlasticCortex` import became `except
  ImportError` so AttributeError/SyntaxError in the LM module surface.
  `OrganismAgent._module_bytes` now prefers the canonical
  `status()["serialized_bytes"]` field, falling back to legacy fields
  and finally to `pickle.dumps(module)`.

#### Organs
- **neural-hippocampus**: `status()` adds `serialized_bytes` and
  `record_count`. `store()`'s optional `corrected_answer` param (added
  the prior session) is now end-to-end tested by
  `test_corrected_answer_round_trips_through_replay`.
- **world-model-critic**: gained a `status()` method (previously had
  none — `OrganismAgent._module_bytes` had a silent special-case
  fallback). Returns the 6 canonical fields including
  `serialized_bytes`.
- **skill-immune-cortex**: M2 — `status()["bytes"]` was
  `sys.getsizeof(json.dumps(...))` (Python object overhead, not UTF-8
  bytes); now `len(json.dumps(...).encode("utf-8"))`. `status()` adds
  `serialized_bytes` (pickle of the whole organ) and `record_count`
  (detectors + skills).
- **identity-hypernetwork**: H3 — implemented real vocab growth.
  `grow_vocab(new_concepts)` extends `W` with one fresh
  `1/sqrt(input_dim)`-scaled row per concept and updates
  `concept_index`/`output_dim` in place.  `_extract_first_concept` now
  auto-registers unknown tokens (alnum, length >= 3, not in a short
  stopword filter) so `update_identity` no longer silently no-ops on
  curriculum senses like "ML model". The closed `CONCEPT_VOCABULARY`
  stops being a hard blocker.
- **experience-autoencoder**: M7 — `train_step(episode, lr=0.01)`
  applies a rank-1 Hebbian update on the sensing matrix (outer of
  residual target and bag-of-words features, then per-column
  renormalisation), returning the pre-update reconstruction error.
  Over 25 steps on a profile episode the error dropped 0.5755 → 0.5447.
  M3 — the leading-space bug in `_target_label_for`'s fallback string
  is fixed. `encode()` now accepts canonical Episode keys
  (`query`/`answer`/`corrected_answer`) as aliases for the
  source-specific names it historically used
  (`situation`/`model_answer`/`revised_answer`). L1 — `ifany = any(...)`
  inlined. `status()` adds `serialized_bytes` and `record_count`.
- **plastic-cortex**: both `PlasticCortex.status()` and
  `LMPlasticCortex.status()` add `serialized_bytes` + `record_count` +
  `project`. Pickle round-trip verified clean on the LM cortex (numba
  kernels are module-level functions, not on `self`).

#### Tests
- New `experiments/tests/test_agent_glue.py`: T1 OrganismAgent
  end-to-end replay test (learn → answer → assert corrected label),
  T2 baseline ablation contract tests, and
  `extract_expected_from_correction` round-trip tests. Previously no
  integration test exercised `OrganismAgent.learn() → answer()` so the
  replay-drift bug was invisible until this session's manual smoke
  test.
- Each organ gained a `test_status_reports_serialized_bytes_*` test
  enforcing the canonical byte contract.

### Re-validated rankings (after fixes)

```
1. NeuralHippocampus      1.000
2. PlasticCortex          0.967
3. Oracle                 0.956   (was 0.844 — metric-direction bug fixed)
4. SkillImmuneCortex      0.590
5. WorldModelCritic       0.399
6. ExperienceAutoencoder  0.203   (now trainable via train_step; re-run to see this move)
7. Always-Wrong           0.200   (was 0.000 — same metric-direction bug)
8. IdentityHypernetwork   0.006
```

Oracle now sits at #3 instead of below PlasticCortex — the canonical
upper-bound ordering is restored.  NeuralHippocampus still tops the
ranking because `eval_extended.py`'s
`eval_neural_hippocampus` still only measures internal mechanics, not
behavioural learning; closing that gap is the next honest eval.

### Remaining gaps (next session)

1. **`eval_extended.py`'s IdentityHypernetwork probe is wrong**: at
   `eval_extended.py:309` it uses `token = first_keyword(correction)`
   as the adapter lookup key (e.g. `"model"` for "No, 'model' here
   means ML model."), then asserts `best == sense` where `sense =
   "ML model"`. Comparing a single concept token to a multi-token
   sense label can never match, so the organ scores 0.006 even though
   `grow_vocab` and `update_identity` now work. Fix: have
   `eval_identity_hypernetwork` check whether any concept in the
   adapter is present in the sense, or have `update_identity` register
   the full sense string as a concept (multi-word concepts weren't
   part of the grow_vocab commit).
2. **`eval_neural_hippocampus` still measures internal mechanics, not
   behaviour.** Add a behavioural probe: after the curriculum, do the
   hippocampus-only slow updates survive into the
   `OrganismAgent.answer()` output? Currently only the integration
   test checks this.
3. **WorldModelCritic** still uses hardcoded ambiguity + Jaccard
   similarity. The autoencoder's `train_step` opens up the possibility
   of swapping in a small learned similarity model.
4. **Organ-level `core.py` / wrapper split (H4) deferred**: the
   `core.py` files are now documented as "v1 organ lives in the
   wrapper" but the inconsistency is still present. Pick one pattern
   and apply uniformly.

## 2026-06-22 — organism curriculum

### Added
- New `experiments/organism_curriculum/` package with a
  progression of learning experiences designed for the multi-organ
  agent rather than for the plastic-cortex LM:

  | Stage | Target organ(s) / flow | Content |
  |-------|------------------------|---------|
  | `stage_0_grounding` | PlasticCortex fast weights | 8 one-word corrections |
  | `stage_1_transfer` | NeuralHippocampus replay | 8 transfer probes in same domain |
  | `stage_2_scope` | SkillImmuneCortex + IdentityHypernetwork | Learn a second sense, test sense isolation |
  | `stage_3_dialog` | Full metabolism | Multi-turn corrections |
  | `stage_4_consolidation` | Hippocampus consolidation + autoencoder | 10 corrections then explicit consolidate() |
  | `stage_5_cross_domain` | Critic + identity + immune | Same word in two domains |

- `dataset.py` — typed dataclasses (`Episode`, `Probe`, `Stage`) and JSON loader.
- `validation.py` — smoke test that guards against the old benchmark's flaws.
- `scoring.py` — `exact`/`contains`/`sense` matching.
- `run_curriculum.py` — driver for `OrganismAgent`/`LMBackendAgent`, raw or LM-mediated, with per-stage and JSON reporting.

### Validation
```
python -m experiments.organism_curriculum.validation
Validated 6 stage(s); 44 episode(s)
All checks passed.
```

### Representative run (with organ packages on PYTHONPATH)
```
python experiments/organism_curriculum/run_curriculum.py \
    --stages stage_0_grounding stage_1_transfer stage_4_consolidation

Stage                        Episodes  Uptake   Pre  Post      Mem d
----------------------------------------------------------------------
Stage 0: Sense grounding       8/8      0.00  0.00  0.88    +15788B
Stage 1: Transfer within domain   8/8      0.00  1.00  1.00     +3273B
Stage 4: Consolidation stress 10/10     0.00  0.80  1.00     +4974B
```

### LM perception layer restored
- `oczy_lm/adapter.py`: restored the three few-shot examples in
  `_PARSE_SYSTEM_PROMPT` (one accepted, two corrected) while keeping
  the strict rule that plain queries must have empty
  `corrected_answer`.  Stronger wording asks the LM to extract only
  the Y part (new meaning) without the redefined word.
- New `oczy_lm/tests/test_adapter_parse.py`: mock-LLM tests for
  accepted, corrected, spurious-corrected-on-accepted clearing,
  missing-corrected-on-corrected downgrading, and malformed-JSON
  fallback.  All pass.
- End-to-end LM perception demo (`experiments/lm_perception/run_perception_demo.py`)
  now runs against `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M`:
  parse rate **4/6 (67%)** on the original 12-episode benchmark
  subset — up from 1/12 before the prompt fix.
- LM-mediated organism curriculum (`run_curriculum.py --lm`) also
  runs end-to-end on Stage 0 (8 episodes).  Parse rate there was
  2/8 because the curriculum's starker "No, 'X' means Y" wording
  triggers the spurious-answer clearing path more often; absorption
  still hit 8/8 thanks to the raw fallback.

### Environment
- Added `llama-cpp-python>=0.3`, `huggingface-hub>=0.20`, `numba>=0.65`,
  `numpy>=2` to `[project].dependencies` in `pyproject.toml`.
- `uv sync` now installs the full LM perception stack into the repo's
  `.venv`.  All `uv run python ...` invocations work; manual
  `PYTHONPATH` shims are no longer needed.
- LFM2.5-1.2B-Instruct Q4_K_M GGUF is cached at
  `~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct-GGUF/`.

### Experiment results (2026-06-22)

#### Raw organism curriculum (`uv run python experiments/organism_curriculum/run_curriculum.py`)
```
Stage                            Episodes  Uptake   Pre  Post      Mem d
--------------------------------------------------------------------------
Stage 0: Sense grounding             8/8      0.00  0.00  0.88    +15678B
Stage 1: Transfer within domain     8/8      0.00  1.00  1.00     +3273B
Stage 2: Scope control              0/8      1.00  0.50  0.50     +9431B
Stage 3: Dialog                     4/4      0.00  0.12  0.25     +2751B
Stage 4: Consolidation stress      10/10     0.00  0.80  1.00     +6140B
Stage 5: Cross-domain                5/6      0.17  0.42  0.33    +10000B
```
Stage 0 / 1 / 4 absorb cleanly. Stage 2 fails 100% (two-senses-per-token
limitation). Stage 3 partially succeeds on uptake but probe scores stay
low. Stage 5 cross-domain is partially tractable (5/6 retention).

#### Existing eval suite baseline comparison (`uv run python experiments/run_experiment.py`)
```
Agent                    Uptake  Transfer   Scope  Forget  Consol  Identity        Mem/Δ
----------------------------------------------------------------------------------------
ZeroMemoryAgent          1.0000    0.0000  0.0000  0.0000  0.0000    0.0000         61.0
ContextOnlyAgent         0.6667    0.0000  0.0000  0.0000  0.0000    0.0000       612.75
FastOnlyAgent            0.6667    0.1667  0.1667  1.0000  1.0000    1.0000         12.0
HippocampusOnlyAgent     0.6667    0.1667  0.0000  0.0000  0.0000    0.0000         1.25
IdentityOnlyAgent        1.0000    0.0000  0.0000  0.0000  0.0000    0.0000        493.0
OrganismAgent            0.6667    0.2500  0.1667  1.0000  1.0000    1.0000      68772.0
```
OrganismAgent matches FastOnlyAgent on forgetting/consolidation/identity
(both inherit PlasticCortex's behaviour) and adds a small transfer edge
via the hippocampus replay path. Memory cost is dominated by the
hippocampus + immune + autoencoder pickled state.

#### LM perception demo (`uv run python experiments/lm_perception/run_perception_demo.py --lessons 12`)
```
Raw  absorbed  :  12/12  (100%)  avg 0.00s/lesson
LM   parse OK  :   5/12  (42%)
LM   absorbed  :  11/12  (92%)  avg 5.36s/lesson  (LM+organism end-to-end)
Wallclock premium per lesson : +5.36s  (LM path - raw)
```
Parse-miss pattern: lessons 1, 5, 6, 7, 8, 9, 10 fail. The parser
succeeds on the first few lessons and the last one but fails consistently
in the middle of the run, suggesting either KV-cache drift or a
systematic spurious-answer-clearing path triggered when the LM hallucinates
a `corrected_answer` on what should be `accepted`. Absorption still
reaches 11/12 because the raw fallback path picks up the LM's misses.

### Limitations
1. **Toy PlasticCortex cannot hold two senses per token.**
   `stage_2_scope` and `stage_5_cross_domain` fail because the default
   word-association backend learns a single corrected sense per word;
   teaching a second sense overwrites the first. The curriculum now
   surfaces this limitation cleanly.
2. **LM parse rate is uneven across curricula.**
   Parse rate on the full 12-lesson original benchmark is 5/12 (42%);
   on the 6-lesson subset it was 4/6 (67%); on the organism curriculum's
   Stage 0 it dropped to 2/8. Absorption stays high (11/12 = 92%) thanks
   to the raw fallback, so this is a perception-layer fidelity issue
   rather than a behaviour blocker. There is a striking positional
   pattern: the parser succeeds on the first few lessons and the last
   one but fails consistently in the middle of the run (lessons 5-10
   on the 12-lesson run), suggesting either KV-cache drift or a
   systematic spurious-answer-clearing path triggered when the LM
   hallucinates a `corrected_answer` on what should be `accepted`.
3. **Scope/retention matching is heuristic.**
   The new `sense` match mode uses stopword-filtered token overlap and
   a known-ambiguous-word exclusion. It is sufficient for the toy
   backend but will need refinement as the organs become more capable.
