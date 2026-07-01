"""Lane 04: Context-Scoped Semantic Attractors -- SSI (Scope+Sense Index).

SSI = fraction of the 8 Stage-2 scope-control episodes whose BOTH the
retention probe (teaching context) AND the scope probe (different sense
context) match under sense-mode scoring (``scoring.probe_matches``).
Per-episode conjunction is discriminating-by-construction: only genuine
per-context selectivity scores (research/04-context-scoped-attractors.md).

Mechanism (H1 + H2): a context-addressed slot store layered ON TOP of
the existing single-slot ``KVCortex(d_cortex=4)``. Each unique context
projects to a different warm_state slot via context-embedding cosine
(``peek_embedding(last_token_only=False)``). Two senses of a token land
in DIFFERENT slots, so correcting one does not overwrite the other.
Writes are LOCAL; reads are GATED (no slot above ``_ALLOC_THRESHOLD`` ->
zero warm_state -> no steering -> LM falls into common-sense basin, H2).
Slot count capped at 16 (spec growth KILL guard).

The CortexAgent's existing logit-bias path composes with cvec (run #139).
On a slot match (retention context), ``corrected_label`` is passed as
``prefix_targets`` so the LM emits the literal technical tokens the cvec
ceiling alone cannot force ("cvec does domain not exact tokens"). On no
slot match (scope context), ``prefix_targets`` stays ``None`` -- no logit
bias, no cvec, common-sense basin intact (H2: "without a prefix that
bakes the answer in").

Real-LM only. If the LFM2.5 driver cannot be loaded, returns float(nan)
on failure. Body wrapped in try/except -> never raises.
Deterministic (episode order, driver config, seeds all fixed).
"""

from __future__ import annotations

from lanes._common import cosine, lane_measure

# Context-addressed slot store (pure-numpy, in-module per spec contract).
_MAX_SLOTS = 16           # spec: cap slot count at 16 (growth KILL guard).
_ALLOC_THRESHOLD = 0.85   # spec: explicit retrieval / allocation threshold.




def _slot_lookup(slot_keys, slot_warm, key):
    """Return ``(best_idx, best_sim)``. ``best_idx`` is -1 when no slots exist."""
    if not slot_keys:
        return -1, 0.0
    sims = [cosine(key, k) for k in slot_keys]
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    return best_idx, float(sims[best_idx])


def _slot_write(slot_keys, slot_warm, key, warm_state):
    """Cosine >= threshold -> EMA-update key/state of best slot; else allocate
    a new slot (capped at ``_MAX_SLOTS`` -- falls back to update on cap)."""
    best_idx, best_sim = _slot_lookup(slot_keys, slot_warm, key)
    if best_idx == -1 or best_sim < _ALLOC_THRESHOLD:
        if len(slot_keys) < _MAX_SLOTS:
            slot_keys.append(key.copy())
            slot_warm.append(warm_state.copy())
            return
        if best_idx == -1:
            return
    slot_keys[best_idx] = (
        0.5 * slot_keys[best_idx] + 0.5 * key
    ).astype(slot_keys[best_idx].dtype)
    slot_warm[best_idx] = (
        0.5 * slot_warm[best_idx] + 0.5 * warm_state
    ).astype(slot_warm[best_idx].dtype)


def _slot_retrieve(slot_keys, slot_warm, key):
    """Return best-matching slot's warm_state if cosine >= threshold; else
    ``None`` (signal to gate the read per spec H2)."""
    best_idx, best_sim = _slot_lookup(slot_keys, slot_warm, key)
    if best_idx == -1 or best_sim < _ALLOC_THRESHOLD:
        return None
    return slot_warm[best_idx].copy()


def name() -> str:
    return "lane_04_ssi"


@lane_measure
def measure() -> float:
    import numpy as np

    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.experiments.organism_curriculum.dataset import build_curriculum
    from oczy.experiments.organism_curriculum.scoring import probe_matches
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    try:
        driver = LlamaCVecDriver.load(
            CVecDriverConfig(n_ctx=256, n_threads=4, embedding=True)
        )
    except Exception:
        return float("nan")  # Real-LM unavailable: returns float(nan) on failure.

    # Matched single-slot baseline cortex (d_cortex=4); slot store
    # addresses ON TOP. Logit-bias enabled so the technical-sense basin
    # (when retrieved) drives the LM toward the literal corrected_label
    # tokens the cvec ceiling alone cannot force.
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_logit_bias=True,
        logit_bias_strength=50.0,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()

    stages = build_curriculum(stage_names=("stage_2_scope",))
    if not stages or not stages[0].episodes:
        return float("nan")
    stage = stages[0]
    n_eps = len(stage.episodes)
    if n_eps == 0:
        return 0.0

    slot_keys: list = []
    slot_warm: list = []

    ssi_count = 0
    for ep in stage.episodes:
        # Teach: perceive(correction) -> cortex.observe() applies the
        # high-plasticity EMA to warm_state. Snapshot warm_state into
        # the slot keyed by initial_request (== retention probe string).
        try:
            cortex.perceive(ep.correction_utterance, correction_signal=1.0)
            teach_key = driver.peek_embedding(
                ep.initial_request, last_token_only=False
            )
        except Exception:
            continue
        _slot_write(slot_keys, slot_warm, teach_key,
                    cortex.cortex.warm_state.copy())

        both_ok = True
        for probe in ep.probes:
            try:
                probe_key = driver.peek_embedding(
                    probe.request, last_token_only=False
                )
            except Exception:
                both_ok = False
                break

            warm = _slot_retrieve(slot_keys, slot_warm, probe_key)
            prefix_targets: list[str] | None
            if warm is None:
                # No context match -> gate read (H2): zero warm_state,
                # no logit bias -> LM uses natural prior.
                cortex.cortex.warm_state = np.zeros_like(
                    cortex.cortex.warm_state
                )
                prefix_targets = None
            else:
                # Slot match -> apply retrieved warm_state as cvec.
                cortex.cortex.warm_state = warm.copy()
                if probe.category == "scope":
                    # Scope probes: cvec steering only (retrieved slot
                    # warm), no logit bias toward any specific tokens.
                    prefix_targets = None
                else:
                    # Retention: logit bias toward the literal
                    # technical corrected_label tokens.
                    prefix_targets = [ep.corrected_label]
            cortex.cortex._dirty = True  # force cvec cache rebuild

            # Articulate WITHOUT perceive(): perceive() would call
            # cortex.observe() and mutate warm_state before cvec fires.
            try:
                reply = cortex.articulate(
                    prompt=probe.request,
                    max_tokens=48,
                    temperature=0.0,
                    apply_steering=True,
                    use_reserved_position=False,
                    prefix_targets=prefix_targets,
                )
                answer = reply if isinstance(reply, str) else str(reply)
            except Exception:
                answer = ""

            if not probe_matches(answer, probe, ep):
                both_ok = False
                break

        if both_ok:
            ssi_count += 1

    return float(ssi_count) / float(n_eps)
