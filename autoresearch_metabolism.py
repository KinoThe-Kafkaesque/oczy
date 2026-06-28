#!/usr/bin/env python3
"""Phase-2 autoresearch harness: real-LM drift-drives-answer logit-rise probe.

research/05 C2's actual softer criterion: "domain-word LOGIT at the probe
position rises with K (Spearman rho > 0.5, p < 0.05)". Segment 12 measured
the STRONGER criterion (target token in greedy output) and produced an honest
null. Segment 13 measures the softer criterion: does H1+H2 drift
move the LM's logit at the target token position, even when greedy decoding
isn't strong enough to actually emit it?

Per-K-point flow:
  1. Fresh CortexAgent (H1+H2 cortex code unchanged from segments 1-11).
  2. SVD-init proj_c from 16 paraphrases of CORRECTION (F3 fix retained).
  3. Apply K correction cycles (H1+H2 path fires per F2 fix; replays=[h,h,h]).
  4. Apply cvecs via driver.set_cvecs_per_layer at articulate_scale=0.03.
  5. Forward-only on PROBE via driver._llm.eval(prompt_ids) (NO generation).
  6. Extract (n_vocab,) logits at last position via driver._llm._ctx.get_logits().
  7. Read logit at target token id (" marmalade", BPE-natural continuation).
  8. Read logit at CONTROL token id (" banana", real word NOT in CORRECTION).

Primary metric: drift_logit_spearman = Spearman(K_samples, logit_at_target_K - logit_at_target_0).
Secondary: control_logit_spearman = Spearman(K_samples, logit_at_control_K - logit_at_control_0).
Anti-gaming: target delta should rise more than control delta (specificity).
"""

from __future__ import annotations

import sys

import numpy as np

from plastic_cortex.kv_cortex import KVCortexConfig
from src.oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig


CORRECTION = "The secret passphrase for level 7 is marmalade."
PROBE = "Answer briefly.\nQuestion: What is the secret passphrase for level 7?\nAnswer:"

# K-sweep. 9 points (was 7 in segment 15): extending to K=25,30 to tighten
# the borderline segment-15 p=0.0469. Pre-registered at segment 16 bump as
# CONFIRMATION iter — no metric def change, no cortex code change, no harness
# architecture change, only additional K points to broaden the Spearman sample.
# Pre-registered outcomes:
#   - rho REMAINS > 0.5 AND p DROPS further below 0.05 -> segment-15 result confirmed
#   - rho DROPS below 0.5 OR p RISES above 0.05 -> segment-15 was borderline/variance
#   - K=25,30 target delta CONTINUES upward arc -> real nonlinear compounding
#   - K=25,30 RETREATS toward baseline -> K=20 was a local peak
K_SAMPLES = [0, 1, 2, 5, 10, 15, 20, 25, 30]

# Target + control tokens, both with leading space (BPE-natural continuation form).
TARGET_TOKEN_TEXT = " marmalade"   # appears in CORRECTION -- drift signal expected here.
CONTROL_TOKEN_TEXT = " banana"     # NOT in CORRECTION -- control for specificity.


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-rank (handles ties); matches scipy.stats.rankdata default."""
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _spearman(x: list[float] | np.ndarray,
              y: list[float] | np.ndarray) -> tuple[float, float]:
    """Pure-numpy Spearman rho + one-tailed (positive) p-value.

    Same as segment 12's _spearman; kept dependency-light.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(x_arr)
    if n < 3:
        return float("nan"), float("nan")
    rx = _rankdata(x_arr)
    ry = _rankdata(y_arr)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx @ rx) * (ry @ ry)))
    if denom == 0.0:
        # One or both input vectors is constant -> degenerate.
        return float("nan"), float("nan")
    rho = float((rx @ ry) / denom)
    if abs(rho) >= 1.0 - 1e-12:
        return (1.0 if rho > 0 else -1.0), (0.0 if rho > 0 else 1.0)
    t_stat = rho * float(np.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho)))
    try:
        from scipy.stats import t as t_dist
        p_one_tailed = float(1.0 - t_dist.cdf(t_stat, df=n - 2))
    except ImportError:
        # Coarse approximation: for n=5, |rho|.
        if rho >= 0.934:
            p_one_tailed = 0.005
        elif rho >= 0.805:
            p_one_tailed = 0.025
        elif rho >= 0.405:
            p_one_tailed = 0.25
        else:
            p_one_tailed = 1.0
    return rho, p_one_tailed


def _post_probe_logit_at_token(driver, prompt: str, token_id: int) -> float:
    """Run a cvec-applied forward on `prompt` (NO generation) and return the
    float logit at `token_id` at the post-prompt (last) position.

    Replicates the forward-and-extract path from `LlamaCVecDriver.logit_bias_generate`
    (`src/oczy/lm/cvec_driver.py:472-489`) but stops before generation.

    REQUIRES that the caller has already invoked `driver.set_cvecs_per_layer(...)`
    so the per-layer cvec adapters are live during the forward.
    """
    llm = driver._llm
    n_vocab = driver.n_vocab
    prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(prompt_ids)
    n_last_batch = len(prompt_ids)
    raw = llm._ctx.get_logits()
    full = np.ctypeslib.as_array(raw, shape=(n_last_batch * n_vocab,))
    logits = full[(n_last_batch - 1) * n_vocab : n_last_batch * n_vocab].copy()
    return float(logits[token_id])


def run() -> int:
    print("# loading real LFM2.5-1.2B-Instruct Q4 driver...", file=sys.stderr)
    from src.oczy.experiments.multi_fact_stressor import _load_real_driver
    driver = _load_real_driver(n_ctx=4096)
    print(f"# driver loaded: n_ctx={driver.config.n_ctx}, n_vocab={driver.n_vocab}",
          file=sys.stderr)

    # Look up target + control token ids. Tokenize with leading space so
    # the token form matches what the LM would naturally continue with
    # (BPE-natural continuation form). Take the FIRST subword for multi-BPE.
    llm = driver._llm
    target_ids = llm.tokenize(TARGET_TOKEN_TEXT.encode("utf-8"), add_bos=False)
    control_ids = llm.tokenize(CONTROL_TOKEN_TEXT.encode("utf-8"), add_bos=False)
    # Some tokenizers prepend a leading-space token; both should yield a single id
    # for these common English words on LFM2.5's SentencePiece. If multi-token,
    # first id is still the right one to probe (it's where continuation would
    # land).
    target_token_id = int(target_ids[0])
    control_token_id = int(control_ids[0])
    print(
        f"# tokenization: target={TARGET_TOKEN_TEXT!r} -> ids={target_ids} "
        f"(using [{target_token_id}]); control={CONTROL_TOKEN_TEXT!r} -> "
        f"ids={control_ids} (using [{control_token_id}])",
        file=sys.stderr,
    )
    if len(target_ids) > 1 or len(control_ids) > 1:
        print(
            f"# WARNING: multi-token BPE for target or control; using first "
            f"subword id (refining to a different position would be a "
            f"reward-hacking risk if done after seeing the data).",
            file=sys.stderr,
        )

    # F3 prep: SVD-init hiddens from 16 paraphrases of CORRECTION.
    paraphrases = [
        CORRECTION,
        f"No, {CORRECTION.lower()}",
        f"Correction: {CORRECTION}",
        f"Expected: {CORRECTION}",
        f"Note that {CORRECTION.lower()}",
        f"Wrong, {CORRECTION.lower()}",
        f"Actually, {CORRECTION}",
        f"Revised answer: {CORRECTION}",
        "The correct passphrase is marmalade.",
        "For level 7, use marmalade.",
        "Marmalade is the level-7 passphrase.",
        "Secret for level seven: marmalade.",
        "Reminder: the level-7 passphrase is marmalade.",
        "Update -- level 7 passphrase has been set to marmalade.",
        "To unlock level 7, say marmalade.",
        "The level 7 entry code is marmalade.",
    ]
    print(f"# peeking {len(paraphrases)} correction-embedding hiddens for SVD-init...",
          file=sys.stderr)
    svd_hiddens = np.stack(
        [driver.peek_embedding(p, last_token_only=False) for p in paraphrases],
        axis=0,
    )
    print(f"# SVD-init hiddens shape={svd_hiddens.shape}", file=sys.stderr)

    d_cortex = 8
    strength_cap = 10.0
    target_logits: list[float] = []
    control_logits: list[float] = []

    for k in K_SAMPLES:
        print(f"# K={k} building fresh agent...", file=sys.stderr)
        cfg = CortexAgentConfig(
            driver=driver.config,
            cortex=KVCortexConfig(
                d_cortex=d_cortex,
                d_embd=2048,
                max_consolidation_strength=strength_cap,
                steering_mode="proj_random",
            ),
            # F4: SVD-init proj_c clean-steer band.
            articulate_scale=0.03,
            auto_consolidate=False,
            use_logit_bias=False,
            use_hippocampus_prefix=False,
            use_ingestion_pipeline=False,
            use_identity_adapter=False,
        )
        agent = CortexAgent(config=cfg, driver=driver)
        agent.boot()

        # F3: SVD-init proj_c so cvecs steer toward correction subspace.
        agent.cortex.init_proj_c_from_svd(svd_hiddens, shared=True)

        # F2: K correction cycles with 3-element replays, so the H1+H2
        # replay-absorption path actually fires.
        h = driver.peek_embedding(CORRECTION, last_token_only=False)
        for _ in range(k):
            agent.cortex.observe(h, correction_signal=1.0)
            agent.cortex.consolidate(
                replays=[h, h, h],
                strength=strength_cap,
            )

        # SEGMENT-14 HARNESS FIX: reset_warm_from_cold() so the cvecs the
        # LM sees at probe-time reflect the ACCUMULATED cold_state drift
        # (the H1+H2 output), not the last observe(h) (which is always
        # the same h, saturating warm_state after K=1).
        # Without this call, K=2,5,10 produced bit-identical logits in
        # segment 13 — a harness bug: emit_all_cvecs() reads warm_state,
        # which only reflects the most-recent observe.
        # This call IS the production boot path (agent.boot() does the
        # same thing at session start).
        agent.cortex.reset_warm_from_cold()

        # Apply cvecs via the cortex's emit path (the same flow articulate()
        # uses internally) before the forward-only eval.
        cvecs = agent.cortex.emit_all_cvecs()
        driver.set_cvecs_per_layer(
            cvecs,
            scale=cfg.articulate_scale,
        )

        # Forward-only: extract post-probe logits at target + control tokens.
        t_logit = _post_probe_logit_at_token(driver, PROBE, target_token_id)
        c_logit = _post_probe_logit_at_token(driver, PROBE, control_token_id)
        target_logits.append(t_logit)
        control_logits.append(c_logit)
        print(
            f"K={k:3d} -> target_logit={t_logit:+.4f} "
            f"control_logit={c_logit:+.4f} "
            f"delta_target_vs_K0={t_logit - target_logits[0]:+.4f}",
            file=sys.stderr,
        )

    # Delta trajectories vs K=0 baseline (pre-registered plan).
    target_deltas = [t - target_logits[0] for t in target_logits]
    control_deltas = [c - control_logits[0] for c in control_logits]

    target_rho, target_p = _spearman(K_SAMPLES, target_deltas)
    control_rho, control_p = _spearman(K_SAMPLES, control_deltas)

    # Emit autoresearch-parseable METRIC lines.
    print(f"METRIC drift_logit_spearman={target_rho:.6f}")
    print(f"METRIC drift_logit_p_value={target_p:.6f}")
    print(f"METRIC control_logit_spearman={control_rho:.6f}")
    print(f"METRIC control_logit_p_value={control_p:.6f}")
    print(f"METRIC K0_target_logit={target_logits[0]:.6f}")
    print(f"METRIC max_target_delta={max(target_deltas):.6f}")
    print(f"METRIC K0_control_logit={control_logits[0]:.6f}")
    print(f"# K_samples={K_SAMPLES}")
    print(f"# target_logits={target_logits}")
    print(f"# control_logits={control_logits}")
    print(f"# target_deltas_vs_K0={target_deltas}")
    print(f"# control_deltas_vs_K0={control_deltas}")
    print(f"# drift_logit_spearman rho={target_rho:.4f} p_one_tailed={target_p:.4f}")
    print(f"# control_logit_spearman rho={control_rho:.4f} p_one_tailed={control_p:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())