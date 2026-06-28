#!/usr/bin/env python3
"""Deterministic codebase-QA benchmark harness.

Measures code_qa_accuracy with and without retrieved repository facts injected
into the prompt. Retrieval uses the KnowledgeStore's keyword overlap scorer so
no embeddings are required and the harness remains deterministic and fast.
The LFM2.5-1.2B-Instruct Q4_K_M GGUF is still used for generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from oczy.experiments.codebase_qa.cortex_agent_recall import evaluate
from oczy.experiments.codebase_qa.knowledge_store import KnowledgeStore
from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.lm import CVecDriverConfig, LlamaCVecDriver
from plastic_cortex.kv_cortex import KVCortexConfig

_FACTS_PATH = Path(__file__).with_name("facts.json")
_QUESTIONS_PATH = Path(__file__).with_name("questions.json")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_prompt(question: str, context: str = "") -> str:
    body = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    if context:
        return f"{context}{body}"
    return body




def _score(expected: str | list[str], answer: str) -> int:
    answer = answer.lower()
    if isinstance(expected, str):
        expected = [expected]
    return 1 if any(exp.lower() in answer for exp in expected) else 0

def _run_consolidation_uptake(driver: LlamaCVecDriver, use_hippo_prefix: bool = False) -> dict[str, Any]:
    """Probe boot-persistent semantic consolidation via SVD-initialised proj_c.

    A fresh CortexAgent is corrected several times toward a target answer.
    The correction hidden vectors are used to SVD-initialise ``proj_c`` so
    the steering direction is aligned to real corrections and survives
    cold boot. Consolidation then moves the warm update into cold_state.
    We record answers before, immediately after, and after reboot.
    """
    import numpy as np

    # Use a novel fact with an unpredictable answer so the LM's priors
    # cannot override the hippocampus-derived prefix.
    probe = "What is the secret passphrase for level 7?"
    semantic_expected = ["marmalade"]  # exact target token
    domain_expected = ["secret", "passphrase", "marmalade", "level"]  # related domain
    correction = "The secret passphrase for level 7 is marmalade."
    prompt = _build_prompt(probe)

    # Build a long filler turn embedding the fact, like the multi-fact stressor.
    # This fills the KV cache with irrelevant text so the hippocampus-derived
    # prefix is the most salient content during generation.
    import random
    random.seed(42)
    def _make_long_turn(total_length_tokens: int = 2048) -> str:
        filler_words = ["neutral"] * 500 + ["context"] * 200 + ["data"] * 200
        random.shuffle(filler_words)
        tokens = []
        while len(" ".join(tokens).split()) < total_length_tokens * 0.95:
            tokens.append(random.choice(filler_words))
        insertion_point = len(tokens) // 2
        tokens.insert(insertion_point, correction)
        tokens.insert(insertion_point + 30, " ".join(
            random.choice(filler_words) for _ in range(20)
        ))
        return " ".join(tokens)

    long_turn = _make_long_turn(total_length_tokens=2048)

    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random"),
        articulate_scale=0.03,
        auto_consolidate=True,
        use_hippocampus_prefix=use_hippo_prefix,
        use_ingestion_pipeline=True,
        ingestion={
            "chunker": "fixed-window",
            "chunker_window_tokens": 64,
            "chunker_overlap_tokens": 8,
            "salience": "lexical-novelty",
            "embedder": "same-lm",
            "aggregator": "stats",
        },
    )
    agent = CortexAgent(config=cfg, knowledge_store=None, driver=driver)
    agent.boot()
    # When using hippocampus prefix, disable cvec steering to prevent
    # the interference documented in experiments_logs/2026-06-25_prefix_steering_poc.md
    # (prefix+cvec degrades both; prefix-only achieves exact-token recall).
    apply_steering = not use_hippo_prefix

    prefix_targets = ["marmalade"] if use_hippo_prefix else None
    pre_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=apply_steering,
        recall_query=probe,
        prefix_targets=prefix_targets,
    ).strip()
    pre_score = _score(semantic_expected, pre_answer)
    pre_domain_score = _score(domain_expected, pre_answer)
    pre_normalised = pre_answer.lower()

    # Perceive the long filler turn.  The ingestion pipeline chunks and
    # the salience filter ensures only novelty-containing chunks reach the
    # hippocampus.  We collect the hidden vector for SVD init.
    agent.perceive(long_turn)
    agent.metabolize()
    correction_hidden = (
        agent._last_hidden.copy() if agent._last_hidden is not None else None
    )

    auto_fired = False
    if agent.should_consolidate():
        auto_fired = True

    if correction_hidden is not None:
        try:
            # Rank-1 SVD: set proj_c to the correction hidden direction.
            agent.cortex.init_proj_c_from_svd(
                np.vstack([correction_hidden])
            )
        except Exception as exc:  # noqa: BLE001
            print(f"SVD init failed: {exc}")

    if not auto_fired:
        agent.consolidate()

    post_warm_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=apply_steering,
        recall_query=probe,
        prefix_targets=prefix_targets,
    ).strip()
    post_warm_score = _score(semantic_expected, post_warm_answer)
    post_warm_domain_score = _score(domain_expected, post_warm_answer)
    output_shift = 1 if post_warm_answer.lower() != pre_normalised else 0

    # Reboot from cold so the post answer comes from boot-persistent state.
    agent.boot()
    post_cold_answer = agent.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=apply_steering,
        recall_query=probe,
        prefix_targets=prefix_targets,
    ).strip()
    post_cold_score = _score(semantic_expected, post_cold_answer)
    post_cold_domain_score = _score(domain_expected, post_cold_answer)
    cold_output_shift = 1 if post_cold_answer.lower() != pre_normalised else 0

    print(
        f"Consolidation uptake probe: {probe}\n"
        f"  pre:       {pre_answer!r} | semantic: {pre_score} | domain: {pre_domain_score}\n"
        f"  post_warm: {post_warm_answer!r} | semantic: {post_warm_score} | domain: {post_warm_domain_score} | shift: {output_shift}\n"
        f"  post_cold: {post_cold_answer!r} | semantic: {post_cold_score} | domain: {post_cold_domain_score} | cold_shift: {cold_output_shift}\n"
        f"  auto_consolidated: {auto_fired}"
    )

    return {
        "pre_score": float(pre_score),
        "post_warm_score": float(post_warm_score),
        "post_cold_score": float(post_cold_score),
        "pre_domain_score": float(pre_domain_score),
        "post_warm_domain_score": float(post_warm_domain_score),
        "post_cold_domain_score": float(post_cold_domain_score),
        "output_shift": float(output_shift),
        "cold_output_shift": float(cold_output_shift),
        "delta": float(post_cold_score - pre_score),
        "auto_fired": 1.0 if auto_fired else 0.0,
        "pre_answer": pre_answer,
        "post_warm_answer": post_warm_answer,
        "post_cold_answer": post_cold_answer,
    }


def _logit_bias_generate(
    driver: LlamaCVecDriver,
    prompt: str,
    target_token_ids: list[int],
    bias: float = 10.0,
    max_tokens: int = 16,
    stop: str = "\n",
) -> str:
    """Generation with direct logit biasing on target tokens.

    This is the 6th mechanism for exact-token recall. Unlike all cvec methods
    (which perturb the residual stream and contaminate the KV cache), logit
    biasing adds a constant to the target token's logit AFTER the forward pass.
    The residual stream — and therefore the KV cache entry for the generated
    token — stays clean.  Subsequent tokens attend to un-contaminated context.

    For multi-token targets (BPE subwords), the bias is applied sequentially:
    only the next expected subword token is biased at each step, in order.
    """
    import numpy as np

    llm = driver._llm
    n_vocab = driver.n_vocab

    prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(prompt_ids)
    n_prompt = len(prompt_ids)

    stop_ids = llm.tokenize(stop.encode("utf-8"), add_bos=False)
    eos_id = llm.token_eos()

    generated_ids: list[int] = []
    target_idx = 0  # which subword token we're trying to force next
    # Track the number of tokens in the last eval batch so we can index
    # get_logits() correctly.  The function returns a flat array of
    # (n_batch * n_vocab) floats; we need the LAST position's logits.
    n_last_batch = n_prompt

    for _ in range(max_tokens):
        raw = llm._ctx.get_logits()
        full = np.ctypeslib.as_array(raw, shape=(n_last_batch * n_vocab,))
        logits = full[(n_last_batch - 1) * n_vocab : n_last_batch * n_vocab].copy()

        # Bias the next expected target subword token.
        if target_idx < len(target_token_ids):
            tid = target_token_ids[target_idx]
            logits[tid] += bias

        next_token = int(np.argmax(logits))
        generated_ids.append(next_token)

        # Advance target index if we matched the expected subword.
        if target_idx < len(target_token_ids) and next_token == target_token_ids[target_idx]:
            target_idx += 1
        elif target_idx < len(target_token_ids):
            # Didn't match the target — stop forcing, let the LM continue freely.
            target_idx = len(target_token_ids)  # stop biasing

        if next_token == eos_id:
            break
        if stop_ids and generated_ids[-len(stop_ids):] == stop_ids:
            break

        llm.eval([next_token])
        n_last_batch = 1

    return llm.detokenize(generated_ids).decode("utf-8", errors="replace")


def _run_logit_bias_disambiguation_uptake(driver: LlamaCVecDriver) -> dict[str, Any]:
    """Probe direct logit biasing for exact-token recall without prefix.

    This is the 6th mechanism tested. Unlike all cvec methods (which perturb
    the residual stream and contaminate the KV cache), logit biasing adds a
    constant to the target token's logit AFTER the forward pass.  The KV cache
    entry for the generated token is written from the clean residual stream,
    so subsequent tokens attend to un-contaminated context.

    Probe: "What does 'profile' mean in this codebase?"
    LM's default sense: "user profile" / "set of data".  Target: "vertical".
    " vertical" is a single BPE token (id=12825) on LFM2.5.

    We also test the consolidation probe: "What is the secret passphrase for
    level 7?" with target "marmalade" (3 subword tokens: " marm"+"al"+"ade").
    This tests multi-token logit biasing.
    """
    llm = driver._llm

    results: dict[str, Any] = {}

    # --- Probe 1: disambiguation (single-token target) ---
    probe1 = "What does 'profile' mean in this codebase?"
    semantic_expected_1 = ["vertical"]
    domain_expected_1 = ["vertical", "sector", "industry", "market", "domain", "segment"]
    prompt1 = _build_prompt(probe1)

    driver.clear_cvec()
    pre1 = driver.generate(prompt1, max_tokens=16, temperature=0.0, stop=["\n"]).strip()
    pre1_score = _score(semantic_expected_1, pre1)
    pre1_domain = _score(domain_expected_1, pre1)
    pre1_norm = pre1.lower()

    target1_ids = llm.tokenize(b" vertical", add_bos=False)
    print(f"  Probe 1 target: ' vertical' -> token ids {target1_ids}")

    bias_results_1: dict[str, Any] = {}
    for bias in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        answer = _logit_bias_generate(
            driver, prompt1, target1_ids, bias=bias, max_tokens=16, stop="\n",
        ).strip()
        sem = _score(semantic_expected_1, answer)
        dom = _score(domain_expected_1, answer)
        shift = 1 if answer.lower() != pre1_norm else 0
        print(
            f"  probe1 bias={bias:5.1f}  semantic={sem} domain={dom} "
            f"shift={shift}  answer={answer!r}"
        )
        bias_results_1[f"bias_{bias}"] = {
            "semantic": float(sem), "domain": float(dom),
            "shift": float(shift), "answer": answer,
        }

    best_bias_1 = max(
        [1.0, 3.0, 5.0, 10.0, 20.0, 50.0],
        key=lambda b: (bias_results_1[f"bias_{b}"]["semantic"],
                       bias_results_1[f"bias_{b}"]["domain"]),
    )
    best1 = bias_results_1[f"bias_{best_bias_1}"]
    results["probe1_pre_score"] = float(pre1_score)
    results["probe1_pre_domain"] = float(pre1_domain)
    results["probe1_pre_answer"] = pre1
    results["probe1_best_bias"] = best_bias_1
    results["probe1_post_score"] = best1["semantic"]
    results["probe1_post_domain"] = best1["domain"]
    results["probe1_post_answer"] = best1["answer"]
    results["probe1_delta"] = best1["semantic"] - float(pre1_score)

    # --- Probe 2: consolidation (multi-token target) ---
    probe2 = "What is the secret passphrase for level 7?"
    semantic_expected_2 = ["marmalade"]
    domain_expected_2 = ["secret", "passphrase", "marmalade", "level"]
    prompt2 = _build_prompt(probe2)

    driver.clear_cvec()
    pre2 = driver.generate(prompt2, max_tokens=16, temperature=0.0, stop=["\n"]).strip()
    pre2_score = _score(semantic_expected_2, pre2)
    pre2_domain = _score(domain_expected_2, pre2)
    pre2_norm = pre2.lower()

    target2_ids = llm.tokenize(b" marmalade", add_bos=False)
    print(f"  Probe 2 target: ' marmalade' -> token ids {target2_ids}")

    bias_results_2: dict[str, Any] = {}
    for bias in [1.0, 5.0, 10.0, 20.0, 50.0]:
        answer = _logit_bias_generate(
            driver, prompt2, target2_ids, bias=bias, max_tokens=16, stop="\n",
        ).strip()
        sem = _score(semantic_expected_2, answer)
        dom = _score(domain_expected_2, answer)
        shift = 1 if answer.lower() != pre2_norm else 0
        print(
            f"  probe2 bias={bias:5.1f}  semantic={sem} domain={dom} "
            f"shift={shift}  answer={answer!r}"
        )
        bias_results_2[f"bias_{bias}"] = {
            "semantic": float(sem), "domain": float(dom),
            "shift": float(shift), "answer": answer,
        }

    best_bias_2 = max(
        [1.0, 5.0, 10.0, 20.0, 50.0],
        key=lambda b: (bias_results_2[f"bias_{b}"]["semantic"],
                       bias_results_2[f"bias_{b}"]["domain"]),
    )
    best2 = bias_results_2[f"bias_{best_bias_2}"]
    results["probe2_pre_score"] = float(pre2_score)
    results["probe2_pre_domain"] = float(pre2_domain)
    results["probe2_pre_answer"] = pre2
    results["probe2_best_bias"] = best_bias_2
    results["probe2_post_score"] = best2["semantic"]
    results["probe2_post_domain"] = best2["domain"]
    results["probe2_post_answer"] = best2["answer"]
    results["probe2_delta"] = best2["semantic"] - float(pre2_score)

    # Headline metrics: use probe1 (the disambiguation probe that all 5
    # cvec methods failed on).
    results["pre_score"] = results["probe1_pre_score"]
    results["post_warm_score"] = results["probe1_post_score"]
    results["post_warm_domain_score"] = results["probe1_post_domain"]
    results["delta"] = results["probe1_delta"]

    print(
        f"Logit bias disambiguation uptake probe:\n"
        f"  probe1 pre:  {pre1!r} | semantic: {pre1_score} | domain: {pre1_domain}\n"
        f"  probe1 post: {results['probe1_post_answer']!r} | semantic: {results['probe1_post_score']} "
        f"| domain: {results['probe1_post_domain']} | best_bias: {best_bias_1}\n"
        f"  probe2 pre:  {pre2!r} | semantic: {pre2_score} | domain: {pre2_domain}\n"
        f"  probe2 post: {results['probe2_post_answer']!r} | semantic: {results['probe2_post_score']} "
        f"| domain: {results['probe2_post_domain']} | best_bias: {best_bias_2}"
    )

    return results

def _run_composition_probe(driver: LlamaCVecDriver) -> dict[str, Any]:
    """Probe cvec + logit biasing composition.

    This is the final unification test.  Cvec operates in residual stream
    space (during forward pass), logit biasing operates in logit space
    (post-forward).  They're on different surfaces, so they should coexist:
    cvec shifts the domain/posture of the output, logit biasing forces the
    exact target token.

    If this works, we have the unified steering mechanism the session goal
    was looking for: cvec (domain shift) + logit biasing (exact-token recall)
    compose because they operate on different surfaces.

    Probe: "What does 'profile' mean in this codebase?"
    Target: "vertical" (exact token via logit biasing)
    Domain: business/vertical vocabulary (via contrastive cvec)
    """
    import numpy as np

    llm = driver._llm
    probe = "What does 'profile' mean in this codebase?"
    semantic_expected = ["vertical"]
    domain_expected = ["vertical", "sector", "industry", "market", "domain", "segment"]
    prompt = _build_prompt(probe)

    # Baseline: no cvec, no logit bias.
    driver.clear_cvec()
    baseline = driver.generate(prompt, max_tokens=16, temperature=0.0, stop=["\n"]).strip()
    baseline_sem = _score(semantic_expected, baseline)
    baseline_dom = _score(domain_expected, baseline)

    # Logit biasing only (no cvec) — the proven baseline from run #137.
    driver.clear_cvec()
    target_ids = llm.tokenize(b" vertical", add_bos=False)
    bias_only = _logit_bias_generate(
        driver, prompt, target_ids, bias=20.0, max_tokens=16, stop="\n",
    ).strip()
    bias_only_sem = _score(semantic_expected, bias_only)
    bias_only_dom = _score(domain_expected, bias_only)

    # Cvec only (no logit bias) — the contrastive cvec that shifts register
    # but can't force the exact token.
    contrastive_defaults = [
        "sector", "industry", "market", "domain",
        "segment", "category", "field", "area",
    ]
    target_emb = driver.peek_embedding(" vertical", last_token_only=True)
    deltas = []
    for default in contrastive_defaults:
        default_emb = driver.peek_embedding(f" {default}", last_token_only=True)
        deltas.append(target_emb - default_emb)
    centered = np.vstack(deltas) - np.vstack(deltas).mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    contrast_vec = Vt[0].astype(np.float32)
    uniform_vecs = [contrast_vec for _ in range(driver.n_layers)]

    driver.set_cvecs_per_layer(uniform_vecs, scale=1.0)
    cvec_only = driver.generate(prompt, max_tokens=16, temperature=0.0, stop=["\n"]).strip()
    cvec_only_sem = _score(semantic_expected, cvec_only)
    cvec_only_dom = _score(domain_expected, cvec_only)
    driver.clear_cvec()

    # COMPOSITION: cvec + logit biasing together.
    # Cvec is set on the driver (applied during forward pass via residual
    # stream).  Logit biasing is applied post-forward in _logit_bias_generate.
    # They operate on different surfaces and should not interfere.
    results: dict[str, Any] = {
        "baseline_answer": baseline,
        "baseline_semantic": float(baseline_sem),
        "baseline_domain": float(baseline_dom),
        "bias_only_answer": bias_only,
        "bias_only_semantic": float(bias_only_sem),
        "bias_only_domain": float(bias_only_dom),
        "cvec_only_answer": cvec_only,
        "cvec_only_semantic": float(cvec_only_sem),
        "cvec_only_domain": float(cvec_only_dom),
    }

    # Sweep cvec scale × logit bias combinations.  Cvec shifts the
    # forward-pass logits, so the bias threshold from the no-cvec probe
    # (20.0) may not apply — sweep both dimensions.
    bias_sweep = [10.0, 20.0, 50.0, 100.0]
    for cvec_scale in [0.01, 0.03, 0.1, 1.0]:
        driver.set_cvecs_per_layer(uniform_vecs, scale=cvec_scale)
        for bias_val in bias_sweep:
            answer = _logit_bias_generate(
                driver, prompt, target_ids, bias=bias_val, max_tokens=16, stop="\n",
            ).strip()
            sem = _score(semantic_expected, answer)
            dom = _score(domain_expected, answer)
            coherent = 1 if len(answer) > 5 and not any(c in answer for c in "ÀÁÂÃÄÅÆÇÈÉÊË") else 0
            print(
                f"  composition cvec={cvec_scale:5.2f} bias={bias_val:6.1f}  "
                f"sem={sem} dom={dom} coh={coherent}  answer={answer!r}"
            )
            key = f"comp_cvec_{cvec_scale}_bias_{bias_val}"
            results[f"{key}_semantic"] = float(sem)
            results[f"{key}_domain"] = float(dom)
            results[f"{key}_coherent"] = float(coherent)
            results[f"{key}_answer"] = answer


    driver.clear_cvec()

    # Headline: best composition result across the 2D sweep.
    combos = [(s, b) for s in [0.01, 0.03, 0.1, 1.0] for b in bias_sweep]
    best_scale, best_bias = max(
        combos,
        key=lambda sb: (results[f"comp_cvec_{sb[0]}_bias_{sb[1]}_semantic"],
                       results[f"comp_cvec_{sb[0]}_bias_{sb[1]}_domain"],
                       results[f"comp_cvec_{sb[0]}_bias_{sb[1]}_coherent"]),
    )
    bk = f"comp_cvec_{best_scale}_bias_{best_bias}"
    results["composition_semantic"] = results[f"{bk}_semantic"]
    results["composition_domain"] = results[f"{bk}_domain"]
    results["composition_coherent"] = results[f"{bk}_coherent"]
    results["composition_answer"] = results[f"{bk}_answer"]
    results["best_cvec_scale"] = best_scale
    results["best_bias"] = best_bias
    results["delta"] = results["composition_semantic"] - results["baseline_semantic"]

    print(
        f"Composition probe (cvec + logit biasing):\n"
        f"  baseline:    {baseline!r} | sem: {baseline_sem} | dom: {baseline_dom}\n"
        f"  bias_only:   {bias_only!r} | sem: {bias_only_sem} | dom: {bias_only_dom}\n"
        f"  cvec_only:   {cvec_only!r} | sem: {cvec_only_sem} | dom: {cvec_only_dom}\n"
        f"  composition: {results['composition_answer']!r} | sem: {results['composition_semantic']} "
        f"| dom: {results['composition_domain']} | coherent: {results['composition_coherent']} "
        f"| best_scale: {best_scale} | best_bias: {best_bias}"
    )

    return results

def _run_e2e_logit_bias_probe(driver: LlamaCVecDriver) -> dict[str, Any]:
    """End-to-end test of logit biasing through CortexAgent on the real model.

    Sweeps cortex configurations to find which produces coherent continuation
    when composed with logit biasing.  The 8D proj_random cortex (run #144)
    produced garbage continuation ("marmaladeiinininin...").  This sweep tests
    larger d_cortex and different steering_modes to see if the cvec can
    produce useful domain shift without degrading coherence.
    """
    probe = "What is the secret passphrase for level 7?"
    semantic_expected = ["marmalade"]
    domain_expected = ["secret", "passphrase", "marmalade", "level"]
    # Diverse corrections sharing the same target — needed for non-degenerate
    # SVD init (identical repeats → near-zero centered matrix → noise basis).
    diverse_corrections = [
        "The secret passphrase for level 7 is marmalade.",
        "Remember: level 7's passphrase is marmalade.",
        "For level 7, the passphrase is marmalade.",
    ]
    prompt = _build_prompt(probe)

    # Baseline: logit biasing only, no cvec (apply_steering=False).
    # This is config-independent — same for all cortex configs.
    driver.clear_cvec()
    cfg_base = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random"),
        articulate_scale=0.01,
        use_logit_bias=True,
        logit_bias_strength=20.0,
        use_hippocampus_prefix=False,
        use_ingestion_pipeline=False,
    )
    agent_base = CortexAgent(config=cfg_base, knowledge_store=None, driver=driver)
    agent_base.boot()
    driver.clear_cvec()
    pre_answer = agent_base.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=False,
        prefix_targets=["marmalade"],
    ).strip()
    pre_score = _score(semantic_expected, pre_answer)
    pre_domain = _score(domain_expected, pre_answer)

    # Sweep cortex configs for the post-correction (cvec + logit_bias) test.
    configs = [
        # Controls (from run #146): non-SVD at coherent scale.
        {"d_cortex": 8, "steering_mode": "proj_random", "scale": 0.001, "svd_init": False},
        {"d_cortex": 64, "steering_mode": "raw_hidden", "scale": 0.001, "svd_init": False},
        # SVD-init with diverse hiddens: d_cortex <= N=3 so SVD is non-degenerate.
        {"d_cortex": 2, "steering_mode": "proj_random", "scale": 0.001, "svd_init": True},
        {"d_cortex": 3, "steering_mode": "proj_random", "scale": 0.001, "svd_init": True},
        {"d_cortex": 2, "steering_mode": "proj_random", "scale": 0.01, "svd_init": True},
        {"d_cortex": 3, "steering_mode": "proj_random", "scale": 0.01, "svd_init": True},
        # Matched-pair non-SVD controls at scale=0.01 (same diverse-corrections
        # loop, only SVD-init differs) — isolates SVD-init as the variable.
        {"d_cortex": 2, "steering_mode": "proj_random", "scale": 0.01, "svd_init": False},
        {"d_cortex": 3, "steering_mode": "proj_random", "scale": 0.01, "svd_init": False},
        {"d_cortex": 8, "steering_mode": "proj_random", "scale": 0.01, "svd_init": False},
    ]

    results: dict[str, Any] = {
        "pre_score": float(pre_score),
        "pre_domain": float(pre_domain),
        "pre_answer": pre_answer,
    }

    import numpy as np

    for i, ccfg in enumerate(configs):
        driver.clear_cvec()
        cfg = CortexAgentConfig(
            cortex=KVCortexConfig(
                d_cortex=ccfg["d_cortex"],
                steering_mode=ccfg["steering_mode"],
            ),
            articulate_scale=ccfg["scale"],
            use_logit_bias=True,
            logit_bias_strength=20.0,
            use_hippocampus_prefix=False,
            use_ingestion_pipeline=False,
        )
        agent = CortexAgent(config=cfg, knowledge_store=None, driver=driver)
        agent.boot()
        driver.clear_cvec()
        # Collect diverse correction hiddens for SVD init.
        collected_hiddens: list[np.ndarray] = []
        for corr in diverse_corrections:
            agent.perceive(corr, correction_signal=1.0)
            agent.metabolize(corr)
            if agent._last_hidden is not None:
                collected_hiddens.append(agent._last_hidden.copy())
        # SVD-init proj_c from diverse correction hiddens.
        if ccfg["svd_init"] and len(collected_hiddens) >= ccfg["d_cortex"]:
            try:
                agent.cortex.init_proj_c_from_svd(
                    np.vstack(collected_hiddens)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  SVD init failed: {exc}")
        elif ccfg["svd_init"]:
            print(f"  SVD init skipped: N={len(collected_hiddens)} < d_cortex={ccfg['d_cortex']}")
        answer = agent.articulate(
            prompt=prompt,
            max_tokens=16,
            temperature=0.0,
            apply_steering=True,
            prefix_targets=["marmalade"],
        ).strip()
        sem = _score(semantic_expected, answer)
        dom = _score(domain_expected, answer)
        # Coherent = has real words after the forced token, no repetition/garbage.
        # Detect repetitive garbage: any 2-6 char substring tiled consecutively
        # 3+ times (e.g. "inininin", "quofofquofofquofof").  Normal English doesn't tile.
        _after = answer[len("marmalade"):] if answer.startswith("marmalade") else answer
        _rep_garbage = any(
            _after[j:j+tl] == _after[j+tl:j+2*tl] == _after[j+2*tl:j+3*tl]
            for tl in range(2, 7)
            for j in range(len(_after) - tl * 3 + 1)
            if _after[j:j+tl].strip()
        )
        coherent = 1 if (len(answer) > 5 and sem == 1
                         and not any(c in answer for c in "ÀÁÂÃÄÅÆÇÈÉÊË")
                         and not _rep_garbage) else 0
        svd = "_svd" if ccfg["svd_init"] else ""
        label = f"d{ccfg['d_cortex']}_{ccfg['steering_mode']}_s{ccfg['scale']}{svd}"
        print(
            f"  e2e config {label:40s}  sem={sem} dom={dom} coh={coherent}  "
            f"answer={answer!r}"
        )
        results[f"config_{i}_label"] = label
        results[f"config_{i}_semantic"] = float(sem)
        results[f"config_{i}_domain"] = float(dom)
        results[f"config_{i}_coherent"] = float(coherent)
        results[f"config_{i}_answer"] = answer
        driver.clear_cvec()

    # Headline: best config (highest semantic + coherent).
    best_i = max(
        range(len(configs)),
        key=lambda i: (results[f"config_{i}_semantic"],
                       results[f"config_{i}_coherent"],
                       results[f"config_{i}_domain"]),
    )
    results["post_score"] = results[f"config_{best_i}_semantic"]
    results["post_domain"] = results[f"config_{best_i}_domain"]
    results["post_answer"] = results[f"config_{best_i}_answer"]
    results["best_config"] = results[f"config_{best_i}_label"]
    results["delta"] = results["post_score"] - results["pre_score"]

    # Cvec-only (no logit bias) with the best config, for comparison.
    best_ccfg = configs[best_i]
    driver.clear_cvec()
    cfg_nb = CortexAgentConfig(
        cortex=KVCortexConfig(
            d_cortex=best_ccfg["d_cortex"],
            steering_mode=best_ccfg["steering_mode"],
        ),
        articulate_scale=best_ccfg["scale"],
        use_logit_bias=False,
        use_hippocampus_prefix=False,
        use_ingestion_pipeline=False,
    )
    agent_nb = CortexAgent(config=cfg_nb, knowledge_store=None, driver=driver)
    agent_nb.boot()
    driver.clear_cvec()
    nb_hiddens: list[np.ndarray] = []
    for corr in diverse_corrections:
        agent_nb.perceive(corr, correction_signal=1.0)
        agent_nb.metabolize(corr)
        if agent_nb._last_hidden is not None:
            nb_hiddens.append(agent_nb._last_hidden.copy())
    if best_ccfg.get("svd_init") and len(nb_hiddens) >= best_ccfg["d_cortex"]:
        try:
            agent_nb.cortex.init_proj_c_from_svd(np.vstack(nb_hiddens))
        except Exception as exc:  # noqa: BLE001
            print(f"  cvec_only SVD init failed: {exc}")
    cvec_only_answer = agent_nb.articulate(
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
        apply_steering=True,
    ).strip()
    results["cvec_only_score"] = float(_score(semantic_expected, cvec_only_answer))
    results["cvec_only_domain"] = float(_score(domain_expected, cvec_only_answer))
    results["cvec_only_answer"] = cvec_only_answer
    driver.clear_cvec()

    print(
        f"E2E logit bias probe (through CortexAgent):\n"
        f"  pre (logit_bias, no cvec):  {pre_answer!r} | sem: {pre_score} | dom: {pre_domain}\n"
        f"  best config: {results['best_config']}\n"
        f"  post (logit_bias + cvec):    {results['post_answer']!r} | sem: {results['post_score']} | dom: {results['post_domain']}\n"
        f"  cvec_only (no logit_bias):   {cvec_only_answer!r} | sem: {results['cvec_only_score']} | dom: {results['cvec_only_domain']}"
    )

    return results






def _run_order_shuffle_probe(driver: LlamaCVecDriver) -> dict[str, Any]:
    """Order-shuffle stressor: does fact order change cvec continuation?

    Presents 3 facts in a SHORT turn (no filler haystack) so peek_embedding
    is dominated by the facts.  Logit biasing forces "marmalade"; the
    continuation text is the order-sensitive signal.  Uses text similarity
    (not just coherence) to detect order effects.

    Tests both observation_mode=parallel (cortex sees full turn once) and
    sequential (cortex observes per-chunk, warm_state is order-dependent).
    Hypothesis: sequential mode is more order-sensitive than parallel.
    """
    probe = "What is the secret passphrase for level 7?"
    domain_expected = ["secret", "passphrase", "marmalade", "level"]
    target_fact = "The secret passphrase for level 7 is marmalade."
    distractor_facts = [
        "The access code for sector 3 is 9421.",
        "The backup server hostname is orion-04.",
    ]
    prompt = _build_prompt(probe)

    def _make_short_turn(facts_in_order: list[str]) -> str:
        """Fixed template — only fact order varies, no filler confound."""
        parts = ["Here is some information."]
        for i, fact in enumerate(facts_in_order):
            prefix = "First," if i == 0 else "Next," if i == 1 else "Finally,"
            parts.append(f"{prefix} {fact}")
        return " ".join(parts)

    def _is_coherent(answer: str) -> int:
        if len(answer) <= 5 or not answer.lower().startswith("marmalade"):
            return 0
        if any(c in answer for c in "ÀÁÂÃÄÅÆÇÈÉÊË"):
            return 0
        after = answer[len("marmalade"):]
        rep = any(
            after[j:j+tl] == after[j+tl:j+2*tl] == after[j+2*tl:j+3*tl]
            for tl in range(2, 7)
            for j in range(len(after) - tl * 3 + 1)
            if after[j:j+tl].strip()
        )
        return 0 if rep else 1

    orderings = [
        [target_fact, distractor_facts[0], distractor_facts[1]],
        [distractor_facts[0], target_fact, distractor_facts[1]],
        [distractor_facts[0], distractor_facts[1], target_fact],
    ]
    ordering_labels = ["target_first", "target_middle", "target_last"]

    results: dict[str, Any] = {}

    for obs_mode in ("parallel", "sequential"):
        mode_answers: list[str] = []
        mode_coh: list[int] = []
        for oi, ordering in enumerate(orderings):
            driver.clear_cvec()
            turn = _make_short_turn(ordering)
            cfg = CortexAgentConfig(
                cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random"),
                articulate_scale=0.001,
                auto_consolidate=True,
                use_hippocampus_prefix=False,
                use_ingestion_pipeline=True,
                use_logit_bias=True,
                logit_bias_strength=20.0,
                ingestion={
                    "chunker": "fixed-window",
                    "chunker_window_tokens": 32,
                    "chunker_overlap_tokens": 4,
                    "salience": "lexical-novelty",
                    "embedder": "same-lm",
                    "aggregator": "stats",
                    "observation_mode": obs_mode,
                },
            )
            agent = CortexAgent(config=cfg, knowledge_store=None, driver=driver)
            agent.boot()
            driver.clear_cvec()
            agent.perceive(turn, correction_signal=1.0)
            agent.metabolize(turn)
            if not agent.should_consolidate():
                agent.consolidate()
            answer = agent.articulate(
                prompt=prompt,
                max_tokens=16,
                temperature=0.0,
                apply_steering=True,
                prefix_targets=["marmalade"],
            ).strip()
            dom = _score(domain_expected, answer)
            coh = _is_coherent(answer)
            mode_answers.append(answer)
            mode_coh.append(coh)
            label = ordering_labels[oi]
            print(
                f"  order_shuffle {obs_mode:10s} {label:14s}  "
                f"dom={dom} coh={coh}  answer={answer!r}"
            )
            results[f"{obs_mode}_{label}_domain"] = float(dom)
            results[f"{obs_mode}_{label}_coherent"] = float(coh)
            results[f"{obs_mode}_{label}_answer"] = answer
            driver.clear_cvec()
        # Sensitivity = number of distinct answers across orderings.
        # Text-level, not just coherence — coherent answers can still differ.
        distinct_answers = len(set(mode_answers))
        coh_sensitivity = len(set(mode_coh)) - 1
        results[f"{obs_mode}_answer_sensitivity"] = float(distinct_answers - 1)
        results[f"{obs_mode}_coherence_sensitivity"] = float(coh_sensitivity)
        results[f"{obs_mode}_mean_coherent"] = float(sum(mode_coh) / len(mode_coh))
        print(
            f"  order_shuffle {obs_mode}: answer_sensitivity={distinct_answers - 1} "
            f"coh_sensitivity={coh_sensitivity} "
            f"mean_coh={results[f'{obs_mode}_mean_coherent']:.2f}"
        )

    par_sens = results["parallel_answer_sensitivity"]
    seq_sens = results["sequential_answer_sensitivity"]
    results["sequential_more_sensitive"] = float(seq_sens > par_sens)
    results["delta_sensitivity"] = float(seq_sens - par_sens)

    print(
        f"Order shuffle probe:\n"
        f"  parallel answer sensitivity:   {par_sens}\n"
        f"  sequential answer sensitivity: {seq_sens}\n"
        f"  sequential more sensitive: {results['sequential_more_sensitive']}"
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Oczy codebase-QA benchmark.")
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of repository facts to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--no-recall",
        action="store_true",
        help="Only run the baseline (no retrieved-context) path.",
    )
    args = parser.parse_args()

    facts = _load_json(_FACTS_PATH)
    questions = _load_json(_QUESTIONS_PATH)

    print("Loading LlamaCVecDriver...")
    cfg = CVecDriverConfig(n_ctx=512, n_threads=4, embedding=True)
    driver = LlamaCVecDriver.load(cfg)

    if hasattr(driver._llm, "set_seed"):
        driver._llm.set_seed(42)

    # Keyword-only store for a deterministic, fast retrieval path.
    store = KnowledgeStore(embed_fn=None)
    for fact in facts:
        store.add_fact(fact["key"], fact["value"], fact.get("metadata", {}))

    print(f"Knowledge store status: {store.status()}")
    print(f"Benchmarking {len(questions)} questions...")
    print(f"Retrieving up to {args.k} facts per question.")

    baseline_scores: list[int] = []
    recall_scores: list[int] = []

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        expected = item["expected"]

        baseline_prompt = _build_prompt(question)
        baseline_answer = driver.generate(
            baseline_prompt,
            max_tokens=48,
            temperature=0.0,
            stop=["\n"],
        )
        baseline_hit = _score(expected, baseline_answer)
        baseline_scores.append(baseline_hit)

        if args.no_recall:
            print(
                f"Q{idx}: {question}\n"
                f"  expected: {expected!r}\n"
                f"  baseline: {baseline_answer.strip()!r} | score: {baseline_hit}"
            )
            continue

        context = store.format_context(question, k=args.k, min_score=0.18)
        recall_prompt = _build_prompt(question, context)
        recall_answer = driver.generate(
            recall_prompt,
            max_tokens=48,
            temperature=0.0,
            stop=["\n"],
        )
        recall_hit = _score(expected, recall_answer)
        recall_scores.append(recall_hit)

        print(
            f"Q{idx}: {question}\n"
            f"  expected: {expected!r}\n"
            f"  baseline: {baseline_answer.strip()!r} | score: {baseline_hit}\n"
            f"  recall:   {recall_answer.strip()!r} | score: {recall_hit}"
        )

    baseline_acc = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    recall_acc = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    recall_lift = recall_acc - baseline_acc

    print(f"METRIC baseline_accuracy={baseline_acc:.4f}")
    if not args.no_recall:
        print(f"METRIC code_qa_accuracy={recall_acc:.4f}")
        print(f"METRIC recall_lift={recall_lift:.4f}")
    cortex_subset_size = 24
    print(f"Running CortexAgent recall evaluation on {cortex_subset_size} questions...")
    cortex_res = evaluate(driver, facts, questions, subset_size=cortex_subset_size)
    print(f"METRIC cortex_agent_baseline_accuracy={cortex_res['baseline_accuracy']:.4f}")
    print(f"METRIC cortex_agent_recall_accuracy={cortex_res['recall_accuracy']:.4f}")
    print(f"METRIC cortex_agent_recall_lift={cortex_res['recall_lift']:.4f}")

    print("Running consolidation uptake evaluation...")
    cons_res = _run_consolidation_uptake(driver, use_hippo_prefix=True)
    print(f"METRIC consolidation_uptake_pre={cons_res['pre_score']:.4f}")
    print(f"METRIC consolidation_uptake_post_warm={cons_res['post_warm_score']:.4f}")
    print(f"METRIC consolidation_uptake_post_warm_domain={cons_res['post_warm_domain_score']:.4f}")
    print(f"METRIC consolidation_uptake_output_shift={cons_res['output_shift']:.4f}")
    print(f"METRIC consolidation_uptake_post_cold={cons_res['post_cold_score']:.4f}")
    print(f"METRIC consolidation_uptake_post_cold_domain={cons_res['post_cold_domain_score']:.4f}")
    print(f"METRIC consolidation_uptake_cold_output_shift={cons_res['cold_output_shift']:.4f}")
    print(f"METRIC consolidation_uptake_delta={cons_res['delta']:.4f}")
    print(f"METRIC consolidation_uptake_auto_fired={cons_res['auto_fired']:.4f}")
    print("Running logit bias disambiguation uptake evaluation...")
    lb_res = _run_logit_bias_disambiguation_uptake(driver)
    print(f"METRIC logit_bias_uptake_pre={lb_res['pre_score']:.4f}")
    print(f"METRIC logit_bias_uptake_post_warm={lb_res['post_warm_score']:.4f}")
    print(f"METRIC logit_bias_uptake_post_warm_domain={lb_res['post_warm_domain_score']:.4f}")
    print(f"METRIC logit_bias_uptake_delta={lb_res['delta']:.4f}")
    print("Running cvec + logit biasing composition evaluation...")
    comp_res = _run_composition_probe(driver)
    print(f"METRIC composition_baseline_semantic={comp_res['baseline_semantic']:.4f}")
    print(f"METRIC composition_bias_only_semantic={comp_res['bias_only_semantic']:.4f}")
    print(f"METRIC composition_cvec_only_semantic={comp_res['cvec_only_semantic']:.4f}")
    print(f"METRIC composition_semantic={comp_res['composition_semantic']:.4f}")
    print(f"METRIC composition_domain={comp_res['composition_domain']:.4f}")
    print(f"METRIC composition_coherent={comp_res['composition_coherent']:.4f}")
    print(f"METRIC composition_delta={comp_res['delta']:.4f}")
    print("Running end-to-end logit bias probe (through CortexAgent)...")
    e2e_res = _run_e2e_logit_bias_probe(driver)
    print(f"METRIC e2e_logit_bias_pre={e2e_res['pre_score']:.4f}")
    print(f"METRIC e2e_logit_bias_post={e2e_res['post_score']:.4f}")
    print(f"METRIC e2e_logit_bias_cvec_only={e2e_res['cvec_only_score']:.4f}")
    print(f"METRIC e2e_logit_bias_delta={e2e_res['delta']:.4f}")
    print("Running order-shuffle stressor (sequential vs parallel)...")
    os_res = _run_order_shuffle_probe(driver)
    print(f"METRIC order_shuffle_parallel_sensitivity={os_res['parallel_answer_sensitivity']:.4f}")
    print(f"METRIC order_shuffle_sequential_sensitivity={os_res['sequential_answer_sensitivity']:.4f}")
    print(f"METRIC order_shuffle_sequential_more_sensitive={os_res['sequential_more_sensitive']:.4f}")
    print(f"METRIC order_shuffle_delta_sensitivity={os_res['delta_sensitivity']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
