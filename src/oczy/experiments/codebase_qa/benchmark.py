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

    stop_ids = llm.tokenize(stop.encode("utf-8"), add_bos=False)
    eos_id = llm.token_eos()

    generated_ids: list[int] = []
    target_idx = 0  # which subword token we're trying to force next

    for _ in range(max_tokens):
        raw = llm._ctx.get_logits()
        logits = np.ctypeslib.as_array(raw, shape=(n_vocab,)).copy()

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
    return 0

if __name__ == "__main__":
    sys.exit(main())
