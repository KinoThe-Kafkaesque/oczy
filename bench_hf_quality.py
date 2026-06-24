#!/usr/bin/env python3
"""Side-by-side output comparison across the three loaded HF models.

Runs a small but diverse prompt suite against each model with greedy
decoding and prints the responses so they can be inspected directly.

The point is not to claim a benchmark score but to expose where each
model actually fails so the user can judge which one is "smarter" on
the tasks they care about.

Prompt suite (~7 items covering different capability axes):
  1  fact             : Single factual lookup
  2  math             : Multi-step arithmetic
  3  reasoning        : Classic syllogism / set-logic puzzle
  4  follow-fmt       : Strict output-format adherence
  5  code             : Small code generation
  6  instruction-faith: Reject the false premise in the prompt
  7  paraphrase       : Concise rephrase
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = [
    ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
    ("LFM2.5-350M", "LiquidAI/LFM2.5-350M"),
    ("LFM2.5-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct"),
]

# Each prompt is (label, kind, prompt, max_new_tokens, expected_short).
# `expected_short` is a phrase that a correct answer should contain or
# closely paraphrase; we show whether it appears verbatim in the output
# as a quick heuristic signal -- but we print the full output so the
# human can judge beyond the substring check.
PROMPTS: list[tuple[str, str, str, int, str]] = [
    (
        "fact_capital_australia",
        "factual",
        "What is the capital of Australia? Answer with just the city name.",
        20,
        "canberra",
    ),
    (
        "math_3step",
        "arithmetic",
        "A shop sells apples at 3 for $1. You buy 12 apples with a $10 bill. "
        "How much change do you get? Show your steps.",
        200,
        "$6",
    ),
    (
        "reasoning_syllogism",
        "reasoning",
        "All zips are zaps. Some zaps are zongs. Therefore, is it definitely true, "
        "definitely false, or undetermined, that some zips are zongs? "
        "Answer with one word and a one-sentence justification.",
        100,
        "undetermined",
    ),
    (
        "format_strict_json",
        "instruction-following",
        "Return ONLY a JSON object with keys 'a' and 'b' where a=2 and b=3. "
        "No prose, no markdown, no explanation.",
        80,
        '"a": 2',
    ),
    (
        "code_reverse_list",
        "code",
        "Write a Python function that reverses a list in place (without "
        "calling list.reverse() or slicing). Return only the function body.",
        250,
        "def ",
    ),
    (
        "instruction_reject_premise",
        "instruction-faith",
        "Why did the Apollo 11 astronauts take selfies with penguins on Mars? "
        "If the question contains a false premise, say so.",
        150,
        "false",
    ),
    (
        "paraphrase_one_sentence",
        "concision",
        "In one sentence, explain what an operating system does.",
        80,
        "manages",
    ),
]


@dataclass
class ModelContext:
    label: str
    repo_id: str
    tok: AutoTokenizer
    model: AutoModelForCausalLM
    is_chat: bool  # Does this model expose a chat template?


def load_all(thread_count: int) -> list[ModelContext]:
    torch.set_num_threads(thread_count)
    out: list[ModelContext] = []
    for label, repo in MODELS:
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        is_chat = bool(getattr(tok, "chat_template", None))
        dt = time.perf_counter() - t0
        print(f"  loaded {label} ({repo}) in {dt:.1f}s; chat_template={is_chat}")
        out.append(ModelContext(label, repo, tok, model, is_chat))
    return out


def render_prompt(ctx: ModelContext, user: str) -> str:
    """Apply chat template if available, else fall back to raw user
    text followed by a newline (works acceptably for base models)."""
    if ctx.is_chat:
        try:
            rendered = ctx.tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered
        except Exception:
            return user + "\n"
    return user + "\n"


def run_one(ctx: ModelContext, user: str, max_new: int) -> tuple[str, float]:
    """Generate one response.  Returns (decoded_text, elapsed_seconds)."""
    text = render_prompt(ctx, user)
    inputs = ctx.tok(text, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = ctx.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=ctx.tok.eos_token_id,
            use_cache=True,
        )
    # If the rendered prompt included an assistant turn marker we want to
    # avoid printing it back.  The simplest way is to decode the whole
    # output and strip the prompt prefix -- skip the first input_ids
    # worth of tokens.
    n_in = inputs.input_ids.shape[1]
    new_ids = out[0, n_in:]
    elapsed = time.perf_counter() - t0
    return ctx.tok.decode(new_ids, skip_special_tokens=True), elapsed


def contains_expected(text: str, expected: str) -> bool:
    if not expected:
        return False
    return expected.lower() in text.lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new-override", type=int, default=None,
                    help="If set, override per-prompt max_new with this value.")
    args = ap.parse_args()

    print(f"threads={args.threads}, dtype=float32")
    print("loading models...")
    ctxs = load_all(args.threads)

    # Scores: pass/fail on the substring heuristic.
    score: dict[str, dict[str, bool]] = {c.label: {} for c in ctxs}

    for idx, (label, kind, user, max_new, expected) in enumerate(PROMPTS, 1):
        if args.max_new_override is not None:
            max_new = args.max_new_override
        print(f"\n{'=' * 78}")
        print(f"PROMPT {idx} [{kind}]: {label}")
        print(f"  {user}")
        print(f"  (contains-heuristic target: {expected!r})")
        print(f"{'-' * 78}")
        for ctx in ctxs:
            try:
                text, elapsed = run_one(ctx, user, max_new)
            except Exception as e:
                print(f"\n[{ctx.label}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                score[ctx.label][label] = False
                continue
            hits = contains_expected(text, expected)
            score[ctx.label][label] = hits
            # Trim trailing whitespace; collapse internal blank lines.
            display = "\n".join(
                line.rstrip() for line in text.rstrip().splitlines()
            )
            print(f"\n[{ctx.label}] ({elapsed:.1f}s, hit={hits})")
            # Indent for readability.
            for line in display.splitlines():
                print("    " + line)

    # Summary table.
    print("\n\n" + "=" * 78)
    print("SUBSTRING-PASS SUMMARY")
    print("=" * 78)
    headers = ["model"] + [lbl.split("_")[0] for lbl, *_ in PROMPTS] + ["score"]
    print("  ".join(h.ljust(18) for h in headers))
    print("-" * (18 * len(headers)))
    for ctx in ctxs:
        cells = [ctx.label.ljust(18)]
        n_pass = 0
        for lbl, *_ in PROMPTS:
            ok = score[ctx.label].get(lbl, False)
            cells.append(("Y" if ok else ".").ljust(18))
            if ok:
                n_pass += 1
        cells.append(f"{n_pass}/{len(PROMPTS)}".ljust(18))
        print("  ".join(cells))

    # Free memory before exit.
    for ctx in ctxs:
        del ctx.model, ctx.tok
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())