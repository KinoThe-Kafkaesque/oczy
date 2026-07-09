#!/usr/bin/env python3
"""Show full outputs for a single backend config (math + paraphrase + syllogism).

So we can read the actual quality, not the 80-char heuristic-truncated one
printed by bench_cross_backend.py.
"""
from __future__ import annotations

import argparse

from bench_cross_backend import (
    HFBackend, GGUFBackend, PROMPTS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["HF", "GGUF"], required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--filename", default="")
    p.add_argument("--label", default="showme")
    p.add_argument("--disc-mb", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--prompts", nargs="*",
                   default=["math_3step", "reasoning_syllogism",
                            "paraphrase_one_sentence"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.backend == "HF":
        be = HFBackend(args.repo_id, args.label)
    else:
        be = GGUFBackend(args.repo_id, args.filename, args.label,
                         args.disc_mb)
    be.load(args.threads)
    for plabel, kind, prompt, max_new, check in PROMPTS:
        if plabel not in args.prompts:
            continue
        print(f"\n{'=' * 70}")
        print(f"[{plabel}] ({kind})  max_new={max_new}")
        print(f"prompt: {prompt}")
        text, secs = be.generate(prompt, max_new)
        print(f"\n-- output ({secs:.1f}s) --")
        print(text)
        print(f"-- end -- (check function: {check.__name__}: {check(text)})")
    be.unload()


if __name__ == "__main__":
    main()