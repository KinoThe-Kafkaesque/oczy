#!/usr/bin/env python3
"""Shared agent runner — called by Harbor agent classes inside the sandbox.

Handles both Oczy (knowledge-augmented) and vanilla LFM inference.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _run_vanilla(model_path: str, instruction: str) -> str:
    """Raw LFM2.5-1.2B inference without augmentation."""
    from llama_cpp import Llama

    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
    prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
    output = llm(prompt, max_tokens=256, temperature=0.0, stop=["<|user|>", "\n\n"])
    return output["choices"][0]["text"].strip()


def _run_oczy(model_path: str, instruction: str, facts_path: str) -> str:
    """Oczy: prepend recalled facts to the prompt."""
    # Add oczy src to path.
    oczy_root = os.environ.get(
        "OCZY_ROOT",
        str(Path(__file__).resolve().parent.parent.parent / "src"),
    )
    sys.path.insert(0, oczy_root)

    from llama_cpp import Llama
    from oczy.experiments.codebase_qa.knowledge_store import KnowledgeStore

    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)

    # Load facts if available.
    store = KnowledgeStore(embed_fn=None)
    if facts_path and Path(facts_path).exists():
        with open(facts_path) as f:
            for fact in json.load(f):
                store.add_fact(fact["key"], fact["value"], fact.get("metadata", {}))

    prompt = store.format_context(instruction) + "\n\n"
    prompt += f"<|user|>\n{instruction}\n<|assistant|>\n"

    output = llm(prompt, max_tokens=256, temperature=0.0, stop=["<|user|>", "\n\n"])
    return output["choices"][0]["text"].strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--mode", choices=["oczy", "vanilla"], required=True)
    p.add_argument("--facts", default="", help="Path to facts.json")
    p.add_argument("--instruction", required=True, help="Task instruction")
    args = p.parse_args()

    if args.mode == "oczy":
        answer = _run_oczy(args.model, args.instruction, args.facts)
    else:
        answer = _run_vanilla(args.model, args.instruction)

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
