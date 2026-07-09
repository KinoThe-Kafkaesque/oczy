#!/usr/bin/env python3
"""Oczy agent wrapper for Harbor QA benchmark.

Prepends recalled knowledge-store facts to the prompt before generating.
This is the simplest form of "Oczy agent" — knowledge retrieval without
cvec steering or scope-slot reranking.
"""

import json
import os
import sys
from pathlib import Path

from llama_cpp import Llama

from oczy.experiments.codebase_qa.knowledge_store import KnowledgeStore


def _find_model() -> str:
    env = os.environ.get("OCZY_MODEL_PATH", "")
    if env and Path(env).exists():
        return env
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    for root, _dirs, files in os.walk(str(cache)):
        if "LFM2.5-1.2B-Instruct-Q4_K_M.gguf" in files:
            return os.path.join(root, "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
    local = Path.home() / ".cache" / "oczy" / "models" / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    if local.exists():
        return str(local)
    raise FileNotFoundError("GGUF model not found. Set OCZY_MODEL_PATH.")


def _load_store() -> KnowledgeStore:
    facts_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "oczy" / "experiments" / "codebase_qa" / "facts.json"
    )
    store = KnowledgeStore(embed_fn=None)
    if facts_path.exists():
        with facts_path.open() as f:
            for fact in json.load(f):
                store.add_fact(fact["key"], fact["value"], fact.get("metadata", {}))
    return store


def main() -> int:
    question = sys.stdin.read().strip()
    if not question:
        print("ERROR: no question provided on stdin", file=sys.stderr)
        return 1

    model_path = _find_model()
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8, verbose=False)

    # Oczy: recall relevant facts and prepend to prompt.
    store = _load_store()
    prompt = store.format_context(question) + "\n\n"
    prompt += f"<|user|>\n{question}\n<|assistant|>\n"

    output = llm(
        prompt,
        max_tokens=128,
        temperature=0.0,
        stop=["<|user|>", "\n\n"],
    )
    answer = output["choices"][0]["text"].strip()
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
