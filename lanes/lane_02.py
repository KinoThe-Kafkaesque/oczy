"""Lane 02: KV-slot fact injection — cvec capacity boundary.

Measures ``capacity_facts_at_rank1`` for the cvec-only path: how many of
the FACTS in ``src/oczy/experiments/multi_fact_stressor.py`` can be made
to surface their target token at rank 1 of the LM's next-token logits
when the fact itself is applied as a residual cvec (uniform across all
layers) at scale 0.03. The KV-chunk path
(``research/02-kv-slot-fact-injection.md`` H1) needs the binding fork
and is out of scope here — this lane reports only the cvec-side ceiling
that the kv-chunk path is hypothesised to exceed.

Real-driver only: the LFM2.5-1.2B-Instruct hybrid conv/attention
recurrent state cannot be exercised by the mock. If the GGUF is
missing or load fails, returns ``float('nan')``. Whole body wrapped in
try/except — never raises.
"""

from __future__ import annotations


_PROBE_SIZE = 3  # skylark / rook / marmalade probe set
_CVEC_SCALE = 0.03


def name() -> str:
    return "lane_02_capacity_cvec"


def measure() -> float:
    try:
        import numpy as np

        from src.oczy.experiments.multi_fact_stressor import (
            FACTS,
            QUERIES,
            TARGETS,
            _resolve_gguf_path,
        )
        from src.oczy.lm.cvec_driver import LlamaCVecDriver
        from llama_cpp import Llama

        facts = list(FACTS[:_PROBE_SIZE])
        queries = list(QUERIES[:_PROBE_SIZE])
        targets = list(TARGETS[:_PROBE_SIZE])
        if not facts or not (len(facts) == len(queries) == len(targets)):
            return float("nan")

        resolved = _resolve_gguf_path()
        if resolved is None:
            return float("nan")

        llm = Llama(
            model_path=str(resolved),
            n_ctx=512,
            n_threads=4,
            embedding=True,
            verbose=False,
        )
        driver = LlamaCVecDriver(llm)
        n_vocab = driver.n_vocab

        reached = 0
        for fact, query, target in zip(facts, queries, targets, strict=True):
            driver.clear_cvec()
            # Derive the cvec from the fact's own mean-pooled embedding.
            vec = driver.peek_embedding(fact, last_token_only=False)
            if vec.shape[0] != driver.n_embd:
                continue
            if driver.set_cvec_uniform(vec, scale=_CVEC_SCALE) != 0:
                continue
            probe = f"Answer briefly.\nQuestion: {query}\nAnswer:"
            prompt_ids = llm.tokenize(probe.encode("utf-8"), add_bos=True)
            llm.reset()
            llm.eval(prompt_ids)
            # First subword of " <target>" is the rank-1 candidate to test.
            target_ids = llm.tokenize((" " + target).encode("utf-8"), add_bos=False)
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            raw = llm._ctx.get_logits()
            full = np.ctypeslib.as_array(raw, shape=(len(prompt_ids) * n_vocab,))
            last = full[(len(prompt_ids) - 1) * n_vocab : len(prompt_ids) * n_vocab]
            if int(np.argmax(last)) == target_id:
                reached += 1

        driver.clear_cvec()
        return float(reached)
    except Exception:
        return float("nan")