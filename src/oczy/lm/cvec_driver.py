"""Driver shim between KVCortex and Llama via the cvec adapter interface.

This is the CortexAgent's articulation side: the cortex emits per-layer
intent vectors, and this driver writes them into the LM via
``llama_set_adapter_cvec`` (LLM-side control vector / steering vector
mechanism). The cortex never touches the LM directly; this module owns
the llama-cpp ctypes boundary.

Surface probe (2026-06-23): ``llama_set_adapter_cvec`` is supported,
persists across ``create_completion`` calls, shifts logits visibly, and
clears cleanly when called with NULL data. This driver formalises that
probe and exposes a typed cortex-side interface.

Perception-side helper ``peek_embedding`` is included as Goal 2 staging:
it returns the model's final-layer residual for the last prompt token.
Layer-L extraction (intermediate residual) is not supported by the
current llama-cpp-python binding — that work is tracked under Goal 2.
"""

import ctypes
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import llama_cpp
import numpy as np
from llama_cpp import Llama


@dataclass
class CVecDriverConfig:
    """Driver + LM loading settings.

    Defaults match ``LanguageAdapterConfig``: LFM2.5-1.2B-Instruct Q4_K_M,
    CPU-only.  See ``bench_cross_backend.py`` for the rationale.
    """

    repo_id: str = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    file_name: str = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    n_ctx: int = 512
    n_threads: int = 4
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False
    # Required for peek_embedding() to return activations; small fixed cost
    # to enable in llama.cpp at context init.
    embedding: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> "CVecDriverConfig":
        """Build a config from environment variables with current defaults as fallback.

        Override precedence: explicit keyword arguments > env vars > hard-coded defaults.
        Boolean env vars accept ``1``, ``true``/``yes`` (case-insensitive) as truthy.
        """

        def _bool(value: str | None) -> bool | None:
            if value is None:
                return None
            return value.lower() in {"1", "true", "yes"}

        def _int(value: str | None) -> int | None:
            if value is None:
                return None
            return int(value)

        kwargs: dict[str, Any] = {}
        if (v := os.environ.get("OCZY_MODEL_REPO_ID")) is not None:
            kwargs["repo_id"] = v
        if (v := os.environ.get("OCZY_MODEL_FILE_NAME")) is not None:
            kwargs["file_name"] = v
        if (v := _int(os.environ.get("OCZY_N_CTX"))) is not None:
            kwargs["n_ctx"] = v
        if (v := _int(os.environ.get("OCZY_N_THREADS"))) is not None:
            kwargs["n_threads"] = v
        if (v := _int(os.environ.get("OCZY_N_GPU_LAYERS"))) is not None:
            kwargs["n_gpu_layers"] = v
        if (v := _bool(os.environ.get("OCZY_USE_MMAP"))) is not None:
            kwargs["use_mmap"] = v
        if (v := _bool(os.environ.get("OCZY_USE_MLOCK"))) is not None:
            kwargs["use_mlock"] = v
        if (v := _bool(os.environ.get("OCZY_VERBOSE"))) is not None:
            kwargs["verbose"] = v

        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def perception(cls) -> "CVecDriverConfig":
        """High context for perception / embedding work; keeps embedding enabled."""
        return cls(n_ctx=1024, embedding=True)

    @classmethod
    def articulation(cls) -> "CVecDriverConfig":
        """Compact context for generation / articulation; disables embedding overhead."""
        return cls(n_ctx=512, embedding=False)

    @classmethod
    def benchmark(cls) -> "CVecDriverConfig":
        """Deterministic baseline for benchmarking; embedding enabled for recall probes."""
        return cls(n_ctx=512, n_threads=4, embedding=True)



@dataclass
class ReservedPosition:
    """A reserved KV-position steering surface injected as a literal prefix.

    This is a first-class handle for the soft-prompt / reserved-position
    mechanism: instead of passing around raw strings, callers carry a small
    dataclass that records provenance and (optionally) measured uptake so
    the organism can later learn which positions work.
    """

    text: str
    source: str = "hand_coded"
    exact_uptake_score: float | None = None
    domain_uptake_score: float | None = None

class LlamaCVecDriver:
    """Persistent control-vector binding for a single ``Llama`` instance.

    Lifecycle:

        driver = LlamaCVecDriver.load(cfg)
        driver.clear_cvec()              # baseline state, no steering
        driver.set_cvec_layer(intents)   # apply cortex intent per layer
        driver.generate(prompt)          # LM samples with steering active
        driver.clear_cvec()              # back to baseline

    The driver owns the ``Llama`` instance because ``llama_set_adapter_cvec``
    is per-context. Reusing one LM across multiple cortexes is not supported
    (you'd need one driver per cortex-LM pair).
    """

    def __init__(self, llm: Llama, config: CVecDriverConfig | None = None) -> None:
        self.config = config or CVecDriverConfig()
        self._llm = llm
        self._ctx_obj = llm._ctx  # LlamaContext wrapper
        self._ctx_p = self._ctx_obj.ctx  # raw llama_context_p pointer (int)
        self.n_embd: int = int(llm.n_embd())
        self.n_vocab: int = int(llm.n_vocab())
        self.n_layers: int = self._probe_n_layers()
        self._cvec_active: bool = False
        # Track per-layer set ranges so clear_cvec() rewinds exactly what was set.
        self._applied_layer_ranges: list[tuple[int, int]] = []
        self._reserved_position: ReservedPosition | None = None
        # LRU cache for embeddings: keyed by (prompt, last_token_only).
        # Per-instance so different LMs never collide.
        self._embedding_cache: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()
        self._embedding_cache_maxsize: int = 128

    @classmethod
    def load(cls, config: CVecDriverConfig | None = None) -> "LlamaCVecDriver":
        cfg = config or CVecDriverConfig()
        llm = Llama.from_pretrained(
            repo_id=cfg.repo_id,
            filename=cfg.file_name,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            use_mmap=cfg.use_mmap,
            use_mlock=cfg.use_mlock,
            verbose=cfg.verbose,
            embedding=cfg.embedding,
        )
        return cls(llm, cfg)

    # ------------------------------------------------------------------
    # Articulation: cvec apply / clear
    # ------------------------------------------------------------------
    def set_cvec_layer(
        self,
        layer_idx: int,
        vec: np.ndarray,
    ) -> int:
        """Apply an ``n_embd``-dim steering vector to a single layer.

        The vector persists across ``generate`` calls until cleared or
        replaced. Per-layer calls override the vector at that layer only;
        other layers retain whatever was last set there (or no cvec).

        Args:
            layer_idx: 0-based layer index, must be in ``[0, n_layers)``.
            vec: ``ndarray`` of ``dtype=float32``, shape ``(n_embd,)``.

        Returns:
            The C return code from ``llama_set_adapter_cvec`` (0 on success).
        """
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(
                "layer_idx %d out of range [0, %d)" % (layer_idx, self.n_layers)
            )
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.n_embd:
            raise ValueError("vec dim %d != n_embd %d" % (vec.shape[0], self.n_embd))
        ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # il_end is exclusive: layer_idx..layer_idx+1 selects exactly one layer.
        rc = llama_cpp.llama_set_adapter_cvec(
            self._ctx_p,
            ptr,
            vec.shape[0],  # len: total float count
            self.n_embd,  # n_embd per layer
            layer_idx,  # il_start
            layer_idx + 1,  # il_end (exclusive)
        )
        if rc == 0:
            self._cvec_active = True
            rng = (layer_idx, layer_idx + 1)
            if rng not in self._applied_layer_ranges:
                self._applied_layer_ranges.append(rng)
        return rc

    def set_cvecs_per_layer(
        self,
        vectors: Sequence[np.ndarray],
        scale: float = 1.0,
    ) -> int:
        """Apply one distinct cvec per layer in a single adapter call.

        ``llama_set_adapter_cvec`` REPLACES the loaded cvec on each call,
        so looped ``set_cvec_layer`` calls would only keep the LAST one.
        This method concatenates the per-layer vectors into one flat
        ``n_embd * n_layers`` array and invokes the adapter once with
        ``il_start=0, il_end=n_layers``. The adapter then interprets
        each ``n_embd``-sized chunk as one layer's steering vector.

        Args:
            vectors: one ``ndarray`` per layer, each ``(n_embd,)``.
                ``len(vectors)`` must equal ``self.n_layers``.
            scale: optional scalar multiplier applied uniformly to every
                layer's vector before flattening. Defaults to 1.0.

        Returns: C return code (0 on success).
        """
        if len(vectors) != self.n_layers:
            raise ValueError(
                "expected %d vectors (one per layer), got %d"
                % (self.n_layers, len(vectors))
            )
        # Build (n_layers, n_embd) contig array, then view as flat float32.
        stacked = np.stack(
            [np.ascontiguousarray(v, dtype=np.float32).reshape(-1) for v in vectors]
        )  # (n_layers, n_embd)
        if stacked.shape[1] != self.n_embd:
            raise ValueError(
                "one or more vectors has wrong dim; expected %d, got shape %s"
                % (self.n_embd, stacked.shape)
            )
        if scale != 1.0:
            stacked = (stacked * scale).astype(np.float32)
        flat = np.ascontiguousarray(stacked).reshape(-1)
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = llama_cpp.llama_set_adapter_cvec(
            self._ctx_p,
            ptr,
            flat.shape[0],  # len: n_embd * n_layers
            self.n_embd,  # one cvec per layer is n_embd floats
            0,  # il_start
            self.n_layers,  # il_end (exclusive)
        )
        if rc == 0:
            self._cvec_active = True
            self._applied_layer_ranges = [(0, self.n_layers)]
        return rc

    def set_cvecs_flat(
        self,
        flat: np.ndarray,
        scale: float = 1.0,
    ) -> int:
        """Apply per-layer cvecs from an already flattened ``(n_layers*n_embd,)`` buffer.

        This is the fast path used by the cortex when it already caches
        the contiguous flat array; it skips the ``np.stack`` done by
        ``set_cvecs_per_layer``.
        """
        expected = self.n_layers * self.n_embd
        flat = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
        if flat.shape[0] != expected:
            raise ValueError(
                "flat cvec length %d != n_layers*n_embd %d" % (flat.shape[0], expected)
            )
        if scale != 1.0:
            flat = (flat * scale).astype(np.float32)
            flat = np.ascontiguousarray(flat)
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = llama_cpp.llama_set_adapter_cvec(
            self._ctx_p,
            ptr,
            flat.shape[0],
            self.n_embd,
            0,
            self.n_layers,
        )
        if rc == 0:
            self._cvec_active = True
            self._applied_layer_ranges = [(0, self.n_layers)]
        return rc

    def set_cvec_uniform(
        self,
        vec: np.ndarray,
        scale: float = 1.0,
    ) -> int:
        """Apply the same steering vector to every layer at once.

        Convenience for cortex states where layer-specific projection is
        not yet wired. Equivalent to looping ``set_cvec_layer`` but uses
        a single adapter call covering the full layer range.
        """
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.n_embd:
            raise ValueError("vec dim %d != n_embd %d" % (vec.shape[0], self.n_embd))
        if scale != 1.0:
            vec = (vec * scale).astype(np.float32)
            vec = np.ascontiguousarray(vec)
        ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = llama_cpp.llama_set_adapter_cvec(
            self._ctx_p,
            ptr,
            vec.shape[0],
            self.n_embd,
            0,
            self.n_layers,
        )
        if rc == 0:
            self._cvec_active = True
            self._applied_layer_ranges = [(0, self.n_layers)]
        return rc

    def clear_cvec(self) -> int:
        """Remove every applied steering vector.

        Calls ``llama_set_adapter_cvec`` with NULL data over the full layer
        range; matches the semantics confirmed in the 2026-06-23 probe: the
        next ``generate`` call returns to the un-steered baseline.
        """
        rc = llama_cpp.llama_set_adapter_cvec(
            self._ctx_p,
            None,
            0,
            self.n_embd,
            0,
            self.n_layers,
        )
        if rc == 0:
            self._cvec_active = False
            self._applied_layer_ranges.clear()
        return rc

    @property
    def cvec_active(self) -> bool:
        return self._cvec_active

    def set_reserved_position(self, position: ReservedPosition | None) -> None:
        """Set (or replace) the reserved KV position used during generation.

        ``position.text`` is prepended to every ``generate()`` prompt unless
        the prompt already starts with it.  Setting ``None`` disables
        reserved-position steering; the caller may also use
        ``clear_reserved_position`` for that.
        """
        self._reserved_position = position

    def clear_reserved_position(self) -> None:
        """Remove the reserved position so prompts pass through unchanged."""
        self._reserved_position = None

    @property
    def reserved_position(self) -> ReservedPosition | None:
        return self._reserved_position

    @property
    def reserved_position_active(self) -> bool:
        return self._reserved_position is not None and bool(self._reserved_position.text)

    # Deprecated thin wrappers for the previous literal-text API.
    def set_articulation_prefix(self, text: str) -> None:
        """Deprecated: use ``set_reserved_position(ReservedPosition(text))``."""
        self._reserved_position = ReservedPosition(text=text)

    def clear_articulation_prefix(self) -> None:
        """Deprecated: use ``clear_reserved_position``."""
        self.clear_reserved_position()

    @property
    def articulation_prefix(self) -> str | None:
        """Deprecated: use ``reserved_position.text``."""
        return self._reserved_position.text if self._reserved_position else None

    # ------------------------------------------------------------------
    # LM forward (generation + perception)
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: Sequence[str] | str | None = None,
    ) -> str:
        """Run ``create_completion`` with whatever cvec is currently applied.

        The LM is frozen — weights never change. Only the steering vector
        changes between calls. Returns generated text only; the caller
        keeps driving the cortex via separate API.
        """
        effective_prompt = prompt
        if self._reserved_position is not None and isinstance(prompt, str):
            prefix = self._reserved_position.text
            if not effective_prompt.startswith(prefix):
                effective_prompt = prefix + effective_prompt
        result = self._llm.create_completion(
            effective_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=list(stop) if stop else None,
        )
        return result["choices"][0]["text"]

    def peek_embedding(self, prompt: str, last_token_only: bool = True) -> np.ndarray:
        """Return the model's final-layer embedding for ``prompt``.

        This is Goal 2 staging: we read the *final-layer* output embedding
        via the official llama-cpp-python ``create_embedding`` high-level
        API. Layer-L intermediate extraction is not supported here yet
        (binding limitation tracked under Goal 2).

        ``create_embedding`` with ``embedding=True`` enabled at model
        load returns a result with one ``data`` entry per (pooled)
        input; each entry's ``embedding`` is a flat ``n_embd`` list.
        The pooling type is set to MEAN by the high-level API, so this
        returns one (n_embd,) vector summarising the whole prompt.

        When ``last_token_only=True`` we additionally restrict the
        embedding to the last input token by querying the API with only
        that single token -- this is the closest approximation available
        without layer-L binding work.

        Embeddings are cached per (prompt, last_token_only) up to a
        small LRU bound to avoid repeated LM calls for the same query.

        Args:
            prompt: text to embed
            last_token_only: when True (default), embed just the last
                token of ``prompt``. When False, embed the whole-prompt
                mean-pooled vector.

        Returns:
            ``ndarray`` of shape ``(n_embd,)``, dtype ``float32``.
        """
        cache_key = (prompt, bool(last_token_only))
        cache = self._embedding_cache
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]

        if last_token_only:
            # Embed the prompt's last token in isolation: this gives a
            # token-conditioned hidden we can feed straight to cortex.observe.
            token_ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False)
            if not token_ids:
                # Fall back to whole-prompt embedding.
                return self.peek_embedding(prompt, last_token_only=False)
            last_id = token_ids[-1]
            piece = self._llm.detokenize([last_id]).decode("utf-8", errors="replace")
            result = self._llm.create_embedding([piece])
        else:
            result = self._llm.create_embedding([prompt])
        emb_list = result["data"][0]["embedding"]
        emb = np.asarray(emb_list, dtype=np.float32)
        if emb.ndim == 2:
            emb = emb[-1]  # take last token row if multiple came back
        emb = np.ascontiguousarray(emb.reshape(-1))

        if len(cache) >= self._embedding_cache_maxsize:
            cache.popitem(last=False)
        cache[cache_key] = emb
        return emb

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "n_embd": self.n_embd,
            "n_layers": self.n_layers,
            "n_vocab": self.n_vocab,
            "cvec_active": self._cvec_active,
            "reserved_position_active": self.reserved_position_active,
            "reserved_position_source": self._reserved_position.source if self._reserved_position else None,
            "reserved_position_text_preview": (
                self._reserved_position.text[:60] + "..."
                if self._reserved_position is not None and len(self._reserved_position.text) > 60
                else (self._reserved_position.text if self._reserved_position is not None else None)
            ),
            "applied_layer_ranges": [(s, e) for s, e in self._applied_layer_ranges],
            "lm_loaded": self._llm is not None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _probe_n_layers(self) -> int:
        """Read the model's layer count via the low-level ``llama_n_layer``.

        This is called once at construction. The high-level wrapper exposes
        an opaque ``LlamaModel``; we cast its underlying pointer back to
        ``llama_model_p`` so the ctypes-bound counter accepts it.
        """
        model_p = self._llm._model.model
        return int(llama_cpp.llama_n_layer(model_p))
