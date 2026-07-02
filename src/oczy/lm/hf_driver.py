"""HFDriver: HuggingFace/PyTorch substrate driver for Sprint 1.

Mirrors the public surface of ``LlamaCVecDriver`` (frozen-legacy, see
``cvec_driver.py`` and ``LEGACY.md``) while adding the two capabilities
llama.cpp never provided:

* Goal 1 — KV-slot writes via ``past_key_values`` splicing.
* Goal 2 — per-layer hidden reads via ``output_hidden_states=True``.

Lifecycle::

    driver = HFDriver.load("Qwen/Qwen2.5-0.5B")
    driver.set_cvec_uniform(vec)       # steering vector
    result = driver.generate(prompt)   # greedy decode with cvec
    driver.peek_layer(prompt, 5)       # per-layer hidden
    handle = driver.encode_kv(fact)    # capture KV cache
    driver.generate_with_kv(prompt, handle)  # splice KV
    driver.clear_cvec()

The driver owns the HuggingFace model+tokenizer pair.  CVec steering uses
PyTorch forward pre-hooks on each decoder layer, which is precise (adds
before the layer, affects only that layer's residual) and cheap to
register / remove.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cvec_driver import ReservedPosition

# ---------------------------------------------------------------------------
# Parallel-task model choice (S1.1).  Fall back to a constructor-required
# arg if the module hasn't landed yet.
# ---------------------------------------------------------------------------
try:
    from .hf_model_choice import DEFAULT_MODEL_ID  # type: ignore[import-untyped]
except ImportError:
    DEFAULT_MODEL_ID: str | None = None  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# KV-slot handle
# ---------------------------------------------------------------------------


@dataclass
class KVHandle:
    """Opaque handle to a pre-computed KV cache for splicing.

    ``past_key_values`` follows the HuggingFace convention: tuple of
    ``(key, value)`` per layer, each ``[1, n_heads, seq_len, head_dim]``.
    ``seq_len`` is the number of tokens in the cached prefix.
    """

    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seq_len: int


# ---------------------------------------------------------------------------
# HFDriver
# ---------------------------------------------------------------------------


class HFDriver:
    """Persistent control-vector binding for a single HuggingFace model.

    Constructor takes a pre-loaded model + tokenizer pair; callers
    should use ``HFDriver.load(model_id=…)`` for standard setup.

    The model is held in ``float32`` CPU eval mode.  Weights never
    change — only the steering surface (cvec, reserved position,
    logit bias, KV slots) changes between calls.
    """

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: "torch.nn.Module",
        tokenizer: Any,
        model_id: str,
    ) -> None:
        cfg = model.config
        self._model: torch.nn.Module = model
        self._tokenizer: Any = tokenizer
        self.model_id: str = model_id
        self.n_embd: int = int(cfg.hidden_size)
        self.n_vocab: int = int(cfg.vocab_size)
        self.n_layers: int = int(cfg.num_hidden_layers)

        # Cvec tracking
        self._cvec_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._cvec_active: bool = False

        # Reserved position
        self._reserved_position: ReservedPosition | None = None

        # Embedding cache
        self._embedding_cache: OrderedDict[
            tuple[str, bool], np.ndarray
        ] = OrderedDict()
        self._embedding_cache_maxsize: int = 128

        self._loaded: bool = True

    @classmethod
    def load(cls, model_id: str | None = None) -> "HFDriver":
        """Lazy model + tokenizer init via HuggingFace.

        ``model_id`` defaults to whatever ``hf_model_choice.DEFAULT_MODEL_ID``
        says (if the parallel S1.1 module exists); falls back to requiring an
        explicit argument.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mid = model_id
        if mid is None:
            if DEFAULT_MODEL_ID is not None:
                mid = DEFAULT_MODEL_ID
            else:
                raise ValueError(
                    "HFDriver.load() requires model_id=...; "
                    "hf_model_choice module not yet available (S1.1 parallel task)"
                )

        model = AutoModelForCausalLM.from_pretrained(
            mid,
            dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(mid)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Fix pad_token_id warning on the model config.
        if model.config.pad_token_id is None or model.config.pad_token_id < 0:
            model.config.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model, tokenizer, mid)

    # ------------------------------------------------------------------
    # Cvec: forward pre-hook steering
    # ------------------------------------------------------------------

    def set_cvec_layer(self, layer_idx: int, vec: np.ndarray) -> None:
        """Apply an ``n_embd``-dim steering vector to a single layer.

        The vector is added to the residual stream input of *that one
        decoder layer* via a forward pre-hook.  Other layers are
        unaffected; repeated calls on the same layer replace the
        previous hook for that layer.
        """
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(
                "layer_idx %d out of range [0, %d)" % (layer_idx, self.n_layers)
            )
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.n_embd:
            raise ValueError(
                "vec dim %d != n_embd %d" % (vec.shape[0], self.n_embd)
            )
        bias = torch.from_numpy(vec.copy()).float()

        # Remove any existing hook for this specific layer.
        self._remove_layer_cvec_hook(layer_idx)

        target = self._get_decoder_layer(layer_idx)

        def _pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
            hs = args[0]
            return (hs + bias.view(1, 1, -1).to(device=hs.device, dtype=hs.dtype),)

        handle = target.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        self._cvec_hooks.append(handle)
        self._cvec_active = True

    def set_cvecs_per_layer(
        self, vectors: Sequence[np.ndarray], scale: float = 1.0
    ) -> None:
        """Apply one distinct cvec per layer.

        ``len(vectors)`` must equal ``self.n_layers``.  Each hook is
        registered independently; ``scale`` is applied to every vector
        before registration.
        """
        if len(vectors) != self.n_layers:
            raise ValueError(
                "expected %d vectors (one per layer), got %d"
                % (self.n_layers, len(vectors))
            )
        for i, v in enumerate(vectors):
            v = np.ascontiguousarray(v, dtype=np.float32).reshape(-1)
            if v.shape[0] != self.n_embd:
                raise ValueError(
                    "vector[%d] dim %d != n_embd %d"
                    % (i, v.shape[0], self.n_embd)
                )
            if scale != 1.0:
                v = v * scale
            self.set_cvec_layer(i, v)

    def set_cvec_uniform(self, vec: np.ndarray, scale: float = 1.0) -> None:
        """Apply the same steering vector to every layer at once."""
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.n_embd:
            raise ValueError(
                "vec dim %d != n_embd %d" % (vec.shape[0], self.n_embd)
            )
        bias = torch.from_numpy((vec * scale).copy()).float()

        for layer_idx in range(self.n_layers):
            self._remove_layer_cvec_hook(layer_idx)
            target = self._get_decoder_layer(layer_idx)

            def _pre_hook(
                _module: torch.nn.Module,
                args: tuple[torch.Tensor],
                _bias: torch.Tensor = bias,
            ) -> tuple[torch.Tensor]:
                hs = args[0]
                return (
                    hs + _bias.view(1, 1, -1).to(device=hs.device, dtype=hs.dtype),
                )

            handle = target.register_forward_pre_hook(_pre_hook, with_kwargs=False)
            self._cvec_hooks.append(handle)
        self._cvec_active = True

    def clear_cvec(self) -> None:
        """Remove every registered steering-vector hook.

        Idempotent — safe to call when no cvec is active.
        """
        for h in self._cvec_hooks:
            h.remove()
        self._cvec_hooks.clear()
        self._cvec_active = False

    @property
    def cvec_active(self) -> bool:
        return self._cvec_active

    # ------------------------------------------------------------------
    # Reserved position
    # ------------------------------------------------------------------

    def set_reserved_position(self, position: ReservedPosition | None) -> None:
        self._reserved_position = position

    def clear_reserved_position(self) -> None:
        self._reserved_position = None

    @property
    def reserved_position(self) -> ReservedPosition | None:
        return self._reserved_position

    @property
    def reserved_position_active(self) -> bool:
        return bool(
            self._reserved_position is not None
            and self._reserved_position.text
        )

    # Deprecated thin wrappers.
    def set_articulation_prefix(self, text: str) -> None:
        self._reserved_position = ReservedPosition(text=text)

    def clear_articulation_prefix(self) -> None:
        self.clear_reserved_position()

    @property
    def articulation_prefix(self) -> str | None:
        return (
            self._reserved_position.text if self._reserved_position else None
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _apply_reserved_prefix(self, prompt: str) -> str:
        if self._reserved_position is not None:
            prefix = self._reserved_position.text
            if not prompt.startswith(prefix):
                return prefix + prompt
        return prompt

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long)

    def _greedy_step(self, logits: torch.Tensor) -> int:
        """Last-token argmax, returns a Python int."""
        last = logits[0, -1, :]
        return int(last.argmax().item())

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: Sequence[str] | str | None = None,
    ) -> str:
        """Greedy-deterministic decode honoring active cvec / reserved position."""
        if temperature != 0.0:
            raise NotImplementedError(
                "HFDriver.generate only supports temperature=0.0 (greedy)"
            )

        effective = self._apply_reserved_prefix(prompt)

        if stop is None:
            stop_list: list[str] = []
        elif isinstance(stop, str):
            stop_list = [stop]
        else:
            stop_list = list(stop)

        input_ids = self._tokenize(effective)
        out = self._model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = self._greedy_step(out.logits)

        generated_ids: list[int] = [next_token]
        eos_id = self._tokenizer.eos_token_id

        for _ in range(max_tokens - 1):
            if next_token == eos_id:
                break

            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            out = self._model(
                input_ids=token_tensor,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = self._greedy_step(out.logits)
            generated_ids.append(next_token)

            # Check stop sequences on the generated text so far.
            current = self._tokenizer.decode(generated_ids)
            for s in stop_list:
                if s and current.endswith(s):
                    generated_ids = []
                    break
            if not generated_ids:
                break

        if not generated_ids:
            return ""
        return self._tokenizer.decode(generated_ids)

    def logit_bias_generate(
        self,
        prompt: str,
        target_token_ids: list[int],
        bias: float = 20.0,
        max_tokens: int = 32,
        stop: Sequence[str] | str | None = None,
    ) -> str:
        """Generate with logit biasing on target tokens (composes with cvec).

        Mirror of LlamaCVecDriver.logit_bias_generate: biases the target
        token's logit at each step until the target is fully emitted.
        """
        effective = self._apply_reserved_prefix(prompt)

        if stop is None:
            stop_list: list[str] = []
        elif isinstance(stop, str):
            stop_list = [stop]
        else:
            stop_list = list(stop)

        input_ids = self._tokenize(effective)
        out = self._model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].clone()

        # Apply bias to first target token.
        target_idx = 0
        if target_token_ids:
            tid = target_token_ids[0]
            if 0 <= tid < self.n_vocab:
                logits[tid] += bias

        next_token = int(logits.argmax().item())
        generated_ids: list[int] = [next_token]
        if target_token_ids and next_token == target_token_ids[0]:
            target_idx = 1
        else:
            target_idx = len(target_token_ids)

        eos_id = self._tokenizer.eos_token_id

        for _ in range(max_tokens - 1):
            if next_token == eos_id:
                break

            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            out = self._model(
                input_ids=token_tensor,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1, :].clone()

            if target_idx < len(target_token_ids):
                tid = target_token_ids[target_idx]
                if 0 <= tid < self.n_vocab:
                    logits[tid] += bias

            next_token = int(logits.argmax().item())
            generated_ids.append(next_token)

            if target_idx < len(target_token_ids):
                if next_token == target_token_ids[target_idx]:
                    target_idx += 1
                else:
                    target_idx = len(target_token_ids)

            current = self._tokenizer.decode(generated_ids)
            for s in stop_list:
                if s and current.endswith(s):
                    generated_ids = []
                    break
            if not generated_ids:
                break

        if not generated_ids:
            return ""
        return self._tokenizer.decode(generated_ids)

    # ------------------------------------------------------------------
    # Perception: embeddings + per-layer hiddens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def peek_embedding(
        self, prompt: str, last_token_only: bool = True
    ) -> np.ndarray:
        """Return final-layer hidden for ``prompt``.

        When ``last_token_only=True`` (default), returns the last
        token's hidden.  When ``False``, mean-pools all tokens.
        Results are cached per (prompt, last_token_only).
        """
        cache_key = (prompt, last_token_only)
        cache = self._embedding_cache
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]

        input_ids = self._tokenize(prompt)
        out = self._model(input_ids=input_ids, output_hidden_states=True)
        final_hidden = out.hidden_states[-1]  # (1, seq_len, n_embd)

        if last_token_only:
            vec = final_hidden[0, -1, :]  # (n_embd,)
        else:
            vec = final_hidden[0].mean(dim=0)  # (n_embd,)

        emb = vec.cpu().to(dtype=torch.float32).numpy()
        emb = np.ascontiguousarray(emb)

        if len(cache) >= self._embedding_cache_maxsize:
            cache.popitem(last=False)
        cache[cache_key] = emb
        return emb

    @torch.no_grad()
    def peek_layer(
        self,
        prompt: str,
        layer_idx: int,
        pooling: str = "last",
    ) -> np.ndarray:
        """Return the hidden state at decoder layer ``layer_idx``.

        Args:
            prompt: text to forward through the model.
            layer_idx: 0-based decoder layer index.
            pooling: ``"last"`` (default — last-token hidden),
                     ``"mean"`` (mean-pool all tokens), or
                     ``"max"`` (max-pool over tokens).

        Returns:
            ``ndarray`` of shape ``(n_embd,)``, dtype ``float32``.
        """
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(
                "layer_idx %d out of range [0, %d)"
                % (layer_idx, self.n_layers)
            )
        if pooling not in ("last", "mean", "max"):
            raise ValueError(
                "pooling must be 'last', 'mean', or 'max', got %r" % pooling
            )

        input_ids = self._tokenize(prompt)
        out = self._model(input_ids=input_ids, output_hidden_states=True)
        # hidden_states: (embeddings, layer_0_output, …, layer_{n_layers-1}_output, final_norm)
        # Layer 0 output is at index 1.
        hidden = out.hidden_states[layer_idx + 1]  # (1, seq_len, n_embd)

        if pooling == "last":
            vec = hidden[0, -1, :]
        elif pooling == "max":
            vec = hidden[0].max(dim=0).values
        else:
            vec = hidden[0].mean(dim=0)

        emb = vec.cpu().to(dtype=torch.float32).numpy()
        return np.ascontiguousarray(emb)

    # ------------------------------------------------------------------
    # Goal 1: KV-slot API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_kv(self, text: str) -> KVHandle:
        """Encode ``text`` and capture its full KV cache.

        The returned ``KVHandle`` can be spliced into
        ``generate_with_kv`` or ``token_ranks_with_kv``, giving the
        model access to ``text`` without it being visible in the
        prompt tokens.
        """
        input_ids = self._tokenize(text)
        seq_len = input_ids.shape[1]
        out = self._model(input_ids=input_ids, use_cache=True)
        pkv = out.past_key_values
        if pkv is None:
            raise RuntimeError(
                "use_cache=True did not produce past_key_values"
            )
        return KVHandle(past_key_values=pkv, seq_len=seq_len)

    @torch.no_grad()
    def generate_with_kv(
        self,
        prompt: str,
        handle: KVHandle,
        max_tokens: int = 32,
    ) -> str:
        """Generate with a pre-computed KV cache spliced ahead of the prompt.

        The model attends to ``handle``'s content but does not see it
        in the visible sequence — the fact is attendable, not present.
        """
        effective = self._apply_reserved_prefix(prompt)
        input_ids = self._tokenize(effective)

        # Position IDs: fact used 0..seq_len-1, prompt continues from seq_len.
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(
            handle.seq_len,
            handle.seq_len + prompt_len,
            dtype=torch.long,
        ).unsqueeze(0)

        out = self._model(
            input_ids=input_ids,
            past_key_values=handle.past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )

        past = out.past_key_values
        next_token = self._greedy_step(out.logits)
        generated_ids: list[int] = [next_token]
        eos_id = self._tokenizer.eos_token_id
        current_pos = handle.seq_len + prompt_len

        for _ in range(max_tokens - 1):
            if next_token == eos_id:
                break

            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            pos_tensor = torch.tensor([[current_pos]], dtype=torch.long)
            out = self._model(
                input_ids=token_tensor,
                past_key_values=past,
                position_ids=pos_tensor,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = self._greedy_step(out.logits)
            generated_ids.append(next_token)
            current_pos += 1

        return self._tokenizer.decode(generated_ids)

    @torch.no_grad()
    def token_ranks(
        self,
        prompt: str,
        targets: list[str],
    ) -> list[dict[str, Any]]:
        """Return the logit rank of each target token after the prompt.

        For each target string, compute the rank of its first token
        among the vocabulary logits (0 = best, vocab_size-1 = worst).
        Used as the baseline for KV-slot measurements.
        """
        input_ids = self._tokenize(prompt)
        out = self._model(input_ids=input_ids)
        logits = out.logits[0, -1, :].cpu()

        results: list[dict[str, Any]] = []
        for target in targets:
            target_ids = self._tokenizer.encode(
                target, add_special_tokens=False
            )
            if not target_ids:
                results.append(
                    {"target": target, "rank": self.n_vocab, "top1": ""}
                )
                continue
            tid = target_ids[0]
            rank = int((logits > logits[tid]).sum().item())
            top1_id = int(logits.argmax().item())
            top1 = self._tokenizer.decode([top1_id])
            results.append(
                {
                    "target": target,
                    "rank": rank,
                    "top1": top1,
                    "target_id": tid,
                }
            )
        return results

    @torch.no_grad()
    def token_ranks_with_kv(
        self,
        prompt: str,
        handle: KVHandle,
        targets: list[str],
    ) -> list[dict[str, Any]]:
        """Like ``token_ranks`` but with an encoded KV cache spliced."""
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(
            handle.seq_len,
            handle.seq_len + prompt_len,
            dtype=torch.long,
        ).unsqueeze(0)

        out = self._model(
            input_ids=input_ids,
            past_key_values=handle.past_key_values,
            position_ids=position_ids,
        )
        logits = out.logits[0, -1, :].cpu()

        results: list[dict[str, Any]] = []
        for target in targets:
            target_ids = self._tokenizer.encode(
                target, add_special_tokens=False
            )
            if not target_ids:
                results.append(
                    {"target": target, "rank": self.n_vocab, "top1": ""}
                )
                continue
            tid = target_ids[0]
            rank = int((logits > logits[tid]).sum().item())
            top1_id = int(logits.argmax().item())
            top1 = self._tokenizer.decode([top1_id])
            results.append(
                {
                    "target": target,
                    "rank": rank,
                    "top1": top1,
                    "target_id": tid,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        rp = self._reserved_position
        preview: str | None = None
        if rp is not None:
            t = rp.text
            preview = t[:60] + "..." if len(t) > 60 else t

        return {
            "model_id": self.model_id,
            "n_embd": self.n_embd,
            "n_layers": self.n_layers,
            "n_vocab": self.n_vocab,
            "cvec_active": self._cvec_active,
            "reserved_position_active": self.reserved_position_active,
            "reserved_position_source": rp.source if rp else None,
            "reserved_position_text_preview": preview,
            "lm_loaded": self._loaded,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.clear_cvec()
        self._embedding_cache.clear()
        self._model = None  # type: ignore[assignment]
        self._tokenizer = None  # type: ignore[assignment]
        self._loaded = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "HFDriver":
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_decoder_layer(self, layer_idx: int) -> torch.nn.Module:
        """Return the PyTorch module for a specific decoder layer.

        Handles common model architectures (Llama, Qwen, GPT-2, etc.)
        by trying standard attribute paths.
        """
        model = self._model
        for attr_path in (
            "model.layers",  # Llama, Qwen2, Mistral, Gemma
            "model.decoder.layers",  # T5-style, BART
            "transformer.h",  # GPT-2
            "transformer.decoder.layers",
        ):
            parts = attr_path.split(".")
            obj = model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if isinstance(obj, torch.nn.ModuleList):
                    return obj[layer_idx]
                if isinstance(obj, (list, tuple)) and layer_idx < len(obj):
                    return obj[layer_idx]

        # Last resort: find any ModuleList of decoder layers.
        for _name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                if len(module) == self.n_layers:
                    return module[layer_idx]

        raise AttributeError(
            "Cannot find decoder layer %d; model architecture not recognized. "
            "Expected model.model.layers, model.model.decoder.layers, "
            "model.transformer.h, or similar." % layer_idx
        )

    def _remove_layer_cvec_hook(self, layer_idx: int) -> None:
        """Currently a no-op; hooks are managed en masse by clear_cvec().

        Single-layer cvec replacement is handled by clear_cvec followed
        by re-registration of all desired hooks in set_cvec_layer /
        set_cvec_uniform.
        """
        pass
