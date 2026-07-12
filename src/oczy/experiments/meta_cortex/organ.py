"""Frozen language organ boundary for Research/20 meta-cortex DEV.

This module owns the real :class:`QwenFrozenOrgan` adapter and the
:class:`FrozenLanguageOrgan` protocol.  It is the *only* boundary between
the differentiable cortex (:mod:`meta_cortex.model`) and the frozen Hugging
Face language model.  The core model stays driver-independent; tests may
substitute a small differentiable frozen organ without shipping a mock
production fallback.

Design invariants (immutable spec — ``research/20-...:51-94``,
``experiments/09-.../README.md:34-46``):

* Every language-organ parameter is ``requires_grad=False`` and excluded
  from any optimizer.  Gradients may flow *through* the organ to the
  soft bank during developmental training, but organ weights never update.
* Final-layer mean-pooled features are extracted directly under
  ``torch.inference_mode()`` — never via ``HFDriver.peek_embedding``,
  which caches raw prompt strings (``hf_driver.py:498-528``).  The adapter
  keeps **no** experience-text/token/embedding cache.
* Teacher forcing follows the proven R19 ``inputs_embeds`` path
  (``s19_language_organ_core.py:475-525``): frozen request/answer token
  embeddings, prepend ``[B,L,D]`` soft bank, forward with LM frozen but
  autograd enabled, CE only at target positions.  Prompt IDs must be an
  exact prefix of full-chat IDs; empty targets or template mismatch fail
  closed.
* Greedy bank-conditioned generation follows R19 ``arm_b_generate``
  (``s19_language_organ_core.py:723-802``) with ``inputs_embeds`` for the
  first pass and KV-cache continuation, under ``inference_mode`` and
  ``temperature=0`` only.
* Hashing adapts ``s19_language_organ_core.py:108-185``: model ID,
  canonical config, tokenizer vocab/special tokens, and every named
  parameter's raw bytes.
* No fake fallback: CLI load errors propagate as :class:`FrozenOrganError`.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F

from oczy.lm.hf_driver import HFDriver

from .contracts import DEFAULT_FEATURE_DIM, DialogueMessage

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FrozenOrganError(Exception):
    """Raised on fail-closed load/tokenization/hash/freeze failures.

    The CLI catches this to exit nonzero without substituting a fake
    organ or organ-only mode.
    """


# ---------------------------------------------------------------------------
# Offline-aware model load-target resolution
# ---------------------------------------------------------------------------


def _resolve_load_target(model_id: str) -> str:
    """Resolve the path/id to pass to ``HFDriver.load``.

    Resolution order:
      1. ``OCZY_MODEL_DIR`` (if set and points to a non-empty directory)
      2. ``OCZY_HF_MODEL_DIR`` (if set and points to a non-empty directory)
      3. ``model_id`` — but ONLY outside remote/offline mode.

    A non-empty directory check (``os.listdir``) rejects arbitrary empty
    paths that would otherwise be accepted as a valid mount point.

    Under ``OCZY_REMOTE_CPU_ONLY=1``, ``HF_HUB_OFFLINE=1``, or
    ``TRANSFORMERS_OFFLINE=1``, the hub-id fallback is forbidden: if neither
    env var resolves to a local directory, a :class:`FrozenOrganError` is
    raised with an actionable ASI message *before* any HuggingFace call —
    preventing the ``LocalEntryNotFoundError`` that occurs when a hub id is
    passed to ``from_pretrained`` with ``local_files_only=True``.

    The caller's ``model_id`` is preserved separately for organ identity
    (hashing/provenance) and is NOT mutated here; only the load target is
    returned.
    """
    offline = (
        os.environ.get("OCZY_REMOTE_CPU_ONLY") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )

    for env_var in ("OCZY_MODEL_DIR", "OCZY_HF_MODEL_DIR"):
        env_dir = os.environ.get(env_var)
        if env_dir and os.path.isdir(env_dir) and os.listdir(env_dir):
            return env_dir

    if offline:
        raise FrozenOrganError(
            f"ASI offline_model_unavailable: OCZY_REMOTE_CPU_ONLY/"
            f"HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is active but no local "
            f"model directory was found. Set OCZY_MODEL_DIR or "
            f"OCZY_HF_MODEL_DIR to an existing snapshot directory for "
            f"pinned model_id={model_id!r}. Network fallback is disabled "
            f"in offline/remote mode."
        )

    return model_id

# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------


def render_chat(messages: Sequence[DialogueMessage], tokenizer: Any) -> str:
    """Render ``messages`` through the tokenizer's chat template.

    A single fixed chat-template path is used so that the rendered prompt
    is deterministic for a given tokenizer + message sequence.  The
    template descriptor (see :func:`_template_descriptor`) is hashable
    and included in :meth:`QwenFrozenOrgan.parameter_hash`.

    Args:
        messages: Sequence of :class:`DialogueMessage` (role, content).
        tokenizer: A Hugging Face tokenizer with ``apply_chat_template``.

    Returns:
        The rendered prompt string.

    Raises:
        FrozenOrganError: If the tokenizer lacks a chat template or
            ``apply_chat_template`` fails.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise FrozenOrganError(
            "tokenizer has no apply_chat_template; cannot render chat"
        )
    raw = [
        {"role": msg.role, "content": msg.content} for msg in messages
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            raw,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise FrozenOrganError(
            f"apply_chat_template failed: {exc}"
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise FrozenOrganError(
            "apply_chat_template returned empty or non-string output"
        )
    return rendered


def _template_descriptor(tokenizer: Any) -> str:
    """Return a hashable descriptor for the tokenizer's chat template.

    This captures the template source text (or its repr) so that a
    template change is detectable in the organ hash.
    """
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        if tmpl is None:
            return "none"
        if isinstance(tmpl, str):
            return tmpl
        return repr(tmpl)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FrozenLanguageOrgan(Protocol):
    """Protocol for a frozen language organ backing the meta-cortex.

    Implementations must:

    * Expose ``feature_dim`` matching the cortex's ``ModelConfig.feature_dim``.
    * Encode texts to final-layer mean-pooled ``[N, D]`` float32 features
      without caching raw text.
    * Provide teacher-forced bank-conditioned loss and specificity KL that
      allow gradients to flow to ``soft_bank`` but never to organ parameters.
    * Provide deterministic greedy generation conditioned only on messages
      and the soft bank — never on target/correction/task-ID.
    * Hash and assert frozen state.
    """

    feature_dim: int

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Return ``[N, feature_dim]`` mean-pooled final-layer features.

        Runs under inference mode; output is detached float32.  No raw
        text or embedding cache is retained.
        """
        ...

    def teacher_forced_logits(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[T, V]`` logits at target token positions.

        ``soft_bank`` is ``[B, L, D]`` with ``B == 1``.  LM parameters are
        frozen but autograd is enabled so gradients flow to ``soft_bank``.
        """
        ...

    def teacher_forced_loss(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar cross-entropy on target positions."""
        ...

    def specificity_kl(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
        reference_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scalar KL divergence at target positions.

        Computes ``KL(bank-conditioned || reference)`` where ``reference``
        is either ``reference_bank`` or the organ-only (no-bank) logits.
        Used to penalize non-specific bank influence on unrelated tasks.
        """
        ...

    def generate(
        self,
        messages: Sequence[DialogueMessage],
        soft_bank: torch.Tensor,
        max_new_tokens: int,
    ) -> str:
        """Deterministic greedy generation conditioned on messages + bank.

        No target, correction, or task-ID argument.  Runs under inference
        mode with ``temperature=0`` only.
        """
        ...

    def parameter_hash(self) -> str:
        """Return SHA-256 over model ID, config, tokenizer, and weights."""
        ...

    def assert_frozen(self) -> None:
        """Raise :class:`FrozenOrganError` if any LM param has requires_grad."""
        ...

    def close(self) -> None:
        """Release the underlying driver/model resources."""
        ...


# ---------------------------------------------------------------------------
# Real Qwen adapter
# ---------------------------------------------------------------------------


class QwenFrozenOrgan:
    """Real :class:`HFDriver`-backed frozen language organ.

    Loads ``Qwen/Qwen2.5-0.5B-Instruct`` (or a compatible model) through
    :meth:`HFDriver.load`, freezes every LM parameter, and provides
    no-cache feature extraction, teacher-forced bank-conditioned loss,
    specificity KL, and deterministic greedy generation.

    The adapter owns **no** experience-text/token/embedding cache.  All
    feature extraction is done directly under ``torch.inference_mode()``
    rather than via ``HFDriver.peek_embedding`` (which caches raw prompts).
    """

    def __init__(
        self,
        driver: HFDriver,
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        """Wrap an existing :class:`HFDriver`.

        Args:
            driver: A loaded HFDriver with model + tokenizer.
            feature_dim: Expected hidden size; must match
                ``driver.n_embd``.

        Raises:
            FrozenOrganError: If ``feature_dim`` does not match the
                model's hidden size, or if freezing fails.
        """
        if int(driver.n_embd) != int(feature_dim):
            raise FrozenOrganError(
                f"feature_dim mismatch: expected {feature_dim}, "
                f"got driver.n_embd={driver.n_embd}"
            )
        self._driver = driver
        self._model = driver._model
        self._tokenizer = driver._tokenizer
        self._model_id = driver.model_id
        self.feature_dim: int = int(feature_dim)
        self._vocab_size: int = int(driver.n_vocab)
        self._closed: bool = False

        # Freeze every LM parameter and verify.
        self._model.eval()
        self._freeze_all_parameters()
        self.assert_frozen()

        # Cache the frozen parameter hash at construction time so we can
        # detect any mutation after training/validation.  This is a hash
        # of model weights/config/tokenizer — not of experience data.
        self._initial_hash = self.parameter_hash()

    @classmethod
    def load(
        cls,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> QwenFrozenOrgan:
        """Load a model through :meth:`HFDriver.load` and wrap it.

        Raises:
            FrozenOrganError: If the model cannot be loaded, the hidden
                size mismatches, or freezing fails.  Never falls back to
                a fake or organ-only mode.
        """
        load_target = _resolve_load_target(model_id)
        try:
            driver = HFDriver.load(model_id=load_target)
        except Exception as exc:
            raise FrozenOrganError(
                f"Failed to load frozen language organ '{model_id}' "
                f"(load_target={load_target!r}): {exc}"
            ) from exc
        organ = cls(driver, feature_dim=feature_dim)
        # Preserve the explicit model_id as organ identity (hashing/
        # provenance) even when the actual load target was a verified
        # local directory (OCZY_MODEL_DIR/OCZY_HF_MODEL_DIR).
        organ._model_id = model_id
        return organ

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze_all_parameters(self) -> None:
        """Set every LM parameter ``requires_grad_(False)``."""
        for param in self._model.parameters():
            param.requires_grad_(False)

    def _check_open(self) -> None:
        if self._closed:
            raise FrozenOrganError("organ is closed")

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to ``[1, S]`` long tensor (add_special_tokens=True)."""
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            raise FrozenOrganError("tokenization produced empty token list")
        return torch.tensor([ids], dtype=torch.long)

    def _embed_fn(self) -> Any:
        """Return the model's input embedding function."""
        return cast(Any, self._model).get_input_embeddings()

    def _validate_soft_bank(self, soft_bank: torch.Tensor) -> None:
        """Validate soft bank shape ``[B, L, D]`` with B==1, D==feature_dim."""
        if not isinstance(soft_bank, torch.Tensor):
            raise FrozenOrganError("soft_bank must be a torch.Tensor")
        if soft_bank.dim() != 3:
            raise FrozenOrganError(
                f"soft_bank must be 3D [B, L, D], got shape {tuple(soft_bank.shape)}"
            )
        b, _l, d = soft_bank.shape
        if b != 1:
            raise FrozenOrganError(
                f"soft_bank batch must be 1, got {b}"
            )
        if d != self.feature_dim:
            raise FrozenOrganError(
                f"soft_bank feature dim must be {self.feature_dim}, got {d}"
            )
        if not torch.isfinite(soft_bank).all():
            raise FrozenOrganError("soft_bank contains non-finite values")

    def _build_teacher_forced_embeds(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, list[int]]:
        """Build ``inputs_embeds`` for teacher forcing.

        Returns ``(inputs_embeds, prompt_len, n_answer, answer_ids)`` where
        ``prompt_len`` is the number of prompt tokens (including bank) and
        ``n_answer`` is the number of target tokens.

        The soft bank is prepended: ``[bank, prompt_embeds, answer_embeds]``.
        Logits at position ``prompt_len - 1`` predict the first answer token.
        """
        # Render the chat prompt (with generation prompt).
        prompt_text = render_chat(messages, self._tokenizer)

        # Tokenize prompt and target.
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=True)
        if not prompt_ids:
            raise FrozenOrganError("prompt tokenization produced empty list")

        # Target: prepend a space to match typical continuation tokenization
        # (R19 uses " " + corrected_response).  This is the standard HF
        # convention for answer continuation after a generation prompt.
        target_text = " " + target.lstrip()
        answer_ids = self._tokenizer.encode(target_text, add_special_tokens=False)
        if not answer_ids:
            raise FrozenOrganError(
                "target tokenization produced empty token list; "
                "cannot teacher-force an empty target"
            )

        # Verify prompt IDs are an exact prefix of full-chat IDs.
        # Full chat = prompt_text + target_text tokenized together should
        # start with prompt_ids.  We verify by checking that the prompt
        # token IDs are a prefix of (prompt_ids + answer_ids) — which is
        # always true by construction, but we also verify the template
        # did not re-tokenize differently when target is appended.
        full_text = prompt_text + target_text
        full_ids = self._tokenizer.encode(full_text, add_special_tokens=True)
        if not full_ids[: len(prompt_ids)] == prompt_ids:
            raise FrozenOrganError(
                "chat template mismatch: prompt IDs are not an exact prefix "
                "of full-chat IDs; target may have been absorbed into the "
                "template"
            )

        embed_fn = self._embed_fn()

        # Prompt embeddings (frozen).
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_embeds = embed_fn(prompt_tensor)  # [1, S, D]

        # Answer embeddings (frozen, for teacher forcing).
        answer_tensor = torch.tensor([answer_ids], dtype=torch.long)
        answer_embeds = embed_fn(answer_tensor)  # [1, A, D]

        # Prepend soft bank: [bank, prompt, answer]
        # soft_bank is [1, L, D] — keep it as-is to preserve gradient flow.
        inputs_embeds = torch.cat(
            [soft_bank, prompt_embeds, answer_embeds],
            dim=1,
        )  # [1, L+S+A, D]

        prompt_len = soft_bank.shape[1] + len(prompt_ids)
        n_answer = len(answer_ids)

        return inputs_embeds, prompt_len, n_answer, answer_ids

    def _forward_teacher_forced(
        self,
        inputs_embeds: torch.Tensor,
        prompt_len: int,
        n_answer: int,
    ) -> torch.Tensor:
        """Forward pass and extract ``[T, V]`` logits at target positions.

        LM parameters are frozen but autograd is enabled so gradients flow
        to ``inputs_embeds`` (and thus to ``soft_bank``).
        """
        # Do NOT use torch.no_grad() or inference_mode here — we need
        # gradients to flow through the frozen LM to the soft bank.
        out = self._model(inputs_embeds=inputs_embeds, use_cache=False)
        # Logits at position (prompt_len - 1) predict the first answer token.
        start = prompt_len - 1
        end = start + n_answer
        logits = out.logits[0, start:end, :]  # [n_answer, V]
        return logits

    # ------------------------------------------------------------------
    # Protocol: encode_texts
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Return ``[N, feature_dim]`` mean-pooled final-layer features.

        Extracts final-layer hidden states directly (not via
        ``HFDriver.peek_embedding``, which caches raw prompt strings).
        Output is detached float32.  No text or embedding cache is
        retained by this adapter.
        """
        self._check_open()
        if not texts:
            raise FrozenOrganError("encode_texts received empty text sequence")

        features: list[torch.Tensor] = []
        for text in texts:
            if not isinstance(text, str):
                raise FrozenOrganError(
                    f"encode_texts expects str, got {type(text).__name__}"
                )
            input_ids = self._tokenize(text)  # [1, S]
            out = self._model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            final_hidden = out.hidden_states[-1]  # [1, S, D]
            mean_pooled = final_hidden[0].mean(dim=0)  # [D]
            features.append(mean_pooled.to(dtype=torch.float32))

        result = torch.stack(features, dim=0)  # [N, D]
        # Detach to ensure no graph is retained — feature extraction is
        # a perception path, not a training path.
        return result.detach()

    # ------------------------------------------------------------------
    # Protocol: teacher_forced_logits
    # ------------------------------------------------------------------

    def teacher_forced_logits(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[T, V]`` logits at target token positions.

        ``soft_bank`` is ``[1, L, D]``.  LM parameters are frozen but
        autograd is enabled so gradients flow to ``soft_bank``.
        """
        self._check_open()
        self._validate_soft_bank(soft_bank)
        if not messages:
            raise FrozenOrganError("teacher_forced_logits received empty messages")
        if not target or not target.strip():
            raise FrozenOrganError(
                "teacher_forced_logits received empty target; cannot "
                "teacher-force an empty target"
            )

        inputs_embeds, prompt_len, n_answer, _ = self._build_teacher_forced_embeds(
            messages, target, soft_bank
        )
        return self._forward_teacher_forced(inputs_embeds, prompt_len, n_answer)

    # ------------------------------------------------------------------
    # Protocol: teacher_forced_loss
    # ------------------------------------------------------------------

    def teacher_forced_loss(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar cross-entropy on target positions.

        Gradients flow through the frozen LM to ``soft_bank`` but never
        to LM parameters (all ``requires_grad=False``).
        """
        self._check_open()
        self._validate_soft_bank(soft_bank)
        if not messages:
            raise FrozenOrganError("teacher_forced_loss received empty messages")
        if not target or not target.strip():
            raise FrozenOrganError(
                "teacher_forced_loss received empty target; cannot "
                "teacher-force an empty target"
            )
        inputs_embeds, prompt_len, n_answer, answer_ids = (
            self._build_teacher_forced_embeds(messages, target, soft_bank)
        )
        logits = self._forward_teacher_forced(inputs_embeds, prompt_len, n_answer)
        targets = torch.tensor(answer_ids, dtype=torch.long)
        loss = F.cross_entropy(logits, targets)
        return loss

    # ------------------------------------------------------------------
    # Protocol: specificity_kl
    # ------------------------------------------------------------------

    def specificity_kl(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
        reference_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scalar KL(bank-conditioned || reference) at target positions.

        When ``reference_bank`` is provided, it replaces ``soft_bank`` for
        the reference distribution.  When ``None``, the reference is the
        organ-only (no-bank) forward pass.

        The KL direction is ``KL(bank || reference)``: we want the
        bank-conditioned distribution to be *close* to the reference on
        unrelated (specificity) tasks, penalizing non-specific influence.
        """
        self._check_open()
        self._validate_soft_bank(soft_bank)
        if not messages:
            raise FrozenOrganError("specificity_kl received empty messages")
        if not target or not target.strip():
            raise FrozenOrganError(
                "specificity_kl received empty target"
            )

        # Bank-conditioned logits (gradients flow to soft_bank).
        bank_logits = self.teacher_forced_logits(messages, target, soft_bank)
        bank_log_probs = F.log_softmax(bank_logits, dim=-1)

        # Reference logits (no gradient to soft_bank).
        with torch.no_grad():
            if reference_bank is not None:
                self._validate_soft_bank(reference_bank)
                ref_logits = self.teacher_forced_logits(
                    messages, target, reference_bank
                )
            else:
                # Organ-only: no bank prepended.
                ref_logits = self._organ_only_logits(messages, target)
            ref_probs = F.softmax(ref_logits, dim=-1)

        # KL(bank || ref) = sum bank_probs * (log bank_probs - log ref_probs).
        # This penalizes the bank-conditioned distribution for diverging
        # from the reference on unrelated (specificity) tasks.
        bank_probs = bank_log_probs.exp()
        # Clamp reference probabilities to avoid log(0) = -inf.
        ref_log_probs = ref_probs.clamp(min=1e-12).log()
        kl = (bank_probs * (bank_log_probs - ref_log_probs)).sum(dim=-1)
        return kl.mean()

    def _organ_only_logits(
        self,
        messages: Sequence[DialogueMessage],
        target: str,
    ) -> torch.Tensor:
        """Return ``[T, V]`` logits without any soft bank prepended.

        Used as the reference for :meth:`specificity_kl`.  No gradient
        flows to any bank.
        """
        prompt_text = render_chat(messages, self._tokenizer)
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=True)
        if not prompt_ids:
            raise FrozenOrganError("prompt tokenization produced empty list")

        target_text = " " + target.lstrip()
        answer_ids = self._tokenizer.encode(target_text, add_special_tokens=False)
        if not answer_ids:
            raise FrozenOrganError(
                "target tokenization produced empty token list"
            )

        embed_fn = self._embed_fn()
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_embeds = embed_fn(prompt_tensor)  # [1, S, D]
        answer_tensor = torch.tensor([answer_ids], dtype=torch.long)
        answer_embeds = embed_fn(answer_tensor)  # [1, A, D]

        inputs_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)
        prompt_len = len(prompt_ids)
        n_answer = len(answer_ids)

        with torch.no_grad():
            out = self._model(inputs_embeds=inputs_embeds, use_cache=False)
            start = prompt_len - 1
            end = start + n_answer
            logits = out.logits[0, start:end, :]
        return logits

    # ------------------------------------------------------------------
    # Protocol: generate
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        messages: Sequence[DialogueMessage],
        soft_bank: torch.Tensor,
        max_new_tokens: int,
    ) -> str:
        """Deterministic greedy generation conditioned on messages + bank.

        No target, correction, or task-ID argument.  Uses ``inputs_embeds``
        for the first pass (bank + prompt) and KV-cache continuation.
        Runs under inference mode with ``temperature=0`` only.
        """
        self._check_open()
        self._validate_soft_bank(soft_bank)
        if not messages:
            raise FrozenOrganError("generate received empty messages")
        if max_new_tokens <= 0:
            raise FrozenOrganError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )

        prompt_text = render_chat(messages, self._tokenizer)
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=True)
        if not prompt_ids:
            raise FrozenOrganError("prompt tokenization produced empty list")

        embed_fn = self._embed_fn()
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_embeds = embed_fn(prompt_tensor)  # [1, S, D]

        # Prepend soft bank: [bank, prompt]
        inputs_embeds = torch.cat([soft_bank, prompt_embeds], dim=1)

        # First forward pass with KV cache.
        out = self._model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        next_token = int(out.logits[0, -1, :].argmax().item())

        generated_ids: list[int] = [next_token]
        eos_id = self._tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            if next_token == eos_id:
                break
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            out = self._model(
                input_ids=token_tensor,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = int(out.logits[0, -1, :].argmax().item())
            generated_ids.append(next_token)

        if not generated_ids:
            return ""
        return self._tokenizer.decode(generated_ids)

    # ------------------------------------------------------------------
    # Protocol: parameter_hash
    # ------------------------------------------------------------------

    def parameter_hash(self) -> str:
        """Return SHA-256 over model ID, config, tokenizer, and weights.

        Adapts ``s19_language_organ_core.py:108-185``.  This is a frozen
        identity hash — it must not change across training/validation.
        """
        self._check_open()

        cfg = getattr(self._model, "config", None)
        cfg_dict: dict[str, Any]
        if cfg is not None:
            cfg_dict = (
                cfg.to_dict() if hasattr(cfg, "to_dict") else {"repr": repr(cfg)}
            )
        else:
            cfg_dict = {}

        # Hash every byte of every named parameter tensor (bit-identical check).
        param_hashes: list[str] = []
        for name, param in sorted(self._model.named_parameters()):
            data = param.detach().cpu().numpy().tobytes()
            param_hashes.append(f"{name}:{hashlib.sha256(data).hexdigest()}")

        # Hash the tokenizer vocab and special tokens.
        tokenizer = self._tokenizer
        try:
            vocab_items = sorted(tokenizer.get_vocab().items())
            vocab_payload = json.dumps(vocab_items, default=str)
            tokenizer_hash = hashlib.sha256(
                vocab_payload.encode()
            ).hexdigest()
        except Exception:
            tokenizer_hash = hashlib.sha256(repr(tokenizer).encode()).hexdigest()

        # Include the chat template descriptor.
        template_desc = _template_descriptor(tokenizer)
        template_hash = hashlib.sha256(template_desc.encode()).hexdigest()

        payload = json.dumps(
            {
                "model_id": self._model_id,
                "config": cfg_dict,
                "param_hashes": param_hashes,
                "tokenizer_hash": tokenizer_hash,
                "template_hash": template_hash,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Protocol: assert_frozen
    # ------------------------------------------------------------------

    def assert_frozen(self) -> None:
        """Raise :class:`FrozenOrganError` if any LM param has requires_grad.

        Also verifies the model is in eval mode.
        """
        self._check_open()
        unfrozen: list[str] = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                unfrozen.append(name)
        if unfrozen:
            raise FrozenOrganError(
                f"{len(unfrozen)} LM parameter(s) have requires_grad=True: "
                f"{unfrozen[:5]}{'...' if len(unfrozen) > 5 else ''}"
            )
        if not self._model.training:
            return
        raise FrozenOrganError(
            "LM model is in training mode; expected eval mode"
        )

    # ------------------------------------------------------------------
    # Protocol: close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying driver/model resources.

        After close, any method call raises :class:`FrozenOrganError`.
        """
        if self._closed:
            return
        self._closed = True
        # Clear references to allow GC of the large model.
        self._model = None  # type: ignore[assignment]
        self._tokenizer = None  # type: ignore[assignment]
        self._driver = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Audit helpers (not part of the protocol but used by CLI/training)
    # ------------------------------------------------------------------

    def initial_hash(self) -> str:
        """Return the parameter hash recorded at construction time.

        This is the baseline for before/after training comparison.
        """
        return self._initial_hash

    def organ_parameter_ids(self) -> set[int]:
        """Return the set of ``id()`` of all LM parameters.

        Used by the outer trainer to verify optimizer parameter-ID
        disjointness — no LM parameter may appear in the cortex optimizer.
        """
        self._check_open()
        return {id(param) for param in self._model.parameters()}

    @property
    def model_id(self) -> str:
        """Return the model ID string."""
        return self._model_id

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size
