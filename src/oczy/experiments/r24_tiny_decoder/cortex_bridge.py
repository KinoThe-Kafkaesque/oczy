"""Phase C cortex integration: r = R_theta(F,S,query) -> frozen decoder -> answer

Uses existing MetaCortex (write/consolidate/read) which already outputs r[64].
Replaces the Qwen organ path: instead of
  soft_bank = coupler(r) -> Qwen(inputs_embeds) -> logits
we do
  decoder(query_tokens, r) -> logits   where decoder is frozen.

Gradients may pass through frozen decoder into cortex+readout during
developmental training; decoder params are excluded from optimizer.
During unseen-task evaluation only F/S may change (no optimizer).

Includes causal interventions per Controls 4-8.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from oczy.experiments.meta_cortex.model import MetaCortex, CortexState
from oczy.experiments.meta_cortex.contracts import CORTEX_DIM

from .decoder import TinySharedDecoder
from .vocab import PAD_ID, encode_bytes, encode_with_eos

__all__ = ["CortexDecoderBridge"]

class CortexDecoderBridge:
    """Bridges MetaCortex state + query text -> frozen decoder loss / generation."""

    def __init__(self, cortex: MetaCortex, decoder: TinySharedDecoder, device: torch.device):
        self.cortex = cortex
        self.decoder = decoder
        self.device = device
        # freeze decoder explicitly
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

    def _query_to_tokens(self, query_text: str) -> torch.Tensor:
        ids = encode_bytes(query_text)
        # guard max_len
        assert len(ids) < self.decoder.config.max_len
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,Tq]

    def _answer_to_tokens(self, answer_text: str) -> torch.Tensor:
        ids = encode_with_eos(answer_text)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,Ta]

    def _read_r(self, state: CortexState, query_text: str) -> torch.Tensor:
        """Cortex read -> r[64] using dummy feature dim 896 pooled hash as placeholder.
        In real integration, HFDriver.encode_texts would give 896-d feature.
        For tiny POC we hash query text -> 896-d random projection seed, deterministic.
        """
        # deterministic query feature: hash bytes -> 896 vector via seeded RNG
        import hashlib
        h = int(hashlib.sha256(query_text.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device=self.device)
        g.manual_seed(h % (2**31))
        feat = torch.randn((1, self.cortex.config.feature_dim), generator=g, device=self.device)
        r = self.cortex.read(state, feat)  # [1,64]
        return r

    def loss_for_state(self, state: CortexState, query_text: str, answer_text: str) -> torch.Tensor:
        q = self._query_to_tokens(query_text)
        a = self._answer_to_tokens(answer_text)
        r = self._read_r(state, query_text)  # [1,64], grad flows to cortex via read
        # decoder forward with requires_grad on r but not decoder
        loss = self.decoder.teacher_forced_loss(q, r, a)
        return loss

    def generate(self, state: CortexState, query_text: str, max_new_tokens: int = 32) -> str:
        q = self._query_to_tokens(query_text)
        r = self._read_r(state, query_text)
        with torch.inference_mode():
            gen_ids = self.decoder.generate_greedy(q, r, max_new_tokens=max_new_tokens)[0].tolist()
            from .vocab import decode_bytes, EOS_ID
            if EOS_ID in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(EOS_ID)]
            return decode_bytes(gen_ids)

    # ---- causal interventions ----
    def zeroed_state_loss(self, state: CortexState, query_text: str, answer_text: str) -> torch.Tensor:
        zero = self.cortex.zero_state(state)
        return self.loss_for_state(zero, query_text, answer_text)

    def swapped_state_loss(self, state: CortexState, donor: CortexState, query_text: str, answer_text: str) -> torch.Tensor:
        swapped = self.cortex.swap_state(state, donor)
        return self.loss_for_state(swapped, query_text, answer_text)
