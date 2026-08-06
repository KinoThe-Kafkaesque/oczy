"""Tiny shared frozen decoder for R24.

- byte vocab 260 (256 + specials)
- 2-4 layer tiny Transformer, 2 heads, d_model 64-128
- query-token attention (self-attn over query+answer)
- FiLM / additive conditioning from r[64] (CORTEX_DIM)
- exact autoregressive byte CE
- frozen artifact hash preservation
- single shared decoder: params O(1) not O(memory_units)

Conditioning ablations:
  - FiLM: x = gamma(r) * x + beta(r)      (multiplicative, per proposal + R25)
  - additive: x = x + W_proj(r)            (R02/R09/R19 baseline)

Usage:
  query_tokens [B,T_q] -> embedded
  r [B,64] -> gamma/beta -> fused at layer 0 (or each layer if deep_film=True)
  autoregressive decode for answer tokens [B,T_a]
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE

__all__ = ["TinyDecoderConfig", "TinySharedDecoder"]

@dataclass(frozen=True, slots=True)
class TinyDecoderConfig:
    vocab_size: int = VOCAB_SIZE  # 260
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 2
    max_len: int = 512  # max query+answer len (covers catalog max 23 + query <60)
    r_dim: int = 64  # CORTEX_DIM
    conditioning: Literal["film", "additive"] = "film"
    deep_film: bool = False  # if True apply FiLM at every layer
    dropout: float = 0.1
    # artifact metadata
    version: str = "r24-tiny-decoder/v1"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_layers in (2,3,4)
        assert self.conditioning in ("film","additive")

class FiLM(nn.Module):
    def __init__(self, r_dim: int, d_model: int):
        super().__init__()
        self.gamma = nn.Linear(r_dim, d_model)
        self.beta = nn.Linear(r_dim, d_model)
        # init gamma near 1, beta near 0 so zero-r is ~identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # x [B,T,D], r [B,D_r]
        g = self.gamma(r).unsqueeze(1)  # [B,1,D]
        b = self.beta(r).unsqueeze(1)
        return g * x + b

class TinySharedDecoder(nn.Module):
    """Shared frozen decoder: (query_tokens, r[64]) -> answer logits."""

    def __init__(self, config: TinyDecoderConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        # conditioning layers
        if config.conditioning == "film":
            self.films = nn.ModuleList([FiLM(config.r_dim, config.d_model) for _ in range(config.n_layers if config.deep_film else 1)])
            if config.conditioning == "film" and not config.deep_film:
                # single film at input; still wrap as ModuleList len 1 for uniformity
                pass
        else:  # additive
            self.additive_projs = nn.ModuleList([nn.Linear(config.r_dim, config.d_model) for _ in range(config.n_layers if config.deep_film else 1)])
            for p in self.additive_projs:
                nn.init.zeros_(p.weight); nn.init.zeros_(p.bias)

        # transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads, dim_feedforward=4*config.d_model, dropout=config.dropout, batch_first=True)
            for _ in range(config.n_layers)
        ])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # hash at init for frozen verification
        self._initial_hash = self.parameter_hash()

    # ---- conditioning helper ----
    def _apply_conditioning(self, x: torch.Tensor, r: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.config
        if cfg.conditioning == "film":
            if cfg.deep_film:
                return self.films[layer_idx](x, r)
            else:
                if layer_idx == 0:
                    return self.films[0](x, r)
                return x
        else:  # additive
            if cfg.deep_film:
                add = self.additive_projs[layer_idx](r).unsqueeze(1)
                return x + add
            else:
                if layer_idx == 0:
                    add = self.additive_projs[0](r).unsqueeze(1)
                    return x + add
                return x

    def forward(self, query_tokens: torch.Tensor, r: torch.Tensor, answer_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """
        query_tokens: [B,Tq] byte ids (includes BOS if desired)
        r: [B,64]
        answer_tokens: [B,Ta] for teacher forcing; if None, only query is forwarded

        Returns logits [B, Tq+Ta, vocab] or [B,Tq,vocab] if no answer.
        """
        B = query_tokens.shape[0]
        if answer_tokens is not None:
            tokens = torch.cat([query_tokens, answer_tokens], dim=1)  # [B, Tq+Ta]
        else:
            tokens = query_tokens
        T = tokens.shape[1]
        assert T <= self.config.max_len, f"seq len {T} > max_len {self.config.max_len}"
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(tokens) + self.pos_emb(pos)  # [B,T,D]

        for idx, layer in enumerate(self.layers):
            x = self._apply_conditioning(x, r, idx)
            # causal mask for autoregressive training
            causal = torch.triu(torch.ones(T,T, device=x.device, dtype=torch.bool), diagonal=1)
            x = layer(x, src_mask=causal)

        x = self.ln(x)
        logits = self.head(x)  # [B,T,V]
        return logits

    def teacher_forced_loss(self, query_tokens: torch.Tensor, r: torch.Tensor, answer_tokens: torch.Tensor) -> torch.Tensor:
        """Exact autoregressive byte CE on answer tokens only."""
        # Build input = [query, answer[:, :-1]] and target = answer
        # logits at position Tq-1 predicts answer[0], ..., Tq+Ta-2 predicts answer[Ta-1]
        B, Ta = answer_tokens.shape
        Tq = query_tokens.shape[1]
        # input answer is shifted right: drop last token for input, keep full for target
        # For Ta=1, input has no answer part
        if Ta > 1:
            answer_input = answer_tokens[:, :-1]  # [B,Ta-1]
            logits = self.forward(query_tokens, r, answer_input)  # [B, Tq+Ta-1, V]
            # logits positions Tq-1 .. Tq+Ta-2 correspond to answer[0..Ta-1]
            # But forward returns logits for all input positions; last logits predicts next token
            # So answer target aligns to logits[:, Tq:, :] when answer_input is used? Let's handle cleanly:
            # Simpler: compute full forward on [query, answer] and slice logits[:, Tq-1 : Tq+Ta-1] vs answer
            full_logits = self.forward(query_tokens, r, answer_tokens)  # [B,Tq+Ta,V]
            # use shift: logits at Tq-1 predicts answer[0]
            # PAD handling: query padded with PAD, but we slice only answer region
            # We take logits at positions Tq-1 to Tq+Ta-2 inclusive
            # Edge: if Tq==0 impossible
            target_logits = full_logits[:, Tq-1 : Tq+Ta-1, :]  # [B,Ta,V]
            loss = F.cross_entropy(target_logits.reshape(-1, self.config.vocab_size), answer_tokens.reshape(-1), ignore_index=PAD_ID)
            return loss
        else:
            # single token answer (unlikely, EOS included so Ta>=1)
            logits = self.forward(query_tokens, r)  # [B,Tq,V]
            last_logits = logits[:, -1:, :].expand(-1, Ta, -1)  # [B,1,V]
            loss = F.cross_entropy(last_logits.reshape(-1, self.config.vocab_size), answer_tokens.reshape(-1), ignore_index=PAD_ID)
            return loss

    def generate_greedy(self, query_tokens: torch.Tensor, r: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        """Greedy decode: iteratively append token."""
        self.eval()
        with torch.inference_mode():
            tokens = query_tokens  # [B,Tq]
            for _ in range(max_new_tokens):
                logits = self.forward(tokens, r)  # [B,T,V]
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
                tokens = torch.cat([tokens, nxt], dim=1)
                if (nxt == EOS_ID).all():
                    break
            return tokens[:, query_tokens.shape[1]:]  # only generated

    def parameter_hash(self) -> str:
        # canonical hash over sorted named params
        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            h.update(name.encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(str(p.dtype).encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def assert_frozen(self):
        for p in self.parameters():
            if p.requires_grad:
                raise RuntimeError("decoder not frozen: param requires_grad True")

    def freeze(self):
        pre = self.parameter_hash()
        for p in self.parameters():
            p.requires_grad = False
        frozen_hash = self.parameter_hash()
        assert frozen_hash == pre, f"hash changed on freeze: {pre[:8]}->{frozen_hash[:8]}"
        return frozen_hash
