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
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import EOS_ID, PAD_ID, VOCAB_SIZE

Conditioning = Literal["film", "additive", "prefix", "none"]

__all__ = ["Conditioning", "TinyDecoderConfig", "TinySharedDecoder"]


@dataclass(frozen=True, slots=True)
class TinyDecoderConfig:
    vocab_size: int = VOCAB_SIZE  # 260
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 2
    max_len: int = 512  # max query+answer len (covers catalog max 23 + query <60)
    r_dim: int = 64  # CORTEX_DIM
    conditioning: Conditioning = "film"
    deep_film: bool = False  # if True apply FiLM/additive at every layer
    n_prefix_tokens: int = 4  # projection count for prefix conditioning
    dropout: float = 0.1
    # artifact metadata
    version: str = "r24-tiny-decoder/v1"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_layers in (2, 3, 4)
        assert self.conditioning in ("film", "additive", "prefix", "none")
        assert self.n_prefix_tokens >= 1
        if self.conditioning == "prefix" and self.deep_film:
            raise ValueError("deep_film is not meaningful with prefix conditioning")


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
        # transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=4 * config.d_model,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Construct variant-specific modules after every shared parameter so paired
        # seeds yield a bit-identical backbone across conditioning ablations.
        if config.conditioning == "film":
            self.films = nn.ModuleList(
                [
                    FiLM(config.r_dim, config.d_model)
                    for _ in range(config.n_layers if config.deep_film else 1)
                ]
            )
        elif config.conditioning == "additive":
            self.additive_projs = nn.ModuleList(
                [
                    nn.Linear(config.r_dim, config.d_model)
                    for _ in range(config.n_layers if config.deep_film else 1)
                ]
            )
            for projection in self.additive_projs:
                assert isinstance(projection, nn.Linear)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        elif config.conditioning == "prefix":
            self.prefix_proj = nn.Linear(config.r_dim, config.n_prefix_tokens * config.d_model)

        # hash at init for frozen verification
        self._initial_hash = self.parameter_hash()

    # ---- conditioning helper ----
    def _apply_conditioning(self, x: torch.Tensor, r: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.config
        if cfg.conditioning in {"prefix", "none"}:
            return x
        if cfg.conditioning == "film":
            if cfg.deep_film:
                return self.films[layer_idx](x, r)
            if layer_idx == 0:
                return self.films[0](x, r)
            return x
        if cfg.deep_film:
            add = self.additive_projs[layer_idx](r).unsqueeze(1)
            return x + add
        if layer_idx == 0:
            add = self.additive_projs[0](r).unsqueeze(1)
            return x + add
        return x

    def forward(
        self,
        query_tokens: torch.Tensor,
        r: torch.Tensor,
        answer_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token-position logits while conditioning on one shared state."""
        batch_size = query_tokens.shape[0]
        if answer_tokens is not None:
            tokens = torch.cat([query_tokens, answer_tokens], dim=1)
        else:
            tokens = query_tokens
        token_count = tokens.shape[1]
        prefix_count = self.config.n_prefix_tokens if self.config.conditioning == "prefix" else 0
        total_count = token_count + prefix_count
        assert total_count <= self.config.max_len, (
            f"seq len {total_count} > max_len {self.config.max_len}"
        )
        positions = torch.arange(total_count, device=tokens.device)
        token_positions = positions[prefix_count:]
        token_x = self.tok_emb(tokens) + self.pos_emb(token_positions).unsqueeze(0)
        if prefix_count:
            prefix_x = self.prefix_proj(r).view(batch_size, prefix_count, self.config.d_model)
            prefix_x = prefix_x + self.pos_emb(positions[:prefix_count]).unsqueeze(0)
            x = torch.cat([prefix_x, token_x], dim=1)
        else:
            x = token_x

        causal = torch.triu(
            torch.ones(total_count, total_count, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for idx, layer in enumerate(self.layers):
            x = self._apply_conditioning(x, r, idx)
            x = layer(x, src_mask=causal)

        x = self.ln(x)
        logits = self.head(x)
        if prefix_count:
            logits = logits[:, prefix_count:]
        return logits

    def teacher_forced_loss_per_example(
        self,
        query_tokens: torch.Tensor,
        r: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive byte cross-entropy, reduced over tokens per example."""
        _, answer_len = answer_tokens.shape
        query_len = query_tokens.shape[1]
        full_logits = self.forward(query_tokens, r, answer_tokens)
        target_logits = full_logits[:, query_len - 1 : query_len + answer_len - 1]
        flat_loss = F.cross_entropy(
            target_logits.transpose(1, 2),
            answer_tokens,
            ignore_index=PAD_ID,
            reduction="none",
        )
        valid = answer_tokens != PAD_ID
        return (flat_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def teacher_forced_loss(
        self,
        query_tokens: torch.Tensor,
        r: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Exact autoregressive byte CE on answer tokens only."""
        return self.teacher_forced_loss_per_example(query_tokens, r, answer_tokens).mean()

    def generate_greedy(
        self, query_tokens: torch.Tensor, r: torch.Tensor, max_new_tokens: int = 32
    ) -> torch.Tensor:
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
            return tokens[:, query_tokens.shape[1] :]  # only generated

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
