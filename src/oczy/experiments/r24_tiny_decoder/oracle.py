"""Oracle rule encoder for Phase A.

Per proposal:
  complete rule/examples -> oracle rule encoder -> z*[64]
  query tokens + z*[64] -> decoder -> expected answer

Requirements:
  - same z* must serve multiple inputs from one rule (per-rule embedding or encoder pooling)
  - complete rules, not paraphrases, split across train/DEV (enforced by taskgen split_audit)
  - no meta-test enters corpus
  - include same-query/different-rule pairs

We implement two encoders:
  1) PerRuleEmbedding: Embedding[hash(rule_fingerprint) -> 64] — simplest, satisfies all requirements,
     trains jointly with decoder via decoder CE; guarantees multi-input sharing.
  2) TextEncoder: small Transformer pooling over oracle_context text -> 64.

The per-rule embedding is the default for Phase A: it is the cleanest oracle supervision,
directly giving z* without distillation. After Phase A, discard encoder and use decoder alone.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import torch
import torch.nn as nn

from oczy.experiments.meta_cortex.contracts import MetaTask, CORTEX_DIM

__all__ = ["PerRuleOracleEncoder", "TextOracleEncoder"]

class PerRuleOracleEncoder(nn.Module):
    """Embedding lookup keyed by rule_fingerprint."""

    def __init__(self, rule_to_idx: Dict[str,int], r_dim: int = CORTEX_DIM, init_scale: float = 0.02):
        super().__init__()
        self.rule_to_idx = rule_to_idx
        self.r_dim = r_dim
        self.emb = nn.Embedding(len(rule_to_idx), r_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=init_scale)

    def forward(self, rule_fingerprints: list[str]) -> torch.Tensor:
        # rule_fingerprints: list len B
        idx = torch.tensor([self.rule_to_idx[fp] for fp in rule_fingerprints], dtype=torch.long, device=self.emb.weight.device)
        return self.emb(idx)  # [B,64]

    def encode_tasks(self, tasks: list[MetaTask]) -> torch.Tensor:
        fps = [t.rule_fingerprint for t in tasks]
        return self.forward(fps)

class TextOracleEncoder(nn.Module):
    """Small transformer that pools oracle_context text (byte-embedded) to z*.

    Reads oracle_context messages (complete mapping + worked examples) as byte tokens.
    """
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 2, r_dim: int = 64, vocab_size: int = 260, max_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, batch_first=True) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, r_dim)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids [B,T], mask [B,T] 1=valid
        B,T = input_ids.shape
        assert T <= self.max_len
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B,-1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for lyr in self.layers:
            x = lyr(x)
        if mask is not None:
            # mean pool over valid
            mask_f = mask.unsqueeze(-1).float()  # [B,T,1]
            summed = (x * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1)
            pooled = summed / denom
        else:
            pooled = x.mean(dim=1)
        z = torch.tanh(self.proj(pooled))  # [B,64] bounded
        return z
