"""R24 Phase-A v2 oracle encoders.

The v1 encoder mean-pooled every byte in a complete rule.  V2 keeps the same
shared 64-dimensional contract but exposes pre-registered pooling ablations:
mean, a learned CLS token, learned-query attention, and generic line-set
attention.  None of the variants sees the probe query, so one rule state still
has to serve every query for that rule.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .vocab import PAD_ID, VOCAB_SIZE

OraclePooling = Literal["mean", "cls", "attention", "line_attention"]


class LearnedRuleOracleEncoder(nn.Module):
    """Train-only ceiling: one learned shared state per known rule fingerprint."""

    def __init__(self, rule_fingerprints: list[str], r_dim: int = 64) -> None:
        super().__init__()
        ordered = sorted(set(rule_fingerprints))
        if not ordered:
            raise ValueError("at least one rule fingerprint is required")
        self.rule_to_index = {fingerprint: index for index, fingerprint in enumerate(ordered)}
        self.embedding = nn.Embedding(len(ordered), r_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, rule_fingerprints: list[str]) -> torch.Tensor:
        try:
            indices = [self.rule_to_index[fingerprint] for fingerprint in rule_fingerprints]
        except KeyError as error:
            raise ValueError(
                "learned-rule oracle cannot evaluate unseen rules; use only as a train-rule ceiling"
            ) from error
        device = self.embedding.weight.device
        return self.embedding(torch.tensor(indices, dtype=torch.long, device=device))


class PoolingOracleEncoder(nn.Module):
    """Encode complete rule text into one bounded ``r[64]`` vector."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        r_dim: int = 64,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 512,
        pooling: OraclePooling = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls", "attention", "line_attention"}:
            raise ValueError(f"unsupported oracle pooling: {pooling}")
        self.pooling = pooling
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        # One extra position is available for the learned CLS token.
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        if pooling == "cls":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.pool_query, std=0.02)
        elif pooling in {"attention", "line_attention"}:
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.pool_query, std=0.02)
            self.pool_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        self.proj = nn.Linear(d_model, r_dim)

    def _transform(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"oracle length {seq_len} exceeds max_len={self.max_len}")
        if self.pooling == "cls":
            cls = self.pool_query.expand(batch_size, -1, -1)
            positions = torch.arange(seq_len + 1, device=input_ids.device)
            token_x = self.tok_emb(input_ids)
            x = torch.cat([cls, token_x], dim=1) + self.pos_emb(positions).unsqueeze(0)
            valid = torch.cat(
                [torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device), mask.bool()],
                dim=1,
            )
        else:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = self.tok_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)
            valid = mask.bool()
        key_padding_mask = ~valid
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x, valid

    @staticmethod
    def _mean_pool(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _attention_pool(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        query = self.pool_query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool_attention(query, x, x, key_padding_mask=~valid, need_weights=False)
        return pooled[:, 0]

    def _line_pool(
        self, x: torch.Tensor, valid: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool bytes within each line without family-specific parsing."""
        # Newline is byte value 10.  Each byte belongs to the line preceding it.
        line_ids = ((input_ids == 10) & valid).long().cumsum(dim=1)
        max_lines = int(line_ids.masked_fill(~valid, 0).max().item()) + 1
        batch_size, _, width = x.shape
        lines = x.new_zeros((batch_size, max_lines, width))
        counts = x.new_zeros((batch_size, max_lines, 1))
        scatter_idx = line_ids.unsqueeze(-1).expand(-1, -1, width)
        lines.scatter_add_(1, scatter_idx, x * valid.unsqueeze(-1).to(x.dtype))
        counts.scatter_add_(1, line_ids.unsqueeze(-1), valid.unsqueeze(-1).to(x.dtype))
        line_valid = counts[..., 0] > 0
        lines = lines / counts.clamp_min(1.0)
        return lines, line_valid

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = input_ids != PAD_ID
        x, valid = self._transform(input_ids, mask)
        if self.pooling == "mean":
            pooled = self._mean_pool(x, valid)
        elif self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "attention":
            pooled = self._attention_pool(x, valid)
        else:
            # CLS is never used in this branch, so x aligns with input_ids.
            lines, line_valid = self._line_pool(x, valid, input_ids)
            pooled = self._attention_pool(lines, line_valid)
        return torch.tanh(self.proj(pooled))
