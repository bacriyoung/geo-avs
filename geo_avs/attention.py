from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class GeoGatedCrossAttention(nn.Module):
    """Cross-attention with pairwise geometry gates.

    The gate is computed from `abs(geo_query - geo_key)` and multiplies the
    softmax attention weights. This avoids the row-scalar gate pitfall where a
    per-query gate disappears after renormalization.
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        gate_dim: int = 8,
        dropout: float = 0.0,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.renormalize = renormalize

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_value_dim, embed_dim)
        self.v_proj = nn.Linear(key_value_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        hidden = max(16, gate_dim * 2)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        geo_query: torch.Tensor,
        geo_key: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if query.ndim == 2:
            query = query[None]
            key_value = key_value[None]
            geo_query = geo_query[None]
            if geo_key is not None:
                geo_key = geo_key[None]
            squeeze_batch = True

        if geo_key is None:
            if geo_query.shape[1] == key_value.shape[1]:
                geo_key = geo_query
            else:
                geo_key = geo_query.new_zeros((geo_query.shape[0], key_value.shape[1], geo_query.shape[-1]))

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key_value))
        v = self._shape(self.v_proj(key_value))
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if attn_mask is not None:
            logits = logits.masked_fill(~attn_mask[:, None].bool(), torch.finfo(logits.dtype).min)

        attn = logits.softmax(dim=-1)
        pair_geo = (geo_query[:, :, None, :] - geo_key[:, None, :, :]).abs()
        gate = self.gate_mlp(pair_geo).squeeze(-1)
        attn = attn * gate[:, None]
        if self.renormalize:
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.embed_dim)
        out = self.out_proj(out)

        if squeeze_batch:
            out = out[0]
            attn = attn[0]
        if return_attention:
            return out, attn
        return out
