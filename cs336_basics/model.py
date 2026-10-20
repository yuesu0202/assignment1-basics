import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """y = W x, with no bias. W has shape (out_features, in_features), like nn.Linear."""

    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store W (not W^T): shape (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # N(0, 2 / (d_in + d_out)), truncated at [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in) -> (..., d_out). Works for any number of leading dims.
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    

class Embedding(nn.Module):
    """Lookup table: token ID -> d_model vector. Weight shape (num_embeddings, embedding_dim)."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # One row per token; d_model is the final dimension
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # N(0, 1), truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...) of int64  ->  (..., embedding_dim)
        return self.weight[token_ids]