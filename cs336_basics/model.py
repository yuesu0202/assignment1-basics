import math

import torch
import torch.nn as nn
from einops import einsum, rearrange


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
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device = device, dtype = dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
           
        return result.to(in_dtype)
    

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            # ~8/3 * d_model, rounded to a multiple of 64 for hardware efficiency
            d_ff = max(64, round(8 * d_model / 3 / 64) * 64)
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)   # gate branch
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)   # value branch
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)   # project back
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))       
        value = self.w3(x)            
        return self.w2(gate * value)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even to form pairs"
        self.d_k = d_k

        # Per-pair frequency: 1 / theta^((2k-2)/d_k) for k = 1..d_k/2  ->  shape (d_k/2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)   # (max_seq_len,)
        angles = torch.outer(positions, inv_freq)                                   # (max_seq_len, d_k/2)

        # Fixed values, not learned, and not saved in state_dict
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions]          # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]

        # If x has extra batch-like dims (e.g. heads), add size-1 axes before seq so they broadcast:
        # positions (batch, seq) vs x (batch, heads, seq, d_k) -> cos (batch, 1, seq, d_k/2)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2].float()            # first element of each pair
        x_odd = x[..., 1::2].float()             # second element of each pair

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave back: [e0, o0, e1, o1, ...]
        out = rearrange(torch.stack([out_even, out_odd], dim=-1), "... half two -> ... (half two)")
        return out.to(x.dtype)
