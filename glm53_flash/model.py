"""Readable scaled GLM-5.3-Flash-style hybrid-attention MoE model.

This is an educational scale model inspired by GLM-5.3-Flash. It is not a
drop-in reproduction of the released frontier model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    dim: int = 192
    layers: int = 12
    heads: int = 6
    expert_hidden: int = 384
    experts: int = 8
    top_k: int = 2
    streams: int = 4
    sparse_window: int = 32
    sparse_stride: int = 32
    max_sequence_length: int = 192

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to [batch, time, heads, head_dim]."""
    length, width = q.shape[1], q.shape[-1]
    if width % 2:
        raise ValueError("head dimension must be even for RoPE")
    positions = torch.arange(length, device=q.device, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (torch.arange(0, width, 2, device=q.device).float() / width))
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, :, None, :].to(q.dtype)
    sin = angles.sin()[None, :, None, :].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class LinearAttention(nn.Module):
    """Causal positive-feature linear attention with recurrent prefix state."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rope(q, k)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        v = v.view(batch, length, self.heads, self.head_dim)
        kv = torch.einsum("bthd,bthe->bthde", k, v).cumsum(dim=1)
        k_prefix = k.cumsum(dim=1)
        numerator = torch.einsum("bthd,bthde->bthe", q, kv)
        denominator = torch.einsum("bthd,bthd->bth", q, k_prefix).unsqueeze(-1)
        output = numerator / denominator.clamp_min(1e-6)
        return self.out(output.reshape(batch, length, dim))


class SparseAttention(nn.Module):
    """True gathered causal attention over a local window plus strided anchors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.window = config.sparse_window
        self.stride = config.sparse_stride
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def _indices(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        for position in range(length):
            anchors = list(range(0, position + 1, self.stride))
            local = list(range(max(0, position - self.window + 1), position + 1))
            rows.append(sorted(set(anchors + local)))
        width = max(map(len, rows))
        indices = torch.zeros((length, width), dtype=torch.long, device=device)
        valid = torch.zeros((length, width), dtype=torch.bool, device=device)
        for row, values in enumerate(rows):
            indices[row, : len(values)] = torch.tensor(values, device=device)
            valid[row, : len(values)] = True
        return indices, valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        indices, valid = self._indices(length, x.device)
        gathered_k = k[:, :, indices, :]
        gathered_v = v[:, :, indices, :]
        scores = torch.einsum("bhtd,bhtkd->bhtk", q, gathered_k) * self.head_dim**-0.5
        scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        output = torch.einsum("bhtk,bhtkd->bhtd", weights, gathered_v)
        return self.out(output.transpose(1, 2).reshape(batch, length, dim))


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class SparseMoE(nn.Module):
    """Top-k routed experts plus one always-active shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Linear(config.dim, config.experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(config.dim, config.expert_hidden) for _ in range(config.experts)]
        )
        self.shared = Expert(config.dim, config.expert_hidden)
        self.top_k = config.top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        logits = self.router(flat)
        top_values, top_indices = logits.topk(self.top_k, dim=-1)
        top_weights = torch.softmax(top_values.float(), dim=-1).to(flat.dtype)
        output = self.shared(flat)
        usage = torch.zeros(len(self.experts), device=x.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_id)
            if positions.numel() == 0:
                continue
            contribution = expert(flat[positions])
            contribution = contribution * top_weights[positions, slots, None].to(contribution.dtype)
            output.index_add_(0, positions, contribution)
            usage[expert_id] = positions.numel()
        usage = usage / max(1, flat.shape[0] * self.top_k)
        return output.view(original_shape), usage


class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = SparseAttention(config) if sparse else LinearAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x))
        ffn, usage = self.moe(self.ffn_norm(x))
        return x + ffn, usage


class HyperConnection(nn.Module):
    """Four residual streams with learned input mixing and output routing."""

    def __init__(self, config: ModelConfig, block: HybridBlock):
        super().__init__()
        self.block = block
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams))
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams))

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = torch.einsum(
            "s,btsd->btd", torch.softmax(self.input_logits, dim=0), streams
        )
        transformed, usage = self.block(mixed)
        routed = torch.softmax(self.output_logits, dim=0)
        streams = streams + transformed.unsqueeze(2) * routed[None, None, :, None]
        return streams, usage


class GLM53FlashFromScratch(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.heads:
            raise ValueError("dim must be divisible by heads")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            HyperConnection(config, HybridBlock(config, sparse=((index + 1) % 4 == 0)))
            for index in range(config.layers)
        ])
        self.final_norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_embeddings(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the language model from precomputed token-like embeddings."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded inputs must have shape [batch, time, dim]")
        if embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("embedded input exceeds max_sequence_length")
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.streams, -1).contiguous()
        usages = []
        for layer in self.layers:
            streams, usage = layer(streams)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2))
        return self.output(hidden), torch.stack(usages)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embedding(input_ids))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        # Active count is an architectural estimate: all non-routed parameters,
        # the shared expert, and top-k of routed expert parameters per layer.
        routed_total = sum(
            sum(parameter.numel() for expert in layer.block.moe.experts for parameter in expert.parameters())
            for layer in self.layers
        )
        active = total - routed_total + routed_total * self.config.top_k // self.config.experts
        return {"total": total, "active_per_token_estimate": active}
