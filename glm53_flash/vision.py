"""Tiny, readable GLM-5.3-Flash-style image path for the byte language model.

The production topology is preserved at teaching scale: RGB patch embedding,
non-causal vision transformer blocks, 2 x 2 spatial downsampling, a gated
multimodal projector, and placeholder replacement inside a token sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GLM53FlashFromScratch, RMSNorm


@dataclass(frozen=True)
class MiniVisionConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_size: int = 32
    depth: int = 2
    heads: int = 4
    intermediate_size: int = 64
    spatial_merge_size: int = 2
    projection_intermediate_size: int = 64
    rms_norm_eps: float = 1e-5

    @property
    def patch_grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def merged_grid_size(self) -> int:
        return self.patch_grid_size // self.spatial_merge_size

    @property
    def visual_token_count(self) -> int:
        return self.merged_grid_size**2

    def validate(self) -> None:
        if self.in_channels != 3:
            raise ValueError("the miniature vision path requires RGB input")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.patch_grid_size % self.spatial_merge_size:
            raise ValueError("patch grid must be divisible by spatial_merge_size")
        if self.hidden_size % self.heads:
            raise ValueError("vision hidden_size must be divisible by heads")
        if self.depth < 1:
            raise ValueError("vision depth must be positive")


@dataclass(frozen=True)
class ImageTokenIds:
    """Special IDs appended after the 260-token byte vocabulary."""

    start: int = 260
    placeholder: int = 261
    end: int = 262
    bos: int = 1

    @property
    def required_vocab_size(self) -> int:
        return max(self.start, self.placeholder, self.end) + 1


class RGBPatchEmbedding(nn.Module):
    """Embed non-overlapping RGB patches and add a tiny learned 2D position table."""

    def __init__(self, config: MiniVisionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position = nn.Parameter(
            torch.empty(1, config.patch_grid_size**2, config.hidden_size)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        expected = (self.config.in_channels, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape [batch, {expected[0]}, {expected[1]}, {expected[2]}]"
            )
        feature_map = self.proj(images)
        height, width = feature_map.shape[-2:]
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens + self.position, (height, width)


class VisionSelfAttention(nn.Module):
    """Full bidirectional vision attention with per-head query and key RMSNorm."""

    def __init__(self, config: MiniVisionConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden_size // config.heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, hidden_size = hidden_states.shape
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q = self.q_norm(q.view(batch, length, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, length, self.heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.proj(attention.transpose(1, 2).reshape(batch, length, hidden_size))


class VisionSwiGLU(nn.Module):
    def __init__(self, config: MiniVisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VisionTransformerBlock(nn.Module):
    """Pre-normalized non-causal attention followed by a SwiGLU MLP."""

    def __init__(self, config: MiniVisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = VisionSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VisionSwiGLU(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class SpatialMerger2x2(nn.Module):
    """Learn one output token from each non-overlapping 2 x 2 patch neighborhood."""

    def __init__(self, vision_dim: int, output_dim: int, merge_size: int = 2):
        super().__init__()
        if merge_size != 2:
            raise ValueError("this teaching merger intentionally implements 2 x 2 merging")
        self.merge_size = merge_size
        # Rearrangement plus a linear map is exactly a non-overlapping stride-2
        # convolution, while also avoiding a Conv2d backward bug in PyTorch 2.5 MPS.
        self.downsample = nn.Linear(vision_dim * merge_size**2, output_dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        batch, token_count, hidden_size = hidden_states.shape
        height, width = grid_size
        if token_count != height * width or height % 2 or width % 2:
            raise ValueError("vision tokens must form an even 2D grid")
        feature_map = hidden_states.transpose(1, 2).reshape(
            batch, hidden_size, height, width
        )
        neighborhoods = feature_map.reshape(
            batch, hidden_size, height // 2, 2, width // 2, 2
        ).permute(0, 2, 4, 1, 3, 5)
        neighborhoods = neighborhoods.reshape(batch, height // 2 * (width // 2), -1)
        return self.downsample(neighborhoods)


class MultimodalProjector(nn.Module):
    """GLM-style projection MLP that emits language-model-width image tokens."""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiniGLMVisionEncoder(nn.Module):
    """The architecture-faithful miniature vision tower."""

    def __init__(self, config: MiniVisionConfig, language_hidden_size: int):
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embedding = RGBPatchEmbedding(config)
        self.blocks = nn.ModuleList([VisionTransformerBlock(config) for _ in range(config.depth)])
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.spatial_merger = SpatialMerger2x2(
            config.hidden_size, language_hidden_size, config.spatial_merge_size
        )
        self.projector = MultimodalProjector(
            language_hidden_size, config.projection_intermediate_size
        )

    @property
    def visual_token_count(self) -> int:
        return self.config.visual_token_count

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states, grid_size = self.patch_embedding(images)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.spatial_merger(hidden_states, grid_size)
        return self.projector(hidden_states)


def build_multimodal_input_ids(
    text_input_ids: torch.Tensor,
    *,
    visual_token_count: int,
    token_ids: ImageTokenIds = ImageTokenIds(),
) -> torch.Tensor:
    """Insert ``<image_start> <image>... <image_end>`` after an optional BOS."""
    if text_input_ids.ndim != 2 or text_input_ids.shape[1] < 1:
        raise ValueError("text_input_ids must have shape [batch, time] with time >= 1")
    if visual_token_count < 1:
        raise ValueError("visual_token_count must be positive")
    batch = text_input_ids.shape[0]
    image_span = torch.full(
        (batch, visual_token_count + 2),
        token_ids.placeholder,
        dtype=text_input_ids.dtype,
        device=text_input_ids.device,
    )
    image_span[:, 0] = token_ids.start
    image_span[:, -1] = token_ids.end
    starts_with_bos = text_input_ids[:, 0] == token_ids.bos
    if bool(starts_with_bos.all()):
        return torch.cat((text_input_ids[:, :1], image_span, text_input_ids[:, 1:]), dim=1)
    if bool(starts_with_bos.any()):
        raise ValueError("all rows must agree on whether a BOS token is present")
    return torch.cat((image_span, text_input_ids), dim=1)


class VisionLanguageModel(nn.Module):
    """Replace explicit image placeholders, then call the existing tiny GLM LM."""

    def __init__(
        self,
        language_model: GLM53FlashFromScratch,
        vision_config: MiniVisionConfig | None = None,
        token_ids: ImageTokenIds = ImageTokenIds(),
    ):
        super().__init__()
        if language_model.config.vocab_size < token_ids.required_vocab_size:
            raise ValueError(
                f"language model vocab_size must be at least {token_ids.required_vocab_size}"
            )
        self.language_model = language_model
        self.vision_config = vision_config or MiniVisionConfig()
        self.token_ids = token_ids
        self.vision_encoder = MiniGLMVisionEncoder(
            self.vision_config, language_model.config.dim
        )

    @property
    def visual_token_count(self) -> int:
        return self.vision_encoder.visual_token_count

    @property
    def inserted_token_count(self) -> int:
        return self.visual_token_count + 2

    def prepare_input_ids(self, text_input_ids: torch.Tensor) -> torch.Tensor:
        return build_multimodal_input_ids(
            text_input_ids,
            visual_token_count=self.visual_token_count,
            token_ids=self.token_ids,
        )

    def embed_multimodal(
        self, images: torch.Tensor, multimodal_input_ids: torch.Tensor
    ) -> torch.Tensor:
        visual = self.vision_encoder(images)
        if visual.shape[:2] != (multimodal_input_ids.shape[0], self.visual_token_count):
            raise ValueError("image batch and placeholder sequence do not match")
        placeholder_mask = multimodal_input_ids == self.token_ids.placeholder
        counts = placeholder_mask.sum(dim=1)
        if not bool((counts == self.visual_token_count).all()):
            raise ValueError("each row must contain exactly one placeholder per visual token")
        embedded = self.language_model.embedding(multimodal_input_ids)
        expanded_mask = placeholder_mask.unsqueeze(-1).expand_as(embedded)
        return embedded.masked_scatter(expanded_mask, visual.to(embedded.dtype))

    def forward(
        self, images: torch.Tensor, text_input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multimodal_input_ids = self.prepare_input_ids(text_input_ids)
        embedded = self.embed_multimodal(images, multimodal_input_ids)
        return self.language_model.forward_embeddings(embedded)


class DirectPatchImageEncoder(nn.Module):
    """Baseline only: directly project RGB patches to LM tokens with no vision stack."""

    def __init__(self, dim: int, patch_size: int = 8, image_size: int = 32):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_projection = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.token_count = (image_size // patch_size) ** 2
        self.position = nn.Parameter(torch.empty(1, self.token_count, dim))
        self.modality = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.modality, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (3, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape [batch, {expected[0]}, {expected[1]}, {expected[2]}]"
            )
        patches = self.patch_projection(images).flatten(2).transpose(1, 2)
        return patches + self.position + self.modality


class DirectPatchVisionLanguageModel(nn.Module):
    """Clearly labeled reproduction of the old direct-patch teaching baseline."""

    def __init__(self, language_model: GLM53FlashFromScratch, patch_size: int = 8):
        super().__init__()
        self.language_model = language_model
        self.image_encoder = DirectPatchImageEncoder(
            language_model.config.dim, patch_size=patch_size
        )

    @property
    def visual_token_count(self) -> int:
        return self.image_encoder.token_count

    @property
    def inserted_token_count(self) -> int:
        return self.visual_token_count

    def forward(
        self, images: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.image_encoder(images)
        text = self.language_model.embedding(input_ids)
        return self.language_model.forward_embeddings(torch.cat((visual, text), dim=1))


# Compatibility name for code that imported the old encoder directly.
PatchImageEncoder = DirectPatchImageEncoder


def answer_only_labels(
    token_ids: torch.Tensor,
    *,
    prompt_length: int,
    visual_token_count: int,
    boundary_token_count: int = 2,
) -> torch.Tensor:
    """Align next-byte labels while masking visual, boundary, and prompt positions."""
    if prompt_length < 1 or prompt_length >= token_ids.shape[1]:
        raise ValueError("prompt_length must leave at least one answer token")
    inserted_token_count = visual_token_count + boundary_token_count
    input_length = token_ids.shape[1] - 1
    labels = torch.full(
        (token_ids.shape[0], inserted_token_count + input_length),
        -100,
        dtype=torch.long,
        device=token_ids.device,
    )
    answer_targets = token_ids[:, prompt_length:]
    start = inserted_token_count + prompt_length - 1
    labels[:, start : start + answer_targets.shape[1]] = answer_targets
    return labels
