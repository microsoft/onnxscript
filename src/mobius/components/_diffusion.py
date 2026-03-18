# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for diffusion transformer models.

Components used across DiT, SD3/MMDiT, Flux, and HunyuanDiT architectures.
Includes adaptive layer normalization variants, patch embedding, timestep
embedding, and feed-forward / self-attention modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._activations import SiLU
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d

if TYPE_CHECKING:
    import onnx_ir as ir


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Norm with zero-init modulation for DiT.

    Produces 6 modulation parameters (shift, scale, gate for both
    attention and FFN) from the timestep embedding.

    Used by DiT, SD3/MMDiT, and Flux transformer blocks.

    Replicates HuggingFace diffusers' ``AdaLayerNormZero``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LayerNorm(hidden_size, eps=eps)
        # 6 modulation parameters: shift1, scale1, gate1, shift2, scale2, gate2
        self.linear = Linear(hidden_size, hidden_size * 6)
        self._silu = SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, timestep_emb: ir.Value):
        emb = self._silu(op, timestep_emb)
        emb = self.linear(op, emb)
        # [B, 6*C] → 6 x [B, C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = op.Split(
            emb,
            num_outputs=6,
            axis=-1,
            _outputs=6,
        )
        # Apply to hidden_states
        normed = self.norm(op, hidden_states)
        return (
            normed,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )


class AdaLayerNormOutput(nn.Module):
    """Final adaptive layer norm with scale+shift from timestep embedding.

    Used as the output normalization in DiT, SD3, and Flux models.

    Replicates HuggingFace diffusers' ``AdaLayerNormContinuous``
    (with elementwise_affine=True variant).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LayerNorm(hidden_size, eps=eps)
        self.linear = Linear(hidden_size, hidden_size * 2)
        self._silu = SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, timestep_emb: ir.Value):
        emb = self._silu(op, timestep_emb)
        emb = self.linear(op, emb)
        shift, scale = op.Split(emb, num_outputs=2, axis=-1, _outputs=2)
        one = op.Constant(value_float=1.0)
        hidden_states = self.norm(op, hidden_states)
        hidden_states = op.Mul(hidden_states, op.Add(one, op.Unsqueeze(scale, [1])))
        hidden_states = op.Add(hidden_states, op.Unsqueeze(shift, [1]))
        return hidden_states


class PatchEmbed(nn.Module):
    """Patch embedding: Conv2d with stride = patch_size.

    Converts spatial input [B, C, H, W] to a sequence of patch tokens
    [B, num_patches, hidden_size]. Used by DiT, SD3, Flux, and HunyuanDiT.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.proj = Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # [B, C, H, W] → [B, hidden, H/p, W/p]
        x = self.proj(op, x)
        batch = op.Shape(x, start=0, end=1)
        channels = op.Shape(x, start=1, end=2)
        height = op.Shape(x, start=2, end=3)
        width = op.Shape(x, start=3, end=4)
        # → [B, hidden, H/p * W/p] → [B, H/p * W/p, hidden]
        spatial = op.Mul(height, width)
        x = op.Reshape(x, op.Concat(batch, channels, spatial, axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class TimestepEmbedding(nn.Module):
    """Projects timestep embedding to hidden dim: Linear → SiLU → Linear.

    Standard timestep MLP used by all diffusion transformer architectures.

    Replicates HuggingFace diffusers' ``TimestepEmbedding``.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim)
        self.linear_2 = Linear(time_embed_dim, time_embed_dim)
        self._silu = SiLU()

    def forward(self, op: builder.OpBuilder, sample: ir.Value):
        sample = self.linear_1(op, sample)
        sample = self._silu(op, sample)
        sample = self.linear_2(op, sample)
        return sample


class DiffusionFFN(nn.Module):
    """GELU MLP for diffusion transformers.

    Two-layer MLP with GELU (tanh approximation) activation.
    Used by DiT, SD3/MMDiT, and Flux transformer blocks.
    """

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        self.linear_1 = Linear(hidden_size, intermediate_size)
        self.linear_2 = Linear(intermediate_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.linear_1(op, hidden_states)
        # GELU with tanh approximation
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.linear_2(op, hidden_states)
        return hidden_states


class DiffusionSelfAttention(nn.Module):
    """Multi-head self-attention for diffusion transformers.

    Standard QKV self-attention with separate projections and output
    projection wrapped in nn.Sequential (index 0) for HF weight compat.

    Used by DiT, SD3/MMDiT (joint attention), and Flux.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.to_q = Linear(hidden_size, hidden_size)
        self.to_k = Linear(hidden_size, hidden_size)
        self.to_v = Linear(hidden_size, hidden_size)
        self.to_out = nn.Sequential(Linear(hidden_size, hidden_size))
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        q = self.to_q(op, hidden_states)
        k = self.to_k(op, hidden_states)
        v = self.to_v(op, hidden_states)
        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return self.to_out(op, attn_out)
