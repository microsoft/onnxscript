# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Vision encoder components (SigLIP-style).

Provides modules for vision encoding in multimodal models:
- PatchEmbedding: Conv2d-based image patch tokenization
- VisionAttention: Bidirectional multi-head attention (no causal mask, no KV cache)
- VisionMLP: Two-layer MLP with configurable activation
- VisionEncoderLayer: Pre-norm transformer layer (LayerNorm + Attention + MLP)
- VisionEncoder: Stack of VisionEncoderLayers
- VisionModel: Full vision model (embeddings + encoder + final norm)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig

if TYPE_CHECKING:
    import onnx_ir as ir


class PatchEmbedding(nn.Module):
    """Conv2d patch embedding for images.

    Converts [batch, channels, height, width] images to
    [batch, num_patches, hidden_size] patch embeddings with
    learned positional embeddings added.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int = 3,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        # Conv2d weight: [out_channels, in_channels, kH, kW]
        self.patch_embedding = nn.Parameter(
            [hidden_size, num_channels, patch_size, patch_size],
            name="patch_embedding.weight",
        )
        self.patch_embedding_bias = nn.Parameter([hidden_size], name="patch_embedding.bias")
        # Learnable position embeddings: [num_patches, hidden_size]
        # (broadcasts over the batch dimension when added to patches)
        self.position_embedding = nn.Parameter(
            [self.num_patches, hidden_size],
            name="position_embedding.weight",
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [batch, channels, height, width]
        # Conv2d: extract patches
        patches = op.Conv(
            pixel_values,
            self.patch_embedding,
            self.patch_embedding_bias,
            kernel_shape=[
                self.patch_embedding.shape[2],
                self.patch_embedding.shape[3],
            ],
            strides=[
                self.patch_embedding.shape[2],
                self.patch_embedding.shape[3],
            ],
        )
        # patches: [batch, hidden_size, grid_h, grid_w]
        # Reshape to [batch, hidden_size, num_patches]
        batch_size = op.Shape(patches, start=0, end=1)
        hidden_dim = op.Shape(patches, start=1, end=2)
        minus_one = op.Constant(value_ints=[-1])
        new_shape = op.Concat(batch_size, hidden_dim, minus_one, axis=0)
        patches = op.Reshape(patches, new_shape)
        # Transpose to [batch, num_patches, hidden_size]
        patches = op.Transpose(patches, perm=[0, 2, 1])
        # Add position embeddings
        embeddings = op.Add(patches, self.position_embedding)
        return embeddings


class VisionAttention(nn.Module):
    """Bidirectional multi-head attention for vision encoders.

    Uses the ONNX Attention op (opset 23). Unlike text attention,
    this has no causal mask and no KV cache.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = _VisionLinear(hidden_size, hidden_size)
        self.k_proj = _VisionLinear(hidden_size, hidden_size)
        self.v_proj = _VisionLinear(hidden_size, hidden_size)
        self.out_proj = _VisionLinear(hidden_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: [batch, seq_len, hidden_size]
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # op.Attention expects [batch, seq_len, num_heads * head_dim]
        # No mask, no past KV for vision (bidirectional)
        attn_output = op.Attention(
            q,
            k,
            v,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scale,
            _outputs=1,
        )

        return self.out_proj(op, attn_output)


class _VisionLinear(nn.Module):
    """Linear layer with bias for vision encoder."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        result = op.MatMul(x, weight_t)
        return op.Add(result, self.bias)


class VisionMLP(nn.Module):
    """Two-layer MLP with GELU activation for vision encoders."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = _VisionLinear(hidden_size, intermediate_size)
        self.fc2 = _VisionLinear(intermediate_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states, approximate="tanh")
        return self.fc2(op, hidden_states)


class VisionLayerNorm(nn.Module):
    """LayerNorm for vision encoder (not RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return op.LayerNormalization(
            hidden_states,
            self.weight,
            self.bias,
            epsilon=self.eps,
            axis=-1,
        )


class VisionEncoderLayer(nn.Module):
    """Pre-norm vision transformer encoder layer.

    Structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_norm1 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.self_attn = VisionAttention(hidden_size, num_heads)
        self.layer_norm2 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class VisionEncoder(nn.Module):
    """Stack of vision encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        return hidden_states


class _VisionModelInner(nn.Module):
    """Inner module wrapping embeddings + encoder + post_layernorm.

    This exists so that when ``VisionModel`` stores it as
    ``self.vision_model``, the resulting parameter names match HuggingFace's
    ``vision_tower.vision_model.embeddings.*`` convention.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        assert vc.hidden_size is not None
        assert vc.intermediate_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        assert vc.image_size is not None
        assert vc.patch_size is not None

        self.embeddings = PatchEmbedding(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            hidden_size=vc.hidden_size,
        )
        self.encoder = VisionEncoder(
            num_layers=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_attention_heads,
            norm_eps=vc.norm_eps,
        )
        self.post_layernorm = VisionLayerNorm(vc.hidden_size, eps=vc.norm_eps)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.encoder(op, hidden_states)
        hidden_states = self.post_layernorm(op, hidden_states)
        return hidden_states


class VisionModel(nn.Module):
    """Full SigLIP-style vision model.

    Combines patch embedding, encoder layers, and final layer norm
    to produce vision features from pixel values.

    The inner ``vision_model`` wrapper ensures parameter names follow
    HuggingFace's ``vision_tower.vision_model.*`` naming convention.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_model = _VisionModelInner(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        return self.vision_model(op, pixel_values)
