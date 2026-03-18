# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for standalone vision encoder models.

Tests ViT and CLIP vision encoders with random weights against
PyTorch references. Each test builds a tiny model, transfers weights,
runs a forward pass, and compares hidden states.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mobius import build_from_module, models
from mobius._configs import ArchitectureConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights

# ---------------------------------------------------------------------------
# Shared tiny config values
# ---------------------------------------------------------------------------
_HIDDEN = 32
_INTERMEDIATE = 64
_HEADS = 2
_LAYERS = 1
_IMAGE_SIZE = 32
_PATCH_SIZE = 8
_CHANNELS = 3
_NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# PyTorch reference: ViT
# ---------------------------------------------------------------------------
class _TorchViTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.up_proj(x)))


class _TorchViTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        b, n, c = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn.transpose(1, 2).reshape(b, n, c))


class _TorchViTEncoderLayer(nn.Module):
    """Pre-norm encoder layer matching ViTModel's _ViTEncoderLayer."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=_NORM_EPS)
        self.self_attn = _TorchViTAttention(hidden_size, num_heads)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=_NORM_EPS)
        self.mlp = _TorchViTMLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.layernorm_before(x))
        x = x + self.mlp(self.layernorm_after(x))
        return x


class _TorchViTModel(nn.Module):
    """PyTorch reference matching ViTModel ONNX parameter names.

    State dict keys:
        embeddings.patch_embeddings.projection.{weight,bias}
        embeddings.cls_token, embeddings.position_embeddings
        encoder.layer.0.{layernorm_before,self_attn.*,layernorm_after,mlp.*}
        layernorm.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        num_patches = (_IMAGE_SIZE // _PATCH_SIZE) ** 2
        # Embeddings
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(
            _CHANNELS, _HIDDEN, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE
        )
        self.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, _HIDDEN))
        self.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, _HIDDEN)
        )
        # Encoder
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [_TorchViTEncoderLayer(_HIDDEN, _INTERMEDIATE, _HEADS) for _ in range(_LAYERS)]
        )
        # Final norm
        self.layernorm = nn.LayerNorm(_HIDDEN, eps=_NORM_EPS)

    def forward(self, pixel_values):
        # Patch embed
        x = self.embeddings.patch_embeddings.projection(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        # CLS + position
        batch = x.shape[0]
        cls = self.embeddings.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.embeddings.position_embeddings
        # Encoder
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)


# ---------------------------------------------------------------------------
# PyTorch reference: CLIP Vision
# ---------------------------------------------------------------------------
class _TorchCLIPVisionModel(nn.Module):
    """PyTorch reference matching CLIPVisionModel ONNX parameter names.

    State dict keys:
        embeddings.class_embedding
        embeddings.patch_embedding.projection.{weight,bias}
        embeddings.position_embedding.weight
        pre_layrnorm.{weight,bias}
        encoder.0.{layer_norm1,self_attn.*,layer_norm2,mlp.*}
        post_layernorm.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        num_patches = (_IMAGE_SIZE // _PATCH_SIZE) ** 2
        # Embeddings
        self.embeddings = nn.Module()
        self.embeddings.class_embedding = nn.Parameter(torch.zeros(_HIDDEN))
        self.embeddings.patch_embedding = nn.Module()
        self.embeddings.patch_embedding.projection = nn.Conv2d(
            _CHANNELS, _HIDDEN, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE
        )
        self.embeddings.position_embedding = nn.Embedding(num_patches + 1, _HIDDEN)
        # Pre/post norms
        self.pre_layrnorm = nn.LayerNorm(_HIDDEN, eps=_NORM_EPS)
        self.post_layernorm = nn.LayerNorm(_HIDDEN, eps=_NORM_EPS)
        # Encoder as indexed ModuleList (keys: encoder.0.*, encoder.1.*, ...)
        self.encoder = nn.ModuleList([_TorchCLIPEncoderLayer() for _ in range(_LAYERS)])

    def forward(self, pixel_values):
        # Patch embed
        x = self.embeddings.patch_embedding.projection(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        # CLS token
        batch = x.shape[0]
        cls = self.embeddings.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # Position embedding
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embeddings.position_embedding(position_ids)
        # Pre-norm
        x = self.pre_layrnorm(x)
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        return self.post_layernorm(x)


class _TorchCLIPEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(_HIDDEN, eps=_NORM_EPS)
        self.self_attn = _TorchCLIPAttention()
        self.layer_norm2 = nn.LayerNorm(_HIDDEN, eps=_NORM_EPS)
        self.mlp = _TorchCLIPMLP()

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _TorchCLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = _HEADS
        self.head_dim = _HIDDEN // _HEADS
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN)
        self.k_proj = nn.Linear(_HIDDEN, _HIDDEN)
        self.v_proj = nn.Linear(_HIDDEN, _HIDDEN)
        self.o_proj = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x):
        b, n, c = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn.transpose(1, 2).reshape(b, n, c))


class _TorchCLIPMLP(nn.Module):
    """CLIP uses quick_gelu: x * sigmoid(1.702 * x)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTERMEDIATE)
        self.fc2 = nn.Linear(_INTERMEDIATE, _HIDDEN)

    def forward(self, x):
        x = self.fc1(x)
        x = x * torch.sigmoid(1.702 * x)  # quick_gelu
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.integration_fast
class TestViTEncoderParity:
    """ViT encoder: ONNX vs PyTorch with random weights."""

    def test_vit_hidden_states_match(self):
        """Build tiny ViT, transfer random weights, compare hidden states."""
        import onnx_ir as ir

        config = ArchitectureConfig(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_attention_heads=_HEADS,
            num_key_value_heads=_HEADS,
            head_dim=_HIDDEN // _HEADS,
            num_hidden_layers=_LAYERS,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="gelu",
            rms_norm_eps=_NORM_EPS,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
            image_size=_IMAGE_SIZE,
            patch_size=_PATCH_SIZE,
            num_channels=_CHANNELS,
        )
        config.dtype = ir.DataType.FLOAT

        # Build ONNX model
        onnx_module = models.ViTModel(config)
        pkg = build_from_module(onnx_module, config, task="image-classification")
        assert "model" in pkg

        # Build PyTorch reference
        ref_model = _TorchViTModel().float().eval()

        # Transfer weights: PyTorch state_dict keys match ONNX param names
        # directly — skip preprocess_weights (which expects HF-format names)
        state_dict = ref_model.state_dict()
        apply_weights(pkg["model"], dict(state_dict))

        # Forward pass
        rng = np.random.default_rng(42)
        pixel_values = rng.standard_normal((1, _CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)).astype(
            np.float32
        )

        with torch.no_grad():
            ref_out = ref_model(torch.from_numpy(pixel_values)).numpy()

        session = OnnxModelSession(pkg["model"])
        onnx_out = session.run({"pixel_values": pixel_values})
        session.close()

        assert_logits_close(
            onnx_out["last_hidden_state"],
            ref_out,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
@pytest.mark.integration_fast
class TestCLIPVisionEncoderParity:
    """CLIP vision encoder: ONNX vs PyTorch with random weights."""

    def test_clip_vision_hidden_states_match(self):
        """Build tiny CLIP vision, transfer random weights, compare output."""
        import onnx_ir as ir

        config = ArchitectureConfig(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_attention_heads=_HEADS,
            num_key_value_heads=_HEADS,
            head_dim=_HIDDEN // _HEADS,
            num_hidden_layers=_LAYERS,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="quick_gelu",
            rms_norm_eps=_NORM_EPS,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
            image_size=_IMAGE_SIZE,
            patch_size=_PATCH_SIZE,
            num_channels=_CHANNELS,
        )
        config.dtype = ir.DataType.FLOAT

        # Build ONNX model
        onnx_module = models.CLIPVisionModel(config)
        pkg = build_from_module(onnx_module, config, task="image-classification")
        assert "model" in pkg

        # Build PyTorch reference
        ref_model = _TorchCLIPVisionModel().float().eval()

        # Transfer weights directly — names already match ONNX params
        state_dict = ref_model.state_dict()
        apply_weights(pkg["model"], dict(state_dict))

        # Forward pass
        rng = np.random.default_rng(42)
        pixel_values = rng.standard_normal((1, _CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)).astype(
            np.float32
        )

        with torch.no_grad():
            ref_out = ref_model(torch.from_numpy(pixel_values)).numpy()

        session = OnnxModelSession(pkg["model"])
        onnx_out = session.run({"pixel_values": pixel_values})
        session.close()

        assert_logits_close(
            onnx_out["last_hidden_state"],
            ref_out,
            rtol=1e-3,
            atol=1e-3,
        )
