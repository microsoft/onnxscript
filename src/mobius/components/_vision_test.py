# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for vision encoder components."""

from __future__ import annotations

import pytest

from mobius._configs import VisionConfig
from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._vision import (
    PatchEmbedding,
    VisionAttention,
    VisionEncoder,
    VisionEncoderLayer,
    VisionLayerNorm,
    VisionMLP,
    VisionModel,
)


class TestPatchEmbedding:
    def test_has_parameters(self):
        emb = PatchEmbedding(image_size=32, patch_size=8, hidden_size=64)
        param_names = [n for n, _ in emb.named_parameters()]
        assert any("patch_embedding" in n for n in param_names)
        assert any("position_embedding" in n for n in param_names)

    def test_num_patches(self):
        emb = PatchEmbedding(image_size=32, patch_size=8, hidden_size=64)
        assert emb.num_patches == 16  # (32/8)^2

    def test_forward(self):
        emb = PatchEmbedding(image_size=32, patch_size=8, hidden_size=64)
        b, op, graph = create_test_builder()
        pixels = create_test_input(b, "pixels", [1, 3, 32, 32])
        result = emb(op, pixels)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0


class TestVisionAttention:
    def test_has_projections(self):
        attn = VisionAttention(hidden_size=64, num_heads=4)
        param_names = [n for n, _ in attn.named_parameters()]
        assert any("q_proj" in n for n in param_names)
        assert any("k_proj" in n for n in param_names)
        assert any("v_proj" in n for n in param_names)
        assert any("out_proj" in n for n in param_names)

    def test_forward(self):
        attn = VisionAttention(hidden_size=64, num_heads=4)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = attn(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_uses_attention_op(self):
        """Verify that VisionAttention uses the ONNX Attention op, not decomposed SDPA."""
        attn = VisionAttention(hidden_size=64, num_heads=4)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = attn(op, hidden)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Attention") == 1


class TestVisionMLP:
    def test_forward(self):
        mlp = VisionMLP(hidden_size=64, intermediate_size=128)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = mlp(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0


class TestVisionLayerNorm:
    def test_has_weight_and_bias(self):
        norm = VisionLayerNorm(hidden_size=64)
        param_names = [n for n, _ in norm.named_parameters()]
        assert "weight" in param_names
        assert "bias" in param_names

    def test_forward(self):
        norm = VisionLayerNorm(hidden_size=64)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = norm(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0


class TestVisionEncoderLayer:
    def test_forward(self):
        layer = VisionEncoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = layer(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_has_attn_and_mlp(self):
        layer = VisionEncoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("self_attn" in n for n in param_names)
        assert any("mlp" in n for n in param_names)
        assert any("layer_norm1" in n for n in param_names)
        assert any("layer_norm2" in n for n in param_names)


class TestVisionEncoder:
    def test_num_layers(self):
        enc = VisionEncoder(num_layers=3, hidden_size=64, intermediate_size=128, num_heads=4)
        assert len(enc.layers) == 3

    def test_forward(self):
        enc = VisionEncoder(num_layers=2, hidden_size=64, intermediate_size=128, num_heads=4)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, 16, 64])
        result = enc(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0


def _make_vision_config():
    return make_config(
        vision=VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
            norm_eps=1e-6,
        ),
    )


class TestVisionModel:
    def test_has_embeddings_encoder_norm(self):
        config = _make_vision_config()
        model = VisionModel(config)
        param_names = [n for n, _ in model.named_parameters()]
        assert any("embeddings" in n for n in param_names)
        assert any("encoder" in n for n in param_names)
        assert any("post_layernorm" in n for n in param_names)

    def test_forward(self):
        config = _make_vision_config()
        model = VisionModel(config)
        b, op, graph = create_test_builder()
        pixels = create_test_input(b, "pixel_values", [1, 3, 32, 32])
        result = model(op, pixels)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_requires_vision_config(self):
        config = make_config()  # No vision config
        with pytest.raises(AssertionError):
            VisionModel(config)
