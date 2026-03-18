# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Whisper encoder/decoder components."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._whisper import (
    Conv1d,
    WhisperAttention,
    WhisperDecoderLayer,
    WhisperEncoderLayer,
)


class TestConv1d:
    """Tests for 1D convolution layer."""

    def test_weight_shape(self):
        conv = Conv1d(80, 512, kernel_size=3, padding=1)
        assert list(conv.weight.shape) == [512, 80, 3]

    def test_bias_shape(self):
        conv = Conv1d(80, 512, kernel_size=3)
        assert list(conv.bias.shape) == [512]

    def test_parameter_count(self):
        conv = Conv1d(80, 512, kernel_size=3)
        params = list(conv.parameters())
        assert len(params) == 2  # weight + bias

    def test_forward_builds_graph(self):
        conv = Conv1d(80, 512, kernel_size=3, padding=1)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 80, 100])

        result = conv(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1


class TestWhisperAttention:
    """Tests for Whisper attention (no RoPE, with bias)."""

    def test_projection_shapes(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        assert list(attn.q_proj.weight.shape) == [64, 64]
        assert list(attn.k_proj.weight.shape) == [64, 64]
        assert list(attn.v_proj.weight.shape) == [64, 64]
        assert list(attn.out_proj.weight.shape) == [64, 64]

    def test_q_proj_has_bias(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        assert attn.q_proj.bias is not None

    def test_k_proj_no_bias(self):
        """Whisper K projection has no bias (matches HuggingFace)."""
        attn = WhisperAttention(d_model=64, num_heads=4)
        assert attn.k_proj.bias is None

    def test_causal_flag(self):
        attn = WhisperAttention(d_model=64, num_heads=4, is_causal=True)
        assert attn._is_causal is True

    def test_self_attention_forward(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        output, (pk, pv) = attn(op, hidden)
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_cross_attention_forward(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        builder, op, graph = create_test_builder()
        decoder_hidden = create_test_input(builder, "dec", [1, 4, 64])
        encoder_hidden = create_test_input(builder, "enc", [1, 8, 64])

        output, _ = attn(op, decoder_hidden, key_value_states=encoder_hidden)
        builder._adapt_outputs([output])
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_past_kv(self):
        attn = WhisperAttention(d_model=64, num_heads=4, is_causal=True)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        past_key = create_test_input(builder, "pk", [1, 8, 4, 16])
        past_value = create_test_input(builder, "pv", [1, 8, 4, 16])

        output, (pk, pv) = attn(op, hidden, past_key_value=(past_key, past_value))
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_pre_scales_query(self):
        """Whisper pre-scales Q (not the Attention op)."""
        attn = WhisperAttention(d_model=64, num_heads=4)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        attn(op, hidden)
        # Pre-scaling: Mul before Attention
        assert count_op_type(graph, "Mul") >= 1


class TestWhisperEncoderLayer:
    """Tests for Whisper encoder layer."""

    def test_has_submodules(self):
        layer = WhisperEncoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "self_attn_layer_norm")
        assert hasattr(layer, "fc1")
        assert hasattr(layer, "fc2")
        assert hasattr(layer, "final_layer_norm")

    def test_forward_builds_graph(self):
        layer = WhisperEncoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        result = layer(op, hidden)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Attention") >= 1
        assert count_op_type(graph, "Gelu") >= 1
        assert count_op_type(graph, "LayerNormalization") >= 2

    def test_residual_connections(self):
        layer = WhisperEncoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        layer(op, hidden)
        assert count_op_type(graph, "Add") >= 2


class TestWhisperDecoderLayer:
    """Tests for Whisper decoder layer (self + cross attention + FFN)."""

    def test_has_submodules(self):
        layer = WhisperDecoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "encoder_attn")
        assert hasattr(layer, "fc1")
        assert hasattr(layer, "fc2")

    def test_self_attn_is_causal(self):
        layer = WhisperDecoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        assert layer.self_attn._is_causal is True

    def test_cross_attn_is_not_causal(self):
        layer = WhisperDecoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        assert layer.encoder_attn._is_causal is False

    def test_forward_builds_graph(self):
        layer = WhisperDecoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64])
        encoder_hidden = create_test_input(builder, "encoder_hidden", [1, 8, 64])

        output, present_kv = layer(op, hidden, encoder_hidden)
        builder._adapt_outputs([output, present_kv[0], present_kv[1]])
        # Self-attention + cross-attention = 2 Attention ops
        assert count_op_type(graph, "Attention") >= 2
        assert count_op_type(graph, "Gelu") >= 1
        assert count_op_type(graph, "LayerNormalization") >= 3

    def test_forward_with_past_kv(self):
        layer = WhisperDecoderLayer(d_model=64, num_heads=4, ffn_dim=128)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        encoder_hidden = create_test_input(builder, "encoder_hidden", [1, 8, 64])
        past_key = create_test_input(builder, "pk", [1, 4, 4, 16])
        past_value = create_test_input(builder, "pv", [1, 4, 4, 16])

        output, (pk, pv) = layer(
            op,
            hidden,
            encoder_hidden,
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 2
