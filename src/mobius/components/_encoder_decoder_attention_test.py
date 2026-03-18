# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EncoderDecoderAttention component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._encoder_decoder_attention import (
    EncoderDecoderAttention,
)


class TestEncoderDecoderAttention:
    """Tests for encoder-decoder (seq2seq) attention module."""

    def test_projection_shapes(self):
        config = make_config()
        attn = EncoderDecoderAttention(config)
        # All projections: (num_heads * head_dim, hidden_size) = (64, 64)
        assert list(attn.q_proj.weight.shape) == [64, 64]
        assert list(attn.k_proj.weight.shape) == [64, 64]
        assert list(attn.v_proj.weight.shape) == [64, 64]
        assert list(attn.out_proj.weight.shape) == [64, 64]

    def test_has_bias_by_default(self):
        config = make_config()
        attn = EncoderDecoderAttention(config)
        assert attn.q_proj.bias is not None

    def test_no_bias(self):
        config = make_config()
        attn = EncoderDecoderAttention(config, bias=False)
        assert attn.q_proj.bias is None

    def test_no_relative_bias_by_default(self):
        config = make_config()
        attn = EncoderDecoderAttention(config)
        assert attn.relative_attention_bias is None

    def test_relative_bias_enabled(self):
        config = make_config(relative_attention_num_buckets=32)
        attn = EncoderDecoderAttention(config, has_relative_attention_bias=True)
        assert attn.relative_attention_bias is not None
        assert list(attn.relative_attention_bias.weight.shape) == [32, 4]

    def test_self_attention_forward(self):
        """Self-attention: no key_value_states provided."""
        config = make_config()
        attn = EncoderDecoderAttention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        output, (pk, pv) = attn(op, hidden)
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_cross_attention_forward(self):
        """Cross-attention: K/V from encoder hidden states."""
        config = make_config()
        attn = EncoderDecoderAttention(config)
        builder, op, graph = create_test_builder()
        decoder_hidden = create_test_input(builder, "dec", [1, 4, 64])
        encoder_hidden = create_test_input(builder, "enc", [1, 8, 64])

        output, (pk, pv) = attn(op, decoder_hidden, key_value_states=encoder_hidden)
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_self_attention_with_past_kv(self):
        config = make_config()
        attn = EncoderDecoderAttention(config, is_causal=True)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        past_key = create_test_input(builder, "pk", [1, 8, 4, 16])
        past_value = create_test_input(builder, "pv", [1, 8, 4, 16])

        output, (pk, pv) = attn(op, hidden, past_key_value=(past_key, past_value))
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_cross_attention_with_past_kv_ignores_cache(self):
        """Cross-attention with past KV must NOT concatenate with cache.

        Encoder output is constant across decode steps, so passing
        past_key_value should not feed into the Attention op's cache
        inputs (which would double the cross-attention sequence length).
        """
        config = make_config()
        attn = EncoderDecoderAttention(config)
        builder, op, graph = create_test_builder()
        decoder_hidden = create_test_input(builder, "dec", [1, 1, 64])
        encoder_hidden = create_test_input(builder, "enc", [1, 8, 64])
        past_key = create_test_input(builder, "pk", [1, 4, 8, 16])
        past_value = create_test_input(builder, "pv", [1, 4, 8, 16])

        output, (pk, pv) = attn(
            op,
            decoder_hidden,
            key_value_states=encoder_hidden,
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1
        # Verify the past_key input is NOT connected to the Attention op
        attn_node = next(n for n in graph if n.op_type == "Attention")
        # Inputs 4 and 5 are past_key and past_value — should be empty
        assert attn_node.inputs[4] is None or attn_node.inputs[4].name == ""
        assert attn_node.inputs[5] is None or attn_node.inputs[5].name == ""

    def test_causal_flag_set(self):
        config = make_config()
        attn = EncoderDecoderAttention(config, is_causal=True)
        assert attn.is_causal is True

    def test_parameter_count_with_bias(self):
        config = make_config()
        attn = EncoderDecoderAttention(config, bias=True)
        params = list(attn.parameters())
        # 4 weights + 4 biases = 8
        assert len(params) == 8

    def test_parameter_count_no_bias(self):
        config = make_config()
        attn = EncoderDecoderAttention(config, bias=False)
        params = list(attn.parameters())
        # 4 weights only
        assert len(params) == 4

    def test_with_attention_bias(self):
        config = make_config()
        attn = EncoderDecoderAttention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "attn_bias", [1, 4, 8, 8])

        output, _ = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output])
        assert count_op_type(graph, "Attention") >= 1
