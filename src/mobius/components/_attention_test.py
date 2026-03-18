# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Attention components."""

from __future__ import annotations

import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._attention import Attention, Qwen35Attention


class TestAttention:
    """Tests for the standard multi-head Attention module."""

    def test_projection_weight_shapes(self):
        config = make_config()
        attn = Attention(config)
        # q_proj: (num_heads * head_dim, hidden_size) = (4*16, 64)
        assert list(attn.q_proj.weight.shape) == [64, 64]
        # k_proj: (num_kv_heads * head_dim, hidden_size) = (2*16, 64)
        assert list(attn.k_proj.weight.shape) == [32, 64]
        # v_proj same as k_proj
        assert list(attn.v_proj.weight.shape) == [32, 64]
        # o_proj: (hidden_size, num_heads * head_dim)
        assert list(attn.o_proj.weight.shape) == [64, 64]

    def test_no_qk_norm_by_default(self):
        config = make_config()
        attn = Attention(config)
        assert attn.q_norm is None
        assert attn.k_norm is None

    def test_qk_norm_enabled(self):
        config = make_config(attn_qk_norm=True)
        attn = Attention(config)
        assert attn.q_norm is not None
        assert attn.k_norm is not None

    def test_qk_norm_full_enabled(self):
        config = make_config(attn_qk_norm=True, attn_qk_norm_full=True)
        attn = Attention(config)
        assert attn.q_norm is not None
        # Full norm: weight shape = (num_heads * head_dim,)
        assert list(attn.q_norm.weight.shape) == [64]
        assert list(attn.k_norm.weight.shape) == [32]

    def test_qk_norm_per_head(self):
        config = make_config(attn_qk_norm=True, attn_qk_norm_full=False)
        attn = Attention(config)
        # Per-head norm: weight shape = (head_dim,)
        assert list(attn.q_norm.weight.shape) == [16]
        assert list(attn.k_norm.weight.shape) == [16]

    def test_custom_scale(self):
        config = make_config()
        attn = Attention(config, scale=0.5)
        assert attn.scaling == pytest.approx(0.5)

    def test_default_scale(self):
        config = make_config()
        attn = Attention(config)
        assert attn.scaling == pytest.approx(16**-0.5)

    def test_forward_builds_graph(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])

        output, (present_key, present_value) = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output, present_key, present_value])
        assert graph.num_nodes() > 0
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_past_kv(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        bias = create_test_input(builder, "bias", [1, 4, 1, 9])
        past_key = create_test_input(builder, "pk", [1, 8, 2, 16])
        past_value = create_test_input(builder, "pv", [1, 8, 2, 16])

        output, (pk, pv) = attn(
            op,
            hidden,
            attention_bias=bias,
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_rope(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, _ = attn(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output])
        assert graph.num_nodes() > 0

    def test_forward_with_qk_norm_builds_graph(self):
        config = make_config(attn_qk_norm=True)
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])

        output, _ = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output])
        assert count_op_type(graph, "RMSNormalization") >= 2

    def test_gqa_head_counts(self):
        config = make_config(num_attention_heads=8, num_key_value_heads=2, head_dim=16)
        attn = Attention(config)
        assert attn.num_attention_heads == 8
        assert attn.num_key_value_heads == 2

    def test_parameter_count(self):
        config = make_config()
        attn = Attention(config)
        params = list(attn.parameters())
        # q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight = 4
        assert len(params) == 4

    def test_parameter_count_with_bias(self):
        config = make_config(attn_qkv_bias=True, attn_o_bias=True)
        attn = Attention(config)
        params = list(attn.parameters())
        # 4 weights + 4 biases = 8
        assert len(params) == 8


class TestQwen35Attention:
    """Tests for Qwen3.5 gated attention."""

    def test_q_proj_doubled(self):
        config = make_config(
            partial_rotary_factor=0.5,
        )
        attn = Qwen35Attention(config)
        # Q proj is doubled: 2 * (num_heads * head_dim)
        assert list(attn.q_proj.weight.shape) == [128, 64]

    def test_has_offset_rms_norm(self):
        from mobius.components._rms_norm import OffsetRMSNorm

        config = make_config(partial_rotary_factor=0.5)
        attn = Qwen35Attention(config)
        assert isinstance(attn.q_norm, OffsetRMSNorm)
        assert isinstance(attn.k_norm, OffsetRMSNorm)

    def test_forward_builds_graph(self):
        config = make_config(partial_rotary_factor=0.5)
        attn = Qwen35Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, (pk, pv) = attn(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])
        # Should have Attention op + Sigmoid (for gate) + Mul (output gating)
        assert count_op_type(graph, "Attention") >= 1
        assert count_op_type(graph, "Sigmoid") >= 1
