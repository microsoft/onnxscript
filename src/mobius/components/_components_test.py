# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Attention, MLP, and DecoderLayer modules."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._attention import Attention
from mobius.components._decoder import DecoderLayer
from mobius.components._mlp import MLP


class TestAttention:
    def test_attention_params(self):
        config = make_config()
        attn = Attention(config)
        param_names = [n for n, _ in attn.named_parameters()]
        assert "q_proj.weight" in param_names
        assert "k_proj.weight" in param_names
        assert "v_proj.weight" in param_names
        assert "o_proj.weight" in param_names

    def test_attention_no_bias(self):
        config = make_config(attn_qkv_bias=False, attn_o_bias=False)
        attn = Attention(config)
        assert attn.q_proj.bias is None
        assert attn.o_proj.bias is None

    def test_attention_with_bias(self):
        config = make_config(attn_qkv_bias=True, attn_o_bias=True)
        attn = Attention(config)
        assert attn.q_proj.bias is not None
        assert attn.o_proj.bias is not None

    def test_attention_qk_norm(self):
        config = make_config(attn_qk_norm=True)
        attn = Attention(config)
        assert attn.q_norm is not None
        assert attn.k_norm is not None

    def test_attention_no_qk_norm(self):
        config = make_config(attn_qk_norm=False)
        attn = Attention(config)
        assert attn.q_norm is None
        assert attn.k_norm is None

    def test_attention_forward(self):
        config = make_config()
        attn = Attention(config)
        builder, op, _graph = create_test_builder()
        hidden_states = create_test_input(builder, "h", [2, 4, 64])
        attn_bias = create_test_input(builder, "bias", [2, 1, 4, 8])
        cos = create_test_input(builder, "cos", [2, 4, 8])
        sin = create_test_input(builder, "sin", [2, 4, 8])
        past_key = create_test_input(builder, "pk", [2, 2, 4, 16])
        past_value = create_test_input(builder, "pv", [2, 2, 4, 16])

        result = attn(
            op,
            hidden_states,
            attn_bias,
            (cos, sin),
            (past_key, past_value),
        )
        assert len(result) == 2
        attn_output, (present_key, present_value) = result
        assert attn_output is not None
        assert present_key is not None
        assert present_value is not None

    def test_attention_gqa(self):
        """Test grouped query attention (num_kv_heads < num_attention_heads)."""
        config = make_config(num_attention_heads=8, num_key_value_heads=2)
        attn = Attention(config)
        assert attn.num_attention_heads == 8
        assert attn.num_key_value_heads == 2

    def test_attention_uses_attention_op(self):
        """Verify that Attention uses the ONNX Attention op, not decomposed SDPA."""
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden_states = create_test_input(builder, "h", [2, 4, 64])
        attn_bias = create_test_input(builder, "bias", [2, 1, 4, 8])
        cos = create_test_input(builder, "cos", [2, 4, 8])
        sin = create_test_input(builder, "sin", [2, 4, 8])
        past_key = create_test_input(builder, "pk", [2, 2, 4, 16])
        past_value = create_test_input(builder, "pv", [2, 2, 4, 16])

        attn(op, hidden_states, attn_bias, (cos, sin), (past_key, past_value))
        assert count_op_type(graph, "Attention") == 1


class TestMLP:
    def test_mlp_params(self):
        config = make_config()
        mlp = MLP(config)
        param_names = [n for n, _ in mlp.named_parameters()]
        assert "gate_proj.weight" in param_names
        assert "up_proj.weight" in param_names
        assert "down_proj.weight" in param_names

    def test_mlp_no_bias(self):
        config = make_config(mlp_bias=False)
        mlp = MLP(config)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_mlp_forward(self):
        config = make_config()
        mlp = MLP(config)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 4, 64])
        result = mlp(op, x)
        assert result is not None
        assert count_op_type(graph, "MatMul") >= 3

    def test_mlp_with_different_activations(self):
        for act in ["silu", "gelu", "relu"]:
            config = make_config(hidden_act=act)
            mlp = MLP(config)
            builder, op, _graph = create_test_builder()
            x = create_test_input(builder, "x", [2, 4, 64])
            result = mlp(op, x)
            assert result is not None


class TestDecoderLayer:
    def test_decoder_layer_params(self):
        config = make_config()
        layer = DecoderLayer(config)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("input_layernorm" in n for n in param_names)
        assert any("post_attention_layernorm" in n for n in param_names)
        assert any("self_attn" in n for n in param_names)
        assert any("mlp" in n for n in param_names)

    def test_decoder_layer_forward(self):
        config = make_config()
        layer = DecoderLayer(config)
        builder, op, graph = create_test_builder()
        hidden_states = create_test_input(builder, "h", [2, 4, 64])
        attn_bias = create_test_input(builder, "bias", [2, 1, 4, 8])
        cos = create_test_input(builder, "cos", [2, 4, 8])
        sin = create_test_input(builder, "sin", [2, 4, 8])
        past_key = create_test_input(builder, "pk", [2, 2, 4, 16])
        past_value = create_test_input(builder, "pv", [2, 2, 4, 16])

        result = layer(
            op,
            hidden_states,
            attn_bias,
            (cos, sin),
            (past_key, past_value),
        )
        assert len(result) == 2
        out, (_pk, _pv) = result
        assert out is not None
        assert count_op_type(graph, "Add") >= 2
