# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DecoderLayer component."""

from __future__ import annotations

import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._decoder import (
    DecoderLayer,
    PostNormDecoderLayer,
    create_decoder_layer,
)
from mobius.components._rms_norm import OffsetRMSNorm, RMSNorm


class TestDecoderLayer:
    """Tests for the pre-norm DecoderLayer."""

    def test_has_attention_and_mlp(self):
        config = make_config()
        layer = DecoderLayer(config)
        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")

    def test_pre_norm_has_two_norms(self):
        config = make_config()
        layer = DecoderLayer(config)
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "post_attention_layernorm")

    def test_default_norm_is_rmsnorm(self):
        config = make_config()
        layer = DecoderLayer(config)
        assert isinstance(layer.input_layernorm, RMSNorm)

    def test_custom_norm_class(self):
        config = make_config()
        layer = DecoderLayer(config, norm_class=OffsetRMSNorm)
        assert isinstance(layer.input_layernorm, OffsetRMSNorm)

    def test_parameter_count(self):
        config = make_config()
        layer = DecoderLayer(config)
        params = list(layer.parameters())
        # Attention: q, k, v, o = 4 weights
        # MLP: gate, up, down = 3 weights
        # Norms: input_layernorm.weight, post_attn_layernorm.weight = 2
        assert len(params) == 9

    def test_forward_builds_graph(self):
        config = make_config()
        layer = DecoderLayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, (pk, pv) = layer(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
            past_key_value=None,
        )
        builder._adapt_outputs([output, pk, pv])
        assert graph.num_nodes() > 0
        # Pre-norm: RMSNorm for input_layernorm + post_attention_layernorm
        assert count_op_type(graph, "RMSNormalization") >= 2
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_past_kv(self):
        config = make_config()
        layer = DecoderLayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        bias = create_test_input(builder, "bias", [1, 4, 1, 9])
        cos = create_test_input(builder, "cos", [1, 1, 16])
        sin = create_test_input(builder, "sin", [1, 1, 16])
        past_key = create_test_input(builder, "pk", [1, 8, 2, 16])
        past_value = create_test_input(builder, "pv", [1, 8, 2, 16])

        output, (pk, pv) = layer(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv])
        assert count_op_type(graph, "Attention") >= 1

    def test_residual_connections(self):
        config = make_config()
        layer = DecoderLayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        layer(
            op,
            hidden,
            bias,
            (cos, sin),
            None,
        )
        # Two residual Add ops (attn + mlp)
        assert count_op_type(graph, "Add") >= 2

    def test_residual_multiplier(self):
        config = make_config()
        layer = DecoderLayer(config, residual_multiplier=0.5)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        layer(op, hidden, bias, (cos, sin), None)
        # Residual multiplier adds extra Mul ops
        assert count_op_type(graph, "Mul") >= 2


class TestPostNormDecoderLayer:
    """Tests for the post-norm DecoderLayer variant."""

    def test_post_norm_has_feedforward_norm(self):
        config = make_config()
        layer = PostNormDecoderLayer(config)
        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "post_feedforward_layernorm")
        # No input_layernorm in post-norm
        assert not hasattr(layer, "input_layernorm")

    def test_forward_builds_graph(self):
        config = make_config()
        layer = PostNormDecoderLayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, _ = layer(op, hidden, bias, (cos, sin), None)
        builder._adapt_outputs([output])
        assert graph.num_nodes() > 0


class TestCreateDecoderLayer:
    """Tests for the config-driven factory."""

    def test_creates_default_layer(self):
        config = make_config()
        layer = create_decoder_layer(config)
        assert isinstance(layer, DecoderLayer)

    def test_reads_residual_multiplier(self):
        config = make_config(residual_multiplier=0.5)
        layer = create_decoder_layer(config)
        assert layer._residual_multiplier == pytest.approx(0.5)

    def test_reads_attention_multiplier(self):
        config = make_config(attention_multiplier=0.25)
        layer = create_decoder_layer(config)
        assert layer.self_attn.scaling == pytest.approx(0.25)

    def test_post_norm_flag(self):
        config = make_config()
        layer = create_decoder_layer(config, post_norm=True)
        assert layer._post_norm is True
