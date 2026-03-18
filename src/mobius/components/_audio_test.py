# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio encoder components (Conformer-style)."""

from __future__ import annotations

from mobius._testing import create_test_builder, create_test_input
from mobius.components._audio import (
    ConformerAttention,
    ConformerConvModule,
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerFeedForward,
    MeanVarianceNorm,
    NeMoConvSubsampling,
    T5RelativeAttentionBias,
)

_DIM = 32
_HEADS = 2
_LINEAR = 64
_KERNEL = 3
_BATCH = 1
_TIME = 16
_INPUT_SIZE = 16


class TestMeanVarianceNorm:
    def test_has_parameters(self):
        mvn = MeanVarianceNorm(_INPUT_SIZE)
        param_names = [n for n, _ in mvn.named_parameters()]
        assert "global_mean" in param_names
        assert "global_invstd" in param_names

    def test_forward(self):
        mvn = MeanVarianceNorm(_INPUT_SIZE)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _INPUT_SIZE])
        out = mvn(op, x)
        assert out is not None


class TestNeMoConvSubsampling:
    def test_has_parameters(self):
        sub = NeMoConvSubsampling(_INPUT_SIZE, conv_channels=_DIM, feat_out=_DIM)
        param_names = [n for n, _ in sub.named_parameters()]
        assert any("conv" in n for n in param_names)
        assert any("out" in n for n in param_names)

    def test_forward(self):
        sub = NeMoConvSubsampling(_INPUT_SIZE, conv_channels=_DIM, feat_out=_DIM)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _INPUT_SIZE])
        out = sub(op, x)
        assert out is not None


class TestT5RelativeAttentionBias:
    def test_has_parameters(self):
        bias = T5RelativeAttentionBias(num_buckets=20, num_heads=_HEADS, max_distance=10)
        param_names = [n for n, _ in bias.named_parameters()]
        assert any("bias_values" in n for n in param_names)

    def test_forward(self):
        bias = T5RelativeAttentionBias(num_buckets=20, num_heads=_HEADS, max_distance=10)
        builder, op, _graph = create_test_builder()
        seq_len = create_test_input(builder, "seq_len", [1])
        out = bias(op, seq_len)
        assert out is not None


class TestConformerFeedForward:
    def test_has_parameters(self):
        ff = ConformerFeedForward(_DIM, _LINEAR)
        param_names = [n for n, _ in ff.named_parameters()]
        assert any("net" in n for n in param_names)
        assert any("layer_norm" in n for n in param_names)

    def test_forward(self):
        ff = ConformerFeedForward(_DIM, _LINEAR)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        out = ff(op, x)
        assert out is not None


class TestConformerConvModule:
    def test_has_parameters(self):
        conv = ConformerConvModule(_DIM, _KERNEL)
        param_names = [n for n, _ in conv.named_parameters()]
        assert any("glu" in n for n in param_names)
        assert any("dw_sep_conv_1d" in n for n in param_names)
        assert any("ext_pw_conv_1d" in n for n in param_names)
        assert any("layer_norm" in n for n in param_names)

    def test_forward(self):
        conv = ConformerConvModule(_DIM, _KERNEL)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        out = conv(op, x)
        assert out is not None


class TestConformerAttention:
    def test_has_parameters(self):
        attn = ConformerAttention(_DIM, _HEADS)
        param_names = [n for n, _ in attn.named_parameters()]
        assert any("linear_q" in n for n in param_names)
        assert any("linear_k" in n for n in param_names)
        assert any("linear_v" in n for n in param_names)
        assert any("linear_out" in n for n in param_names)

    def test_forward(self):
        attn = ConformerAttention(_DIM, _HEADS)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        bias = create_test_input(builder, "bias", [1, _HEADS, _TIME, _TIME])
        out = attn(op, x, bias)
        assert out is not None


class TestConformerEncoderLayer:
    def test_has_parameters(self):
        layer = ConformerEncoderLayer(_DIM, _HEADS, _LINEAR, _KERNEL)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("self_attn" in n for n in param_names)
        assert any("feed_forward" in n for n in param_names)
        assert any("conv" in n for n in param_names)

    def test_forward(self):
        layer = ConformerEncoderLayer(_DIM, _HEADS, _LINEAR, _KERNEL)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        bias = create_test_input(builder, "bias", [1, _HEADS, _TIME, _TIME])
        out = layer(op, x, bias)
        assert out is not None


class TestConformerEncoder:
    def test_has_parameters(self):
        enc = ConformerEncoder(
            input_size=_INPUT_SIZE,
            attention_dim=_DIM,
            attention_heads=_HEADS,
            num_blocks=1,
            linear_units=_LINEAR,
            kernel_size=_KERNEL,
            conv_channels=_DIM,
            t5_bias_max_distance=10,
        )
        param_names = [n for n, _ in enc.named_parameters()]
        assert any("embed" in n for n in param_names)
        assert any("encoders" in n for n in param_names)

    def test_forward(self):
        enc = ConformerEncoder(
            input_size=_INPUT_SIZE,
            attention_dim=_DIM,
            attention_heads=_HEADS,
            num_blocks=1,
            linear_units=_LINEAR,
            kernel_size=_KERNEL,
            conv_channels=_DIM,
            t5_bias_max_distance=10,
        )
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _INPUT_SIZE])
        out = enc(op, x)
        assert out is not None
