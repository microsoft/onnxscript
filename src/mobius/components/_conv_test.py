# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for convolution components."""

from __future__ import annotations

import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._codec_conv import CausalConvNd
from mobius.components._conv import (
    BatchNorm2d,
    Conv2d,
    Conv2dNoBias,
    ConvTranspose2d,
)


class TestConv2d:
    """Tests for 2D convolution with bias."""

    def test_weight_shape(self):
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        assert list(conv.weight.shape) == [16, 3, 3, 3]

    def test_bias_exists(self):
        conv = Conv2d(3, 16, kernel_size=3)
        assert conv.bias is not None
        assert list(conv.bias.shape) == [16]

    def test_parameter_count(self):
        conv = Conv2d(3, 16, kernel_size=3)
        params = list(conv.parameters())
        assert len(params) == 2  # weight + bias

    def test_forward_builds_graph(self):
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 3, 32, 32])

        result = conv(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1

    def test_stride_setting(self):
        conv = Conv2d(3, 16, kernel_size=3, stride=2)
        assert conv._stride == 2

    def test_groups(self):
        conv = Conv2d(16, 16, kernel_size=3, groups=16)
        # Depthwise: (out, in//groups, k, k) = (16, 1, 3, 3)
        assert list(conv.weight.shape) == [16, 1, 3, 3]

    def test_forward_with_stride(self):
        conv = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 3, 32, 32])

        conv(op, x)
        assert count_op_type(graph, "Conv") >= 1


class TestConv2dNoBias:
    """Tests for 2D convolution without bias."""

    def test_no_bias(self):
        conv = Conv2dNoBias(3, 16, kernel_size=3)
        params = list(conv.parameters())
        assert len(params) == 1  # weight only

    def test_weight_shape(self):
        conv = Conv2dNoBias(3, 16, kernel_size=5)
        assert list(conv.weight.shape) == [16, 3, 5, 5]

    def test_forward_builds_graph(self):
        conv = Conv2dNoBias(3, 16, kernel_size=3, padding=1)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 3, 32, 32])

        result = conv(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1


class TestBatchNorm2d:
    """Tests for 2D batch normalization."""

    def test_parameter_count(self):
        bn = BatchNorm2d(16)
        params = list(bn.parameters())
        # weight, bias, running_mean, running_var = 4
        assert len(params) == 4

    def test_parameter_shapes(self):
        bn = BatchNorm2d(16)
        assert list(bn.weight.shape) == [16]
        assert list(bn.bias.shape) == [16]
        assert list(bn.running_mean.shape) == [16]
        assert list(bn.running_var.shape) == [16]

    def test_forward_builds_graph(self):
        bn = BatchNorm2d(16)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 16, 32, 32])

        result = bn(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "BatchNormalization") >= 1


class TestConvTranspose2d:
    """Tests for transposed 2D convolution."""

    def test_weight_shape(self):
        conv = ConvTranspose2d(16, 8, kernel_size=4, stride=2)
        assert list(conv.weight.shape) == [16, 8, 4, 4]

    def test_has_bias(self):
        conv = ConvTranspose2d(16, 8, kernel_size=4)
        assert conv.bias is not None
        assert list(conv.bias.shape) == [8]

    def test_parameter_count(self):
        conv = ConvTranspose2d(16, 8, kernel_size=4)
        params = list(conv.parameters())
        assert len(params) == 2  # weight + bias

    def test_forward_builds_graph(self):
        conv = ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 16, 8, 8])

        result = conv(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "ConvTranspose") >= 1


# ---------------------------------------------------------------------------
# CausalConvNd
# ---------------------------------------------------------------------------


class TestCausalConvNd:
    """Tests for N-d causal convolution."""

    # ── weight shapes ──────────────────────────────────────────────────────

    def test_weight_shape_1d(self):
        conv = CausalConvNd(8, 16, kernel_size=3, ndim=1)
        assert list(conv.weight.shape) == [16, 8, 3]

    def test_weight_shape_2d(self):
        conv = CausalConvNd(8, 16, kernel_size=3, ndim=2)
        assert list(conv.weight.shape) == [16, 8, 3, 3]

    def test_weight_shape_3d(self):
        conv = CausalConvNd(8, 16, kernel_size=3, ndim=3)
        assert list(conv.weight.shape) == [16, 8, 3, 3, 3]

    def test_depthwise_weight_shape(self):
        # Depthwise: groups == in_channels → weight (out, 1, ...)
        conv = CausalConvNd(8, 8, kernel_size=3, ndim=2, groups=8)
        assert list(conv.weight.shape) == [8, 1, 3, 3]

    # ── bias ───────────────────────────────────────────────────────────────

    def test_bias_present_by_default(self):
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=1)
        assert conv.bias is not None
        assert list(conv.bias.shape) == [8]

    def test_no_bias(self):
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=1, bias=False)
        assert conv.bias is None

    # ── graph construction ─────────────────────────────────────────────────

    def test_forward_1d_builds_conv_and_pad(self):
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=1)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, 16])
        result = conv(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1
        assert count_op_type(graph, "Pad") >= 1

    def test_forward_2d_builds_conv_and_pad(self):
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=2)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, 8, 8])
        result = conv(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1
        # Causal padding on last dim: pad_left = (3-1)*1 = 2 > 0 → Pad op present
        assert count_op_type(graph, "Pad") >= 1

    def test_forward_3d_builds_conv_and_pad(self):
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=3)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, 4, 4, 4])
        result = conv(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1
        assert count_op_type(graph, "Pad") >= 1

    def test_forward_kernel_size_1_no_pad(self):
        # kernel_size=1 → pad_left=0 → no Pad op
        conv = CausalConvNd(4, 8, kernel_size=1, ndim=2)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, 8, 8])
        result = conv(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Conv") >= 1
        assert count_op_type(graph, "Pad") == 0

    def test_dilation_increases_pad(self):
        # With dilation=2, pad_left = (3-1)*2 = 4 — still produces a Pad op.
        conv = CausalConvNd(4, 8, kernel_size=3, ndim=1, dilation=2)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, 16])
        result = conv(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Pad") >= 1

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="ndim must be 1, 2, or 3"):
            CausalConvNd(4, 8, kernel_size=3, ndim=4)
