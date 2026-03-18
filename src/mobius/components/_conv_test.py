# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for convolution components."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
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
