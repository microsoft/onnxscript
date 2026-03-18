# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for QuantizedLinear component."""

from __future__ import annotations

import math

import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._quantized_linear import QuantizedLinear

# Test dimensions
IN_FEATURES = 64
OUT_FEATURES = 32
BITS_4 = 4
BITS_8 = 8
BLOCK_SIZE = 32


class TestQuantizedLinearInit:
    def test_creates_packed_weight(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        blob_size = 32 * 4 // 8  # = 16
        assert ql.weight.shape == [OUT_FEATURES, n_blocks, blob_size]

    def test_creates_scales(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        assert ql.scales.shape == [OUT_FEATURES, n_blocks]

    def test_no_zero_points_by_default(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        assert ql.zero_points is None

    def test_creates_zero_points_when_requested(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True)
        n_blocks = math.ceil(IN_FEATURES / BLOCK_SIZE)
        # 4-bit: two zero-point values packed per byte
        zp_dim = math.ceil(n_blocks / 2)
        assert ql.zero_points is not None
        assert ql.zero_points.shape == [OUT_FEATURES, zp_dim]

    def test_no_bias_by_default(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        assert ql.bias is None

    def test_creates_bias_when_requested(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True)
        assert ql.bias is not None
        assert ql.bias.shape == [OUT_FEATURES]

    def test_8bit_packed_shape(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=8, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        blob_size = 32 * 8 // 8  # = 32
        assert ql.weight.shape == [OUT_FEATURES, n_blocks, blob_size]

    def test_rejects_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=3)

    def test_rejects_non_power_of_2_block_size(self):
        with pytest.raises(ValueError, match="block_size must be a power of 2 >= 16"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, block_size=48)

    def test_rejects_small_block_size(self):
        with pytest.raises(ValueError, match="block_size must be a power of 2 >= 16"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, block_size=8)


class TestQuantizedLinearForward:
    def test_graph_has_matmulnbits_node(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "MatMulNBits") == 1

    def test_matmulnbits_domain_is_microsoft(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert node.domain == "com.microsoft"
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_matmulnbits_attributes(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["K"] == IN_FEATURES
                assert attrs["N"] == OUT_FEATURES
                assert attrs["bits"] == 4
                assert attrs["block_size"] == 32
                break

    def test_3_inputs_without_zero_points(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert len(node.inputs) == 3
                break

    def test_4_inputs_with_zero_points(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert len(node.inputs) == 4
                break

    def test_bias_adds_add_node(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "MatMulNBits") == 1
        assert count_op_type(graph, "Add") == 1

    def test_no_add_without_bias(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Add") == 0

    def test_8bit_forward(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=8, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result])
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["bits"] == 8
                break

    def test_parameter_names(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True, bias=True)
        names = [n for n, _ in ql.named_parameters()]
        assert "weight" in names
        assert "scales" in names
        assert "zero_points" in names
        assert "bias" in names


class TestMakeQuantizedLinearFactory:
    """Tests for the make_quantized_linear_factory closure."""

    def test_factory_returns_class(self):
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=32)
        assert isinstance(factory, type)

    def test_factory_creates_quantized_linear(self):
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=32, has_zero_point=True)
        instance = factory(64, 128)
        assert isinstance(instance, QuantizedLinear)
        assert instance._bits == 4
        assert instance._block_size == 32
        assert instance.zero_points is not None

    def test_factory_matches_linear_signature(self):
        """Factory class must accept (in_features, out_features, bias=True)."""
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=128)
        instance = factory(32, 64, bias=False)
        assert instance._k == 32
        assert instance._n == 64
        assert instance.bias is None
