# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for activation functions."""

from __future__ import annotations

import pytest

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._activations import (
    ACT2FN,
    gelu,
    get_activation,
    linear,
    quick_gelu,
    relu,
    silu,
    tanh,
)


class TestActivations:
    def _run_activation(self, act_fn):
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = act_fn(op, x)
        assert result is not None

    def test_silu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = silu(op, x)
        assert result is not None
        assert count_op_type(graph, "Sigmoid") >= 1
        assert count_op_type(graph, "Mul") >= 1

    def test_relu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = relu(op, x)
        assert result is not None
        assert count_op_type(graph, "Relu") == 1

    def test_tanh(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = tanh(op, x)
        assert result is not None
        assert count_op_type(graph, "Tanh") == 1

    def test_gelu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = gelu(op, x)
        assert result is not None
        assert graph.num_nodes() >= 1

    def test_quick_gelu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = quick_gelu(op, x)
        assert result is not None
        assert count_op_type(graph, "Sigmoid") >= 1

    def test_linear_activation(self):
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = linear(op, x)
        assert result is x  # Identity

    def test_get_activation_valid(self):
        for name in ACT2FN:
            fn = get_activation(name)
            assert callable(fn)

    def test_get_activation_invalid(self):
        with pytest.raises(KeyError):
            get_activation("nonexistent_activation")

    def test_get_activation_none_raises_value_error(self):
        with pytest.raises(ValueError, match="hidden_act is None"):
            get_activation(None)

    @pytest.mark.parametrize("act_name", list(ACT2FN.keys()))
    def test_all_activations_produce_output(self, act_name):
        act_fn = get_activation(act_name)
        self._run_activation(act_fn)
