# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MLP component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._mlp import MLP


class TestMLP:
    """Tests for the gated MLP (SwiGLU-style) component."""

    def test_projection_shapes(self):
        config = make_config()
        mlp = MLP(config)
        # gate_proj: (intermediate_size, hidden_size) = (128, 64)
        assert list(mlp.gate_proj.weight.shape) == [128, 64]
        # up_proj: same
        assert list(mlp.up_proj.weight.shape) == [128, 64]
        # down_proj: (hidden_size, intermediate_size) = (64, 128)
        assert list(mlp.down_proj.weight.shape) == [64, 128]

    def test_parameter_count_no_bias(self):
        config = make_config()
        mlp = MLP(config)
        params = list(mlp.parameters())
        # gate_proj.weight, up_proj.weight, down_proj.weight = 3
        assert len(params) == 3

    def test_parameter_count_with_bias(self):
        config = make_config(mlp_bias=True)
        mlp = MLP(config)
        params = list(mlp.parameters())
        # 3 weights + 3 biases = 6
        assert len(params) == 6

    def test_forward_builds_graph(self):
        config = make_config()
        mlp = MLP(config)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        result = mlp(op, x)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0
        # gate_proj + up_proj + down_proj = at least 3 MatMul ops
        assert count_op_type(graph, "MatMul") >= 3

    def test_forward_has_mul_for_gating(self):
        config = make_config()
        mlp = MLP(config)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        mlp(op, x)
        # gate * up = at least 1 Mul op
        assert count_op_type(graph, "Mul") >= 1

    def test_silu_activation(self):
        config = make_config(hidden_act="silu")
        mlp = MLP(config)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        mlp(op, x)
        # SiLU = x * sigmoid(x): Sigmoid + Mul
        assert count_op_type(graph, "Sigmoid") >= 1

    def test_gelu_activation(self):
        config = make_config(hidden_act="gelu")
        mlp = MLP(config)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        mlp(op, x)
        assert count_op_type(graph, "Gelu") >= 1

    def test_different_intermediate_size(self):
        config = make_config(intermediate_size=256)
        mlp = MLP(config)
        assert list(mlp.gate_proj.weight.shape) == [256, 64]
        assert list(mlp.down_proj.weight.shape) == [64, 256]
