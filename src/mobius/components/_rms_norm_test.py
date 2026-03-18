# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RMSNorm component."""

from __future__ import annotations

from onnxscript import nn

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._rms_norm import RMSNorm, apply_rms_norm


class TestRMSNorm:
    def test_rms_norm_creates_parameters(self):
        norm = RMSNorm(64, eps=1e-6)
        params = list(norm.parameters())
        # Should have weight only (eps is now a float attribute, not a parameter)
        assert len(params) == 1

    def test_rms_norm_weight_shape(self):
        norm = RMSNorm(128)
        assert list(norm.weight.shape) == [128]

    def test_rms_norm_forward_builds_graph(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        norm = RMSNorm(64, eps=1e-5)
        result = norm(op, x)
        assert result is not None
        # Should use the ONNX RMSNormalization op
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_rms_norm_different_eps(self):
        norm1 = RMSNorm(64, eps=1e-5)
        norm2 = RMSNorm(64, eps=1e-6)
        assert norm1.variance_epsilon != norm2.variance_epsilon

    def test_apply_rms_norm_function(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        weight = nn.Parameter([64], name="test_weight")
        weight._realize(builder)

        result = apply_rms_norm(op, x, weight, 1e-6)
        assert result is not None
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_multiple_rms_norms_in_same_graph(self):
        """Ensure two RMSNorm modules can coexist in the same graph."""
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])

        norm1 = RMSNorm(64, eps=1e-6)
        norm2 = RMSNorm(64, eps=1e-6)

        # Manually set names to avoid collision
        builder.push_module("norm1")
        for p in norm1._parameters.values():
            p._realize(builder)
        r1 = norm1.forward(op, x)
        builder.pop_module()

        builder.push_module("norm2")
        for p in norm2._parameters.values():
            p._realize(builder)
        r2 = norm2.forward(op, x)
        builder.pop_module()

        assert r1 is not None
        assert r2 is not None
        assert count_op_type(graph, "RMSNormalization") == 2
