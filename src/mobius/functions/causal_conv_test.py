# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CausalConvWithState ir.Function builder."""

from __future__ import annotations

import pytest

from mobius.functions.causal_conv import causal_conv_nd_with_state


class TestCausalConvNdWithState:
    """Tests for the causal_conv_nd_with_state ir.Function builder."""

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_function_has_four_inputs(self, ndim: int):
        func = causal_conv_nd_with_state(kernel_size=4, channels=16, ndim=ndim)
        assert len(func.graph.inputs) == 4
        names = [inp.name for inp in func.graph.inputs]
        assert names == ["input", "weight", "bias", "conv_state"]

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_function_has_two_outputs(self, ndim: int):
        func = causal_conv_nd_with_state(kernel_size=4, channels=16, ndim=ndim)
        assert len(func.graph.outputs) == 2
        names = [out.name for out in func.graph.outputs]
        assert names == ["output", "present_state"]

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_function_contains_conv_node(self, ndim: int):
        func = causal_conv_nd_with_state(kernel_size=3, channels=8, ndim=ndim)
        op_types = [node.op_type for node in func.graph]
        assert "Conv" in op_types

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_function_contains_concat_for_state_prepend(self, ndim: int):
        func = causal_conv_nd_with_state(kernel_size=3, channels=8, ndim=ndim)
        op_types = [node.op_type for node in func.graph]
        assert "Concat" in op_types

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_kernel_size_1_builds(self, ndim: int):
        """kernel_size=1 means state_width=0 — state is empty but graph must build."""
        func = causal_conv_nd_with_state(kernel_size=1, channels=4, ndim=ndim)
        assert len(func.graph.inputs) == 4
        assert len(func.graph.outputs) == 2

    @pytest.mark.parametrize("activation", ["silu", "swish", "none"])
    def test_activations(self, activation: str):
        func = causal_conv_nd_with_state(
            kernel_size=4, channels=8, ndim=1, activation=activation
        )
        op_types = [node.op_type for node in func.graph]
        if activation in ("silu", "swish"):
            assert "Sigmoid" in op_types
        else:
            assert "Sigmoid" not in op_types

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="ndim must be 1, 2, or 3"):
            causal_conv_nd_with_state(kernel_size=4, channels=8, ndim=4)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            causal_conv_nd_with_state(kernel_size=4, channels=8, activation="relu")

    def test_function_op_name_is_causal_conv_with_state(self):
        """Op type name must match call sites (op.CausalConvWithState)."""
        func = causal_conv_nd_with_state(kernel_size=4, channels=8, ndim=1)
        assert func.name == "CausalConvWithState"

    def test_function_domain(self):
        func = causal_conv_nd_with_state(kernel_size=4, channels=8, ndim=1)
        assert func.domain == "com.microsoft"
