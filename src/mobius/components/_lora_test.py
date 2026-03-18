# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LoRALinear component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._lora import LoRALinear


class TestLoRALinear:
    """Tests for Linear layer with LoRA adapters."""

    def test_no_adapters_is_plain_linear(self):
        layer = LoRALinear(64, 128)
        params = list(layer.parameters())
        # Just weight, no bias by default
        assert len(params) == 1

    def test_single_adapter_parameters(self):
        layer = LoRALinear(64, 128, lora_adapters=[("default", 8, 1.0)])
        param_names = [n for n, _ in layer.named_parameters()]
        assert "_lora_A_default" in param_names
        assert "_lora_B_default" in param_names
        assert "_lora_scale_default" in param_names

    def test_adapter_shapes(self):
        layer = LoRALinear(64, 128, lora_adapters=[("default", 8, 1.0)])
        # lora_A: (rank, in_features) = (8, 64)
        assert list(layer._lora_A_default.shape) == [8, 64]
        # lora_B: (out_features, rank) = (128, 8)
        assert list(layer._lora_B_default.shape) == [128, 8]

    def test_multiple_adapters(self):
        layer = LoRALinear(
            64,
            128,
            lora_adapters=[
                ("vision", 8, 1.0),
                ("speech", 4, 0.5),
            ],
        )
        assert len(layer._adapters) == 2
        assert hasattr(layer, "_lora_A_vision")
        assert hasattr(layer, "_lora_A_speech")
        assert list(layer._lora_A_speech.shape) == [4, 64]

    def test_forward_no_adapters(self):
        layer = LoRALinear(64, 128)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        result = layer(op, x)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "MatMul") >= 1

    def test_forward_with_adapter(self):
        layer = LoRALinear(64, 128, lora_adapters=[("default", 8, 1.0)])
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        result = layer(op, x)
        builder._adapt_outputs([result])
        # Base MatMul + lora_A MatMul + lora_B MatMul = at least 3
        assert count_op_type(graph, "MatMul") >= 3
        # scale Mul + Add for residual
        assert count_op_type(graph, "Add") >= 1

    def test_forward_multiple_adapters(self):
        layer = LoRALinear(
            64,
            128,
            lora_adapters=[
                ("vision", 8, 1.0),
                ("speech", 4, 0.5),
            ],
        )
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 8, 64])

        result = layer(op, x)
        builder._adapt_outputs([result])
        # Base + 2*(lora_A + lora_B) = 5 MatMul ops
        assert count_op_type(graph, "MatMul") >= 5

    def test_with_bias(self):
        layer = LoRALinear(64, 128, bias=True, lora_adapters=[("default", 8, 1.0)])
        assert layer.bias is not None
