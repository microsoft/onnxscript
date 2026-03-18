# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MoE components."""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._testing import (
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._moe import MoELayer, SparseMixerGate, TopKGate


class TestTopKGate:
    def test_gate_has_weight_parameter(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        param_names = [n for n, _ in gate.named_parameters()]
        assert "weight" in param_names

    def test_gate_weight_shape(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        assert gate.weight.shape == ir.Shape([4, 64])

    def test_gate_forward_produces_outputs(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts])
        assert graph.num_nodes() > 0


class TestSparseMixerGate:
    def test_gate_has_weight_parameter(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        param_names = [n for n, _ in gate.named_parameters()]
        assert "weight" in param_names

    def test_gate_weight_shape(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        assert gate.weight.shape == ir.Shape([4, 64])

    def test_gate_forward_produces_outputs(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts])
        assert graph.num_nodes() > 0

    def test_gate_custom_jitter_eps(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2, jitter_eps=0.05)
        assert gate.jitter_eps == pytest.approx(0.05)

    def test_gate_top_k_1(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=1)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts])
        assert graph.num_nodes() > 0


class TestMoELayer:
    def test_moe_layer_has_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("gate" in n for n in param_names)

    def test_moe_layer_has_experts(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("experts.0" in n for n in param_names)
        assert any("experts.3" in n for n in param_names)

    def test_moe_layer_num_experts(self):
        config = make_config(num_local_experts=8, num_experts_per_tok=2)
        layer = MoELayer(config)
        assert len(layer.experts) == 8

    def test_moe_layer_forward(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        result = layer(op, hidden)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_moe_layer_requires_expert_config(self):
        config = make_config()  # No MoE config
        with pytest.raises(AssertionError):
            MoELayer(config)

    def test_moe_layer_with_custom_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        gate = SparseMixerGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        layer = MoELayer(config, gate=gate)
        assert isinstance(layer.gate, SparseMixerGate)

    def test_moe_layer_forward_with_sparse_mixer_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        gate = SparseMixerGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        layer = MoELayer(config, gate=gate)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        result = layer(op, hidden)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0
