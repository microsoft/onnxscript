# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Q-Former component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._qformer import (
    QFormer,
    QFormerAttention,
    QFormerLayer,
)

# Test dimensions
HIDDEN = 64
NUM_HEADS = 4
INTERMEDIATE = 128
NUM_QUERIES = 8
NUM_PATCHES = 16
NUM_LAYERS = 2
ENCODER_HIDDEN = 96  # Different from HIDDEN to test cross-dim


class TestQFormerAttention:
    def test_has_projections(self):
        attn = QFormerAttention(hidden_size=HIDDEN, num_attention_heads=NUM_HEADS)
        param_names = [n for n, _ in attn.named_parameters()]
        assert any("q_proj" in n for n in param_names)
        assert any("k_proj" in n for n in param_names)
        assert any("v_proj" in n for n in param_names)
        assert any("out_proj" in n for n in param_names)

    def test_self_attention_forward(self):
        attn = QFormerAttention(hidden_size=HIDDEN, num_attention_heads=NUM_HEADS)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, NUM_QUERIES, HIDDEN])
        result = attn(op, hidden)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_self_attention_uses_attention_op(self):
        attn = QFormerAttention(hidden_size=HIDDEN, num_attention_heads=NUM_HEADS)
        b, op, graph = create_test_builder()
        hidden = create_test_input(b, "hidden", [1, NUM_QUERIES, HIDDEN])
        result = attn(op, hidden)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Attention") == 1

    def test_cross_attention_forward(self):
        """Cross-attention: Q from queries, K/V from encoder features."""
        attn = QFormerAttention(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            kv_hidden_size=ENCODER_HIDDEN,
        )
        b, op, graph = create_test_builder()
        queries = create_test_input(b, "queries", [1, NUM_QUERIES, HIDDEN])
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, ENCODER_HIDDEN])
        result = attn(op, queries, key_value_states=visual)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Attention") == 1

    def test_cross_attention_same_hidden_size(self):
        """Cross-attention where encoder and Q-Former share hidden size."""
        attn = QFormerAttention(hidden_size=HIDDEN, num_attention_heads=NUM_HEADS)
        b, op, graph = create_test_builder()
        queries = create_test_input(b, "queries", [1, NUM_QUERIES, HIDDEN])
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, HIDDEN])
        result = attn(op, queries, key_value_states=visual)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0


class TestQFormerLayer:
    def test_has_self_attn_cross_attn_mlp(self):
        layer = QFormerLayer(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("self_attn" in n for n in param_names)
        assert any("cross_attn" in n for n in param_names)
        assert any("mlp" in n for n in param_names)
        assert any("self_attn_layernorm" in n for n in param_names)
        assert any("cross_attn_layernorm" in n for n in param_names)
        assert any("mlp_layernorm" in n for n in param_names)

    def test_forward(self):
        layer = QFormerLayer(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        b, op, graph = create_test_builder()
        queries = create_test_input(b, "queries", [1, NUM_QUERIES, HIDDEN])
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, HIDDEN])
        result = layer(op, queries, visual)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_forward_different_encoder_hidden_size(self):
        """Encoder hidden size differs from Q-Former hidden size."""
        layer = QFormerLayer(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
            encoder_hidden_size=ENCODER_HIDDEN,
        )
        b, op, graph = create_test_builder()
        queries = create_test_input(b, "queries", [1, NUM_QUERIES, HIDDEN])
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, ENCODER_HIDDEN])
        result = layer(op, queries, visual)
        b._adapt_outputs([result])
        # Self-attention + cross-attention = 2 Attention ops
        assert count_op_type(graph, "Attention") == 2

    def test_has_three_layernorms(self):
        layer = QFormerLayer(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        param_names = [n for n, _ in layer.named_parameters()]
        layernorm_names = [n for n in param_names if "layernorm" in n]
        # 3 norms x 2 params each (weight + bias) = 6
        assert len(layernorm_names) == 6


class TestQFormer:
    def test_has_query_tokens_and_layers(self):
        qformer = QFormer(
            num_query_tokens=NUM_QUERIES,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        param_names = [n for n, _ in qformer.named_parameters()]
        assert any("query_tokens" in n for n in param_names)
        assert any("layers" in n for n in param_names)
        assert any("layernorm" in n for n in param_names)
        assert len(qformer.layers) == NUM_LAYERS

    def test_forward(self):
        qformer = QFormer(
            num_query_tokens=NUM_QUERIES,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        b, op, graph = create_test_builder()
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, HIDDEN])
        result = qformer(op, visual)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_forward_with_different_encoder_hidden_size(self):
        qformer = QFormer(
            num_query_tokens=NUM_QUERIES,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
            encoder_hidden_size=ENCODER_HIDDEN,
        )
        b, op, graph = create_test_builder()
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, ENCODER_HIDDEN])
        result = qformer(op, visual)
        b._adapt_outputs([result])
        # 2 layers x 2 attn ops each = 4 Attention ops
        assert count_op_type(graph, "Attention") == NUM_LAYERS * 2

    def test_attention_op_count(self):
        """Each layer has self-attn + cross-attn = 2 Attention ops."""
        qformer = QFormer(
            num_query_tokens=NUM_QUERIES,
            num_layers=3,
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE,
        )
        b, op, graph = create_test_builder()
        visual = create_test_input(b, "visual", [1, NUM_PATCHES, HIDDEN])
        result = qformer(op, visual)
        b._adapt_outputs([result])
        assert count_op_type(graph, "Attention") == 6  # 3 layers x 2
