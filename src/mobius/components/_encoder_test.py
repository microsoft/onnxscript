# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for encoder components (EncoderAttention, EncoderLayer, BertEmbeddings)."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._encoder import (
    BertEmbeddings,
    EncoderAttention,
    EncoderLayer,
)


class TestEncoderAttention:
    """Tests for bidirectional encoder attention."""

    def test_projection_shapes(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4)
        assert list(attn.q_proj.weight.shape) == [64, 64]
        assert list(attn.k_proj.weight.shape) == [64, 64]
        assert list(attn.v_proj.weight.shape) == [64, 64]
        assert list(attn.out_proj.weight.shape) == [64, 64]

    def test_has_bias_by_default(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4)
        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None

    def test_no_bias(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4, bias=False)
        assert attn.q_proj.bias is None

    def test_parameter_count_with_bias(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4, bias=True)
        params = list(attn.parameters())
        # 4 weights + 4 biases = 8
        assert len(params) == 8

    def test_forward_builds_graph(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        result = attn(op, hidden)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_attention_mask(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        mask = create_test_input(builder, "mask", [1, 1, 8, 8])

        result = attn(op, hidden, attention_mask=mask)
        builder._adapt_outputs([result])
        assert count_op_type(graph, "Attention") >= 1

    def test_head_dim_computed(self):
        attn = EncoderAttention(hidden_size=64, num_attention_heads=4)
        assert attn.head_dim == 16


class TestEncoderLayer:
    """Tests for post-norm BERT-style encoder layer."""

    def test_has_submodules(self):
        layer = EncoderLayer(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
        )
        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")
        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "post_mlp_layernorm")

    def test_forward_builds_graph(self):
        layer = EncoderLayer(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
        )
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        result = layer(op, hidden)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0
        assert count_op_type(graph, "Attention") >= 1
        # Post-norm: two LayerNormalization ops
        assert count_op_type(graph, "LayerNormalization") >= 2

    def test_forward_with_mask(self):
        layer = EncoderLayer(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
        )
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        mask = create_test_input(builder, "mask", [1, 1, 8, 8])

        result = layer(op, hidden, attention_mask=mask)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_residual_connections(self):
        layer = EncoderLayer(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
        )
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        layer(op, hidden)
        # Two residual Adds (attn + mlp)
        assert count_op_type(graph, "Add") >= 2

    def test_custom_activation(self):
        layer = EncoderLayer(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act="relu",
        )
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])

        layer(op, hidden)
        assert count_op_type(graph, "Relu") >= 1


class TestBertEmbeddings:
    """Tests for BERT embeddings (word + position + token_type)."""

    def test_has_three_embeddings(self):
        emb = BertEmbeddings(
            vocab_size=100,
            hidden_size=64,
            max_position_embeddings=32,
        )
        assert hasattr(emb, "word_embeddings")
        assert hasattr(emb, "position_embeddings")
        assert hasattr(emb, "token_type_embeddings")
        assert hasattr(emb, "layernorm")

    def test_embedding_shapes(self):
        emb = BertEmbeddings(
            vocab_size=100,
            hidden_size=64,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
        assert list(emb.word_embeddings.weight.shape) == [100, 64]
        assert list(emb.position_embeddings.weight.shape) == [32, 64]
        assert list(emb.token_type_embeddings.weight.shape) == [2, 64]

    def test_forward_builds_graph(self):
        emb = BertEmbeddings(
            vocab_size=100,
            hidden_size=64,
            max_position_embeddings=32,
        )
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [1, 8], dtype=ir.DataType.INT64)
        token_type_ids = create_test_input(
            builder, "token_type_ids", [1, 8], dtype=ir.DataType.INT64
        )

        result = emb(op, input_ids, token_type_ids)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0
        # Three Gather ops (word + position + token_type)
        assert count_op_type(graph, "Gather") >= 3
        # LayerNorm at the end
        assert count_op_type(graph, "LayerNormalization") >= 1
