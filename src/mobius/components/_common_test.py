# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Linear, Embedding, and attention bias utilities."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._common import (
    Embedding,
    Linear,
    create_attention_bias,
    create_padding_mask,
)


class TestLinear:
    def test_linear_with_bias(self):
        linear = Linear(64, 128, bias=True)
        params = list(linear.parameters())
        assert len(params) == 2  # weight + bias
        assert list(linear.weight.shape) == [128, 64]
        assert list(linear.bias.shape) == [128]

    def test_linear_without_bias(self):
        linear = Linear(64, 128, bias=False)
        params = list(linear.parameters())
        assert len(params) == 1  # weight only
        assert linear.bias is None

    def test_linear_forward(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        linear = Linear(64, 128, bias=True)
        result = linear(op, x)
        assert result is not None
        assert count_op_type(graph, "Transpose") >= 1
        assert count_op_type(graph, "MatMul") >= 1
        assert count_op_type(graph, "Add") >= 1

    def test_linear_no_bias_forward(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        linear = Linear(64, 128, bias=False)
        result = linear(op, x)
        assert result is not None
        assert count_op_type(graph, "MatMul") >= 1
        assert count_op_type(graph, "Add") == 0


class TestEmbedding:
    def test_embedding_params(self):
        emb = Embedding(1000, 64)
        params = list(emb.parameters())
        assert len(params) == 1
        assert list(emb.weight.shape) == [1000, 64]

    def test_embedding_forward(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        emb = Embedding(1000, 64)
        result = emb(op, input_ids)
        assert result is not None
        assert count_op_type(graph, "Gather") >= 1

    def test_embedding_with_padding_idx(self):
        emb = Embedding(1000, 64, padding_idx=0)
        assert emb.padding_idx == 0


class TestCreateAttentionBias:
    def test_creates_bias(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        bias = create_attention_bias(op, input_ids, attention_mask)
        assert bias is not None
        assert graph.num_nodes() > 0

    def test_creates_bias_with_sliding_window(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        bias = create_attention_bias(op, input_ids, attention_mask, sliding_window=4)
        assert bias is not None
        assert count_op_type(graph, "Less") >= 1


class TestCreatePaddingMask:
    def test_creates_bool_mask_with_2d_input_ids(self):
        """Standard path: input_ids is 2D [batch, q_len]."""
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        mask = create_padding_mask(op, input_ids, attention_mask)
        assert mask is not None
        assert count_op_type(graph, "Cast") >= 1
        assert count_op_type(graph, "Unsqueeze") >= 1
        assert count_op_type(graph, "Expand") >= 1

    def test_creates_bool_mask_with_3d_hidden_states(self):
        """inputs_embeds path: input_ids is actually 3D hidden_states."""
        builder, op, graph = create_test_builder()
        hidden_states = create_test_input(
            builder, "hidden_states", [2, 4, 256], dtype=ir.DataType.FLOAT
        )
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        mask = create_padding_mask(op, hidden_states, attention_mask)
        assert mask is not None
        # Should still produce a valid graph (no crash from 3D input).
        assert count_op_type(graph, "Cast") >= 1
        assert count_op_type(graph, "Expand") >= 1

    def test_uses_simpler_ops_than_attention_bias(self):
        """Padding mask uses simple broadcast ops, not the causal CumSum chain."""
        builder_pad, op_pad, graph_pad = create_test_builder()
        input_ids_pad = create_test_input(
            builder_pad, "input_ids", [2, 4], dtype=ir.DataType.INT64
        )
        mask_pad = create_test_input(
            builder_pad, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        create_padding_mask(op_pad, input_ids_pad, mask_pad)

        # Padding mask avoids the CumSum/GreaterOrEqual/Where causal-bias chain
        assert count_op_type(graph_pad, "Cast") >= 1
        assert count_op_type(graph_pad, "Unsqueeze") >= 1
        assert count_op_type(graph_pad, "Expand") >= 1
        assert count_op_type(graph_pad, "CumSum") == 0
        assert count_op_type(graph_pad, "GreaterOrEqual") == 0
        assert count_op_type(graph_pad, "Where") == 0

        builder_bias, op_bias, graph_bias = create_test_builder()
        input_ids_bias = create_test_input(
            builder_bias, "input_ids", [2, 4], dtype=ir.DataType.INT64
        )
        mask_bias = create_test_input(
            builder_bias, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        create_attention_bias(op_bias, input_ids_bias, mask_bias)

        # Full causal bias requires CumSum + GreaterOrEqual + Where chain
        assert count_op_type(graph_bias, "CumSum") >= 1
        assert count_op_type(graph_bias, "GreaterOrEqual") >= 1
        assert count_op_type(graph_bias, "Where") >= 1
