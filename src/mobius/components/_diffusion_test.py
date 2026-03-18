# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared diffusion transformer components."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._diffusion import (
    AdaLayerNormOutput,
    AdaLayerNormZero,
    DiffusionFFN,
    DiffusionSelfAttention,
    PatchEmbed,
    TimestepEmbedding,
)


class TestAdaLayerNormZero:
    """Tests for AdaLayerNormZero component."""

    def test_parameters_created(self):
        """Norm and linear parameters exist."""
        mod = AdaLayerNormZero(hidden_size=64)
        params = dict(mod.named_parameters())
        assert "norm.weight" in params
        assert "norm.bias" in params
        assert "linear.weight" in params
        assert "linear.bias" in params

    def test_forward_produces_7_outputs(self):
        """Forward returns (normed, shift, scale, gate) x2 = 7 values."""
        mod = AdaLayerNormZero(hidden_size=64)
        builder, op, _graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64], ir.DataType.FLOAT)
        temb = create_test_input(builder, "temb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, hidden, temb)
        assert len(result) == 7

    def test_graph_contains_split(self):
        """Forward should Split the 6*C modulation into 6 chunks."""
        mod = AdaLayerNormZero(hidden_size=64)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64], ir.DataType.FLOAT)
        temb = create_test_input(builder, "temb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, hidden, temb)
        graph.outputs.extend(result)
        assert count_op_type(graph, "Split") >= 1

    def test_graph_contains_layer_norm(self):
        """Forward should include LayerNormalization for the norm."""
        mod = AdaLayerNormZero(hidden_size=64)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64], ir.DataType.FLOAT)
        temb = create_test_input(builder, "temb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, hidden, temb)
        graph.outputs.extend(result)
        assert count_op_type(graph, "LayerNormalization") >= 1


class TestAdaLayerNormOutput:
    """Tests for AdaLayerNormOutput component."""

    def test_parameters_created(self):
        """Norm and linear parameters exist."""
        mod = AdaLayerNormOutput(hidden_size=64)
        params = dict(mod.named_parameters())
        assert "norm.weight" in params
        assert "linear.weight" in params

    def test_forward_returns_single_output(self):
        """Forward returns a single modulated output."""
        mod = AdaLayerNormOutput(hidden_size=64)
        builder, op, _graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64], ir.DataType.FLOAT)
        temb = create_test_input(builder, "temb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, hidden, temb)
        # Should return a single value, not a tuple
        assert not isinstance(result, tuple)

    def test_graph_contains_split(self):
        """Scale and shift are split from linear output."""
        mod = AdaLayerNormOutput(hidden_size=64)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 64], ir.DataType.FLOAT)
        temb = create_test_input(builder, "temb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, hidden, temb)
        graph.outputs.append(result)
        assert count_op_type(graph, "Split") >= 1


class TestPatchEmbed:
    """Tests for PatchEmbed component."""

    def test_parameters_created(self):
        """Conv2d projection parameters exist."""
        mod = PatchEmbed(in_channels=4, hidden_size=64, patch_size=2)
        params = dict(mod.named_parameters())
        assert "proj.weight" in params

    def test_forward_produces_sequence(self):
        """Output should be transposed to [B, num_patches, hidden]."""
        mod = PatchEmbed(in_channels=4, hidden_size=64, patch_size=2)
        builder, op, graph = create_test_builder()
        sample = create_test_input(builder, "sample", [1, 4, 8, 8], ir.DataType.FLOAT)
        result = mod(op, sample)
        graph.outputs.append(result)
        # Should contain Conv, Reshape, Transpose
        assert count_op_type(graph, "Conv") >= 1
        assert count_op_type(graph, "Transpose") >= 1


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding component."""

    def test_parameters_created(self):
        """Two linear layers' parameters exist."""
        mod = TimestepEmbedding(in_channels=64, time_embed_dim=128)
        params = dict(mod.named_parameters())
        assert "linear_1.weight" in params
        assert "linear_2.weight" in params

    def test_forward_graph(self):
        """Forward should apply linear → SiLU → linear."""
        mod = TimestepEmbedding(in_channels=64, time_embed_dim=128)
        builder, op, graph = create_test_builder()
        t_emb = create_test_input(builder, "t_emb", [1, 64], ir.DataType.FLOAT)
        result = mod(op, t_emb)
        graph.outputs.append(result)
        # Two MatMul ops for the two Linear layers
        assert count_op_type(graph, "MatMul") >= 2


class TestDiffusionFFN:
    """Tests for DiffusionFFN component."""

    def test_parameters_created(self):
        """Two linear layers' parameters exist."""
        mod = DiffusionFFN(hidden_size=64)
        params = dict(mod.named_parameters())
        assert "linear_1.weight" in params
        assert "linear_2.weight" in params

    def test_default_intermediate_size(self):
        """Default intermediate size is 4x hidden."""
        mod = DiffusionFFN(hidden_size=64)
        # linear_1 projects from 64 to 256 (4x)
        shape = mod.linear_1.weight.shape
        assert shape is not None
        assert 256 in list(shape)

    def test_custom_intermediate_size(self):
        """Custom intermediate size is respected."""
        mod = DiffusionFFN(hidden_size=64, intermediate_size=128)
        shape = mod.linear_1.weight.shape
        assert shape is not None
        assert 128 in list(shape)

    def test_forward_graph(self):
        """Forward should apply linear → GELU → linear."""
        mod = DiffusionFFN(hidden_size=64)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 4, 64], ir.DataType.FLOAT)
        result = mod(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "Gelu") >= 1


class TestDiffusionSelfAttention:
    """Tests for DiffusionSelfAttention component."""

    def test_parameters_created(self):
        """Q, K, V and output projection parameters exist."""
        mod = DiffusionSelfAttention(hidden_size=64, num_heads=4)
        params = dict(mod.named_parameters())
        assert "to_q.weight" in params
        assert "to_k.weight" in params
        assert "to_v.weight" in params
        assert any("to_out" in n for n in params)

    def test_forward_graph(self):
        """Forward should contain an Attention op."""
        mod = DiffusionSelfAttention(hidden_size=64, num_heads=4)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 4, 64], ir.DataType.FLOAT)
        result = mod(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "Attention") >= 1
