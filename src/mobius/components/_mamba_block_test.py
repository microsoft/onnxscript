# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MambaBlock component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._mamba_block import MambaBlock


class TestMambaBlock:
    """Tests for MambaBlock graph construction."""

    def test_default_dt_rank(self):
        """dt_rank defaults to ceil(d_model / 16)."""
        block = MambaBlock(d_model=64, d_inner=128)
        # ceil(64 / 16) = 4
        assert block.dt_rank == 4

    def test_default_dt_rank_non_divisible(self):
        """dt_rank rounds up for non-divisible d_model."""
        block = MambaBlock(d_model=100, d_inner=200)
        # ceil(100 / 16) = 7
        assert block.dt_rank == 7

    def test_custom_dt_rank(self):
        """dt_rank can be overridden."""
        block = MambaBlock(d_model=64, d_inner=128, dt_rank=8)
        assert block.dt_rank == 8

    def test_parameters_created(self):
        """All sub-module parameters are created."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16)
        params = list(block.parameters())
        # in_proj.weight, conv1d.weight, conv1d.bias,
        # ssm.x_proj.weight, ssm.dt_proj.weight, ssm.dt_proj.bias,
        # ssm.A_log, ssm.D, out_proj.weight
        assert len(params) == 9

    def test_in_proj_shape(self):
        """Input projection maps d_model → 2*d_inner."""
        block = MambaBlock(d_model=64, d_inner=128)
        # in_proj: (2*d_inner, d_model) = (256, 64)
        assert list(block.in_proj.weight.shape) == [256, 64]

    def test_out_proj_shape(self):
        """Output projection maps d_inner → d_model."""
        block = MambaBlock(d_model=64, d_inner=128)
        # out_proj: (d_model, d_inner) = (64, 128)
        assert list(block.out_proj.weight.shape) == [64, 128]

    def test_conv1d_shape(self):
        """Conv1D weight has correct shape."""
        block = MambaBlock(d_model=64, d_inner=128, conv_kernel=4)
        # conv1d.weight: (d_inner, 1, conv_kernel) = (128, 1, 4)
        assert list(block.conv1d.weight.shape) == [128, 1, 4]

    def test_forward_builds_graph(self):
        """Forward pass constructs a valid ONNX graph."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16, conv_kernel=4)
        test_builder, op, _graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [2, 1, 64])
        conv_state = create_test_input(test_builder, "conv_state", [2, 128, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [2, 128, 16])

        output, new_conv, new_ssm = block(op, hidden, conv_state, ssm_state)

        assert output is not None
        assert new_conv is not None
        assert new_ssm is not None

    def test_conv_op_present(self):
        """Graph contains Conv op from depthwise conv1d."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        assert count_op_type(graph, "Conv") >= 1

    def test_split_ops_present(self):
        """Graph contains Split ops for in_proj and SSM projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # Two splits: in_proj (x/z) and SSM x_proj (dt/B/C)
        assert count_op_type(graph, "Split") >= 2

    def test_silu_activation_via_sigmoid(self):
        """SiLU (x * sigmoid(x)) produces Sigmoid + Mul ops."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # SiLU appears in conv path + z gating = at least 2 Sigmoid ops
        assert count_op_type(graph, "Sigmoid") >= 2

    def test_matmul_ops_for_projections(self):
        """Graph contains MatMul ops for linear projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # in_proj, x_proj, dt_proj, out_proj = at least 4 MatMul ops
        assert count_op_type(graph, "MatMul") >= 4

    def test_custom_conv_kernel(self):
        """Custom conv_kernel size is reflected in parameters."""
        block = MambaBlock(d_model=32, d_inner=64, conv_kernel=8)
        assert list(block.conv1d.weight.shape) == [64, 1, 8]
        assert block.conv_kernel == 8
