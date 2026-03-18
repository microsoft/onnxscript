# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SelectiveScan (S6) component."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._ssm import SelectiveScan


class TestSelectiveScan:
    """Tests for SelectiveScan graph construction."""

    def test_parameters_created(self):
        """All expected parameters are created."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        params = {p.name for p in ssm.parameters()}
        # x_proj (Linear, no bias), dt_proj (Linear, with bias), A_log, D
        assert any("weight" in n for n in params)  # x_proj.weight
        assert ssm.A_log is not None
        assert ssm.D is not None

    def test_parameter_shapes(self):
        """Parameter shapes match the specified dimensions."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        assert list(ssm.A_log.shape) == [64, 16]
        assert list(ssm.D.shape) == [64]
        # x_proj: (dt_rank + 2*d_state, d_inner) = (36, 64)
        assert list(ssm.x_proj.weight.shape) == [36, 64]
        # dt_proj: (d_inner, dt_rank) = (64, 4)
        assert list(ssm.dt_proj.weight.shape) == [64, 4]

    def test_forward_builds_graph(self):
        """Forward pass constructs a valid ONNX graph."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        test_builder, op, _graph = create_test_builder()
        x = create_test_input(test_builder, "x", [2, 1, 64])
        state = create_test_input(test_builder, "ssm_state", [2, 64, 16])

        y, new_state = ssm(op, x, state)

        assert y is not None
        assert new_state is not None

    def test_discretization_ops_present(self):
        """Graph contains Exp ops for A discretization."""
        ssm = SelectiveScan(d_inner=32, d_state=8, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 32])
        state = create_test_input(test_builder, "ssm_state", [1, 32, 8])

        ssm(op, x, state)

        # Exp is used for both A discretization and softplus
        assert count_op_type(graph, "Exp") >= 1
        # Softplus for dt
        assert count_op_type(graph, "Softplus") >= 1
        # Split for dt/B/C from x_proj output
        assert count_op_type(graph, "Split") >= 1

    def test_skip_connection_ops(self):
        """Graph contains Add for skip connection (D * x)."""
        ssm = SelectiveScan(d_inner=32, d_state=8, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 32])
        state = create_test_input(test_builder, "ssm_state", [1, 32, 8])

        ssm(op, x, state)

        # Multiple Add ops: state update + skip connection
        assert count_op_type(graph, "Add") >= 2

    def test_different_d_state(self):
        """Works with different state dimensions."""
        for d_state in (4, 16, 64):
            ssm = SelectiveScan(d_inner=32, d_state=d_state, dt_rank=2)
            assert list(ssm.A_log.shape) == [32, d_state]
            # x_proj output size: dt_rank + 2*d_state
            assert list(ssm.x_proj.weight.shape) == [
                2 + 2 * d_state,
                32,
            ]

    def test_state_input_dtype(self):
        """State input accepts FLOAT dtype."""
        ssm = SelectiveScan(d_inner=16, d_state=4, dt_rank=2)
        test_builder, op, _graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 16], dtype=ir.DataType.FLOAT)
        state = create_test_input(
            test_builder, "ssm_state", [1, 16, 4], dtype=ir.DataType.FLOAT
        )

        y, _new_state = ssm(op, x, state)
        assert y is not None
