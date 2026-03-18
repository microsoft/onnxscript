# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Scan subgraph utilities."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import create_test_builder, create_test_input
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)


class TestCreateBodyGraph:
    """Tests for create_body_graph helper."""

    def test_returns_graph_and_builder(self):
        state = ir.Value(
            name="state", shape=ir.Shape([1, 8]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        scan_in = ir.Value(
            name="scan_in", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.INT64)
        )
        graph, _builder = create_body_graph([state], [scan_in])
        assert isinstance(graph, ir.Graph)
        assert len(graph.inputs) == 2

    def test_inputs_are_state_plus_scan(self):
        s1 = ir.Value(name="s1", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.FLOAT))
        s2 = ir.Value(name="s2", shape=ir.Shape([3]), type=ir.TensorType(ir.DataType.FLOAT))
        si = ir.Value(name="si", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))
        graph, _ = create_body_graph([s1, s2], [si])
        assert len(graph.inputs) == 3
        assert graph.inputs[0].name == "s1"
        assert graph.inputs[1].name == "s2"
        assert graph.inputs[2].name == "si"

    def test_custom_name(self):
        state = ir.Value(name="s", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.FLOAT))
        graph, _ = create_body_graph([state], [], name="my_body")
        assert graph.name == "my_body"

    def test_empty_scan_inputs(self):
        state = ir.Value(name="s", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.FLOAT))
        graph, _ = create_body_graph([state], [])
        assert len(graph.inputs) == 1


class TestRenameSubgraphValues:
    """Tests for rename_subgraph_values collision avoidance."""

    def test_prefixes_node_names(self):
        state = ir.Value(name="s", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.FLOAT))
        graph, body_builder = create_body_graph([state], [])
        body_op = body_builder.op
        # Add a simple op
        result = body_op.Identity(state)
        result.name = "result_out"
        graph.outputs.append(result)

        rename_subgraph_values(graph, "body_")

        node_names = [n.name for n in graph]
        assert all(n.startswith("body_") for n in node_names)

    def test_preserves_input_names(self):
        state = ir.Value(name="s", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.FLOAT))
        graph, body_builder = create_body_graph([state], [])
        body_op = body_builder.op
        result = body_op.Identity(state)
        result.name = "result_out"
        graph.outputs.append(result)

        rename_subgraph_values(graph, "body_")

        assert graph.inputs[0].name == "s"

    def test_preserves_output_names(self):
        state = ir.Value(name="s", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.FLOAT))
        graph, body_builder = create_body_graph([state], [])
        body_op = body_builder.op
        result = body_op.Identity(state)
        result.name = "result_out"
        graph.outputs.append(result)

        rename_subgraph_values(graph, "body_")

        assert graph.outputs[0].name == "result_out"


class TestCompactScanOutput:
    """Tests for compact_scan_output padding removal."""

    def test_builds_graph(self):
        builder, op, graph = create_test_builder()
        scan_result = create_test_input(builder, "scan_result", [3, 10])
        lengths = create_test_input(builder, "lengths", [3], dtype=ir.DataType.INT64)

        result = compact_scan_output(op, scan_result, lengths)
        builder._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_has_compress_op(self):
        builder, op, graph = create_test_builder()
        scan_result = create_test_input(builder, "scan_result", [3, 10])
        lengths = create_test_input(builder, "lengths", [3], dtype=ir.DataType.INT64)

        compact_scan_output(op, scan_result, lengths)

        op_types = [n.op_type for n in graph]
        assert "Compress" in op_types
