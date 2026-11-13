# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _materialize_reshape_shape


class MaterializeReshapeShapeTest(unittest.TestCase):
    def test_reference_attributes_not_materialized(self):
        for node_index, attribute in ((0, "start"), (1, "allowzero")):
            with self.subTest(attribute=attribute):
                model = ir.from_onnx_text(
                    """
                    <ir_version: 8, opset_import: ["" : 14]>
                    agraph (float[6] data, float[2, 3] ref) => (float[2, 3] output)
                    {
                        shape = Shape(ref)
                        output = Reshape(data, shape)
                    }
                    """
                )
                model.graph.node(node_index).attributes[attribute] = ir.RefAttr(
                    attribute, attribute, ir.AttributeType.INT
                )
                self.assertEqual(_materialize_reshape_shape.rules.apply_to_model(model), 0)

    def test_runtime_shape_is_not_proven_by_output_annotation(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 14]>
            agraph (float[6] data, int64[2] shape) => (float[2, 3] output)
            {
                output = Reshape(data, shape)
            }
            """
        )
        self.assertEqual(_materialize_reshape_shape.rules.apply_to_model(model), 0)

    def test_zero_dim_copies_input_dimension(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 14]>
            agraph (float[2, 3] data, float[0, 3] ref) => (float[2, 3] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
            """
        )
        original = model.clone()
        self.assertEqual(_materialize_reshape_shape.rules.apply_to_model(model), 1)
        reshape = model.graph.outputs[0].producer()
        self.assertEqual(reshape.inputs[1].const_value.numpy().tolist(), [2, 3])
        testing.assert_numerically_equal(
            original,
            model,
            (np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3), dtype=np.float32)),
        )

    def test_opset_and_unknown_dim_boundaries(self):
        for opset in (5, 13, 14):
            for dims, min_opset in (
                ([2, 6], 5),
                ([None, 6], 5),
                ([0, 6], 14),
                (["B", 0, 6], None),
                (["B", "C"], None),
            ):
                with self.subTest(opset=opset, dims=dims):
                    model = ir.from_onnx_text(
                        f"""
                        <ir_version: 8, opset_import: ["" : {opset}]>
                        agraph (float[N, 2, 3] data, float ref) => (float output)
                        {{
                            shape = Shape(ref)
                            output = Reshape<allowzero=1>(data, shape)
                        }}
                        """
                    )
                    reshape = model.graph.outputs[0].producer()
                    if opset < 14:
                        del reshape.attributes["allowzero"]
                    if 0 in dims:
                        model.graph.inputs[0].shape = ir.Shape([0, 0, 3])
                    model.graph.inputs[1].shape = ir.Shape(dims)
                    model.graph.outputs[0].shape = ir.Shape(dims)
                    expected = min_opset is not None and opset >= min_opset
                    count = _materialize_reshape_shape.rules.apply_to_model(model)
                    self.assertEqual(count, int(expected))
                    if expected:
                        reshape = model.graph.outputs[0].producer()
                        self.assertEqual(
                            reshape.inputs[1].const_value.numpy().tolist(),
                            [-1 if not isinstance(d, int) else d for d in dims],
                        )
                        self.assertEqual(
                            reshape.attributes.get_int("allowzero", None),
                            1 if opset >= 14 else None,
                        )

    def test_fully_static_output_shape_materializes(self):
        """When output shape is fully static, replace dynamic shape input with constant."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data, float[2, 3] ref) => (float[2, 3] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([2, 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        shape_input = reshape_nodes[0].inputs[1]
        self.assertIsNotNone(shape_input.const_value)
        self.assertEqual(shape_input.const_value.numpy().tolist(), [2, 3])

    def test_one_symbolic_dim_uses_minus_one(self):
        """When output has one symbolic dim, replace it with -1."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data, float[B, 3] ref) => (float[B, 3] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape(["B", 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        shape_input = reshape_nodes[0].inputs[1]
        self.assertIsNotNone(shape_input.const_value)
        self.assertEqual(shape_input.const_value.numpy().tolist(), [-1, 3])

    def test_two_symbolic_dims_not_materialized(self):
        """When output has two symbolic dims, the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data, float[B, C] ref) => (float[B, C] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape(["B", "C"])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_zero_dim_with_symbolic_dim_not_materialized(self):
        """[-1, 0, 512] is invalid with allowzero=1, so the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: [ "" : 18]>
            agraph (float[B, 8, 0, 64] data, float[B, 0, 512] ref) => (float[B, 0, 512] output)
            {
                shape = Shape(ref)
                output = Reshape<allowzero=1>(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape(["B", 0, 512])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_constant_shape_input_not_replaced(self):
        """When the shape input is already a constant, the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[2, 3] output)
            {
                shape = Constant<value: tensor = int64[2] {2, 3}>()
                output = Reshape(data, shape)
            }
        """
        )
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_unknown_output_shape_with_proven_target_materializes(self):
        """A proven shape input does not require an output shape annotation."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float output)
            {
                shape = Shape(data)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = None
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)

    def test_allowzero_attribute_preserved(self):
        """The allowzero attribute should be preserved on the new Reshape."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data, float[2, 3] ref) => (float[2, 3] output)
            {
                shape = Shape(ref)
                output = Reshape<allowzero=1>(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([2, 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        allowzero = reshape_nodes[0].attributes.get_int("allowzero", 0)
        self.assertEqual(allowzero, 1)

    def test_numerical_correctness_static(self):
        """Verify numerical equivalence for fully static materialization."""
        # Build a model where a dynamic Concat produces the shape for Reshape.
        # After materialization, the Reshape uses a constant shape.
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[12] data, float[3, 4] ref) => (float[3, 4] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
        """
        original = ir.from_onnx_text(model_text)
        model = ir.from_onnx_text(model_text)
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([3, 4])
                break
        _materialize_reshape_shape.rules.apply_to_model(model)
        testing.assert_numerically_equal(
            original,
            model,
            (
                np.arange(12).astype(np.float32),
                np.zeros((3, 4), dtype=np.float32),
            ),
        )


if __name__ == "__main__":
    unittest.main()
