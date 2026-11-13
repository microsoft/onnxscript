# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
import parameterized
from onnx.reference import ReferenceEvaluator

from onnxscript import optimizer

_OPTIMIZERS = [
    ("fold_ir", optimizer.fold_constants, False),
    ("fold_proto", optimizer.fold_constants, True),
    ("optimize_ir", optimizer.optimize, False),
    ("optimize_proto", optimizer.optimize, True),
]


class InitializerSafetyTest(unittest.TestCase):
    def _apply(self, model, transform, as_proto):
        if as_proto:
            proto = ir.serde.serialize_model(model)
            result = transform(proto)
            return ir.serde.deserialize_model(
                result if isinstance(result, onnx.ModelProto) else proto
            )
        result = transform(model)
        return result if isinstance(result, ir.Model) else model

    def _run(self, model, feeds):
        proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(proto)
        return ReferenceEvaluator(proto).run(None, feeds)

    @parameterized.parameterized.expand(_OPTIMIZERS)
    def test_overridable_if_condition(self, _, transform, as_proto):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2] x, bool condition = {1}) => (float[2] y) {
                y = If(condition) <
                    then_branch = t() => (float[2] a) { a = Identity(x) },
                    else_branch = e() => (float[2] b) { b = Neg(x) }
                >
            }
            """
        )
        x = np.array([2, -3], dtype=np.float32)
        feeds = [
            {"x": x},
            {"x": x, "condition": np.array(False)},
            {"x": x, "condition": np.array(True)},
        ]
        expected = [self._run(model, feed)[0] for feed in feeds]

        model = self._apply(model, transform, as_proto)

        self.assertIn("condition", model.graph.initializers)
        self.assertEqual(model.graph.node(0).op_type, "If")
        for feed, output in zip(feeds, expected):
            np.testing.assert_array_equal(self._run(model, feed)[0], output)

    @parameterized.parameterized.expand(_OPTIMIZERS)
    def test_overridable_gather_data(self, _, transform, as_proto):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g (int64[2] data = {3, 5}) => (int64[1] y)
            <int64[1] index = {0}> {
                y = Gather<axis = 0>(data, index)
            }
            """
        )
        feeds = [{}, {"data": np.array([7, 11], dtype=np.int64)}]
        expected = [self._run(model, feed)[0] for feed in feeds]

        model = self._apply(model, transform, as_proto)

        self.assertIn("data", model.graph.initializers)
        for feed, output in zip(feeds, expected):
            np.testing.assert_array_equal(self._run(model, feed)[0], output)

    @parameterized.parameterized.expand(_OPTIMIZERS)
    def test_shape_folding_preserves_input_default(self, _, transform, as_proto):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2] data = {3, 5}) => (int64[1] y) {
                y = Shape(data)
            }
            """
        )
        model = self._apply(model, transform, as_proto)

        self.assertEqual([value.name for value in model.graph.inputs], ["data"])
        self.assertIn("data", model.graph.initializers)
        self.assertNotIn("Shape", [node.op_type for node in model.graph])
        for feeds in ({}, {"data": np.array([7, 11], dtype=np.float32)}):
            np.testing.assert_array_equal(self._run(model, feeds)[0], [2])

    @parameterized.parameterized.expand(
        [
            *_OPTIMIZERS,
            ("dce_ir", optimizer.remove_unused_nodes, False),
            ("dce_proto", optimizer.remove_unused_nodes, True),
        ]
    )
    def test_unused_initialized_input_preserves_default(self, _, transform, as_proto):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2] x, float[2] two = {2, 2}) => (float[2] y) {
                unused = Add(two, two)
                y = Mul(x, x)
            }
            """
        )
        model = self._apply(model, transform, as_proto)

        self.assertEqual([value.name for value in model.graph.inputs], ["x", "two"])
        self.assertIn("two", model.graph.initializers)
        x = np.array([3, 4], dtype=np.float32)
        np.testing.assert_array_equal(self._run(model, {"x": x})[0], x * x)

    @parameterized.parameterized.expand(
        [
            ("ir", False, False, "Clip"),
            ("proto", True, False, "Clip"),
            ("forced", False, True, "Clip"),
            ("required", False, False, "Mul"),
        ]
    )
    def test_missing_external_input_is_not_omitted(self, _, as_proto, force, op_type):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g () => (float[2] y, float[2] squared)
            <float[2] x = {-1, 1}, float minimum = {0}> {
                y = Clip(x, minimum)
                squared = Mul(x, x)
            }
            """
        )
        model.graph.node(0).op_type = op_type
        expected = self._run(model, {})
        original_minimum = model.graph.initializers["minimum"].const_value
        with tempfile.TemporaryDirectory() as directory:
            model.graph.initializers["minimum"].const_value = ir.ExternalTensor(
                location="unavailable.data",
                base_dir=directory,
                offset=0,
                length=4,
                dtype=ir.DataType.FLOAT,
                shape=ir.Shape([]),
                name="minimum",
            )

            def fold(value):
                return optimizer.fold_constants(
                    value, should_fold=lambda node: True if force else None
                )

            with self.assertLogs("onnxscript.optimizer", level="WARNING") as logs:
                model = self._apply(model, fold, as_proto)

        self.assertTrue(any("minimum" in message for message in logs.output))
        self.assertEqual([node.op_type for node in model.graph], [op_type])
        self.assertIn("squared", model.graph.initializers)
        self.assertIsInstance(
            model.graph.initializers["minimum"].const_value, ir.ExternalTensor
        )
        model.graph.initializers["minimum"].const_value = original_minimum
        actual = self._run(model, {})
        for output, reference in zip(actual, expected):
            np.testing.assert_array_equal(output, reference)

    def test_external_shape_can_fold_without_accessing_data(self):
        tensor = ir.ExternalTensor(
            location="unavailable.data",
            offset=0,
            length=8,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape([2]),
            name="data",
        )
        data = ir.Value(
            name="data",
            shape=tensor.shape,
            type=ir.TensorType(tensor.dtype),
            const_value=tensor,
        )
        shape = ir.Node("", "Shape", [data])
        shape.outputs[0].name = "y"
        shape.outputs[0].shape = ir.Shape([1])
        shape.outputs[0].type = ir.TensorType(ir.DataType.INT64)
        model = ir.Model(
            ir.Graph(
                [],
                shape.outputs,
                nodes=[shape],
                initializers=[data],
                opset_imports={"": 18},
                name="external_shape",
            ),
            ir_version=10,
        )

        with mock.patch.object(
            ir.ExternalTensor, "numpy", side_effect=AssertionError("Data was accessed")
        ):
            optimizer.fold_constants(model)

        self.assertNotIn("Shape", [node.op_type for node in model.graph])
        np.testing.assert_array_equal(self._run(model, {})[0], [2])

    def test_invalid_tensor_data_error_is_not_suppressed(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g () => (float[2] y)
            <float[2] data = {-1, 2}> {
                y = Neg(data)
            }
            """
        )
        model.graph.initializers["data"].const_value = ir.tensor(
            np.array([-1, 2], dtype=np.float32)
        )
        with mock.patch.object(
            ir.Tensor, "numpy", side_effect=ValueError("Invalid data")
        ), self.assertRaisesRegex(ValueError, "Invalid data"):
            optimizer.fold_constants(model)

    def test_omitted_optional_input_can_still_fold(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18]>
            g () => (float[2] y)
            <float[2] x = {-1, 2}, float maximum = {1}> {
                y = Clip(x, , maximum)
            }
            """
        )
        expected = self._run(model, {})[0]

        optimizer.fold_constants(model)

        self.assertEqual(len(model.graph), 0)
        np.testing.assert_array_equal(self._run(model, {})[0], expected)


if __name__ == "__main__":
    unittest.main()
