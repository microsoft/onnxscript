# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for onnxscript.optimizer._inliner"""

from __future__ import annotations

import unittest
from typing import Callable, Sequence

import onnx
from onnx import parser

from onnxscript import ir
from onnxscript.optimizer._inliner import inline


def _name_checker(renameable: Sequence[str] | None) -> Callable[[str, str], bool]:
    """Construct function to check if actual value name matches expected value name.

    This is used to avoid hard-coding the expected names in the test cases.
    """
    # Default to exact match if no renaming is allowed.
    if renameable is None:
        return lambda a, b: a == b
    # If some names are allowed to be renamed, keep track of the renaming.
    # And check that the renaming is consistent across all nodes.
    renaming_map: dict[str, str] = {}

    def check(actual: str, expected: str) -> bool:
        if expected in renameable:
            # actual name can be different, as long as it is consistently used.
            if expected in renaming_map:
                return renaming_map[expected] == actual
            renaming_map[expected] = actual
            return True
        else:
            return actual == expected

    return check


class InlinerTest(unittest.TestCase):
    def _check(
        self, input_model: str, expected_model: str, renameable: Sequence[str] | None = None
    ) -> None:
        name_check = _name_checker(renameable)
        model_proto = parser.parse_model(input_model)
        model_ir = ir.serde.deserialize_model(model_proto)
        inline(model_ir)
        proto = ir.serde.serialize_model(model_ir)
        text = onnx.printer.to_text(proto)
        print(text)
        expected_proto = parser.parse_model(expected_model)
        expected_ir = ir.serde.deserialize_model(expected_proto)
        self.assertEqual(len(model_ir.graph), len(expected_ir.graph))
        for node, expected_node in zip(model_ir.graph, expected_ir.graph):
            # TODO: handle node renaming
            self.assertEqual(node.op_type, expected_node.op_type)
            self.assertEqual(len(node.inputs), len(expected_node.inputs))
            for input, expected_input in zip(node.inputs, expected_node.inputs):
                self.assertEqual(input is None, expected_input is None)
                if input is not None:
                    self.assertTrue(name_check(input.name, expected_input.name))
            self.assertEqual(len(node.attributes), len(expected_node.attributes))
            for key, value in node.attributes.items():
                self.assertIn(key, expected_node.attributes)
                expected_value = expected_node.attributes[key]
                self.assertTrue(isinstance(value, ir.Attr))
                self.assertTrue(isinstance(expected_value, ir.Attr))
                self.assertEqual(value.type, expected_value.type)
                if (
                    value.type != ir.AttributeType.GRAPH
                    and value.type != ir.AttributeType.GRAPHS
                ):
                    self.assertEqual(value.value, expected_value.value)
                else:
                    self.fail("Graph attributes are not supported yet")
                    # TODO: handle graph attributes
            self.assertEqual(len(node.outputs), len(expected_node.outputs))
            for output, expected_output in zip(node.outputs, expected_node.outputs):
                self.assertTrue(name_check(output.name, expected_output.name))

    def test_single_call(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = Mul(temp, temp)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp = Add(X, X)
                Y = Mul(temp, temp)
            }
        """
        self._check(input_model, expected_model, renameable=["temp"])

    def test_two_calls(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo (X)
                Y = local.foo (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = Mul(temp, temp)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp1 = Add(X, X)
                T = Mul(temp1, temp1)
                temp2 = Add(T, T)
                Y = Mul(temp2, temp2)
            }
        """
        self._check(input_model, expected_model, renameable=["temp1", "temp2"])

    def test_nested_call(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp = Add(X, X)
                Y = Mul(temp, temp)
            }
        """
        self._check(input_model, expected_model, renameable=["temp"])

    def test_attr_parameter(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo <alpha = 0.5> (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo <alpha> (x) => (y) {
                y = Selu <alpha: float = @alpha> (x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = Selu <alpha: float = 0.5> (X)
            }
        """
        self._check(input_model, expected_model)

    def test_attr_parameter_with_default_value(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo <alpha = 0.5> (X)
                Y = local.foo (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo <alpha: float=0.6> (x) => (y) {
                y = Selu <alpha: float = @alpha> (x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = Selu <alpha: float = 0.5> (X)
                Y = Selu <alpha: float = 0.6> (T)
            }
        """
        self._check(input_model, expected_model)


if __name__ == "__main__":
    unittest.main()
