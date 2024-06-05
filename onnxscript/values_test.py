# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import inspect
import typing
import unittest

import onnxscript
from onnxscript import values


class TracedOnnxFunctionTest(unittest.TestCase):
    def test_init(self):
        def function(input1, input2, attr1: int, attr2: int = 1):
            return input1 + input2 + attr1 + attr2

        opset = values.Opset("test", 1)
        traced_function = values.TracedOnnxFunction(opset, function)
        self.assertEqual(traced_function.opset, opset)
        self.assertEqual(traced_function.name, function.__name__)
        self.assertEqual(traced_function.func, function)

    def test_param_schemas_in_correct_order_with_mixed_inputs_and_attrs(self):
        opset = values.Opset("test", 1)

        def function(input1, input2, attr1: int, attr2: float, input3, attr3: str = "default"):
            return opset.CustomOp(input1 + input2, input3, attr1, attr2, attr3)

        traced_function = values.TracedOnnxFunction(opset, function)
        param_schemas = traced_function.param_schemas()
        expected_ordered_param_names = [
            "input1",
            "input2",
            "attr1",
            "attr2",
            "input3",
            "attr3",
        ]
        self.assertEqual(len(param_schemas), len(expected_ordered_param_names))
        for i, param_schema in enumerate(param_schemas):
            self.assertEqual(param_schema.name, expected_ordered_param_names[i])

    def test_it_preserves_the_function_signature(self):
        opset = values.Opset("test", 1)

        def function(input1, input2, attr1: int, attr2: float, input3, attr3: str = "default"):
            return opset.CustomOp(input1 + input2, input3, attr1, attr2, attr3)

        traced_function = values.TracedOnnxFunction(opset, function)
        signature = inspect.signature(traced_function)
        self.assertEqual(signature.parameters["input1"].name, "input1")
        self.assertEqual(signature.parameters["input2"].name, "input2")
        self.assertEqual(signature.parameters["attr1"].name, "attr1")
        self.assertEqual(signature.parameters["attr2"].name, "attr2")
        self.assertEqual(signature.parameters["input3"].name, "input3")
        self.assertEqual(signature.parameters["attr3"].name, "attr3")

        annotations = typing.get_type_hints(traced_function)
        self.assertEqual(annotations["attr1"], int)
        self.assertEqual(annotations["attr2"], float)
        self.assertEqual(annotations["attr3"], str)


class OnnxFunctionTest(unittest.TestCase):
    def test_param_schemas_in_correct_order_with_mixed_inputs_and_attrs(self):
        opset = values.Opset("test", 1)

        @onnxscript.script(default_opset=opset)
        def function(input1, input2, attr1: int, attr2: float, input3, attr3: str = "default"):
            return opset.CustomOp(input1 + input2, input3, attr1, attr2, attr3)

        param_schemas = function.param_schemas()
        expected_ordered_param_names = [
            "input1",
            "input2",
            "attr1",
            "attr2",
            "input3",
            "attr3",
        ]
        self.assertEqual(len(param_schemas), len(expected_ordered_param_names))
        for i, param_schema in enumerate(param_schemas):
            self.assertEqual(param_schema.name, expected_ordered_param_names[i])

    def test_it_preserves_the_function_signature(self):
        opset = values.Opset("test", 1)

        @onnxscript.script(default_opset=opset)
        def function(input1, input2, attr1: int, attr2: float, input3, attr3: str = "default"):
            return opset.CustomOp(input1 + input2, input3, attr1, attr2, attr3)

        signature = inspect.signature(function)
        self.assertEqual(signature.parameters["input1"].name, "input1")
        self.assertEqual(signature.parameters["input2"].name, "input2")
        self.assertEqual(signature.parameters["attr1"].name, "attr1")
        self.assertEqual(signature.parameters["attr2"].name, "attr2")
        self.assertEqual(signature.parameters["input3"].name, "input3")
        self.assertEqual(signature.parameters["attr3"].name, "attr3")

        annotations = typing.get_type_hints(function)
        self.assertEqual(annotations["attr1"], int)
        self.assertEqual(annotations["attr2"], float)
        self.assertEqual(annotations["attr3"], str)


if __name__ == "__main__":
    unittest.main()
