# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import inspect
import typing
import unittest
from unittest import mock

import onnx

import onnxscript
from onnxscript import FLOAT, INT64, opset18, script
from onnxscript._internal import values


class TracedOnnxFunctionTest(unittest.TestCase):
    def test_init(self):
        def function(input1, input2, attr1: int, attr2: int = 1):
            return input1 + input2 + attr1 + attr2

        opset = values.Opset("test", 1)
        traced_function = values.TracedOnnxFunction(opset, function)
        self.assertEqual(traced_function.opset, opset)
        self.assertEqual(traced_function.name, function.__name__)
        self.assertEqual(traced_function.func, function)

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
    def test_model_type_override_lengths(self):
        @script()
        def function(x: FLOAT["N"], y: FLOAT["N"]):  # noqa: F821, UP037
            return opset18.Identity(x), opset18.Identity(y)

        original = function.to_function_proto().SerializeToString()
        for name in ("input_types", "output_types"):
            for count in (0, 1, 3):
                with self.subTest(name=name, count=count):
                    with mock.patch.object(
                        function.function_ir, "get_called_functions"
                    ) as get_called_functions:
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{name} must contain exactly 2 types for function "
                            rf"'function'; got {count}\.",
                        ):
                            function.to_model_proto(**{name: [FLOAT["N"]] * count})
                        get_called_functions.assert_not_called()
                    self.assertEqual(
                        function.to_function_proto().SerializeToString(), original
                    )

        for overrides in (
            {},
            {"input_types": None, "output_types": None},
            {"input_types": [FLOAT["N"]] * 2, "output_types": [FLOAT["N"]] * 2},
        ):
            with self.subTest(overrides=overrides):
                model = function.to_model_proto(**overrides)
                self.assertEqual(len(model.graph.input), 2)
                self.assertEqual(len(model.graph.output), 2)

    def test_zero_arity_type_overrides(self):
        @script()
        def no_inputs():
            return opset18.Constant(value_float=1.0)

        @script(default_opset=opset18)
        def no_outputs(x: FLOAT["N"]):  # noqa: F821, UP037
            return ()

        for function, name, expected in (
            (no_inputs, "input_types", 0),
            (no_inputs, "output_types", 1),
            (no_outputs, "input_types", 1),
            (no_outputs, "output_types", 0),
        ):
            for count in (0, 1, 2):
                with self.subTest(function=function.name, name=name, count=count):
                    overrides = {name: [FLOAT["N"]] * count}
                    if count == expected:
                        model = function.to_model_proto(**overrides)
                        signature = (
                            model.graph.input if name == "input_types" else model.graph.output
                        )
                        self.assertEqual(len(signature), expected)
                    else:
                        with self.assertRaisesRegex(
                            ValueError, rf"{name} must contain exactly {expected} types"
                        ):
                            function.to_model_proto(**overrides)

    def test_model_type_override_precedence(self):
        @script(
            io_types=FLOAT["D"],
            input_types=[INT64["D"]],
            output_types=[INT64["D"]],
        )
        def function(x: FLOAT["A"]) -> FLOAT["A"]:  # noqa: F821, UP037
            return opset18.Identity(x)

        cases = (
            ({}, onnx.TensorProto.INT64, "D"),
            (
                {"input_types": [FLOAT["C"]], "output_types": [FLOAT["C"]]},
                onnx.TensorProto.FLOAT,
                "C",
            ),
            (
                {"input_types": None, "output_types": None},
                onnx.TensorProto.FLOAT,
                "A",
            ),
        )
        for overrides, elem_type, dimension in cases:
            with self.subTest(overrides=overrides):
                model = function.to_model_proto(**overrides)
                for value in (*model.graph.input, *model.graph.output):
                    self.assertEqual(value.type.tensor_type.elem_type, elem_type)
                    self.assertEqual(value.type.tensor_type.shape.dim[0].dim_param, dimension)

    def test_call_can_replace_invalid_decorator_type_overrides(self):
        @script(input_types=[], output_types=[FLOAT["N"], FLOAT["N"]])
        def function(x):
            return opset18.Identity(x)

        with self.assertRaisesRegex(ValueError, "input_types must contain exactly 1 types"):
            function.to_model_proto()
        with self.assertRaisesRegex(ValueError, "output_types must contain exactly 1 types"):
            function.to_model_proto(input_types=[FLOAT["N"]])
        model = function.to_model_proto(input_types=[FLOAT["N"]], output_types=[FLOAT["N"]])
        onnx.checker.check_model(model)
        model = function.to_model_proto(
            input_types=None, output_types=None, io_types=FLOAT["N"]
        )
        onnx.checker.check_model(model)

    def test_io_types_fill_missing_types_without_replacing_annotations(self):
        @script(io_types=FLOAT["D"])
        def function(x: FLOAT["A"]):  # noqa: F821, UP037
            return opset18.Identity(x)

        for overrides, output_type, dimension in (
            ({}, onnx.TensorProto.FLOAT, "D"),
            ({"io_types": INT64["C"]}, onnx.TensorProto.INT64, "C"),
        ):
            with self.subTest(overrides=overrides):
                model = function.to_model_proto(**overrides)
                input_type = model.graph.input[0].type.tensor_type
                self.assertEqual(input_type.elem_type, onnx.TensorProto.FLOAT)
                self.assertEqual(input_type.shape.dim[0].dim_param, "A")
                actual_output_type = model.graph.output[0].type.tensor_type
                self.assertEqual(actual_output_type.elem_type, output_type)
                self.assertEqual(actual_output_type.shape.dim[0].dim_param, dimension)
        model = function.to_model_proto(io_types=None)
        self.assertFalse(model.graph.output[0].HasField("type"))

    def test_attributes_are_not_model_inputs(self):
        @script()
        def function(x: FLOAT["N"], alpha: float = 1.0) -> FLOAT["N"]:  # noqa: F821, UP037
            return opset18.LeakyRelu(x, alpha=alpha)

        model = function.to_model_proto(input_types=[FLOAT["N"]])
        self.assertEqual(len(model.graph.input), 1)
        onnx.checker.check_model(model)
        with self.assertRaisesRegex(ValueError, "input_types must contain exactly 1 types"):
            function.to_model_proto(input_types=[FLOAT["N"], FLOAT["N"]])

    def test_explicit_model_validation_and_graph_ownership(self):
        @script()
        def identity(x):
            return opset18.Identity(x)

        graph = identity.graph()
        self.assertIs(identity.graph(), graph)
        clone = graph.clone()
        clone.inputs[0].name = "independent_input"
        self.assertEqual(graph.inputs[0].name, "x")

        incomplete_model = identity.to_model_proto()
        self.assertFalse(incomplete_model.graph.input[0].HasField("type"))
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(incomplete_model)

        model = identity.to_model_proto(input_types=[FLOAT["N"]], output_types=[FLOAT["N"]])
        onnx.checker.check_model(model)
        model.graph.input[0].name = "independent_model_input"
        self.assertEqual(graph.inputs[0].name, "x")
        self.assertFalse(identity.to_model_proto().graph.input[0].HasField("type"))

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
