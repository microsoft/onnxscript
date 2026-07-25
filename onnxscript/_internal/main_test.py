# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import unittest

import onnx

import onnxscript
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.values import Opset


class ScriptCustomOpTypeTest(unittest.TestCase):
    def test_custom_op_type_is_used_consistently(self):
        custom_opset = Opset("com.example.custom", 1)

        @script(custom_opset, op_type="MY_NEW_NAME_OP", producer_name="onnxscript-test")
        def custom_op(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        @script()
        def caller(x: FLOAT) -> FLOAT:
            return custom_op(x)

        self.assertEqual(custom_op.name, "MY_NEW_NAME_OP")
        self.assertEqual(custom_op.__name__, "custom_op")
        self.assertEqual(custom_op.op_signature.name, "MY_NEW_NAME_OP")
        self.assertEqual(custom_op.function_ir.graph.name, "MY_NEW_NAME_OP")
        self.assertEqual(custom_op.to_function_proto().name, "MY_NEW_NAME_OP")

        model = caller.to_model_proto()
        call_node = model.graph.node[0]
        self.assertEqual(call_node.op_type, "MY_NEW_NAME_OP")
        self.assertEqual(call_node.domain, "com.example.custom")
        self.assertEqual(
            {(function.domain, function.name) for function in model.functions},
            {("com.example.custom", "MY_NEW_NAME_OP")},
        )
        self.assertNotIn("op_type", custom_op.kwargs)
        self.assertEqual(custom_op.to_model_proto(io_types=FLOAT).producer_name, "onnxscript-test")
        onnx.checker.check_model(model)

    def test_issue_example_uses_default_local_opset(self):
        @script(op_type="MY_NEW_NAME_OP")
        def custom_op(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        self.assertEqual(custom_op.name, "MY_NEW_NAME_OP")
        self.assertEqual(custom_op.to_function_proto().name, "MY_NEW_NAME_OP")

    def test_default_op_type_is_python_function_name(self):
        @script()
        def default_name(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        self.assertEqual(default_name.name, "default_name")
        self.assertEqual(default_name.to_function_proto().name, "default_name")

    def test_none_uses_default_name_and_public_signature_is_stable(self):
        @script(op_type=None)
        def default_name(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        self.assertIsNone(inspect.signature(script).parameters["op_type"].default)
        self.assertEqual(default_name.name, "default_name")

    def test_custom_name_can_be_reused_in_different_domains(self):
        first_opset = Opset("com.example.first", 1)
        second_opset = Opset("com.example.second", 1)

        @script(first_opset, op_type="SharedName")
        def first(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        @script(second_opset, op_type="SharedName")
        def second(x: FLOAT) -> FLOAT:
            return op.Neg(x)

        @script()
        def intermediate(x: FLOAT) -> FLOAT:
            return first(x)

        @script()
        def caller(x: FLOAT) -> tuple[FLOAT, FLOAT]:
            return intermediate(x), second(x)

        model = caller.to_model_proto()
        self.assertTrue(
            {
                ("com.example.first", "SharedName"),
                ("com.example.second", "SharedName"),
            }.issubset(
                {(function.domain, function.name) for function in model.functions}
            )
        )
        onnx.checker.check_model(model)

    def test_distinct_functions_cannot_share_an_identifier(self):
        custom_opset = Opset("com.example.custom", 1)

        @script(custom_opset, op_type="Duplicate")
        def first(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        @script(custom_opset, op_type="Duplicate")
        def second(x: FLOAT) -> FLOAT:
            return op.Neg(x)

        @script()
        def caller(x: FLOAT) -> tuple[FLOAT, FLOAT]:
            return first(x), second(x)

        with self.assertRaisesRegex(ValueError, "same identifier"):
            caller.to_model_proto()

    def test_repeated_calls_to_same_function_are_deduplicated(self):
        custom_opset = Opset("com.example.custom", 1)

        @script(custom_opset, op_type="CalledTwice")
        def custom_op(x: FLOAT) -> FLOAT:
            return op.Abs(x)

        @script()
        def caller(x: FLOAT) -> tuple[FLOAT, FLOAT]:
            return custom_op(x), custom_op(x)

        model = caller.to_model_proto()
        self.assertEqual(len(model.functions), 1)

    def test_invalid_custom_op_types_are_rejected(self):
        invalid_values = ("", "   ", 42)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):

                    @script(op_type=value)
                    def invalid(x: FLOAT) -> FLOAT:
                        return op.Abs(x)

    def test_non_c_identifier_names_are_supported(self):
        for value in ("1LeadingDigit", "Name-With-Punctuation", "Name With Spaces"):
            with self.subTest(value=value):

                @script(op_type=value)
                def custom_op(x: FLOAT) -> FLOAT:
                    return op.Abs(x)

                self.assertEqual(custom_op.name, value)


if __name__ == "__main__":
    unittest.main()
