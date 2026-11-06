# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Negative syntax boundaries and non-graph side effects of the script language.

Positive, executable counterparts are in eager_mode_test.EagerModeTest.
"""

import contextlib
import io
import unittest

import numpy as np
import onnx
import parameterized

from onnxscript import FLOAT, INT64, evaluator, graph, script, values
from onnxscript import opset17 as op
from onnxscript._internal.converter import TranslationError


class LanguageSupportTest(unittest.TestCase):
    def test_return_inside_if_is_rejected(self):
        def early_return(x):
            if x > 0:
                return x
            return -x

        with self.assertRaisesRegex(ValueError, "Return statements.*control-flow"):
            script(default_opset=op)(early_return)

    def test_return_inside_loop_is_rejected(self):
        def early_return(x):
            for _ in range(3):
                return x
            return -x

        with self.assertRaisesRegex(ValueError, "Return statements.*control-flow"):
            script(default_opset=op)(early_return)

    def test_return_inside_while_is_rejected(self):
        def early_return(x):
            while x:
                return x
            return -x

        with self.assertRaisesRegex(ValueError, "Return statements.*control-flow"):
            script(default_opset=op)(early_return)

    def test_return_inside_constant_if_is_rejected(self):
        def early_return(x):
            if True:
                return x
            return -x

        with self.assertRaisesRegex(ValueError, "Return statements.*control-flow"):
            script(default_opset=op)(early_return)

    def test_chained_comparison_is_rejected(self):
        def chained(x):
            return 0 < x < 3

        with self.assertRaisesRegex(TranslationError, "Multiple comparisons"):
            script(default_opset=op)(chained)

    def test_dynamic_step_with_default_bound_is_rejected(self):
        def dynamic_step(x, step):
            return x[::step]

        with self.assertRaisesRegex(RuntimeError, "Default start/stop.*step direction"):
            script(default_opset=op)(dynamic_step)

    def test_non_integer_slice_is_rejected(self):
        def non_integer(x):
            return x[0.5:]

        with self.assertRaisesRegex(RuntimeError, "Slice component type must be int"):
            script(default_opset=op)(non_integer)

    def test_outer_return_annotation_survives_nested_graph(self):
        def typed(x) -> FLOAT:
            @graph()
            def body(state, element) -> tuple[FLOAT, FLOAT]:
                return state, element

            return x, x

        def untyped(x) -> FLOAT:
            @graph()
            def body(state, element):
                return state, element

            return x, x

        for function in (typed, untyped):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(
                SyntaxError, "Mismatch in number of return values and types"
            ):
                script(default_opset=op)(function)

    def test_nested_graph_capture_cannot_be_reassigned_before_use(self):
        def changed_capture(x):
            captured = op.Neg(x)

            @graph()
            def body(state, element):
                updated = op.Add(state, element)
                output = op.Add(updated, captured)
                return updated, output

            captured = op.Abs(x)
            zero = op.Constant(value_float=0.0)
            _, result = op.Scan(zero, x, body=body, num_scan_inputs=1)
            return result

        with self.assertRaisesRegex(TranslationError, "Outer scope variable.*modified"):
            script()(changed_capture)

    def test_nonlocal_in_nested_graph_is_rejected(self):
        def mutate_capture(x):
            captured = op.Neg(x)

            @graph()
            def body(state, element):
                nonlocal captured
                captured = op.Add(state, element)
                return captured, captured

            zero = op.Constant(value_float=0.0)
            _, result = op.Scan(zero, x, body=body, num_scan_inputs=1)
            return result

        with self.assertRaisesRegex(ValueError, "Unsupported statement type.*Nonlocal"):
            script()(mutate_capture)

    @parameterized.parameterized.expand(["onnxruntime", "reference"])
    def test_missing_graph_decorator_is_an_eager_error(self, backend):
        @script()
        def undecorated(x: INT64["N"]) -> INT64["N"]:  # noqa: F821
            def body(state: INT64, element: INT64) -> tuple[INT64, INT64]:
                updated = op.Add(state, element)
                return updated, updated

            zero = op.Constant(value_int=0)
            _, result = op.Scan(zero, x, body=body, num_scan_inputs=1)
            return result

        onnx.checker.check_model(undecorated.to_model_proto(ir_version=9))
        runtime = (
            evaluator.ORTEvaluator()
            if backend == "onnxruntime"
            else evaluator.OnnxReferenceRuntimeEvaluator()
        )
        with evaluator.default_as(runtime), self.assertRaisesRegex(TypeError, "@graph"):
            undecorated(np.array([1, 2], dtype=np.int64))

    def test_print_is_eager_only(self):
        @script()
        def debug(x: FLOAT[2]) -> FLOAT[2]:
            print("debug marker")
            return op.Identity(x)

        self.assertEqual(
            [node.op_type for node in debug.to_function_proto().node], ["Identity"]
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            actual = debug(np.array([1, 2], dtype=np.float32))
        self.assertEqual(stream.getvalue(), "debug marker\n")
        np.testing.assert_array_equal(actual, [1, 2])

    @parameterized.parameterized.expand(["onnxruntime", "reference"])
    def test_schema_less_custom_domain_is_preserved(self, backend):
        custom = values.Opset("test.language.custom", 1)

        @script()
        def custom_call(x):
            return custom.Transform(x, mode="fast")

        proto = custom_call.to_function_proto()
        self.assertEqual(len(proto.node), 1)
        node = proto.node[0]
        self.assertEqual((node.domain, node.op_type), ("test.language.custom", "Transform"))
        self.assertEqual(list(node.input), ["x"])
        self.assertEqual(node.attribute[0].name, "mode")
        self.assertEqual(node.attribute[0].s, b"fast")
        self.assertIn(
            ("test.language.custom", 1), [(v.domain, v.version) for v in proto.opset_import]
        )
        runtime = (
            evaluator.ORTEvaluator()
            if backend == "onnxruntime"
            else evaluator.OnnxReferenceRuntimeEvaluator()
        )
        with evaluator.default_as(runtime), self.assertRaisesRegex(
            AttributeError, "Attribute Transform not found"
        ):
            custom_call(np.array([1, 2], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
