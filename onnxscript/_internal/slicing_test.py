# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Parity tests for scalar indexing combined with slices."""

import itertools
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import parameterized

from onnxscript import INT64, evaluator, opset15, script
from onnxscript._internal import converter


@script(default_opset=opset15)
def last_row_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[-1, :2]


@script(default_opset=opset15)
def last_column_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[:2, -1]


@script(default_opset=opset15)
def last_element(x: INT64["N", "M"]) -> INT64:  # noqa: F821
    return x[-1, -1]


@script(default_opset=opset15)
def penultimate_row_last_column(x: INT64["N", "M"]) -> INT64:  # noqa: F821
    return x[-2, -1]


@script(default_opset=opset15)
def reversed_last_row(x: INT64["N", "M"]) -> INT64["M"]:  # noqa: F821
    return x[-1, ::-1]


@script(default_opset=opset15)
def first_row_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[0, :2]


class TestScalarSlicing(unittest.TestCase):
    def test_dynamic_step_defaults_report_source_and_guidance(self):
        def missing_start(x, step):
            return x[:3:step]

        def missing_stop(x, step):
            return x[1::step]

        def missing_both(x, step):
            return x[::step]

        for function, source in (
            (missing_start, "return x[:3:step]"),
            (missing_stop, "return x[1::step]"),
            (missing_both, "return x[::step]"),
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaises(converter.TranslationError) as raised:
                    script(default_opset=opset15)(function)
                message = str(raised.exception)
                self.assertIn("Slice default start/stop", message)
                self.assertIn("step direction is unknown", message)
                self.assertIn("Specify both start and stop, or use a constant step", message)
                self.assertIn(f"Function '{function.__name__}', line 2", message)
                self.assertIn(source, message)
                self.assertIn("^", message)

    def test_non_integer_slice_components_report_source_and_guidance(self):
        def invalid_start(x):
            return x[1.5:3]

        def invalid_stop(x):
            return x[1:"end"]

        def invalid_step(x):
            return x[1:3:0.5]

        for function, type_name, source in (
            (invalid_start, "float", "return x[1.5:3]"),
            (invalid_stop, "str", 'return x[1:"end"]'),
            (invalid_step, "float", "return x[1:3:0.5]"),
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaises(converter.TranslationError) as raised:
                    script(default_opset=opset15)(function)
                message = str(raised.exception)
                self.assertIn(f"Slice component type must be int, not {type_name}", message)
                self.assertIn("Use integer start, stop, and step values", message)
                self.assertIn(f"Function '{function.__name__}', line 2", message)
                self.assertIn(source, message)
                self.assertIn("^", message)

    def test_dynamic_step_with_explicit_bounds(self):
        @script(default_opset=opset15)
        def bounded_slice(x: INT64["N"], step: INT64) -> INT64["M"]:  # noqa: F821
            return x[1:6:step]

        model = bounded_slice.to_model_proto()
        onnx.checker.check_model(model)
        data = np.arange(8, dtype=np.int64)
        for step in (1, 2, -1):
            with self.subTest(step=step):
                with evaluator.default_as(evaluator.OnnxReferenceRuntimeEvaluator()):
                    actual = bounded_slice(data, np.array(step, dtype=np.int64))
                np.testing.assert_array_equal(actual, data[1:6:step])

    @parameterized.parameterized.expand(
        itertools.product(
            [
                (last_row_slice, (-1, slice(None, 2))),
                (last_column_slice, (slice(None, 2), -1)),
                (last_element, (-1, -1)),
                (penultimate_row_last_column, (-2, -1)),
                (reversed_last_row, (-1, slice(None, None, -1))),
                (first_row_slice, (0, slice(None, 2))),
            ],
            [(4, 3), (2, 5)],
            ["eager", "onnxruntime"],
        )
    )
    def test_matches_numpy(self, function_and_index, shape, mode):
        function, index = function_and_index
        data = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        if mode == "eager":
            with evaluator.default_as(evaluator.OnnxReferenceRuntimeEvaluator()):
                actual = function(data)
        else:
            model = function.to_model_proto(ir_version=9)
            onnx.checker.check_model(model)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, {"x": data})[0]
        np.testing.assert_array_equal(actual, data[index])
        self.assertEqual(actual.shape, data[index].shape)


if __name__ == "__main__":
    unittest.main()
